"""Explicit, recoverable conversion of the flat private store to the shared layout.

This one-off maintenance operation reads one known layout and schema. Normal
registry access never falls back to these paths. Source data and derivatives
are not modified.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from nro.configuration.store import fingerprint
from nro.engine.io import atomic_write_json, sync_directory
from nro.orchestration.branch_store import _decode
from nro.orchestration.branches import branch_id
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import APPLICATION_ID, SCHEMA_VERSION, RegistryLock

# Scientific identities and historical producer details must survive relocation.
PROTECTED = {
    "artifact_contract",
    "configuration",
    "software",
    "contract",
    "artifact_contract_json",
    "contract_json",
    "resolved_yaml",
    "revision_fingerprint",
    "artifact_fingerprint",
    "contract_fingerprint",
    "lineage_fingerprint",
    "config_fingerprint",
    "definition_fingerprint",
}
LOCK_NAMES = {
    "registry.lock",
    "registry.lock.recovery",
    "execution-cache.lock",
    "execution-cache.recovery-lock",
    "edit.lock",
    "edit.recovery-lock",
    "registry.recovery-lock",
    "publish.lock",
    "publish.recovery-lock",
}


@dataclass(frozen=True)
class CutoverPlan:
    """A read-only preview of file placement, copy size, and source identity."""

    root: Path
    mappings: tuple[tuple[str, str], ...]
    files: int
    bytes: int
    source_fingerprint: str


def journal_path(root: Path) -> Path:
    """Return the sibling journal that blocks clients during publication gaps."""
    return ControlPaths(root).cutover_journal


def _ignored(name: str) -> bool:
    return name in LOCK_NAMES or name.startswith("registry.lock.released-")


def _inventory(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Private-state directory is missing or redirected: {root}")
    files = {}
    for parent, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if not _ignored(name))
        for name in (*directories, *names):
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"Cutover does not follow symlinks: {path}")
        for name in sorted(names):
            if _ignored(name):
                continue
            path = Path(parent) / name
            if not path.is_file():
                raise ValueError(f"Unexpected private-state entry: {path}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stat = path.stat()
            files[str(path.relative_to(root))] = [
                stat.st_size,
                stat.st_mtime_ns,
                digest,
                stat.st_uid,
                stat.st_gid,
                stat.st_mode,
            ]
    return files


@contextmanager
def _database(path: Path, *, write=False):
    db = sqlite3.connect(path.as_uri() + ("?mode=rw" if write else "?mode=ro"), uri=True)
    try:
        db.row_factory = sqlite3.Row
        yield db
        if write:
            db.commit()
    finally:
        db.close()


def _quiescent(root: Path) -> None:
    with _database(root / "registry.sqlite3") as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            raise ValueError("Cutover source is not an nro scheduler registry")
        row = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if (
            row is None
            or int(row[0]) != SCHEMA_VERSION
            or db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
        ):
            raise ValueError(
                f"Cutover requires registry schema {SCHEMA_VERSION}; it does not migrate schemas"
            )
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Source registry integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("Source registry has broken foreign-key references")
        queries = {
            "active demand": "SELECT 1 FROM requests WHERE state='active' LIMIT 1",
            "active attempts": "SELECT 1 FROM attempts WHERE state IN ('queued','running','cancel_requested') LIMIT 1",
            "workers": "SELECT 1 FROM workers WHERE state IN ('idle','running','draining','shutdown_requested') LIMIT 1",
            "allocations": "SELECT 1 FROM scheduler_submissions WHERE state IN ('prepared','submitted','running','cancel_requested') LIMIT 1",
            "maintenance": "SELECT 1 FROM metadata WHERE key='maintenance_mode' LIMIT 1",
        }
        for label, query in queries.items():
            if db.execute(query).fetchone():
                raise ValueError(
                    f"Cutover requires a quiescent store: {label} remain; coordinate with the site maintainer"
                )
    for path in (root / "ingestion").glob("*.json"):
        if json.loads(path.read_text()).get("state") in {"queued", "running"}:
            raise ValueError("Cutover requires queued/running ingestion to finish or be cancelled")
    for path in (root / "ingestion/reviews").glob("*.json"):
        if json.loads(path.read_text()).get("expires", 0) > time.time():
            raise ValueError("Cutover requires active ingestion review leases to end")


def _mappings(root: Path) -> tuple[tuple[str, str], ...]:
    paths = ControlPaths(root)
    targets = {
        "registry.sqlite3": paths.database,
        "requests": paths.scheduler / "requests",
        "workers": paths.scheduler / "workers",
        "ingestion": paths.ingestion,
        "implementations": paths.implementations,
        "execution-sites": paths.execution_sites,
    }
    targets.update(
        {
            name: paths.branch("main") / name
            for name in ("manifests", "events", "snapshots", "workflows")
        }
    )
    unexpected = {p.name for p in root.iterdir()} - set(targets) - {"branches"}
    unexpected = {name for name in unexpected if not _ignored(name)}
    if unexpected:
        raise ValueError(
            f"Unrecognized source entries; no files will be discarded: {sorted(unexpected)}"
        )
    branch_root = root / "branches"
    if branch_root.exists():
        if (branch_root / "registration-pending.json").exists():
            raise ValueError("Finish interrupted branch registration before cutover")
        catalog = branch_root / "registrations.json"
        topology = _decode(json.loads(catalog.read_text()))
        unexpected = {p.name for p in branch_root.iterdir()} - {"registrations.json", "registries"}
        if any(not _ignored(name) for name in unexpected):
            raise ValueError("Unrecognized branch metadata in cutover source")
        targets["branches/registrations.json"] = paths.catalog
        expected = {branch_id(name) for name in topology.records}
        if {p.name for p in (branch_root / "registries").iterdir()} != expected:
            raise ValueError("Branch databases do not match the catalog")
        for name, record in topology.records.items():
            relative = f"branches/registries/{branch_id(name)}"
            with _database(root / relative / "registry.sqlite3") as db:
                identity = dict(db.execute("SELECT key,value FROM identity"))
            if identity != dict(
                branch=name, registry_id=record.registry_id, scheduler_control=str(root)
            ):
                raise ValueError(f"Branch database does not match its registration: {name}")
            targets[relative] = paths.branch(name)
    return tuple(
        (name, str(path.relative_to(root)))
        for name, path in sorted(targets.items())
        if (root / name).exists()
    )


def preview(root: Path) -> CutoverPlan:
    """Inspect the flat store without locks, writes, scheduler queries, or data changes.

    Read and hash private files to estimate copy size and detect changes between
    confirmation and execution. Preview is not a reservation; execution repeats
    the checks under maintenance locks.
    """
    root = ControlPaths(root).root
    if journal_path(root).exists():
        raise ValueError("An interrupted cutover exists; resume it or request rollback")
    _quiescent(root)
    mappings = _mappings(root)
    inventory = _inventory(root)
    return CutoverPlan(
        root,
        mappings,
        len(inventory),
        sum(row[0] for row in inventory.values()),
        fingerprint(inventory),
    )


def _rewrite(value, root: Path, mappings):
    if isinstance(value, str):
        for source, destination in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
            old = str(root / source)
            if value == old or value.startswith(old + "/"):
                return str(root / destination) + value[len(old) :]
        return value
    if isinstance(value, list):
        return [_rewrite(item, root, mappings) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PROTECTED:
                if key != "software" and _contains_movable(item, root, mappings):
                    raise ValueError(
                        f"Scientific field {key} contains private paths; review is required before cutover"
                    )
                result[key] = item
            else:
                result[key] = _rewrite(item, root, mappings)
        return result
    return value


def _contains_movable(value, root: Path, mappings) -> bool:
    if isinstance(value, str):
        return _rewrite(value, root, mappings) != value
    if isinstance(value, list):
        return any(_contains_movable(item, root, mappings) for item in value)
    if isinstance(value, dict):
        return any(_contains_movable(item, root, mappings) for item in value.values())
    return False


def _rewrite_db(path: Path, root: Path, mappings) -> None:
    with _database(path, write=True) as db:
        tables = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            for row in db.execute(f"SELECT rowid AS _cutover_rowid, * FROM {quoted}").fetchall():
                changes = {}
                for key in row.keys():
                    value = row[key]
                    if not isinstance(value, str):
                        continue
                    if key in PROTECTED:
                        decoded = yaml.safe_load(value) if key == "resolved_yaml" else value
                        if key.endswith("_json"):
                            decoded = json.loads(value)
                        if _contains_movable(decoded, root, mappings):
                            raise ValueError(
                                f"Scientific field {key} contains private paths; review is required before cutover"
                            )
                        continue
                    try:
                        decoded = json.loads(value)
                    except (ValueError, TypeError):
                        changed = _rewrite(value, root, mappings)
                    else:
                        rewritten = _rewrite(decoded, root, mappings)
                        changed = json.dumps(rewritten) if rewritten != decoded else value
                    if changed != value:
                        changes[key] = changed
                if changes:
                    columns = ", ".join('"' + key.replace('"', '""') + '"=?' for key in changes)
                    db.execute(
                        f"UPDATE {quoted} SET {columns} WHERE rowid=?",
                        (*changes.values(), row["_cutover_rowid"]),
                    )
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Staged registry integrity check failed")


def _stage(source: Path, staged: Path, mappings) -> None:
    staged.mkdir(mode=0o2775)
    _preserve_owner(source, staged)
    for old, new in mappings:
        src, dst = source / old, staged / new
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                copy_function=_copy_file,
                ignore=lambda directory, names: [name for name in names if _ignored(name)],
            )
            for parent, directories, files in os.walk(src):
                directories[:] = [name for name in directories if not _ignored(name)]
                _preserve_owner(Path(parent), dst / Path(parent).relative_to(src))
        else:
            _copy_file(src, dst)
    for path in staged.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(staged)
        if relative.parts[:2] == ("shared", "cache"):
            continue
        if path.name == "registry.sqlite3":
            _rewrite_db(path, source, mappings)
        elif path.suffix == ".json":
            value = json.loads(path.read_text())
            changed = _rewrite(value, source, mappings)
            if changed != value:
                before = path.stat()
                atomic_write_json(path, changed, mode=before.st_mode & 0o7777, durable=True)
                after = path.stat()
                if (before.st_uid, before.st_gid) != (after.st_uid, after.st_gid):
                    os.chown(path, before.st_uid, before.st_gid)
        elif path.suffix in {".yml", ".yaml"}:
            value = yaml.safe_load(path.read_text())
            if _rewrite(value, source, mappings) != value:
                raise ValueError(
                    f"Configuration contains private paths requiring scientific review: {relative}"
                )
    atomic_write_json(
        staged / "shared/cutover.json",
        {"source": str(source), "completed_layout": "shared-branches"},
        durable=True,
    )
    for parent, directories, files in os.walk(staged, topdown=False):
        for name in files:
            with (Path(parent) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(Path(parent))


def _preserve_owner(source: Path, target: Path) -> None:
    before, after = source.stat(), target.stat()
    if (before.st_uid, before.st_gid) != (after.st_uid, after.st_gid):
        os.chown(target, before.st_uid, before.st_gid)
    shutil.copystat(source, target)


def _copy_file(source: Path, target: Path) -> Path:
    shutil.copy2(source, target)
    _preserve_owner(Path(source), Path(target))
    return target


def _load_journal(root: Path) -> dict:
    journal = json.loads(journal_path(root).read_text())
    if journal.get("root") != str(root) or journal.get("state") not in {
        "copying",
        "prepared",
        "published",
    }:
        raise ValueError("Invalid cutover journal")
    token = journal.get("token", "")
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError("Invalid cutover token")
    for key in ("staged", "backup"):
        expected = root.with_name(f"{root.name}.cutover-{key}-{token}")
        if journal.get(key) != str(expected) or expected.resolve() != expected:
            raise ValueError("Cutover journal contains an unexpected destination")
    if all(path.exists() for path in (root, Path(journal["staged"]), Path(journal["backup"]))):
        raise ValueError("Ambiguous cutover directories; manual recovery required")
    return journal


def _save(root: Path, journal: dict) -> None:
    atomic_write_json(journal_path(root), journal, mode=0o664, durable=True)


def execute(root: Path, *, expected_fingerprint: str | None = None) -> Path:
    """Copy and publish a confirmed flat store, or resume a prepared publication.

    Hold the old cache, registry, and branch locks while copying and checking
    quiescence. Keep the original tree as a rollback copy. Interrupted copying
    requires rollback; interrupted publication resumes from its journal. Never
    cancel work, discard an unknown entry, or edit public ownership receipts.
    """
    root = ControlPaths(root).root
    with RegistryLock(
        root.with_name(root.name + ".cutover.lock"),
        root.with_name(root.name + ".cutover.recovery-lock"),
    ):
        pending = journal_path(root).exists()
        if pending:
            journal = _load_journal(root)
            if journal["state"] == "copying":
                raise ValueError("Copying was interrupted; roll back this cutover before retrying")
        else:
            plan = preview(root)
            if expected_fingerprint is not None and expected_fingerprint != plan.source_fingerprint:
                raise ValueError("Private state changed after preview; inspect a new preview")
            if shutil.disk_usage(root.parent).free < plan.bytes + max(
                plan.bytes // 10, 1024 * 1024
            ):
                raise ValueError("Insufficient free space for a verified copy and rollback store")
            token = uuid.uuid4().hex
            journal = dict(
                root=str(root),
                token=token,
                state="copying",
                fingerprint=plan.source_fingerprint,
                mappings=plan.mappings,
                staged=str(root.with_name(f"{root.name}.cutover-staged-{token}")),
                backup=str(root.with_name(f"{root.name}.cutover-backup-{token}")),
            )
        backup, staged = Path(journal["backup"]), Path(journal["staged"])
        source = backup if backup.exists() else root
        with ExitStack() as stack:
            locks = []
            for name, recovery in [
                ("execution-cache.lock", "execution-cache.recovery-lock"),
                ("registry.lock", "registry.lock.recovery"),
            ]:
                locks.append(stack.enter_context(RegistryLock(source / name, source / recovery)))
            if (source / "branches").exists():
                locks.append(
                    stack.enter_context(
                        RegistryLock(
                            source / "branches/edit.lock", source / "branches/edit.recovery-lock"
                        )
                    )
                )
            _quiescent(source)
            if fingerprint(_inventory(source)) != journal["fingerprint"]:
                raise ValueError("Source state changed; publication refused")
            if not pending:
                _save(root, journal)
                _stage(root, staged, journal["mappings"])
                if fingerprint(_inventory(root)) != journal["fingerprint"]:
                    raise ValueError("Source changed while staging; roll back before retrying")
                journal.update(state="prepared", staged_fingerprint=fingerprint(_inventory(staged)))
                _save(root, journal)
            target = root if backup.exists() and root.exists() else staged
            if fingerprint(_inventory(target)) != journal["staged_fingerprint"]:
                raise ValueError("Staged state changed; publication refused")
            if not backup.exists():
                try:
                    root.rename(backup)
                finally:
                    if backup.exists() and not root.exists():
                        for lock in locks:
                            lock.path = backup / lock.path.relative_to(root)
                            lock.recovery_path = backup / lock.recovery_path.relative_to(root)
                sync_directory(root.parent)
            if not root.exists():
                staged.rename(root)
                sync_directory(root.parent)
            journal["state"] = "published"
            _save(root, journal)
        journal_path(root).rename(root / "shared/cutover-journal.json")
        sync_directory(root / "shared")
        sync_directory(root.parent)
        return backup


def rollback(root: Path) -> Path | None:
    """Undo an unfinished cutover; retain staging as evidence instead of deleting it.

    Completed cutovers cannot be rolled back through this command because new
    work may have changed the published state. Coordinate that recovery separately.
    """
    root = ControlPaths(root).root
    with RegistryLock(
        root.with_name(root.name + ".cutover.lock"),
        root.with_name(root.name + ".cutover.recovery-lock"),
    ):
        journal = _load_journal(root)
        backup, staged = Path(journal["backup"]), Path(journal["staged"])
        if backup.exists() and fingerprint(_inventory(backup)) != journal["fingerprint"]:
            raise ValueError("Rollback copy changed; manual recovery required")
        if backup.exists() and root.exists():
            if fingerprint(_inventory(root)) != journal["staged_fingerprint"]:
                raise ValueError("Published state changed; automatic rollback refused")
            root.rename(staged)
        if backup.exists():
            backup.rename(root)
        if not root.is_dir():
            raise ValueError("Original store is missing; manual recovery required")
        journal_path(root).rename(
            root.with_name(root.name + ".cutover-rolled-back-" + journal["token"] + ".json")
        )
        sync_directory(root.parent)
        return staged if staged.exists() else None
