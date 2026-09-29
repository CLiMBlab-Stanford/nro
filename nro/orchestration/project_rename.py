"""Transactionally rename one BIDS project across nro-managed state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from nro.configuration.branch_definitions import read_selection
from nro.configuration.definition_migrations import MANIFEST, update_store
from nro.configuration.site import make_site_document, read_site_definition, site_definition_path
from nro.configuration.store import fingerprint
from nro.engine.io import atomic_write_json
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.ownership import (
    OWNERSHIP_VERSION,
    ownership_record_fingerprint,
    work_item_record_path,
)
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.registry import ensure_shared_directory

_PROJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_ACTIVE = ("queued", "running", "cancel_requested")


@dataclass(frozen=True)
class ProjectMove:
    """One atomic directory rename in a project migration."""

    source: Path
    destination: Path


def _project(value: str) -> str:
    if not _PROJECT.fullmatch(value):
        raise ValueError(f"Invalid project identifier: {value!r}")
    return value


def _replace_project(value: str, old: str, new: str, key_map: dict[str, str]) -> str:
    """Translate an exact identity or project path without touching prose."""
    if value in key_map:
        return key_map[value]
    if value == old:
        return new
    token = f"/{old}/"
    if token in value:
        value = value.replace(token, f"/{new}/")
    if value.endswith(f"/{old}"):
        value = value[: -len(old)] + new
    return value


def _translate(value: Any, old: str, new: str, key_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_project(value, old, new, key_map)
    if isinstance(value, list):
        return [_translate(item, old, new, key_map) for item in value]
    if isinstance(value, dict):
        return {
            _replace_project(key, old, new, key_map): _translate(item, old, new, key_map)
            for key, item in value.items()
        }
    return value


def _json(value: str | None, old: str, new: str, key_map: dict[str, str]) -> str | None:
    if value is None:
        return None
    return json.dumps(
        _translate(json.loads(value), old, new, key_map),
        sort_keys=True,
        separators=(",", ":"),
    )


def _move_candidates(values: dict, topology, old: str, new: str) -> tuple[ProjectMove, ...]:
    roots = [
        (Path(values["bids"]), Path(values["bids"])),
        (Path(values["work"]), Path(values["work"])),
    ]
    development = Path(values["development"])
    control = ControlPaths(Path(values["registry"]))
    for name in topology.records:
        paths = BranchPaths(name, Path(values["bids"]), Path(values["work"]), development)
        if name != "main":
            roots.extend(
                (
                    (paths.output_bids, paths.output_bids),
                    (paths.private_project(old).parent, paths.private_project(new).parent),
                )
            )
        events = control.branch(name) / "events"
        roots.append((events, events))
    unique: dict[Path, ProjectMove] = {}
    for source_root, destination_root in roots:
        source = source_root / old
        destination = destination_root / new
        unique[source] = ProjectMove(source, destination)
    return tuple(unique.values())


def _moves(values: dict, topology, old: str, new: str) -> tuple[ProjectMove, ...]:
    return tuple(
        move
        for move in _move_candidates(values, topology, old, new)
        if move.source.exists() or move.source.is_symlink()
    )


def _definition_roots(values: dict, store: BranchStore) -> tuple[Path, ...]:
    roots = [Path(values["definitions"]).resolve()]
    for name, record in store.read().topology.records.items():
        selected = read_selection(store.control, name, record.registry_id)
        if selected is not None and selected not in roots:
            roots.append(selected)
    return tuple(roots)


def _definition_updates(
    values: dict, store: BranchStore, old: str, new: str
) -> dict[Path, dict[Path, bytes]]:
    """Build managed definition transactions for project-indexed documents."""
    updates: dict[Path, dict[Path, bytes]] = {}
    for root in _definition_roots(values, store):
        for path in sorted((root / "markup").glob("*_markup.yml")):
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(document, dict) and old in document:
                updates.setdefault(root, {})[path.relative_to(root)] = _rewrite_markup(
                    path, old, new
                ).encode("utf-8")
    shared = Path(values["definitions"]).resolve()
    site = site_definition_path(shared)
    if site.is_file():
        settings_values, bidsify = read_site_definition(shared)
        sources = bidsify.get("project_sources", {})
        if old in sources:
            if new in sources:
                raise ValueError(f"Site definitions already route destination project {new}")
            bidsify = dict(bidsify)
            bidsify["project_sources"] = {
                (new if key == old else key): value for key, value in sources.items()
            }
            document = make_site_document(settings_values, bidsify=bidsify)
            updates.setdefault(shared, {})[site.relative_to(shared)] = yaml.safe_dump(
                document, sort_keys=False
            ).encode("utf-8")
    return updates


def _unhandled_definition_references(
    values: dict,
    store: BranchStore,
    old: str,
    updates: dict[Path, dict[Path, bytes]],
) -> tuple[Path, ...]:
    """Find project mentions outside the definition fields this command understands."""
    handled = {root / relative for root, changes in updates.items() for relative in changes}
    references = []
    for root in _definition_roots(values, store):
        for path in root.rglob("*"):
            if (
                path in handled
                or not path.is_file()
                or path.is_symlink()
                or any(part.startswith(".") for part in path.relative_to(root).parts)
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(old)}(?![A-Za-z0-9_-])", text):
                references.append(path)
    return tuple(sorted(references))


def _ingestion_files(control: ControlPaths, topology, old: str) -> tuple[Path, ...]:
    roots = [control.ingestion]
    roots.extend(control.branch(name) / "ingestion" for name in topology.records if name != "main")
    matched = []
    for root in roots:
        for path in sorted(root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("project") == old:
                matched.append(path)
    return tuple(matched)


def _receipt_files(moves: Iterable[ProjectMove], old: str) -> tuple[Path, ...]:
    matched = []
    for move in moves:
        if move.source.name != old or move.source.parent.name not in {"BIDS", "bids"}:
            continue
        for path in move.source.glob("derivatives/nro/*/*/.nro/work_items/*/*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("project") == old:
                matched.append(path)
    return tuple(sorted(matched))


def _scene_files(moves: Iterable[ProjectMove], old: str) -> tuple[Path, ...]:
    """Return generated Workbench scenes whose absolute links need translation."""
    matched = []
    for move in moves:
        if move.source.name == old and move.source.parent.name in {"BIDS", "bids"}:
            matched.extend(move.source.glob("derivatives/scenes/**/*.scene"))
    return tuple(sorted(path for path in matched if path.is_file() and not path.is_symlink()))


def _metadata_files(moves: Iterable[ProjectMove], old: str) -> tuple[Path, ...]:
    """Find managed structured metadata that embeds absolute project paths."""
    matched = []
    suffixes = {".json", ".yaml", ".yml", ".scene"}
    for move in moves:
        if move.source.name != old or move.source.parent.name not in {"BIDS", "bids", "WORK"}:
            continue
        for path in move.source.rglob("*"):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix.lower() not in suffixes
                or ".nro/work_items" in path.as_posix()
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if f"/{old}/" in text or text.rstrip().endswith(f"/{old}"):
                matched.append(path)
    return tuple(sorted(set(matched)))


def _source_symlinks(bids_root: Path, old: str) -> dict[Path, str]:
    """Inventory raw BIDS links without traversing them or derivative trees."""
    project = bids_root / old
    links = {}
    for parent, directories, files in os.walk(project, followlinks=False):
        directory = Path(parent)
        relative = directory.relative_to(project)
        if relative.parts and relative.parts[0] == "derivatives":
            directories[:] = []
            continue
        for name in tuple(directories):
            path = directory / name
            if path.is_symlink():
                links[path] = os.readlink(path)
                directories.remove(name)
        for name in files:
            path = directory / name
            if path.is_symlink():
                links[path] = os.readlink(path)
    return links


def _copy_or_link(source: str | Path, destination: str | Path) -> str:
    """Hard-link one file when possible, otherwise preserve its metadata in a copy."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def _materialize_source_symlink(path: Path) -> None:
    target = path.resolve(strict=True)
    temporary = path.with_name(f".{path.name}.materialize-{uuid.uuid4().hex}")
    saved = path.with_name(f".{path.name}.symlink-{uuid.uuid4().hex}")
    try:
        if target.is_dir():
            shutil.copytree(target, temporary, symlinks=False, copy_function=_copy_or_link)
        elif target.is_file():
            _copy_or_link(target, temporary)
        else:
            raise ValueError(f"Unsupported raw BIDS symbolic-link target: {path} -> {target}")
        os.replace(path, saved)
        try:
            os.replace(temporary, path)
        except BaseException:
            os.replace(saved, path)
            raise
        saved.unlink()
    finally:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        else:
            temporary.unlink(missing_ok=True)
        saved.unlink(missing_ok=True)


def _restore_source_symlink(path: Path, target: str) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    path.symlink_to(target)


def _restore_recorded_symlink(path: Path, target: str) -> None:
    """Restore a link only when its parent survived an interrupted transaction."""
    if not path.parent.is_dir():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    path.symlink_to(target)


def _absolute_symlinks(
    moves: Iterable[ProjectMove], old: str, new: str, *, excluded: Iterable[Path] = ()
) -> dict[Path, tuple[str, str]]:
    """Find links into renamed roots without following directory links."""
    updates = {}
    excluded = set(excluded)
    for move in moves:
        for parent, directories, files in os.walk(move.source, followlinks=False):
            directory = Path(parent)
            names = [*directories, *files]
            for name in names:
                path = directory / name
                if path in excluded or not path.is_symlink():
                    continue
                target = os.readlink(path)
                if not Path(target).is_absolute():
                    continue
                translated = _replace_project(target, old, new, {})
                if translated != target:
                    updates[path] = (target, translated)
    return updates


def _replace_symlink(path: Path, target: str) -> None:
    temporary = path.with_name(f".{path.name}.project-rename-{uuid.uuid4().hex}")
    try:
        os.symlink(target, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _incomplete_journal(control: ControlPaths, old: str, new: str) -> tuple[Path, dict] | None:
    root = control.shared / "project-renames"
    matches = []
    for path in root.glob("*/journal.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("old") == old
            and record.get("new") == new
            and record.get("state") == "prepared"
        ):
            matches.append((path.stat().st_mtime_ns, path, record))
    if not matches:
        return None
    _mtime, path, record = max(matches)
    return path, record


def _rename_commit_journal(registry) -> Path | None:
    """Return the journal committed with the central SQL transaction, if any."""
    with registry.connection() as db:
        row = db.execute("SELECT value FROM metadata WHERE key='project_rename_commit'").fetchone()
    return None if row is None else Path(str(row[0]))


def _clear_rename_markers(registry) -> None:
    with registry.connection(write=True) as db:
        db.execute(
            """DELETE FROM metadata
               WHERE key='project_rename_commit'
                  OR (key='maintenance_mode' AND value='project rename')"""
        )


def _recover_interrupted(registry, old: str, new: str) -> dict | None:
    """Finish a committed rename or roll an interrupted pre-commit transaction back."""
    committed_journal = _rename_commit_journal(registry)
    found = None
    if committed_journal is not None:
        if not committed_journal.is_file():
            raise RuntimeError(f"Committed project rename journal is missing: {committed_journal}")
        record = json.loads(committed_journal.read_text(encoding="utf-8"))
        if record.get("old") != old or record.get("new") != new:
            raise RuntimeError(
                "A different committed project rename must be recovered first: "
                f"{record.get('old')} -> {record.get('new')} ({committed_journal})"
            )
        found = (committed_journal, record)
    if found is None:
        found = _incomplete_journal(ControlPaths(registry.paths.control), old, new)
    if found is None:
        return None
    journal_path, record = found
    with registry.connection() as db:
        old_registered = bool(
            db.execute("SELECT 1 FROM bids_projects WHERE project=?", (old,)).fetchone()
            or db.execute("SELECT 1 FROM work_items WHERE project=?", (old,)).fetchone()
            or db.execute("SELECT 1 FROM requests WHERE project=?", (old,)).fetchone()
        )
        new_registered = bool(
            db.execute("SELECT 1 FROM bids_projects WHERE project=?", (new,)).fetchone()
            or db.execute("SELECT 1 FROM work_items WHERE project=?", (new,)).fetchone()
            or db.execute("SELECT 1 FROM requests WHERE project=?", (new,)).fetchone()
        )
    moves = tuple(
        ProjectMove(Path(item["source"]), Path(item["destination"]))
        for item in record.get("moves", ())
    )
    source_missing = moves and all(
        not move.source.exists() and not move.source.is_symlink() for move in moves
    )
    destinations_present = moves and all(
        move.destination.exists() or move.destination.is_symlink() for move in moves
    )
    centrally_committed = committed_journal == journal_path
    if (
        (centrally_committed or (new_registered and not old_registered))
        and source_missing
        and destinations_present
    ):
        completed = {**record, "state": "complete", "recovered": True}
        atomic_write_json(journal_path, completed, sort_keys=True, mode=0o664, durable=True)
        _clear_rename_markers(registry)
        return {
            key: value
            for key, value in completed.items()
            if key not in {"state", "file_backups", "scheduler_backup", "branch_backups"}
        } | {"journal": str(journal_path.parent), "executed": True}
    if centrally_committed or (new_registered and not old_registered):
        raise RuntimeError(
            f"Committed project rename has incomplete directory moves; inspect {journal_path}"
        )
    if new_registered and old_registered:
        raise RuntimeError(
            f"Interrupted project rename has ambiguous registry identities; inspect {journal_path}"
        )
    for move in reversed(moves):
        if move.destination.exists() and not move.source.exists():
            move.destination.rename(move.source)
    for item in record.get("absolute_symlink_records", ()):
        _restore_recorded_symlink(Path(item["path"]), str(item["old_target"]))
    for item in record.get("source_symlink_records", ()):
        _restore_recorded_symlink(Path(item["path"]), str(item["target"]))
    for name, backup in record.get("branch_backups", {}).items():
        destination = ControlPaths(registry.paths.control).branch(name) / "registry.sqlite3"
        shutil.copy2(backup, destination)
    for item in record.get("file_backups", ()):
        path, backup = Path(item["path"]), Path(item["backup"])
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)
        if ".nro/work_items" in path.as_posix():
            receipt = json.loads(backup.read_text(encoding="utf-8"))
            new_key = work_item_key(
                new,
                str(receipt["module"]),
                str(receipt["lineage_fingerprint"]),
                str(receipt["participant"]),
                receipt["entities"],
            )
            sibling = path.with_name(new_key.split(":", 1)[-1] + ".json")
            if sibling != path:
                sibling.unlink(missing_ok=True)
    _clear_rename_markers(registry)
    atomic_write_json(
        journal_path,
        {**record, "state": "rolled_back", "recovered": True},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    return None


def _key_maps(db: sqlite3.Connection, old: str, new: str) -> tuple[dict[int, str], dict[str, str]]:
    rows = db.execute(
        """SELECT item.id,item.work_item_key,item.module,item.participant,item.entities_json,
                  lineage.lineage_fingerprint,execution.branch,execution.registry_id,
                  execution.logical_key
           FROM work_items item
           JOIN module_lineages lineage ON lineage.id=item.module_lineage_id
           LEFT JOIN work_item_execution execution ON execution.work_item_id=item.id
           WHERE item.project=? ORDER BY item.id""",
        (old,),
    ).fetchall()
    stored: dict[int, str] = {}
    mapping: dict[str, str] = {}
    for row in rows:
        logical = work_item_key(
            new,
            str(row["module"]),
            str(row["lineage_fingerprint"]),
            str(row["participant"]),
            json.loads(row["entities_json"]),
        )
        scheduler_key = (
            logical if row["branch"] in {None, "main"} else f"{row['registry_id']}:{logical}"
        )
        stored[int(row["id"])] = scheduler_key
        mapping[str(row["work_item_key"])] = scheduler_key
        if row["logical_key"]:
            mapping[str(row["logical_key"])] = logical
    return stored, mapping


def _receipt_key_map(paths: Iterable[Path], new: str) -> dict[str, str]:
    """Derive logical identities for owned artifacts absent from live scheduler state."""
    mapping = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        mapping[str(record["work_item_key"])] = work_item_key(
            new,
            str(record["module"]),
            str(record["lineage_fingerprint"]),
            str(record["participant"]),
            record["entities"],
        )
    return mapping


def preview(registry, *, checkout: Path, values: dict, old: str, new: str) -> dict:
    """Describe a safe rename without changing files or registries."""
    old, new = _project(old), _project(new)
    if old == new:
        raise ValueError("Old and new project identifiers are identical")
    store = BranchStore(registry.paths.control)
    topology = store.read().topology
    if topology.registered_checkout(checkout) != "main":
        raise ValueError("Project renames require the registered main checkout")
    candidates = _move_candidates(values, topology, old, new)
    moves = tuple(move for move in candidates if move.source.exists() or move.source.is_symlink())
    source = Path(values["bids"]) / old
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"Source BIDS project is missing or redirected: {source}")
    conflicts = [
        move.destination
        for move in candidates
        if move.destination.exists() or move.destination.is_symlink()
    ]
    if conflicts:
        raise ValueError(
            "Project rename destinations already exist: "
            + ", ".join(map(str, sorted(set(conflicts))))
        )
    for move in moves:
        if move.source.is_symlink() or move.source.parent.resolve() != move.source.parent:
            raise ValueError(
                f"Managed project root is redirected through a symbolic link: {move.source}"
            )
        if move.source.stat().st_dev != move.destination.parent.stat().st_dev:
            raise ValueError(f"Project rename is not atomic across filesystems: {move.source}")

    with registry.connection() as db:
        active_attempts = int(
            db.execute(
                """SELECT COUNT(*) FROM attempts attempt JOIN work_items item
                 ON item.id=attempt.work_item_id
                 WHERE item.project=? AND attempt.state IN ('queued','running','cancel_requested')""",
                (old,),
            ).fetchone()[0]
        )
        active_steps = int(
            db.execute(
                """SELECT COUNT(*) FROM resource_step_tasks task JOIN work_items item
                 ON item.id=task.work_item_id
                 WHERE item.project=? AND task.state IN ('pending','running')""",
                (old,),
            ).fetchone()[0]
        )
        active_requests = int(
            db.execute(
                "SELECT COUNT(*) FROM requests WHERE project=? AND state='active'", (old,)
            ).fetchone()[0]
        )
        work_items = int(
            db.execute("SELECT COUNT(*) FROM work_items WHERE project=?", (old,)).fetchone()[0]
        )
    ingestion = _ingestion_files(ControlPaths(registry.paths.control), topology, old)
    active_ingestion = 0
    for path in ingestion:
        record = json.loads(path.read_text(encoding="utf-8"))
        active_ingestion += record.get("state") in _ACTIVE
    blockers = []
    if active_attempts or active_steps or active_requests:
        blockers.append(
            f"project has {active_requests} active request(s), {active_attempts} active attempt(s), "
            f"and {active_steps} active resource step(s); stop its work first"
        )
    if active_ingestion:
        blockers.append(f"project has {active_ingestion} active BIDSification request(s)")
    receipts = _receipt_files(moves, old)
    scenes = _scene_files(moves, old)
    metadata = _metadata_files(moves, old)
    source_symlinks = _source_symlinks(Path(values["bids"]), old)
    symlinks = _absolute_symlinks(moves, old, new, excluded=source_symlinks)
    definition_updates = _definition_updates(values, store, old, new)
    unhandled = _unhandled_definition_references(values, store, old, definition_updates)
    if unhandled:
        blockers.append(
            "definitions contain project references outside supported markup or "
            "BIDSification routing fields: " + ", ".join(map(str, unhandled))
        )
    return {
        "old": old,
        "new": new,
        "moves": [
            {"source": str(item.source), "destination": str(item.destination)} for item in moves
        ],
        "work_items": work_items,
        "ownership_receipts": len(receipts),
        "scene_files": len(scenes),
        "metadata_files": len(metadata),
        "source_symlinks": len(source_symlinks),
        "source_symlink_records": [
            {"path": str(path), "target": target}
            for path, target in sorted(source_symlinks.items())
        ],
        "absolute_symlinks": len(symlinks),
        "absolute_symlink_records": [
            {"path": str(path), "old_target": targets[0], "new_target": targets[1]}
            for path, targets in sorted(symlinks.items())
        ],
        "ingestion_records": len(ingestion),
        "definition_files": [
            str(root / relative)
            for root, changes in definition_updates.items()
            for relative in changes
        ],
        "unhandled_definition_references": [str(path) for path in unhandled],
        "blockers": blockers,
    }


def _rewrite_markup(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text) or {}
    if not isinstance(document, dict) or old not in document:
        return text
    if new in document:
        raise ValueError(f"Markup already defines destination project {new}: {path}")
    pattern = re.compile(rf"(?m)^{re.escape(old)}:(?=\s*(?:#.*)?$)")
    replaced, count = pattern.subn(f"{new}:", text)
    if count != 1:
        raise ValueError(f"Cannot safely rename the top-level markup key in {path}")
    parsed = yaml.safe_load(replaced) or {}
    if old in parsed or new not in parsed:
        raise ValueError(f"Markup project rename did not validate: {path}")
    return replaced


def _rewrite_receipts(
    receipt_paths: Iterable[Path], old: str, new: str, key_map: dict[str, str]
) -> tuple[int, tuple[Path, ...]]:
    changed = 0
    created = []
    for old_path in receipt_paths:
        record = json.loads(old_path.read_text(encoding="utf-8"))
        translated = _translate(record, old, new, key_map)
        if translated.get("record_version") == OWNERSHIP_VERSION:
            translated["record_fingerprint"] = ownership_record_fingerprint(translated)
        new_key = str(translated["work_item_key"])
        new_path = work_item_record_path(
            old_path.parents[7],
            old_path.parents[4].name,
            str(translated["directory_label"]),
            str(translated["module"]),
            new_key,
        )
        # The containing project is renamed later; retain its current root here.
        if new_path != old_path and new_path.exists():
            raise ValueError(f"Renamed ownership receipt already exists: {new_path}")
        atomic_write_json(new_path, translated, sort_keys=True, mode=0o664, durable=True)
        if new_path != old_path:
            old_path.unlink()
            created.append(new_path)
        changed += 1
    return changed, tuple(created)


def _rewrite_branch_registry(scientific, mapping: dict[str, str], old: str, new: str) -> int:
    changed = 0
    with scientific.connection(write=True) as db:
        rows = db.execute("SELECT * FROM work_items").fetchall()
        for row in rows:
            contract = json.loads(row["contract_json"])
            if contract.get("project") != old:
                continue
            key = mapping.get(str(row["work_item_key"]))
            if key is None:
                # A branch registry may retain planning history after both its
                # scheduler record and artifact were purged. It has no durable
                # identity to migrate and should not survive the rename.
                db.execute("DELETE FROM work_items WHERE work_item_key=?", (row["work_item_key"],))
                changed += 1
                continue
            translated = _translate(contract, old, new, mapping)
            observation = _json(row["observation_json"], old, new, mapping)
            encoded = json.dumps(translated, sort_keys=True, separators=(",", ":"))
            db.execute(
                """UPDATE work_items SET work_item_key=?,contract_json=?,
                          contract_fingerprint=?,observation_json=? WHERE work_item_key=?""",
                (key, encoded, fingerprint(translated), observation, row["work_item_key"]),
            )
            changed += 1
        # Planning caches are performance data keyed by source paths and project identity.
        for table in ("planning_cache", "planning_files"):
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone():
                db.execute(f"DELETE FROM {table}")
    return changed


def _rewrite_central(
    db: sqlite3.Connection,
    old: str,
    new: str,
    *,
    changed_metadata: Iterable[Path] = (),
) -> tuple[int, dict[str, str]]:
    stored, mapping = _key_maps(db, old, new)
    if db.execute("SELECT 1 FROM bids_projects WHERE project=?", (new,)).fetchone():
        raise ValueError(f"Destination project is already registered: {new}")
    project = db.execute("SELECT * FROM bids_projects WHERE project=?", (old,)).fetchone()
    if project is not None:
        db.execute(
            "INSERT INTO bids_projects(project,path,discovered_at) VALUES (?,?,?)",
            (
                new,
                _replace_project(str(project["path"]), old, new, mapping),
                project["discovered_at"],
            ),
        )
        db.execute(
            "UPDATE bids_participants SET project=?,path=replace(path,?,?) WHERE project=?",
            (new, f"/{old}/", f"/{new}/", old),
        )
    json_columns = {
        "requests": (("selectors_json",), "project=?"),
        "work_items": (
            (
                "entities_json",
                "artifact_contract_json",
                "command_json",
                "input_paths_json",
                "expected_outputs_json",
            ),
            "project=?",
        ),
        "artifacts": (
            ("metadata_json",),
            "work_item_id IN (SELECT id FROM work_items WHERE project=?)",
        ),
        "completions": (
            ("artifact_contract_json", "provenance_json", "command_json"),
            "work_item_id IN (SELECT id FROM work_items WHERE project=?)",
        ),
        "work_item_execution": (
            (
                "context_json",
                "binding_sources_json",
                "provenance_json",
                "scientific_contract_json",
            ),
            "work_item_id IN (SELECT id FROM work_items WHERE project=?)",
        ),
        "request_plans": (
            ("payload_json",),
            "request_id IN (SELECT id FROM requests WHERE project=?)",
        ),
        "branch_work_items": (
            ("scientific_contract_json",),
            "work_item_id IN (SELECT id FROM work_items WHERE project=?)",
        ),
        "attempt_execution": (
            ("context_json", "provenance_json", "command_json"),
            "attempt_id IN (SELECT attempt.id FROM attempts attempt JOIN work_items item ON item.id=attempt.work_item_id WHERE item.project=?)",
        ),
    }
    for table, (columns, where) in json_columns.items():
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            continue
        primary = "_nro_rowid"
        rows = db.execute(
            f"SELECT rowid AS {primary},* FROM {table} WHERE {where}", (old,)
        ).fetchall()
        for row in rows:
            updates = {column: _json(row[column], old, new, mapping) for column in columns}
            if all(updates[column] == row[column] for column in columns):
                continue
            clause = ",".join(f"{column}=?" for column in columns)
            db.execute(
                f"UPDATE {table} SET {clause} WHERE rowid=?",
                (*updates.values(), row[primary]),
            )
    path_columns = {
        "work_items": (("runtime_config_path", "output_root", "artifact_reason"), "project=?"),
        "artifacts": (("path",), "work_item_id IN (SELECT id FROM work_items WHERE project=?)"),
        "attempts": (
            ("log_path", "error_message"),
            "work_item_id IN (SELECT id FROM work_items WHERE project=?)",
        ),
    }
    for table, (columns, where) in path_columns.items():
        for column in columns:
            db.execute(
                f"UPDATE {table} SET {column}=replace({column},?,?) "
                f"WHERE {where} AND {column} LIKE ?",
                (f"/{old}/", f"/{new}/", old, f"%/{old}/%"),
            )
    for item_id, key in stored.items():
        db.execute(
            "UPDATE work_items SET work_item_key=?,project=? WHERE id=?", (key, new, item_id)
        )
    for table in ("work_item_execution", "compiled_revisions", "branch_work_items"):
        rows = db.execute(f"SELECT rowid,logical_key FROM {table}").fetchall()
        for row in rows:
            if row["logical_key"] in mapping:
                db.execute(
                    f"UPDATE {table} SET logical_key=? WHERE rowid=?",
                    (mapping[row["logical_key"]], row["rowid"]),
                )
    for row in db.execute(
        "SELECT registry_id,logical_key,scientific_contract_json FROM branch_work_items"
    ):
        contract = json.loads(row["scientific_contract_json"])
        if contract.get("project") != new:
            continue
        db.execute(
            "UPDATE compiled_revisions SET fingerprint=? WHERE registry_id=? AND logical_key=?",
            (fingerprint(contract), row["registry_id"], row["logical_key"]),
        )
    db.execute("UPDATE requests SET project=? WHERE project=?", (new, old))
    for row in db.execute(
        "SELECT id,artifact_contract_json FROM work_items WHERE project=?", (new,)
    ):
        value = json.loads(row["artifact_contract_json"])
        db.execute(
            "UPDATE work_items SET artifact_fingerprint=? WHERE id=?",
            (fingerprint(value), row["id"]),
        )
    for row in db.execute(
        """SELECT completion.work_item_id,completion.artifact_contract_json
             FROM completions completion JOIN work_items item ON item.id=completion.work_item_id
             WHERE item.project=?""",
        (new,),
    ):
        value = json.loads(row["artifact_contract_json"])
        db.execute(
            "UPDATE completions SET artifact_fingerprint=? WHERE work_item_id=?",
            (fingerprint(value), row["work_item_id"]),
        )
    changed_paths = {str(path) for path in changed_metadata}
    if changed_paths:
        for row in db.execute(
            """SELECT artifact.id,artifact.path,artifact.digest_algorithm
                 FROM artifacts artifact JOIN work_items item
                   ON item.id=artifact.work_item_id
                 WHERE item.project=?""",
            (new,),
        ):
            if row["path"] not in changed_paths:
                continue
            path = Path(row["path"])
            stat = path.stat()
            digest = None
            if row["digest_algorithm"] == "sha256":
                checksum = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        checksum.update(chunk)
                digest = checksum.hexdigest()
            elif row["digest_algorithm"] is not None:
                raise ValueError(
                    f"Unsupported artifact digest during project rename: {row['digest_algorithm']}"
                )
            db.execute(
                "UPDATE artifacts SET size=?,mtime_ns=?,digest=? WHERE id=?",
                (stat.st_size, stat.st_mtime_ns, digest, row["id"]),
            )
    if project is not None:
        db.execute("DELETE FROM bids_projects WHERE project=?", (old,))
    return len(stored), mapping


def execute(registry, *, checkout: Path, values: dict, old: str, new: str) -> dict:
    """Apply a previously previewable rename and retain a recovery journal."""
    old, new = _project(old), _project(new)
    recovered = _recover_interrupted(registry, old, new)
    if recovered is not None:
        return recovered
    report = preview(registry, checkout=checkout, values=values, old=old, new=new)
    if report["blockers"]:
        raise ValueError("Project rename is blocked: " + "; ".join(report["blockers"]))
    old, new = report["old"], report["new"]
    from nro.orchestration.planner_client import shutdown as shutdown_planner

    shutdown_planner(registry.paths.control)
    store = BranchStore(registry.paths.control)
    topology = store.read().topology
    moves = tuple(
        ProjectMove(Path(item["source"]), Path(item["destination"])) for item in report["moves"]
    )
    receipts = _receipt_files(moves, old)
    metadata = _metadata_files(moves, old)
    source_symlinks = _source_symlinks(Path(values["bids"]), old)
    symlinks = _absolute_symlinks(moves, old, new, excluded=source_symlinks)
    ingestion = _ingestion_files(ControlPaths(registry.paths.control), topology, old)
    definition_updates = _definition_updates(values, store, old, new)
    journal = ControlPaths(registry.paths.control).shared / "project-renames" / uuid.uuid4().hex
    ensure_shared_directory(journal)
    atomic_write_json(
        journal / "journal.json",
        {**report, "state": "prepared"},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )

    # Databases and edited metadata are inexpensive to back up. Project directories
    # are rolled back by reversing their same-filesystem atomic renames.
    with registry.connection() as db:
        backup = sqlite3.connect(journal / "scheduler.sqlite3")
        try:
            db.backup(backup)
        finally:
            backup.close()
    (journal / "scheduler.sqlite3").chmod(0o664)
    branch_backups: dict[str, Path] = {}
    for name in topology.records:
        scientific = store.registry(name)
        backup = journal / f"branch-{scientific.record.registry_id}.sqlite3"
        with scientific.connection() as db:
            copy = sqlite3.connect(backup)
            try:
                db.backup(copy)
            finally:
                copy.close()
        backup.chmod(0o664)
        branch_backups[name] = backup
    file_backups: dict[Path, Path] = {}
    definition_paths = tuple(
        root / relative for root, changes in definition_updates.items() for relative in changes
    )
    manifests = tuple(root / MANIFEST for root in definition_updates if (root / MANIFEST).is_file())
    for index, path in enumerate((*definition_paths, *manifests, *ingestion, *receipts, *metadata)):
        backup = journal / "files" / f"{index:08d}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        file_backups[path] = backup
    journal_record = {
        **report,
        "file_backups": [
            {"path": str(path), "backup": str(backup)} for path, backup in file_backups.items()
        ],
        "scheduler_backup": str(journal / "scheduler.sqlite3"),
        "branch_backups": {name: str(path) for name, path in branch_backups.items()},
    }
    atomic_write_json(
        journal / "journal.json",
        {**journal_record, "state": "prepared"},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )

    moved: list[ProjectMove] = []
    created_receipts: tuple[Path, ...] = ()
    with registry.connection(write=True) as db:
        existing = db.execute("SELECT value FROM metadata WHERE key='maintenance_mode'").fetchone()
        if existing is not None:
            raise ValueError(f"Shared registry is already in {existing[0]} maintenance")
        db.execute("INSERT INTO metadata(key,value) VALUES ('maintenance_mode','project rename')")
    completed = False
    changed = report["work_items"]
    try:
        final_preview = preview(registry, checkout=checkout, values=values, old=old, new=new)
        if final_preview["blockers"]:
            raise ValueError(
                "Project rename became blocked: " + "; ".join(final_preview["blockers"])
            )
        with registry.connection() as db:
            _stored, mapping = _key_maps(db, old, new)
        mapping.update(_receipt_key_map(receipts, new))
        for name, record in topology.records.items():
            local_mapping = {
                key: value.split(":", 1)[-1]
                if value.startswith(record.registry_id + ":")
                else value
                for key, value in mapping.items()
            }
            _rewrite_branch_registry(store.registry(name), local_mapping, old, new)
        for root, updates in definition_updates.items():
            update_store(root, updates)
        for path in ingestion:
            value = _translate(json.loads(path.read_text(encoding="utf-8")), old, new, mapping)
            atomic_write_json(path, value, sort_keys=True, mode=0o660, durable=True)
        for path in source_symlinks:
            _materialize_source_symlink(path)
        metadata = tuple(sorted(set(metadata) | set(_metadata_files(moves, old))))
        for path in metadata:
            text = path.read_text(encoding="utf-8")
            translated = _replace_project(text, old, new, mapping)
            if translated != text:
                from nro.engine.io import atomic_write_text

                atomic_write_text(
                    path,
                    translated,
                    mode=path.stat().st_mode & 0o777,
                    durable=True,
                )
        _receipt_count, created_receipts = _rewrite_receipts(receipts, old, new, mapping)
        for path, (_old_target, new_target) in symlinks.items():
            _replace_symlink(path, new_target)
        for move in moves:
            move.source.rename(move.destination)
            moved.append(move)
        changed_metadata = tuple(
            Path(_replace_project(str(path), old, new, {})) for path in metadata
        )
        with registry.connection(write=True) as db:
            changed, _committed_mapping = _rewrite_central(
                db, old, new, changed_metadata=changed_metadata
            )
            db.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES ('project_rename_commit',?)",
                (str(journal / "journal.json"),),
            )
        completed = True
    except BaseException:
        if _rename_commit_journal(registry) == journal / "journal.json":
            completed = True
        else:
            for move in reversed(moved):
                if move.destination.exists() and not move.source.exists():
                    move.destination.rename(move.source)
            for path, (old_target, _new_target) in symlinks.items():
                if path.is_symlink():
                    _replace_symlink(path, old_target)
            for path, target in source_symlinks.items():
                _restore_source_symlink(path, target)
            for name, backup in branch_backups.items():
                shutil.copy2(backup, store.registry(name).database)
            for path in created_receipts:
                path.unlink(missing_ok=True)
            for path, backup in file_backups.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)
            _clear_rename_markers(registry)
            atomic_write_json(
                journal / "journal.json",
                {**journal_record, "state": "rolled_back"},
                sort_keys=True,
                mode=0o664,
                durable=True,
            )
            raise
    finally:
        if not completed:
            _clear_rename_markers(registry)
    if not completed:
        raise RuntimeError("Project rename did not reach its commit point")
    atomic_write_json(
        journal / "journal.json",
        {**journal_record, "state": "complete"},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    _clear_rename_markers(registry)
    return {**report, "work_items": changed, "journal": str(journal), "executed": True}
