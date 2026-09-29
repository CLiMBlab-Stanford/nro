"""Transactional conversion of nro public metadata to portable references."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from nro.engine.io import atomic_write_json, atomic_write_text
from nro.engine.references import (
    derivative_dataset_description,
    portable_public_payload,
    public_document_roots,
)
from nro.orchestration.artifact_records import file_record
from nro.orchestration.branches import BranchPaths
from nro.orchestration.ownership import (
    LEGACY_OWNERSHIP_VERSION,
    LINEAGE_RECORD_NAME,
    OWNERSHIP_DIRECTORY,
    OWNERSHIP_VERSION,
    convert_legacy_ownership_record,
    read_ownership_records,
)


@dataclass(frozen=True)
class ProvenanceMigrationReport:
    """Preview or outcome of one public-provenance conversion."""

    scanned: int
    changed: tuple[Path, ...]
    errors: tuple[str, ...]


def _bids_roots(registry, site_values: Mapping[str, object] | None) -> tuple[Path, ...]:
    """Return main and registered branch-owned public BIDS roots."""
    roots = [Path(registry.paths.bids_root)]
    if site_values is None:
        return tuple(roots)
    required = ("bids", "work", "development")
    if not all(str(site_values.get(key, "")).strip() for key in required):
        return tuple(roots)
    from nro.orchestration.branch_store import BranchStore

    try:
        topology = BranchStore(registry.paths.control).read().topology
    except (FileNotFoundError, ValueError):
        return tuple(roots)
    for name in sorted(topology.records):
        if name == "main":
            continue
        paths = BranchPaths(name, *(Path(str(site_values[key])) for key in required))
        roots.append(paths.output_bids)
    return tuple(dict.fromkeys(path.absolute() for path in roots))


def _journal_root(registry) -> Path:
    return Path(registry.paths.control) / "shared" / "provenance-migrations"


def _refresh_inventory(registry, paths: Iterable[Path]) -> None:
    """Align private integrity evidence with restored or converted metadata."""
    with registry.connection(write=True) as database:
        for path in paths:
            resolved = str(path.resolve())
            if not path.is_file():
                database.execute(
                    "DELETE FROM artifacts WHERE direction='output' AND path=?", (resolved,)
                )
                continue
            record = file_record(path)
            database.execute(
                """UPDATE artifacts SET size=?,mtime_ns=?,digest_algorithm=?,digest=?
                   WHERE direction='output' AND path=?""",
                (
                    record["size"],
                    record["mtime_ns"],
                    "sha256" if "sha256" in record else None,
                    record.get("sha256"),
                    resolved,
                ),
            )


def _restore_journal(registry, journal: Path, record: dict) -> None:
    """Roll back one prepared or interrupted public-metadata transaction."""
    restored = []
    for entry in reversed(record.get("files", [])):
        path = Path(entry["path"])
        if entry["existed"]:
            backup = journal / entry["backup"]
            if not backup.is_file():
                raise RuntimeError(f"Portable provenance backup is missing: {backup}")
            shutil.copy2(backup, path)
        else:
            path.unlink(missing_ok=True)
        restored.append(path)
    _refresh_inventory(registry, restored)
    atomic_write_json(
        journal / "journal.json",
        {**record, "state": "rolled_back"},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )


def _recover_interrupted(registry) -> None:
    """Restore any conversion that did not reach its durable commit marker."""
    root = _journal_root(registry)
    if not root.is_dir():
        return
    for journal in sorted(path for path in root.iterdir() if path.is_dir()):
        marker = journal / "journal.json"
        if not marker.is_file():
            raise RuntimeError(f"Portable provenance journal is incomplete: {journal}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        state = record.get("state")
        if state not in {"complete", "rolled_back"}:
            _restore_journal(registry, journal, record)
        shutil.rmtree(journal)


def _unfinished_journals(registry) -> tuple[Path, ...]:
    """Return migration journals that require execution-time recovery."""
    root = _journal_root(registry)
    if not root.is_dir():
        return ()
    unfinished = []
    for journal in sorted(path for path in root.iterdir() if path.is_dir()):
        marker = journal / "journal.json"
        if not marker.is_file():
            unfinished.append(journal)
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("state") not in {"complete", "rolled_back"}:
            unfinished.append(journal)
    return tuple(unfinished)


def _document(path: Path) -> tuple[object, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text), "json"
    return yaml.safe_load(text), "yaml"


def _serialized(value: object, kind: str) -> str:
    if kind == "json":
        return json.dumps(value, indent=2, sort_keys=True) + "\n"
    return yaml.safe_dump(value, sort_keys=False)


def _portable_document(path: Path, value: object) -> object:
    if path.name == "dataset_description.json":
        return value
    if OWNERSHIP_DIRECTORY in path.parts:
        if not isinstance(value, dict):
            raise ValueError("ownership document is not a mapping")
        version = value.get("record_version")
        if version == OWNERSHIP_VERSION:
            return value
        if version != LEGACY_OWNERSHIP_VERSION:
            raise ValueError(f"unsupported ownership record version {version!r}")
        configuration_class = None
        if path.name == LINEAGE_RECORD_NAME:
            configuration_class = str(value.get("configuration_class") or "")
        return convert_legacy_ownership_record(
            value,
            roots=public_document_roots(path),
            configuration_class=configuration_class,
        )
    return portable_public_payload(path, value)


def _candidates(project_root: Path) -> tuple[Path, ...]:
    root = project_root / "derivatives" / "nro"
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
        )
    )


def migrate_public_provenance(
    registry,
    *,
    projects: Iterable[str],
    execute: bool = False,
    version: str,
    site_values: Mapping[str, object] | None = None,
) -> ProvenanceMigrationReport:
    """Preview or atomically apply portable-reference metadata conversion.

    The database transaction refreshes only integrity evidence for rewritten
    metadata. Work-item contracts, generations, states, and dependency edges are
    intentionally unchanged.
    """
    replacements: dict[Path, str] = {}
    errors: list[str] = []
    scanned = 0
    selected = tuple(sorted(set(projects)))
    if execute:
        with registry.connection() as database:
            active = int(
                database.execute(
                    """SELECT COUNT(*) FROM attempts
                       WHERE state IN ('queued','running','cancel_requested')"""
                ).fetchone()[0]
            )
        if active:
            raise ValueError("Portable provenance migration requires all attempts to be stopped")
        _recover_interrupted(registry)
    else:
        errors.extend(
            f"Interrupted migration requires --execute recovery: {path}"
            for path in _unfinished_journals(registry)
        )
    bids_roots = _bids_roots(registry, site_values)
    project_roots = tuple(root / project for root in bids_roots for project in selected)
    for project_root in project_roots:
        derivative_root = project_root / "derivatives/nro"
        if not derivative_root.is_dir():
            continue
        for path in _candidates(project_root):
            scanned += 1
            try:
                value, kind = _document(path)
                converted = _portable_document(path, value)
            except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
                errors.append(f"{path}: {error}")
                continue
            if converted != value:
                replacements[path] = _serialized(converted, kind)
        description = project_root / "derivatives/nro/dataset_description.json"
        try:
            expected_description = derivative_dataset_description(project_root, version=version)
        except ValueError as error:
            message = f"{description}: {error}"
            if message not in errors:
                errors.append(message)
            continue
        current_description = None
        if description.is_file():
            try:
                current_description = json.loads(description.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{description}: {error}")
        if current_description != expected_description:
            replacements[description] = _serialized(expected_description, "json")
    if errors or not execute:
        return ProvenanceMigrationReport(scanned, tuple(replacements), tuple(errors))
    if not replacements:
        return ProvenanceMigrationReport(scanned, (), ())
    root = _journal_root(registry)
    root.mkdir(parents=True, exist_ok=True, mode=0o2775)
    journal = root / uuid.uuid4().hex
    backups = journal / "files"
    backups.mkdir(parents=True, mode=0o2775)
    entries = []
    for index, path in enumerate(replacements):
        relative = f"files/{index:08d}"
        existed = path.is_file()
        if existed:
            shutil.copy2(path, journal / relative)
        entries.append({"path": str(path.resolve()), "backup": relative, "existed": existed})
    journal_record = {
        "format": 1,
        "state": "prepared",
        "projects": list(selected),
        "files": entries,
    }
    atomic_write_json(
        journal / "journal.json",
        journal_record,
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    try:
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "applying"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
        for path, rendered in replacements.items():
            existed = path.is_file()
            stat = path.stat() if existed else None
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
            atomic_write_text(
                path,
                rendered,
                mode=(stat.st_mode & 0o777) if stat is not None else 0o664,
                durable=True,
            )
            if stat is not None:
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        validation_errors = []
        for bids_root in bids_roots:
            represented = tuple(
                project for project in selected if (bids_root / project / "derivatives/nro").is_dir()
            )
            if not represented:
                continue
            lineages, receipts, found = read_ownership_records(
                bids_root,
                represented,
                source_bids_root=registry.paths.bids_root,
            )
            _ = (lineages, receipts)
            validation_errors.extend(found)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        _refresh_inventory(registry, replacements)
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "complete"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
    except Exception:
        _restore_journal(registry, journal, journal_record)
        raise
    finally:
        marker = journal / "journal.json"
        if marker.is_file():
            state = json.loads(marker.read_text(encoding="utf-8")).get("state")
            if state in {"complete", "rolled_back"}:
                shutil.rmtree(journal)
    return ProvenanceMigrationReport(scanned, tuple(replacements), ())
