"""Repair integrity evidence omitted by the original portable-metadata migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from nro.engine.references import configured_reference_roots
from nro.orchestration.artifact_records import file_record
from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.provenance_migration import (
    _document,
    _ownership_dataset_view,
    _portable_document,
    _serialized,
)

HOTFIX_ID = "v0278-private-portable-metadata"
SUMMARY = "Refresh private artifact evidence missed by the portable-metadata migration."
_METADATA_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


def _canonical_portable_file(path: Path, project_root: Path) -> bool:
    """Return whether a file is the exact canonical output of project migration."""
    try:
        value, kind, _digest = _document(path)
        roots = configured_reference_roots(
            project_root,
            derivative_root=project_root / "derivatives/nro",
        )
        converted = _portable_document(path, value, roots=roots)
        converted = _ownership_dataset_view(path, converted, project_root, roots=roots)
        return path.read_text(encoding="utf-8") == _serialized(converted, kind)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return False


def _matches_record(row, record: dict) -> bool:
    """Compare one stored artifact row with a current filesystem record."""
    if row["size"] is None or row["mtime_ns"] is None:
        return False
    if int(row["size"]) != int(record["size"]):
        return False
    if row["digest_algorithm"] == "sha256" and row["digest"] is not None:
        return row["digest"] == record.get("sha256")
    return int(row["mtime_ns"]) == int(record["mtime_ns"])


def _candidates(database, bids_root: Path, projects: tuple[str, ...]):
    """Find records carrying the exact signature of the historical omission."""
    roots = {
        project: (bids_root / project).resolve()
        for project in sorted(set(projects))
        if (bids_root / project).is_dir()
    }
    found = []
    rows = database.execute(
        """SELECT id,path,size,mtime_ns,digest_algorithm,digest
           FROM artifacts WHERE direction='private' ORDER BY path,id"""
    )
    canonical: dict[Path, bool] = {}
    records: dict[Path, dict] = {}
    for row in rows:
        path = Path(str(row["path"]))
        if path.suffix.lower() not in _METADATA_SUFFIXES:
            continue
        project = next(
            (name for name, root in roots.items() if path.is_relative_to(root / "derivatives/nro")),
            None,
        )
        if project is None or not path.is_file():
            continue
        record = records.get(path)
        if record is None:
            record = file_record(path)
            records[path] = record
        if row["mtime_ns"] is None or int(row["mtime_ns"]) != int(record["mtime_ns"]):
            continue
        if _matches_record(row, record):
            continue
        if not canonical.setdefault(path, _canonical_portable_file(path, roots[project])):
            continue
        found.append((row, project, path, record))
    return found


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Repair only canonical files whose migration-preserved timestamp still matches."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    with registry.connection(write=execute) as database:
        if execute:
            placeholders = ",".join("?" for _ in selected)
            active = database.execute(
                f"""SELECT COUNT(*) FROM attempts AS a
                    JOIN work_items AS w ON w.id=a.work_item_id
                    WHERE w.project IN ({placeholders})
                      AND a.state IN ('queued','running','cancel_requested')""",
                selected,
            ).fetchone()[0]
            if active:
                raise ValueError("Hotfix requires selected-project attempts to be stopped")
        candidates = _candidates(database, registry.paths.bids_root, selected)
        if execute:
            updates = []
            for row, _project, _path, record in candidates:
                updates.append(
                    (
                        record["size"],
                        record["mtime_ns"],
                        "sha256" if "sha256" in record else None,
                        record.get("sha256"),
                        int(row["id"]),
                    )
                )
            database.executemany(
                """UPDATE artifacts SET size=?,mtime_ns=?,digest_algorithm=?,digest=?
                   WHERE id=?""",
                updates,
            )
            grouped: dict[str, list[dict]] = {project: [] for project in selected}
            for _row, project, path, record in candidates:
                grouped[project].append(
                    {"path": str(path), "sha256": record.get("sha256"), "size": record["size"]}
                )
            for project, repaired in grouped.items():
                key = f"hotfix:{HOTFIX_ID}:{project}"
                value = json.dumps(
                    {
                        "hotfix": HOTFIX_ID,
                        "project": project,
                        "records": len(repaired),
                        "evidence_sha256": hashlib.sha256(
                            json.dumps(repaired, sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                database.execute(
                    """INSERT INTO metadata(key,value) VALUES (?,?)
                       ON CONFLICT(key) DO NOTHING""",
                    (key, value),
                )
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=tuple(sorted({path for _row, _project, path, _record in candidates})),
        records=len(candidates),
        applied=execute,
    )
