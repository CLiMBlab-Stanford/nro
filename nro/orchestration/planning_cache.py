"""Content-address compiled plans without weakening artifact freshness checks."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
from typing import Mapping

from nro.configuration.store import fingerprint

PLANNER_CACHE_VERSION = 1
_SOURCE_SUFFIXES = (".json", ".tsv", ".bval", ".bvec", ".nii", ".nii.gz")


def _kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return "nifti-header"
    if name.endswith(_SOURCE_SUFFIXES):
        return "content"
    return None


def _digest(path: Path, kind: str) -> str:
    if kind == "content":
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    opener = gzip.open if path.name.lower().endswith(".nii.gz") else open
    try:
        with opener(path, "rb") as stream:
            header = stream.read(560)
    except (OSError, EOFError):
        with path.open("rb") as stream:
            header = stream.read(560)
    return hashlib.sha256(header).hexdigest()


def _source_files(project_root: Path, subject_dir: Path) -> tuple[Path, ...]:
    files = [path for path in project_root.iterdir() if path.is_file() and _kind(path)]
    for parent, directories, names in os.walk(subject_dir, followlinks=False):
        directories[:] = [
            name for name in directories if not name.startswith(".") and name != "derivatives"
        ]
        files.extend(
            path for name in names if (path := Path(parent) / name).is_file() and _kind(path)
        )
    return tuple(sorted(set(files)))


def _signature(path: Path, kind: str) -> tuple[int, int, str, str]:
    for _attempt in range(3):
        before = path.stat()
        digest = _digest(path, kind)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return before.st_size, before.st_mtime_ns, kind, digest
    raise RuntimeError(f"Source BIDS file changed while planning: {path}")


def participant_source_manifest(registry, project_root: Path, subject_dir: Path) -> str:
    """Fingerprint source structure, metadata contents, and image headers.

    Small metadata files are checksummed on each pass. Cached size and modification
    time avoid reopening unchanged image headers. Workers still assess artifact inputs
    by content.
    """
    files = _source_files(project_root, subject_dir)
    read_records = getattr(registry, "planning_file_records", None)
    write_records = getattr(registry, "record_planning_files", None)
    absolute_paths = tuple(str(path.absolute()) for path in files)
    cached = read_records(absolute_paths) if read_records is not None else {}
    records: dict[str, tuple[int, int, str, str]] = {}
    changed: dict[str, tuple[int, int, str, str]] = {}
    for path, absolute in zip(files, absolute_paths, strict=True):
        kind = _kind(path)
        assert kind is not None
        stat = path.stat()
        existing = cached.get(absolute)
        if (
            kind == "nifti-header"
            and existing is not None
            and existing[:3] == (stat.st_size, stat.st_mtime_ns, kind)
        ):
            record = existing
        else:
            record = _signature(path, kind)
            if record != existing:
                changed[absolute] = record
        records[absolute] = record
    if write_records is not None:
        write_records(changed)
    payload = [
        {
            "path": str(Path(path).relative_to(project_root)),
            "size": record[0],
            "mtime_ns": record[1],
            "kind": record[2],
            "digest": record[3],
        }
        for path, record in sorted(records.items())
    ]
    return fingerprint(payload)


def plan_scope_key(
    *,
    project: str,
    participant: str,
    target: str,
    workflow_id: str,
    selected_runs: tuple[str, ...],
    target_pairs: tuple[tuple[str, int], ...],
    task_models: tuple[str, ...],
    memory_gb: int,
    max_memory_gb: int,
) -> str:
    """Identify one user-visible participant-planning request."""
    return fingerprint(
        {
            "project": project,
            "participant": participant,
            "target": target,
            "workflow": workflow_id,
            "runs": selected_runs,
            "target_pairs": target_pairs,
            "task_models": task_models,
            "memory_gb": memory_gb,
            "max_memory_gb": max_memory_gb,
        }
    )


def plan_cache_key(*, scope_key: str, scientific_inputs: Mapping) -> str:
    """Bind a request scope to every input that can change its compiled graph."""
    return fingerprint(
        {
            "planner_cache": PLANNER_CACHE_VERSION,
            "scope": scope_key,
            "scientific_inputs": scientific_inputs,
        }
    )
