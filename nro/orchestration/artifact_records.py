"""Create compact filesystem records for completed artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

SMALL_DIGEST_LIMIT = 1024 * 1024


def file_record(path: str | Path, *, include_digest: bool = True) -> dict:
    """Describe one regular file, optionally including a compact-file digest."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if resolved.exists():
            raise ValueError(f"Completion artifact is not a regular file: {resolved}")
        raise ValueError(f"Completion artifact is missing: {resolved}")
    stat = resolved.stat()
    record = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if include_digest and stat.st_size <= SMALL_DIGEST_LIMIT:
        record["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return record


def inventory(paths: Iterable[str | Path]) -> list[dict]:
    """Return deterministic metadata records for unique resolved files."""
    return [
        file_record(path, include_digest=False)
        for path in sorted({Path(path).resolve() for path in paths})
    ]


def inventories_match(left: Iterable[dict], right: Iterable[dict]) -> bool:
    """Compare filesystem evidence while ignoring retired digest fields."""

    def projected(records: Iterable[dict]) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            sorted(
                (
                    str(record["path"]),
                    int(record["size"]),
                    int(record["mtime_ns"]),
                )
                for record in records
            )
        )

    try:
        return projected(left) == projected(right)
    except (KeyError, TypeError, ValueError):
        return False


def read_json_mapping(path: str | Path) -> dict | None:
    """Read one JSON mapping, returning ``None`` for absent or invalid data."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def is_control_artifact(path: str | Path, control_root: str | Path) -> bool:
    """Return whether a path belongs to private orchestration control state."""
    try:
        Path(path).resolve().relative_to(Path(control_root).resolve())
        return True
    except ValueError:
        return False
