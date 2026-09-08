"""Serialization and atomic file-publication primitives."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Iterator


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def json_path_default(value: Any) -> str | list[Any]:
    """Serialize paths and tuples in manifests written with ``json.dump``."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def manifest_value(value: Any) -> Any:
    """Convert nested path collections into JSON-compatible manifest values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {key: manifest_value(item) for key, item in value.items()}
    return value


def flatten_paths(value: Any) -> list[Path]:
    """Collect paths recursively from nested mappings and sequences."""
    if isinstance(value, Path):
        return [value]
    if isinstance(value, (list, tuple)):
        return [path for item in value for path in flatten_paths(item)]
    if isinstance(value, dict):
        return [path for item in value.values() for path in flatten_paths(item)]
    return []


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a file with metadata before atomically publishing it."""
    with atomic_output_path(destination) as temporary:
        shutil.copy2(source, temporary)


def write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    """Write human-readable JSON atomically."""
    atomic_write_json(path, value, sort_keys=sort_keys)


def atomic_save_npy(path: Path, array: Any) -> None:
    """Write a NumPy array and atomically publish the completed file."""
    import numpy as np

    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)


def atomic_save_npz(
    path: Path,
    *,
    compressed: bool = False,
    **arrays: Any,
) -> None:
    """Write a NumPy archive and atomically publish the completed file."""
    import numpy as np

    save = np.savez_compressed if compressed else np.savez
    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as stream:
            save(stream, **arrays)


def require_nonempty_file(path: Path, description: str) -> None:
    """Reject a missing or empty required file with a user-facing error."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing required file for {description}: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"{description} is empty or invalid: {path}")


def gzip_file_is_valid(path: Path) -> bool:
    """Return whether a path is a nonempty, completely readable gzip file."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
    except (OSError, EOFError):
        return False
    return True


def invalid_gzip_files(paths: Iterable[Path]) -> list[Path]:
    """Return existing ``.nii.gz`` paths that are not readable gzip files."""
    return [
        path
        for path in paths
        if path.name.endswith(".nii.gz") and path.exists() and not gzip_file_is_valid(path)
    ]


def temporary_sibling(path: Path) -> Path:
    """Return a unique hidden sibling while preserving the target's suffixes.

    Several neuroimaging tools infer their output format from compound suffixes
    such as ``.nii.gz``.  Keeping those suffixes on the temporary name lets a
    producer write exactly the intended format before the result is published.
    """
    path = Path(path)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f".{stem}.tmp-{uuid.uuid4().hex}{suffix}")


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Stage one file beside its destination and publish it atomically.

    The caller writes and, when appropriate, validates the yielded path.  The
    canonical destination is replaced only after the context exits normally
    and the staged result is a nonempty regular file.  Interrupted or failed
    writes therefore cannot expose a partial artifact at the canonical path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(path)
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(
                f"Staged output is missing or empty and cannot be published: {temporary}"
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    mode: int | None = None,
    durable: bool = False,
) -> None:
    """Replace a text file atomically after writing it beside the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(text, encoding="utf-8")
        if mode is not None:
            os.chmod(temporary, mode)
        if durable:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    sort_keys: bool = False,
    mode: int | None = None,
    durable: bool = False,
) -> None:
    """Serialize JSON and atomically replace the destination."""
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=sort_keys) + "\n",
        mode=mode,
        durable=durable,
    )
