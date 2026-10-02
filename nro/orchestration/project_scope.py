"""Define the project namespaces owned by nro migration operations."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

_BIDS_DATATYPES = frozenset(
    {
        "anat",
        "beh",
        "dwi",
        "eeg",
        "fmap",
        "func",
        "ieeg",
        "meg",
        "micr",
        "motion",
        "mrs",
        "nirs",
        "perf",
        "pet",
    }
)
_OPAQUE_SOURCE_DIRECTORIES = frozenset({"phenotype", "sourcedata", "stimuli"})
_BIDS_LABEL = re.compile(r"[A-Za-z0-9]+")
_ROOT_METADATA = re.compile(
    r"(?:dataset_description\.json|participants\.(?:json|tsv)|samples\.(?:json|tsv)|"
    r"README(?:\.(?:md|rst|txt))?|CHANGES(?:\.(?:md|rst|txt))?|"
    r"LICENSE(?:\.(?:md|rst|txt))?|\.bidsignore|"
    r"(?:[A-Za-z0-9]+-[A-Za-z0-9]+_)+[A-Za-z0-9]+\.(?:json|tsv))"
)


def _entity_directory(name: str, entity: str) -> bool:
    prefix = f"{entity}-"
    return name.startswith(prefix) and bool(_BIDS_LABEL.fullmatch(name[len(prefix) :]))


def _subject_file(name: str, subject: str, session: str | None = None) -> bool:
    prefix = subject if session is None else f"{subject}_{session}"
    return name.startswith(f"{prefix}_")


def is_source_datatype_directory(relative: Path) -> bool:
    """Return whether a project-relative directory directly holds BIDS data files."""
    parts = Path(relative).parts
    return (len(parts) == 2 and parts[1] in _BIDS_DATATYPES) or (
        len(parts) == 3 and _entity_directory(parts[1], "ses") and parts[2] in _BIDS_DATATYPES
    )


def _scope_directory_names(relative: Path, names: list[str]) -> list[str]:
    """Return child directories that belong to the supported source BIDS tree."""
    if not relative.parts:
        return [
            name
            for name in names
            if name in _OPAQUE_SOURCE_DIRECTORIES or _entity_directory(name, "sub")
        ]
    if relative.parts[0] in _OPAQUE_SOURCE_DIRECTORIES:
        return names
    subject = relative.parts[0]
    if not _entity_directory(subject, "sub"):
        return []
    if len(relative.parts) == 1:
        return [name for name in names if name in _BIDS_DATATYPES or _entity_directory(name, "ses")]
    second = relative.parts[1]
    if second in _BIDS_DATATYPES:
        return names
    if not _entity_directory(second, "ses"):
        return []
    if len(relative.parts) == 2:
        return [name for name in names if name in _BIDS_DATATYPES]
    return names if relative.parts[2] in _BIDS_DATATYPES else []


def _scope_file_names(relative: Path, names: list[str]) -> list[str]:
    """Return files that belong to the supported source BIDS tree."""
    if not relative.parts:
        return [
            name
            for name in names
            if _ROOT_METADATA.fullmatch(name)
            or name in _OPAQUE_SOURCE_DIRECTORIES
            or _entity_directory(name, "sub")
        ]
    if relative.parts[0] in _OPAQUE_SOURCE_DIRECTORIES:
        return names
    subject = relative.parts[0]
    if not _entity_directory(subject, "sub"):
        return []
    if len(relative.parts) == 1:
        return [
            name
            for name in names
            if _subject_file(name, subject)
            or name in _BIDS_DATATYPES
            or _entity_directory(name, "ses")
        ]
    second = relative.parts[1]
    if second in _BIDS_DATATYPES:
        return names
    if not _entity_directory(second, "ses"):
        return []
    if len(relative.parts) == 2:
        return [
            name
            for name in names
            if _subject_file(name, subject, second) or name in _BIDS_DATATYPES
        ]
    return names if relative.parts[2] in _BIDS_DATATYPES else []


def source_bids_walk(
    project_root: Path,
    *,
    onerror: Callable[[OSError], None] | None = None,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Walk source BIDS namespaces while excluding code and unmanaged content.

    ``sourcedata``, ``stimuli``, and ``phenotype`` are intentionally opaque:
    BIDS assigns those namespaces but does not impose the participant datatype
    hierarchy used for imaging data.
    """
    project_root = Path(project_root)
    for parent, directories, files in os.walk(
        project_root,
        topdown=True,
        followlinks=False,
        onerror=onerror,
    ):
        relative = Path(parent).relative_to(project_root)
        directories[:] = _scope_directory_names(relative, directories)
        files[:] = _scope_file_names(relative, files)
        yield parent, directories, files
