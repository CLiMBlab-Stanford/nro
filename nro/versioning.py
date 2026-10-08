"""Validate the plain semantic versions used for nro main releases."""

from __future__ import annotations

import importlib.metadata
import os
import re
import tomllib
from pathlib import Path


def application_version() -> str:
    """Read the selected application version from its source or distribution."""
    source = os.environ.get("NRO_EXECUTION_SOURCE_ROOT")
    if source:
        project = Path(source) / "pyproject.toml"
        try:
            value = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
            raise RuntimeError(f"Cannot read nro version from {project}") from error
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Invalid nro version in {project}")
        return value
    return importlib.metadata.version("nro")


def parse_release_version(value: object) -> tuple[int, int, int]:
    """Parse a release version containing only major, minor, and patch numbers."""
    if not isinstance(value, str) or not re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value
    ):
        raise ValueError("Release versions must have the form MAJOR.MINOR.PATCH")
    return tuple(map(int, value.split(".")))


def require_release_advance(previous: str | None, proposed: str) -> None:
    """Require the initial 0.0.1 release or a later semantic version."""
    version = parse_release_version(proposed)
    if previous is None:
        if version != (0, 0, 1):
            raise ValueError("The first main release must be 0.0.1")
    elif version <= parse_release_version(previous):
        raise ValueError("A main release must advance by at least one patch version")
