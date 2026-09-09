"""Validate the plain semantic versions used for nro main releases."""

from __future__ import annotations

import re


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
