"""Reject a proposed main release whose package version did not advance."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from nro.versioning import parse_release_version, require_release_advance


def project_version(document: bytes) -> str:
    """Read the project version from serialized pyproject metadata."""
    value = tomllib.loads(document.decode()).get("project", {}).get("version")
    parse_release_version(value)
    return value


def main(argv: list[str]) -> int:
    """Compare the worktree version with the pyproject at one base commit."""
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} BASE_COMMIT")
    base_document = subprocess.run(
        ["git", "show", f"{argv[1]}:pyproject.toml"],
        check=True,
        capture_output=True,
    ).stdout
    previous = project_version(base_document)
    proposed = project_version(Path("pyproject.toml").read_bytes())
    try:
        require_release_advance(previous, proposed)
    except ValueError as error:
        raise SystemExit(f"{error}: {previous} -> {proposed}") from error
    print(f"main release version advances: {previous} -> {proposed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
