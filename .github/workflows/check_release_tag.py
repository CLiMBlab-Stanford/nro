"""Validate a version tag before publishing its GitHub Release."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from nro.versioning import parse_release_version


def git(*args: str) -> str:
    """Run a read-only Git query and return its stripped output."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main(argv: list[str]) -> int:
    """Check tag syntax, package version, target commit, and main ancestry."""
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} TAG")
    tag = argv[1]
    if not tag.startswith("v"):
        raise SystemExit("release tags must have the form vMAJOR.MINOR.PATCH")
    tagged_version = tag[1:]
    try:
        parse_release_version(tagged_version)
    except ValueError as error:
        raise SystemExit(error) from error

    project = tomllib.loads(Path("pyproject.toml").read_text())
    packaged_version = project.get("project", {}).get("version")
    if packaged_version != tagged_version:
        raise SystemExit(f"tag {tag} does not match package version {packaged_version!r}")

    head = git("rev-parse", "HEAD")
    if git("cat-file", "-t", f"refs/tags/{tag}") != "tag":
        raise SystemExit(f"release tag {tag} must be annotated")
    target = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    if target != head:
        raise SystemExit(f"tag {tag} does not identify the checked-out commit")
    try:
        git("merge-base", "--is-ancestor", head, "refs/remotes/origin/main")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"tag {tag} does not identify a commit on main") from error
    print(f"validated release tag {tag} at {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
