"""Read the running nro version and validate main-release version numbers."""

from __future__ import annotations

import functools
import os
import re
import subprocess
import tomllib
from pathlib import Path

UNRELEASED = "unreleased"


def application_version() -> str:
    """Report the running nro version, or how unreleased code differs from a release.

    Only code that matches a published release reports a bare version such as
    `0.49.0`. A verified application snapshot reports the version of the
    release recorded for its installation, or `unreleased+source.DIGEST` when
    no recorded release matches it. Code run from a Git checkout reports
    `BASE+dev.COUNT.gHASH`, where BASE is the newest release it contains, with
    `.uncommitted` appended when tracked files have uncommitted changes. Code
    with no release to describe reports `unreleased`.
    """
    source = os.environ.get("NRO_EXECUTION_SOURCE_ROOT")
    if source:
        return _snapshot_version(Path(source), os.environ.get("NRO_EXECUTION_SOURCE_DIGEST"))
    return _checkout_version(Path(__file__).resolve().parents[1])


def _snapshot_version(root: Path, digest: str | None) -> str:
    """Return the recorded release for a matching snapshot."""
    project = root / "pyproject.toml"
    try:
        value = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"Cannot read nro version from {project}") from error
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid nro version in {project}")
    if not digest:
        return UNRELEASED
    for record, key in _release_records():
        release = record.get("release")
        if (
            record.get(key) == digest
            and isinstance(release, dict)
            and release.get("version") == value
        ):
            return value
    return f"{UNRELEASED}+source.{digest[:12]}"


def _release_records():
    """Yield records that bind a release to a source digest.

    The installation record covers commands run from an installed checkout.
    Workers do not run from that checkout, so the central implementation
    record covers them. A record that cannot be read binds no release.
    """
    try:
        from nro.site.configuration import installation_record

        yield installation_record(), "application_digest"
    except (OSError, ValueError):
        pass
    try:
        import json

        from nro.orchestration.scheduler_implementation import implementation_path
        from nro.site.configuration import settings

        path = implementation_path(Path(settings()[0]["registry"]))
        yield json.loads(path.read_text(encoding="utf-8")), "source_digest"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        pass


@functools.lru_cache(maxsize=8)
def _checkout_version(root: Path) -> str:
    """Describe a Git checkout relative to the newest release it contains.

    Releases are tagged on merge commits in `main`, which `dev` does not
    contain. A checkout therefore contains a release when HEAD descends from
    the tagged commit or from the commit merged into it. A clean checkout
    whose tree matches the release tree is that release.
    """

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )

    try:
        head = git("rev-parse", "--verify", "--quiet", "HEAD^{commit}")
        tags = git("tag", "--list", "v[0-9]*.[0-9]*.[0-9]*")
        if head.returncode or tags.returncode:
            return UNRELEASED
        releases = []
        for tag in tags.stdout.split():
            try:
                releases.append((parse_release_version(tag[1:]), tag))
            except ValueError:
                continue
        for _version, tag in sorted(releases, reverse=True):
            commits = [
                result.stdout.strip()
                for result in (
                    git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"),
                    git("rev-parse", "--verify", "--quiet", f"{tag}^2"),
                )
                if not result.returncode
            ]
            base = next(
                (
                    commit
                    for commit in commits
                    if not git("merge-base", "--is-ancestor", commit, "HEAD").returncode
                ),
                None,
            )
            if base is None:
                continue
            uncommitted = bool(git("status", "--porcelain", "--untracked-files=no").stdout)
            same_tree = (
                git("rev-parse", "HEAD^{tree}").stdout == git("rev-parse", f"{tag}^{{tree}}").stdout
            )
            if same_tree and not uncommitted:
                return tag[1:]
            count = git("rev-list", "--count", f"{base}..HEAD").stdout.strip()
            commit = git("rev-parse", "--short=7", "HEAD").stdout.strip()
            suffix = ".uncommitted" if uncommitted else ""
            return f"{tag[1:]}+dev.{count}.g{commit}{suffix}"
    except (OSError, subprocess.SubprocessError):
        return UNRELEASED
    return UNRELEASED


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
