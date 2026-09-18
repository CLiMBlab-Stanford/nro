"""Select validation from declared development boundaries."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TestScope:
    """A source boundary and the tests required when it changes."""

    name: str
    paths: tuple[str, ...]
    tests: tuple[str, ...]
    boundaries: tuple[str, ...]
    downstream: tuple[str, ...]
    full: bool


@dataclass(frozen=True)
class TestSelection:
    """Resolved validation for a set of changed repository paths."""

    spheres: tuple[str, ...]
    tests: tuple[str, ...]
    unmapped: tuple[str, ...]
    full: bool


def load_scopes(path: Path) -> dict[str, TestScope]:
    """Read and validate the tracked change-impact map."""
    path = Path(path)
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1 or not isinstance(document.get("scope"), dict):
        raise ValueError("Unsupported development test-scope document")
    scopes = {}
    for name, value in document["scope"].items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Development sphere {name!r} must be a table")
        scopes[name] = TestScope(
            name=name,
            paths=tuple(map(str, value.get("paths", ()))),
            tests=tuple(map(str, value.get("tests", ()))),
            boundaries=tuple(map(str, value.get("boundaries", ()))),
            downstream=tuple(map(str, value.get("downstream", ()))),
            full=bool(value.get("full", False)),
        )
        if not scopes[name].paths or not scopes[name].tests:
            raise ValueError(f"Development sphere {name!r} needs paths and tests")
    unknown = {
        downstream
        for scope in scopes.values()
        for downstream in scope.downstream
        if downstream not in scopes
    }
    if unknown:
        raise ValueError("Unknown downstream development sphere(s): " + ", ".join(sorted(unknown)))
    root = path.parent.parent
    missing_tests = sorted(
        {
            test
            for scope in scopes.values()
            for test in (*scope.tests, *scope.boundaries)
            if not (root / test).is_file()
        }
    )
    if missing_tests:
        raise ValueError(
            "Development test scope references missing test files: "
            + ", ".join(missing_tests)
        )
    return scopes


def _matches(path: str, pattern: str) -> bool:
    """Match repository paths while treating ``**`` as zero or more directories."""
    if fnmatch.fnmatchcase(path, pattern):
        return True
    return "/**/" in pattern and fnmatch.fnmatchcase(path, pattern.replace("/**/", "/"))


def select_tests(
    scopes: Mapping[str, TestScope],
    changed: Iterable[str],
    *,
    requested: Iterable[str] = (),
) -> TestSelection:
    """Resolve changed paths and explicit spheres to a fail-closed test selection."""
    changed = tuple(dict.fromkeys(str(Path(path)) for path in changed if str(path).strip()))
    selected = set(requested)
    unknown_requested = selected - scopes.keys()
    if unknown_requested:
        raise ValueError("Unknown development sphere(s): " + ", ".join(sorted(unknown_requested)))
    unmapped = []
    for path in changed:
        owners = {
            name for name, scope in scopes.items() if any(_matches(path, p) for p in scope.paths)
        }
        if not owners:
            unmapped.append(path)
        selected.update(owners)
    pending = list(selected)
    while pending:
        name = pending.pop()
        for downstream in scopes[name].downstream:
            if downstream not in selected:
                selected.add(downstream)
                pending.append(downstream)
    full = bool(unmapped) or any(scopes[name].full for name in selected)
    tests = tuple(
        dict.fromkeys(
            test
            for name in sorted(selected)
            for test in (*scopes[name].tests, *scopes[name].boundaries)
        )
    )
    return TestSelection(tuple(sorted(selected)), tests, tuple(sorted(unmapped)), full)


def changed_paths(root: Path) -> tuple[str, ...]:
    """Return every source and destination path changed from ``HEAD``.

    Git reports a rename as one status record with two paths. Both locations
    matter to change-scoped validation: the removed implementation can own a
    different development sphere from its replacement.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=False,
    )
    records = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    paths: list[str] = []
    index = 0
    while index < len(records) and records[index]:
        record = records[index]
        if len(record) < 4:
            raise ValueError(f"Malformed Git status record: {record!r}")
        status = record[:2]
        paths.append(record[3:])
        index += 1
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise ValueError(f"Git status omitted the source of {record!r}")
            paths.append(records[index])
            index += 1
    return tuple(dict.fromkeys(paths))


def run_selection(root: Path, selection: TestSelection, *, full: bool = False) -> int:
    """Run the selected pytest targets, or the complete suite when required."""
    targets = () if full or selection.full else selection.tests
    command = [sys.executable, "-m", "pytest", *(targets or ("tests",))]
    return subprocess.run(command, cwd=root, check=False).returncode
