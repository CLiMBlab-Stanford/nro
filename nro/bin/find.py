"""List source BIDS images that match project, participant, and entity selectors."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Pattern

from nro.engine.bids import NON_RUN_ENTITIES, BidsImage, discover_bids_images

_DERIVATIVE_SELECTOR_KEYS = frozenset(
    (NON_RUN_ENTITIES - {"sub"}) | {"model", "model-set", "workflow", "module", "lineage"}
)


def _split_alternatives(value: str) -> tuple[str, ...]:
    """Split top-level commas while preserving regular-expression groups."""
    alternatives: list[str] = []
    current: list[str] = []
    closing: list[str] = []
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}"}
    for character in value:
        if escaped:
            current.extend(("\\", character))
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character in pairs:
            closing.append(pairs[character])
        elif closing and character == closing[-1]:
            closing.pop()
        if character == "," and not closing:
            alternatives.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    alternatives.append("".join(current).strip())
    return tuple(alternatives)


def _compile_patterns(values: Iterable[str], *, label: str) -> tuple[Pattern[str], ...]:
    patterns = []
    for value in values:
        try:
            patterns.append(re.compile(value))
        except re.error as error:
            raise ValueError(
                f"Invalid regular expression for {label}: {value!r}: {error}"
            ) from error
    return tuple(patterns)


def _parse_entity_patterns(
    values: Sequence[str] | None,
) -> dict[str, tuple[Pattern[str], ...] | None]:
    grouped: dict[str, list[str] | None] = {}
    for raw in values or ():
        if "=" not in raw:
            raise ValueError(f"Run selector must use entity=value syntax, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip().removeprefix("--")
        if not key or "_" in key:
            raise ValueError(f"Invalid BIDS entity selector: {raw!r}")
        if key in _DERIVATIVE_SELECTOR_KEYS:
            raise ValueError(f"{key!r} is not a selectable source BIDS image entity")
        alternatives = tuple(item for item in _split_alternatives(value) if item)
        if not alternatives:
            if key in grouped:
                raise ValueError(
                    f"Cannot combine an absent-value selector with other values for {key!r}"
                )
            grouped[key] = None
            continue
        if grouped.get(key) is None and key in grouped:
            raise ValueError(
                f"Cannot combine an absent-value selector with other values for {key!r}"
            )
        grouped.setdefault(key, [])
        assert grouped[key] is not None
        grouped[key].extend(alternatives)
    return {
        key: None if patterns is None else _compile_patterns(patterns, label=key)
        for key, patterns in grouped.items()
    }


def _matches_patterns(
    entities: Mapping[str, str], selectors: Mapping[str, tuple[Pattern[str], ...] | None]
) -> bool:
    for key, patterns in selectors.items():
        value = entities.get(key)
        if patterns is None:
            if value is not None:
                return False
        elif value is None or not any(pattern.fullmatch(value) for pattern in patterns):
            return False
    return True


def _projects(bids_root: Path, requested: Sequence[str]) -> tuple[Path, ...]:
    if requested:
        return tuple(bids_root / project for project in dict.fromkeys(requested))
    return tuple(
        path
        for path in sorted(bids_root.iterdir())
        if path.is_dir() and any(candidate.is_dir() for candidate in path.glob("sub-*"))
    )


def find_images(
    bids_root: Path,
    *,
    projects: Sequence[str] = (),
    participants: Sequence[str] = (),
    runs: Sequence[str] = (),
    tasks: Sequence[str] = (),
) -> tuple[BidsImage, ...]:
    """Return source images matching full regular expressions for selected entities."""
    participant_patterns = _compile_patterns(
        (value.removeprefix("sub-") for value in participants), label="participant"
    )
    selectors = _parse_entity_patterns(runs)
    task_patterns = _compile_patterns(tasks, label="task")
    matches: list[BidsImage] = []
    for project in _projects(bids_root, projects):
        if not project.is_dir():
            continue
        for subject in sorted(path for path in project.glob("sub-*") if path.is_dir()):
            participant = subject.name.removeprefix("sub-")
            if participant_patterns and not any(
                pattern.fullmatch(participant) for pattern in participant_patterns
            ):
                continue
            for image in discover_bids_images(subject):
                if not _matches_patterns(image.entities, selectors):
                    continue
                if task_patterns:
                    task = image.entities.get("task")
                    if task is None or not any(
                        pattern.fullmatch(task) for pattern in task_patterns
                    ):
                        continue
                matches.append(image)
    return tuple(sorted(matches, key=lambda item: str(item.path)))


def build_parser(*, prog: str = "nro.bin.find") -> argparse.ArgumentParser:
    """Build the source-image discovery parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "-p",
        "--participant",
        nargs="+",
        action="extend",
        default=None,
        metavar="REGEX",
        help="Match participant IDs by full regular expression",
    )
    parser.add_argument(
        "-P",
        "--project",
        nargs="+",
        action="extend",
        default=None,
        metavar="PROJECT",
        help="Select exact BIDS project names",
    )
    parser.add_argument(
        "-r",
        "--run",
        nargs="+",
        action="extend",
        default=None,
        metavar="ENTITY=REGEX[,REGEX...]",
        help="Match source-image BIDS entities by full regular expression",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        action="extend",
        default=None,
        metavar="REGEX",
        help="Match task entities by full regular expression",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON array of paths")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.find") -> None:
    """Discover matching source images and print their absolute paths."""
    args = build_parser(prog=prog).parse_args(argv)
    from nro.configuration.site import bids_root

    try:
        matches = find_images(
            bids_root(),
            projects=args.project or (),
            participants=args.participant or (),
            runs=args.run or (),
            tasks=args.task or (),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    paths = [str(image.path.absolute()) for image in matches]
    if args.json:
        print(json.dumps(paths, indent=2))
    elif paths:
        print("\n".join(paths))


if __name__ == "__main__":
    main()
