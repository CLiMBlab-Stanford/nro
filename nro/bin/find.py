"""List source BIDS images that match project, participant, and entity selectors."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nro.engine.bids import (
    NON_RUN_ENTITIES,
    BidsImage,
    discover_bids_images,
    matches_selectors,
    parse_entity_selectors,
)
from nro.engine.patterns import is_pattern, matches_any, validate_patterns

_DERIVATIVE_SELECTOR_KEYS = frozenset(
    (NON_RUN_ENTITIES - {"sub"}) | {"model", "model-set", "workflow", "module", "lineage"}
)


def _entity_selectors(values: Sequence[str] | None) -> dict[str, tuple[str, ...] | None]:
    selectors = parse_entity_selectors(values)
    for key in selectors:
        if key in _DERIVATIVE_SELECTOR_KEYS:
            raise ValueError(f"{key!r} is not a selectable source BIDS image entity")
    return selectors


def _projects(bids_root: Path, requested: Sequence[str]) -> tuple[Path, ...]:
    if requested and not any(is_pattern(value) for value in requested):
        return tuple(bids_root / project for project in dict.fromkeys(requested))
    if requested:
        return tuple(
            path
            for path in sorted(bids_root.iterdir())
            if path.is_dir() and matches_any(path.name, requested)
        )
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
    """Return source images matching the selected projects, participants, and entities.

    Every selector value follows the shared rules in ``nro.engine.patterns``.
    """
    participant_patterns = tuple(value.removeprefix("sub-") for value in participants)
    validate_patterns((*projects, *participant_patterns))
    selectors = _entity_selectors(runs)
    task_patterns = tuple(tasks)
    validate_patterns(task_patterns)
    matches: list[BidsImage] = []
    for project in _projects(bids_root, projects):
        if not project.is_dir():
            continue
        for subject in sorted(path for path in project.glob("sub-*") if path.is_dir()):
            participant = subject.name.removeprefix("sub-")
            if participant_patterns and not matches_any(participant, participant_patterns):
                continue
            for image in discover_bids_images(subject):
                if not matches_selectors(image.entities, selectors):
                    continue
                task = image.entities.get("task")
                if task_patterns and (task is None or not matches_any(task, task_patterns)):
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
        metavar="ID",
        help="Select BIDS participant IDs; accepts globs and re:REGEX",
    )
    parser.add_argument(
        "-P",
        "--project",
        nargs="+",
        action="extend",
        default=None,
        metavar="PROJECT",
        help="Select projects; accepts globs and re:REGEX",
    )
    parser.add_argument(
        "-r",
        "--run",
        nargs="+",
        action="extend",
        default=None,
        metavar="ENTITY=VALUE[,VALUE...]",
        help="Match source-image BIDS entities; values accept globs and re:REGEX",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        action="extend",
        default=None,
        metavar="TASK",
        help="Match task entities; accepts globs and re:REGEX",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON array of paths")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.find") -> None:
    """Discover matching source images and print their absolute paths."""
    args = build_parser(prog=prog).parse_args(argv)
    from nro.site.configuration import bids_root

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
