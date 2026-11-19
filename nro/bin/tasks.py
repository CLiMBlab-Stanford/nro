"""Summarize the BOLD task IDs found in source BIDS projects."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nro.bin.find import find_images
from nro.engine.bids import BidsImage, parse_entity_selectors, resolve_bids_table

_FIXED_ENTITIES = frozenset({"datatype", "suffix"})


@dataclass
class TaskSummary:
    """Count the selected source BOLD runs that share one task ID."""

    task: str
    runs: int = 0
    runs_with_events: int = 0
    participants: set[tuple[str, str]] = field(default_factory=set)
    projects: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        """Return the JSON form; participants are counted per project."""
        return {
            "task": self.task,
            "runs": self.runs,
            "runs_with_events": self.runs_with_events,
            "participants": len(self.participants),
            "projects": dict(sorted(self.projects.items())),
        }


def _has_events(image: BidsImage) -> bool:
    try:
        resolve_bids_table(image.path, suffix="events")
    except (FileNotFoundError, ValueError):
        return False
    return True


def summarize_tasks(bids_root: Path, images: Iterable[BidsImage]) -> tuple[TaskSummary, ...]:
    """Group BOLD images by task, ordered by task ID ignoring case.

    Images outside ``func``, with another suffix, or without a task entity are
    ignored. A run counts as having events when exactly one most specific
    ``events.tsv`` applies to it under BIDS inheritance.
    """
    summaries: dict[str, TaskSummary] = {}
    for image in images:
        task = image.entities.get("task")
        if image.datatype != "func" or image.suffix != "bold" or not task:
            continue
        project = image.path.relative_to(bids_root).parts[0]
        summary = summaries.setdefault(task, TaskSummary(task))
        summary.runs += 1
        summary.runs_with_events += _has_events(image)
        summary.participants.add((project, image.participant))
        summary.projects[project] += 1
    return tuple(sorted(summaries.values(), key=lambda item: (item.task.casefold(), item.task)))


def render_table(summaries: Sequence[TaskSummary]) -> str:
    """Format summaries as aligned columns with one header line."""
    rows = [("TASK", "RUNS", "EVENTS", "PARTICIPANTS", "PROJECTS")]
    rows.extend(
        (
            item.task,
            str(item.runs),
            str(item.runs_with_events),
            str(len(item.participants)),
            " ".join(f"{project}:{count}" for project, count in sorted(item.projects.items())),
        )
        for item in summaries
    )
    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    lines = []
    for row in rows:
        cells = [
            row[0].ljust(widths[0]),
            *(value.rjust(width) for value, width in zip(row[1:4], widths[1:], strict=True)),
            row[4],
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines) + "\n"


def build_parser(*, prog: str = "nro.bin.tasks") -> argparse.ArgumentParser:
    """Build the task-inventory parser."""
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
        help="Match BOLD run entities; values accept globs and re:REGEX",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        action="extend",
        default=None,
        metavar="TASK",
        help="Match task entities; accepts globs and re:REGEX",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="Print a JSON array, or write it to PATH",
    )
    parser.add_argument(
        "--no-pager",
        action="store_true",
        help="Print the table directly instead of opening less",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.tasks") -> None:
    """Count selected source BOLD runs by task and page or export the table."""
    args = build_parser(prog=prog).parse_args(argv)
    from nro.engine.cli import page_text
    from nro.engine.io import atomic_write_text
    from nro.site.configuration import bids_root

    root = bids_root()
    try:
        fixed = _FIXED_ENTITIES.intersection(parse_entity_selectors(args.run))
        if fixed:
            raise ValueError(
                f"nro tasks counts func BOLD runs only; remove {', '.join(sorted(fixed))} from --run"
            )
        images = find_images(
            root,
            projects=args.project or (),
            participants=args.participant or (),
            runs=args.run or (),
            tasks=args.task or (),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    summaries = summarize_tasks(root, images)
    if args.json is not None:
        text = json.dumps([item.as_dict() for item in summaries], indent=2) + "\n"
        if args.json == "-":
            print(text, end="")
        else:
            atomic_write_text(Path(args.json), text)
        return
    if not summaries:
        raise SystemExit("No source BOLD runs match the selection")
    page_text(render_table(summaries), use_pager=not args.no_pager, header_lines=1)


if __name__ == "__main__":
    main()
