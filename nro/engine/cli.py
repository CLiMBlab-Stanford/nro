"""Command-line selection vocabulary and output primitives."""

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

from nro.engine.bids import parse_selectors
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE


@dataclass(frozen=True)
class CoreSelection:
    participants: tuple[str, ...]
    projects: tuple[str, ...]
    modules: tuple[str, ...]
    workflows: tuple[str, ...]
    runs: dict[str, tuple[str, ...] | None]
    spaces: tuple[str, ...]
    smoothing: tuple[int, ...]

    @property
    def instance_entities(self) -> dict[str, tuple[str, ...] | None]:
        entities = dict(self.runs)
        if self.spaces:
            entities["space"] = self.spaces
        if self.smoothing:
            entities["smoothing"] = tuple(str(value) for value in self.smoothing)
        return entities


def add_core_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    module_choices: Sequence[str],
    planner_defaults: bool = False,
    default_module: str = "networks",
    default_workflow: str = "main",
) -> None:
    """Add the selection options shared by user-facing orchestration tools."""
    parser.add_argument(
        "-p", "--participant", nargs="+", action="extend", default=None,
        metavar="ID", help="Select one or more BIDS participant IDs",
    )
    parser.add_argument(
        "-P", "--project", nargs="+", action="extend", default=None,
        metavar="PROJECT", help="Select one or more projects",
    )
    parser.add_argument(
        "-m", "--module", nargs="+", action="extend", choices=module_choices,
        default=None, metavar="MODULE", help="Select one or more modules",
    )
    parser.add_argument(
        "-w", "--workflow", nargs="+", action="extend", default=None,
        metavar="WORKFLOW", help="Select one or more workflow IDs",
    )
    parser.add_argument(
        "-r", "--run", nargs="+", action="extend", default=None,
        metavar="ENTITY=VALUE[,VALUE...]",
        help="Match BIDS runs; values are comma-delimited alternatives per entity",
    )
    parser.add_argument(
        "-s", "--space", nargs="+", action="extend", default=None,
        metavar="SPACE", help="Select one or more output spaces",
    )
    parser.add_argument(
        "-S", "--smoothing", nargs="+", action="extend", type=int, default=None,
        metavar="MM", help="Select one or more smoothing FWHM values in mm",
    )
    if planner_defaults:
        parser.set_defaults(
            _planner_defaults=True,
            _default_module=default_module,
            _default_workflow=default_workflow,
        )


def core_selection(args: argparse.Namespace) -> CoreSelection:
    """Normalize shared CLI values and parse run selectors."""
    planner_defaults = bool(getattr(args, "_planner_defaults", False))
    smoothing = tuple(
        dict.fromkeys(
            args.smoothing
            or ((DEFAULT_SMOOTHING_MM,) if planner_defaults else ())
        )
    )
    if any(value < 0 for value in smoothing):
        raise ValueError("--smoothing values must be nonnegative integers")
    return CoreSelection(
        participants=tuple(
            dict.fromkeys(value.removeprefix("sub-") for value in (args.participant or ()))
        ),
        projects=tuple(dict.fromkeys(args.project or ())),
        modules=tuple(
            dict.fromkeys(
                args.module
                or ((getattr(args, "_default_module"),) if planner_defaults else ())
            )
        ),
        workflows=tuple(
            dict.fromkeys(
                args.workflow
                or ((getattr(args, "_default_workflow"),) if planner_defaults else ())
            )
        ),
        runs=parse_selectors(args.run),
        spaces=tuple(
            dict.fromkeys(args.space or ((DEFAULT_SPACE,) if planner_defaults else ()))
        ),
        smoothing=smoothing,
    )


def _less_supports_header(less: str) -> bool:
    """Return whether this less executable advertises sticky headers."""
    try:
        result = subprocess.run(
            [less, "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return "--header" in ((result.stdout or "") + (result.stderr or ""))


def page_text(
    text: str,
    *,
    use_pager: bool = True,
    header_lines: int = 0,
) -> None:
    """Page text interactively while preserving pipe and automation output."""
    if not use_pager or not sys.stdout.isatty():
        sys.stdout.write(text)
        return
    less = shutil.which("less")
    if less is None:
        sys.stdout.write(text)
        return
    command = [less, "-R"]
    if header_lines > 0 and _less_supports_header(less):
        command.extend(("--header", str(header_lines)))
    try:
        subprocess.run(command, input=text, text=True, check=False)
    except KeyboardInterrupt:
        pass


def stderr(message: str) -> None:
    """Write a message to standard error immediately."""
    sys.stderr.write(message)
    sys.stderr.flush()
