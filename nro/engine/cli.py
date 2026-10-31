"""Command-line selection vocabulary and output primitives."""

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import yaml

from nro.engine.bids import matches_selectors, parse_selectors
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE


@lru_cache(maxsize=256)
def _model_source_tasks(identifier: str, path: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    """Read source tasks once for one observed model-definition revision."""
    del path, mtime_ns, size
    from nro.modules.contrasts.task_models import load_task_model, source_tasks

    return source_tasks(load_task_model(identifier), identifier)


@dataclass(frozen=True)
class CoreSelection:
    """Normalized data, workflow, spatial-target, and first-level selectors."""

    participants: tuple[str, ...]
    projects: tuple[str, ...]
    modules: tuple[str, ...]
    workflows: tuple[str, ...]
    runs: dict[str, tuple[str, ...] | None]
    spaces: tuple[str, ...]
    smoothing: tuple[int, ...]
    analyses: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    model_sets: tuple[str, ...] | None = None
    lineages: tuple[str, ...] = ()

    @property
    def work_item_entities(self) -> dict[str, tuple[str, ...] | None]:
        """Return run and target entity filters used to match work items."""
        entities = dict(self.runs)
        if "task" in entities:
            entities["source_task"] = entities.pop("task")
        if self.spaces:
            entities["space"] = self.spaces
        if self.smoothing:
            entities["smoothing"] = tuple(str(value) for value in self.smoothing)
        if self.models:
            entities["model"] = self.models
        if self.analyses:
            entities["analysis"] = self.analyses
        if self.model_sets:
            from nro.modules.contrasts.task_models import model_ids_in_sets

            entities["model_id"] = model_ids_in_sets(self.model_sets)
        return entities


def matches_work_item_selectors(entities: dict, selectors: dict) -> bool:
    """Match work-item entities, including qualified models resolved from sets."""
    selectors = dict(selectors)
    identifier = f"{entities.get('task', '')}/{entities.get('model', '')}"
    analyses = selectors.pop("analysis", ())
    if analyses and entities.get("analysis", entities.get("task")) not in analyses:
        return False
    source_tasks = selectors.pop("source_task", ())
    if source_tasks:
        available = (entities.get("task"),)
        if entities.get("model") and entities.get("task"):
            identifier = f"{entities['task']}/{entities['model']}"
            try:
                from nro.modules.contrasts.task_models import model_path

                path = model_path(identifier)
                stat = Path(path).stat()
                available = _model_source_tasks(
                    identifier, str(path), stat.st_mtime_ns, stat.st_size
                )
            except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
                # Removed definitions remain selectable by analysis or
                # qualified model.  The historical task entity is the best
                # available fallback for source-task selection.
                pass
        if not set(source_tasks).intersection(filter(None, available)):
            return False
    models = selectors.pop("model", ())
    if models and entities.get("model") not in models and identifier not in models:
        return False
    return matches_selectors({**entities, "model_id": identifier}, selectors)


def matches_module_lineage(module: str, lineage_id: str, selected: Sequence[str]) -> bool:
    """Match a module-lineage label in bare or module-qualified form."""
    return not selected or lineage_id in selected or f"{module}/{lineage_id}" in selected


def add_core_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    module_choices: Sequence[str],
    planner_defaults: bool = False,
    default_modules: Sequence[str] = (),
    default_workflow: str = "main",
    module_help: str | None = None,
) -> None:
    """Add the selection options shared by user-facing orchestration tools."""
    parser.add_argument(
        "-p",
        "--participant",
        nargs="+",
        action="extend",
        default=None,
        metavar="ID",
        help="Select one or more BIDS participant IDs",
    )
    parser.add_argument(
        "-P",
        "--project",
        nargs="+",
        action="extend",
        default=None,
        metavar="PROJECT",
        help="Select one or more projects",
    )
    parser.add_argument(
        "-m",
        "--module",
        nargs="+",
        action="extend",
        choices=module_choices,
        default=None,
        metavar="MODULE",
        help=module_help
        or (
            "Select one or more modules"
            + ("; defaults to all workflow endpoints" if planner_defaults else "")
        ),
    )
    parser.add_argument(
        "-w",
        "--workflow",
        nargs="+",
        action="extend",
        default=None,
        metavar="WORKFLOW",
        help="Select one or more workflow IDs",
    )
    parser.add_argument(
        "-i",
        "--lineage",
        nargs="+",
        action="extend",
        default=None,
        metavar="ID",
        help="Select module-lineage IDs, optionally as MODULE/ID",
    )
    parser.add_argument(
        "-r",
        "--run",
        nargs="+",
        action="extend",
        default=None,
        metavar="ENTITY=VALUE[,VALUE...]",
        help="Match BIDS runs; values are comma-delimited alternatives per entity",
    )
    parser.add_argument(
        "-s",
        "--space",
        nargs="+",
        action="extend",
        default=None,
        metavar="SPACE",
        help="Select one or more output spaces",
    )
    parser.add_argument(
        "-S",
        "--smoothing",
        nargs="+",
        action="extend",
        type=int,
        default=None,
        metavar="MM",
        help="Select one or more smoothing FWHM values in mm",
    )
    parser.add_argument("--task", nargs="+", action="extend", help="Select source BIDS tasks")
    parser.add_argument(
        "--analysis",
        nargs="+",
        action="extend",
        help="Select contrasts analysis labels",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        action="extend",
        help="Select contrasts variants or ANALYSIS/VARIANT IDs",
    )
    parser.add_argument(
        "--model-set",
        nargs="+",
        action="extend",
        help="Select contrasts model sets; requests default to main unless --model is given",
    )
    if planner_defaults:
        parser.set_defaults(
            _planner_defaults=True,
            _default_modules=tuple(default_modules),
            _default_workflow=default_workflow,
        )


def core_selection(
    args: argparse.Namespace, *, apply_planner_defaults: bool | None = None
) -> CoreSelection:
    """Normalize shared CLI values and parse run selectors.

    ``apply_planner_defaults`` can suppress endpoint, workflow, space, and
    smoothing defaults for modes that select existing registry records.
    """
    planner_defaults = bool(getattr(args, "_planner_defaults", False))
    if apply_planner_defaults is not None:
        planner_defaults = apply_planner_defaults
    smoothing = tuple(
        dict.fromkeys(args.smoothing or ((DEFAULT_SMOOTHING_MM,) if planner_defaults else ()))
    )
    if any(value < 0 for value in smoothing):
        raise ValueError("--smoothing values must be nonnegative integers")
    runs = parse_selectors(args.run)
    tasks = tuple(dict.fromkeys(getattr(args, "task", None) or ()))
    if tasks:
        if "task" in runs:
            tasks = tuple(value for value in tasks if value in (runs["task"] or ()))
            if not tasks:
                raise ValueError("--task and --run task= select disjoint tasks")
        runs["task"] = tasks
    return CoreSelection(
        participants=tuple(
            dict.fromkeys(value.removeprefix("sub-") for value in (args.participant or ()))
        ),
        projects=tuple(dict.fromkeys(args.project or ())),
        modules=tuple(
            dict.fromkeys(args.module or (args._default_modules if planner_defaults else ()))
        ),
        workflows=tuple(
            dict.fromkeys(
                args.workflow or ((getattr(args, "_default_workflow"),) if planner_defaults else ())
            )
        ),
        runs=runs,
        spaces=tuple(dict.fromkeys(args.space or ((DEFAULT_SPACE,) if planner_defaults else ()))),
        smoothing=smoothing,
        analyses=tuple(dict.fromkeys(getattr(args, "analysis", None) or ())),
        models=tuple(dict.fromkeys(getattr(args, "model", None) or ())),
        model_sets=tuple(dict.fromkeys(args.model_set))
        if getattr(args, "model_set", None)
        else None,
        lineages=tuple(dict.fromkeys(args.lineage or ())),
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
    command = [less, "-R", "-S"]
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
