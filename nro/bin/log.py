"""Browse matching nro worker or current derivative-instance logs with less."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from nro.engine.cli import matches_instance_selectors as matches_selectors
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.configuration.paths import BIDS_PATH
from nro.orchestration.registry import Registry
from nro.orchestration.catalog import MODULES
from nro.orchestration.selection import selected_projects


def _matching_instance_ids(
    registry: Registry,
    *,
    projects: set[str] | None = None,
    participants: Iterable[str],
    modules: set[str],
    workflows: set[str],
    selectors: dict[str, tuple[str, ...] | None],
) -> set[int]:
    participant_set = {value.removeprefix("sub-") for value in participants}
    project_set = projects or {registry.paths.project}
    result: set[int] = set()
    for row in registry.instance_rows(read_only=True):
        if str(row["project"]) not in project_set:
            continue
        if participant_set and str(row["participant"]) not in participant_set:
            continue
        if modules and str(row["module"]) not in modules:
            continue
        if workflows and not workflows.intersection(
            str(row.get("workflow_ids") or "").split(",")
        ):
            continue
        entities = json.loads(row["entities_json"])
        if selectors and not matches_selectors(entities, selectors):
            continue
        result.add(int(row["id"]))
    return result


def _existing(paths: Iterable[Path]) -> list[Path]:
    unique = {path.expanduser().resolve() for path in paths if path.is_file()}
    return sorted(
        unique,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )


def collect_log_paths(
    registry: Registry,
    *,
    instance_ids: set[int],
    instance_level: bool,
    instance_filtered: bool,
) -> list[Path]:
    """Resolve matching logs, newest first, without invoking a pager."""
    if instance_filtered and not instance_ids:
        return []
    if instance_level:
        # ``instance_rows`` exposes only the current attempt log, matching the
        # instance-level view's current-state semantics.
        rows = registry.instance_rows(read_only=True)
        return _existing(
            Path(str(row["log_path"]))
            for row in rows
            if row.get("log_path")
            and (not instance_filtered or int(row["id"]) in instance_ids)
        )

    if not instance_filtered:
        return _existing(registry.existing_control_path().joinpath("workers").glob("slurm-*.log"))

    placeholders = ",".join("?" for _ in instance_ids)
    job_ids: set[str] = set()
    with registry.read_connection() as db:
        job_ids.update(
            str(row["slurm_job_id"])
            for row in db.execute(
                f"""SELECT DISTINCT w.slurm_job_id
                    FROM attempts a JOIN workers w ON w.id=a.worker_id
                    WHERE a.instance_id IN ({placeholders}) AND w.slurm_job_id IS NOT NULL""",
                tuple(sorted(instance_ids)),
            )
        )
    return _existing(registry.paths.workers / f"slurm-{job_id}.log" for job_id in job_ids)


def build_parser(*, prog: str = "nro.bin.log") -> argparse.ArgumentParser:
    """Construct the log parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument(
        "-i", "--instance-level", action="store_true",
        help="Browse current derivative-instance logs instead of Slurm worker logs",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.log") -> None:
    """Open matching worker or instance logs in the configured pager.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bids_root = Path(args.bids_root).expanduser().resolve()
    selectors = selection.instance_entities
    modules = set(selection.modules)
    projects = selected_projects(bids_root, selection.projects)
    if not projects:
        raise SystemExit("No nro projects found in the central registry")
    registry = Registry.for_project(projects[0], bids_root=bids_root)
    if not registry.existing_database_path().is_file():
        raise SystemExit("No central nro registry found")
    instance_filtered = bool(
        selection.projects
        or selection.participants
        or selection.modules
        or selection.workflows
        or selectors
    )
    instance_ids = _matching_instance_ids(
        registry,
        projects=set(projects),
        participants=selection.participants,
        modules=modules,
        workflows=set(selection.workflows),
        selectors=selectors,
    )
    paths = collect_log_paths(
        registry,
        instance_ids=instance_ids,
        instance_level=args.instance_level,
        instance_filtered=instance_filtered,
    )
    paths = _existing(paths)
    if not paths:
        level = "instance" if args.instance_level else "worker"
        print(f"No matching {level}-level logs.")
        return
    less = shutil.which("less")
    if less is None:
        raise SystemExit("Cannot browse logs because 'less' is not available in PATH")
    try:
        subprocess.run([less, "-R", "--", *(str(path) for path in paths)], check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
