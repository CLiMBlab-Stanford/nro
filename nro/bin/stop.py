"""Stop selected nro work and its downstream dependents without removing demand."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.orchestration.catalog import MODULES
from nro.orchestration.registry import Registry
from nro.orchestration.selection import selected_projects
from nro.orchestration.worker_control import cancel_worker_allocations


def build_parser(*, prog: str = "nro.bin.stop") -> argparse.ArgumentParser:
    """Construct the stop parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument(
        "-W",
        "--worker",
        "--workers",
        action="store_true",
        help="Stop all workers and their active work; cannot be combined with selectors",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Stop all workers and their active work, then stop the scheduler",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=(
            "Stop matching demand from every user and stop the shared attempt; "
            "does not delete completed derivatives or registry history"
        ),
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help="Stop only the selected derivative rather than its downstream dependents",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.stop") -> None:
    """Stop selected work or shut down workers without deleting artifacts or demand.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    from nro.site import configuration as site

    bids_root = site.bids_root()
    modules = list(selection.modules)
    selectors = selection.work_item_entities
    total = {"work_items": 0, "requests": 0, "attempts": 0}
    from nro.orchestration.scheduler_implementation import implementation_path

    values = site.settings()[0]
    branch_execution = (
        site.installation_record().get("mode") == "branch"
        or implementation_path(Path(values["registry"])).is_file()
    )
    pool_control = args.worker or args.scheduler
    if pool_control and (
        selection.projects
        or selection.participants
        or selection.modules
        or selection.workflows
        or selection.lineages
        or selection.runs
        or selection.spaces
        or selection.smoothing
        or selection.models
        or selection.model_sets
        or args.only
        or args.force
    ):
        raise SystemExit(
            "--worker and --scheduler control the global pool and cannot use selectors"
        )
    if branch_execution and not pool_control:
        from nro.orchestration.scheduler_client import stop

        for project in selected_projects(bids_root, selection.projects):
            result = stop(
                Path(values["registry"]),
                bids_root,
                checkout=site.CHECKOUT,
                project=project,
                selection=dict(
                    participants=selection.participants,
                    modules=modules,
                    workflows=selection.workflows,
                    lineages=selection.lineages,
                    selectors=selectors,
                    include_dependents=not args.only,
                    force=args.force,
                ),
            )
            for key, value in result.items():
                total[key] += value
        print(
            f"Stopped {total['work_items']} work-item demand(s) across {total['requests']} request(s); "
            f"signalled {total['attempts']} running attempt(s)."
        )
        return
    if pool_control:
        if branch_execution:
            from nro.orchestration.scheduler_client import pool_operation, shutdown_service

            shutdown = pool_operation(
                Path(values["registry"]),
                bids_root,
                checkout=site.CHECKOUT,
                operation="stop_workers",
            )
            stopped_jobs, failures = shutdown["stopped_jobs"], shutdown["failures"]
        else:
            registry = Registry.for_project("", bids_root=bids_root)
            if not registry.existing_database_path().is_file():
                raise SystemExit("No central nro registry found")
            shutdown = registry.request_worker_shutdown()
            stopped_jobs, failures = cancel_worker_allocations(registry, shutdown)
        print(
            f"Requested shutdown of {shutdown['workers']} worker(s); interrupted "
            f"{shutdown['attempts']} active work-item attempt(s); cancelled "
            f"{shutdown['submission_count']} pending/active "
            f"worker submission(s), including {stopped_jobs} Slurm job(s)."
        )
        for failure in failures:
            print(f"WARNING: could not cancel Slurm job {failure}")
        if args.scheduler:
            if branch_execution:
                scheduler = shutdown_service(
                    Path(values["registry"]), bids_root, checkout=site.CHECKOUT
                )
                if scheduler["stopping"]:
                    print("Requested scheduler shutdown.")
                else:
                    print("No live scheduler was running.")
            else:
                print("No scheduler service is active in this installation mode.")
        return
    projects = selected_projects(bids_root, selection.projects)
    if not projects:
        raise SystemExit("No nro projects found in the central registry")
    central_registry = Registry.for_project(projects[0], bids_root=bids_root)
    for project in projects:
        registry = Registry.for_project(project, bids_root=bids_root)
        result = registry.request_stop(
            participants=selection.participants,
            modules=modules,
            workflows=selection.workflows,
            lineages=selection.lineages,
            selectors=selectors,
            include_dependents=not args.only,
            force=args.force,
        )
        for key, value in result.items():
            total[key] += value
    for submission_id, job_id in central_registry.unused_queued_worker_jobs():
        result = subprocess.run(["scancel", job_id], text=True, capture_output=True)
        central_registry.update_submission(
            submission_id, state="cancelled" if result.returncode == 0 else "error"
        )
    print(
        f"Stopped {total['work_items']} work-item demand(s) "
        f"across {total['requests']} request(s); "
        f"signalled {total['attempts']} running attempt(s)."
    )


if __name__ == "__main__":
    main()
