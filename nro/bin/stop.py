"""Cancel selected nro derivative demand and its downstream dependents."""

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
        "--workers",
        action="store_true",
        help="Shut down the current user's lab-wide worker pool without cancelling instance demand",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=(
            "Cancel matching demand from every user and stop the shared attempt; "
            "does not delete completed derivatives or registry history"
        ),
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help="Cancel only the selected derivative rather than its downstream dependents",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.stop") -> None:
    """Cancel selected demand or shut down workers without deleting artifacts.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    from nro.configuration import site

    bids_root = site.bids_root()
    modules = list(selection.modules)
    selectors = selection.instance_entities
    total = {"instances": 0, "requests": 0, "attempts": 0}
    from nro.orchestration.scheduler_implementation import implementation_path

    values = site.settings()[0]
    branch_execution = (
        site.installation_record().get("mode") == "branch"
        or implementation_path(Path(values["registry"])).is_file()
    )
    if branch_execution and not args.workers:
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
                    selectors=selectors,
                    include_dependents=not args.only,
                    force=args.force,
                ),
            )
            for key, value in result.items():
                total[key] += value
        print(
            f"Cancelled {total['instances']} instance demand(s) across {total['requests']} request(s); "
            f"signalled {total['attempts']} running attempt(s)."
        )
        return
    if args.workers:
        if (
            selection.projects
            or selection.participants
            or selection.modules
            or selection.workflows
            or selection.runs
            or selection.spaces
            or selection.smoothing
            or selection.models
            or selection.model_sets
            or args.only
            or args.force
        ):
            raise SystemExit(
                "--workers controls the lab-wide pool and cannot be combined with selectors"
            )
        if branch_execution:
            from nro.orchestration.scheduler_client import pool_operation

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
            f"{shutdown['attempts']} active instance attempt(s); cancelled "
            f"{shutdown['submission_count']} pending/active "
            f"worker submission(s), including {stopped_jobs} Slurm job(s)."
        )
        for failure in failures:
            print(f"WARNING: could not cancel Slurm job {failure}")
        return
    projects = selected_projects(bids_root, selection.projects)
    if not projects:
        raise SystemExit("No nro projects found in the central registry")
    central_registry = Registry.for_project(projects[0], bids_root=bids_root)
    for project in projects:
        registry = Registry.for_project(project, bids_root=bids_root)
        result = registry.request_cancellation(
            participants=selection.participants,
            modules=modules,
            workflows=selection.workflows,
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
        f"{'Force-cancelled' if args.force else 'Cancelled'} {total['instances']} instance demand(s) "
        f"across {total['requests']} request(s); "
        f"signalled {total['attempts']} running attempt(s)."
    )


if __name__ == "__main__":
    main()
