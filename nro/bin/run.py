"""Request workflow endpoints and supply a shared Slurm worker pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nro.configuration.store import ConfigStore
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.orchestration.catalog import MODULES, normalize_module, terminal_modules
from nro.orchestration.discovery import register_existing_artifacts
from nro.orchestration.manifests import assess_registry
from nro.orchestration.planner import Planner, PlanningResult
from nro.orchestration.registry import Registry
from nro.orchestration.selection import discover_bids_inventory
from nro.orchestration.submission import _submit_workers, _write_worker_script
from nro.orchestration.worker_control import stop_worker_pool_for_repair

DEFAULT_CONCURRENCY = 50
DEFAULT_WORKER_IDLE_TIMEOUT = 30


def _confirm_repair_with_workers(activity: dict) -> bool:
    workers = len(activity["workers"])
    submissions = len(activity["submissions"])
    print(
        f"Registry repair must stop the shared worker pool ({workers} active "
        f"worker(s), {submissions} active/pending submission(s)).",
        file=sys.stderr,
    )
    try:
        response = input("Stop all workers and continue with repair? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() in {"y", "yes"}


def _unavailable_records(plan: PlanningResult) -> list[dict[str, str]]:
    return [
        {
            "project": item.project,
            "participant": item.participant,
            "module": item.module,
            "reason": item.reason,
        }
        for item in dict.fromkeys(plan.unavailable)
    ]


def _report_unavailable(plan: PlanningResult) -> None:
    records = _unavailable_records(plan)
    if not records:
        return
    print(
        f"Skipped {len(records)} unavailable participant/module selection(s):",
        file=sys.stderr,
    )
    for item in records:
        print(
            f"  {item['project']}/sub-{item['participant']} {item['module']}: {item['reason']}",
            file=sys.stderr,
        )


def build_parser(*, prog: str = "nro.bin.run") -> argparse.ArgumentParser:
    """Construct the run parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(
        parser,
        module_choices=MODULES,
        planner_defaults=True,
        default_modules=terminal_modules(),
    )
    from nro.configuration.site import settings

    site, _ = settings()
    parser.add_argument("--partition", default=site["partition"])
    parser.add_argument("--account", default=site["account"] or None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum shared worker concurrency (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--time", type=int, default=24, metavar="HOURS")
    parser.add_argument("--memory", type=int, default=32, metavar="GB")
    parser.add_argument("--max-memory", type=int, default=256, metavar="GB")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument(
        "--worker-idle-timeout",
        type=int,
        default=DEFAULT_WORKER_IDLE_TIMEOUT,
        metavar="SECONDS",
        help=f"Exit a worker after this many idle seconds (default: {DEFAULT_WORKER_IDLE_TIMEOUT})",
    )
    parser.add_argument(
        "--drain-minutes",
        type=int,
        default=15,
        help="Stop workers from claiming new work this long before wall time",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run one worker locally instead of submitting Slurm jobs",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Register and assess the request without starting workers",
    )
    parser.add_argument(
        "--no-inherit",
        action="store_true",
        help="Compute selected work in this development branch without reusing ancestor derivatives",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Repair scientific records without new demand; after central activation "
            "this is branch-scoped, otherwise it rebuilds the standalone registry. "
            "Asks before stopping active work"
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.run") -> None:
    """Plan selected work, persist demand, and optionally supply workers.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    from nro.configuration import site

    record = site.installation_record()
    if record.get("mode") in {"shared", "branch"} and not record.get("ready") and not args.repair:
        raise SystemExit("The installation is undergoing setup or maintenance")
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bids_root = site.bids_root()
    if args.repair:
        explicit_selections = {
            "--participant": args.participant,
            "--project": args.project,
            "--module": args.module,
            "--workflow": args.workflow,
            "--run": args.run,
            "--space": args.space,
            "--smoothing": args.smoothing,
            "--task": args.task,
            "--model": args.model,
            "--model-set": args.model_set,
        }
        supplied = [name for name, value in explicit_selections.items() if value is not None]
        if supplied:
            raise SystemExit(
                "--repair covers all projects and cannot be combined with selection options: "
                + ", ".join(supplied)
            )
        from nro.orchestration.scheduler_implementation import implementation_path

        values = site.settings()[0]
        if (
            record.get("mode") == "branch"
            or implementation_path(Path(values["registry"])).is_file()
        ):
            from nro.orchestration.branch_repair import repair_checkout

            def confirm(activity):
                print(
                    f"Repair will stop {activity['attempts']} active attempt(s) for branch {activity['branch']}. "
                    "Other branches and the shared worker pool remain available.",
                    file=sys.stderr,
                )
                try:
                    return input(
                        "Stop these attempts and repair this branch? [y/N] "
                    ).strip().lower() in {"y", "yes"}
                except (EOFError, KeyboardInterrupt):
                    return False

            try:
                result = repair_checkout(
                    Path(values["registry"]), bids_root, site.CHECKOUT, confirm=confirm
                )
            except (ValueError, RuntimeError, OSError) as error:
                raise SystemExit(str(error)) from error
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(
                    f"Repaired {result['registry']}; restored {result['instances']} scientific record(s), without demand."
                )
            return
        registry = Registry.for_project("", bids_root=bids_root)
        activity = registry.worker_pool_activity(for_repair=True)
        if (activity["workers"] or activity["submissions"]) and not _confirm_repair_with_workers(
            activity
        ):
            raise SystemExit("Registry repair cancelled; no state was changed")
        try:
            shutdown = stop_worker_pool_for_repair(registry)
        except RuntimeError as error:
            raise SystemExit(
                f"Registry was not repaired because the worker pool could not be "
                f"stopped safely: {error}. Worker shutdown remains requested; "
                "resolve any reported allocation and run --repair again."
            ) from error
        if shutdown["cancellation_failures"]:
            print(
                "WARNING: direct Slurm cancellation failed for "
                + "; ".join(shutdown["cancellation_failures"])
                + "; all affected workers nevertheless stopped cooperatively.",
                file=sys.stderr,
            )
        registry.reinitialize()
        inventory = discover_bids_inventory(bids_root)
        registry.replace_bids_inventory(inventory)
        discovery = register_existing_artifacts(
            registry,
            bids_root=bids_root,
            inventory=inventory,
            memory_gb=args.memory,
            max_memory_gb=args.max_memory,
        )
        result = {
            "repaired": True,
            "registry": str(registry.paths.database),
            "projects": list(inventory),
            "participants": sum(len(values) for values in inventory.values()),
            "artifacts": discovery.artifacts,
            "instances": discovery.instances,
            "unavailable_artifacts": list(discovery.unavailable),
            "requests": [],
            "submitted_workers": [],
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"Repaired {registry.paths.database}; discovered {len(inventory)} "
                f"project(s), {result['participants']} participant(s), and "
                f"{discovery.artifacts} existing artifact(s); registered "
                f"{discovery.instances} instance(s) including dependencies."
            )
            if discovery.unavailable:
                print(
                    f"Skipped {len(discovery.unavailable)} unavailable "
                    "participant/workflow combination(s).",
                    file=sys.stderr,
                )
        return

    modules = tuple(normalize_module(value) for value in selection.modules)
    if (selection.models or selection.model_sets) and "firstlevels" not in modules:
        raise SystemExit("--model and --model-set require -m firstlevels")
    if set(modules).issubset({"anat", "func"}) and (
        args.space is not None or args.smoothing is not None
    ):
        raise SystemExit(
            "--space and --smoothing apply only when the terminal module is "
            "clean, dynconn, microparcellation, networks, or firstlevels"
        )
    if set(modules) == {"anat"} and selection.runs:
        raise SystemExit("--run does not apply to the anatomical module")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.memory < 1 or args.max_memory < args.memory:
        raise SystemExit("--memory must be positive and --max-memory must be at least --memory")
    if args.drain_minutes < 0 or args.drain_minutes * 60 >= args.time * 60 * 60:
        raise SystemExit("--drain-minutes must be nonnegative and shorter than --time")
    if args.worker_idle_timeout < 1:
        raise SystemExit("--worker-idle-timeout must be at least 1 second")
    inventory = discover_bids_inventory(bids_root)
    actual_projects = list(inventory)
    if selection.projects:
        absent_projects = sorted(set(selection.projects) - set(actual_projects))
        if absent_projects:
            raise SystemExit(
                "No BIDS participants were found in project(s): " + ", ".join(absent_projects)
            )
        projects = list(selection.projects)
    else:
        projects = actual_projects
    if not projects:
        raise SystemExit(f"No BIDS projects were found under {bids_root}")

    from nro.orchestration.scheduler_implementation import implementation_path

    values = site.settings()[0]
    branch_execution = (
        record.get("mode") == "branch" or implementation_path(Path(values["registry"])).is_file()
    )
    if branch_execution:
        from nro.orchestration.branch_store import BranchStore

        registry = BranchStore(Path(values["registry"])).registry_for_checkout(site.CHECKOUT)
        new_registry = False
    else:
        registry = Registry.for_project(projects[0], bids_root=bids_root)
        new_registry = not registry.existing_database_path().is_file()
        registry.replace_bids_inventory(inventory)
    if new_registry:
        register_existing_artifacts(
            registry,
            bids_root=bids_root,
            inventory=inventory,
            memory_gb=args.memory,
            max_memory_gb=args.max_memory,
        )
    store = ConfigStore()
    workflows = {workflow_id: store.resolve(workflow_id) for workflow_id in selection.workflows}
    registered_workflows = {
        workflow_id: registry.register_workflow(workflow)
        for workflow_id, workflow in workflows.items()
    }

    planner = Planner(registry, bids_root=bids_root)
    scientific_revisions = (
        {row.key: row.revision for row in registry.instances()} if branch_execution else {}
    )
    plan = planner.plan(
        projects=projects,
        requested_participants=selection.participants,
        modules=modules,
        workflows=workflows,
        registered_workflows=registered_workflows,
        selectors=selection.runs,
        spaces=selection.spaces,
        smoothing_levels=selection.smoothing,
        memory_gb=args.memory,
        max_memory_gb=args.max_memory,
        models=selection.models,
        model_sets=selection.model_sets,
    )
    if selection.participants:
        absent = sorted(set(selection.participants) - plan.present_participants)
        if absent:
            raise SystemExit("No matching BIDS participant(s) were found: " + ", ".join(absent))
    if not plan.requests:
        unavailable = _unavailable_records(plan)
        if unavailable:
            details = "; ".join(
                f"{item['project']}/sub-{item['participant']} {item['module']}: {item['reason']}"
                for item in unavailable
            )
            raise SystemExit(f"No requested work is available: {details}")
        raise SystemExit("The requested project, participant, and run filters matched no work")

    participant_count = plan.participant_count
    planned_projects = plan.projects
    all_instances = plan.instances
    matched_participants = {
        project: list(participants) for project, participants in plan.matched_participants.items()
    }
    if branch_execution:
        if args.local and args.no_submit:
            raise SystemExit("--local and --no-submit are mutually exclusive")
        from nro.orchestration.branch_requests import register_requests
        from nro.orchestration.scheduler_client import supply

        request_ids = register_requests(
            registry,
            plan,
            selectors={
                "runs": selection.runs,
                "spaces": list(selection.spaces),
                "smoothing": list(selection.smoothing),
                "models": list(selection.models),
                "model_sets": list(selection.model_sets or (() if selection.models else ("main",))),
            },
            concurrency=args.concurrency,
            partition=args.partition,
            expected_revisions=scientific_revisions,
            inherit=not args.no_inherit,
        )
        result = supply(
            Path(values["registry"]),
            bids_root,
            checkout=site.CHECKOUT,
            request_ids=request_ids,
            options={
                key: getattr(args, key)
                for key in (
                    "local",
                    "no_submit",
                    "memory",
                    "max_memory",
                    "partition",
                    "account",
                    "time",
                    "cpus",
                    "worker_idle_timeout",
                    "drain_minutes",
                )
            },
        )
        result.update(
            requests=request_ids,
            projects=list(planned_projects),
            participants=matched_participants,
            instances=len(all_instances),
            unavailable=_unavailable_records(plan),
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"Created {len(request_ids)} branch request(s); submitted {len(result['submitted_workers'])} worker(s)."
            )
            _report_unavailable(plan)
        return
    if args.no_inherit:
        raise SystemExit("--no-inherit applies only when branch execution is active")
    request_ids = list(
        planner.register_requests(
            plan,
            selectors={
                "runs": selection.runs,
                "spaces": list(selection.spaces),
                "smoothing": list(selection.smoothing),
                "models": list(selection.models),
                "model_sets": list(selection.model_sets or (() if selection.models else ("main",))),
            },
            concurrency=args.concurrency,
            partition=args.partition,
        )
    )
    registered_instances = registry.instance_ids(tuple(all_instances))
    # Reassess the complete active graph before allocating more workers.  A
    # request can make an old upstream derivative stale while a downstream
    # attempt from an earlier request is still executing.  Such an attempt
    # must be stopped before any new allocation proceeds.
    all_states = assess_registry(registry, projects=planned_projects)
    states = {instance_id: all_states[instance_id] for instance_id in registered_instances.values()}
    prematurely_running = registry.cancel_attempts_with_stale_upstreams()
    registry.reconcile_requests()
    fresh = sum(state == "fresh" for state, _reason in states.values())
    submitted: list[str] = []
    if args.local and args.no_submit:
        raise SystemExit("--local and --no-submit are mutually exclusive")
    if args.local:
        from nro.orchestration.scheduler_implementation import run_local_worker

        run_local_worker(registry, memory_gb=args.memory, drain_seconds=args.drain_minutes * 60)
    elif not args.no_submit:
        tier = args.memory
        scripts: dict[int, Path] = {}
        while True:
            scripts[tier] = _write_worker_script(
                registry,
                bids_root=bids_root,
                partition=args.partition,
                account=args.account,
                hours=args.time,
                memory_gb=tier,
                cpus=args.cpus,
                idle_timeout=args.worker_idle_timeout,
                drain_seconds=args.drain_minutes * 60,
            )
            if tier >= args.max_memory:
                break
            tier = min(args.max_memory, tier * 2)
        submitted = _submit_workers(registry, request_ids[0], scripts[args.memory], args.memory)
    result = {
        "requests": request_ids,
        "projects": list(planned_projects),
        "workflows": list(selection.workflows),
        "modules": list(modules),
        "participants": matched_participants,
        "spaces": list(selection.spaces),
        "smoothing": list(selection.smoothing),
        "concurrency": args.concurrency,
        "instances": len(all_instances),
        "fresh": fresh,
        "cancel_requested": len(prematurely_running),
        "submitted_workers": submitted,
        "unavailable": _unavailable_records(plan),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"Created {len(request_ids)} request(s) for {participant_count} "
            f"project/participant match(es): {len(all_instances)} instance(s), "
            f"{fresh} already fresh."
        )
        if submitted:
            print(f"Submitted {len(submitted)} worker(s): {', '.join(submitted)}")
        elif not args.local and not args.no_submit:
            print(
                "No new workers submitted; the shared pool already meets the active concurrency limit."
            )
        if prematurely_running:
            print(
                f"Requested cancellation of {len(prematurely_running)} prematurely downstream "
                "instance attempt(s); their workers will stop only those instance processes."
            )
        _report_unavailable(plan)

    from nro.orchestration.execution_cache import cleanup_cache

    cleanup_cache(registry)


if __name__ == "__main__":
    main()
