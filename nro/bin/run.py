"""Plan a terminal module request and supply a shared Slurm worker pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.catalog import MODULES, normalize_module
from nro.orchestration.manifests import assess_registry
from nro.orchestration.planner import Planner, PlanningResult
from nro.orchestration.registry import Registry
from nro.orchestration.worker import Worker
from nro.orchestration.worker_control import stop_worker_pool_for_repair
from nro.configuration.store import ConfigStore
from nro.engine.io import atomic_write_text
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.orchestration.selection import discover_bids_inventory


DEFAULT_MODULE = "networks"
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
            f"  {item['project']}/sub-{item['participant']} {item['module']}: "
            f"{item['reason']}",
            file=sys.stderr,
        )


def _worker_source_root() -> Path:
    """Return the repository root that workers import."""
    return Path(__file__).resolve().parents[2]


def _write_worker_script(
    registry: Registry,
    *,
    bids_root: Path,
    partition: str,
    account: str | None,
    hours: int,
    memory_gb: int,
    cpus: int,
    idle_timeout: int = DEFAULT_WORKER_IDLE_TIMEOUT,
    drain_seconds: int = 15 * 60,
) -> Path:
    profile_payload = json.dumps(
        {
            "bids_root": str(bids_root),
            "partition": partition,
            "account": account,
            "hours": hours,
            "cpus": cpus,
            "idle_timeout": idle_timeout,
            "drain_seconds": drain_seconds,
        },
        sort_keys=True,
    )
    profile = hashlib.sha256(profile_payload.encode("utf-8")).hexdigest()[:12]
    path = registry.paths.workers / f"worker-large-{memory_gb}gb-{profile}.sbatch"
    command = [
        sys.executable, "-m", "nro.orchestration.worker",
        "--bids-root", str(bids_root), "--resource-class", "large",
        "--memory-gb", str(memory_gb),
        "--idle-timeout", str(idle_timeout),
        "--walltime-seconds", str(hours * 60 * 60),
        "--drain-seconds", str(drain_seconds),
        "--profile", profile,
    ]
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=nro-worker",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={hours}:00:00",
        f"#SBATCH --mem={memory_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={registry.paths.workers}/slurm-%j.log",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines.extend((
        "set -euo pipefail",
        f"cd {shlex.quote(str(_worker_source_root()))}",
        "exec " + shlex.join(command),
    ))
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _submit_workers(
    registry: Registry,
    request_id: str,
    script: Path,
    memory_gb: int,
) -> list[str]:
    submitted: list[str] = []
    registry.reconcile_scheduler_submissions()
    for submission_id, _token in registry.reserve_worker_submissions(
        request_id=request_id, resource_class="large", memory_gb=memory_gb
    ):
        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(script)],
                check=True,
                text=True,
                capture_output=True,
            )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id:
                raise RuntimeError(f"sbatch returned no job ID: {result.stdout!r}")
            registry.update_submission(submission_id, state="submitted", slurm_job_id=job_id)
            submitted.append(job_id)
        except BaseException:
            registry.update_submission(submission_id, state="error")
            raise
    return submitted


def build_parser(*, prog: str = "nro.bin.run") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(
        parser,
        module_choices=MODULES,
        planner_defaults=True,
        default_module=DEFAULT_MODULE,
    )
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument("--partition", default="sphinx")
    parser.add_argument("--account", default="nlp")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Maximum shared worker concurrency (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument("--time", type=int, default=24, metavar="HOURS")
    parser.add_argument("--memory", type=int, default=32, metavar="GB")
    parser.add_argument("--max-memory", type=int, default=256, metavar="GB")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument(
        "--worker-idle-timeout", type=int, default=DEFAULT_WORKER_IDLE_TIMEOUT,
        metavar="SECONDS",
        help=f"Exit a worker after this many idle seconds (default: {DEFAULT_WORKER_IDLE_TIMEOUT})",
    )
    parser.add_argument(
        "--drain-minutes", type=int, default=15,
        help="Stop workers from claiming new work this long before wall time",
    )
    parser.add_argument("--local", action="store_true", help="Run one worker locally instead of submitting Slurm jobs")
    parser.add_argument("--no-submit", action="store_true", help="Register and assess the request without starting workers")
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Stop the shared worker pool, destroy and rebuild the lab-wide private "
            "registry, discover the source BIDS tree, and leave derivative work "
            "unregistered; asks for confirmation when workers exist"
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.run") -> None:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bids_root = Path(args.bids_root).expanduser().resolve()
    if args.repair:
        explicit_selections = {
            "--participant": args.participant,
            "--project": args.project,
            "--module": args.module,
            "--workflow": args.workflow,
            "--run": args.run,
            "--space": args.space,
            "--smoothing": args.smoothing,
        }
        supplied = [name for name, value in explicit_selections.items() if value is not None]
        if supplied:
            raise SystemExit(
                "--repair is lab-wide and cannot be combined with selection options: "
                + ", ".join(supplied)
            )
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
        result = {
            "repaired": True,
            "registry": str(registry.paths.database),
            "projects": list(inventory),
            "participants": sum(len(values) for values in inventory.values()),
            "instances": 0,
            "requests": [],
            "submitted_workers": [],
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"Repaired {registry.paths.database}; discovered {len(inventory)} "
                f"project(s) and {result['participants']} participant(s). The "
                "derivative registry is empty."
            )
        return

    modules = tuple(normalize_module(value) for value in selection.modules)
    if set(modules).issubset({"anat", "func"}) and (
        args.space is not None or args.smoothing is not None
    ):
        raise SystemExit(
            "--space and --smoothing apply only when the terminal module is "
            "clean, microparcellation, or networks"
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
                "No BIDS participants were found in project(s): "
                + ", ".join(absent_projects)
            )
        projects = list(selection.projects)
    else:
        projects = actual_projects
    if not projects:
        raise SystemExit(f"No BIDS projects were found under {bids_root}")

    registry = Registry.for_project(projects[0], bids_root=bids_root)
    registry.replace_bids_inventory(inventory)
    store = ConfigStore()
    workflows = {
        workflow_id: store.resolve(workflow_id)
        for workflow_id in selection.workflows
    }
    registered_workflows = {
        workflow_id: registry.register_workflow(workflow)
        for workflow_id, workflow in workflows.items()
    }

    planner = Planner(registry, bids_root=bids_root)
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
    )
    if selection.participants:
        absent = sorted(set(selection.participants) - plan.present_participants)
        if absent:
            raise SystemExit(
                "No matching BIDS participant(s) were found: " + ", ".join(absent)
            )
    if not plan.requests:
        unavailable = _unavailable_records(plan)
        if unavailable:
            details = "; ".join(
                f"{item['project']}/sub-{item['participant']} {item['module']}: "
                f"{item['reason']}"
                for item in unavailable
            )
            raise SystemExit(f"No requested work is available: {details}")
        raise SystemExit(
            "The requested project, participant, and run filters matched no work"
        )

    participant_count = plan.participant_count
    planned_projects = plan.projects
    all_instances = plan.instances
    matched_participants = {
        project: list(participants)
        for project, participants in plan.matched_participants.items()
    }
    request_ids = list(
        planner.register_requests(
            plan,
            selectors={
                "runs": selection.runs,
                "spaces": list(selection.spaces),
                "smoothing": list(selection.smoothing),
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
    states = {
        instance_id: all_states[instance_id]
        for instance_id in registered_instances.values()
    }
    prematurely_running = registry.cancel_attempts_with_stale_upstreams()
    registry.reconcile_requests()
    fresh = sum(state == "fresh" for state, _reason in states.values())
    submitted: list[str] = []
    if args.local and args.no_submit:
        raise SystemExit("--local and --no-submit are mutually exclusive")
    if args.local:
        Worker(
            registry,
            resource_class="large",
            memory_gb=args.memory,
            idle_timeout=1.0,
            poll_interval=0.2,
            drain_seconds=args.drain_minutes * 60,
        ).run()
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
        submitted = _submit_workers(
            registry, request_ids[0], scripts[args.memory], args.memory
        )
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
            print("No new workers submitted; the shared pool already meets the active concurrency limit.")
        if prematurely_running:
            print(
                f"Requested cancellation of {len(prematurely_running)} prematurely downstream "
                "instance attempt(s); their workers will stop only those instance processes."
            )
        _report_unavailable(plan)


if __name__ == "__main__":
    main()
