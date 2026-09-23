"""Handle user-facing central scheduler operations."""

import json
import sys
from pathlib import Path

from nro.orchestration.branch_store import BranchStore
from nro.orchestration.resources import (
    GPU_RESOURCE_CLASS,
    SCHEDULABLE_RESOURCE_CLASSES,
    memory_tiers,
)


def supply(registry, request_ids: list[str], options: dict, *, checkout: Path) -> dict:
    """Supply central workers only for requests owned by the authorized checkout."""
    branches = BranchStore(registry.paths.control)
    name = branches.read().topology.registered_checkout(checkout)
    owner = branches.read().topology.records[name].registry_id
    with registry.connection() as db:
        for request_id in request_ids:
            row = db.execute(
                "SELECT registry_id FROM request_owners WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None or row[0] != owner:
                raise ValueError("Worker supply request belongs to another branch")
    registry.cancel_attempts_with_stale_upstreams()
    registry.reconcile_requests()
    from nro.orchestration.scheduler_implementation import run_local_worker
    from nro.orchestration.submission import _submit_workers, _write_worker_script

    submitted = []
    if options["local"]:
        lower_memory = 0
        for tier in memory_tiers(int(options["memory"]), int(options["max_memory"])):
            if registry.worker_capacity_needed(
                request_id=request_ids[0] if request_ids else None,
                resource_class=GPU_RESOURCE_CLASS,
                memory_gb=tier,
                minimum_memory_gb=lower_memory,
            ):
                raise ValueError("Lesion-aware GPU anatomy requires a scheduled GPU worker")
            lower_memory = tier
        process = run_local_worker(
            registry,
            memory_gb=options["memory"],
            drain_seconds=options["drain_minutes"] * 60,
            cpus=options["cpus"],
            poll_interval=options.get("worker_poll_interval", 5.0),
            stdout=sys.stderr,
            wait=False,
        )
        submitted = [process.nro_worker_id]
    elif not options.get("no_submit", False):
        scripts = {}
        for resource_class in SCHEDULABLE_RESOURCE_CLASSES:
            for tier in memory_tiers(options["memory"], options["max_memory"]):
                scripts[(resource_class, tier)] = _write_worker_script(
                    registry,
                    bids_root=registry.paths.bids_root,
                    partition=options["partition"],
                    account=options["account"],
                    hours=options["time"],
                    memory_gb=tier,
                    cpus=options["cpus"],
                    resource_class=resource_class,
                    idle_timeout=options["worker_idle_timeout"],
                    drain_seconds=options["drain_minutes"] * 60,
                )
        for resource_class in SCHEDULABLE_RESOURCE_CLASSES:
            class_submitted: list[str] = []
            lower_memory = 0
            for tier in memory_tiers(options["memory"], options["max_memory"]):
                class_submitted = _submit_workers(
                    registry,
                    request_ids[0] if request_ids else None,
                    scripts[(resource_class, tier)],
                    tier,
                    resource_class=resource_class,
                    minimum_memory_gb=lower_memory,
                )
                if class_submitted:
                    break
                lower_memory = tier
            submitted.extend(class_submitted)
    return {"submitted_workers": submitted}


def supply_needed(registry, request_ids: list[str], options: dict, *, checkout: Path) -> dict:
    """Assess whether a supply request needs a new worker service."""
    branches = BranchStore(registry.paths.control)
    name = branches.read().topology.registered_checkout(checkout)
    owner = branches.read().topology.records[name].registry_id
    with registry.connection() as db:
        for request_id in request_ids:
            row = db.execute(
                "SELECT registry_id FROM request_owners WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None or row[0] != owner:
                raise ValueError("Worker supply request belongs to another branch")
    registry.cancel_attempts_with_stale_upstreams()
    registry.reconcile_requests()
    needed = False
    for resource_class in SCHEDULABLE_RESOURCE_CLASSES:
        lower_memory = 0
        for tier in memory_tiers(int(options["memory"]), int(options["max_memory"])):
            if registry.worker_capacity_needed(
                request_id=request_ids[0] if request_ids else None,
                resource_class=resource_class,
                memory_gb=tier,
                minimum_memory_gb=lower_memory,
            ):
                needed = True
                break
            lower_memory = tier
        if needed:
            break
    return {"needed": needed}


def status(registry, *, checkout: Path, mode: str) -> dict:
    """Report this branch's registered selections, retaining upstream error details."""
    branches = BranchStore(registry.paths.control)
    topology = branches.read().topology
    name = topology.registered_checkout(checkout)
    owner = topology.records[name].registry_id
    from nro.bidsify.store import IngestionStore

    ingestion = [
        {
            **{
                key: row[key]
                for key in (
                    "id",
                    "server",
                    "project",
                    "participant",
                    "session",
                    "state",
                    "stage",
                    "issues",
                )
            },
            "branch": name,
        }
        for row in IngestionStore(registry, branch=name).rows()
    ]
    if not registry.existing_database_path().is_file():
        return {"rows": [], "visible_ids": [], "ingestion": ingestion, "dependencies": []}
    from nro.orchestration.manifests import assess_registry, preview_registry

    if mode == "verify":
        registry.reconcile_attempt_timeouts()
        assess_registry(registry, compiled=True, recover_public=True)
    elif mode not in {"cached", "preview"}:
        raise ValueError("Unknown status mode")
    states = preview_registry(registry, compiled=True) if mode == "preview" else None
    rows = registry.work_item_status_snapshot(read_only=True, artifact_states=states)
    with registry.connection() as db:
        visible = {
            row[0]
            for row in db.execute(
                "SELECT work_item_id FROM branch_work_items WHERE registry_id=?", (owner,)
            )
        }
        scientific = {
            row["work_item_id"]: (row["logical_key"], row["revision"])
            for row in db.execute(
                """
            SELECT b.work_item_id,b.logical_key,r.revision FROM branch_work_items b
            JOIN compiled_revisions r ON r.registry_id=b.registry_id AND r.logical_key=b.logical_key
            WHERE b.registry_id=?""",
                (owner,),
            )
        }
        if name == "main":
            visible.update(
                row[0]
                for row in db.execute(
                    """SELECT i.id FROM work_items i
                    LEFT JOIN work_item_execution e ON e.work_item_id=i.id
                    WHERE e.work_item_id IS NULL AND i.artifact_state!='missing'"""
                )
            )
        workflows = {}
        # A repair deliberately removes request history.  Current workflow
        # bindings still describe which workflows can reproduce each retained
        # work item, so status verification must not depend on an old request.
        for row in db.execute(
            """SELECT DISTINCT b.work_item_id,w.workflow_id
               FROM branch_work_items b
               JOIN work_items i ON i.id=b.work_item_id
               JOIN workflow_bindings wb ON wb.module_lineage_id=i.module_lineage_id
               JOIN workflow_revisions w ON w.id=wb.workflow_revision_id
               WHERE b.registry_id=? AND w.workflow_id LIKE ?""",
            (owner, owner + ":%"),
        ):
            workflows.setdefault(row[0], set()).add(row[1].removeprefix(owner + ":"))
        for row in db.execute(
            """SELECT ri.work_item_id,w.workflow_id FROM request_artifacts ri
            JOIN requests r ON r.id=ri.request_id JOIN request_owners o ON o.request_id=r.id
            JOIN workflow_revisions w ON w.id=r.workflow_revision_id WHERE o.registry_id=?""",
            (owner,),
        ):
            workflows.setdefault(row[0], set()).add(row[1].removeprefix(owner + ":"))
        active_resume_workflows = {}
        latest_resume_workflow = {}
        for row in db.execute(
            """SELECT ri.work_item_id,w.workflow_id,r.state,ri.demand_state,ri.role
               FROM request_work_items ri
               JOIN requests r ON r.id=ri.request_id
               JOIN request_owners o ON o.request_id=r.id
               JOIN workflow_revisions w ON w.id=r.workflow_revision_id
               WHERE o.registry_id=? AND r.state!='registered'
               ORDER BY r.updated_at DESC,r.created_at DESC,r.id DESC""",
            (owner,),
        ):
            if row["role"] != "target":
                continue
            work_item_id = int(row["work_item_id"])
            workflow = str(row["workflow_id"]).removeprefix(owner + ":")
            latest_resume_workflow.setdefault(work_item_id, workflow)
            if row["state"] == "active" and row["demand_state"] == "active":
                active_resume_workflows.setdefault(work_item_id, set()).add(workflow)
        worker_logs = {
            int(row["work_item_id"]): str(
                registry.paths.workers / f"slurm-{row['slurm_job_id']}.log"
            )
            for row in db.execute(
                """SELECT a.work_item_id,w.slurm_job_id FROM attempts a
                   JOIN workers w ON w.id=a.worker_id
                   WHERE w.slurm_job_id IS NOT NULL ORDER BY a.id"""
            )
        }
    for row in rows:
        if row["id"] in visible:
            logical_key, revision = scientific.get(row["id"], (row["work_item_key"], None))
            row["logical_key"] = logical_key
            row["scientific_revision"] = revision
        row["workflow_ids"] = ",".join(sorted(workflows.get(row["id"], ()))) or (
            row.get("workflow_ids", "") if name == "main" else ""
        )
        resume_workflows = active_resume_workflows.get(int(row["id"]))
        if resume_workflows is None:
            latest = latest_resume_workflow.get(int(row["id"]))
            resume_workflows = {latest} if latest else set()
        row["resume_workflow_ids"] = ",".join(sorted(resume_workflows))
        row["worker_log_path"] = worker_logs.get(int(row["id"]))
    reported = set(visible)
    for row in rows:
        if int(row["id"]) in visible:
            reported.update(int(value) for value in row.get("root_failure_ids", ()))
    rows = [row for row in rows if int(row["id"]) in reported]
    dependencies = [
        (work_item_id, upstream_id)
        for work_item_id, upstream_id in registry.work_item_dependencies(read_only=True)
        if work_item_id in reported and upstream_id in reported
    ]
    return {
        "rows": rows,
        "visible_ids": sorted(visible),
        "ingestion": ingestion,
        "dependencies": dependencies,
    }


def stop(registry, *, checkout: Path, selection: dict) -> dict:
    """Cancel selected branch demand without cancelling another branch's requests."""
    branches = BranchStore(registry.paths.control)
    topology = branches.read().topology
    name = topology.registered_checkout(checkout)
    return registry.request_cancellation(
        **selection, branch_registry_id=topology.records[name].registry_id
    )


def logs(
    registry,
    *,
    checkout: Path,
    selection: dict,
    worker_level: bool,
    running_only: bool = False,
) -> dict:
    """Resolve logs of selected branch artifacts or the workers that executed them."""
    from nro.engine.cli import matches_module_lineage, matches_work_item_selectors

    name = BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    report = status(registry, checkout=checkout, mode="cached")
    visible = set(report["visible_ids"])
    requested_modules = set(selection["modules"])
    bidsify_selected = "bidsify" in requested_modules
    scientific_modules = requested_modules - {"bidsify"}
    scientific_selected = not requested_modules or bool(scientific_modules)
    selected = [
        row
        for row in report["rows"]
        if scientific_selected
        and row["id"] in visible
        and all(
            not selection[key] or row[field] in selection[key]
            for key, field in (
                ("projects", "project"),
                ("participants", "participant"),
            )
        )
        and (not scientific_modules or row["module"] in scientific_modules)
        and matches_module_lineage(
            row["module"], row.get("directory_label", ""), selection.get("lineages", ())
        )
        and (
            not selection["workflows"]
            or set(selection["workflows"]).intersection(
                str(row.get("workflow_ids") or "").split(",")
            )
        )
        and matches_work_item_selectors(json.loads(row["entities_json"]), selection["selectors"])
        and (not running_only or row.get("status") == "Running")
    ]
    if worker_level:
        ids = {row["id"] for row in selected}
        with registry.connection() as db:
            paths = [
                str(registry.paths.workers / f"slurm-{row['slurm_job_id']}.log")
                for row in db.execute("""SELECT a.work_item_id,w.slurm_job_id FROM attempts a
                    JOIN workers w ON w.id=a.worker_id WHERE w.slurm_job_id IS NOT NULL""")
                if row["work_item_id"] in ids
            ]
    else:
        paths = [row["log_path"] for row in selected if row.get("log_path")]
    if (
        bidsify_selected
        and not selection["workflows"]
        and not selection.get("lineages")
        and set(selection["selectors"]) <= {"ses"}
    ):
        from nro.bidsify.store import IngestionStore

        sessions = selection["selectors"].get("ses", ())
        store = IngestionStore(registry, branch=name)
        paths.extend(
            str(store.root / f"{row['id']}.log")
            for row in report["ingestion"]
            if (
                not selection.get("ingestion_projects")
                or row["project"] in selection["ingestion_projects"]
            )
            and (not selection["participants"] or row["participant"] in selection["participants"])
            and (not sessions or row["session"] in sessions)
            and (not running_only or row.get("state") == "running")
        )
    return {"paths": sorted(set(paths))}


def pool_operation(registry, *, checkout: Path, operation: str, concurrency=None) -> dict:
    """Apply explicit lab-wide pool controls from an authorized checkout."""
    BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    if operation == "concurrency":
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Concurrency must be a positive integer")
        return {"updated_requests": registry.set_active_concurrency(concurrency)}
    if operation != "stop_workers":
        raise ValueError("Unknown pool operation")
    from nro.orchestration.worker_control import cancel_worker_allocations

    shutdown = registry.request_worker_shutdown()
    stopped, failures = cancel_worker_allocations(registry, shutdown)
    return dict(shutdown, stopped_jobs=stopped, failures=failures)


def require_environment_idle(registry, *, checkout: Path, environment: Path) -> dict:
    """Reject maintenance while any demanded recipe or attempt uses this environment."""
    BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    if not environment.is_absolute():
        raise ValueError("Environment path must be absolute")
    with registry.connection() as db:
        commands = [
            row[0]
            for row in db.execute("""SELECT DISTINCT i.command_json FROM work_items i
            JOIN request_work_items ri ON ri.work_item_id=i.id JOIN requests r ON r.id=ri.request_id
            WHERE r.state='active' AND ri.demand_state='active'
            UNION SELECT COALESCE(e.command_json,i.command_json) FROM attempts a
            JOIN work_items i ON i.id=a.work_item_id LEFT JOIN attempt_execution e ON e.attempt_id=a.id
            WHERE a.state IN ('queued','running','cancel_requested')""")
        ]
        if any(
            Path(json.loads(command)[0]).absolute().is_relative_to(environment.absolute())
            for command in commands
        ):
            raise ValueError(
                "Outstanding work uses this environment; stop its demand and attempts before maintenance"
            )
        from nro.bidsify.index import IngestionIndex

        if any(
            row["state"] in {"queued", "running"}
            and row.get("execution")
            and Path(row["execution"]["python"]).absolute().is_relative_to(environment.absolute())
            for row in IngestionIndex(registry).execution_records()
        ):
            raise ValueError(
                "Outstanding ingestion uses this environment; stop it before maintenance"
            )
    return {"idle": True}


def installation_activity(registry, *, checkout: Path) -> dict:
    """Report work that must finish or stop before shared installation maintenance."""
    BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    activity = registry.worker_pool_activity()
    with registry.connection() as db:
        attempts = int(
            db.execute(
                "SELECT COUNT(*) FROM attempts WHERE state IN ('running','cancel_requested')"
            ).fetchone()[0]
        )
    from nro.bidsify.index import IngestionIndex

    ingestion = sum(row["state"] == "running" for row in IngestionIndex(registry).rows())
    return {
        "workers": len(activity["workers"]),
        "submissions": len(activity["submissions"]),
        "attempts": attempts,
        "ingestion": ingestion,
    }


def installation_prepare(registry, *, checkout: Path, action: str) -> dict:
    """Install a global barrier and begin a drain or immediate stop."""
    if action not in {"drain", "stop"}:
        raise ValueError("Installation action must be drain or stop")
    checkout = checkout.resolve()
    BranchStore(registry.paths.control).read().topology.registered_checkout(checkout)
    with registry.connection(write=True) as db:
        rows = {
            str(row["key"]): str(row["value"])
            for row in db.execute(
                "SELECT key,value FROM metadata WHERE key IN "
                "('maintenance_mode','installation_checkout','installation_action')"
            )
        }
        if rows.get("maintenance_mode") not in {None, "installation"}:
            raise ValueError(
                f"Shared registry is already in {rows['maintenance_mode']} maintenance"
            )
        if rows.get("installation_checkout") not in {None, str(checkout)}:
            raise ValueError("Shared installation maintenance belongs to another checkout")
        db.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (
                ("maintenance_mode", "installation"),
                ("installation_checkout", str(checkout)),
                ("installation_action", action),
            ),
        )
    return installation_progress(registry, checkout=checkout)


def installation_progress(registry, *, checkout: Path) -> dict:
    """Advance installation quiescence and report whether direct maintenance is safe."""
    checkout = checkout.resolve()
    with registry.connection() as db:
        rows = {
            str(row["key"]): str(row["value"])
            for row in db.execute(
                "SELECT key,value FROM metadata WHERE key IN "
                "('maintenance_mode','installation_checkout','installation_action')"
            )
        }
    if rows.get("maintenance_mode") != "installation" or rows.get("installation_checkout") != str(
        checkout
    ):
        raise ValueError("This checkout does not own installation maintenance")
    action = rows.get("installation_action", "drain")
    activity = installation_activity(registry, checkout=checkout)
    stopped = 0
    failures: list[str] = []
    if action == "stop" or not (activity["attempts"] or activity["ingestion"]):
        from nro.orchestration.worker_control import active_pool_members, cancel_worker_allocations

        shutdown = registry.request_worker_shutdown(all_users=True)
        stopped, failures = cancel_worker_allocations(
            registry,
            shutdown,
            update_registry=False,
        )
        if not failures:
            workers, jobs = active_pool_members(registry.worker_pool_activity())
            if not workers and not jobs:
                for submission_id, _job_id in shutdown["submissions"]:
                    registry.update_submission(submission_id, state="cancelled")
                registry.recover_orphaned_attempts()
                registry.confirm_worker_shutdown(row["id"] for row in shutdown["worker_rows"])
        activity = installation_activity(registry, checkout=checkout)
    done = not any(activity.values()) and not failures
    return {
        **activity,
        "action": action,
        "done": done,
        "stopped_jobs": stopped,
        "failures": failures,
    }
