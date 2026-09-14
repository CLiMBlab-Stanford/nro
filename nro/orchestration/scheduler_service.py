"""Run the ephemeral service that owns the central scheduler registry."""

import argparse
import fcntl
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.artifact_resolution import scientific_contracts
from nro.orchestration.branch_admission import _admit_resolved
from nro.orchestration.branch_reconciliation import candidates_locked, resolve_payload
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.compiled_request import decode_spec
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.scheduler_bus import DEFAULT_IDLE_GRACE_SECONDS, HEARTBEAT_SECONDS
from nro.orchestration.source_snapshots import SourceSnapshot

_STOP = False


def _request_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def admit(
    registry,
    payload: dict,
    *,
    checkout: Path,
    site_values: dict,
    assess: bool = True,
    source_verified: bool = False,
    request_id: str | None = None,
) -> str:
    """Keep an installed environment out of maintenance until demand is published."""
    record_path = checkout / ".nro-installation.json"
    if not record_path.exists():
        return _admit(
            registry,
            payload,
            checkout=checkout,
            site_values=site_values,
            assess=assess,
            source_verified=source_verified,
            request_id=request_id,
        )
    lock = checkout / ".nro-install.lock"
    if not lock.is_file():
        raise ValueError("Installed checkout lacks its maintenance lock; rerun installation")
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("The submitting installation is undergoing maintenance") from error
        record = json.loads(record_path.read_text())
        if record.get("checkout") != str(checkout) or not record.get("ready"):
            raise ValueError("The submitting installation is not ready")
        if str(Path(record["environment"]) / "bin/python") != payload["python"]:
            raise ValueError("Request interpreter does not match the installed environment")
        return _admit(
            registry,
            payload,
            checkout=checkout,
            site_values=site_values,
            assess=assess,
            source_verified=source_verified,
            request_id=request_id,
        )


def admit_many(
    registry,
    entries: list[dict],
    *,
    checkout: Path,
    site_values: dict,
    message_id: str = "direct",
) -> list[str]:
    """Admit one invocation's requests after one source check and registry assessment."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("Admission batch must contain at least one request")
    projects = []
    sources = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"project", "payload"}:
            raise ValueError("Invalid admission batch entry")
        project, payload = entry["project"], entry["payload"]
        if not isinstance(project, str) or not project or payload.get("project") != project:
            raise ValueError("Admission batch project does not match its payload")
        projects.append(project)
        descriptor = payload.get("source")
        if not isinstance(descriptor, dict) or set(descriptor) != {"root", "digest"}:
            raise ValueError("Admission batch has an invalid source")
        sources[(descriptor["root"], descriptor["digest"])] = SourceSnapshot(
            Path(descriptor["root"]), descriptor["digest"]
        )
    for source in sources.values():
        source.verify_manifest()
    from nro.orchestration.manifests import assess_registry

    assess_registry(registry, projects=tuple(dict.fromkeys(projects)), compiled=True)
    from nro.orchestration.registry import Registry

    return [
        admit(
            Registry.for_project(
                entry["project"],
                bids_root=registry.paths.bids_root,
                registry_path=registry.paths.control,
            ),
            entry["payload"],
            checkout=checkout,
            site_values=site_values,
            assess=False,
            source_verified=True,
            request_id=f"{message_id}-{index}",
        )
        for index, entry in enumerate(entries)
    ]


def _admit(
    registry,
    payload: dict,
    *,
    checkout: Path,
    site_values: dict,
    assess: bool = True,
    source_verified: bool = False,
    request_id: str | None = None,
) -> str:
    """Validate and admit a detached graph without opening its scientific registry.

    The transport holds the cache publication lock until this call returns.
    Scientific revisions reject delayed requests from another checkout of the
    same branch. Central registration authorizes the checkout, while the source
    digest binds the request to the captured implementation.
    """
    branches = BranchStore(registry.paths.control)
    source = SourceSnapshot(Path(payload["source"]["root"]), payload["source"]["digest"])
    if not source_verified:
        source.verify()
    if type(payload.get("demand", True)) is not bool:
        raise ValueError("Demand must be a boolean")
    if type(payload["concurrency"]) is not int or payload["concurrency"] < 1:
        raise ValueError("Concurrency must be a positive integer")
    if not Path(payload["python"]).is_absolute() or not Path(payload["python"]).is_file():
        raise ValueError("The job interpreter is unavailable")
    context = ExecutionContext.from_dict(payload["context"])
    if any(
        getattr(context.paths, key) != Path(site_values[key]).resolve()
        for key in ("bids", "work", "development")
    ):
        raise ValueError("Compiled request paths differ from the central site")
    if context.project != registry.paths.project:
        raise ValueError("Compiled request belongs to another project")
    specs = tuple(decode_spec(value) for value in payload["specifications"])
    contracts = payload.get("contracts")
    if contracts is None:
        contracts = scientific_contracts(specs)
    elif (
        not isinstance(contracts, dict)
        or set(contracts) != {spec.key for spec in specs}
        or not all(isinstance(value, dict) for value in contracts.values())
    ):
        raise ValueError("Scientific contracts must cover the complete request")
    if set(payload["revisions"]) != set(contracts):
        raise ValueError("Scientific revisions must cover the complete request")
    if assess:
        from nro.orchestration.manifests import assess_registry

        assess_registry(registry, projects=(context.project,), compiled=True)
    if payload["branch"] == "main":
        from nro.orchestration.scheduler_implementation import implementation_path

        active = json.loads(implementation_path(registry.paths.control).read_text())
        if (
            Path(str(active.get("checkout", ""))).resolve() != checkout.resolve()
            or active.get("release") != payload["release"]
            or active.get("source_digest") != source.digest
        ):
            raise ValueError("Main request does not match its approved release")
    with branches._lock():
        topology = branches.read().topology
        name = topology.registered_checkout(checkout)
        if (
            name != payload["branch"]
            or topology.records[name].registry_id != payload["registry_id"]
        ):
            raise ValueError("Request ownership does not match the submitting checkout")
        with registry.connection(write=True) as db:
            for key, contract in contracts.items():
                revision = payload["revisions"][key]
                if type(revision) is not int or revision < 1:
                    raise ValueError("Scientific revisions must be positive integers")
                digest = fingerprint(contract)
                row = db.execute(
                    "SELECT revision,fingerprint FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
                    (payload["registry_id"], key),
                ).fetchone()
                if row and (
                    revision < row["revision"]
                    or (revision == row["revision"] and digest != row["fingerprint"])
                ):
                    raise ValueError(
                        "A newer scientific request was admitted; refresh this checkout before retrying"
                    )
                db.execute(
                    """INSERT INTO compiled_revisions VALUES (?,?,?,?)
                    ON CONFLICT(registry_id,logical_key) DO UPDATE SET revision=excluded.revision,
                    fingerprint=excluded.fingerprint""",
                    (payload["registry_id"], key, revision, digest),
                )
            plan = resolve_payload(topology, payload, candidates_locked(db, context.project))
            return _admit_resolved(registry, db, plan, payload, request_id=request_id)


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
        tier, scripts = options["memory"], {}
        while True:
            scripts[tier] = _write_worker_script(
                registry,
                bids_root=registry.paths.bids_root,
                partition=options["partition"],
                account=options["account"],
                hours=options["time"],
                memory_gb=tier,
                cpus=options["cpus"],
                idle_timeout=options["worker_idle_timeout"],
                drain_seconds=options["drain_minutes"] * 60,
            )
            if tier >= options["max_memory"]:
                break
            tier = min(options["max_memory"], tier * 2)
        submitted = _submit_workers(
            registry,
            request_ids[0] if request_ids else None,
            scripts[options["memory"]],
            options["memory"],
        )
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
    needed = registry.worker_capacity_needed(
        request_id=request_ids[0] if request_ids else None,
        resource_class="large",
        memory_gb=int(options["memory"]),
    )
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
        assess_registry(registry, compiled=True)
    elif mode not in {"cached", "preview"}:
        raise ValueError("Unknown status mode")
    states = preview_registry(registry, compiled=True) if mode == "preview" else None
    rows = registry.instance_status_snapshot(read_only=True, artifact_states=states)
    with registry.connection() as db:
        visible = {
            row[0]
            for row in db.execute(
                "SELECT instance_id FROM branch_instances WHERE registry_id=?", (owner,)
            )
        }
        scientific = {
            row["instance_id"]: (row["logical_key"], row["revision"])
            for row in db.execute(
                """
            SELECT b.instance_id,b.logical_key,r.revision FROM branch_instances b
            JOIN compiled_revisions r ON r.registry_id=b.registry_id AND r.logical_key=b.logical_key
            WHERE b.registry_id=?""",
                (owner,),
            )
        }
        if name == "main":
            visible.update(
                row[0]
                for row in db.execute(
                    """SELECT i.id FROM instances i
                    LEFT JOIN instance_execution e ON e.instance_id=i.id
                    WHERE e.instance_id IS NULL AND i.artifact_state!='missing'"""
                )
            )
        workflows = {}
        for row in db.execute(
            """SELECT ri.instance_id,w.workflow_id FROM request_artifacts ri
            JOIN requests r ON r.id=ri.request_id JOIN request_owners o ON o.request_id=r.id
            JOIN workflow_revisions w ON w.id=r.workflow_revision_id WHERE o.registry_id=?""",
            (owner,),
        ):
            workflows.setdefault(row[0], set()).add(row[1].removeprefix(owner + ":"))
        worker_logs = {
            int(row["instance_id"]): str(
                registry.paths.workers / f"slurm-{row['slurm_job_id']}.log"
            )
            for row in db.execute(
                """SELECT a.instance_id,w.slurm_job_id FROM attempts a
                   JOIN workers w ON w.id=a.worker_id
                   WHERE w.slurm_job_id IS NOT NULL ORDER BY a.id"""
            )
        }
    for row in rows:
        if row["id"] in visible:
            logical_key, revision = scientific.get(row["id"], (row["instance_key"], None))
            row["logical_key"] = logical_key
            row["scientific_revision"] = revision
        row["workflow_ids"] = ",".join(sorted(workflows.get(row["id"], ()))) or (
            row.get("workflow_ids", "") if name == "main" else ""
        )
        row["worker_log_path"] = worker_logs.get(int(row["id"]))
    return {
        "rows": rows,
        "visible_ids": sorted(visible),
        "ingestion": ingestion,
        "dependencies": registry.instance_dependencies(read_only=True),
    }


def stop(registry, *, checkout: Path, selection: dict) -> dict:
    """Cancel selected branch demand without cancelling another branch's requests."""
    branches = BranchStore(registry.paths.control)
    topology = branches.read().topology
    name = topology.registered_checkout(checkout)
    return registry.request_cancellation(
        **selection, branch_registry_id=topology.records[name].registry_id
    )


def logs(registry, *, checkout: Path, selection: dict, instance_level: bool) -> dict:
    """Resolve logs of selected branch artifacts or the workers that executed them."""
    from nro.engine.cli import matches_instance_selectors

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
        and (
            not selection["workflows"]
            or set(selection["workflows"]).intersection(row["workflow_ids"].split(","))
        )
        and matches_instance_selectors(json.loads(row["entities_json"]), selection["selectors"])
    ]
    if instance_level:
        paths = [row["log_path"] for row in selected if row.get("log_path")]
    else:
        ids = {row["id"] for row in selected}
        with registry.connection() as db:
            paths = [
                str(registry.paths.workers / f"slurm-{row['slurm_job_id']}.log")
                for row in db.execute("""SELECT a.instance_id,w.slurm_job_id FROM attempts a
                    JOIN workers w ON w.id=a.worker_id WHERE w.slurm_job_id IS NOT NULL""")
                if row["instance_id"] in ids
            ]
    if bidsify_selected and not selection["workflows"] and set(selection["selectors"]) <= {"ses"}:
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
            for row in db.execute("""SELECT DISTINCT i.command_json FROM instances i
            JOIN request_instances ri ON ri.instance_id=i.id JOIN requests r ON r.id=ri.request_id
            WHERE r.state='active' AND ri.demand_state='active'
            UNION SELECT COALESCE(e.command_json,i.command_json) FROM attempts a
            JOIN instances i ON i.id=a.instance_id LEFT JOIN attempt_execution e ON e.attempt_id=a.id
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


def _validate_worker_event(registry, message: dict, *, registering: bool = False) -> None:
    """Fence stale workers and reject events older than the last applied sequence."""
    worker_id = str(message["worker_id"])
    token = str(message["worker_token"])
    sequence = int(message["sequence"])
    token_key = f"worker_token:{worker_id}"
    sequence_key = f"worker_sequence:{worker_id}"
    with registry.connection(write=True) as db:
        current_token = db.execute(
            "SELECT value FROM metadata WHERE key=?", (token_key,)
        ).fetchone()
        current_sequence = db.execute(
            "SELECT value FROM metadata WHERE key=?", (sequence_key,)
        ).fetchone()
        if registering:
            if current_token is not None and current_token[0] != token:
                worker = db.execute("SELECT state FROM workers WHERE id=?", (worker_id,)).fetchone()
                if worker is not None and worker[0] not in {"exited", "terminated", "lost"}:
                    raise ValueError("Worker identity is already owned by another process")
            previous = 0
        else:
            if current_token is None or current_token[0] != token:
                raise ValueError("Worker fencing token is obsolete")
            previous = int(current_sequence[0]) if current_sequence else 0
        if sequence < previous:
            raise ValueError("Worker event sequence predates its applied predecessor")


def _commit_worker_event(registry, message: dict) -> None:
    """Advance a worker sequence only after its operation has succeeded."""
    worker_id = str(message["worker_id"])
    token = str(message["worker_token"])
    sequence = int(message["sequence"])
    token_key = f"worker_token:{worker_id}"
    sequence_key = f"worker_sequence:{worker_id}"
    with registry.connection(write=True) as db:
        db.execute(
            "INSERT INTO metadata(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (token_key, token),
        )
        db.execute(
            "INSERT INTO metadata(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (sequence_key, str(sequence)),
        )


def _worker_script(registry, *, memory_gb: int, profile: str | None) -> Path:
    suffix = f"-{profile}" if profile else ""
    exact = registry.paths.workers / f"worker-large-{memory_gb}gb{suffix}.sbatch"
    if exact.is_file():
        return exact
    candidates = sorted(registry.paths.workers.glob(f"worker-large-{memory_gb}gb-*.sbatch"))
    if candidates:
        return candidates[-1]
    raise ValueError(f"No prepared {memory_gb} GB worker script is available")


def _submit_reserved(
    registry, submission_id: int, script: Path, *, dependency: str | None = None
) -> str:
    """Run sbatch outside a registry transaction and finalize its reserved intent."""
    from nro.orchestration.scheduler_implementation import validate_worker_script

    validate_worker_script(registry.paths.control, script)
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterany:{dependency}")
    command.append(str(script))
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id:
            raise RuntimeError(f"sbatch returned no worker job ID: {result.stdout!r}")
        registry.update_submission(submission_id, state="submitted", slurm_job_id=job_id)
        return job_id
    except BaseException:
        registry.update_submission(submission_id, state="error")
        raise


def worker_operation(registry, message: dict) -> object:
    """Apply one ordered worker event and return its acknowledgement or assignment."""
    action = message["action"]
    _validate_worker_event(registry, message, registering=action == "register")
    result = _apply_worker_operation(registry, message)
    _commit_worker_event(registry, message)
    return result


def _apply_worker_operation(registry, message: dict) -> object:
    """Apply one worker operation after fencing and before sequence publication."""
    action = message["action"]
    worker_id = str(message["worker_id"])
    if action == "register":
        registry.register_worker(
            worker_id,
            resource_class=message["resource_class"],
            memory_gb=int(message["memory_gb"]),
            slurm_job_id=message.get("slurm_job_id"),
            user_name=str(message["user_name"]),
            hostname=str(message["hostname"]),
            pid=int(message["pid"]),
        )
        return None
    if action == "heartbeat":
        registry.heartbeat_worker(worker_id, state=message["state"])
        return None
    if action == "shutdown_requested":
        return registry.worker_shutdown_requested(worker_id)
    if action == "submission_running":
        registry.mark_submission_running(message.get("slurm_job_id", ""))
        return None
    if action == "submission_complete":
        registry.mark_submission_complete(message.get("slurm_job_id"))
        return None
    if action == "reconcile_submissions":
        return registry.reconcile_scheduler_submissions()
    if action == "recover_orphans":
        return registry.recover_orphaned_attempts()
    if action == "claim":
        envelope = registry.current_worker_assignment(worker_id)
        if envelope is None:
            envelope = registry.claim_ready_instance(
                worker_id,
                tuple(message["resource_classes"]),
                memory_gb=int(message["memory_gb"]),
            )
        return None if envelope is None else envelope.as_dict()
    if action == "claim_ingestion":
        from nro.bidsify.index import IngestionIndex

        return IngestionIndex(registry).claim(worker_id, int(message["memory_gb"]))
    if action == "finish_ingestion":
        from nro.bidsify.store import IngestionStore

        IngestionStore(registry, branch=message["branch"]).finish(
            message["request_id"],
            worker_id,
            state=message["state"],
            changes=message.get("changes"),
        )
        return None
    if action == "attempt_cancel_requested":
        return registry.attempt_cancel_requested(int(message["attempt_id"]))
    if action == "attempt_process":
        registry.record_attempt_process(
            int(message["attempt_id"]), int(message["process_group_id"])
        )
        return None
    if action == "finish_attempt":
        registry.finish_attempt(
            int(message["attempt_id"]),
            state=message["state"],
            error_type=message.get("error_type"),
            error_message=message.get("error_message"),
        )
        return None
    if action == "record_oom":
        return registry.record_oom(int(message["attempt_id"]), message=message["message"])
    if action == "cancel_failed_descendants":
        return registry.cancel_attempts_downstream_of_failure(int(message["instance_id"]))
    if action == "runner_graph_signature":
        from nro.orchestration.worker import _runner_graph_signature

        return _runner_graph_signature(registry, int(message["instance_id"]))
    if action == "attempt_summary":
        with registry.connection() as db:
            row = db.execute(
                "SELECT state,error_message FROM attempts WHERE id=?",
                (int(message["attempt_id"]),),
            ).fetchone()
        if row is None:
            return "attempt record missing"
        return f"attempt state={row['state']}" + (
            f"; error={row['error_message']}" if row["error_message"] else ""
        )
    if action == "record_completion":
        from nro.orchestration.manifests import record_completion

        return record_completion(
            registry,
            instance_id=int(message["instance_id"]),
            attempt_id=int(message["attempt_id"]),
            outputs=tuple(Path(path) for path in message["outputs"]),
        )
    if action == "refresh":
        from nro.orchestration.assessment import AssessmentConflict
        from nro.orchestration.branch_reconciliation import reconcile_branch_requests
        from nro.orchestration.manifests import assess_registry

        cancelled = []
        if registry.reserve_artifact_assessment():
            try:
                demanded = registry.demanded_instance_ids()
                if demanded:
                    try:
                        assess_registry(registry, instance_ids=demanded, compiled=True)
                    except AssessmentConflict:
                        pass
                reconcile_branch_requests(registry)
                cancelled = registry.cancel_attempts_with_stale_upstreams()
                registry.reconcile_requests()
            finally:
                registry.finish_artifact_assessment()
        return len(cancelled)
    if action == "required_memory":
        return registry.required_memory_above(int(message["memory_gb"]))
    if action == "request_capacity":
        memory = int(message["memory_gb"])
        kind = message["kind"]
        if kind == "successor":
            reservation = registry.reserve_worker_successor(
                worker_id=worker_id, resource_class="large", memory_gb=memory
            )
            if reservation is None:
                return None
            script = _worker_script(registry, memory_gb=memory, profile=message.get("profile"))
            with registry.connection() as db:
                row = db.execute(
                    "SELECT slurm_job_id FROM workers WHERE id=?", (worker_id,)
                ).fetchone()
            return _submit_reserved(
                registry, reservation[0], script, dependency=row[0] if row else None
            )
        if kind == "adaptive":
            reservation = registry.reserve_adaptive_worker(resource_class="large", memory_gb=memory)
            if reservation is None:
                return None
            script = _worker_script(registry, memory_gb=memory, profile=message.get("profile"))
            return _submit_reserved(registry, reservation[0], script)
        if kind == "expand":
            reservations = registry.reserve_worker_submissions(
                request_id=None, resource_class="large", memory_gb=memory
            )
            if not reservations:
                return []
            script = _worker_script(registry, memory_gb=memory, profile=message.get("profile"))
            return [
                _submit_reserved(registry, submission_id, script)
                for submission_id, _token in reservations
            ]
        raise ValueError("Unknown capacity request")
    if action == "close":
        registry.close_worker(worker_id, state=message["state"])
        return None
    if action == "cleanup_cache":
        from nro.orchestration.execution_cache import cleanup_cache

        cleanup_cache(registry)
        return None
    raise ValueError("Unknown worker operation")


def dispatch(registry, message: dict, *, values: dict, message_id: str) -> object:
    """Apply one validated command through the service's registry authority."""
    global _STOP
    if message["operation"] == "server_shutdown":
        from nro.orchestration.scheduler_bus import publish_shutdown

        publish_shutdown(
            registry.paths.control,
            token=os.environ["NRO_SCHEDULER_TOKEN"],
            generation=int(os.environ["NRO_SCHEDULER_GENERATION"]),
        )
        _STOP = True
        result = {"stopping": True}
    elif message["operation"] == "worker":
        result = worker_operation(registry, message)
    elif message["operation"] == "admit":
        request_id = admit(
            registry,
            message["payload"],
            checkout=Path(message["checkout"]),
            site_values=values,
            request_id=message_id,
        )
        result = {"request_id": request_id}
    elif message["operation"] == "admit_many":
        result = {
            "request_ids": admit_many(
                registry,
                message["entries"],
                checkout=Path(message["checkout"]),
                site_values=values,
                message_id=message_id,
            )
        }
    elif message["operation"] == "supply":
        result = supply(
            registry,
            message["request_ids"],
            message["options"],
            checkout=Path(message["checkout"]),
        )
    elif message["operation"] == "supply_needed":
        result = supply_needed(
            registry,
            message["request_ids"],
            message["options"],
            checkout=Path(message["checkout"]),
        )
    elif message["operation"] == "status":
        result = status(registry, checkout=Path(message["checkout"]), mode=message["mode"])
    elif message["operation"] == "stop":
        result = stop(registry, checkout=Path(message["checkout"]), selection=message["selection"])
    elif message["operation"] == "logs":
        result = logs(
            registry,
            checkout=Path(message["checkout"]),
            selection=message["selection"],
            instance_level=message["instance_level"],
        )
    elif message["operation"] in {"concurrency", "stop_workers"}:
        result = pool_operation(
            registry,
            checkout=Path(message["checkout"]),
            operation=message["operation"],
            concurrency=message.get("concurrency"),
        )
    elif message["operation"] == "environment_idle":
        result = require_environment_idle(
            registry,
            checkout=Path(message["checkout"]),
            environment=Path(message["environment"]),
        )
    elif message["operation"] == "installation_activity":
        result = installation_activity(registry, checkout=Path(message["checkout"]))
    elif message["operation"] == "installation_prepare":
        result = installation_prepare(
            registry,
            checkout=Path(message["checkout"]),
            action=message["action"],
        )
    elif message["operation"] == "installation_progress":
        result = installation_progress(registry, checkout=Path(message["checkout"]))
    elif message["operation"] == "purge_snapshot":
        from nro.orchestration.branch_purge import snapshot

        result = snapshot(registry, checkout=Path(message["checkout"]), site_values=values)
    elif message["operation"] == "purge":
        from nro.orchestration.branch_purge import purge
        from nro.orchestration.scheduler_bus import publish_progress

        result = purge(
            registry,
            checkout=Path(message["checkout"]),
            site_values=values,
            plan=message["plan"],
            logs_only=message["logs_only"],
            dry_run=message["dry_run"],
            progress=lambda phase, completed, total: publish_progress(
                registry.paths.control,
                message_id,
                phase=phase,
                completed=completed,
                total=total,
            ),
        )
    elif message["operation"] == "cache":
        BranchStore(registry.paths.control).read().topology.registered_checkout(
            Path(message["checkout"])
        )
        from nro.orchestration.execution_cache import collect_cache

        collection = collect_cache(
            registry,
            dry_run=message["dry_run"],
            service=message.get("service"),
            approved=None if message["approved"] is None else tuple(map(Path, message["approved"])),
        )
        result = dict(
            paths=list(map(str, collection.paths)),
            retained=list(map(str, collection.retained)),
            reason=collection.reason,
        )
    elif message["operation"] == "repair_prepare":
        from nro.orchestration.branch_repair import prepare

        result = prepare(
            registry,
            checkout=Path(message["checkout"]),
            reservation=message.get("reservation"),
            allow_stop=message.get("allow_stop", False),
        )
    elif message["operation"] == "repair_finish":
        from nro.orchestration.branch_repair import finish

        result = finish(
            registry, checkout=Path(message["checkout"]), reservation=message["reservation"]
        )
    elif message["operation"] == "promotion_preview":
        from nro.orchestration.promotion import preview

        result = preview(
            registry,
            checkout=Path(message["checkout"]),
            source=message["source"],
            requests=message["requests"],
            pr=message["pr"],
            attest=message["attest"],
        )
    elif message["operation"] == "promotion_publish":
        from nro.orchestration.promotion import publish

        result = publish(
            registry,
            checkout=Path(message["checkout"]),
            report=message["report"],
            replace=message["replace"],
            attest=message["attest"],
        )
    elif message["operation"] == "publish":
        topology = BranchStore(registry.paths.control).read().topology
        name = topology.registered_checkout(Path(message["checkout"]))
        with registry.connection() as db:
            owner = db.execute(
                "SELECT registry_id FROM request_owners WHERE request_id=?",
                (message["request"],),
            ).fetchone()
            if owner is None or owner[0] != topology.records[name].registry_id:
                raise ValueError("Publication request belongs to another branch")
        destination = Path(message["destination"]).expanduser().resolve()
        if any(
            destination.is_relative_to(Path(values[key]).resolve())
            for key in ("bids", "work", "development", "registry")
        ):
            raise ValueError(
                "Standalone publication must be outside managed data and control stores"
            )
        from nro.orchestration.publish import publish

        result = {
            "destination": str(
                publish(
                    registry,
                    request_id=message["request"],
                    destination=destination,
                    validate=message["validate"],
                    compiled=True,
                )
            )
        }
    elif message["operation"] == "branch_update":
        from nro.orchestration.branch_operations import update

        result = update(
            registry,
            checkout=Path(message["checkout"]),
            branch=message["branch"],
            action=message["action"],
            revision=message["revision"],
            parent=message.get("parent"),
        )
    else:
        raise ValueError("Unsupported scheduler operation")
    return result


def _message_response(registry, record: dict, *, values: dict) -> dict:
    """Execute one message; its durable response is the processed-message record."""
    from nro.orchestration.registry import Registry

    operation_registry = Registry.for_project(
        record["payload"].get("project", ""),
        bids_root=values["bids"],
        registry_path=values["registry"],
    )
    try:
        response = {
            "result": dispatch(
                operation_registry,
                record["payload"],
                values=values,
                message_id=record["id"],
            )
        }
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        response = {"error": str(error)}
    return response


def _quiet_message(record: dict) -> bool:
    payload = record["payload"]
    return payload.get("operation") == "worker" and payload.get("action") in {
        "heartbeat",
        "shutdown_requested",
        "attempt_cancel_requested",
        "required_memory",
    }


def _process_record(registry, record: dict, *, values: dict) -> dict:
    """Commit one record once, publish its recovery response, and acknowledge it."""
    from nro.orchestration.scheduler_bus import (
        acknowledge_message,
        clear_progress,
        message_path,
        publish_response,
        read_response,
    )

    response = read_response(registry.paths.control, record["id"])
    if response is None:
        response = _message_response(registry, record, values=values)
        publish_response(registry.paths.control, record["id"], response)
    acknowledge_message(message_path(registry.paths.control, record["id"]))
    clear_progress(registry.paths.control, record["id"])
    return response


def _receive_direct(
    listener: socket.socket,
    registry,
    *,
    values: dict,
    token: str,
    generation: int,
) -> tuple[bool, bool]:
    """Handle one waiting TCP request and report whether state may have changed."""
    from nro.orchestration.scheduler_rpc import receive, send, validate_request

    try:
        connection, _peer = listener.accept()
    except BlockingIOError:
        return False, False
    with connection:
        connection.settimeout(60.0)
        try:
            envelope = receive(connection)
            record, durable = validate_request(envelope, token=token)
            response = (
                _process_record(registry, record, values=values)
                if durable
                else _message_response(registry, record, values=values)
            )
            changed = not _quiet_message(record)
        except BaseException as error:
            response = {"error": str(error)}
            changed = False
        try:
            send(connection, response)
        except (ConnectionError, OSError):
            pass
    return True, changed


def _branch_reports(registry) -> dict[str, dict]:
    """Build one cached report for every active branch with an attached checkout."""
    reports = {}
    topology = BranchStore(registry.paths.control).read().topology
    for name, record in topology.records.items():
        if record.retired or not record.checkouts:
            continue
        reports[name] = status(registry, checkout=record.checkouts[0], mode="cached")
    return reports


def publish_status_snapshot(registry, *, generation: int, active: bool) -> None:
    """Publish the complete cached read model without exposing SQLite to readers."""
    from nro.engine.io import atomic_write_json
    from nro.orchestration.control_paths import ControlPaths
    from nro.orchestration.registry import utcnow
    from nro.orchestration.scheduler_bus import PROTOCOL

    with registry.connection() as db:
        workers = [
            dict(row)
            for row in db.execute(
                "SELECT id,state,resource_class,memory_gb,slurm_job_id,updated_at FROM workers"
            )
        ]
        submissions = [
            dict(row)
            for row in db.execute(
                "SELECT id,state,slurm_job_id,memory_gb FROM scheduler_submissions "
                "WHERE state IN ('prepared','submitted','running','cancel_requested')"
            )
        ]
    atomic_write_json(
        ControlPaths(registry.paths.control).service_snapshot,
        {
            "protocol": PROTOCOL,
            "generation": generation,
            "published_at": utcnow(),
            "service_active": active,
            "workers": workers,
            "submissions": submissions,
            "branches": _branch_reports(registry),
        },
        sort_keys=True,
        mode=0o664,
        durable=True,
    )


def _registry_busy(registry) -> bool:
    """Return whether active execution requires the service to remain available."""
    with registry.connection() as db:
        queries = (
            "SELECT 1 FROM attempts WHERE state IN ('queued','running','cancel_requested') LIMIT 1",
            "SELECT 1 FROM workers WHERE state IN ('idle','running','draining','shutdown_requested') LIMIT 1",
            "SELECT 1 FROM scheduler_submissions WHERE state IN ('prepared','submitted','running','cancel_requested') LIMIT 1",
        )
        return any(db.execute(query).fetchone() for query in queries)


def serve(
    *,
    launch_token: str,
    bids_root: Path,
    idle_grace: float = DEFAULT_IDLE_GRACE_SECONDS,
) -> int:
    """Own scheduler access until all durable work has remained quiescent."""
    from nro.configuration.site import settings
    from nro.orchestration.registry import Registry
    from nro.orchestration.scheduler_bus import (
        acknowledge_message,
        activate,
        collect_transport_garbage,
        consume_message,
        deactivate,
        pending_messages,
        publish_active,
        publish_response,
        publish_startup_error,
    )
    from nro.orchestration.scheduler_implementation import require_worker_source

    global _STOP
    _STOP = False
    values = settings()[0]
    control = Path(values["registry"])
    require_worker_source(control)
    registry = Registry.for_project("", bids_root=bids_root, registry_path=control)
    from nro.orchestration.scheduler_rpc import open_listener

    listener = open_listener()
    host = socket.getfqdn()
    port = int(listener.getsockname()[1])
    try:
        registry.initialize()
        with registry.connection(write=True) as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key='scheduler_generation'"
            ).fetchone()
            generation = int(row[0]) + 1 if row else 1
            db.execute(
                "INSERT INTO metadata(key,value) VALUES ('scheduler_generation',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(generation),),
            )
        active = activate(control, launch_token, generation, host=host, port=port)
    except BaseException as error:
        publish_startup_error(control, launch_token, f"{type(error).__name__}: {error}")
        raise
    os.environ["NRO_SCHEDULER_TOKEN"] = launch_token
    os.environ["NRO_SCHEDULER_GENERATION"] = str(generation)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    heartbeat_state = {"generation": generation, "job_id": active.get("job_id")}
    heartbeat_stop = threading.Event()

    def renew_lease() -> None:
        while not heartbeat_stop.wait(HEARTBEAT_SECONDS):
            try:
                publish_active(
                    control,
                    token=launch_token,
                    generation=int(heartbeat_state["generation"]),
                    job_id=heartbeat_state["job_id"],
                    host=host,
                    port=port,
                )
            except OSError as error:
                print(f"Scheduler heartbeat failed: {error}", flush=True)

    heartbeat_thread = threading.Thread(target=renew_lease, name="scheduler-heartbeat")
    heartbeat_thread.start()
    idle_since = None
    last_cleanup = 0.0
    try:
        publish_status_snapshot(registry, generation=generation, active=True)
        while not _STOP:
            now = time.monotonic()
            if now - last_cleanup >= 3600.0:
                collect_transport_garbage(control)
                last_cleanup = now
            changed = False
            handled_direct = False
            while True:
                handled, direct_changed = _receive_direct(
                    listener,
                    registry,
                    values=values,
                    token=launch_token,
                    generation=generation,
                )
                if not handled:
                    break
                handled_direct = True
                changed = changed or direct_changed
            batch = pending_messages(control)
            for path in batch:
                try:
                    record = consume_message(path)
                    _process_record(registry, record, values=values)
                    changed = changed or not _quiet_message(record)
                except BaseException as error:
                    print(f"Scheduler deferred {path}: {type(error).__name__}: {error}", flush=True)
                    try:
                        publish_response(control, path.stem, {"error": str(error)})
                        acknowledge_message(path)
                    except BaseException:
                        # A transient filesystem error leaves the message for retry.
                        pass
            if changed:
                generation += 1
                with registry.connection(write=True) as db:
                    db.execute(
                        "UPDATE metadata SET value=? WHERE key='scheduler_generation'",
                        (str(generation),),
                    )
                heartbeat_state["generation"] = generation
                publish_status_snapshot(registry, generation=generation, active=True)
            if batch or handled_direct or _registry_busy(registry):
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= idle_grace:
                break
            time.sleep(0.02 if handled_direct else 0.1)
        publish_status_snapshot(registry, generation=generation, active=False)
        return 0
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join()
        listener.close()
        deactivate(control, launch_token)


def run_once(*, launch_token: str, bids_root: Path) -> int:
    """Process pending requests under a launch claim without starting a service."""
    from nro.configuration.site import settings
    from nro.orchestration.registry import Registry
    from nro.orchestration.scheduler_bus import (
        consume_message,
        pending_messages,
        publish_startup_error,
        read_launch,
        release_launch,
    )
    from nro.orchestration.scheduler_implementation import require_worker_source

    values = settings()[0]
    control = Path(values["registry"])
    claim = read_launch(control)
    if not claim or claim.get("token") != launch_token:
        raise RuntimeError("One-shot coordinator launch token is obsolete")
    require_worker_source(control)
    registry = Registry.for_project("", bids_root=bids_root, registry_path=control)
    try:
        registry.initialize()
        with registry.connection(write=True) as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key='scheduler_generation'"
            ).fetchone()
            generation = int(row[0]) + 1 if row else 1
            db.execute(
                "INSERT INTO metadata(key,value) VALUES ('scheduler_generation',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(generation),),
            )
        os.environ["NRO_SCHEDULER_TOKEN"] = launch_token
        os.environ["NRO_SCHEDULER_GENERATION"] = str(generation)
        while True:
            batch = pending_messages(control, minimum_age=0.0)
            if not batch:
                break
            for path in batch:
                record = consume_message(path)
                _process_record(registry, record, values=values)
        publish_status_snapshot(registry, generation=generation, active=False)
        return 0
    except BaseException as error:
        publish_startup_error(control, launch_token, f"{type(error).__name__}: {error}")
        raise
    finally:
        release_launch(control, launch_token)


def build_parser() -> argparse.ArgumentParser:
    """Build the internal controller parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--launch-token")
    parser.add_argument("--bids-root", type=Path)
    parser.add_argument("--idle-grace", type=float, default=DEFAULT_IDLE_GRACE_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the controller, or reject the removed one-shot transport."""
    args = build_parser().parse_args(argv)
    if args.serve == args.once or not args.launch_token or args.bids_root is None:
        raise SystemExit("scheduler_service must be started through the scheduler bus")
    if args.once:
        raise SystemExit(run_once(launch_token=args.launch_token, bids_root=args.bids_root))
    raise SystemExit(
        serve(
            launch_token=args.launch_token,
            bids_root=args.bids_root,
            idle_grace=args.idle_grace,
        )
    )


if __name__ == "__main__":
    main()
