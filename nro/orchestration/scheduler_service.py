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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from pathlib import Path

from nro.configuration.site import protected_site_fingerprint
from nro.configuration.store import fingerprint
from nro.orchestration.artifact_resolution import scientific_contracts
from nro.orchestration.branch_admission import _admit_resolved
from nro.orchestration.branch_reconciliation import candidates_locked, resolve_payload
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.compiled_request import decode_spec
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.resources import SCHEDULABLE_RESOURCE_CLASSES
from nro.orchestration.scheduler_bus import DEFAULT_IDLE_GRACE_SECONDS, HEARTBEAT_SECONDS
from nro.orchestration.scheduler_maintenance import refresh_scheduler_state
from nro.orchestration.scheduler_operations import (
    installation_activity,
    installation_prepare,
    installation_progress,
    logs,
    pool_operation,
    require_environment_idle,
    status,
    stop,
    supply,
    supply_needed,
)
from nro.orchestration.source_snapshots import SourceSnapshot

_STOP = False
MAINTENANCE_INTERVAL_SECONDS = 30.0


class _ActivitySignal:
    """Count cross-thread activity without losing a concurrent notification."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    def set(self) -> None:
        """Record one or more events for the service loop."""
        with self._lock:
            self._count += 1

    def consume(self) -> bool:
        """Atomically report and clear activity observed so far."""
        with self._lock:
            active = bool(self._count)
            self._count = 0
            return active


def _request_stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


@contextmanager
def _installation_access(checkout: Path, payloads: tuple[dict, ...]):
    """Hold one shared maintenance guard for an admission batch."""
    record_path = checkout / ".nro-installation.json"
    if not record_path.exists():
        yield
        return
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
        expected_python = str(Path(record["environment"]) / "bin/python")
        if any(payload.get("python") != expected_python for payload in payloads):
            raise ValueError("Request interpreter does not match the installed environment")
        yield


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
    with _installation_access(checkout, (payload,)):
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

    assess_registry(registry, projects=tuple(dict.fromkeys(projects)), compiled=False)
    from nro.orchestration.registry import Registry

    branches = BranchStore(registry.paths.control)
    expected_site = protected_site_fingerprint(Path(site_values["definitions"]))
    payloads = tuple(entry["payload"] for entry in entries)
    results = []
    with _installation_access(checkout, payloads), branches._lock():
        topology = branches.read().topology
        with registry.connection(write=True) as db:
            candidates = {
                project: candidates_locked(db, project) for project in dict.fromkeys(projects)
            }
            for index, entry in enumerate(entries):
                project = entry["project"]
                project_registry = Registry.for_project(
                    project,
                    bids_root=registry.paths.bids_root,
                    registry_path=registry.paths.control,
                )
                results.append(
                    _admit(
                        project_registry,
                        entry["payload"],
                        checkout=checkout,
                        site_values=site_values,
                        assess=False,
                        source_verified=True,
                        request_id=f"{message_id}-{index}",
                        expected_site=expected_site,
                        locked=(topology, db, candidates[project]),
                    )
                )
    return results


def _admit(
    registry,
    payload: dict,
    *,
    checkout: Path,
    site_values: dict,
    assess: bool = True,
    source_verified: bool = False,
    request_id: str | None = None,
    expected_site: str | None = None,
    locked: tuple[object, object, tuple] | None = None,
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
    expected_site = expected_site or protected_site_fingerprint(Path(site_values["definitions"]))
    if payload.get("site_fingerprint") != expected_site:
        raise ValueError(
            "Request site definitions differ from the central protected site; "
            "refresh the branch definitions before retrying"
        )
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

        assess_registry(registry, projects=(context.project,), compiled=False)
    if payload["branch"] == "main":
        from nro.orchestration.scheduler_implementation import implementation_path

        active = json.loads(implementation_path(registry.paths.control).read_text())
        if (
            Path(str(active.get("checkout", ""))).resolve() != checkout.resolve()
            or active.get("release") != payload["release"]
            or active.get("source_digest") != source.digest
        ):
            raise ValueError("Main request does not match its approved release")

    def publish(topology, db, candidates) -> str:
        name = topology.registered_checkout(checkout)
        if (
            name != payload["branch"]
            or topology.records[name].registry_id != payload["registry_id"]
        ):
            raise ValueError("Request ownership does not match the submitting checkout")
        existing_revisions = {
            str(row["logical_key"]): (int(row["revision"]), str(row["fingerprint"]))
            for row in db.execute(
                """SELECT logical_key,revision,fingerprint FROM compiled_revisions
                   WHERE registry_id=?""",
                (payload["registry_id"],),
            )
        }
        changed_revisions = []
        for key, contract in contracts.items():
            revision = payload["revisions"][key]
            if type(revision) is not int or revision < 1:
                raise ValueError("Scientific revisions must be positive integers")
            digest = fingerprint(contract)
            current = existing_revisions.get(key)
            if current and (
                revision < current[0] or (revision == current[0] and digest != current[1])
            ):
                raise ValueError(
                    "A newer scientific request was admitted; refresh this checkout before retrying"
                )
            if current != (revision, digest):
                changed_revisions.append((payload["registry_id"], key, revision, digest))
        db.executemany(
            """INSERT INTO compiled_revisions VALUES (?,?,?,?)
            ON CONFLICT(registry_id,logical_key) DO UPDATE SET revision=excluded.revision,
            fingerprint=excluded.fingerprint""",
            changed_revisions,
        )
        plan = resolve_payload(topology, payload, candidates)
        return _admit_resolved(registry, db, plan, payload, request_id=request_id)

    if locked is not None:
        return publish(*locked)
    with branches._lock():
        topology = branches.read().topology
        with registry.connection(write=True) as db:
            return publish(topology, db, candidates_locked(db, context.project))


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


def _worker_script(registry, *, resource_class: str, memory_gb: int, profile: str | None) -> Path:
    suffix = f"-{profile}" if profile else ""
    exact = registry.paths.workers / f"worker-{resource_class}-{memory_gb}gb{suffix}.sbatch"
    if exact.is_file():
        return exact
    candidates = sorted(
        registry.paths.workers.glob(f"worker-{resource_class}-{memory_gb}gb-*.sbatch")
    )
    if candidates:
        return candidates[-1]
    raise ValueError(f"No prepared {resource_class} {memory_gb} GB worker script is available")


def _worker_script_tiers(
    registry, *, resource_class: str, minimum_memory_gb: int, profile: str | None
) -> tuple[tuple[int, Path], ...]:
    """Return one request profile's eligible scripts in ascending memory order."""
    prefix = f"worker-{resource_class}-"
    suffix = f"gb-{profile}.sbatch" if profile else "gb.sbatch"
    scripts = []
    for candidate in registry.paths.workers.glob(f"{prefix}*{suffix}"):
        value = candidate.name.removeprefix(prefix).removesuffix(suffix)
        if value.isdigit() and int(value) >= minimum_memory_gb:
            scripts.append((int(value), candidate))
    if scripts:
        return tuple(sorted(scripts))
    return (
        (
            minimum_memory_gb,
            _worker_script(
                registry,
                resource_class=resource_class,
                memory_gb=minimum_memory_gb,
                profile=profile,
            ),
        ),
    )


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
    if action == "claim":
        envelope = registry.current_worker_assignment(worker_id)
        if envelope is None:
            envelope = registry.claim_ready_work_item(
                worker_id,
                tuple(message["resource_classes"]),
                memory_gb=int(message["memory_gb"]),
            )
        return None if envelope is None else envelope.as_dict()
    if action == "claim_resource_step":
        envelope = registry.current_worker_assignment(worker_id)
        if envelope is None:
            envelope = registry.claim_resource_step(
                worker_id,
                resource_class=str(message["resource_class"]),
                memory_gb=int(message["memory_gb"]),
            )
        return None if envelope is None else envelope.as_dict()
    if action == "defer_resource_step":
        return registry.defer_resource_step(
            int(message["attempt_id"]),
            step_id=str(message["step_id"]),
            resource_class=str(message["resource_class"]),
            memory_gb=int(message["memory_gb"]),
        )
    if action == "finish_resource_step":
        registry.finish_resource_step(
            int(message["task_id"]),
            int(message["attempt_id"]),
            state=str(message["state"]),
            error_type=message.get("error_type"),
            error_message=message.get("error_message"),
        )
        return None
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
        return registry.cancel_attempts_downstream_of_failure(int(message["work_item_id"]))
    if action == "runner_graph_signature":
        from nro.orchestration.worker import _runner_graph_signature

        return _runner_graph_signature(registry, int(message["work_item_id"]))
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
        from nro.orchestration.completion import record_completion

        return record_completion(
            registry,
            work_item_id=int(message["work_item_id"]),
            attempt_id=int(message["attempt_id"]),
            outputs=tuple(Path(path) for path in message["outputs"]),
        )
    if action == "required_memory":
        return registry.required_memory_above(
            int(message["memory_gb"]),
            resource_classes=tuple(message.get("resource_classes", ())),
        )
    if action == "request_capacity":
        memory = int(message["memory_gb"])
        kind = message["kind"]
        resource_class = str(message["resource_class"])
        if kind == "successor":
            reservation = registry.reserve_worker_successor(
                worker_id=worker_id, resource_class=resource_class, memory_gb=memory
            )
            if reservation is None:
                return None
            script = _worker_script(
                registry,
                resource_class=resource_class,
                memory_gb=memory,
                profile=message.get("profile"),
            )
            with registry.connection() as db:
                row = db.execute(
                    "SELECT slurm_job_id FROM workers WHERE id=?", (worker_id,)
                ).fetchone()
            return _submit_reserved(
                registry, reservation[0], script, dependency=row[0] if row else None
            )
        if kind == "adaptive":
            reservation = registry.reserve_adaptive_worker(
                resource_class=resource_class, memory_gb=memory
            )
            if reservation is None:
                return None
            script = _worker_script(
                registry,
                resource_class=resource_class,
                memory_gb=memory,
                profile=message.get("profile"),
            )
            return _submit_reserved(registry, reservation[0], script)
        if kind == "expand":
            submitted = []
            for candidate_class in SCHEDULABLE_RESOURCE_CLASSES:
                lower_memory = 0
                for candidate_memory, script in _worker_script_tiers(
                    registry,
                    resource_class=candidate_class,
                    minimum_memory_gb=1,
                    profile=message.get("profile"),
                ):
                    reservations = registry.reserve_worker_submissions(
                        request_id=None,
                        resource_class=candidate_class,
                        memory_gb=candidate_memory,
                        minimum_memory_gb=lower_memory,
                    )
                    lower_memory = candidate_memory
                    if not reservations:
                        continue
                    submitted.extend(
                        _submit_reserved(registry, submission_id, script)
                        for submission_id, _token in reservations
                    )
                    break
            return submitted
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
            worker_level=message["worker_level"],
            running_only=bool(message.get("running_only", False)),
        )
    elif message["operation"] in {
        "concurrency",
        "gpu_concurrency",
        "settings",
        "stop_workers",
    }:
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
    elif message["operation"] == "gc":
        from nro.orchestration.garbage_collection import collect
        from nro.orchestration.scheduler_bus import publish_progress

        result = collect(
            registry,
            checkout=Path(message["checkout"]),
            site_values=values,
            selection=message["selection"],
            dry_run=message["dry_run"],
            approved=message["approved"],
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
            registry,
            checkout=Path(message["checkout"]),
            reservation=message["reservation"],
            workflows=message["workflows"],
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
    """Execute one validated scheduler request and encode its result."""
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
        response = {"error": str(error), "error_type": type(error).__name__}
    return response


def _quiet_message(record: dict) -> bool:
    payload = record["payload"]
    return payload.get("operation") == "worker" and payload.get("action") in {
        "heartbeat",
        "shutdown_requested",
        "attempt_cancel_requested",
        "required_memory",
    }


_MAINTENANCE_OPERATIONS = {
    "cache",
    "gc",
    "promotion_publish",
    "publish",
    "purge",
    "repair_finish",
    "repair_prepare",
    "status",
}


def _executor_for(
    record: dict,
    executors: dict[str, ThreadPoolExecutor],
    *,
    durable: bool = True,
) -> ThreadPoolExecutor:
    """Route worker traffic away from long user maintenance operations."""
    operation = record["payload"].get("operation")
    if operation == "worker":
        return executors["worker" if durable else "poll"]
    if operation in _MAINTENANCE_OPERATIONS:
        return executors["maintenance"]
    return executors["command"]


def _handle_connection(
    connection: socket.socket,
    registry,
    *,
    values: dict,
    token: str,
    coordinator,
    executors: dict[str, ThreadPoolExecutor],
    changed: _ActivitySignal,
) -> None:
    """Validate one connection and hand its work to the appropriate pool."""
    from nro.orchestration.scheduler_rpc import receive, send, validate_request

    with connection:
        connection.settimeout(60.0)
        try:
            envelope = receive(connection)
            record, durable = validate_request(envelope, token=token)
            executor = _executor_for(record, executors, durable=durable)
            if durable:
                future = coordinator.submit(record, executor)
                if not _quiet_message(record):
                    future.add_done_callback(lambda _future: changed.set())
                try:
                    response = future.result(timeout=0.05)
                except FutureTimeout:
                    response = {"pending": record["id"]}
            else:
                response = executor.submit(
                    _message_response, registry, record, values=values
                ).result()
            if "pending" not in response and not _quiet_message(record):
                changed.set()
        except BaseException as error:
            response = {"error": str(error), "error_type": type(error).__name__}
        try:
            send(connection, response)
        except (ConnectionError, OSError):
            pass


def _accept_connections(
    listener: socket.socket,
    readers: ThreadPoolExecutor,
    registry,
    **options,
) -> int:
    """Drain ready connections without executing their requests inline."""
    accepted = 0
    while True:
        try:
            connection, _peer = listener.accept()
        except BlockingIOError:
            return accepted
        readers.submit(_handle_connection, connection, registry, **options)
        accepted += 1


def _listen(
    listener: socket.socket,
    readers: ThreadPoolExecutor,
    registry,
    *,
    stop_event: threading.Event,
    activity_event: _ActivitySignal,
    **options,
) -> None:
    """Accept connections independently of maintenance and snapshot work."""
    while not stop_event.is_set():
        accepted = _accept_connections(listener, readers, registry, **options)
        if accepted:
            activity_event.set()
        stop_event.wait(0.02 if accepted else 0.1)


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
            "SELECT 1 FROM scheduler_requests WHERE state IN ('pending','running') LIMIT 1",
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
        activate,
        collect_transport_garbage,
        deactivate,
        publish_active,
        publish_startup_error,
    )
    from nro.orchestration.scheduler_implementation import require_worker_source
    from nro.orchestration.scheduler_requests import RequestCoordinator, prepare

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
    executors = {
        "poll": ThreadPoolExecutor(max_workers=4, thread_name_prefix="scheduler-poll"),
        "worker": ThreadPoolExecutor(max_workers=4, thread_name_prefix="scheduler-worker"),
        "command": ThreadPoolExecutor(max_workers=4, thread_name_prefix="scheduler-command"),
        "maintenance": ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="scheduler-maintenance"
        ),
    }
    readers = ThreadPoolExecutor(max_workers=16, thread_name_prefix="scheduler-rpc")
    changed_event = _ActivitySignal()
    request_activity = _ActivitySignal()
    listener_stop = threading.Event()
    coordinator = RequestCoordinator(
        registry,
        lambda record: _message_response(registry, record, values=values),
    )

    def note_completion(record: dict, future) -> None:
        try:
            future.result()
        except BaseException:
            pass
        if not _quiet_message(record):
            changed_event.set()

    for record in prepare(registry):
        future = coordinator.submit(record, _executor_for(record, executors))
        future.add_done_callback(lambda completed, item=record: note_completion(item, completed))
    listener_thread = threading.Thread(
        target=_listen,
        name="scheduler-listener",
        args=(listener, readers, registry),
        kwargs={
            "stop_event": listener_stop,
            "activity_event": request_activity,
            "values": values,
            "token": launch_token,
            "coordinator": coordinator,
            "executors": executors,
            "changed": changed_event,
        },
    )
    listener_thread.start()
    idle_since = None
    last_cleanup = 0.0
    last_maintenance = 0.0
    try:
        cancelled = refresh_scheduler_state(registry)
        if cancelled:
            print(
                f"Scheduler cancelled {cancelled} attempt(s) with stale upstream artifacts",
                flush=True,
            )
        last_maintenance = time.monotonic()
        publish_status_snapshot(registry, generation=generation, active=True)
        while not _STOP:
            now = time.monotonic()
            changed = False
            if now - last_cleanup >= 3600.0:
                collect_transport_garbage(control)
                last_cleanup = now
            if now - last_maintenance >= MAINTENANCE_INTERVAL_SECONDS and _registry_busy(registry):
                cancelled = refresh_scheduler_state(registry)
                if cancelled:
                    print(
                        f"Scheduler cancelled {cancelled} attempt(s) with stale upstream artifacts",
                        flush=True,
                    )
                last_maintenance = time.monotonic()
                changed = True
            handled_direct = request_activity.consume()
            if changed_event.consume():
                changed = True
            if changed:
                generation += 1
                with registry.connection(write=True) as db:
                    db.execute(
                        "UPDATE metadata SET value=? WHERE key='scheduler_generation'",
                        (str(generation),),
                    )
                heartbeat_state["generation"] = generation
                publish_status_snapshot(registry, generation=generation, active=True)
            if handled_direct or _registry_busy(registry):
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
        listener_stop.set()
        listener_thread.join()
        listener.close()
        readers.shutdown(wait=True, cancel_futures=False)
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
        deactivate(control, launch_token)


def run_once(*, launch_token: str, bids_root: Path) -> int:
    """Process one retryable stdin request under a fenced launch claim."""
    from nro.configuration.site import settings
    from nro.orchestration.registry import Registry
    from nro.orchestration.scheduler_bus import publish_startup_error, read_launch, release_launch
    from nro.orchestration.scheduler_implementation import require_worker_source
    from nro.orchestration.scheduler_requests import RequestCoordinator, prepare

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
        envelope = json.loads(sys.stdin.read())
        from nro.orchestration.scheduler_bus import validate_message

        if not isinstance(envelope, dict) or set(envelope) != {"durable", "record"}:
            raise ValueError("Invalid one-shot scheduler request")
        durable = envelope["durable"]
        if not isinstance(durable, bool):
            raise ValueError("Invalid one-shot scheduler durability flag")
        record = validate_message(envelope["record"])
        coordinator = RequestCoordinator(
            registry,
            lambda item: _message_response(registry, item, values=values),
        )
        for pending in prepare(registry):
            if pending["id"] != record["id"]:
                coordinator.run(pending)
        response = (
            coordinator.run(record)
            if durable
            else _message_response(registry, record, values=values)
        )
        publish_status_snapshot(registry, generation=generation, active=False)
        print(json.dumps(response, separators=(",", ":"), sort_keys=True), flush=True)
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
