"""Reusable foreground worker for the nro derivative registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

import yaml

from nro.configuration.paths import BIDS_PATH
from nro.engine.io import atomic_write_json
from nro.orchestration.assessment import AssessmentConflict
from nro.orchestration.contracts import ExecutionEnvelope
from nro.orchestration.dependency_state import AttemptInvalidated
from nro.orchestration.execution import (
    ExecutionLauncher,
    ShutdownUnconfirmed,
    SubprocessExecutionLauncher,
)
from nro.orchestration.execution_cache import cleanup_cache
from nro.orchestration.manifests import assess_registry, record_completion
from nro.orchestration.registry import Registry, ensure_shared_directory, utcnow
from nro.orchestration.scheduler_implementation import validate_worker_script

COMPATIBLE = {
    "large": ("large", "medium", "small"),
    "medium": ("medium", "small"),
    "small": ("small",),
}


def _append_event(registry: Registry, instance: ExecutionEnvelope, event: dict) -> None:
    path = instance.log_path.parent / "events.jsonl"
    ensure_shared_directory(path.parent)
    record = {
        "timestamp": utcnow(),
        "instance_id": instance.instance_id,
        **event,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        path.chmod(0o664)
    except PermissionError:
        pass


def _update_orchestration_step(
    log_path: Path,
    *,
    status: str,
    started_at: float,
    manifest_path: str,
    error: str | None = None,
) -> None:
    """Track worker-owned completion/provenance work in the shared step ledger."""
    path = log_path.parent / "current-steps.json"
    ensure_shared_directory(path.parent)
    key = "orchestration:completion-manifest"
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(current, dict):
            current = {}
    except (OSError, json.JSONDecodeError):
        current = {}
    record = {
        "step_id": key,
        "name": "Completion Manifest and Provenance",
        "status": status,
        "timestamp": utcnow(),
        "outputs": [manifest_path],
        "command": None,
        "cwd": os.getcwd(),
        "reason": "Validate and fingerprint instance outputs after module completion.",
        "elapsed_seconds": time.monotonic() - started_at,
        "error": error,
    }
    current[key] = record
    atomic_write_json(path, current, sort_keys=True, mode=0o664)
    events = path.with_name("step-events.jsonl")
    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        events.chmod(0o664)
    except PermissionError:
        pass


def _outputs(instance: ExecutionEnvelope) -> tuple[Path, ...]:
    """Resolve public instance outputs from the immutable module DAG ledger.

    Workers must never discover derivatives by walking an output directory:
    stale or unrelated files would then become part of the instance contract.
    Variable-cardinality directory nodes expose one fixed manifest, whose
    public inventory is expanded here for instance-level missing-file detection.
    """
    root = instance.output_root
    declared = instance.expected_outputs
    if not declared:
        raise RuntimeError(f"Instance has an empty fixed output contract: {instance.instance_id}")
    resolved_root = root.resolve()
    values: set[Path] = set()
    for candidate in declared:
        resolved = candidate.expanduser().resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(
                f"Instance output contract escapes its derivative root: {resolved}"
            ) from error
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise RuntimeError(f"Instance did not produce nonempty declared output: {resolved}")
        values.add(resolved)

    missing_references: set[Path] = set()

    def referenced(value, *, base: Path) -> None:
        if isinstance(value, dict):
            for item in value.values():
                referenced(item, base=base)
        elif isinstance(value, (list, tuple)):
            for item in value:
                referenced(item, base=base)
        elif isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved = candidate.resolve()
            try:
                resolved.relative_to(resolved_root)
            except ValueError:
                return
            if resolved.is_file() and resolved.stat().st_size > 0:
                values.add(resolved)
            else:
                missing_references.add(resolved)

    pending = [
        path
        for path in values
        if path.name.endswith(("_manifest.json", "_manifest.yaml", "_manifest.yml"))
    ]
    visited: set[Path] = set()
    while pending:
        manifest_path = pending.pop()
        if manifest_path in visited:
            continue
        visited.add(manifest_path)
        try:
            if manifest_path.suffix == ".json":
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, yaml.YAMLError, TypeError) as error:
            raise RuntimeError(
                f"Instance publication manifest is unreadable: {manifest_path}: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Instance publication manifest is not a mapping: {manifest_path}")
        before = set(values)
        inventory = manifest.get("public_outputs")
        if inventory is None:
            inventory = manifest.get("outputs", {})
        referenced(inventory, base=manifest_path.parent)
        pending.extend(
            path
            for path in values - before
            if path.name.endswith(("_manifest.json", "_manifest.yaml", "_manifest.yml"))
        )
    if missing_references:
        raise RuntimeError(
            "Instance publication inventory contains missing or empty output(s): "
            + ", ".join(str(path) for path in sorted(missing_references))
        )
    return tuple(sorted(values))


def _failure_summary(log_path: Path, return_code: int) -> str:
    try:
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            tail = deque(stream, maxlen=500)
    except OSError:
        tail = ()
    failed_step = None
    error = None
    for line in tail:
        if "Failed Step:" in line:
            failed_step = line.split("Failed Step:", 1)[1].strip()
        if "Error:" in line:
            error = line.split("Error:", 1)[1].strip()
        elif "FATAL:" in line:
            error = line.split("FATAL:", 1)[1].strip()
    details = []
    if failed_step:
        details.append(f"failed step {failed_step}")
    if error:
        details.append(error)
    suffix = "; ".join(details) if details else f"see {log_path}"
    return f"Derivative command exited with status {return_code}: {suffix}"


def _runner_graph_signature(registry: Registry, instance_id: int) -> str:
    """Fingerprint the artifact topology and transitive BIDS state.

    Source mtimes and sizes distinguish a genuinely new BIDS data state at an
    existing pathname. Commands and source code are execution details rather
    than evidence that the derivative contract changed.
    """
    with registry.connection() as db:
        closure = [
            dict(row)
            for row in db.execute(
                """
                WITH RECURSIVE ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT dependency.upstream_instance_id
                    FROM instance_dependencies dependency
                    JOIN ancestors ON dependency.instance_id=ancestors.id
                )
                SELECT instance.id, instance.instance_key,
                       instance.artifact_fingerprint, instance.input_paths_json
                FROM instances instance JOIN ancestors ON ancestors.id=instance.id
                ORDER BY instance.instance_key
                """,
                (instance_id,),
            )
        ]
        closure_ids = {int(row["id"]) for row in closure}
        keys_by_id = {int(row["id"]): str(row["instance_key"]) for row in closure}
        placeholders = ",".join("?" for _ in closure_ids)
        edges = sorted(
            (keys_by_id[int(row["instance_id"])], keys_by_id[int(row["upstream_instance_id"])])
            for row in db.execute(
                f"""SELECT instance_id, upstream_instance_id FROM instance_dependencies
                    WHERE instance_id IN ({placeholders})
                      AND upstream_instance_id IN ({placeholders})""",
                (*sorted(closure_ids), *sorted(closure_ids)),
            )
        )
    root = next(row for row in closure if int(row["id"]) == instance_id)
    source_paths = sorted(
        {
            str(Path(value).expanduser().resolve())
            for row in closure
            for value in json.loads(str(row["input_paths_json"]))
        }
    )
    source_state = []
    for value in source_paths:
        path = Path(value)
        try:
            stat = path.stat()
            source_state.append(
                {"path": value, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
            )
        except OSError:
            source_state.append({"path": value, "missing": True})
    payload = {
        "artifact_fingerprint": root["artifact_fingerprint"],
        "instances": [
            {
                "instance_key": str(row["instance_key"]),
                "artifact_fingerprint": str(row["artifact_fingerprint"]),
            }
            for row in closure
        ],
        "edges": edges,
        "source_state": source_state,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _looks_like_oom(log_path: Path, return_code: int) -> bool:
    if return_code in {-9, 137}:
        return True
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")[-200_000:].lower()
    except OSError:
        return False
    return any(
        marker in text for marker in ("out of memory", "out_of_memory", "oom_kill", "oom-kill")
    )


class Worker:
    """Claim derivative instances or ingestion stages and supervise their execution."""

    def __init__(
        self,
        registry: Registry,
        *,
        resource_class: str,
        memory_gb: int = 32,
        idle_timeout: float = 30.0,
        poll_interval: float = 5.0,
        walltime_seconds: float | None = None,
        drain_seconds: float = 15 * 60,
        profile: str | None = None,
        launcher: ExecutionLauncher | None = None,
    ) -> None:
        """Configure registry access, resource limits, polling, and the execution launcher."""
        self.registry = registry
        self.resource_class = resource_class
        self.memory_gb = memory_gb
        self.idle_timeout = idle_timeout
        self.poll_interval = poll_interval
        self.deadline = (
            time.monotonic() + walltime_seconds if walltime_seconds is not None else None
        )
        self.drain_seconds = drain_seconds
        self.profile = profile
        self.worker_id = (
            f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.launcher = launcher or SubprocessExecutionLauncher()
        self.stop_requested = False

    def _log(self, message: str) -> None:
        """Write a concise lifecycle record to the Slurm worker stream."""
        print(f"{utcnow()} nro worker {self.worker_id}: {message}", flush=True)

    def _attempt_summary(self, attempt_id: int) -> str:
        with self.registry.connection() as db:
            row = db.execute(
                "SELECT state, error_message FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
        if row is None:
            return "attempt record missing"
        summary = f"attempt state={row['state']}"
        if row["error_message"]:
            summary += f"; error={row['error_message']}"
        return summary

    def _submit_successor(self) -> None:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return
        script = self.registry.paths.workers / f"worker-{self.resource_class}.sbatch"
        tier_name = f"worker-{self.resource_class}-{self.memory_gb}gb"
        if self.profile:
            tier_name += f"-{self.profile}"
        tier_script = self.registry.paths.workers / f"{tier_name}.sbatch"
        if tier_script.is_file():
            script = tier_script
        if not script.is_file():
            return
        reservation = self.registry.reserve_worker_successor(
            worker_id=self.worker_id,
            resource_class=self.resource_class,
            memory_gb=self.memory_gb,
        )
        if reservation is None:
            return
        submission_id, _token = reservation
        try:
            validate_worker_script(self.registry.paths.control, script)
            result = subprocess.run(
                ["sbatch", "--parsable", f"--dependency=afterany:{job_id}", str(script)],
                check=True,
                text=True,
                capture_output=True,
            )
            successor_id = result.stdout.strip().split(";", 1)[0]
            if not successor_id:
                raise RuntimeError(f"sbatch returned no successor job ID: {result.stdout!r}")
            self.registry.update_submission(
                submission_id, state="submitted", slurm_job_id=successor_id
            )
        except BaseException as error:
            self.registry.update_submission(submission_id, state="error")
            print(f"WARNING: could not submit replacement worker: {error}", file=sys.stderr)

    def _submit_adaptive_worker(self, memory_gb: int) -> None:
        if not os.environ.get("SLURM_JOB_ID"):
            return
        profile_suffix = f"-{self.profile}" if self.profile else ""
        script = self.registry.paths.workers / (
            f"worker-{self.resource_class}-{memory_gb}gb{profile_suffix}.sbatch"
        )
        if not script.is_file():
            candidates: list[tuple[int, Path]] = []
            prefix = f"worker-{self.resource_class}-"
            pattern = (
                f"{prefix}*gb-{self.profile}.sbatch" if self.profile else f"{prefix}*gb.sbatch"
            )
            for candidate in self.registry.paths.workers.glob(pattern):
                value = candidate.name.removeprefix(prefix)
                value = value.removesuffix(
                    f"gb-{self.profile}.sbatch" if self.profile else "gb.sbatch"
                )
                if value.isdigit() and int(value) >= memory_gb:
                    candidates.append((int(value), candidate))
            if not candidates:
                print(
                    f"WARNING: no worker script can satisfy a {memory_gb} GB OOM retry",
                    file=sys.stderr,
                )
                return
            memory_gb, script = min(candidates)
        reservation = self.registry.reserve_adaptive_worker(
            resource_class=self.resource_class,
            memory_gb=memory_gb,
        )
        if reservation is None:
            return
        submission_id, _token = reservation
        try:
            validate_worker_script(self.registry.paths.control, script)
            result = subprocess.run(
                ["sbatch", "--parsable", str(script)],
                check=True,
                text=True,
                capture_output=True,
            )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id:
                raise RuntimeError(f"sbatch returned no adaptive worker job ID: {result.stdout!r}")
            self.registry.update_submission(submission_id, state="submitted", slurm_job_id=job_id)
        except BaseException as error:
            self.registry.update_submission(submission_id, state="error")
            print(f"WARNING: could not submit {memory_gb} GB worker: {error}", file=sys.stderr)

    def _expand_ready_pool(self) -> None:
        """Add ordinary workers when a completed dependency exposes parallel work."""
        if not os.environ.get("SLURM_JOB_ID"):
            return
        profile_suffix = f"-{self.profile}" if self.profile else ""
        script = self.registry.paths.workers / (
            f"worker-{self.resource_class}-{self.memory_gb}gb{profile_suffix}.sbatch"
        )
        if not script.is_file():
            print(
                f"WARNING: cannot expand worker pool; script is missing: {script}",
                file=sys.stderr,
            )
            return
        reservations = self.registry.reserve_worker_submissions(
            request_id=None,
            resource_class=self.resource_class,
            memory_gb=self.memory_gb,
        )
        for submission_id, _token in reservations:
            try:
                validate_worker_script(self.registry.paths.control, script)
                result = subprocess.run(
                    ["sbatch", "--parsable", str(script)],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                job_id = result.stdout.strip().split(";", 1)[0]
                if not job_id:
                    raise RuntimeError(
                        f"sbatch returned no expansion worker job ID: {result.stdout!r}"
                    )
                self.registry.update_submission(
                    submission_id, state="submitted", slurm_job_id=job_id
                )
            except BaseException as error:
                self.registry.update_submission(submission_id, state="error")
                print(f"WARNING: could not expand worker pool: {error}", file=sys.stderr)

    def _signal(self, signum, _frame) -> None:
        self.stop_requested = True
        self.launcher.terminate()

    def _execute(self, instance: ExecutionEnvelope) -> None:
        attempt_id = instance.attempt_id
        command = instance.execution.command
        log_path = instance.log_path
        ensure_shared_directory(log_path.parent)
        step_ledger = log_path.parent / "current-steps.json"
        # These files describe one current attempt, just like instance.log.  Do
        # not let obsolete branch-specific nodes from an older attempt leak
        # into the new completion certificate's private-artifact inventory.
        for current_attempt_file in (
            step_ledger,
            log_path.parent / "step-events.jsonl",
            log_path.parent / "runner-graph.json",
        ):
            current_attempt_file.unlink(missing_ok=True)
        cancelled = False
        scheduler_cancelled = False
        completion_started: float | None = None
        started_runner_graph_signature = _runner_graph_signature(
            self.registry, instance.instance_id
        )
        _append_event(
            self.registry,
            instance,
            {
                "event": "attempt_started",
                "attempt_id": attempt_id,
                "worker_id": self.worker_id,
                "command": command,
                "cwd": os.getcwd(),
                "log": str(log_path),
                "inputs": [str(path) for path in instance.input_paths],
                "runtime_config": str(instance.execution.runtime_config),
                "revision_fingerprint": instance.revision_fingerprint,
                "runner_graph_signature": started_runner_graph_signature,
            },
        )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    log_path.chmod(0o664)
                except PermissionError:
                    pass
                log.write(f"{utcnow()} nro worker {self.worker_id}\n")
                log.write(f"Command: {json.dumps(command)}\n")
                log.flush()
                if not str(instance.execution.runtime_config).strip():
                    log.write(
                        "FATAL: Instance has no registry-pinned runtime configuration; "
                        "re-plan it with python -m nro.bin.run\n"
                    )
                    log.flush()
                    raise RuntimeError(
                        "Instance has no registry-pinned runtime configuration; "
                        "re-plan it with python -m nro.bin.run"
                    )

                def cancellation_state() -> tuple[bool, bool]:
                    scheduler_cancelled = self.registry.attempt_cancel_requested(attempt_id)
                    return self.stop_requested or scheduler_cancelled, scheduler_cancelled

                self.registry.record_attempt_process(attempt_id, -1)
                result = self.launcher.run(
                    instance,
                    stdout=log,
                    environment={
                        **os.environ,
                        "NRO_INSTANCE_ID": str(instance.instance_id),
                        "NRO_ATTEMPT_ID": str(attempt_id),
                        "NRO_STEP_LEDGER": str(step_ledger),
                        "NRO_BIDS_PATH": str(self.registry.paths.bids_root),
                        "NRO_WORKER_MEMORY_GB": str(self.memory_gb),
                        "NRO_RUNTIME_CONFIG": str(instance.execution.runtime_config),
                        "NRO_CONFIGURATION_FINGERPRINT": instance.config_fingerprint,
                        "NRO_RUNNER_GRAPH_SIGNATURE": started_runner_graph_signature,
                    },
                    poll_interval=self.poll_interval,
                    cancellation_state=cancellation_state,
                    heartbeat=lambda: self.registry.heartbeat_worker(
                        self.worker_id, state="running"
                    ),
                    process_started=lambda group: self.registry.record_attempt_process(
                        attempt_id, group
                    ),
                )
                cancelled = result.cancelled
                scheduler_cancelled = result.scheduler_cancelled
                return_code = result.return_code
            if cancelled:
                # Registry-issued cancellations already carry their durable
                # cause.  A worker-level signal is infrastructure interruption
                # and is safe to resume under existing demand.
                self.registry.finish_attempt(
                    attempt_id,
                    state="cancelled",
                    error_type=None if scheduler_cancelled else "WorkerTerminated",
                    error_message=None
                    if scheduler_cancelled
                    else "Worker terminated while instance was running",
                )
                _append_event(
                    self.registry,
                    instance,
                    {"event": "attempt_cancelled", "attempt_id": attempt_id},
                )
                return
            if return_code != 0:
                message = _failure_summary(log_path, return_code)
                if _looks_like_oom(log_path, return_code):
                    next_memory = self.registry.record_oom(attempt_id, message=message)
                    self._cancel_failed_descendants(instance)
                    _append_event(
                        self.registry,
                        instance,
                        {
                            "event": "attempt_oom",
                            "attempt_id": attempt_id,
                            "memory_gb": self.memory_gb,
                            "retry_memory_gb": next_memory,
                            "error": message,
                        },
                    )
                    if next_memory is not None:
                        self._submit_adaptive_worker(next_memory)
                    return
                self.registry.finish_attempt(
                    attempt_id,
                    state="error",
                    error_type="CalledProcessError",
                    error_message=message,
                )
                self._cancel_failed_descendants(instance)
                _append_event(
                    self.registry,
                    instance,
                    {
                        "event": "attempt_failed",
                        "attempt_id": attempt_id,
                        "return_code": return_code,
                        "error": message,
                    },
                )
                return
            current_runner_graph_signature = _runner_graph_signature(
                self.registry, instance.instance_id
            )
            if current_runner_graph_signature != started_runner_graph_signature:
                message = (
                    "BIDS/workflow instance DAG changed while the attempt was running; "
                    "discarding this completion and returning the instance to the ready queue"
                )
                self.registry.finish_attempt(
                    attempt_id,
                    state="cancelled",
                    error_type="InstanceGraphChanged",
                    error_message=message,
                )
                _append_event(
                    self.registry,
                    instance,
                    {
                        "event": "attempt_graph_changed",
                        "attempt_id": attempt_id,
                        "error": message,
                    },
                )
                return
            completion_started = time.monotonic()
            _update_orchestration_step(
                log_path,
                status="running",
                started_at=completion_started,
                manifest_path=str(instance.manifest_path),
            )
            manifest = record_completion(
                self.registry,
                instance_id=instance.instance_id,
                attempt_id=attempt_id,
                outputs=_outputs(instance),
            )
            _update_orchestration_step(
                log_path,
                status="success",
                started_at=completion_started,
                manifest_path=str(instance.manifest_path),
            )
            completion_started = None
            self.registry.finish_attempt(attempt_id, state="success")
            _append_event(
                self.registry,
                instance,
                {
                    "event": "attempt_succeeded",
                    "attempt_id": attempt_id,
                    "generation": manifest["generation"],
                    "outputs": [item["path"] for item in manifest["public_outputs"]],
                },
            )
        except ShutdownUnconfirmed as error:
            self.stop_requested = True
            self._log(str(error) + "; retaining the active attempt until shutdown is confirmed")
        except AttemptInvalidated as error:
            self.registry.finish_attempt(
                attempt_id, state="cancelled", error_type="UpstreamStale", error_message=str(error)
            )
            _append_event(
                self.registry,
                instance,
                {"event": "attempt_cancelled", "attempt_id": attempt_id, "error": str(error)},
            )
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            if completion_started is not None:
                try:
                    _update_orchestration_step(
                        log_path,
                        status="error",
                        started_at=completion_started,
                        manifest_path=str(instance.manifest_path),
                        error=message,
                    )
                except BaseException:
                    pass
            self.registry.finish_attempt(
                attempt_id, state="error", error_type=type(error).__name__, error_message=message
            )
            self._cancel_failed_descendants(instance)
            _append_event(
                self.registry,
                instance,
                {
                    "event": "attempt_failed",
                    "attempt_id": attempt_id,
                    "error": message,
                    "traceback": traceback.format_exc(),
                },
            )

    def _execute_ingestion(self, record: dict) -> None:
        """Supervise a noninteractive ingestion stage without derivative completion checks."""
        from types import SimpleNamespace

        from nro.bidsify.store import IngestionStore
        from nro.orchestration.contracts import ExecutionRecipe

        store = IngestionStore(self.registry, branch=record.get("branch", "main"))
        log_path = store.root / f"{record['id']}.log"
        result_path = store.root / f"{record['id']}.result"
        result_path.unlink(missing_ok=True)
        envelope = SimpleNamespace(
            execution=ExecutionRecipe(
                command=(
                    sys.executable,
                    "-m",
                    "nro.bidsify",
                    "--request",
                    record["id"],
                    "--bids-root",
                    str(self.registry.paths.bids_root),
                    "--control",
                    str(self.registry.paths.control),
                ),
                runtime_config=store.root / f"{record['id']}.json",
            )
        )
        self.registry.heartbeat_worker(self.worker_id, state="running")
        try:
            if record.get("execution"):
                from nro.bidsify.execution import stage_command

                envelope.execution = ExecutionRecipe(
                    command=stage_command(record, self.registry),
                    runtime_config=store.root / f"{record['id']}.json",
                )

            def cancelled():
                if self.stop_requested or self.registry.worker_shutdown_requested(self.worker_id):
                    return True
                if store.branch != "main":
                    from nro.orchestration.branch_store import BranchStore

                    owner = (
                        BranchStore(self.registry.paths.control)
                        .read()
                        .topology.records.get(store.branch)
                    )
                    return owner is None or owner.retired
                return False

            with log_path.open("a") as log:
                log_path.chmod(0o660)
                log.write(f"{utcnow()} Stage: {record['stage']}\n")
                log.flush()
                result = self.launcher.run(
                    envelope,
                    stdout=log,
                    environment=dict(os.environ),
                    poll_interval=self.poll_interval,
                    cancellation_state=lambda: (cancelled(), False),
                    heartbeat=lambda: self.registry.heartbeat_worker(
                        self.worker_id, state="running"
                    ),
                )
            if result.cancelled:
                store.finish(record["id"], self.worker_id, state="interrupted")
            elif result.return_code != 0 or not result_path.is_file():
                store.finish(
                    record["id"],
                    self.worker_id,
                    state="failed",
                    changes={
                        "issues": [
                            "Scheduled stage failed; inspect the ingestion log and retry through nro bidsify"
                        ]
                    },
                )
            else:
                changes = json.loads(result_path.read_text())
                store.finish(
                    record["id"], self.worker_id, state=changes.pop("state"), changes=changes
                )
        except Exception:
            store.finish(record["id"], self.worker_id, state="interrupted")
        finally:
            self.registry.heartbeat_worker(self.worker_id, state="idle")
            self._expand_ready_pool()

    def _cancel_failed_descendants(self, instance: ExecutionEnvelope) -> None:
        cancelled = self.registry.cancel_attempts_downstream_of_failure(instance.instance_id)
        if cancelled:
            self._log(
                f"requested cancellation of {len(cancelled)} active downstream attempt(s) "
                f"after instance {instance.instance_id} failed"
            )

    def _refresh_scheduler_state(self) -> None:
        """Periodically make the active registry agree with filesystem evidence."""
        if not self.registry.reserve_artifact_assessment():
            return
        try:
            demanded = self.registry.demanded_instance_ids()
            if demanded:
                try:
                    assess_registry(self.registry, instance_ids=demanded, compiled=True)
                except AssessmentConflict:
                    self._log("artifact assessment deferred because the registry kept changing")
            from nro.orchestration.branch_reconciliation import reconcile_branch_requests

            reconcile_branch_requests(self.registry)
            cancelled = self.registry.cancel_attempts_with_stale_upstreams()
            self.registry.reconcile_requests()
            if cancelled:
                self._log(
                    f"requested cancellation of {len(cancelled)} active instance attempt(s) "
                    "whose upstream artifacts became stale"
                )
        finally:
            self.registry.finish_artifact_assessment()

    def run(self) -> int:
        """Run the claim/execute loop until shutdown, draining, or idle timeout.

        Register and renew the worker lease, update attempts, and close the worker
        record on exit. Scientific failures are recorded per attempt.
        """
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self.registry.register_worker(
            self.worker_id,
            resource_class=self.resource_class,
            memory_gb=self.memory_gb,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        self.registry.mark_submission_running(os.environ.get("SLURM_JOB_ID", ""))
        self._log(
            f"started (Slurm job {os.environ.get('SLURM_JOB_ID', 'none')}; "
            f"class={self.resource_class}; memory={self.memory_gb} GB)"
        )
        idle_since = time.monotonic()
        idle_announced = False
        last_refresh_check = 0.0
        try:
            self.registry.reconcile_scheduler_submissions()
            while not self.stop_requested:
                if self.registry.worker_shutdown_requested(self.worker_id):
                    self.stop_requested = True
                    self._log("shutdown requested by owning user")
                    break
                self.registry.recover_orphaned_attempts()
                if time.monotonic() - last_refresh_check >= 30.0:
                    self._refresh_scheduler_state()
                    last_refresh_check = time.monotonic()
                if (
                    self.deadline is not None
                    and self.deadline - time.monotonic() <= self.drain_seconds
                ):
                    self.registry.heartbeat_worker(self.worker_id, state="draining")
                    self._log("draining before the Slurm wall-time limit")
                    self._submit_successor()
                    break
                instance = self.registry.claim_ready_instance(
                    self.worker_id,
                    COMPATIBLE[self.resource_class],
                    memory_gb=self.memory_gb,
                )
                if instance is None:
                    from nro.bidsify.index import IngestionIndex

                    ingestion = IngestionIndex(self.registry).claim(self.worker_id, self.memory_gb)
                    if ingestion is not None:
                        idle_since = time.monotonic()
                        idle_announced = False
                        self._execute_ingestion(ingestion)
                        continue
                    required_memory = self.registry.required_memory_above(self.memory_gb)
                    if required_memory is not None:
                        self._submit_adaptive_worker(required_memory)
                    self.registry.heartbeat_worker(self.worker_id, state="idle")
                    if not idle_announced:
                        self._log("idle; waiting for a ready instance")
                        idle_announced = True
                    if time.monotonic() - idle_since >= self.idle_timeout:
                        self._log(f"idle timeout reached after {self.idle_timeout:g}s; exiting")
                        break
                    time.sleep(self.poll_interval)
                    continue
                idle_since = time.monotonic()
                idle_announced = False
                attempt_id = instance.attempt_id
                self._log(
                    f"claimed instance {instance.instance_id} (module={instance.module}; "
                    f"participant=sub-{instance.participant}; attempt={attempt_id}); "
                    f"instance log: {instance.log_path}"
                )
                started_at = time.monotonic()
                self._execute(instance)
                self._log(
                    f"released instance {instance.instance_id} after {time.monotonic() - started_at:.3f}s; "
                    f"{self._attempt_summary(attempt_id)}; instance log: {instance.log_path}"
                )
                if self.registry.worker_shutdown_requested(self.worker_id):
                    self.stop_requested = True
                    self._log("shutdown requested by owning user")
                    break
                self._refresh_scheduler_state()
                self._expand_ready_pool()
            return 0
        except BaseException as error:
            self._log(f"fatal worker error: {type(error).__name__}: {error}")
            raise
        finally:
            final_state = "terminated" if self.stop_requested else "exited"
            self.registry.close_worker(self.worker_id, state=final_state)
            self.registry.mark_submission_complete(os.environ.get("SLURM_JOB_ID"))
            self._log(f"stopped (state={final_state})")
            cleanup_cache(self.registry)


def build_parser() -> argparse.ArgumentParser:
    """Build the internal worker-process argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument("--resource-class", choices=tuple(COMPATIBLE), default="large")
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument("--idle-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--walltime-seconds", type=float)
    parser.add_argument("--drain-seconds", type=float, default=15 * 60)
    parser.add_argument("--profile")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one reusable worker until it drains or receives shutdown."""
    args = build_parser().parse_args(argv)
    registry = Registry.for_project("", bids_root=args.bids_root)
    from nro.orchestration.scheduler_implementation import require_worker_source

    require_worker_source(registry.paths.control)
    raise SystemExit(
        Worker(
            registry,
            resource_class=args.resource_class,
            memory_gb=args.memory_gb,
            idle_timeout=args.idle_timeout,
            poll_interval=args.poll_interval,
            walltime_seconds=args.walltime_seconds,
            drain_seconds=args.drain_seconds,
            profile=args.profile,
        ).run()
    )


if __name__ == "__main__":
    main()
