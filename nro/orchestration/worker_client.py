"""Worker-side client for the scheduler's durable message protocol."""

from __future__ import annotations

import getpass
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from nro.engine.io import atomic_write_json, read_json
from nro.orchestration.contracts import ExecutionEnvelope
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.scheduler_client import SchedulerEndpoint, exchange
from nro.orchestration.source_snapshots import SourceSnapshot


class WorkerSchedulerClient:
    """Expose worker operations without giving the worker a database handle."""

    def __init__(self, *, bids_root: Path, control: Path, worker_id: str) -> None:
        """Bind the active source, site, and worker identity used by every event."""
        root = os.environ.get("NRO_EXECUTION_SOURCE_ROOT")
        digest = os.environ.get("NRO_EXECUTION_SOURCE_DIGEST")
        site = os.environ.get("NRO_SITE_CONFIG")
        if not root or not digest or not site:
            raise ValueError("Scheduled workers require a pinned source and site")
        self.paths = SimpleNamespace(
            control=Path(control).resolve(),
            bids_root=Path(bids_root).resolve(),
        )
        self.worker_id = worker_id
        self.token = os.urandom(16).hex()
        self.sequence = 0
        self.endpoint = SchedulerEndpoint(
            Path(control).resolve(),
            Path(bids_root).resolve(),
            SourceSnapshot(Path(root), digest),
            Path(site),
            Path(sys.executable),
        )
        self.worker_root = ControlPaths(control).service_workers / worker_id

    def _call(self, action: str, **fields: Any) -> Any:
        """Send one ordered worker event and return its service result."""
        self.sequence += 1
        return exchange(
            self.endpoint,
            {
                "operation": "worker",
                "action": action,
                "worker_id": self.worker_id,
                "worker_token": self.token,
                "sequence": self.sequence,
                **fields,
            },
            timeout=300.0,
        )

    def register_worker(self, worker_id: str, **fields: Any) -> None:
        """Register this worker and establish its fencing token."""
        self._call(
            "register",
            user_name=getpass.getuser(),
            hostname=socket.gethostname(),
            pid=os.getpid(),
            **fields,
        )

    def heartbeat_worker(self, worker_id: str, *, state: str, **_fields: Any) -> None:
        """Publish the worker's current state without opening the registry."""
        from nro.orchestration.scheduler_bus import read_active
        from nro.orchestration.scheduler_client import _ensure_service

        if read_active(self.endpoint.control) is None:
            _ensure_service(self.endpoint, explicit=False)
        self.sequence += 1
        atomic_write_json(
            self.worker_root / "presence.json",
            {
                "protocol": 1,
                "worker_id": self.worker_id,
                "worker_token": self.token,
                "sequence": self.sequence,
                "state": state,
                "heartbeat": time.time(),
            },
            sort_keys=True,
            mode=0o664,
        )

    def _control(self) -> dict:
        try:
            value = read_json(self.worker_root / "control.json")
        except FileNotFoundError:
            return {}
        if value.get("worker_token") != self.token:
            return {}
        return value

    def worker_shutdown_requested(self, worker_id: str) -> bool:
        """Return whether the current control file requests shutdown."""
        return self._control().get("state") == "shutdown_requested"

    def mark_submission_running(self, slurm_job_id: str) -> None:
        """Associate this running worker with its Slurm submission."""
        self._call("submission_running", slurm_job_id=slurm_job_id)

    def mark_submission_complete(self, slurm_job_id: str | None) -> None:
        """Mark this worker's Slurm submission complete when it has one."""
        self._call("submission_complete", slurm_job_id=slurm_job_id)

    def reconcile_scheduler_submissions(self) -> int:
        """Ask the controller to reconcile known worker allocations."""
        return int(self._call("reconcile_submissions"))

    def recover_orphaned_attempts(self) -> int:
        """Ask the controller to recover work owned by confirmed-dead workers."""
        return int(self._call("recover_orphans"))

    def claim_ready_instance(
        self, worker_id: str, resource_classes: Sequence[str], *, memory_gb: int = 32
    ) -> ExecutionEnvelope | None:
        """Claim one compatible derivative assignment, if any is ready."""
        value = self._call("claim", resource_classes=list(resource_classes), memory_gb=memory_gb)
        return None if value is None else ExecutionEnvelope.from_dict(value)

    def claim_ingestion(self, memory_gb: int) -> dict | None:
        """Claim one compatible ingestion stage, if any is ready."""
        return self._call("claim_ingestion", memory_gb=memory_gb)

    def finish_ingestion(self, request_id: str, *, branch: str, state: str, changes=None) -> None:
        """Publish the terminal result of an ingestion stage."""
        self._call(
            "finish_ingestion",
            request_id=request_id,
            branch=branch,
            state=state,
            changes=changes,
        )

    def attempt_cancel_requested(self, attempt_id: int) -> bool:
        """Return whether the current control file cancels an attempt."""
        return int(attempt_id) in set(self._control().get("cancel_attempts", ()))

    def record_attempt_process(self, attempt_id: int, process_group_id: int) -> None:
        """Record the process group supervised for an attempt."""
        self._call("attempt_process", attempt_id=attempt_id, process_group_id=process_group_id)

    def finish_attempt(self, attempt_id: int, **fields: Any) -> None:
        """Publish an attempt's terminal state and optional error details."""
        self._call("finish_attempt", attempt_id=attempt_id, **fields)

    def record_oom(self, attempt_id: int, *, message: str) -> int | None:
        """Record an out-of-memory result and return its retry tier."""
        value = self._call("record_oom", attempt_id=attempt_id, message=message)
        return None if value is None else int(value)

    def cancel_attempts_downstream_of_failure(self, instance_id: int) -> list[dict]:
        """Cancel active consumers of a failed instance."""
        return self._call("cancel_failed_descendants", instance_id=instance_id)

    def runner_graph_signature(self, instance_id: int) -> str:
        """Return the current transitive graph signature for an instance."""
        return str(self._call("runner_graph_signature", instance_id=instance_id))

    def attempt_summary(self, attempt_id: int) -> str:
        """Return a concise summary of an attempt's saved state."""
        return str(self._call("attempt_summary", attempt_id=attempt_id))

    def record_completion(
        self, *, instance_id: int, attempt_id: int, outputs: Sequence[Path]
    ) -> dict:
        """Validate outputs and publish the completion manifest."""
        return self._call(
            "record_completion",
            instance_id=instance_id,
            attempt_id=attempt_id,
            outputs=[str(path) for path in outputs],
        )

    def refresh_scheduler_state(self) -> int:
        """Reassess demanded artifacts and cancel obsolete attempts."""
        return int(self._call("refresh"))

    def required_memory_above(self, memory_gb: int) -> int | None:
        """Return the smallest ready memory tier above this worker's capacity."""
        value = self._call("required_memory", memory_gb=memory_gb)
        return None if value is None else int(value)

    def request_capacity(self, kind: str, *, memory_gb: int, profile: str | None = None) -> None:
        """Ask the controller to supply eligible worker capacity."""
        self._call("request_capacity", kind=kind, memory_gb=memory_gb, profile=profile)

    def close_worker(self, worker_id: str, *, state: str = "exited") -> None:
        """Publish this worker's terminal state."""
        self._call("close", state=state)

    def cleanup_cache(self) -> None:
        """Ask the controller to remove unreferenced execution captures."""
        self._call("cleanup_cache")
