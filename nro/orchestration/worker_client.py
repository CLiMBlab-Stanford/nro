"""Worker-side client for direct scheduler coordination."""

from __future__ import annotations

import getpass
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from nro.orchestration.contracts import ExecutionEnvelope
from nro.orchestration.dependency_state import AttemptInvalidated
from nro.orchestration.scheduler_client import SchedulerEndpoint, SchedulerError, exchange
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

    def _call(self, action: str, *, durable: bool = True, **fields: Any) -> Any:
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
            require_service=True,
            durable=durable,
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
        """Refresh this worker's registry lease through the scheduler."""
        self._call("heartbeat", state=state, durable=False)

    def worker_shutdown_requested(self, worker_id: str) -> bool:
        """Return whether the scheduler requests this worker to shut down."""
        return bool(self._call("shutdown_requested", durable=False))

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

    def claim_ready_work_item(
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
        """Return whether the scheduler has cancelled an attempt."""
        return bool(self._call("attempt_cancel_requested", attempt_id=attempt_id, durable=False))

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

    def cancel_attempts_downstream_of_failure(self, work_item_id: int) -> list[dict]:
        """Cancel active consumers of a failed work item."""
        return self._call("cancel_failed_descendants", work_item_id=work_item_id)

    def runner_graph_signature(self, work_item_id: int) -> str:
        """Return the current transitive graph signature for an work item."""
        return str(self._call("runner_graph_signature", work_item_id=work_item_id, durable=False))

    def attempt_summary(self, attempt_id: int) -> str:
        """Return a concise summary of an attempt's saved state."""
        return str(self._call("attempt_summary", attempt_id=attempt_id, durable=False))

    def record_completion(
        self, *, work_item_id: int, attempt_id: int, outputs: Sequence[Path]
    ) -> dict:
        """Validate outputs and publish the completion manifest."""
        try:
            return self._call(
                "record_completion",
                work_item_id=work_item_id,
                attempt_id=attempt_id,
                outputs=[str(path) for path in outputs],
            )
        except SchedulerError as error:
            if error.error_type == "AttemptInvalidated":
                raise AttemptInvalidated(str(error)) from error
            raise

    def outputs_visible(self, outputs: Sequence[Path]) -> bool:
        """Return whether the scheduler host can see every published output."""
        return bool(
            exchange(
                self.endpoint,
                {
                    "operation": "output_visibility",
                    "paths": [str(path) for path in outputs],
                },
                timeout=60.0,
                require_service=True,
                durable=False,
            )
        )

    def refresh_scheduler_state(self) -> int:
        """Reassess demanded artifacts and cancel obsolete attempts."""
        return int(self._call("refresh"))

    def required_memory_above(self, memory_gb: int) -> int | None:
        """Return the smallest ready memory tier above this worker's capacity."""
        value = self._call("required_memory", memory_gb=memory_gb, durable=False)
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
