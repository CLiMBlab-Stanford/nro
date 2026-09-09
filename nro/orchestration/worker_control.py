"""Shared control operations for the lab-wide worker pool."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time

from nro.orchestration.registry import Registry


def cancel_worker_allocations(
    registry: Registry,
    shutdown: dict,
    *,
    update_registry: bool = True,
) -> tuple[int, list[str]]:
    """Cancel recorded Slurm allocations and return successes and failures."""
    stopped = 0
    failures: list[str] = []
    submissions: dict[str, list[int]] = {}
    for submission_id, job_id in shutdown["submissions"]:
        submissions.setdefault(job_id, []).append(submission_id)
    for job_id in shutdown["job_ids"]:
        try:
            result = subprocess.run(["scancel", job_id], text=True, capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{job_id}: {error}")
            continue
        if result.returncode == 0:
            stopped += 1
        else:
            detail = result.stderr.strip() or result.stdout.strip() or "scancel failed"
            failures.append(f"{job_id}: {detail}")
        if update_registry:
            for submission_id in submissions.get(job_id, ()):
                registry.update_submission(
                    submission_id,
                    state="cancelled" if result.returncode == 0 else "error",
                )
    return stopped, failures


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _slurm_job_terminal(job_id: str) -> bool | None:
    """Return whether Slurm confirms that an allocation no longer exists."""
    if not shutil.which("squeue"):
        return None
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"],
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0:
        return not bool(result.stdout.strip())
    message = (result.stderr + result.stdout).lower()
    if "invalid job id" in message:
        return True
    return None


def _active_pool_members(activity: dict) -> tuple[list[str], list[str]]:
    """Return worker IDs and job IDs whose termination is not yet confirmed."""
    now = time.time()
    hostname = socket.gethostname()
    workers: list[str] = []
    jobs: set[str] = set()
    for row in activity["workers"]:
        job_id = row.get("slurm_job_id")
        if job_id:
            jobs.add(str(job_id))
            continue
        if row.get("hostname") == hostname:
            try:
                alive = _process_exists(int(row["pid"]))
            except (TypeError, ValueError):
                alive = True
        else:
            lease = row.get("lease_expires_at")
            alive = lease is None or float(lease) > now
        if alive:
            workers.append(str(row["id"]))
    jobs.update(
        str(row["slurm_job_id"]) for row in activity["submissions"] if row.get("slurm_job_id")
    )
    active_jobs = [job_id for job_id in sorted(jobs) if _slurm_job_terminal(job_id) is not True]
    return workers, active_jobs


def wait_for_worker_shutdown(
    registry: Registry,
    *,
    timeout: float = 180.0,
    poll_interval: float = 1.0,
) -> None:
    """Wait until every worker process/allocation is confirmed inactive."""
    deadline = time.monotonic() + timeout
    while True:
        workers, jobs = _active_pool_members(registry.worker_pool_activity(for_repair=True))
        if not workers and not jobs:
            return
        if time.monotonic() >= deadline:
            details = []
            if workers:
                details.append("workers " + ", ".join(workers))
            if jobs:
                details.append("Slurm jobs " + ", ".join(jobs))
            raise RuntimeError(
                "Timed out waiting for the worker pool to stop: " + "; ".join(details)
            )
        time.sleep(poll_interval)


def stop_worker_pool_for_repair(registry: Registry) -> dict:
    """Freeze and synchronously stop the complete lab-wide worker pool."""
    shutdown = registry.request_worker_shutdown(all_users=True, for_repair=True)
    # Repair will discard the existing registry after every allocation stops,
    # so no follow-up scheduler bookkeeping is useful here. This also keeps
    # repair usable when only the registry version, rather than its worker
    # control tables, is obsolete.
    stopped_jobs, failures = cancel_worker_allocations(registry, shutdown, update_registry=False)
    wait_for_worker_shutdown(registry)
    from nro.bidsify.index import IngestionIndex

    # These allocations are now confirmed stopped. Ingestion survives repair,
    # so release its leases before the worker table is discarded.
    with registry._lock():
        IngestionIndex(registry).recover_locked({row["id"] for row in shutdown["worker_rows"]})
    return {**shutdown, "stopped_jobs": stopped_jobs, "cancellation_failures": failures}
