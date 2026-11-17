"""Query Slurm for the state of allocations that nro workers run in.

Each function returns True or False only when Slurm gives a conclusive answer
and None when the query fails or the state is ambiguous. Callers decide how to
treat None. Queries run with `LC_ALL=C` where the parser depends on message
text.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone


def job_terminal(job_id: str) -> bool | None:
    """Check whether a job has left Slurm; query failures remain unknown.

    Slurm may report an expired job ID as an error instead of an empty
    result. Only that specific diagnostic establishes absence on failure.
    Accounting provides a bounded fallback when the live queue is
    temporarily unavailable.
    """
    if shutil.which("squeue"):
        try:
            result = subprocess.run(
                ["squeue", "--noheader", "--jobs", str(job_id), "--format", "%T"],
                check=False,
                text=True,
                capture_output=True,
                timeout=5,
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None:
            if not result.returncode:
                return not bool(result.stdout.strip())
            if (
                not result.stdout.strip()
                and result.stderr.strip() == "slurm_load_jobs error: Invalid job id specified"
            ):
                return True
    return accounting_terminal(job_id)


def accounting_terminal(job_id: str) -> bool | None:
    """Return terminal state from Slurm accounting when it is conclusive."""
    if not shutil.which("sacct"):
        return None
    try:
        result = subprocess.run(
            [
                "sacct",
                "--jobs",
                str(job_id),
                "--noheader",
                "--parsable2",
                "--format",
                "JobIDRaw,State",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    states = {
        fields[1].split(maxsplit=1)[0].split("+", 1)[0].upper()
        for line in result.stdout.splitlines()
        if len(fields := line.split("|")) >= 2 and fields[0].strip() == str(job_id)
    }
    if not states:
        return None
    active = {
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
    }
    terminal = {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
    if states <= terminal:
        return True
    if states & active:
        return False
    return None


def _accounting_states(job_id: str) -> list[str] | None:
    """Return the accounting states of a job and its steps, or None when unavailable."""
    if not shutil.which("sacct"):
        return None
    try:
        result = subprocess.run(
            ["sacct", "--jobs", str(job_id), "--noheader", "--parsable2", "--format", "State"],
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.split("|", 1)[0].strip().upper() for line in result.stdout.splitlines()]


def out_of_memory(job_id: str) -> bool | None:
    """Return whether Slurm accounting records an out-of-memory termination."""
    states = _accounting_states(job_id)
    return None if states is None else any(state.startswith("OUT_OF_MEMORY") for state in states)


def timed_out(job_id: str) -> bool | None:
    """Return whether Slurm accounting identifies a wall-time expiry."""
    states = _accounting_states(job_id)
    return None if states is None else any(state.startswith("TIMEOUT") for state in states)


def run_replaced(job_id: str, *, hostname: str, registered_at: str) -> bool | None:
    """Return whether a requeue replaced the Slurm run that a worker belonged to.

    Slurm keeps the job ID when it requeues a preempted job, so a live job ID
    does not show that the worker's run is still alive. The worker's run has
    ended when the job is pending again, runs on other nodes, or started after
    the worker registered. Query failures and other states remain unknown.
    """
    if not shutil.which("squeue"):
        return None
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", str(job_id), "--format", "%T|%S|%N"],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    if result.returncode or len(lines) != 1 or lines[0].count("|") != 2:
        return None
    state, start, nodes = (field.strip() for field in lines[0].split("|"))
    if state.upper() == "PENDING":
        return True
    if state.upper() != "RUNNING" or not nodes:
        return None
    hosts = nodes.split(",")
    if "[" in nodes:
        try:
            expanded = subprocess.run(
                ["scontrol", "show", "hostnames", nodes],
                check=True,
                text=True,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        hosts = expanded.stdout.split()
    if hostname.split(".", 1)[0] not in {host.split(".", 1)[0] for host in hosts}:
        return True
    try:
        # squeue reports local time without an offset.
        run_started = datetime.fromisoformat(start).astimezone(timezone.utc)
        worker_started = datetime.fromisoformat(registered_at)
    except (TypeError, ValueError):
        return None
    if worker_started.tzinfo is None:
        worker_started = worker_started.replace(tzinfo=timezone.utc)
    return run_started > worker_started
