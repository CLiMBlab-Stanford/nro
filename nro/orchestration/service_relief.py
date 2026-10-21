"""Release idle worker allocations while a control service waits for Slurm."""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nro.engine.io import atomic_write_json, read_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import ensure_shared_directory

SERVICE_START_GRACE_SECONDS = 60.0
RELIEF_INTERVAL_SECONDS = 5 * 60.0

_SLURM_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_WORKER_JOB_NAMES = frozenset({"nro-worker", "nro-long-worker", "nro-gpu-worker"})


@dataclass(frozen=True)
class PendingService:
    """A scheduler allocation waiting to start."""

    name: str
    token: str
    job_id: str
    reason: str


def _scontrol_job(job_id: str) -> str:
    try:
        result = subprocess.run(
            ["scontrol", "--oneliner", "show", "job", job_id],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode:
        return ""
    return result.stdout.strip()


def _field(output: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", output)
    return "" if match is None else match.group(1)


def _pending_job_reason(job_id: str) -> str | None:
    output = _scontrol_job(job_id)
    if not output:
        return None
    if _field(output, "JobState").upper() != "PENDING":
        return None
    return _field(output, "Reason").strip().strip("()") or "pending"


def cancel_pending_workers(control: Path) -> int:
    """Cancel pending worker jobs owned by this control store."""
    try:
        result = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--states=PENDING",
                "--name=" + ",".join(sorted(_WORKER_JOB_NAMES)),
                "--format=%i|%j",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode:
        return 0
    workers = (ControlPaths(control).scheduler / "workers").resolve()
    selected: list[str] = []
    for row in result.stdout.splitlines():
        try:
            job_id, name = (value.strip() for value in row.split("|", 1))
        except ValueError:
            continue
        if name not in _WORKER_JOB_NAMES or _SLURM_JOB_ID.fullmatch(job_id) is None:
            continue
        record = _scontrol_job(job_id)
        command = _field(record, "Command")
        if _field(record, "JobState") != "PENDING" or not command or command == "(null)":
            continue
        path = Path(command).expanduser()
        try:
            owned = path.resolve().parent == workers and path.name.startswith("worker-")
        except OSError:
            owned = False
        if owned:
            selected.append(job_id)
    if not selected:
        return 0
    try:
        cancelled = subprocess.run(
            ["scancel", "--state=PENDING", *selected],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return len(selected) if cancelled.returncode == 0 else 0


def _age_seconds(record: dict, now: float) -> float:
    try:
        created = datetime.fromisoformat(str(record["created_at"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, now - created.timestamp())


def _launches(control: Path) -> tuple[tuple[str, dict], ...]:
    from nro.orchestration import scheduler_bus

    if scheduler_bus.read_active(control) is None:
        launch = scheduler_bus.read_launch(control)
        return () if launch is None else (("scheduler", launch),)
    return ()


def pending_service(
    control: Path,
    *,
    now: float | None = None,
) -> PendingService | None:
    """Return a control service that has remained pending beyond its grace period."""
    checked_at = time.time() if now is None else now
    for name, launch in _launches(control):
        token = str(launch.get("token") or "")
        job_id = str(launch.get("job_id") or "")
        if not token or not job_id.isdigit():
            continue
        if _age_seconds(launch, checked_at) < SERVICE_START_GRACE_SECONDS:
            continue
        reason = _pending_job_reason(job_id)
        if reason is None:
            continue
        return PendingService(name, token, job_id, reason)
    return None


def claim_relief(
    control: Path,
    service: PendingService,
    *,
    worker_id: str,
    now: float | None = None,
) -> bool:
    """Permit one idle worker to retire during each relief interval."""
    checked_at = time.time() if now is None else now
    root = ControlPaths(control).service / "relief"
    ensure_shared_directory(root)
    lock = root / "claim.lock"
    with lock.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state_path = root / f"{service.name}.json"
        try:
            state = read_json(state_path)
        except (FileNotFoundError, OSError, ValueError):
            state = {}
        try:
            retired_at = float(state.get("retired_at", 0))
        except (TypeError, ValueError):
            retired_at = 0.0
        if state.get("token") == service.token and (
            checked_at - retired_at < RELIEF_INTERVAL_SECONDS
        ):
            return False
        atomic_write_json(
            state_path,
            {
                "service": service.name,
                "token": service.token,
                "job_id": service.job_id,
                "reason": service.reason,
                "worker_id": worker_id,
                "retired_at": checked_at,
            },
            sort_keys=True,
            mode=0o664,
        )
    return True


def await_service_relief(
    control: Path,
    *,
    worker_id: str,
    poll_seconds: float = 5.0,
) -> PendingService | None:
    """Wait behind a pending service launch and retire if elected."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    while True:
        service = pending_service(control)
        if service is None:
            return None
        if claim_relief(control, service, worker_id=worker_id):
            cancel_pending_workers(control)
            return service
        time.sleep(max(0.1, poll_seconds))
