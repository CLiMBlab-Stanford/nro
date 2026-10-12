"""Release idle worker allocations that directly block a control service."""

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
RELIEF_INTERVAL_SECONDS = 60.0

_RUNNING_JOB_LIMITS = frozenset(
    {
        "AssocGrpJobsLimit",
        "AssocMaxJobsLimit",
        "QOSGrpJobsLimit",
        "QOSMaxJobsPerAccountLimit",
        "QOSMaxJobsPerUserLimit",
    }
)
_MEMORY = re.compile(r"^(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[KMGTP]?)(?:[cn])?$", re.I)
_SLURM_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_WORKER_JOB_NAMES = frozenset({"nro-worker", "nro-long-worker", "nro-gpu-worker"})


@dataclass(frozen=True)
class PendingService:
    """A scheduler or planner allocation that one idle worker can help start."""

    name: str
    token: str
    job_id: str
    reason: str


@dataclass(frozen=True)
class _SlurmJob:
    state: str
    reason: str
    partition: str
    account: str
    cpus: int
    memory_gb: float


def _memory_gb(value: str) -> float:
    match = _MEMORY.fullmatch(value.strip())
    if match is None:
        return float("inf")
    amount = float(match.group("value"))
    unit = match.group("unit").upper()
    scale = {
        "": 1 / 1024**3,
        "K": 1 / 1024**2,
        "M": 1 / 1024,
        "G": 1,
        "T": 1024,
        "P": 1024**2,
    }
    return amount * scale[unit]


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


def _slurm_job(job_id: str) -> _SlurmJob | None:
    output = _scontrol_job(job_id)
    if not output:
        return None

    state = _field(output, "JobState")
    reason = _field(output, "Reason")
    partition = _field(output, "Partition")
    account = _field(output, "Account")
    cpus_value = _field(output, "NumCPUs")
    memory = _field(output, "MinMemoryNode")
    per_cpu = False
    if not memory:
        memory = _field(output, "MinMemoryCPU")
        per_cpu = bool(memory)
    if not state or not reason or not cpus_value or not memory:
        return None
    try:
        cpus = int(cpus_value)
    except ValueError:
        return None
    memory_gb = _memory_gb(memory)
    if per_cpu:
        memory_gb *= cpus
    return _SlurmJob(
        state=state.upper(),
        reason=reason.strip().strip("()"),
        partition=partition,
        account=account,
        cpus=cpus,
        memory_gb=memory_gb,
    )


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
    from nro.orchestration import planner_bus, scheduler_bus

    if scheduler_bus.read_active(control) is None:
        launch = scheduler_bus.read_launch(control)
        return () if launch is None else (("scheduler", launch),)
    launches: list[tuple[str, dict]] = []
    if planner_bus.read_active(control) is None:
        launch = planner_bus.read_launch(control)
        if launch is not None:
            launches.append(("planner", launch))
    return tuple(launches)


def pending_service(
    control: Path,
    *,
    worker_cpus: int,
    worker_memory_gb: int,
    worker_partition: str,
    worker_account: str,
    now: float | None = None,
) -> PendingService | None:
    """Return a pending service that this idle allocation can help start."""
    checked_at = time.time() if now is None else now
    for name, launch in _launches(control):
        token = str(launch.get("token") or "")
        job_id = str(launch.get("job_id") or "")
        if not token or not job_id.isdigit():
            continue
        if _age_seconds(launch, checked_at) < SERVICE_START_GRACE_SECONDS:
            continue
        job = _slurm_job(job_id)
        if job is None or job.state != "PENDING":
            continue
        same_account = not job.account or not worker_account or job.account == worker_account
        if job.reason in _RUNNING_JOB_LIMITS and same_account:
            return PendingService(name, token, job_id, job.reason)
        if (
            job.reason == "Resources"
            and same_account
            and job.partition
            and job.partition == worker_partition
            and worker_cpus >= job.cpus
            and worker_memory_gb >= job.memory_gb
        ):
            return PendingService(name, token, job_id, job.reason)
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
    worker_memory_gb: int,
    poll_seconds: float = 5.0,
) -> PendingService | None:
    """Wait behind an actionable service launch and retire if elected."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        worker_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    except ValueError:
        worker_cpus = 1
    while True:
        service = pending_service(
            control,
            worker_cpus=worker_cpus,
            worker_memory_gb=worker_memory_gb,
            worker_partition=os.environ.get("SLURM_JOB_PARTITION", ""),
            worker_account=os.environ.get("SLURM_JOB_ACCOUNT", ""),
        )
        if service is None:
            return None
        if claim_relief(control, service, worker_id=worker_id):
            cancel_pending_workers(control)
            return service
        time.sleep(max(0.1, poll_seconds))
