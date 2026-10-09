"""Discover, fence, and launch the persistent planning broker."""

from __future__ import annotations

import getpass
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

from nro.engine.io import atomic_write_json, atomic_write_text, read_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import ensure_shared_directory, utcnow
from nro.orchestration.scheduler_bus import (
    LEASE_SECONDS,
    PROTOCOL,
    STARTING_GRACE_SECONDS,
    LaunchClaim,
    _identifier,
    _launch_abandoned,
)

PLANNER_CPUS = 2
PLANNER_MEMORY_GB = 8
PLANNER_IDLE_GRACE_SECONDS = 12 * 60 * 60.0


def prepare(control: Path) -> ControlPaths:
    """Create the planner control directory without starting the service."""
    paths = ControlPaths(control)
    paths.require_current_layout()
    ensure_shared_directory(paths.planner)
    try:
        paths.planner.chmod(0o2775)
    except PermissionError:
        pass
    return paths


def read_active(control: Path) -> dict | None:
    """Return the live planner endpoint, excluding expired leases."""
    try:
        value = read_json(ControlPaths(control).planner_active)
    except FileNotFoundError:
        return None
    required = {"protocol", "token", "job_id", "host", "pid", "port", "heartbeat"}
    if set(value) != required or value["protocol"] != PROTOCOL:
        return None
    try:
        if not 0 < int(value["port"]) < 65536:
            return None
        if time.time() - float(value["heartbeat"]) > LEASE_SECONDS:
            return None
    except (TypeError, ValueError):
        return None
    return value


def read_launch(control: Path) -> dict | None:
    """Return the current planner launch owner, when complete."""
    try:
        return read_json(ControlPaths(control).planner_launch / "owner.json")
    except FileNotFoundError:
        return None


def claim_launch(control: Path) -> LaunchClaim | None:
    """Claim one planner launch, recovering only a terminal owner."""
    paths = prepare(control)
    launch = paths.planner_launch
    token = uuid.uuid4().hex
    try:
        launch.mkdir(mode=0o2775)
    except FileExistsError:
        record = read_launch(control)
        if record is None:
            try:
                if time.time() - launch.stat().st_mtime <= STARTING_GRACE_SECONDS:
                    return None
            except OSError:
                return None
            record = {"created_at": "invalid"}
        if record.get("created_at") != "invalid" and not _launch_abandoned(record):
            return None
        recovery = paths.planner / "launch.recovery"
        try:
            recovery.mkdir(mode=0o2775)
        except FileExistsError:
            return None
        try:
            record = read_launch(control)
            if record is not None and not _launch_abandoned(record):
                return None
            stale = paths.planner / f"launch.stale-{uuid.uuid4().hex}"
            try:
                launch.rename(stale)
            except FileNotFoundError:
                pass
            shutil.rmtree(stale, ignore_errors=True)
            try:
                launch.mkdir(mode=0o2775)
            except FileExistsError:
                return None
        finally:
            try:
                recovery.rmdir()
            except OSError:
                pass
    owner = launch / "owner.json"
    atomic_write_json(
        owner,
        {
            "protocol": PROTOCOL,
            "token": token,
            "created_at": utcnow(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job_id": "",
        },
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    return LaunchClaim(token, owner)


def update_launch_job(claim: LaunchClaim, job_id: str) -> None:
    """Bind a submitted allocation to its planner launch claim."""
    current = read_json(claim.owner_path)
    if current.get("token") != claim.token:
        raise RuntimeError("Planner launch ownership changed during submission")
    current["job_id"] = _identifier(str(job_id), "planner Slurm job ID")
    atomic_write_json(claim.owner_path, current, sort_keys=True, mode=0o664, durable=True)


def release_launch(control: Path, token: str) -> None:
    """Release planner launch state only when the token still owns it."""
    launch = ControlPaths(control).planner_launch
    record = read_launch(control)
    if record and record.get("token") == token:
        stale = launch.with_name(f"launch.released-{token}")
        try:
            launch.rename(stale)
        except FileNotFoundError:
            return
        shutil.rmtree(stale, ignore_errors=True)


def publish_active(control: Path, *, token: str, job_id: str | None, host: str, port: int) -> None:
    """Publish or refresh the current planner endpoint."""
    atomic_write_json(
        ControlPaths(control).planner_active,
        {
            "protocol": PROTOCOL,
            "token": _identifier(token, "planner fencing token"),
            "job_id": str(job_id or ""),
            "host": str(host),
            "pid": os.getpid(),
            "port": int(port),
            "heartbeat": time.time(),
        },
        sort_keys=True,
        mode=0o664,
        durable=True,
    )


def activate(control: Path, token: str, *, host: str, port: int) -> None:
    """Activate the matching planner launch and fence delayed jobs."""
    claim = read_launch(control)
    if not claim or claim.get("token") != token:
        raise RuntimeError("Planner launch token is obsolete")
    active = read_active(control)
    if active and active.get("token") != token:
        raise RuntimeError("Another planner is already active")
    publish_active(
        control,
        token=token,
        job_id=os.environ.get("SLURM_JOB_ID") or str(claim.get("job_id") or ""),
        host=host,
        port=port,
    )


def deactivate(control: Path, token: str) -> None:
    """Remove planner state owned by one service incarnation."""
    path = ControlPaths(control).planner_active
    try:
        active = read_json(path)
    except FileNotFoundError:
        active = None
    if active and active.get("token") == token:
        path.unlink(missing_ok=True)
    release_launch(control, token)


def write_script(
    control: Path,
    *,
    token: str,
    source,
    site: Path,
    python: Path,
    partition: str,
    account: str | None,
    time_hours: int = 24,
    memory_gb: int = PLANNER_MEMORY_GB,
    cpus: int = PLANNER_CPUS,
) -> Path:
    """Write one pinned Slurm script for the planning broker."""
    import shlex

    paths = prepare(control)
    script = paths.planner / f"planner-{token}.sbatch"
    command = source.command(
        (
            str(python),
            "-m",
            "nro.orchestration.planner_service",
            "--control",
            str(control),
            "--launch-token",
            token,
        ),
        site=site,
    )
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=nro-planner",
        f"#SBATCH --time={time_hours}:00:00",
        f"#SBATCH --mem={memory_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={paths.planner}/planner-%j.log",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines.extend(("set -euo pipefail", "export NRO_PROCESS_ROLE=planner"))
    lines.append("exec " + shlex.join(command))
    atomic_write_text(script, "\n".join(lines) + "\n", mode=0o664)
    return script


def submit(script: Path) -> str:
    """Submit one planner allocation and return its Slurm identity."""
    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(script)], check=True, text=True, capture_output=True
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"Slurm rejected planner submission for {script}" + (f": {detail}" if detail else "")
        ) from error
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError("sbatch returned no planner job ID")
    return job_id
