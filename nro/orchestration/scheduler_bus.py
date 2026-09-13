"""Persist scheduler recovery records and manage service election."""

from __future__ import annotations

import getpass
import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nro.engine.io import atomic_write_json, atomic_write_text, read_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryLock, ensure_shared_directory, utcnow

PROTOCOL = 1
HEARTBEAT_SECONDS = 5.0
LEASE_SECONDS = 30.0
STARTING_GRACE_SECONDS = 30.0
DEFAULT_IDLE_GRACE_SECONDS = 30.0
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid scheduler {label}")
    return value


def prepare(control: Path) -> ControlPaths:
    """Create the shared exchange directories without opening the registry."""
    paths = ControlPaths(control)
    paths.require_current_layout()
    for path in (
        paths.service,
        paths.service_inbox,
        paths.service_responses,
    ):
        ensure_shared_directory(path)
        try:
            path.chmod(0o2775)
        except PermissionError:
            pass
    return paths


def _shard(identifier: str) -> str:
    return identifier.replace("-", "")[:2].lower()


def message_path(control: Path, message_id: str) -> Path:
    """Return the immutable inbox path for one validated message ID."""
    message_id = _identifier(message_id, "message ID")
    return ControlPaths(control).service_inbox / _shard(message_id) / f"{message_id}.json"


def response_path(control: Path, message_id: str) -> Path:
    """Return the response path for one validated message ID."""
    message_id = _identifier(message_id, "message ID")
    return ControlPaths(control).service_responses / f"{message_id}.json"


def create_message(payload: dict[str, Any], *, kind: str = "command") -> dict[str, Any]:
    """Create one validated scheduler record without publishing it."""
    message_id = uuid.uuid4().hex
    return {
        "protocol": PROTOCOL,
        "id": message_id,
        "kind": kind,
        "created_at": utcnow(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "payload": payload,
    }


def publish_message(control: Path, payload: dict[str, Any], *, kind: str = "command") -> str:
    """Publish one immutable recovery record and return its identity."""
    paths = prepare(control)
    record = create_message(payload, kind=kind)
    target = message_path(paths.root, record["id"])
    ensure_shared_directory(target.parent)
    atomic_write_json(target, record, sort_keys=True, mode=0o664, durable=True)
    return str(record["id"])


def publish_response(control: Path, message_id: str, value: dict[str, Any]) -> None:
    """Atomically publish the terminal response for a consumed message."""
    path = response_path(control, message_id)
    atomic_write_json(path, value, sort_keys=True, mode=0o664, durable=True)


def read_response(control: Path, message_id: str) -> dict[str, Any] | None:
    """Read a complete response, or return None while it is absent."""
    path = response_path(control, message_id)
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def pending_messages(
    control: Path, *, limit: int = 100, minimum_age: float = 1.0
) -> tuple[Path, ...]:
    """Return a bounded batch old enough to require filesystem recovery."""
    root = ControlPaths(control).service_inbox
    if not root.is_dir():
        return ()
    cutoff = time.time() - minimum_age
    candidates = []
    for path in root.glob("*/*.json"):
        try:
            if path.stat().st_mtime <= cutoff:
                candidates.append(path)
        except OSError:
            continue
    return tuple(sorted(candidates, key=lambda path: path.name)[:limit])


def consume_message(path: Path) -> dict[str, Any]:
    """Read and validate one immutable inbox record."""
    return validate_message(read_json(path), expected_id=path.stem)


def validate_message(record: Any, *, expected_id: str | None = None) -> dict[str, Any]:
    """Validate one durable or directly received scheduler record."""
    if (
        not isinstance(record, dict)
        or set(record) != {"protocol", "id", "kind", "created_at", "user", "host", "payload"}
        or record["protocol"] != PROTOCOL
        or record["kind"] not in {"command", "worker"}
        or not isinstance(record["payload"], dict)
        or expected_id is not None
        and expected_id != record["id"]
    ):
        raise ValueError(f"Invalid scheduler message: {expected_id or '<direct>'}")
    _identifier(record["id"], "message ID")
    return record


def acknowledge_message(path: Path) -> None:
    """Remove a message only after its registry effect and response are durable."""
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def read_active(control: Path) -> dict[str, Any] | None:
    """Read the current controller lease without treating an expired record as live."""
    path = ControlPaths(control).service_active
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    required = {
        "protocol",
        "token",
        "generation",
        "job_id",
        "host",
        "pid",
        "port",
        "heartbeat",
    }
    if set(value) != required or value["protocol"] != PROTOCOL:
        return None
    try:
        port = int(value["port"])
        if not 0 < port < 65536 or not str(value["host"]):
            return None
        if time.time() - float(value["heartbeat"]) > LEASE_SECONDS:
            return None
    except (TypeError, ValueError):
        return None
    return value


def publish_active(
    control: Path,
    *,
    token: str,
    generation: int,
    job_id: str | None,
    host: str,
    port: int,
) -> dict[str, Any]:
    """Publish or renew the controller's externally visible lease."""
    if not host or not 0 < int(port) < 65536:
        raise ValueError("Scheduler endpoint is invalid")
    record = {
        "protocol": PROTOCOL,
        "token": _identifier(token, "fencing token"),
        "generation": int(generation),
        "job_id": str(job_id or ""),
        "host": str(host),
        "pid": os.getpid(),
        "port": int(port),
        "heartbeat": time.time(),
    }
    atomic_write_json(
        ControlPaths(control).service_active,
        record,
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    return record


def read_launch(control: Path) -> dict[str, Any] | None:
    """Read the current STARTING claim, if it has a complete owner record."""
    path = ControlPaths(control).service_launch / "owner.json"
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def _slurm_terminal(job_id: str) -> bool | None:
    """Return a positive terminal-state result without guessing on scheduler failure."""
    return RegistryLock._slurm_terminal(job_id)


def _launch_abandoned(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    job_id = str(record.get("job_id") or "")
    if job_id:
        if job_id.startswith("local-"):
            if record.get("host") != socket.gethostname():
                try:
                    created = datetime.fromisoformat(str(record["created_at"]))
                    return (
                        datetime.now(timezone.utc) - created
                    ).total_seconds() > STARTING_GRACE_SECONDS
                except (KeyError, TypeError, ValueError):
                    return False
            try:
                os.kill(int(job_id.removeprefix("local-")), 0)
            except ProcessLookupError:
                return True
            except (PermissionError, OSError, ValueError):
                return False
            return False
        return _slurm_terminal(job_id) is True
    try:
        created = datetime.fromisoformat(str(record["created_at"]))
        return (datetime.now(timezone.utc) - created).total_seconds() > STARTING_GRACE_SECONDS
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class LaunchClaim:
    """A unique claim that may submit and activate one controller."""

    token: str
    owner_path: Path


def claim_launch(control: Path) -> LaunchClaim | None:
    """Atomically claim controller startup, recovering only a terminal owner."""
    paths = prepare(control)
    launch = paths.service_launch
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
        if not _launch_abandoned(record):
            if record.get("created_at") != "invalid":
                return None
        recovery = paths.service / "launch.recovery"
        try:
            recovery.mkdir(mode=0o2775)
        except FileExistsError:
            return None
        try:
            record = read_launch(control)
            if record is None:
                try:
                    if time.time() - launch.stat().st_mtime <= STARTING_GRACE_SECONDS:
                        return None
                except OSError:
                    return None
            elif not _launch_abandoned(record):
                return None
            stale = paths.service / f"launch.stale-{uuid.uuid4().hex}"
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
    owner_path = launch / "owner.json"
    atomic_write_json(
        owner_path,
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
    return LaunchClaim(token, owner_path)


def update_launch_job(claim: LaunchClaim, job_id: str) -> None:
    """Bind a submitted Slurm allocation to the launch token."""
    current = read_json(claim.owner_path)
    if current.get("token") != claim.token:
        raise RuntimeError("Controller launch ownership changed during submission")
    current["job_id"] = _identifier(str(job_id), "Slurm job ID")
    atomic_write_json(claim.owner_path, current, sort_keys=True, mode=0o664, durable=True)


def release_launch(control: Path, token: str) -> None:
    """Release a launch claim only when its fencing token still owns it."""
    launch = ControlPaths(control).service_launch
    record = read_launch(control)
    if record and record.get("token") == token:
        stale = launch.with_name(f"launch.released-{token}")
        try:
            launch.rename(stale)
        except FileNotFoundError:
            return
        shutil.rmtree(stale, ignore_errors=True)


def activate(control: Path, token: str, generation: int, *, host: str, port: int) -> dict[str, Any]:
    """Activate the matching launch claim and reject delayed controller jobs."""
    claim = read_launch(control)
    if not claim or claim.get("token") != token:
        raise RuntimeError("Controller launch token is obsolete")
    active = read_active(control)
    if active and active.get("token") != token:
        raise RuntimeError("Another controller is already active")
    return publish_active(
        control,
        token=token,
        generation=generation,
        job_id=os.environ.get("SLURM_JOB_ID") or str(claim.get("job_id") or ""),
        host=host,
        port=port,
    )


def deactivate(control: Path, token: str) -> None:
    """Remove only the active and launch records owned by this controller."""
    path = ControlPaths(control).service_active
    try:
        active = read_json(path)
    except FileNotFoundError:
        active = None
    if active and active.get("token") == token:
        path.unlink(missing_ok=True)
    release_launch(control, token)


def publish_startup_error(control: Path, token: str, error: str) -> None:
    """Publish a controller startup failure for clients awaiting its token."""
    atomic_write_json(
        ControlPaths(control).service / f"startup-error-{_identifier(token, 'launch token')}.json",
        {
            "protocol": PROTOCOL,
            "token": token,
            "error": str(error),
            "created_at": utcnow(),
        },
        sort_keys=True,
        mode=0o664,
        durable=True,
    )


def read_startup_error(control: Path, token: str) -> dict[str, Any] | None:
    """Read the startup failure for one launch token, if present."""
    if not token:
        return None
    try:
        value = read_json(ControlPaths(control).service / f"startup-error-{token}.json")
    except (FileNotFoundError, ValueError, OSError):
        return None
    if value.get("protocol") != PROTOCOL or value.get("token") != token:
        return None
    return value


def write_controller_script(
    control: Path,
    *,
    bids_root: Path,
    token: str,
    source,
    site: Path,
    python: Path,
    partition: str,
    account: str | None,
) -> Path:
    """Write the pinned Slurm script for one controller launch."""
    import shlex

    paths = prepare(control)
    script = paths.service / f"controller-{token}.sbatch"
    command = source.command(
        (
            str(python),
            "-m",
            "nro.orchestration.scheduler_service",
            "--serve",
            "--launch-token",
            token,
            "--bids-root",
            str(bids_root),
        ),
        site=site,
    )
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=nro-scheduler",
        f"#SBATCH --partition={partition}",
        "#SBATCH --time=24:00:00",
        "#SBATCH --mem=1G",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --output={paths.service}/controller-%j.log",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines.extend(("set -euo pipefail", "export NRO_PROCESS_ROLE=scheduler"))
    lines.append("exec " + shlex.join(command))
    atomic_write_text(script, "\n".join(lines) + "\n", mode=0o664)
    return script


def submit_controller(script: Path) -> str:
    """Submit one controller and return its Slurm job identity."""
    result = subprocess.run(
        ["sbatch", "--parsable", str(script)], check=True, text=True, capture_output=True
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError(f"sbatch returned no controller job ID: {result.stdout!r}")
    return job_id


def read_snapshot(control: Path) -> dict[str, Any] | None:
    """Read the last complete scheduler read model without opening SQLite."""
    try:
        value = read_json(ControlPaths(control).service_snapshot)
    except FileNotFoundError:
        return None
    if value.get("protocol") != PROTOCOL or not isinstance(value.get("branches"), dict):
        raise ValueError("Unsupported scheduler status snapshot")
    return value


def publish_shutdown(control: Path, *, token: str, generation: int) -> None:
    """Prevent workers from replacing a controller stopped intentionally."""
    atomic_write_json(
        ControlPaths(control).service / "shutdown.json",
        {
            "protocol": PROTOCOL,
            "token": token,
            "generation": generation,
            "created_at": utcnow(),
        },
        sort_keys=True,
        mode=0o664,
        durable=True,
    )


def shutdown_pending(control: Path) -> bool:
    """Return whether an intentional controller shutdown remains in force."""
    path = ControlPaths(control).service / "shutdown.json"
    try:
        return read_json(path).get("protocol") == PROTOCOL
    except (FileNotFoundError, ValueError, OSError):
        return False


def clear_shutdown(control: Path) -> None:
    """Allow an explicit user command to start orchestration again."""
    (ControlPaths(control).service / "shutdown.json").unlink(missing_ok=True)


def collect_transport_garbage(control: Path, *, age_seconds: float = 86400.0) -> None:
    """Remove old acknowledged responses and obsolete controller scripts."""
    paths = ControlPaths(control)
    cutoff = time.time() - age_seconds
    for path in (
        *paths.service_responses.glob("*.json"),
        *paths.service.glob("controller-*.sbatch"),
        *paths.service.glob("controller-local-*.log"),
        *paths.service.glob("startup-error-*.json"),
    ):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            pass
