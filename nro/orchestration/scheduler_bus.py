"""Persist scheduler recovery records and manage service election."""

from __future__ import annotations

import getpass
import json
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

from nro.configuration.store import fingerprint
from nro.engine.io import atomic_write_json, atomic_write_text, read_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryLock, ensure_shared_directory, utcnow

PROTOCOL = 1
HEARTBEAT_SECONDS = 5.0
LEASE_SECONDS = 30.0
STARTING_GRACE_SECONDS = 30.0
DEFAULT_IDLE_GRACE_SECONDS = 12 * 60 * 60.0
SCHEDULER_CPUS = 4
SCHEDULER_TIME_HOURS = 24
SCHEDULER_MEMORY_GB = 4
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_STATUS_FORMAT = 2
_STATUS_STATIC_FILE = re.compile(r"status-static-[0-9a-f]{64}\.json")
_STATUS_DYNAMIC_FILE = re.compile(r"status-dynamic-[01]\.json")
_STATUS_STATIC_ROW_FIELDS = frozenset(
    {
        "artifact_fingerprint",
        "config_id",
        "configuration_class",
        "configuration_route_json",
        "created_at",
        "directory_label",
        "entities_json",
        "id",
        "lineage_fingerprint",
        "logical_key",
        "max_memory_gb",
        "memory_gb",
        "module",
        "module_lineage_id",
        "output_prefix",
        "output_root",
        "participant",
        "project",
        "resource_class",
        "revision_fingerprint",
        "scientific_revision",
        "scope",
        "work_item_key",
    }
)


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid scheduler {label}")
    return value


def prepare(control: Path) -> ControlPaths:
    """Create the shared exchange directories without opening the registry."""
    paths = ControlPaths(control)
    paths.require_current_layout()
    for path in (paths.service, paths.service_progress):
        ensure_shared_directory(path)
        try:
            path.chmod(0o2775)
        except PermissionError:
            pass
    return paths


def progress_path(control: Path, message_id: str) -> Path:
    """Return the transient progress path for one validated message ID."""
    message_id = _identifier(message_id, "message ID")
    return ControlPaths(control).service_progress / f"{message_id}.json"


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


def publish_progress(
    control: Path,
    message_id: str,
    *,
    phase: str,
    completed: int,
    total: int,
) -> None:
    """Publish the latest bounded progress record for a long operation."""
    if not phase or len(phase) > 160:
        raise ValueError("Scheduler progress phase must contain at most 160 characters")
    if completed < 0 or total < 0 or completed > total:
        raise ValueError("Invalid scheduler progress count")
    path = progress_path(control, message_id)
    atomic_write_json(
        path,
        {
            "protocol": PROTOCOL,
            "id": message_id,
            "phase": phase,
            "completed": int(completed),
            "total": int(total),
            "updated_at": time.time(),
        },
        sort_keys=True,
        mode=0o664,
    )


def read_progress(control: Path, message_id: str) -> dict[str, Any] | None:
    """Read the latest complete progress record for one message."""
    try:
        record = read_json(progress_path(control, message_id))
    except FileNotFoundError:
        return None
    required = {"protocol", "id", "phase", "completed", "total", "updated_at"}
    if (
        not isinstance(record, dict)
        or set(record) != required
        or record["protocol"] != PROTOCOL
        or record["id"] != message_id
    ):
        return None
    return record


def clear_progress(control: Path, message_id: str) -> None:
    """Remove a terminal operation's transient progress record."""
    progress_path(control, message_id).unlink(missing_ok=True)


def validate_message(record: Any, *, expected_id: str | None = None) -> dict[str, Any]:
    """Validate one durable or directly received scheduler record."""
    if (
        not isinstance(record, dict)
        or set(record) != {"protocol", "id", "kind", "created_at", "user", "host", "payload"}
        or record["protocol"] != PROTOCOL
        or record["kind"] not in {"command", "worker"}
        or not isinstance(record["payload"], dict)
        or (expected_id is not None and expected_id != record["id"])
    ):
        raise ValueError(f"Invalid scheduler message: {expected_id or '<direct>'}")
    _identifier(record["id"], "message ID")
    return record


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


def startup_progress_path(control: Path, token: str) -> Path:
    """Return the transient startup record for one launch token."""
    return ControlPaths(control).service / (
        f"startup-progress-{_identifier(token, 'launch token')}.json"
    )


def publish_startup_progress(control: Path, token: str, phase: str) -> None:
    """Expose scheduler initialization after its allocation has started."""
    if not phase or len(phase) > 160:
        raise ValueError("Scheduler startup phase must contain at most 160 characters")
    atomic_write_json(
        startup_progress_path(control, token),
        {
            "protocol": PROTOCOL,
            "token": token,
            "phase": phase,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated_at": time.time(),
        },
        sort_keys=True,
        mode=0o664,
    )


def read_startup_progress(control: Path, token: str) -> dict[str, Any] | None:
    """Read the current initialization phase for one launch token."""
    if not token:
        return None
    try:
        value = read_json(startup_progress_path(control, token))
    except (FileNotFoundError, ValueError, OSError):
        return None
    required = {"protocol", "token", "phase", "host", "pid", "updated_at"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["protocol"] != PROTOCOL
        or value["token"] != token
    ):
        return None
    return value


def clear_startup_progress(control: Path, token: str) -> None:
    """Remove one launch token's transient initialization record."""
    startup_progress_path(control, token).unlink(missing_ok=True)


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
    record = publish_active(
        control,
        token=token,
        generation=generation,
        job_id=os.environ.get("SLURM_JOB_ID") or str(claim.get("job_id") or ""),
        host=host,
        port=port,
    )
    clear_startup_progress(control, token)
    return record


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
    clear_startup_progress(control, token)


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
    time_hours: int = SCHEDULER_TIME_HOURS,
    memory_gb: int = SCHEDULER_MEMORY_GB,
    cpus: int = SCHEDULER_CPUS,
) -> Path:
    """Write the pinned Slurm script for one controller launch."""
    import shlex

    if time_hours < 1 or memory_gb < 1 or cpus < 1:
        raise ValueError("Scheduler time, memory, and CPUs must be positive")

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
        f"#SBATCH --time={time_hours}:00:00",
        f"#SBATCH --mem={memory_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
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


def _split_status_snapshot(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate stable graph data from frequently changing execution state."""
    static_branches = {}
    dynamic_branches = {}
    for name, report in value["branches"].items():
        static_rows = []
        dynamic_rows = []
        for row in report["rows"]:
            static_rows.append(
                {key: item for key, item in row.items() if key in _STATUS_STATIC_ROW_FIELDS}
            )
            dynamic_rows.append(
                {
                    key: item
                    for key, item in row.items()
                    if key == "id" or key not in _STATUS_STATIC_ROW_FIELDS
                }
            )
        static_branches[name] = {
            "rows": static_rows,
            "visible_ids": report["visible_ids"],
            "dependencies": report["dependencies"],
        }
        dynamic_branches[name] = {
            "rows": dynamic_rows,
            "ingestion": report["ingestion"],
        }
    return (
        {"branches": static_branches},
        {
            "generation": value["generation"],
            "workers": value["workers"],
            "submissions": value["submissions"],
            "branches": dynamic_branches,
        },
    )


def _write_status_json(path: Path, value: dict[str, Any]) -> None:
    """Durably publish compact JSON used only as a machine-facing read model."""
    atomic_write_text(
        path,
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
        mode=0o664,
        durable=True,
    )


def publish_snapshot(control: Path, value: dict[str, Any]) -> None:
    """Publish a coherent cached read model while reusing unchanged graph data."""
    if value.get("protocol") != PROTOCOL or not isinstance(value.get("branches"), dict):
        raise ValueError("Invalid scheduler status snapshot")
    paths = prepare(control)
    static, dynamic = _split_status_snapshot(value)
    static_fingerprint = fingerprint(static)
    static_name = f"status-static-{static_fingerprint}.json"
    dynamic_name = f"status-dynamic-{int(value['generation']) % 2}.json"
    static_path = paths.service / static_name
    if not static_path.is_file():
        _write_status_json(
            static_path,
            {
                "protocol": PROTOCOL,
                "format": _STATUS_FORMAT,
                "fingerprint": static_fingerprint,
                **static,
            },
        )
    _write_status_json(
        paths.service / dynamic_name,
        {"protocol": PROTOCOL, "format": _STATUS_FORMAT, **dynamic},
    )
    _write_status_json(
        paths.service_snapshot,
        {
            "protocol": PROTOCOL,
            "format": _STATUS_FORMAT,
            "generation": value["generation"],
            "published_at": value["published_at"],
            "service_active": value["service_active"],
            "static": static_name,
            "dynamic": dynamic_name,
        },
    )


def _status_component(service: Path, name: object, pattern: re.Pattern[str]) -> dict[str, Any]:
    """Read one status component after constraining it to the service directory."""
    if not isinstance(name, str) or pattern.fullmatch(name) is None:
        raise ValueError("Unsupported scheduler status component")
    return read_json(service / name)


def _assemble_status_snapshot(control: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Join one manifest's static graph and dynamic state in memory."""
    service = ControlPaths(control).service
    static = _status_component(service, manifest.get("static"), _STATUS_STATIC_FILE)
    dynamic = _status_component(service, manifest.get("dynamic"), _STATUS_DYNAMIC_FILE)
    if (
        static.get("protocol") != PROTOCOL
        or static.get("format") != _STATUS_FORMAT
        or dynamic.get("protocol") != PROTOCOL
        or dynamic.get("format") != _STATUS_FORMAT
        or dynamic.get("generation") != manifest.get("generation")
        or static.get("fingerprint")
        != str(manifest.get("static", "")).removeprefix("status-static-").removesuffix(".json")
        or not isinstance(static.get("branches"), dict)
        or not isinstance(dynamic.get("branches"), dict)
        or set(static["branches"]) != set(dynamic["branches"])
    ):
        raise ValueError("Unsupported scheduler status snapshot")
    branches = {}
    for name, stable in static["branches"].items():
        changing = dynamic["branches"][name]
        dynamic_rows = {int(row["id"]): row for row in changing["rows"]}
        rows = []
        for row in stable["rows"]:
            identifier = int(row["id"])
            if identifier not in dynamic_rows:
                raise ValueError("Scheduler status components describe different work items")
            rows.append({**row, **dynamic_rows.pop(identifier)})
        if dynamic_rows:
            raise ValueError("Scheduler status components describe different work items")
        branches[name] = {
            "rows": rows,
            "visible_ids": stable["visible_ids"],
            "ingestion": changing["ingestion"],
            "dependencies": stable["dependencies"],
        }
    return {
        "protocol": PROTOCOL,
        "generation": manifest["generation"],
        "published_at": manifest["published_at"],
        "service_active": manifest["service_active"],
        "workers": dynamic["workers"],
        "submissions": dynamic["submissions"],
        "branches": branches,
    }


def read_snapshot(control: Path) -> dict[str, Any] | None:
    """Read the last complete scheduler read model without opening SQLite."""
    path = ControlPaths(control).service_snapshot
    for _attempt in range(3):
        try:
            value = read_json(path)
        except FileNotFoundError:
            return None
        if value.get("protocol") != PROTOCOL:
            raise ValueError("Unsupported scheduler status snapshot")
        if isinstance(value.get("branches"), dict):
            return value
        if value.get("format") != _STATUS_FORMAT:
            raise ValueError("Unsupported scheduler status snapshot")
        try:
            return _assemble_status_snapshot(control, value)
        except (FileNotFoundError, ValueError):
            # A second publication may have advanced through both alternating
            # dynamic slots after this reader loaded its manifest.
            continue
    raise ValueError("Scheduler status components changed repeatedly while being read")


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
    """Remove old progress records and controller files."""
    paths = ControlPaths(control)
    cutoff = time.time() - age_seconds
    try:
        current_static = read_json(paths.service_snapshot).get("static")
    except (FileNotFoundError, ValueError, OSError):
        current_static = None
    for path in (
        *paths.service_progress.glob("*.json"),
        *paths.service.glob("controller-*.sbatch"),
        *paths.service.glob("controller-local-*.log"),
        *paths.service.glob("startup-error-*.json"),
        *paths.service.glob("startup-progress-*.json"),
        *(
            path
            for path in paths.service.glob("status-static-*.json")
            if path.name != current_static
        ),
    ):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            pass
