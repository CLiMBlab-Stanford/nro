"""Coordinate through a live scheduler or a fenced one-shot process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nro.orchestration.execution_cache import cache_lock
from nro.orchestration.scheduler_implementation import (
    capture_worker_implementation,
    implementation_path,
)

DEFAULT_RPC_TIMEOUT_SECONDS = 180.0
UPDATED_STATUS_TIMEOUT_SECONDS = 600.0
CONTROL_RPC_TIMEOUT_SECONDS = 60.0
MAINTENANCE_RPC_TIMEOUT_SECONDS = 900.0
_WAIT_FRAMES = ("·", "•", "●", "•")
_WAIT_COLORS = ("\x1b[95m", "\x1b[94m", "\x1b[96m", "\x1b[92m", "\x1b[93m")
_RESET = "\x1b[0m"
_CLEAR = "\r\x1b[2K"


class SchedulerError(RuntimeError):
    """Report a scheduler transport or service failure."""

    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        """Preserve the remote error category with its user-facing message."""
        super().__init__(message)
        self.error_type = error_type


def _response_result(response: dict):
    """Return a scheduler result or raise its typed service error."""
    if "error" in response:
        raise SchedulerError(
            str(response["error"]),
            error_type=response.get("error_type"),
        )
    if "result" not in response:
        raise SchedulerError("Central scheduler returned a malformed response")
    return response["result"]


@dataclass(frozen=True)
class SchedulerEndpoint:
    """Pin the control store and implementation used to launch its service."""

    control: Path
    bids_root: Path
    source: object
    site: Path
    python: Path
    maintenance: bool = False


def command(
    control: Path,
    bids_root: Path,
    *,
    allow_changed_checkout: bool = False,
    maintenance_checkout: Path | None = None,
) -> SchedulerEndpoint:
    """Capture the approved implementation used for scheduler coordination."""
    control = Path(control).expanduser().resolve()
    bids_root = Path(bids_root).expanduser().resolve()
    if not implementation_path(control).is_file():
        raise ValueError("Activate an approved main scheduler before submitting branch work")
    if maintenance_checkout is not None:
        from nro.orchestration.scheduler_implementation import (
            capture_maintenance_implementation,
        )

        source, site, python = capture_maintenance_implementation(
            control, bids_root, maintenance_checkout
        )
    else:
        source, site, python = capture_worker_implementation(
            control, bids_root, check_checkout=not allow_changed_checkout
        )
    return SchedulerEndpoint(
        control,
        bids_root,
        source,
        site,
        python,
        maintenance=maintenance_checkout is not None,
    )


def _wait_notice(frame: int, message: str) -> bool:
    if not sys.stderr.isatty():
        return False
    marker = _WAIT_FRAMES[frame % len(_WAIT_FRAMES)]
    if "NO_COLOR" not in os.environ:
        marker = f"{_WAIT_COLORS[frame % len(_WAIT_COLORS)]}{marker}{_RESET}"
    sys.stderr.write(f"{_CLEAR}{marker} {message}")
    sys.stderr.flush()
    return True


def _progress_notice(record: dict | None, fallback: str) -> str:
    """Render one bounded progress record for the interactive wait line."""
    if record is None:
        return fallback
    phase = str(record["phase"])
    completed = int(record["completed"])
    total = int(record["total"])
    return f"{phase}: {completed:,}/{total:,}..." if total else f"{phase}..."


def _start_service(endpoint: SchedulerEndpoint) -> str | None:
    """Submit one controller when this caller wins the atomic launch claim."""
    from nro.configuration.site import settings
    from nro.orchestration.scheduler_bus import (
        claim_launch,
        release_launch,
        submit_controller,
        update_launch_job,
        write_controller_script,
    )

    claim = claim_launch(endpoint.control)
    if claim is None:
        return None
    try:
        values = settings(path=endpoint.site)[0]
        run_locally = os.environ.get("NRO_SCHEDULER_LOCAL") == "1"
        if run_locally:
            from nro.orchestration.control_paths import ControlPaths

            local_command = endpoint.source.command(
                (
                    str(endpoint.python),
                    "-m",
                    "nro.orchestration.scheduler_service",
                    "--serve",
                    "--launch-token",
                    claim.token,
                    "--bids-root",
                    str(endpoint.bids_root),
                    "--idle-grace",
                    "1",
                ),
                site=endpoint.site,
            )
            environment = {
                **os.environ,
                "NRO_PROCESS_ROLE": "scheduler",
            }
            environment.pop("SLURM_JOB_ID", None)
            log = ControlPaths(endpoint.control).service / f"controller-local-{claim.token}.log"
            with log.open("ab") as stream:
                process = subprocess.Popen(
                    local_command,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
            job_id = f"local-{process.pid}"
        else:
            script = write_controller_script(
                endpoint.control,
                bids_root=endpoint.bids_root,
                token=claim.token,
                source=endpoint.source,
                site=endpoint.site,
                python=endpoint.python,
                partition=values["partition"],
                account=values.get("account") or None,
            )
            job_id = submit_controller(script)
        update_launch_job(claim, job_id)
        return job_id
    except BaseException:
        release_launch(endpoint.control, claim.token)
        raise


def _run_once(endpoint: SchedulerEndpoint, record: dict, *, durable: bool = True) -> dict | None:
    """Run one retryable request through a fenced local coordinator."""
    from nro.orchestration.scheduler_bus import (
        claim_launch,
        read_progress,
        release_launch,
        update_launch_job,
    )

    claim = claim_launch(endpoint.control)
    if claim is None:
        return None
    command = endpoint.source.command(
        (
            str(endpoint.python),
            "-m",
            "nro.orchestration.scheduler_service",
            "--once",
            "--launch-token",
            claim.token,
            "--bids-root",
            str(endpoint.bids_root),
        ),
        site=endpoint.site,
    )
    environment = {**os.environ, "NRO_PROCESS_ROLE": "scheduler"}
    if endpoint.maintenance:
        environment["NRO_SCHEDULER_MAINTENANCE"] = "1"
    else:
        environment.pop("NRO_SCHEDULER_MAINTENANCE", None)
    environment.pop("SLURM_JOB_ID", None)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        update_launch_job(claim, f"local-{process.pid}")
        assert process.stdin is not None
        process.stdin.write(
            json.dumps(
                {"durable": durable, "record": record},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        process.stdin.close()
        process.stdin = None
        operation = str(record["payload"].get("operation") or "request")
        count = len(record["payload"].get("plan", ())) if operation == "purge" else 0
        suffix = f" for {count:,} work items" if count else ""
        notice_text = f"Applying {operation}{suffix}..."
        started = time.monotonic()
        frame = 0
        notice = False
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= 0.75:
                        progress = read_progress(endpoint.control, str(record["id"]))
                        notice = (
                            _wait_notice(frame, _progress_notice(progress, notice_text)) or notice
                        )
                        frame += 1
        finally:
            if notice:
                sys.stderr.write(_CLEAR)
                sys.stderr.flush()
        if process.returncode:
            detail = stderr.strip() or stdout.strip() or f"status {process.returncode}"
            raise SchedulerError(f"One-shot scheduler update failed: {detail}")
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise SchedulerError("One-shot scheduler returned an invalid response") from error
        if not isinstance(response, dict):
            raise SchedulerError("One-shot scheduler returned an invalid response")
        return response
    except BaseException:
        release_launch(endpoint.control, claim.token)
        raise


def _ensure_coordinator(
    endpoint: SchedulerEndpoint,
    *,
    require_service: bool,
    start_epoch: bool = False,
) -> bool:
    """Provide a live service or a fenced one-shot coordinator as requested."""
    from nro.orchestration.scheduler_bus import clear_shutdown, read_active, shutdown_pending

    if start_epoch:
        clear_shutdown(endpoint.control)
    elif require_service and shutdown_pending(endpoint.control):
        return False
    if read_active(endpoint.control) is None:
        if require_service:
            _start_service(endpoint)
        else:
            return False
    return True


def exchange(
    endpoint: SchedulerEndpoint,
    message: dict,
    *,
    descriptors: tuple[int, ...] = (),
    timeout: float | None = DEFAULT_RPC_TIMEOUT_SECONDS,
    require_service: bool = False,
    start_epoch: bool = False,
    durable: bool = True,
) -> dict:
    """Send one request directly, with a recovery record when required."""
    if descriptors:
        raise ValueError("The durable scheduler transport does not accept file descriptors")
    from nro.orchestration.scheduler_bus import (
        create_message,
        read_active,
        read_launch,
        read_progress,
        read_startup_error,
    )

    kind = "worker" if message.get("operation") == "worker" else "command"
    record = create_message(message, kind=kind)
    message_id = str(record["id"])
    try:
        service_available = _ensure_coordinator(
            endpoint,
            require_service=require_service,
            start_epoch=start_epoch,
        )
        if not service_available and not require_service:
            response = _run_once(endpoint, record, durable=durable)
            if response is not None:
                return _response_result(response)
    except Exception as error:
        raise SchedulerError(f"Could not start scheduler coordination: {error}") from error
    started = time.monotonic()
    last_recovery_check = started
    active_token: str | None = None
    active_started: float | None = None
    frame = 0
    notice = False
    attempted_endpoint: tuple[str, int] | None = None
    while True:
        launch = read_launch(endpoint.control)
        if launch is not None:
            startup_error = read_startup_error(endpoint.control, str(launch.get("token") or ""))
            if startup_error is not None:
                if notice:
                    sys.stderr.write(_CLEAR)
                    sys.stderr.flush()
                raise SchedulerError(
                    "Central scheduler could not start: " + str(startup_error["error"])
                )
        now = time.monotonic()
        active = read_active(endpoint.control)
        if active is not None:
            token = str(active["token"])
            if token != active_token:
                active_token = token
                active_started = now
            endpoint_identity = (
                str(active["token"]),
                int(active["port"]),
            )
            if endpoint_identity != attempted_endpoint:
                from nro.orchestration.scheduler_rpc import request

                attempted_endpoint = endpoint_identity
                response = None
                try:
                    direct_timeout = 3600.0 if timeout is None else max(1.0, timeout)
                    response = request(
                        active,
                        record,
                        timeout=direct_timeout,
                        durable=durable,
                    )
                except (ConnectionError, OSError, TimeoutError, ValueError):
                    response = None
                if response is not None:
                    if response.get("error") == "Scheduler endpoint is obsolete":
                        response = None
                        continue
                    if response.get("pending") == message_id:
                        attempted_endpoint = None
                        time.sleep(0.1)
                        continue
                    if notice:
                        sys.stderr.write(_CLEAR)
                        sys.stderr.flush()
                    return _response_result(response)
        elif active_token is not None:
            launch_token = str(launch.get("token") or "") if launch is not None else ""
            if launch_token and launch_token != active_token:
                active_token = None
                active_started = None

        job_id = str(launch.get("job_id") or "") if launch is not None else ""
        waiting_for_slurm = (
            active_started is None and bool(job_id) and not job_id.startswith("local-")
        )
        if active_started is not None:
            timed_elapsed = now - active_started
        elif waiting_for_slurm:
            timed_elapsed = None
        else:
            timed_elapsed = now - started
        if timeout is not None and timed_elapsed is not None and timed_elapsed >= timeout:
            if notice:
                sys.stderr.write(_CLEAR)
                sys.stderr.flush()
            raise SchedulerError(
                f"Central scheduler did not respond within {timeout:g} seconds; "
                f"request {message_id} remains recorded"
            )
        if now - last_recovery_check >= 5.0:
            service_available = _ensure_coordinator(
                endpoint,
                require_service=require_service,
                start_epoch=False,
            )
            if not service_available and not require_service:
                response = _run_once(endpoint, record, durable=durable)
                if response is not None:
                    if notice:
                        sys.stderr.write(_CLEAR)
                        sys.stderr.flush()
                    return _response_result(response)
            last_recovery_check = now
            attempted_endpoint = None
        elapsed = now - started
        if elapsed >= 0.75:
            progress = read_progress(endpoint.control, message_id) if durable else None
            fallback = (
                "Waiting for the scheduler allocation..."
                if waiting_for_slurm
                else "Waiting for the scheduler..."
            )
            message_text = _progress_notice(progress, fallback)
            notice = _wait_notice(frame, message_text) or notice
            frame += 1
        time.sleep(0.1 if elapsed < 2 else 0.5)


def _cached_status(control: Path, checkout: Path) -> dict:
    """Select one branch report from the last atomic read model."""
    from nro.orchestration.branch_store import BranchStore
    from nro.orchestration.scheduler_bus import read_snapshot

    snapshot = read_snapshot(control)
    if snapshot is None:
        return {"rows": [], "visible_ids": [], "ingestion": [], "dependencies": []}
    name = BranchStore(control).read().topology.registered_checkout(checkout)
    return snapshot["branches"].get(
        name, {"rows": [], "visible_ids": [], "ingestion": [], "dependencies": []}
    )


def _endpoint(
    control: Path,
    bids_root: Path,
    *,
    allow_changed_checkout: bool = False,
    maintenance_checkout: Path | None = None,
) -> SchedulerEndpoint:
    with cache_lock(control):
        return command(
            control,
            bids_root,
            allow_changed_checkout=allow_changed_checkout,
            maintenance_checkout=maintenance_checkout,
        )


def supply(
    control: Path, bids_root: Path, *, checkout: Path, request_ids: list[str], options: dict
) -> dict:
    """Start central workers after admission has published execution pins."""
    endpoint = _endpoint(control, bids_root)
    if options.get("no_submit", False):
        return exchange(
            endpoint,
            dict(
                operation="supply",
                checkout=str(checkout),
                request_ids=request_ids,
                options=options,
            ),
        )
    preflight = exchange(
        endpoint,
        dict(
            operation="supply_needed",
            checkout=str(checkout),
            request_ids=request_ids,
            options=options,
        ),
    )
    if not preflight["needed"]:
        return {"submitted_workers": []}
    result = exchange(
        endpoint,
        dict(operation="supply", checkout=str(checkout), request_ids=request_ids, options=options),
        timeout=None if options["local"] else DEFAULT_RPC_TIMEOUT_SECONDS,
        require_service=True,
        start_epoch=True,
    )
    if options["local"]:
        _wait_for_local_workers(Path(control), result["submitted_workers"])
    return result


def _wait_for_local_workers(control: Path, worker_ids: list[str]) -> None:
    """Retain foreground semantics while the service remains free to coordinate."""
    from nro.orchestration.scheduler_bus import read_snapshot

    pending = set(worker_ids)
    seen: set[str] = set()
    started = time.monotonic()
    while pending:
        snapshot = read_snapshot(control) or {}
        states = {str(row["id"]): str(row["state"]) for row in snapshot.get("workers", ())}
        seen.update(pending.intersection(states))
        failed = {
            worker_id: states[worker_id]
            for worker_id in pending
            if states.get(worker_id) in {"lost", "terminated"}
        }
        if failed:
            details = ", ".join(f"{worker} ({state})" for worker, state in failed.items())
            raise SchedulerError(f"Local scheduler worker failed: {details}")
        pending = {
            worker_id
            for worker_id in pending
            if states.get(worker_id) not in {"exited", "terminated", "lost"}
        }
        if pending:
            if time.monotonic() - started >= 300:
                raise SchedulerError("Local scheduler worker did not finish within 300 seconds")
            time.sleep(0.1)
    if worker_ids and not seen:
        raise SchedulerError("Local scheduler worker exited before registration")


def status(control: Path, bids_root: Path, *, checkout: Path, mode: str) -> dict:
    """Read the cached snapshot or request an authoritative update barrier."""
    if mode == "cached":
        return _cached_status(control, checkout)
    if mode != "verify":
        raise ValueError("Unknown status mode")
    return exchange(
        _endpoint(control, bids_root),
        dict(operation="status", checkout=str(checkout), mode=mode),
        timeout=UPDATED_STATUS_TIMEOUT_SECONDS,
        durable=False,
    )


def stop(control: Path, bids_root: Path, *, checkout: Path, project: str, selection: dict) -> dict:
    """Cancel matching demand through the branch-scoped service operation."""
    return exchange(
        # Cancellation must remain available while a shared checkout is ahead
        # of its active release.  The endpoint still uses the pinned active
        # implementation; only the otherwise-required checkout HEAD match is
        # relaxed so users can quiesce work before installation maintenance.
        _endpoint(control, bids_root, allow_changed_checkout=True),
        dict(operation="stop", checkout=str(checkout), project=project, selection=selection),
        timeout=CONTROL_RPC_TIMEOUT_SECONDS,
    )


def cancel_requests(
    control: Path,
    bids_root: Path,
    *,
    checkout: Path,
    requests: Sequence[tuple[str, str]],
) -> dict[str, int]:
    """Withdraw exact requests created by an interrupted run invocation."""
    grouped: dict[str, list[str]] = {}
    for project, request_id in requests:
        grouped.setdefault(project, []).append(request_id)
    totals = {"work_items": 0, "requests": 0, "attempts": 0}
    for project, request_ids in grouped.items():
        result = stop(
            control,
            bids_root,
            checkout=checkout,
            project=project,
            selection={
                "participants": (),
                "modules": (),
                "workflows": (),
                "lineages": (),
                "selectors": {},
                "include_dependents": False,
                "request_ids": request_ids,
            },
        )
        for key in totals:
            totals[key] += int(result[key])
    return totals


def logs(
    control: Path,
    bids_root: Path,
    *,
    checkout: Path,
    selection: dict,
    worker_level: bool,
    running_only: bool = False,
) -> dict:
    """Resolve logs from the cached read model without starting a service."""
    from nro.engine.cli import matches_module_lineage, matches_work_item_selectors
    from nro.orchestration.control_paths import ControlPaths

    report = _cached_status(control, checkout)
    requested_modules = set(selection["modules"])
    scientific_modules = requested_modules - {"bidsify"}
    scientific_selected = not requested_modules or bool(scientific_modules)
    visible = set(report["visible_ids"])
    selected = [
        row
        for row in report["rows"]
        if scientific_selected
        and row["id"] in visible
        and (not selection["projects"] or row["project"] in selection["projects"])
        and (not selection["participants"] or row["participant"] in selection["participants"])
        and (not scientific_modules or row["module"] in scientific_modules)
        and matches_module_lineage(
            row["module"], row.get("directory_label", ""), selection.get("lineages", ())
        )
        and (
            not selection["workflows"]
            or set(selection["workflows"]).intersection(
                str(row.get("workflow_ids") or "").split(",")
            )
        )
        and matches_work_item_selectors(json.loads(row["entities_json"]), selection["selectors"])
        and (not running_only or row.get("status") == "Running")
    ]
    if worker_level:
        paths = [row["worker_log_path"] for row in selected if row.get("worker_log_path")]
    else:
        paths = [row["log_path"] for row in selected if row.get("log_path")]
    if (
        "bidsify" in requested_modules
        and not selection["workflows"]
        and not selection.get("lineages")
    ):
        sessions = selection["selectors"].get("ses", ())
        control_paths = ControlPaths(control)
        paths.extend(
            str(
                (
                    control_paths.ingestion
                    if row["branch"] == "main"
                    else control_paths.branch(row["branch"]) / "ingestion"
                )
                / f"{row['id']}.log"
            )
            for row in report.get("ingestion", [])
            if (
                not selection.get("ingestion_projects")
                or row["project"] in selection["ingestion_projects"]
            )
            and (not selection["participants"] or row["participant"] in selection["participants"])
            and (not sessions or row["session"] in sessions)
            and row.get("branch")
            and (not running_only or row.get("state") == "running")
        )
    return {"paths": sorted(set(paths))}


def pool_operation(
    control: Path, bids_root: Path, *, checkout: Path, operation: str, concurrency=None
) -> dict:
    """Apply a global pool control through the service."""
    return exchange(
        _endpoint(
            control,
            bids_root,
            allow_changed_checkout=operation == "stop_workers",
        ),
        dict(operation=operation, checkout=str(checkout), concurrency=concurrency),
        timeout=CONTROL_RPC_TIMEOUT_SECONDS,
    )


def shutdown_service(
    control: Path,
    bids_root: Path,
    *,
    checkout: Path,
    allow_changed_checkout: bool = False,
) -> dict:
    """Record shutdown intent and stop the current controller after its response."""
    from nro.orchestration.scheduler_bus import read_active

    if read_active(control) is None:
        return {"stopping": False}
    return exchange(
        _endpoint(
            control,
            bids_root,
            allow_changed_checkout=allow_changed_checkout,
            maintenance_checkout=checkout if allow_changed_checkout else None,
        ),
        {"operation": "server_shutdown", "checkout": str(checkout)},
        timeout=CONTROL_RPC_TIMEOUT_SECONDS,
    )


def maintenance(
    control: Path, bids_root: Path, *, checkout: Path, operation: str, **fields
) -> dict:
    """Run one scoped maintenance operation through the service."""
    if operation not in {
        "purge_snapshot",
        "purge",
        "gc",
        "cache",
        "repair_prepare",
        "repair_finish",
        "promotion_preview",
        "promotion_publish",
        "publish",
        "branch_update",
        "environment_idle",
        "installation_activity",
        "installation_prepare",
        "installation_progress",
    }:
        raise ValueError("Unsupported maintenance operation")
    timeout = (
        None
        if operation in {"purge", "gc", "promotion_publish", "publish"}
        else MAINTENANCE_RPC_TIMEOUT_SECONDS
    )
    return exchange(
        _endpoint(
            control,
            bids_root,
            allow_changed_checkout=operation.startswith("installation_"),
            maintenance_checkout=checkout if operation.startswith("installation_") else None,
        ),
        dict(operation=operation, checkout=str(checkout), **fields),
        timeout=timeout,
    )
