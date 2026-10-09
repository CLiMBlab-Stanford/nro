"""Launch and communicate with the persistent planning broker."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from nro.orchestration.planner_bus import (
    claim_launch,
    read_active,
    read_launch,
    release_launch,
    submit,
    update_launch_job,
    write_script,
)
from nro.orchestration.scheduler_bus import PROTOCOL, _launch_abandoned
from nro.orchestration.scheduler_rpc import receive, send


def ensure(endpoint) -> str | None:
    """Submit one planner when neither a live endpoint nor launch exists."""
    if read_active(endpoint.control) is not None:
        return None
    claim = claim_launch(endpoint.control)
    if claim is None:
        return None
    try:
        from nro.site.configuration import settings

        values = settings(path=endpoint.site)[0]
        if os.environ.get("NRO_SCHEDULER_LOCAL") == "1":
            command = endpoint.source.command(
                (
                    str(endpoint.python),
                    "-m",
                    "nro.orchestration.planner_service",
                    "--control",
                    str(endpoint.control),
                    "--launch-token",
                    claim.token,
                    "--idle-grace",
                    "60",
                ),
                site=endpoint.site,
            )
            environment = {**os.environ, "NRO_PROCESS_ROLE": "planner"}
            environment.pop("SLURM_JOB_ID", None)
            log = Path(endpoint.control) / "shared/planner" / f"planner-local-{claim.token}.log"
            with log.open("ab") as stream:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
            job_id = f"local-{process.pid}"
        else:
            script = write_script(
                endpoint.control,
                token=claim.token,
                source=endpoint.source,
                site=endpoint.site,
                python=endpoint.python,
                partition=values["partition"],
                account=values.get("account") or None,
                time_hours=values["planner_time"],
                memory_gb=values["planner_memory"],
                cpus=values["planner_cpus"],
            )
            job_id = submit(script)
        update_launch_job(claim, job_id)
        return job_id
    except BaseException:
        release_launch(endpoint.control, claim.token)
        raise


def _response(response: dict) -> dict:
    if "error" in response:
        raise RuntimeError(str(response["error"]))
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Planner returned an invalid response")
    return result


def execute(
    control: Path,
    request: dict,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Wait for the claimed planner and execute one FIFO planning request."""
    phase = None

    def report(value: str) -> None:
        nonlocal phase
        if progress is not None and value != phase:
            progress(value)
        phase = value

    while True:
        active = read_active(control)
        if active is not None:
            try:
                report("Planning requested work")
                with socket.create_connection(
                    (str(active["host"]), int(active["port"])), timeout=10.0
                ) as connection:
                    connection.settimeout(None)
                    send(
                        connection,
                        {
                            "protocol": PROTOCOL,
                            "token": str(active["token"]),
                            "message": {"operation": "plan", "request": request},
                        },
                    )
                    return _response(receive(connection))
            except (ConnectionError, OSError, TimeoutError):
                pass
        launch = read_launch(control)
        if launch is None or _launch_abandoned(launch):
            raise RuntimeError("Planning service ended before accepting the request")
        report("Waiting for planner allocation")
        time.sleep(1.0)


def shutdown(control: Path) -> dict:
    """Stop a live planner or cancel its not-yet-active allocation."""
    active = read_active(control)
    if active is not None:
        try:
            with socket.create_connection(
                (str(active["host"]), int(active["port"])), timeout=10.0
            ) as connection:
                connection.settimeout(30.0)
                send(
                    connection,
                    {
                        "protocol": PROTOCOL,
                        "token": str(active["token"]),
                        "message": {"operation": "shutdown"},
                    },
                )
                return _response(receive(connection))
        except (ConnectionError, OSError, TimeoutError):
            pass
    launch = read_launch(control)
    if launch is None:
        return {"stopping": False}
    token = str(launch.get("token") or "")
    job_id = str(launch.get("job_id") or "")
    release_launch(control, token)
    if job_id.startswith("local-") and launch.get("host") == socket.gethostname():
        try:
            os.kill(int(job_id.removeprefix("local-")), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError, ValueError):
            pass
    elif job_id:
        subprocess.run(
            ["scancel", job_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return {"stopping": bool(job_id)}
