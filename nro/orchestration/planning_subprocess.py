"""Run pinned scientific planning without importing branch code into the scheduler."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from nro.orchestration.scheduler_requests import RetryableSchedulerRequest
from nro.orchestration.source_snapshots import SourceSnapshot


def execute(
    message: dict,
    *,
    stop: threading.Event | None = None,
    threads: int = 2,
) -> dict:
    """Execute one pinned planner child and return its detached admission plan."""
    required = {"source", "site", "python", "checkout", "argv"}
    if set(message) != required or not isinstance(message["argv"], list):
        raise ValueError("Invalid planner request")
    descriptor = message["source"]
    if not isinstance(descriptor, dict) or set(descriptor) != {"root", "digest"}:
        raise ValueError("Invalid planner source descriptor")
    source = SourceSnapshot(Path(descriptor["root"]), str(descriptor["digest"]))
    source.verify_manifest()
    site = Path(message["site"])
    python = Path(message["python"])
    checkout = Path(message["checkout"])
    if not all(path.is_absolute() for path in (site, python, checkout)):
        raise ValueError("Planner execution paths must be absolute")
    argv = [str(value) for value in message["argv"] if str(value) != "--json"]
    command = source.command(
        (str(python), "-m", "nro.bin.run", *argv, "--json"),
        site=site,
    )
    thread_limit = str(max(1, min(2, int(threads))))
    environment = {
        **os.environ,
        "NRO_REMOTE_PLANNER": "1",
        "NRO_PLANNER_SOURCE_ROOT": str(source.root),
        "NRO_PLANNER_SITE": str(site),
        "NRO_CHECKOUT": str(checkout),
        "NRO_PROCESS_ROLE": "planner",
        "OMP_NUM_THREADS": thread_limit,
        "OPENBLAS_NUM_THREADS": thread_limit,
        "MKL_NUM_THREADS": thread_limit,
        "NUMEXPR_NUM_THREADS": thread_limit,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": thread_limit,
    }
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                raise RetryableSchedulerRequest("Planning stopped with the scheduler")
    if process.returncode:
        detail = stderr.strip() or stdout.strip() or f"status {process.returncode}"
        raise RuntimeError("Planning failed: " + detail[-20_000:])
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Planner returned an invalid response") from error
    if not isinstance(value, dict):
        raise RuntimeError("Planner returned an invalid response")
    return value
