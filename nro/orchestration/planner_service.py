"""Serve branch-specific planning through one persistent FIFO broker."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nro.orchestration.planner_bus import (
    PLANNER_IDLE_GRACE_SECONDS,
    activate,
    deactivate,
    publish_active,
    read_active,
)
from nro.orchestration.scheduler_bus import HEARTBEAT_SECONDS, PROTOCOL
from nro.orchestration.scheduler_rpc import receive, send
from nro.orchestration.source_snapshots import SourceSnapshot


def _run(message: dict, stop: threading.Event | None = None) -> dict:
    """Execute one pinned `nro run` planner and return its structured result."""
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
    environment = {
        **os.environ,
        "NRO_REMOTE_PLANNER": "1",
        "NRO_CHECKOUT": str(checkout),
        "NRO_PROCESS_ROLE": "planner",
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
                raise RuntimeError("Planning stopped with the orchestration services")
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


def serve(*, control: Path, launch_token: str, idle_grace: float) -> int:
    """Accept one planning request at a time until the idle grace expires."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", 0))
    listener.listen(32)
    listener.settimeout(1.0)
    host = socket.getfqdn()
    port = int(listener.getsockname()[1])
    activate(control, launch_token, host=host, port=port)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            active = read_active(control)
            if active is None or active.get("token") != launch_token:
                stop.set()
                return
            publish_active(
                control,
                token=launch_token,
                job_id=str(active.get("job_id") or ""),
                host=host,
                port=port,
            )

    thread = threading.Thread(target=heartbeat, name="planner-heartbeat", daemon=True)
    thread.start()
    last_activity = time.monotonic()
    planning = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner-work")
    connections = ThreadPoolExecutor(max_workers=8, thread_name_prefix="planner-rpc")

    def handle(connection: socket.socket) -> None:
        nonlocal last_activity
        with connection:
            connection.settimeout(None)
            try:
                envelope = receive(connection)
                if (
                    envelope.get("protocol") != PROTOCOL
                    or envelope.get("token") != launch_token
                    or not isinstance(envelope.get("message"), dict)
                ):
                    raise ValueError("Planner endpoint is obsolete")
                message = envelope["message"]
                if message.get("operation") == "shutdown":
                    stop.set()
                    response = {"result": {"stopping": True}}
                elif message.get("operation") == "plan":
                    response = {"result": planning.submit(_run, message["request"], stop).result()}
                else:
                    raise ValueError("Unsupported planner operation")
            except (ValueError, RuntimeError, OSError) as error:
                response = {"error": str(error), "error_type": type(error).__name__}
            try:
                send(connection, response)
            except (ConnectionError, OSError):
                pass
        last_activity = time.monotonic()

    try:
        while not stop.is_set() and time.monotonic() - last_activity < idle_grace:
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            connections.submit(handle, connection)
        return 0
    finally:
        stop.set()
        listener.close()
        planning.shutdown(wait=True, cancel_futures=True)
        connections.shutdown(wait=True, cancel_futures=True)
        thread.join(timeout=HEARTBEAT_SECONDS + 1)
        deactivate(control, launch_token)


def build_parser() -> argparse.ArgumentParser:
    """Build the internal planner-service parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument("--idle-grace", type=float, default=PLANNER_IDLE_GRACE_SECONDS)
    return parser


def main() -> None:
    """Run the planner broker."""
    args = build_parser().parse_args()
    raise SystemExit(
        serve(
            control=args.control.resolve(),
            launch_token=args.launch_token,
            idle_grace=args.idle_grace,
        )
    )


if __name__ == "__main__":
    main()
