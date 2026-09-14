"""Launch and reuse a user-specific Slurm allocation for Workbench viewers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from nro.engine.io import atomic_write_json, read_json

PROTOCOL = 1
VIEWER_HOURS = 12
VIEWER_MEMORY_GB = 32
VIEWER_CPUS = 2
_MAX_MESSAGE_BYTES = 64 * 1024


def state_directory(control: Path) -> Path:
    """Return the private shared directory for the current user's viewer broker."""
    path = Path(control).expanduser().absolute() / "viewers" / f"uid-{os.getuid()}"
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


@contextmanager
def _launch_lock(root: Path) -> Iterator[None]:
    lock = root / "launch.lock"
    with lock.open("a+", encoding="utf-8") as stream:
        lock.chmod(0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _stored_active(root: Path) -> dict | None:
    try:
        return read_json(root / "active.json")
    except (FileNotFoundError, OSError, ValueError):
        return None


def _active(root: Path) -> dict | None:
    value = _stored_active(root)
    if value is None:
        return None
    required = {"protocol", "token", "host", "port", "pid", "job_id", "started_at"}
    if set(value) != required or value.get("protocol") != PROTOCOL:
        return None
    if not isinstance(value.get("token"), str) or not value["token"]:
        return None
    if not isinstance(value.get("host"), str) or not value["host"]:
        return None
    if type(value.get("port")) is not int or not 0 < value["port"] < 65536:
        return None
    return value


def _exchange(active: dict, payload: dict, *, timeout: float = 5.0) -> dict:
    request = (
        json.dumps(
            {"protocol": PROTOCOL, "token": active["token"], **payload}, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    with socket.create_connection((active["host"], active["port"]), timeout=timeout) as stream:
        stream.sendall(request)
        stream.settimeout(timeout)
        response = b""
        while not response.endswith(b"\n"):
            block = stream.recv(4096)
            if not block:
                break
            response += block
            if len(response) > _MAX_MESSAGE_BYTES:
                raise RuntimeError("Viewer broker returned an oversized response")
    try:
        value = json.loads(response)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Viewer broker returned an invalid response") from error
    if not isinstance(value, dict) or type(value.get("ok")) is not bool:
        raise RuntimeError("Viewer broker returned an invalid response")
    return value


def _live_active(root: Path) -> dict | None:
    active = _active(root)
    if active is None:
        return None
    for _attempt in range(2):
        try:
            response = _exchange(active, {"operation": "ping"}, timeout=10)
        except (OSError, RuntimeError):
            time.sleep(0.25)
            continue
        return active if response.get("ok") else None
    return None


def _cancel_stale_active(root: Path) -> None:
    active = _stored_active(root)
    if active is None:
        (root / "active.json").unlink(missing_ok=True)
        return
    job_id = str(active.get("job_id") or "")
    if job_id.isdigit():
        subprocess.run(
            ["scancel", job_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    token = active.get("token")
    if isinstance(token, str):
        _remove_owned_active(root, token)
    else:
        (root / "active.json").unlink(missing_ok=True)


def _remove_owned_active(root: Path, token: str) -> None:
    path = root / "active.json"
    try:
        value = read_json(path)
    except (FileNotFoundError, OSError, ValueError):
        return
    if value.get("token") == token:
        path.unlink(missing_ok=True)


def _launch(
    root: Path,
    *,
    viewer: Path,
    partition: str,
    account: str | None,
) -> tuple[subprocess.Popen, str, Path]:
    launcher = shutil.which("srun")
    if launcher is None:
        raise ValueError("Cannot open the scene through Slurm because srun is unavailable")
    if not os.environ.get("DISPLAY"):
        raise ValueError(
            "X11 forwarding is unavailable because DISPLAY is not set; reconnect with SSH X forwarding"
        )
    token = uuid.uuid4().hex
    log = root / f"broker-{token}.log"
    command = [
        launcher,
        "--x11",
        f"--partition={partition}",
        "--job-name=nro-view-server",
        "--ntasks=1",
        f"--cpus-per-task={VIEWER_CPUS}",
        f"--mem={VIEWER_MEMORY_GB}G",
        f"--time={VIEWER_HOURS}:00:00",
    ]
    if account:
        command.append(f"--account={account}")
    command.extend(
        (
            sys.executable,
            "-m",
            "nro.engine.viewer_broker",
            "--serve",
            "--state-dir",
            str(root),
            "--token",
            token,
            "--viewer",
            str(viewer),
        )
    )
    log_stream = log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_stream.close()
    return process, token, log


def _await_start(process: subprocess.Popen, root: Path, token: str, log: Path) -> dict:
    announced = False
    try:
        while True:
            active = _active(root)
            if active is not None and active.get("token") == token:
                try:
                    if _exchange(active, {"operation": "ping"}).get("ok"):
                        if announced:
                            print("Viewer allocation is ready.", file=sys.stderr)
                        return active
                except (OSError, RuntimeError):
                    pass
            status = process.poll()
            if status is not None:
                raise ValueError(f"Slurm viewer broker exited with status {status}; see {log}")
            if not announced:
                print("Waiting for the viewer allocation...", file=sys.stderr)
                announced = True
            time.sleep(1)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise ValueError("Viewer allocation cancelled") from None


def open_viewer(
    scene: Path,
    *,
    viewer: Path,
    partition: str,
    account: str | None,
    control: Path,
) -> int:
    """Open one scene through the current user's persistent viewer allocation."""
    scene = Path(scene).expanduser().absolute()
    viewer = Path(viewer).expanduser().absolute()
    if not scene.is_file():
        raise ValueError(f"Workbench scene is unavailable: {scene}")
    if not viewer.is_file():
        raise ValueError(f"Connectome Workbench viewer is unavailable: {viewer}")
    if os.environ.get("SLURM_JOB_ID"):
        if not os.environ.get("DISPLAY"):
            raise ValueError(
                "This Slurm allocation has no X11 display; reconnect with SSH X forwarding"
            )
        return subprocess.Popen(
            [str(viewer), "-scene-load-hd", str(scene), "1"], start_new_session=True
        ).pid

    root = state_directory(control)
    for attempt in range(2):
        with _launch_lock(root):
            active = _live_active(root)
            if active is None:
                _cancel_stale_active(root)
                process, token, log = _launch(
                    root, viewer=viewer, partition=partition, account=account
                )
                active = _await_start(process, root, token, log)
            try:
                response = _exchange(active, {"operation": "open", "scene": str(scene)})
            except OSError:
                response = {"ok": False, "stale": True, "error": "Viewer broker is unavailable"}
            if response.get("ok"):
                return int(response["pid"])
            if not response.get("stale") or attempt:
                raise ValueError(str(response.get("error") or "Viewer broker rejected the scene"))
            _cancel_stale_active(root)
        time.sleep(1)
    raise ValueError("Viewer broker is unavailable")


def _display_available() -> bool:
    probe = shutil.which("xdpyinfo")
    if probe is None:
        return bool(os.environ.get("DISPLAY"))
    try:
        return (
            subprocess.run(
                [probe],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _read_request(stream: socket.socket) -> dict:
    payload = b""
    while not payload.endswith(b"\n"):
        block = stream.recv(4096)
        if not block:
            break
        payload += block
        if len(payload) > _MAX_MESSAGE_BYTES:
            raise ValueError("Viewer request is too large")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Viewer request must be an object")
    return value


def _send_response(stream: socket.socket, response: dict) -> None:
    """Return one response without letting a disconnected client stop the broker."""
    try:
        stream.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
    except OSError:
        pass


def serve(root: Path, token: str, viewer: Path) -> None:
    """Serve authenticated scene-open requests until the Slurm allocation ends."""
    root = Path(root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    viewer = Path(viewer).expanduser().absolute()
    if not viewer.is_file():
        raise ValueError(f"Connectome Workbench viewer is unavailable: {viewer}")
    if not _display_available():
        raise ValueError("The Slurm viewer allocation has no usable X11 display")

    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    children: list[subprocess.Popen] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", 0))
        server.listen()
        server.settimeout(1)
        host = socket.getfqdn()
        active = {
            "protocol": PROTOCOL,
            "token": token,
            "host": host,
            "port": server.getsockname()[1],
            "pid": os.getpid(),
            "job_id": str(os.environ.get("SLURM_JOB_ID") or ""),
            "started_at": time.time(),
        }
        atomic_write_json(root / "active.json", active, sort_keys=True, mode=0o600, durable=True)
        try:
            while running:
                children = [child for child in children if child.poll() is None]
                try:
                    stream, _address = server.accept()
                except socket.timeout:
                    continue
                with stream:
                    try:
                        request = _read_request(stream)
                        if request.get("protocol") != PROTOCOL or request.get("token") != token:
                            raise ValueError("Viewer request authentication failed")
                        operation = request.get("operation")
                        if operation == "ping":
                            response = {"ok": True}
                        elif operation == "open":
                            if not _display_available():
                                response = {
                                    "ok": False,
                                    "stale": True,
                                    "error": "The viewer broker lost its X11 display",
                                }
                                running = False
                            else:
                                scene = Path(str(request.get("scene") or ""))
                                if not scene.is_absolute() or not scene.is_file():
                                    raise ValueError("Requested Workbench scene is unavailable")
                                child_log = root / f"viewer-{uuid.uuid4().hex}.log"
                                log_stream = child_log.open("a", encoding="utf-8")
                                try:
                                    child = subprocess.Popen(
                                        [str(viewer), "-scene-load-hd", str(scene), "1"],
                                        stdin=subprocess.DEVNULL,
                                        stdout=log_stream,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True,
                                    )
                                finally:
                                    log_stream.close()
                                children.append(child)
                                response = {"ok": True, "pid": child.pid}
                        else:
                            raise ValueError("Unknown viewer request")
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        response = {"ok": False, "error": str(error)}
                    _send_response(stream, response)
        finally:
            _remove_owned_active(root, token)


def build_parser() -> argparse.ArgumentParser:
    """Build the internal viewer-broker parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--token", required=True)
    parser.add_argument("--viewer", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the internal viewer broker."""
    args = build_parser().parse_args(argv)
    if not args.serve:
        raise SystemExit("The viewer broker is an internal service")
    serve(args.state_dir, args.token, args.viewer)


if __name__ == "__main__":
    main()
