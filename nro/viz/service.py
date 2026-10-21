"""Run one browser-viewer server per user and checkout in a Slurm job.

State lives in a private directory under the site registry:
``viewers/uid-<UID>/viz-<CHECKOUT_DIGEST>/``. It holds the user's access token,
the pending job, the running server's address, and server logs. Every
``nro viz`` call by the same user from the same checkout shares one server, so
several browser tabs and terminals use one allocation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from nro.engine.io import atomic_write_json, read_json

PORT_BASE = 20000
PORT_RANGE = 40000
PORT_ATTEMPTS = 50
CACHE_FRACTION = 0.6
ACTIVE_FIELDS = {"host", "port", "pid", "job_id", "started_at"}
QUEUED_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "REQUEUED", "RESIZING"}


def state_directory(control: Path, checkout: Path) -> Path:
    """Return the current user's private state directory for one checkout."""
    digest = hashlib.sha256(str(Path(checkout).absolute()).encode()).hexdigest()[:12]
    user = Path(control).expanduser().absolute() / "viewers" / f"uid-{os.getuid()}"
    path = user / f"viz-{digest}"
    for directory in (user, path):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
    return path


def preferred_port(uid: int | None = None) -> int:
    """Return the user's stable first-choice server port."""
    return PORT_BASE + (os.getuid() if uid is None else uid) % PORT_RANGE


def access_token(root: Path) -> str:
    """Read the user's access token, creating it on first use.

    The token persists across servers, so a bookmarked URL keeps working when
    a later server lands on the same host and port.
    """
    from nro.viz.server import new_token

    path = root / "token"
    try:
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    except FileNotFoundError:
        pass
    token = new_token()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    return token


@contextmanager
def launch_lock(root: Path) -> Iterator[None]:
    """Serialize server discovery and submission for one user and checkout."""
    path = root / "launch.lock"
    with path.open("a+", encoding="utf-8") as stream:
        path.chmod(0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read(path: Path) -> dict | None:
    try:
        return read_json(path)
    except (FileNotFoundError, OSError, ValueError):
        return None


def active_server(root: Path, token: str, *, timeout: float = 5.0) -> dict | None:
    """Return the running server's record when it answers an authorized ping."""
    active = _read(root / "active.json")
    if active is None or set(active) != ACTIVE_FIELDS:
        return None
    request = urllib.request.Request(
        f"http://{active['host']}:{active['port']}/api/ping",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if json.loads(response.read()).get("ok"):
                return active
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def job_state(job_id: str) -> str | None:
    """Return a Slurm job's state, or None when Slurm no longer lists it."""
    if not job_id.isdigit() or shutil.which("squeue") is None:
        return None
    completed = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    state = completed.stdout.strip()
    return state or None


def queued_job(root: Path) -> str | None:
    """Return the job ID of a submitted server that Slurm still lists."""
    pending = _read(root / "job.json")
    job_id = str((pending or {}).get("job_id") or "")
    if job_state(job_id) in QUEUED_STATES:
        return job_id
    (root / "job.json").unlink(missing_ok=True)
    return None


def submit(root: Path, values: dict) -> str:
    """Submit the server as a batch job and return its job ID.

    ``values`` are the site settings; the job uses ``viewing_partition``,
    ``account``, ``viewer_cpus``, ``viewer_memory``, and ``viewer_time``.
    """
    launcher = shutil.which("sbatch")
    if launcher is None:
        raise SystemExit("sbatch is unavailable; rerun with --serverless to serve from this host")
    serve_command = shlex.join(
        [
            sys.executable,
            "-m",
            "nro.viz.service",
            "--serve",
            "--state-dir",
            str(root),
            "--memory-gb",
            str(values["viewer_memory"]),
        ]
    )
    command = [
        launcher,
        "--parsable",
        "--job-name=nro-viz",
        "--ntasks=1",
        f"--cpus-per-task={values['viewer_cpus']}",
        f"--mem={values['viewer_memory']}G",
        f"--time={values['viewer_time']}:00:00",
        f"--output={root / 'server-%j.log'}",
        f"--wrap=exec {serve_command}",
    ]
    if values.get("viewing_partition"):
        command.append(f"--partition={values['viewing_partition']}")
    if values.get("account"):
        command.append(f"--account={values['account']}")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_") or key in {"SLURM_CLUSTERS", "SLURM_CONF"}
    }
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"sbatch rejected the viewer server: {completed.stderr.strip()}")
    job_id = completed.stdout.strip().split(";")[0]
    atomic_write_json(
        root / "job.json",
        {"job_id": job_id, "submitted_at": time.time()},
        mode=0o600,
    )
    return job_id


def stop(root: Path) -> list[str]:
    """Cancel the user's queued or running server and return the cancelled job IDs."""
    jobs = []
    for name in ("active.json", "job.json"):
        record = _read(root / name) or {}
        job_id = str(record.get("job_id") or "")
        if job_id.isdigit() and job_id not in jobs:
            jobs.append(job_id)
    for job_id in jobs:
        subprocess.run(
            ["scancel", job_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    for name in ("active.json", "job.json"):
        (root / name).unlink(missing_ok=True)
    return jobs


def bind(root: Path, token: str, *, port: int | None, cache_bytes: int, restart):
    """Bind a server on the requested port, or on the first free stable port.

    Without an explicit port the server tries ``preferred_port()`` and the next
    ``PORT_ATTEMPTS - 1`` ports, then any free port.
    """
    from nro.viz.catalog import Catalogs, branch_rows
    from nro.viz.server import make_server

    catalogs = Catalogs(lambda branch: branch_rows(branch or None))
    candidates = (
        [port]
        if port is not None
        else [preferred_port() + offset for offset in range(PORT_ATTEMPTS)] + [0]
    )
    for candidate in candidates:
        try:
            return make_server(
                catalogs,
                host="0.0.0.0",
                port=candidate,
                token=token,
                restart=restart,
                cache_bytes=cache_bytes,
            )
        except OSError:
            if candidate == candidates[-1]:
                raise
    raise RuntimeError("unreachable")


def serve(root: Path, *, port: int | None = None, memory_gb: int = 32, record: bool = True) -> None:
    """Serve until stopped; restart in place on request or SIGUSR1.

    With ``record``, the server publishes its address in ``active.json`` so
    ``nro viz`` can find it, and removes the record when it exits. A restart
    replaces the process with one that loads the current code and rebinds the
    same port.
    """
    token = access_token(root)
    restarting = threading.Event()

    def restart() -> None:
        restarting.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    def terminate(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    cache_bytes = int(memory_gb * 1024**3 * CACHE_FRACTION)
    server = bind(root, token, port=port, cache_bytes=cache_bytes, restart=restart)
    signal.signal(signal.SIGUSR1, lambda *_: restart())
    signal.signal(signal.SIGTERM, terminate)
    bound = server.server_address[1]
    active = {
        "host": socket.getfqdn(),
        "port": bound,
        "pid": os.getpid(),
        "job_id": str(os.environ.get("SLURM_JOB_ID") or ""),
        "started_at": time.time(),
    }
    if record:
        atomic_write_json(root / "active.json", active, mode=0o600)
    print(f"nro viz server listening on {active['host']}:{bound}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    if restarting.is_set():
        sys.stdout.flush()
        command = restart_argv(root, bound, memory_gb, record=record)
        os.execv(command[0], command)
    if record and (_read(root / "active.json") or {}).get("pid") == os.getpid():
        (root / "active.json").unlink(missing_ok=True)


def restart_argv(root: Path, port: int, memory_gb: int, *, record: bool = True) -> list[str]:
    """Return the command that reruns a server on the same port."""
    command = [
        sys.executable,
        "-m",
        "nro.viz.service",
        "--serve",
        "--state-dir",
        str(root),
        "--port",
        str(port),
        "--memory-gb",
        str(memory_gb),
    ]
    if not record:
        command.append("--unrecorded")
    return command


def build_parser() -> argparse.ArgumentParser:
    """Build the internal server parser used inside the Slurm job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--memory-gb", type=int, default=32)
    parser.add_argument("--unrecorded", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the internal viewer server."""
    args = build_parser().parse_args(argv)
    if not args.serve:
        raise SystemExit("The viewer server is an internal service; run `nro viz`")
    serve(args.state_dir, port=args.port, memory_gb=args.memory_gb, record=not args.unrecorded)


if __name__ == "__main__":
    main()
