"""Serve completed surface derivatives to a web browser through an SSH tunnel."""

from __future__ import annotations

import argparse
import getpass
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

DESCRIPTION = (
    "Start or find your browser-based surface viewer and print how to open it. The "
    "server runs in your own Slurm job on the site's viewing partition, and every "
    "`nro viz` call from this checkout reuses it. The command returns once the server "
    "is running. Optional selectors set the page's initial branch, project, and "
    "participant; all three can be changed in the page."
)
STARTUP_POLL_SECONDS = 2


def build_parser(*, prog: str = "nro viz") -> argparse.ArgumentParser:
    """Construct the viewer command parser."""
    parser = argparse.ArgumentParser(prog=prog, description=DESCRIPTION)
    parser.add_argument("-P", "--project", help="Project shown when the page opens")
    parser.add_argument("-p", "--participant", help="Participant shown when the page opens")
    parser.add_argument("--branch", help="Branch shown when the page opens")
    parser.add_argument(
        "--local-port",
        type=int,
        help="Port on your computer used in the printed tunnel command (default: the server port)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return as soon as the server job is queued instead of when it is running",
    )
    parser.add_argument("--stop", action="store_true", help="Cancel your viewer server")
    parser.add_argument(
        "--serverless",
        action="store_true",
        help="Run a private server in this terminal on the current host until interrupted",
    )
    parser.add_argument("--port", type=int, help="Server port for --serverless")
    parser.add_argument("--tunnel-host", help=argparse.SUPPRESS)
    return parser


def instructions(*, host: str, port: int, local_port: int, tunnel_host: str, url: str) -> str:
    """Describe how to reach a server from the user's own computer."""
    return (
        f"nro viz is serving on {host}:{port}.\n"
        "On your own computer, run:\n\n"
        f"  ssh -N -L {local_port}:{host}:{port} {getpass.getuser()}@{tunnel_host}\n\n"
        "Then open:\n\n"
        f"  {url}\n\n"
        "The server keeps running after this command exits. Stop it with `nro viz --stop`."
    )


def _tunnel_host(args: argparse.Namespace) -> str:
    return args.tunnel_host or os.environ.get("SLURM_SUBMIT_HOST") or socket.getfqdn()


def _announce(args: argparse.Namespace, active: dict, token: str) -> None:
    local_port = args.local_port or int(active["port"])
    query = {"token": token}
    query.update(
        (key, value)
        for key, value in (
            ("branch", args.branch),
            ("project", args.project),
            ("participant", args.participant),
        )
        if value
    )
    url = f"http://localhost:{local_port}/?{urlencode(query)}"
    print(
        instructions(
            host=str(active["host"]),
            port=int(active["port"]),
            local_port=local_port,
            tunnel_host=_tunnel_host(args),
            url=url,
        ),
        flush=True,
    )


def _state() -> Path:
    from nro.site.configuration import CHECKOUT, settings
    from nro.viz.service import state_directory

    return state_directory(Path(settings()[0]["registry"]), CHECKOUT)


def main(argv: list[str] | None = None, *, prog: str = "nro viz") -> None:
    """Find, start, or stop the current user's viewer server."""
    from nro.viz import service

    args = build_parser(prog=prog).parse_args(argv)
    if args.port is not None and not args.serverless:
        raise SystemExit("--port applies only to --serverless; your server uses a stable port")
    root = _state()
    token = service.access_token(root)
    if args.stop:
        jobs = service.stop(root)
        print(
            "Cancelled viewer server job " + ", ".join(jobs)
            if jobs
            else "No viewer server is running"
        )
        return
    if args.serverless:
        print(f"Serving in this terminal; open http://localhost:PORT/?token={token}", flush=True)
        service.serve(root, port=args.port, record=False)
        return

    from nro.site.configuration import settings

    with service.launch_lock(root):
        active = service.active_server(root, token)
        if active is not None:
            _announce(args, active, token)
            return
        job_id = service.queued_job(root)
        if job_id is None:
            job_id = service.submit(root, settings()[0])
            print(f"Submitted viewer server job {job_id}.", file=sys.stderr, flush=True)
        else:
            print(f"Viewer server job {job_id} is already queued.", file=sys.stderr, flush=True)
    if args.no_wait:
        print("Run `nro viz` again once it starts to print the tunnel command and URL.")
        return
    print(
        "Waiting for the server to start (Ctrl-C leaves it queued)...",
        file=sys.stderr,
        flush=True,
    )
    try:
        while True:
            active = service.active_server(root, token)
            if active is not None:
                _announce(args, active, token)
                return
            # Ask Slurm directly: on NFS, another host may see the job record
            # disappear before it sees the server's address file appear.
            if service.job_state(job_id) not in service.QUEUED_STATES:
                raise SystemExit(
                    f"Viewer server job {job_id} ended before it started serving; "
                    f"see {root / f'server-{job_id}.log'}"
                )
            time.sleep(STARTUP_POLL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nJob {job_id} remains queued. Run `nro viz` later for the URL.", file=sys.stderr)


if __name__ == "__main__":
    main()
