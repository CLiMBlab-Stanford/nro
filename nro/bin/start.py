"""Start the shared scheduler without requesting scientific work."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser(*, prog: str = "nro.bin.start") -> argparse.ArgumentParser:
    """Construct the scheduler-start parser without starting the service."""
    from nro.site.configuration import settings

    site, _ = settings()
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--partition", default=site["partition"])
    parser.add_argument("--account", default=site["account"] or None)
    parser.add_argument("--time", type=int, default=24, metavar="HOURS")
    parser.add_argument("--memory", type=int, default=4, metavar="GB")
    parser.add_argument("--cpus", type=int, default=4)
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.start") -> None:
    """Submit a scheduler allocation only when no scheduler is already live."""
    args = build_parser(prog=prog).parse_args(argv)
    if args.time < 1 or args.memory < 1 or args.cpus < 1:
        raise SystemExit("--time, --memory, and --cpus must be positive")
    from nro.orchestration.scheduler_client import start
    from nro.site import configuration as site

    values = site.settings()[0]
    result = start(
        Path(values["registry"]),
        site.bids_root(),
        checkout=site.CHECKOUT,
        options={
            "partition": args.partition,
            "account": args.account,
            "time": args.time,
            "memory": args.memory,
            "cpus": args.cpus,
        },
    )
    state = result["state"]
    job_id = result.get("job_id") or "unknown"
    if state == "submitted":
        print(f"Submitted scheduler allocation {job_id}.")
    elif state == "running":
        print(f"Scheduler is already running ({job_id}).")
    else:
        print(f"Scheduler allocation is already starting ({job_id}).")
