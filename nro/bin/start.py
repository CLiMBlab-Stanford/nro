"""Start the shared scheduler without requesting scientific work."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.site.slurm_arguments import (
    SCHEDULER_GROUPS,
    add_slurm_arguments,
    apply_slurm_arguments,
    scheduler_options,
)


def build_parser(*, prog: str = "nro.bin.start") -> argparse.ArgumentParser:
    """Construct the scheduler-start parser without starting the service."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_slurm_arguments(parser, SCHEDULER_GROUPS)
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.start") -> None:
    """Submit a scheduler allocation only when no scheduler is already live."""
    args = build_parser(prog=prog).parse_args(argv)
    from nro.orchestration.scheduler_client import start
    from nro.site import configuration as site

    values = site.settings()[0]
    try:
        slurm_values = apply_slurm_arguments(args, values)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = start(
        Path(values["registry"]),
        site.bids_root(),
        checkout=site.CHECKOUT,
        options=scheduler_options(slurm_values),
    )
    state = result["state"]
    job_id = result.get("job_id") or "unknown"
    if state == "submitted":
        print(f"Submitted scheduler allocation {job_id}.")
    elif state == "running":
        print(f"Scheduler is already running ({job_id}).")
    else:
        print(f"Scheduler allocation is already starting ({job_id}).")
