"""Command-line options derived from the site's ``slurm.`` settings.

Each ``slurm.`` site key has one option, named by the part of the key after
``slurm.`` with periods and underscores replaced by hyphens. For example,
``slurm.worker.max_memory_gb`` becomes ``--worker-max-memory-gb``. An option
that is not given keeps the site value.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Mapping

from nro.site.configuration import DEFAULTS
from nro.site.policy import SITE_SETTINGS

PREFIX = "slurm."
SLURM_SETTINGS = {name: spec for name, spec in SITE_SETTINGS.items() if name.startswith(PREFIX)}

# Commands add the groups whose allocations they can create.
SCHEDULER_GROUPS = ("partition", "account", "scheduler")
WORKER_GROUPS = (*SCHEDULER_GROUPS, "worker", "long_worker")
VIEWER_GROUPS = ("account", "viewer")


def slurm_flag(name: str) -> str:
    """Return the option for one ``slurm.`` site key."""
    return "--" + name.removeprefix(PREFIX).replace(".", "-").replace("_", "-")


def slurm_group(name: str) -> str:
    """Return the allocation group that one ``slurm.`` site key configures."""
    suffix = name.removeprefix(PREFIX)
    if suffix == "viewer_partition":
        return "viewer"
    return suffix.split(".", 1)[0]


def add_slurm_arguments(parser: argparse.ArgumentParser, groups: Iterable[str]) -> None:
    """Add the options for every ``slurm.`` key in the selected groups."""
    selected = set(groups)
    unknown = selected - {slurm_group(name) for name in SLURM_SETTINGS}
    if unknown:
        raise ValueError(f"Unknown Slurm option groups: {', '.join(sorted(unknown))}")
    options = parser.add_argument_group(
        "Slurm options", "Override site slurm.* settings for allocations this command creates"
    )
    for name, spec in SLURM_SETTINGS.items():
        if slurm_group(name) not in selected:
            continue
        integer = integer_kind(spec.kind)
        options.add_argument(
            slurm_flag(name),
            dest=spec.storage_key,
            type=int if integer else str,
            default=None,
            metavar="N" if integer else "NAME",
            help=f"{spec.description} (default: site {name})",
        )


def integer_kind(kind: str) -> bool:
    """Return whether a site setting kind holds an integer."""
    return kind == "int"


def apply_slurm_arguments(args: argparse.Namespace, values: Mapping[str, object]) -> dict:
    """Return site settings with the given Slurm options applied.

    Each Slurm option on ``args`` is also replaced by its effective value, so
    callers can read ``args.worker_time`` and similar attributes directly.
    Raises ValueError for an option below its site minimum.
    """
    effective = {**DEFAULTS, **values}
    for name, spec in SLURM_SETTINGS.items():
        if not hasattr(args, spec.storage_key):
            continue
        value = getattr(args, spec.storage_key)
        if value is None:
            value = effective[spec.storage_key]
        elif spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{slurm_flag(name)} must be at least {spec.minimum}")
        effective[spec.storage_key] = value
        # An empty site partition or account means "let Slurm choose".
        setattr(args, spec.storage_key, value if integer_kind(spec.kind) else value or None)
    if hasattr(args, "worker_memory") and int(effective["worker_max_memory"]) < int(
        effective["worker_memory"]
    ):
        raise ValueError("--worker-max-memory-gb must be at least --worker-memory-gb")
    return effective


def scheduler_options(values: Mapping[str, object]) -> dict:
    """Return the scheduler allocation options used when a command starts one."""
    values = {**DEFAULTS, **values}
    return {
        "partition": values["partition"],
        "account": values["account"] or None,
        "time": int(values["scheduler_time"]),
        "memory": int(values["scheduler_memory"]),
        "cpus": int(values["scheduler_cpus"]),
    }
