"""Send compiled branch work to the designated central Python environment."""

import json
import subprocess
from pathlib import Path

from nro.orchestration.execution_cache import cache_lock, service_lease
from nro.orchestration.scheduler_implementation import (
    capture_worker_implementation,
    implementation_path,
)


def command(control: Path, bids_root: Path) -> tuple[str, ...]:
    """Capture the central service command while the caller holds the cache lock."""
    if not implementation_path(control).is_file():
        raise ValueError("Activate an approved main scheduler before submitting branch work")
    source, site, python = capture_worker_implementation(control, bids_root)
    return source.command((str(python), "-m", "nro.orchestration.scheduler_service"), site=site)


def exchange(command: tuple[str, ...], message: dict, *, descriptors: tuple[int, ...] = ()) -> dict:
    """Exchange one bounded operation; surface central errors without hiding stderr."""
    result = subprocess.run(
        command,
        input=json.dumps(message, allow_nan=False),
        text=True,
        stdout=subprocess.PIPE,
        pass_fds=descriptors,
    )
    try:
        response = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise RuntimeError("Central scheduler returned an invalid response") from error
    if result.returncode or "error" in response:
        raise RuntimeError(response.get("error", "Central scheduler failed"))
    return response["result"]


def supply(
    control: Path, bids_root: Path, *, checkout: Path, request_ids: list[str], options: dict
) -> dict:
    """Start central workers after admission has published the execution pins."""
    with service_lease(control) as lease:
        with cache_lock(control):
            selected = command(control, bids_root)
        return exchange(
            selected,
            dict(
                operation="supply", checkout=str(checkout), request_ids=request_ids, options=options
            ),
            descriptors=(lease[1],),
        )


def status(control: Path, bids_root: Path, *, checkout: Path, mode: str) -> dict:
    """Read central compiled status while protecting the short-lived service source."""
    with cache_lock(control):
        return exchange(
            command(control, bids_root), dict(operation="status", checkout=str(checkout), mode=mode)
        )


def stop(control: Path, bids_root: Path, *, checkout: Path, project: str, selection: dict) -> dict:
    """Cancel matching demand through the central branch-scoped operation."""
    with cache_lock(control):
        return exchange(
            command(control, bids_root),
            dict(operation="stop", checkout=str(checkout), project=project, selection=selection),
        )


def logs(
    control: Path, bids_root: Path, *, checkout: Path, selection: dict, instance_level: bool
) -> dict:
    """Find branch logs without opening the scheduler database in this checkout."""
    with cache_lock(control):
        return exchange(
            command(control, bids_root),
            dict(
                operation="logs",
                checkout=str(checkout),
                selection=selection,
                instance_level=instance_level,
            ),
        )


def pool_operation(
    control: Path, bids_root: Path, *, checkout: Path, operation: str, concurrency=None
) -> dict:
    """Apply an explicit global pool control using the central implementation."""
    with cache_lock(control):
        return exchange(
            command(control, bids_root),
            dict(operation=operation, checkout=str(checkout), concurrency=concurrency),
        )


def maintenance(
    control: Path, bids_root: Path, *, checkout: Path, operation: str, **fields
) -> dict:
    """Run a scoped maintenance operation while the service retains its source lease."""
    if operation not in {
        "purge_snapshot",
        "purge",
        "cache",
        "repair_prepare",
        "repair_finish",
        "promotion_preview",
        "promotion_publish",
        "publish",
        "branch_update",
    }:
        raise ValueError("Unsupported maintenance operation")
    with service_lease(control) as lease:
        with cache_lock(control):
            selected = command(control, bids_root)
        return exchange(
            selected,
            dict(operation=operation, checkout=str(checkout), service=lease[0], **fields),
            descriptors=(lease[1],),
        )
