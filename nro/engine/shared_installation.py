"""Prepare and publish the main installation without discarding queued demand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Literal

from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import Registry
from nro.orchestration.releases import ReleaseStore, tagged_source
from nro.orchestration.scheduler_implementation import activate
from nro.orchestration.worker_control import (
    cancel_worker_allocations,
    wait_for_worker_shutdown,
)


def _executing(registry: Registry) -> tuple[int, int]:
    """Count running derivative attempts and ingestion stages."""
    with registry.connection() as db:
        attempts = int(
            db.execute(
                "SELECT COUNT(*) FROM attempts WHERE state IN ('running','cancel_requested')"
            ).fetchone()[0]
        )
    from nro.bidsify.index import IngestionIndex

    ingestion = sum(row["state"] == "running" for row in IngestionIndex(registry).rows())
    return attempts, ingestion


MaintenanceAction = Literal["drain", "stop"]


def _maintenance(registry: Registry) -> tuple[str | None, str | None, str | None]:
    with registry.connection() as db:
        rows = {
            str(row["key"]): str(row["value"])
            for row in db.execute(
                "SELECT key,value FROM metadata "
                "WHERE key IN "
                "('maintenance_mode','installation_checkout','installation_action')"
            )
        }
    return (
        rows.get("maintenance_mode"),
        rows.get("installation_checkout"),
        rows.get("installation_action"),
    )


def prepare_pool(
    registry: Registry,
    *,
    checkout: Path,
    confirm: Callable[[dict], MaintenanceAction | None],
    poll_interval: float = 5.0,
    report_interval: float = 30.0,
) -> dict:
    """Quiesce workers under an installation barrier while preserving demand."""
    checkout = Path(checkout).expanduser().resolve()
    registry.initialize()
    mode, owner, saved_action = _maintenance(registry)
    if mode not in {None, "installation"}:
        raise RuntimeError(f"Shared registry is already in {mode} maintenance")
    if mode == "installation" and owner != str(checkout):
        raise RuntimeError(
            f"Shared installation maintenance belongs to {owner or 'another checkout'}"
        )
    activity = registry.worker_pool_activity()
    attempts, ingestion = _executing(registry)
    active = bool(activity["workers"] or activity["submissions"] or attempts or ingestion)
    summary = {
        "workers": len(activity["workers"]),
        "submissions": len(activity["submissions"]),
        "attempts": attempts,
        "ingestion": ingestion,
        "resuming": mode == "installation",
    }
    action = saved_action
    if active and action is None:
        action = confirm(summary)
        if action is None:
            detail = (
                "the worker pool was not changed"
                if mode is None
                else "the existing maintenance barrier remains in place"
            )
            raise RuntimeError(f"Shared installation cancelled; {detail}")
    action = action or "drain"
    if action not in {"drain", "stop"}:
        raise RuntimeError(f"Shared installation has an invalid maintenance action: {action}")
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO metadata(key,value) VALUES ('maintenance_mode','installation')
               ON CONFLICT(key) DO UPDATE SET value=excluded.value"""
        )
        db.execute(
            """INSERT INTO metadata(key,value) VALUES ('installation_checkout',?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (str(checkout),),
        )
        db.execute(
            """INSERT INTO metadata(key,value) VALUES ('installation_action',?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (action,),
        )

    if action == "drain":
        next_report = 0.0
        while True:
            attempts, ingestion = _executing(registry)
            if not attempts and not ingestion:
                break
            now = time.monotonic()
            if now >= next_report:
                print(
                    f"Waiting for {attempts} derivative attempt(s) and "
                    f"{ingestion} ingestion stage(s) to finish...",
                    flush=True,
                )
                next_report = now + report_interval
            time.sleep(poll_interval)

    shutdown = registry.request_worker_shutdown(all_users=True)
    stopped, failures = cancel_worker_allocations(registry, shutdown)
    if failures:
        raise RuntimeError("Could not stop worker allocation(s): " + "; ".join(failures))
    wait_for_worker_shutdown(registry)
    return {**summary, "action": action, "stopped_jobs": stopped}


def publish(checkout: Path, registry: Registry) -> dict:
    """Register main, record its exact release tag, and activate its scheduler."""
    checkout = Path(checkout).expanduser().resolve()
    tagged_source(checkout)
    branches = BranchStore(registry.paths.control)
    snapshot = branches.initialize()
    main = snapshot.topology.records["main"]
    if checkout not in main.checkouts:
        snapshot = branches.authorize_checkout("main", checkout, revision=snapshot.revision)
    release = ReleaseStore(branches).record_tagged(checkout)
    from nro.configuration.site import installation_record
    from nro.engine.bootstrap import RECORD, write_record

    installation = installation_record(checkout)
    installation["release"] = release
    write_record(checkout / RECORD, installation)
    implementation = activate(registry, checkout, installation_maintenance=True)
    return {"release": release, "implementation": implementation}
