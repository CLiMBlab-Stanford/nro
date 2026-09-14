"""Prepare and publish the main installation without discarding queued demand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Literal

from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import Registry
from nro.orchestration.releases import ReleaseStore, tagged_source
from nro.orchestration.scheduler_implementation import activate

MaintenanceAction = Literal["drain", "stop"]


def prepare_pool(
    registry: Registry,
    *,
    checkout: Path,
    confirm: Callable[[dict], MaintenanceAction | None],
    poll_interval: float = 5.0,
    report_interval: float = 30.0,
    rebuild_schema: bool = False,
) -> dict:
    """Quiesce workers under an installation barrier while preserving demand."""
    checkout = Path(checkout).expanduser().resolve()
    from nro.orchestration.scheduler_implementation import implementation_path

    if not implementation_path(registry.paths.control).is_file():
        # Initial activation has no controller implementation to launch. This
        # is the one normal offline bootstrap of the scheduler database.
        registry.initialize()
        activity = registry.worker_pool_activity()
        if activity["workers"] or activity["submissions"]:
            raise RuntimeError(
                "An unmanaged worker pool is still active; stop it before scheduler activation"
            )
        with registry.connection(write=True) as db:
            db.executemany(
                "INSERT INTO metadata(key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (
                    ("maintenance_mode", "installation"),
                    ("installation_checkout", str(checkout)),
                    ("installation_action", "drain"),
                ),
            )
        return {
            "workers": 0,
            "submissions": 0,
            "attempts": 0,
            "ingestion": 0,
            "action": "drain",
            "done": True,
            "stopped_jobs": [],
            "failures": [],
        }
    from nro.orchestration.scheduler_bus import read_active
    from nro.orchestration.scheduler_client import maintenance, shutdown_service

    control, bids_root = registry.paths.control, registry.paths.bids_root
    from nro.orchestration.registry import SCHEMA_VERSION

    stored_schema = registry.stored_schema_version()
    if stored_schema != SCHEMA_VERSION:
        if not rebuild_schema:
            raise RuntimeError(
                f"Scheduler schema {stored_schema} does not match {SCHEMA_VERSION}; "
                "shared installation maintenance must rebuild it"
            )
        if read_active(control) is not None:
            shutdown_service(
                control,
                bids_root,
                checkout=checkout,
                allow_changed_checkout=True,
            )
            deadline = time.monotonic() + 60
            while read_active(control) is not None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Scheduler did not stop before its registry schema rebuild")
                time.sleep(0.1)
        activity = registry.worker_pool_activity(for_repair=True)
        if activity["workers"] or activity["submissions"]:
            choice = confirm(
                {
                    "workers": len(activity["workers"]),
                    "submissions": len(activity["submissions"]),
                    "attempts": 0,
                    "ingestion": 0,
                }
            )
            if choice != "stop":
                raise RuntimeError(
                    "An incompatible scheduler schema cannot be drained by the new release; "
                    "wait for current work to finish or rerun maintenance and choose stop"
                )
            from nro.orchestration.worker_control import stop_worker_pool_for_repair

            stop_worker_pool_for_repair(registry)
        from nro.orchestration.scheduler_repair import repair_for_installation

        repaired = repair_for_installation(registry, checkout=checkout)
        print(
            f"Rebuilt scheduler schema {stored_schema} as {SCHEMA_VERSION}; "
            f"backup: {repaired['backup']}",
            flush=True,
        )
    summary = maintenance(
        control,
        bids_root,
        checkout=checkout,
        operation="installation_activity",
    )
    active = any(summary.values())
    action = None
    if active:
        action = confirm(summary)
        if action is None:
            raise RuntimeError("Shared installation cancelled; the worker pool was not changed")
    action = action or "drain"
    progress = maintenance(
        control,
        bids_root,
        checkout=checkout,
        operation="installation_prepare",
        action=action,
    )
    next_report = 0.0
    while not progress["done"]:
        if progress["failures"]:
            raise RuntimeError(
                "Could not stop worker allocation(s): " + "; ".join(progress["failures"])
            )
        now = time.monotonic()
        if now >= next_report:
            print(
                f"Waiting for {progress['attempts']} derivative attempt(s), "
                f"{progress['ingestion']} ingestion stage(s), and "
                f"{progress['workers'] + progress['submissions']} worker allocation(s)...",
                flush=True,
            )
            next_report = now + report_interval
        time.sleep(poll_interval)
        progress = maintenance(
            control,
            bids_root,
            checkout=checkout,
            operation="installation_progress",
        )
    shutdown_service(
        control,
        bids_root,
        checkout=checkout,
        allow_changed_checkout=True,
    )
    deadline = time.monotonic() + 60
    while read_active(control) is not None:
        if time.monotonic() >= deadline:
            raise RuntimeError("Scheduler did not release installation maintenance")
        time.sleep(0.1)
    return {**summary, **progress}


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
