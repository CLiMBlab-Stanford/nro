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


def _repair_scientific_schemas(registry: Registry) -> list[dict]:
    """Rebuild incompatible branch storage and report each replacement."""
    from nro.orchestration.scheduler_repair import repair_scientific_schemas

    repaired = repair_scientific_schemas(registry)
    for item in repaired:
        if item["backup"] is not None:
            if item.get("action") == "migrated":
                print(
                    f"Migrated scientific schema {item['stored_schema']} to {item['schema']} "
                    f"for branch {item['branch']}; backup: {item['backup']}",
                    flush=True,
                )
            else:
                print(
                    f"Rebuilt scientific schema {item['stored_schema']} as {item['schema']} "
                    f"for branch {item['branch']} with {item['work_items']} recovered work "
                    f"item(s); backup: {item['backup']}",
                    flush=True,
                )
        for message in item["unavailable"]:
            print(f"Could not recover branch {item['branch']}: {message}", flush=True)
    return repaired


def prepare_pool(
    registry: Registry,
    *,
    checkout: Path,
    confirm: Callable[[dict], MaintenanceAction | None],
    poll_interval: float = 5.0,
    report_interval: float = 30.0,
    update_schema: bool = False,
) -> dict:
    """Quiesce workers under an installation barrier while preserving demand."""
    checkout = Path(checkout).expanduser().resolve()
    from nro.engine.maintenance import MaintenanceJournal, audit_shared_state
    from nro.orchestration.selection import discover_bids_inventory

    journal = MaintenanceJournal(registry.paths.control, checkout)
    audit = audit_shared_state(registry, discover_bids_inventory(registry.paths.bids_root))
    journal.record("audited", audit=audit.as_dict())
    if audit.fatal_errors:
        raise RuntimeError("Shared-maintenance audit failed:\n- " + "\n- ".join(audit.fatal_errors))
    for message in audit.ownership_errors:
        print(f"Corrupt or obsolete derivative ownership: {message}", flush=True)
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
        scientific = _repair_scientific_schemas(registry)
        result = {
            "workers": 0,
            "submissions": 0,
            "attempts": 0,
            "ingestion": 0,
            "action": "drain",
            "done": True,
            "stopped_jobs": [],
            "failures": [],
            "scientific": scientific,
        }
        journal.finish(result=result)
        return result
    from nro.orchestration.scheduler_bus import read_active
    from nro.orchestration.scheduler_client import maintenance, shutdown_service

    control, bids_root = registry.paths.control, registry.paths.bids_root
    from nro.orchestration.registry import SCHEMA_VERSION

    stored_schema = registry.stored_schema_version()
    if stored_schema != SCHEMA_VERSION:
        if not update_schema:
            raise RuntimeError(
                f"Scheduler schema {stored_schema} does not match {SCHEMA_VERSION}; "
                "shared installation maintenance must update it"
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
        from nro.orchestration.registry_schema import SCHEMA as SCHEDULER_SCHEMA

        if SCHEDULER_SCHEMA.supports(stored_schema):
            backup = registry.migrate_schema()
            print(
                f"Migrated scheduler schema {stored_schema} to {SCHEMA_VERSION}; backup: {backup}",
                flush=True,
            )
        else:
            from nro.orchestration.scheduler_repair import repair_for_installation

            repaired = repair_for_installation(registry, checkout=checkout)
            print(
                f"Rebuilt scheduler schema {stored_schema} as {SCHEMA_VERSION}; "
                f"backup: {repaired['backup']}",
                flush=True,
            )
    journal.record("quiescing", audit=audit.as_dict())
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
    journal.record("rebuilding", audit=audit.as_dict())
    scientific = _repair_scientific_schemas(registry)
    result = {**summary, **progress, "scientific": scientific}
    journal.finish(result=result)
    return result


def publish(checkout: Path, registry: Registry, *, installation: dict | None = None) -> dict:
    """Publish a validated environment and scheduler binding under the maintenance barrier."""
    checkout = Path(checkout).expanduser().resolve()
    tagged_source(checkout)
    branches = BranchStore(registry.paths.control)
    snapshot = branches.initialize()
    main = snapshot.topology.records["main"]
    if checkout not in main.checkouts:
        snapshot = branches.authorize_checkout("main", checkout, revision=snapshot.revision)
    release = ReleaseStore(branches).record_tagged(checkout)
    from nro.configuration.site import installation_record
    from nro.engine.bootstrap import RECORD
    from nro.engine.io import atomic_write_text
    from nro.orchestration.scheduler_implementation import implementation_path

    installation = dict(installation or installation_record(checkout))
    if installation.get("mode") != "shared" or installation.get("checkout") != str(checkout):
        raise ValueError("Shared publication requires a matching installation candidate")
    if not installation.get("ready"):
        raise ValueError("Shared publication requires a validated installation candidate")
    installation["release"] = release
    record_path = checkout / RECORD
    binding_path = implementation_path(registry.paths.control)
    previous_record = record_path.read_bytes() if record_path.is_file() else None
    previous_binding = binding_path.read_bytes() if binding_path.is_file() else None
    try:
        implementation = activate(
            registry,
            checkout,
            installation_maintenance=True,
            installation=installation,
            installation_path=record_path,
        )
    except BaseException:
        if previous_record is None:
            record_path.unlink(missing_ok=True)
        else:
            atomic_write_text(record_path, previous_record.decode("utf-8"), durable=True)
        if previous_binding is None:
            binding_path.unlink(missing_ok=True)
        else:
            atomic_write_text(binding_path, previous_binding.decode("utf-8"), durable=True)
        raise
    return {"release": release, "implementation": implementation}
