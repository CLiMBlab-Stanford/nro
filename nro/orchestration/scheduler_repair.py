"""Explicit main-maintainer replacement of shared scheduling state."""

import json
from pathlib import Path

from nro.configuration.site import CHECKOUT
from nro.orchestration import scheduler_implementation
from nro.orchestration.branch_registry import SCHEMA_VERSION as BRANCH_SCHEMA_VERSION
from nro.orchestration.branch_registry import BranchRegistry
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import SCHEMA_VERSION, RegistryLock
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.worker_control import stop_worker_pool_for_repair


def repair_scientific_schemas(registry) -> list[dict]:
    """Recover branch mappings and rebuild incompatible scientific registries."""
    from nro.orchestration.branch_repair import (
        _recover_public_work_items,
        _repair_records_locked,
    )

    branches = BranchStore(registry.paths.control)
    if not branches.path.is_file():
        return []
    topology = branches.read().topology
    repaired = []
    records = sorted(
        topology.records.values(),
        key=lambda record: (len(topology.ancestors(record.name)), record.name),
    )
    for record in records:
        scientific = BranchRegistry(registry.paths.control, record)
        stored = scientific.stored_schema_version()
        unavailable = (
            _recover_public_work_items(
                registry,
                branch=record.name,
                registry_id=record.registry_id,
            )
            if record.name != "main"
            else []
        )
        if stored != scientific.stored_schema_version():
            raise RuntimeError("Scientific registry schema changed during repair")
        if stored != BRANCH_SCHEMA_VERSION:
            with registry.connection(write=True) as db:
                work_items = _repair_records_locked(
                    db,
                    branch=record.name,
                    registry_id=record.registry_id,
                )
            scientific.rebuild([], work_items)
            repaired.append(
                {
                    "branch": record.name,
                    "stored_schema": stored,
                    "schema": scientific.stored_schema_version(),
                    "backup": str(scientific.root / "registry-before-repair.sqlite3"),
                    "work_items": len(work_items),
                    "unavailable": unavailable,
                }
            )
        elif unavailable:
            repaired.append(
                {
                    "branch": record.name,
                    "stored_schema": stored,
                    "schema": stored,
                    "backup": None,
                    "work_items": None,
                    "unavailable": unavailable,
                }
            )
    return repaired


def _rebuild(registry) -> dict:
    """Replace quiescent scheduler state and recover records backed by public files."""
    backup = registry.reinitialize(preserve_branch_runtime=True, retain_backup=True)
    from nro.orchestration.discovery import register_existing_artifacts
    from nro.orchestration.selection import discover_bids_inventory

    inventory = discover_bids_inventory(registry.paths.bids_root)
    registry.replace_bids_inventory(inventory)
    found = register_existing_artifacts(
        registry, bids_root=registry.paths.bids_root, inventory=inventory
    )
    scientific = repair_scientific_schemas(registry)
    branch_unavailable = [
        f"{item['branch']}: {message}" for item in scientific for message in item["unavailable"]
    ]
    return dict(
        repaired=True,
        registry=str(registry.paths.database),
        backup=str(backup),
        work_items=found.work_items,
        unavailable=[*found.unavailable, *branch_unavailable],
        schema=SCHEMA_VERSION,
        scientific=scientific,
    )


def _bound_checkout(control: Path) -> Path | None:
    """Read the shared checkout from a valid scheduler binding, when present."""
    path = scheduler_implementation.implementation_path(control)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        checkout = value["checkout"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Scheduler installation binding is invalid") from error
    if not isinstance(checkout, str) or not Path(checkout).is_absolute():
        raise ValueError("Scheduler installation binding is invalid")
    return Path(checkout).resolve()


def repair_for_installation(registry, *, checkout: Path) -> dict:
    """Rebuild an incompatible scheduler after tagged-source validation and pool shutdown."""
    checkout = checkout.resolve()
    branches = BranchStore(registry.paths.control)
    if branches.read().topology.require_checkout(checkout) != "main":
        raise ValueError("Only main can rebuild the shared scheduler during installation")
    if _bound_checkout(registry.paths.control) != checkout:
        raise ValueError("Installation repair must use the designated shared checkout")
    with RegistryLock(
        branches.root / "scheduler-repair.lock", branches.root / "scheduler-repair.recovery-lock"
    ):
        activity = registry.worker_pool_activity(for_repair=True)
        if activity["workers"] or activity["submissions"]:
            raise ValueError("Stop the shared worker pool before rebuilding its scheduler schema")
        return _rebuild(registry)


def repair(registry, *, checkout: Path, confirm, allow_release_transition: bool = False) -> dict:
    """Stop the whole pool and rebuild shared and incompatible scientific state.

    Requests and attempt history are removed from active state. Backups retain
    replaced databases and private certificates. Public artifacts remain in
    place and current work-item contracts are rediscovered or registered on demand.
    """
    checkout = checkout.resolve()
    if checkout != CHECKOUT:
        raise ValueError("Run scheduler repair using the accepting main checkout’s Python")
    branches = BranchStore(registry.paths.control)
    if branches.read().topology.require_checkout(checkout) != "main":
        raise ValueError("Only main can repair shared scheduling state")
    try:
        ReleaseStore(branches).require_approved(checkout)
    except ValueError:
        if not allow_release_transition:
            raise
        from nro.orchestration.releases import tagged_source

        tagged_source(checkout)
    bound_checkout = _bound_checkout(registry.paths.control)
    if bound_checkout is not None and bound_checkout != checkout:
        raise ValueError("Repair must run from the designated shared installation")
    with RegistryLock(
        branches.root / "scheduler-repair.lock", branches.root / "scheduler-repair.recovery-lock"
    ):
        activity = registry.worker_pool_activity(for_repair=True)
        if not confirm(activity):
            raise ValueError("Shared scheduler repair cancelled")
        stop_worker_pool_for_repair(registry)
        return _rebuild(registry)
