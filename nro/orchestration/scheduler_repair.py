"""Explicit main-maintainer replacement of shared scheduling state."""

import json
from pathlib import Path

from nro.configuration.site import CHECKOUT
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import RegistryLock
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.scheduler_implementation import implementation_path
from nro.orchestration.worker_control import stop_worker_pool_for_repair


def repair(registry, *, checkout: Path, confirm) -> dict:
    """Stop the whole pool and rebuild shared state, preserving branch databases.

    This is a release-maintenance operation, distinct from branch scientific
    repair. Requests and attempt history are removed from active state. A backup
    retains the old database and private certificates. Public artifacts and
    scientific runtime configurations remain in place.
    """
    checkout = checkout.resolve()
    if checkout != CHECKOUT:
        raise ValueError("Run scheduler repair using the accepting main checkout’s Python")
    branches = BranchStore(registry.paths.control)
    if branches.read().topology.require_checkout(checkout) != "main":
        raise ValueError("Only main can repair shared scheduling state")
    ReleaseStore(branches).require_approved(checkout)
    binding = implementation_path(registry.paths.control)
    if binding.is_file() and Path(json.loads(binding.read_text())["checkout"]) != checkout:
        raise ValueError("Repair must run from the designated shared installation")
    with RegistryLock(
        branches.root / "scheduler-repair.lock", branches.root / "scheduler-repair.recovery-lock"
    ):
        activity = registry.worker_pool_activity(for_repair=True)
        if not confirm(activity):
            raise ValueError("Shared scheduler repair cancelled")
        stop_worker_pool_for_repair(registry)
        backup = registry.reinitialize(preserve_branch_runtime=True, retain_backup=True)
        from nro.orchestration.discovery import register_existing_artifacts
        from nro.orchestration.selection import discover_bids_inventory

        inventory = discover_bids_inventory(registry.paths.bids_root)
        registry.replace_bids_inventory(inventory)
        found = register_existing_artifacts(
            registry, bids_root=registry.paths.bids_root, inventory=inventory
        )
        return dict(
            repaired=True,
            registry=str(registry.paths.database),
            backup=str(backup),
            instances=found.instances,
            unavailable=list(found.unavailable),
        )
