"""Transactional shared-maintenance tests."""

import json
from pathlib import Path

import pytest

from nro.engine.maintenance import MaintenanceJournal, audit_shared_state
from nro.orchestration.branch_registry import BranchRegistry
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import Registry
from nro.orchestration.scheduler_repair import _rebuild


def test_audit_is_read_only_and_reports_all_registry_schemas(tmp_path: Path) -> None:
    bids = tmp_path / "BIDS"
    registry = Registry.for_project("", bids_root=bids)
    registry.initialize()
    branches = BranchStore(registry.paths.control)
    branches.initialize()
    before = registry.paths.database.read_bytes()

    audit = audit_shared_state(registry, {})

    assert audit.scheduler_schema == registry.stored_schema_version()
    assert set(audit.branch_schemas) == {"main", "dev"}
    assert audit.fatal_errors == ()
    assert registry.paths.database.read_bytes() == before


def test_maintenance_journal_resumes_one_checkout_transaction(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    first = MaintenanceJournal(tmp_path / "control", checkout)
    first.record("audited", detail="one")
    second = MaintenanceJournal(tmp_path / "control", checkout)

    assert second.identifier == first.identifier
    second.finish(detail="done")

    assert not second.path.exists()
    completed = json.loads((second.root / f"{first.identifier}.json").read_text())
    assert completed["phase"] == "complete"


def test_failed_rebuild_restores_scheduler_database(tmp_path: Path, monkeypatch) -> None:
    bids = tmp_path / "BIDS"
    registry = Registry.for_project("", bids_root=bids)
    registry.initialize()
    BranchStore(registry.paths.control).initialize()
    before = registry.paths.database.read_bytes()

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected discovery failure")

    monkeypatch.setattr("nro.orchestration.discovery.register_existing_artifacts", fail)
    with pytest.raises(RuntimeError, match="injected discovery failure"):
        _rebuild(registry)

    assert registry.paths.database.read_bytes() == before
    registry.initialize()


def test_failed_rebuild_restores_every_branch_database(tmp_path: Path, monkeypatch) -> None:
    bids = tmp_path / "BIDS"
    registry = Registry.for_project("", bids_root=bids)
    registry.initialize()
    branches = BranchStore(registry.paths.control)
    snapshot = branches.initialize()
    scientific = {
        name: BranchRegistry(registry.paths.control, record)
        for name, record in snapshot.topology.records.items()
    }
    for branch in scientific.values():
        branch.initialize()
    databases = {name: branch.database for name, branch in scientific.items()}
    before = {name: path.read_bytes() for name, path in databases.items()}

    def fail_after_mutation(_registry):
        for path in databases.values():
            path.write_bytes(b"incomplete replacement")
        raise RuntimeError("injected branch rebuild failure")

    monkeypatch.setattr(
        "nro.orchestration.scheduler_repair.repair_scientific_schemas", fail_after_mutation
    )
    with pytest.raises(RuntimeError, match="injected branch rebuild failure"):
        _rebuild(registry)

    assert {name: path.read_bytes() for name, path in databases.items()} == before
