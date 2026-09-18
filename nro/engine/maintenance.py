"""Read-only audit and durable phase records for shared maintenance."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.branch_registry import SCHEMA_VERSION as BRANCH_SCHEMA_VERSION
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.ownership import read_ownership_records
from nro.orchestration.registry_schema import SCHEMA_VERSION as SCHEDULER_SCHEMA_VERSION


@dataclass(frozen=True)
class MaintenanceAudit:
    """Complete read-only assessment made before shared state is changed."""

    scheduler_schema: int | None
    branch_schemas: Mapping[str, int | None]
    integrity_errors: tuple[str, ...]
    ownership_errors: tuple[str, ...]

    @property
    def fatal_errors(self) -> tuple[str, ...]:
        """Return contradictions that make a safe rebuild impossible."""
        return self.integrity_errors

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable audit record."""
        return {
            "expected_scheduler_schema": SCHEDULER_SCHEMA_VERSION,
            "scheduler_schema": self.scheduler_schema,
            "expected_branch_schema": BRANCH_SCHEMA_VERSION,
            "branch_schemas": dict(self.branch_schemas),
            "integrity_errors": list(self.integrity_errors),
            "ownership_errors": list(self.ownership_errors),
        }


def _inspect_database(path: Path, application_id: int) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            if int(connection.execute("PRAGMA application_id").fetchone()[0]) != application_id:
                return None, f"Not an nro database: {path}"
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                return None, f"SQLite integrity check failed for {path}: {integrity}"
            return int(connection.execute("PRAGMA user_version").fetchone()[0]), None
        finally:
            connection.close()
    except sqlite3.Error as error:
        return None, f"Cannot inspect {path}: {error}"


def audit_shared_state(registry, inventory: Mapping[str, tuple[str, ...]]) -> MaintenanceAudit:
    """Inspect registries and public ownership without changing either."""
    from nro.orchestration.branch_registry import APPLICATION_ID as BRANCH_APPLICATION_ID
    from nro.orchestration.registry_schema import APPLICATION_ID as SCHEDULER_APPLICATION_ID

    scheduler_schema, scheduler_error = _inspect_database(
        registry.paths.database, SCHEDULER_APPLICATION_ID
    )
    errors = [scheduler_error] if scheduler_error else []
    branches = BranchStore(registry.paths.control)
    branch_schemas: dict[str, int | None] = {}
    if branches.path.is_file():
        try:
            records = branches.read().topology.records
        except (OSError, ValueError, json.JSONDecodeError) as error:
            records = {}
            errors.append(f"Cannot read branch registration: {error}")
        for name in sorted(records):
            path = ControlPaths(registry.paths.control).branch(name) / "registry.sqlite3"
            schema, error = _inspect_database(path, BRANCH_APPLICATION_ID)
            branch_schemas[name] = schema
            if error:
                errors.append(error)
    _lineages, _work_items, ownership_errors = read_ownership_records(
        registry.paths.bids_root, inventory
    )
    return MaintenanceAudit(
        scheduler_schema,
        branch_schemas,
        tuple(errors),
        tuple(ownership_errors),
    )


class MaintenanceJournal:
    """Durably record one idempotent shared-maintenance transition."""

    def __init__(self, control: Path, checkout: Path) -> None:
        """Open or resume the transaction journal for this checkout."""
        self.root = ControlPaths(control).shared / "maintenance"
        self.path = self.root / "active.json"
        self.checkout = str(checkout.resolve())
        self.identifier = uuid.uuid4().hex
        if self.path.is_file():
            try:
                current = json.loads(self.path.read_text())
                if current.get("checkout") == self.checkout:
                    self.identifier = str(current["transaction"])
            except (OSError, ValueError, KeyError, TypeError):
                pass

    def record(self, phase: str, **details: object) -> None:
        """Publish the current phase atomically for recovery and diagnosis."""
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.path,
            {
                "transaction": self.identifier,
                "checkout": self.checkout,
                "phase": phase,
                **details,
            },
            durable=True,
        )

    def finish(self, **details: object) -> None:
        """Retain a terminal record and release the active transaction marker."""
        self.record("complete", **details)
        completed = self.root / f"{self.identifier}.json"
        self.path.replace(completed)
