"""Persist branch registrations under the shared scheduler's control directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from nro.configuration.store import fingerprint
from nro.engine.io import atomic_write_json
from nro.orchestration.artifact_resolution import ArtifactCandidate
from nro.orchestration.branch_planning import BranchPlan, resolve_branch_plan
from nro.orchestration.branch_registry import BranchRegistry
from nro.orchestration.branches import BranchPaths, BranchRecord, BranchTopology
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryLock, ensure_shared_directory


@dataclass(frozen=True)
class BranchSnapshot:
    """A validated inheritance tree and the revision required to change it."""

    topology: BranchTopology
    revision: str


def _encode(topology: BranchTopology) -> dict:
    return {
        name: {
            "parent": record.parent,
            "checkouts": [str(path) for path in record.checkouts],
            "retired": record.retired,
            "registry_id": record.registry_id,
        }
        for name, record in sorted(topology.records.items())
    }


def _decode(value: object) -> BranchTopology:
    if not isinstance(value, dict):
        raise ValueError("Branch registrations must be an object")
    records = {}
    for name, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {
            "parent",
            "checkouts",
            "retired",
            "registry_id",
        }:
            raise ValueError(f"Invalid branch registration: {name}")
        checkouts = entry["checkouts"]
        if (
            not isinstance(checkouts, list)
            or not all(isinstance(path, str) and Path(path).is_absolute() for path in checkouts)
            or type(entry["retired"]) is not bool
        ):
            raise ValueError(f"Invalid branch registration: {name}")
        records[name] = BranchRecord(
            name,
            entry["parent"],
            tuple(map(Path, checkouts)),
            entry["retired"],
            entry["registry_id"],
        )
    return BranchTopology(records)


class BranchStore:
    """Serialize registration edits and reject updates based on an outdated tree.

    This store reserves names and authorizes checkouts. Creating it does not
    enable branch execution, move artifacts, or change Git refs. Its records
    live separately from the rebuildable work registry and survive its repair.
    """

    def __init__(self, control: Path, *, lock_timeout: float = 120) -> None:
        """Select the shared control directory without reading or creating it."""
        self.control = Path(control).expanduser().resolve()
        paths = ControlPaths(self.control)
        paths.require_current_layout()
        self.root = paths.shared
        self.path = paths.catalog
        self.pending = self.root / "registration-pending.json"
        self.lock_timeout = lock_timeout

    def _lock(self) -> RegistryLock:
        ensure_shared_directory(self.root)
        return RegistryLock(
            self.root / "edit.lock", self.root / "edit.recovery-lock", timeout=self.lock_timeout
        )

    def read(self) -> BranchSnapshot:
        """Read the last published tree; unfinished registrations remain invisible."""
        topology = _decode(json.loads(self.path.read_text()))
        return BranchSnapshot(topology, fingerprint(_encode(topology)))

    def _finish_pending(self) -> None:
        if not self.pending.exists():
            return
        pending = json.loads(self.pending.read_text())
        if not isinstance(pending, dict) or set(pending) != {"before_revision", "records"}:
            raise ValueError("Invalid pending branch registration")
        if pending["before_revision"] is not None and not isinstance(
            pending["before_revision"], str
        ):
            raise ValueError("Invalid pending branch registration revision")
        after = _decode(pending["records"])
        before = self.read() if self.path.exists() else None
        current_revision = before.revision if before else None
        after_revision = fingerprint(_encode(after))
        if current_revision not in {pending["before_revision"], after_revision}:
            raise ValueError("Branch registration changed during an interrupted publication")
        if before:
            for name, record in before.topology.records.items():
                if (
                    name not in after.records
                    or record.registry_id != after.records[name].registry_id
                ):
                    raise ValueError(
                        "Branch registrations cannot discard or replace registry identities"
                    )
        for name, record in after.records.items():
            registry = BranchRegistry(self.control, record, lock_timeout=self.lock_timeout)
            if before and name in before.topology.records:
                registry.validate_identity()
            else:
                registry.initialize()
        atomic_write_json(self.path, _encode(after), mode=0o664, durable=True)
        self.pending.unlink()

    def _publish(self, before: BranchSnapshot | None, after: BranchTopology) -> BranchSnapshot:
        atomic_write_json(
            self.pending,
            {
                "before_revision": before.revision if before else None,
                "records": _encode(after),
            },
            mode=0o664,
            durable=True,
        )
        self._finish_pending()
        return self.read()

    def initialize(self) -> BranchSnapshot:
        """Reserve main/dev and their central scientific databases; resume an interrupted edit."""
        with self._lock():
            self._finish_pending()
            if not self.path.exists():
                return self._publish(None, BranchTopology.reserved())
            snapshot = self.read()
            for record in snapshot.topology.records.values():
                BranchRegistry(
                    self.control, record, lock_timeout=self.lock_timeout
                ).validate_identity()
            return snapshot

    def _update(
        self, revision: str, change: Callable[[BranchTopology], BranchTopology]
    ) -> BranchSnapshot:
        with self._lock():
            self._finish_pending()
            before = self.read()
            if before.revision != revision:
                raise ValueError(
                    "Branch registrations changed; read the current tree before retrying"
                )
            after = change(before.topology)
            return self._publish(before, after)

    def register(
        self,
        name: str,
        parent: str,
        *,
        revision: str,
        checkout: Path | None = None,
    ) -> BranchSnapshot:
        """Reserve a name and initialize its shared database, optionally attaching a checkout."""

        def change(tree: BranchTopology) -> BranchTopology:
            tree = tree.register(BranchRecord(name, parent))
            return tree.authorize_checkout(name, checkout) if checkout is not None else tree

        return self._update(revision, change)

    def authorize_checkout(self, name: str, checkout: Path, *, revision: str) -> BranchSnapshot:
        """Verify and register a named Git checkout without changing its branch."""
        return self._update(revision, lambda tree: tree.authorize_checkout(name, checkout))

    def reparent(
        self,
        name: str,
        parent: str,
        *,
        revision: str,
        checkout: Path,
    ) -> BranchSnapshot:
        """Change future inheritance without modifying snapshots held by existing requests."""

        def change(tree: BranchTopology) -> BranchTopology:
            tree.require_branch_checkout(name, checkout)
            return tree.reparent(name, parent)

        return self._update(revision, change)

    def retire(self, name: str, *, revision: str, checkout: Path) -> BranchSnapshot:
        """Stop future registrations for a branch while retaining its name and artifacts."""

        def change(tree: BranchTopology) -> BranchTopology:
            tree.require_branch_checkout(name, checkout)
            return tree.retire(name)

        return self._update(revision, change)

    def registry(self, name: str) -> BranchRegistry:
        """Resolve a registered branch's canonical scientific database, including retired branches."""
        with self._lock():
            records = self.read().topology.records
            if name not in records:
                raise ValueError(f"Unregistered branch: {name}")
            registry = BranchRegistry(self.control, records[name], lock_timeout=self.lock_timeout)
            registry.validate_identity()
            return registry

    def registry_for_checkout(self, checkout: Path) -> BranchRegistry:
        """Resolve a checkout through the central binding, never through a local database.

        Verify the current named Git branch and registered root on lookup.
        No registration or database is created by this operation.
        """
        with self._lock():
            tree = self.read().topology
            name = tree.require_checkout(checkout)
            registry = BranchRegistry(
                self.control, tree.records[name], lock_timeout=self.lock_timeout
            )
            registry.validate_identity()
            return registry

    def resolve_plan(
        self,
        checkout: Path,
        paths: BranchPaths,
        instances: Sequence[InstanceSpec],
        terminals: Sequence[str],
        candidates: Sequence[ArtifactCandidate],
        *,
        validate: Callable[[ArtifactCandidate], bool],
        inherit: bool = True,
    ) -> BranchPlan:
        """Authorize and record a scientific plan without submitting worker demand.

        Filesystem validation runs without holding the catalog lock. Recheck the
        checkout, topology, and scientific revisions before recording the plan.
        Concurrent edits reject the result; callers must resolve it again.
        Scientific records in ancestor branches are never changed.
        """
        with self._lock():
            snapshot = self.read()
            name = snapshot.topology.require_checkout(checkout)
            if paths.branch != name:
                raise ValueError("Planning paths do not belong to the authorized checkout")
            registry = BranchRegistry(
                self.control, snapshot.topology.records[name], lock_timeout=self.lock_timeout
            )
            revisions = {item.key: item.revision for item in registry.instances()}
        plan = resolve_branch_plan(
            snapshot.topology,
            paths,
            instances,
            terminals,
            candidates,
            validate=validate,
            inherit=inherit,
        )
        by_key = {item.key: item for item in instances}
        required: set[str] = set()
        pending = list(plan.terminals)
        while pending:
            key = pending.pop()
            if key not in required:
                required.add(key)
                pending.extend(by_key[key].dependencies)
        with self._lock():
            current = self.read()
            if (
                current.revision != snapshot.revision
                or current.topology.require_checkout(checkout) != name
            ):
                raise ValueError("Branch registration changed during planning; resolve again")
            registry.record_graph(
                tuple(by_key[key] for key in sorted(required)),
                expected_revisions={key: revisions.get(key) for key in required},
            )
        return plan
