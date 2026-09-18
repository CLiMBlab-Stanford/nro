"""Repair scientific records without replacing the shared scheduler."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.manifests import assess_registry
from nro.orchestration.ownership import (
    complete_ownership_records,
    materialize_work_item_specs,
    read_ownership_records,
)
from nro.orchestration.registry import utcnow


@dataclass(frozen=True)
class PublicOwnership:
    """Validated lineage roots and work-item receipts in a branch namespace."""

    lineages: list[dict]
    work_items: list[tuple[dict, Path]]
    errors: list[str]


def _has_public_evidence(row) -> bool:
    """Return whether a registered work item still has a declared public file."""
    paths = [Path(value) for value in json.loads(row["expected_outputs_json"])]
    return any(path.is_file() for path in paths)


def _ancestor_work_item(
    db,
    topology,
    branch: str,
    logical_key: str,
) -> tuple[str, int] | None:
    """Resolve one inherited logical key through nearest-first branch precedence."""
    for ancestor in topology.ancestors(branch)[1:]:
        owner = topology.records[ancestor].registry_id
        row = db.execute(
            """SELECT i.work_item_key,i.id FROM branch_work_items b
               JOIN work_items i ON i.id=b.work_item_id
               WHERE b.registry_id=? AND b.logical_key=?""",
            (owner, logical_key),
        ).fetchone()
        if row is not None:
            return str(row["work_item_key"]), int(row["id"])
    return None


def _recover_public_work_items(
    registry,
    *,
    branch: str,
    registry_id: str,
    ownership: PublicOwnership | None = None,
) -> list[str]:
    """Restore one branch's current ownership receipts to the shared scheduler."""
    ownership = ownership or _public_ownership_records(registry, branch=branch)
    lineages = ownership.lineages
    records = ownership.work_items
    errors = list(ownership.errors)
    if not lineages or not records:
        return errors

    lineage_ids = registry.register_owned_lineages(lineages, branch_registry_id=registry_id)
    specs, materialization_errors = materialize_work_item_specs(
        registry,
        records,
        lineage_ids,
        namespace=registry_id,
    )
    errors.extend(materialization_errors)
    contracts = {
        str(record["work_item_key"]): record["scientific_contract"] for record, _ in records
    }
    local = {spec.key for spec in specs}
    topology = BranchStore(registry.paths.control).read().topology
    external: dict[str, tuple[str, int]] = {}
    unresolved: dict[str, set[str]] = {}
    with registry.connection() as db:
        for spec in specs:
            for dependency in spec.dependencies:
                if dependency in local or dependency in external:
                    continue
                inherited = _ancestor_work_item(db, topology, branch, dependency)
                if inherited is None:
                    unresolved.setdefault(spec.key, set()).add(dependency)
                else:
                    external[dependency] = inherited
    while unresolved:
        removed = set(unresolved)
        next_unresolved = {
            spec.key: {dependency for dependency in spec.dependencies if dependency in removed}
            for spec in specs
            if spec.key not in removed
            and any(dependency in removed for dependency in spec.dependencies)
        }
        for key, dependencies in sorted(unresolved.items()):
            errors.append(
                f"Stored work item {key} lacks dependencies: {', '.join(sorted(dependencies))}"
            )
        specs = [spec for spec in specs if spec.key not in removed]
        local.difference_update(removed)
        unresolved = next_unresolved

    if not specs:
        return errors
    logical_by_scheduler = {f"{registry_id}:{spec.key}": spec.key for spec in specs}
    external_ids = {scheduler_key: item_id for scheduler_key, item_id in external.values()}
    compiled = []
    for spec in specs:
        dependencies = tuple(
            f"{registry_id}:{dependency}" if dependency in local else external[dependency][0]
            for dependency in spec.dependencies
        )
        compiled.append(
            spec.evolve(
                key=f"{registry_id}:{spec.key}",
                dependencies=dependencies,
            )
        )
    with registry.connection(write=True) as db:
        ids = registry._upsert_work_item_graph_locked(
            db,
            tuple(
                (
                    spec,
                    spec.as_record(
                        compiled_contract=spec.contract.as_dict(spec.identity),
                    ),
                )
                for spec in compiled
            ),
            now=utcnow(),
            external_ids=external_ids,
            owner_branch=branch,
        )
        for scheduler_key, logical_key in logical_by_scheduler.items():
            work_item_id = ids[scheduler_key]
            contract = contracts[logical_key]
            db.execute(
                """INSERT INTO branch_work_items VALUES (?,?,?,?)
                   ON CONFLICT(registry_id,logical_key) DO UPDATE SET
                   work_item_id=excluded.work_item_id,
                   scientific_contract_json=excluded.scientific_contract_json""",
                (registry_id, logical_key, work_item_id, json.dumps(contract)),
            )
            db.execute(
                """INSERT INTO compiled_revisions VALUES (?,?,?,?)
                   ON CONFLICT(registry_id,logical_key) DO UPDATE SET
                   fingerprint=excluded.fingerprint""",
                (registry_id, logical_key, 1, fingerprint(contract)),
            )
    # Ownership receipts preserve the contract that produced the public files.
    # Current branch code is registered later, after its scientific registry has
    # been rebuilt.  Validate the recovered files against their recorded contract
    # here; assessing them against the scheduler's implementation would apply
    # main-branch policy to development-branch artifacts.
    assess_registry(
        registry,
        projects=tuple(sorted({spec.project for spec in specs})),
        compiled=True,
        recover_public=True,
    )
    return errors


def _public_ownership_records(registry, *, branch: str) -> PublicOwnership:
    """Read complete public ownership records for one branch output namespace."""
    from nro.configuration.site import settings

    values = settings()[0]
    configured_bids = Path(values["bids"]).expanduser().resolve()
    development = (
        Path(values["development"])
        if registry.paths.bids_root == configured_bids
        else registry.paths.bids_root.parent / "NRO_DEV"
    )
    paths = BranchPaths(
        branch,
        registry.paths.bids_root,
        Path(values["work"])
        if registry.paths.bids_root == configured_bids
        else registry.paths.bids_root.parent / "WORK",
        development,
    )
    output_bids = paths.output_bids
    if not output_bids.is_dir():
        return PublicOwnership([], [], [])
    projects = sorted(path.name for path in output_bids.iterdir() if path.is_dir())
    lineages, records, errors = read_ownership_records(output_bids, projects)
    lineages, records, incomplete = complete_ownership_records(lineages, records)
    errors.extend(incomplete)
    return PublicOwnership(lineages, records, errors)


def _repair_records_locked(db, *, branch: str, registry_id: str) -> list[dict]:
    """Retain disk-backed branch records and the dependencies needed to describe them."""
    mappings = [
        dict(row)
        for row in db.execute(
            """SELECT b.logical_key,b.scientific_contract_json,b.work_item_id,
                      i.expected_outputs_json,
                      COALESCE(r.revision,1) AS revision
               FROM branch_work_items b
               JOIN work_items i ON i.id=b.work_item_id
               LEFT JOIN compiled_revisions r
                 ON r.registry_id=b.registry_id AND r.logical_key=b.logical_key
               WHERE b.registry_id=?""",
            (registry_id,),
        )
    ]
    candidates = {int(row["work_item_id"]) for row in mappings}
    unowned_main: set[int] = set()
    if branch == "main":
        unowned_main.update(
            int(row["id"])
            for row in db.execute(
                """SELECT i.id FROM work_items i
                   LEFT JOIN work_item_execution e ON e.work_item_id=i.id
                   WHERE e.work_item_id IS NULL"""
            )
        )
        candidates.update(unowned_main)
    if not candidates:
        return []
    placeholders = ",".join("?" for _ in candidates)
    rows = {
        int(row["id"]): dict(row)
        for row in db.execute(
            f"SELECT id,expected_outputs_json FROM work_items WHERE id IN ({placeholders})",
            tuple(sorted(candidates)),
        )
    }
    retained = {work_item_id for work_item_id, row in rows.items() if _has_public_evidence(row)}
    dependencies: dict[int, set[int]] = {}
    for row in db.execute(
        f"""SELECT work_item_id,upstream_work_item_id FROM work_item_dependencies
            WHERE work_item_id IN ({placeholders})""",
        tuple(sorted(candidates)),
    ):
        dependencies.setdefault(int(row["work_item_id"]), set()).add(
            int(row["upstream_work_item_id"])
        )
    pending = list(retained)
    while pending:
        for upstream in dependencies.get(pending.pop(), ()):
            if upstream in candidates and upstream not in retained:
                retained.add(upstream)
                pending.append(upstream)

    removed = [row["logical_key"] for row in mappings if int(row["work_item_id"]) not in retained]
    if removed:
        db.executemany(
            "DELETE FROM branch_work_items WHERE registry_id=? AND logical_key=?",
            [(registry_id, key) for key in removed],
        )
        db.executemany(
            "DELETE FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
            [(registry_id, key) for key in removed],
        )

    records = {
        row["logical_key"]: {
            "key": row["logical_key"],
            "revision": row["revision"],
            "contract": json.loads(row["scientific_contract_json"]),
        }
        for row in mappings
        if int(row["work_item_id"]) in retained
    }
    if branch == "main" and retained:
        from nro.orchestration.branch_reconciliation import candidates_locked

        projects = [
            row[0]
            for row in db.execute(
                f"SELECT DISTINCT project FROM work_items WHERE id IN ({placeholders})",
                tuple(sorted(candidates)),
            )
        ]
        for project in projects:
            for item in candidates_locked(db, project, fresh_only=False):
                work_item_id = int(item.evidence["work_item_id"])
                if (
                    item.branch == "main"
                    and work_item_id in retained
                    and work_item_id in unowned_main
                ):
                    contract = dict(item.contract)
                    records.setdefault(item.key, dict(key=item.key, revision=1, contract=contract))
                    db.execute(
                        "INSERT OR IGNORE INTO branch_work_items VALUES (?,?,?,?)",
                        (registry_id, item.key, work_item_id, json.dumps(contract)),
                    )
                    db.execute(
                        "INSERT OR IGNORE INTO compiled_revisions VALUES (?,?,?,?)",
                        (registry_id, item.key, 1, fingerprint(contract)),
                    )
    return [records[key] for key in sorted(records)]


def prepare(
    registry, *, checkout: Path, reservation: str | None = None, allow_stop: bool = False
) -> dict:
    """Inspect or reserve branch repair, cancel its demand, and await its attempts."""
    topology = BranchStore(registry.paths.control).read().topology
    name = topology.require_checkout(checkout)
    owner = topology.records[name].registry_id
    key = "branch_maintenance:" + owner
    with registry.connection(write=reservation is not None) as db:
        active = db.execute(
            """SELECT COUNT(*) FROM attempts a JOIN work_item_execution e ON e.work_item_id=a.work_item_id
            WHERE e.registry_id=? AND a.state IN ('queued','running','cancel_requested')""",
            (owner,),
        ).fetchone()[0]
        requests = db.execute(
            "SELECT COUNT(*) FROM requests r JOIN request_owners o ON o.request_id=r.id WHERE o.registry_id=? AND r.state='active'",
            (owner,),
        ).fetchone()[0]
        if reservation is None:
            return dict(branch=name, attempts=active, requests=requests)
        if active and not allow_stop:
            raise ValueError("Branch attempts are active; confirm stopping them before repair")
        db.execute(
            "INSERT INTO metadata(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, reservation),
        )
        db.execute(
            "UPDATE requests SET state='cancelled',updated_at=? WHERE id IN (SELECT request_id FROM request_owners WHERE registry_id=?) AND state='active'",
            (utcnow(), owner),
        )
        db.execute(
            "UPDATE request_work_items SET demand_state='cancelled' WHERE request_id IN (SELECT request_id FROM request_owners WHERE registry_id=?)",
            (owner,),
        )
        db.execute(
            """UPDATE attempts SET state='cancel_requested',error_type='BranchRepair',
            error_message='Branch scientific registry repair requested'
            WHERE state IN ('queued','running') AND work_item_id IN
            (SELECT work_item_id FROM work_item_execution WHERE registry_id=?)""",
            (owner,),
        )
    deadline = time.monotonic() + 30
    while True:
        with registry.connection() as db:
            active = db.execute(
                """SELECT 1 FROM attempts a JOIN work_item_execution e ON e.work_item_id=a.work_item_id
                WHERE e.registry_id=? AND a.state IN ('queued','running','cancel_requested') LIMIT 1""",
                (owner,),
            ).fetchone()
        if active is None:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Branch attempts have not stopped; repair remains reserved. Retry after shutdown completes"
            )
        registry.recover_orphaned_attempts()
        time.sleep(0.1)
    ownership = _public_ownership_records(registry, branch=name)
    recovery_errors = list(ownership.errors)
    if name != "main":
        recovery_errors = (
            _recover_public_work_items(
                registry,
                branch=name,
                registry_id=owner,
                ownership=PublicOwnership(ownership.lineages, ownership.work_items, []),
            )
            + recovery_errors
        )
    with registry.connection(write=True) as db:
        records = _repair_records_locked(db, branch=name, registry_id=owner)
    return dict(
        branch=name,
        work_items=records,
        owned_lineages=ownership.lineages,
        unavailable=recovery_errors,
        reservation=reservation,
    )


def finish(registry, *, checkout: Path, reservation: str, workflows: list[dict]) -> dict:
    """Restore current workflow bindings and release this repair reservation."""
    from nro.orchestration.branch_admission import _workflow

    topology = BranchStore(registry.paths.control).read().topology
    name = topology.require_checkout(checkout)
    owner = topology.records[name].registry_id
    with registry.connection(write=True) as db:
        row = db.execute(
            "SELECT value FROM metadata WHERE key=?", ("branch_maintenance:" + owner,)
        ).fetchone()
        if row is None or row[0] != reservation:
            raise ValueError("Branch maintenance reservation changed")
        for payload in workflows:
            _workflow(db, payload, owner)
        db.execute("DELETE FROM metadata WHERE key=?", ("branch_maintenance:" + owner,))
    return {"repaired": True, "branch": name, "workflows": len(workflows)}


def _register_current_workflows(scientific, *, store=None) -> list[dict]:
    """Compile current definitions and detach their scheduler-facing bindings."""
    from nro.configuration.store import ConfigStore
    from nro.orchestration.compiled_request import export_workflow

    store = store or ConfigStore()
    return [
        export_workflow(scientific, scientific.register_workflow(store.resolve(workflow_id)))
        for workflow_id in store.workflow_ids()
    ]


def repair_checkout(control: Path, bids_root: Path, checkout: Path, *, confirm) -> dict:
    """Rebuild this branch's scientific database after confirmed scoped shutdown."""
    import uuid

    from nro.orchestration.branch_registry import BranchRegistry
    from nro.orchestration.registry import RegistryLock
    from nro.orchestration.scheduler_client import maintenance

    topology = BranchStore(control).read().topology
    name = topology.require_checkout(checkout)
    scientific = BranchRegistry(control, topology.records[name])
    with RegistryLock(scientific.root / "repair.lock", scientific.root / "repair.recovery-lock"):
        activity = maintenance(control, bids_root, checkout=checkout, operation="repair_prepare")
        allowed = bool(activity["attempts"] and confirm(activity))
        if activity["attempts"] and not allowed:
            raise ValueError("Branch repair cancelled; no state was changed")
        reservation = uuid.uuid4().hex
        data = maintenance(
            control,
            bids_root,
            checkout=checkout,
            operation="repair_prepare",
            reservation=reservation,
            allow_stop=allowed,
        )
        # Historical workflow snapshots remain on disk for provenance. Only
        # definitions that resolve now make a recovered lineage reproducible.
        scientific.rebuild([], data["work_items"], owned_lineages=data.get("owned_lineages", []))
        workflows = _register_current_workflows(scientific)
        result = maintenance(
            control,
            bids_root,
            checkout=checkout,
            operation="repair_finish",
            reservation=reservation,
            workflows=workflows,
        )
        return dict(
            result,
            work_items=len(data["work_items"]),
            unavailable=data.get("unavailable", []),
            requests=[],
            submitted_workers=[],
            registry=str(scientific.database),
            backup=str(scientific.root / "registry-before-repair.sqlite3"),
        )
