"""Repair scientific records without replacing the shared scheduler."""

import json
import time
from pathlib import Path

from nro.orchestration.artifact_resolution import scientific_contracts
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.compiled_request import decode_spec
from nro.orchestration.registry import utcnow


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
            """SELECT COUNT(*) FROM attempts a JOIN instance_execution e ON e.instance_id=a.instance_id
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
            "UPDATE request_instances SET demand_state='cancelled' WHERE request_id IN (SELECT request_id FROM request_owners WHERE registry_id=?)",
            (owner,),
        )
        db.execute(
            """UPDATE attempts SET state='cancel_requested',error_type='BranchRepair',
            error_message='Branch scientific registry repair requested'
            WHERE state IN ('queued','running') AND instance_id IN
            (SELECT instance_id FROM instance_execution WHERE registry_id=?)""",
            (owner,),
        )
    deadline = time.monotonic() + 30
    while True:
        with registry.connection() as db:
            active = db.execute(
                """SELECT 1 FROM attempts a JOIN instance_execution e ON e.instance_id=a.instance_id
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
    records = {}
    workflows = []
    with registry.connection() as db:
        payloads = [
            json.loads(row[0])
            for row in db.execute(
                """SELECT p.payload_json FROM request_plans p
            JOIN request_owners o ON o.request_id=p.request_id WHERE o.registry_id=?""",
                (owner,),
            )
        ]
        for payload in payloads:
            workflows.append(payload["workflow"])
            contracts = scientific_contracts(
                tuple(decode_spec(value) for value in payload["specifications"])
            )
            for logical_key, contract in contracts.items():
                revision = payload["revisions"][logical_key]
                if logical_key not in records or records[logical_key]["revision"] < revision:
                    records[logical_key] = dict(
                        key=logical_key, revision=revision, contract=contract
                    )
        for row in db.execute(
            """SELECT b.logical_key,b.scientific_contract_json,COALESCE(r.revision,1) AS revision
            FROM branch_instances b LEFT JOIN compiled_revisions r ON r.registry_id=b.registry_id AND r.logical_key=b.logical_key
            WHERE b.registry_id=?""",
            (owner,),
        ):
            records[row["logical_key"]] = dict(
                key=row["logical_key"],
                revision=row["revision"],
                contract=json.loads(row["scientific_contract_json"]),
            )
        if name == "main":
            from nro.orchestration.branch_reconciliation import candidates_locked

            for project in [row[0] for row in db.execute("SELECT DISTINCT project FROM instances")]:
                for item in candidates_locked(db, project, fresh_only=False):
                    if item.branch == "main":
                        records.setdefault(
                            item.key, dict(key=item.key, revision=1, contract=dict(item.contract))
                        )
    return dict(
        branch=name, workflows=workflows, instances=list(records.values()), reservation=reservation
    )


def finish(registry, *, checkout: Path, reservation: str) -> dict:
    """Release this repair reservation without creating demand or changing capacity."""
    topology = BranchStore(registry.paths.control).read().topology
    name = topology.require_checkout(checkout)
    with registry.connection(write=True) as db:
        count = db.execute(
            "DELETE FROM metadata WHERE key=? AND value=?",
            ("branch_maintenance:" + topology.records[name].registry_id, reservation),
        ).rowcount
        if count != 1:
            raise ValueError("Branch maintenance reservation changed")
    return {"repaired": True, "branch": name}


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
        scientific.rebuild(data["workflows"], data["instances"])
        result = maintenance(
            control,
            bids_root,
            checkout=checkout,
            operation="repair_finish",
            reservation=reservation,
        )
        return dict(
            result,
            instances=len(data["instances"]),
            requests=[],
            submitted_workers=[],
            registry=str(scientific.database),
            backup=str(scientific.root / "registry-before-repair.sqlite3"),
        )
