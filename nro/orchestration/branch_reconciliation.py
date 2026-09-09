"""Rebind active requests using their stored scientific graphs and current artifacts."""

import json
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.artifact_resolution import ArtifactCandidate, scientific_contracts
from nro.orchestration.branch_planning import resolve_branch_plan
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.compiled_request import decode_spec
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.execution_cache import cache_publication
from nro.orchestration.execution_context import ExecutionContext


def candidates_locked(
    db, project: str, *, fresh_only: bool = True
) -> tuple[ArtifactCandidate, ...]:
    """Read compiled producer contracts; no source or scientific module is loaded."""
    rows = [
        dict(row)
        for row in db.execute(
            """SELECT i.*, c.directory_label,c.config_fingerprint,
        e.branch,e.logical_key,e.scientific_contract_json FROM instances i
        JOIN configuration_lineages c ON c.id=i.configuration_lineage_id
        LEFT JOIN instance_execution e ON e.instance_id=i.id WHERE i.project=?""",
            (project,),
        )
    ]
    by_id = {row["id"]: row for row in rows}
    parents = {}
    for edge in db.execute("SELECT instance_id,upstream_instance_id FROM instance_dependencies"):
        if edge["instance_id"] in by_id:
            parents.setdefault(edge["instance_id"], []).append(
                by_id[edge["upstream_instance_id"]]["instance_key"]
            )
    specs = []
    for row in rows:
        contract = json.loads(row["artifact_contract_json"])
        specs.append(
            InstanceSpec.create(
                key=row["instance_key"],
                module=row["module"],
                project=row["project"],
                participant=row["participant"],
                entities=json.loads(row["entities_json"]),
                scope=row["scope"],
                configuration_lineage_id=row["configuration_lineage_id"],
                config_fingerprint=row["config_fingerprint"],
                directory_label=row["directory_label"],
                runtime_config=Path(row["runtime_config_path"]),
                command=json.loads(row["command_json"]),
                dependencies=parents.get(row["id"], ()),
                input_paths=tuple(map(Path, json.loads(row["input_paths_json"]))),
                output_root=Path(row["output_root"]),
                output_prefix=row["output_prefix"],
                expected_outputs=tuple(map(Path, json.loads(row["expected_outputs_json"]))),
                output_format=contract["output"]["format"],
                processing=contract.get("processing", {}),
                resource_class=row["resource_class"],
            )
        )
    contracts = scientific_contracts(specs)
    return tuple(
        ArtifactCandidate(
            row["branch"] or "main",
            row["logical_key"] or row["instance_key"],
            json.loads(row["scientific_contract_json"])
            if row["scientific_contract_json"]
            else contracts[row["instance_key"]],
            row["current_generation"],
            Path(row["output_root"]),
            {"instance_id": row["id"]},
        )
        for row in rows
        if not fresh_only or row["artifact_state"] == "fresh" and row["current_generation"] >= 0
    )


def resolve_payload(topology, payload, candidates):
    """Resolve stored specifications without recompiling their scientific meaning."""
    if payload.get("protocol") != 1:
        raise ValueError("Unsupported compiled request protocol")
    context = ExecutionContext.from_dict(payload["context"])
    if (
        context.paths.branch != payload["branch"]
        or context.project != payload["project"]
        or topology.records[payload["branch"]].registry_id != payload["registry_id"]
    ):
        raise ValueError("Compiled request ownership changed")
    return resolve_branch_plan(
        topology,
        context.paths,
        tuple(decode_spec(value) for value in payload["specifications"]),
        payload["terminals"],
        candidates,
        validate=lambda _: True,
        inherit=payload["inherit"],
    )


@cache_publication
def reconcile_branch_requests(registry) -> int:
    """Replace unavailable inherited inputs locally, preserving request and code pins.

    Only active demand is reconsidered. A newer scientific request supersedes an
    older graph for the same branch identity. Captured attempt dependencies remain
    until shutdown even when the current request is rebound.
    """
    from nro.orchestration import dependency_state
    from nro.orchestration.branch_admission import _admit_resolved
    from nro.orchestration.registry import utcnow
    from nro.orchestration.source_snapshots import SourceSnapshot

    branches = BranchStore(registry.paths.control)
    if not branches.path.exists():
        return 0
    changed = 0
    with branches._lock():
        topology = branches.read().topology
        with registry.connection(write=True) as db:
            requests = [
                dict(row)
                for row in db.execute("""SELECT p.* FROM request_plans p
                JOIN requests r ON r.id=p.request_id WHERE r.state='active' ORDER BY r.created_at DESC""")
            ]
            for request in requests:
                payload = json.loads(request["payload_json"])
                owner = topology.records.get(payload["branch"])
                if owner is None or owner.retired or owner.registry_id != payload["registry_id"]:
                    ids = [
                        row[0]
                        for row in db.execute(
                            "SELECT instance_id FROM request_instances WHERE request_id=?",
                            (request["request_id"],),
                        )
                    ]
                    db.execute(
                        "UPDATE requests SET state='cancelled' WHERE id=?", (request["request_id"],)
                    )
                    dependency_state.invalidate(
                        db, ids, now=utcnow(), reason="Owning branch retired", include_roots=True
                    )
                    continue
                needs_replay = False
                for selected in payload.get("inherited", []):
                    row = db.execute(
                        "SELECT artifact_state,artifact_fingerprint FROM instances WHERE id=?",
                        (selected["id"],),
                    ).fetchone()
                    if (
                        row is None
                        or row["artifact_state"] != "fresh"
                        or row["artifact_fingerprint"] != selected["contract"]
                        or selected["branch"] not in topology.ancestors(payload["branch"])
                    ):
                        needs_replay = True
                        break
                if not needs_replay:
                    continue
                active_targets = {
                    row[0]
                    for row in db.execute(
                        """SELECT e.logical_key
                    FROM request_instances ri JOIN instance_execution e ON e.instance_id=ri.instance_id
                    WHERE ri.request_id=? AND ri.role='target' AND ri.demand_state='active' """,
                        (request["request_id"],),
                    )
                }
                if not active_targets:
                    continue
                by_key = {value["key"]: value for value in payload["specifications"]}
                required, pending = set(), list(active_targets)
                while pending:
                    key = pending.pop()
                    if key not in required:
                        required.add(key)
                        pending.extend(by_key[key]["dependencies"])
                payload = dict(
                    payload,
                    terminals=sorted(active_targets),
                    specifications=[by_key[key] for key in sorted(required)],
                )
                specifications = tuple(decode_spec(value) for value in payload["specifications"])
                contracts = scientific_contracts(specifications)
                superseded = any(
                    fingerprint(json.loads(row["scientific_contract_json"]))
                    != fingerprint(contracts[row["logical_key"]])
                    for row in db.execute(
                        "SELECT logical_key,scientific_contract_json FROM branch_instances WHERE registry_id=?",
                        (payload["registry_id"],),
                    )
                    if row["logical_key"] in contracts
                )
                if superseded:
                    db.execute(
                        "UPDATE requests SET state='superseded' WHERE id=?",
                        (request["request_id"],),
                    )
                    continue
                SourceSnapshot(
                    Path(payload["source"]["root"]), payload["source"]["digest"]
                ).verify()
                plan = resolve_payload(topology, payload, candidates_locked(db, payload["project"]))
                _admit_resolved(registry, db, plan, payload, request_id=request["request_id"])
                changed += 1
    return changed
