"""Admit authorized branch recipes to the single shared request/attempt index."""

from __future__ import annotations

import getpass
import json
import uuid
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.branch_planning import BranchPlan
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.execution_cache import cache_publication
from nro.orchestration.registry import utcnow


def _workflow(db, payload: dict, owner: str) -> tuple[int, dict[int, int]]:
    revision = payload["revision"]
    lineages, bindings, dependencies = (
        payload[key] for key in ("lineages", "bindings", "dependencies")
    )
    needed = {row["configuration_lineage_id"] for row in bindings}
    while True:
        expanded = needed | {
            row["upstream_configuration_lineage_id"]
            for row in dependencies
            if row["configuration_lineage_id"] in needed
        }
        if expanded == needed:
            break
        needed = expanded
    mapping = {}
    for row in lineages:
        if row["id"] not in needed:
            continue
        signature = fingerprint({"owner": owner, "lineage": row["lineage_fingerprint"]})
        db.execute(
            """INSERT OR IGNORE INTO configuration_lineages
            (derivative_class,config_id,config_fingerprint,lineage_fingerprint,resolved_yaml,directory_label,created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (
                row["derivative_class"],
                row["config_id"],
                row["config_fingerprint"],
                signature,
                row["resolved_yaml"],
                row["directory_label"],
                row["created_at"],
            ),
        )
        mapping[row["id"]] = db.execute(
            """SELECT id FROM configuration_lineages
            WHERE derivative_class=? AND lineage_fingerprint=?""",
            (row["derivative_class"], signature),
        ).fetchone()[0]
    for row in dependencies:
        if row["configuration_lineage_id"] in needed:
            db.execute(
                "INSERT OR IGNORE INTO configuration_lineage_dependencies VALUES (?,?,?)",
                (
                    mapping[row["configuration_lineage_id"]],
                    mapping[row["upstream_configuration_lineage_id"]],
                    row["role"],
                ),
            )
    name = owner + ":" + revision["workflow_id"]
    db.execute(
        """INSERT OR IGNORE INTO workflow_revisions
        (workflow_id,revision,definition_fingerprint,source_path,resolved_yaml,created_at) VALUES (?,?,?,?,?,?)""",
        (
            name,
            revision["revision"],
            revision["definition_fingerprint"],
            revision["source_path"],
            revision["resolved_yaml"],
            revision["created_at"],
        ),
    )
    revision_id = db.execute(
        "SELECT id FROM workflow_revisions WHERE workflow_id=? AND revision=?",
        (name, revision["revision"]),
    ).fetchone()[0]
    for row in bindings:
        db.execute(
            "INSERT OR IGNORE INTO workflow_bindings VALUES (?,?,?)",
            (revision_id, row["derivative_class"], mapping[row["configuration_lineage_id"]]),
        )
    return revision_id, mapping


@cache_publication
def admit_plan(
    registry,
    branches: BranchStore,
    checkout: Path,
    plan: BranchPlan,
    registered,
    *,
    source,
    site: Path,
    python: Path,
    concurrency: int,
    partition: str | None,
    selectors: dict,
) -> str | None:
    """Create local demand and inherited read edges under the global scheduler lock.

    Validate branch and scientific identity again after planning. Ancestors must
    still be fresh at exactly their selected generation. Their rows, recipes,
    and demand are never changed. The source and environment are supplied by the
    authorized caller and retained in execution recipes, not scientific identity.
    """
    if concurrency < 1 or not python.is_absolute():
        raise ValueError("Admission requires positive concurrency and an absolute interpreter")
    source.verify()
    from nro.orchestration.releases import ReleaseStore
    from nro.orchestration.source_snapshots import source_fingerprint

    release = ReleaseStore(branches).require_approved(checkout) if plan.branch == "main" else None
    with branches._lock():
        snapshot = branches.read()
        name = snapshot.topology.require_checkout(checkout)
        if name != plan.branch or branches.control != registry.paths.control.resolve():
            raise ValueError("Plan does not belong to this checkout and scheduler")
        owner = snapshot.topology.records[name].registry_id
        if source_fingerprint(checkout) != source.digest:
            raise ValueError("Source changed before admission; plan and capture it again")
        # The catalog lock is already held; constructing the bound handle avoids
        # reacquiring it through registry_for_checkout.
        from nro.orchestration.branch_registry import BranchRegistry

        scientific = BranchRegistry(branches.control, snapshot.topology.records[name])
        recorded = {item.key: item for item in scientific.instances()}
        for item in plan.instances:
            if (
                item.context.paths.branch != name
                or item.context.paths.bids != registry.paths.bids_root.resolve()
                or item.spec.project != registry.paths.project
            ):
                raise ValueError("Plan paths differ from the authorized project")
            if item.artifact and item.artifact.branch not in snapshot.topology.ancestors(name):
                raise ValueError("Selected producer is no longer an ancestor")
            if item.spec.key not in recorded or recorded[
                item.spec.key
            ].contract_fingerprint != fingerprint(item.contract):
                raise ValueError("Scientific plan changed before admission; resolve it again")
        from nro.orchestration.compiled_request import encode_spec, export_workflow

        payload = dict(
            protocol=1,
            branch=name,
            registry_id=owner,
            project=registry.paths.project,
            context=plan.instances[0].context.as_dict(),
            specifications=[
                encode_spec(spec)
                for spec in (plan.specifications or tuple(item.spec for item in plan.instances))
            ],
            revisions={
                spec.key: recorded[spec.key].revision
                for spec in (plan.specifications or tuple(item.spec for item in plan.instances))
            },
            terminals=list(plan.terminals),
            inherit=plan.inherit,
            workflow=export_workflow(scientific, registered),
            source=dict(root=str(source.root), digest=source.digest),
            site=str(site),
            python=str(python),
            selectors=selectors,
            concurrency=concurrency,
            partition=partition,
            release=release,
        )
        with registry.connection(write=True) as db:
            return _admit_resolved(registry, db, plan, payload)


def _admit_resolved(
    registry, db, plan: BranchPlan, payload: dict, request_id: str | None = None
) -> str:
    """Publish a resolved compiled request; the caller holds admission locks."""
    from nro.orchestration.source_snapshots import SourceSnapshot

    name, owner = payload["branch"], payload["registry_id"]
    if db.execute(
        "SELECT 1 FROM metadata WHERE key=?", ("branch_maintenance:" + owner,)
    ).fetchone():
        raise ValueError(
            "Branch scientific repair is in progress; resume repair before submitting work"
        )
    source = SourceSnapshot(Path(payload["source"]["root"]), payload["source"]["digest"])
    site, python = Path(payload["site"]), Path(payload["python"])
    selectors, concurrency, partition = (
        payload[key] for key in ("selectors", "concurrency", "partition")
    )
    release = payload["release"]
    external = {}
    keys = {}
    for item in plan.instances:
        artifact = item.artifact
        if artifact is None:
            keys[item.spec.key] = item.spec.key if name == "main" else owner + ":" + item.spec.key
            continue
        row = db.execute(
            """SELECT i.* FROM instances i LEFT JOIN instance_execution e ON e.instance_id=i.id
            WHERE COALESCE(e.branch,'main')=? AND COALESCE(e.logical_key,i.instance_key)=?""",
            (artifact.branch, artifact.key),
        ).fetchone()
        if (
            row is None
            or row["artifact_state"] != "fresh"
            or row["current_generation"] != artifact.generation
            or Path(row["output_root"]).resolve() != artifact.root.resolve()
        ):
            raise ValueError("Inherited artifact changed before admission; resolve it again")
        keys[item.spec.key] = row["instance_key"]
        external[row["instance_key"]] = int(row["id"])
    revision_id, lineages = _workflow(db, payload["workflow"], owner)
    specs = []
    for item in plan.work:
        spec = item.spec
        specs.append(
            spec.evolve(
                key=keys[spec.key],
                configuration_lineage_id=lineages[spec.configuration_lineage_id],
                dependencies=tuple(keys[key] for key in spec.dependencies),
                input_paths=tuple(item.context.input_path(path) for path in spec.input_paths),
                output_root=item.context.output_path(spec.output_root),
                expected_outputs=tuple(
                    item.context.output_path(path) for path in spec.expected_outputs
                ),
                command=source.command((str(python), *spec.command[1:]), site=site),
            )
        )
    now = utcnow()
    ids = registry._upsert_instance_graph_locked(
        db,
        tuple(
            (spec, spec.as_record(compiled_contract=spec.contract.as_dict(spec.identity)))
            for spec in specs
        ),
        now=now,
        external_ids=external,
        owner_branch=name,
    )
    for item in plan.work:
        instance_id = ids[keys[item.spec.key]]
        previous = db.execute(
            "SELECT * FROM instance_execution WHERE instance_id=?", (instance_id,)
        ).fetchone()
        if previous is not None:
            from nro.orchestration import dependency_state

            science_changed = fingerprint(
                json.loads(previous["scientific_contract_json"])
            ) != fingerprint(item.contract)
            if science_changed:
                dependency_state.invalidate(
                    db,
                    [instance_id],
                    now=now,
                    include_roots=True,
                    reason="Resolved scientific contract changed",
                )
                db.execute(
                    "UPDATE instances SET command_json=?, runtime_config_path=? WHERE id=?",
                    (
                        json.dumps(
                            source.command((str(python), *item.spec.command[1:]), site=site)
                        ),
                        str(item.spec.runtime_config),
                        instance_id,
                    ),
                )
        sources = []
        for key in dict.fromkeys(item.spec.dependencies):
            upstream = ids[keys[key]]
            contract = db.execute(
                "SELECT artifact_fingerprint FROM instances WHERE id=?", (upstream,)
            ).fetchone()[0]
            sources.append(dict(id=upstream, contract=contract, inherited=keys[key] in external))
        db.execute(
            """INSERT INTO instance_execution VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(instance_id) DO UPDATE SET context_json=excluded.context_json,
            binding_sources_json=excluded.binding_sources_json,
            provenance_json=excluded.provenance_json, scientific_contract_json=excluded.scientific_contract_json""",
            (
                instance_id,
                name,
                owner,
                item.spec.key,
                json.dumps(item.context.as_dict()),
                json.dumps(sources),
                json.dumps(dict(branch=name, source_digest=source.digest, release=release)),
                json.dumps(item.contract),
            ),
        )
    request_id = request_id or uuid.uuid4().hex
    db.execute(
        """INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET workflow_revision_id=excluded.workflow_revision_id,
        target_module=excluded.target_module,selectors_json=excluded.selectors_json,
        state=excluded.state,updated_at=excluded.updated_at""",
        (
            request_id,
            getpass.getuser(),
            payload["project"],
            revision_id,
            ",".join(plan.terminals),
            json.dumps(selectors),
            concurrency,
            partition,
            ("active" if plan.work else "satisfied")
            if payload.get("demand", True)
            else "registered",
            now,
            now,
        ),
    )
    db.execute("INSERT OR IGNORE INTO request_owners VALUES (?,?,?)", (request_id, name, owner))
    payload = dict(
        payload,
        inherited=[
            dict(
                id=ids[keys[item.spec.key]],
                branch=item.artifact.branch,
                contract=db.execute(
                    "SELECT artifact_fingerprint FROM instances WHERE id=?",
                    (ids[keys[item.spec.key]],),
                ).fetchone()[0],
            )
            for item in plan.instances
            if item.artifact is not None
        ],
    )
    db.execute("DELETE FROM request_instances WHERE request_id=?", (request_id,))
    db.execute("DELETE FROM request_artifacts WHERE request_id=?", (request_id,))
    db.executemany(
        "INSERT OR IGNORE INTO request_artifacts VALUES (?,?)",
        [(request_id, ids[keys[item.spec.key]]) for item in plan.instances],
    )
    for item in plan.instances:
        db.execute(
            """INSERT INTO branch_instances VALUES (?,?,?,?)
            ON CONFLICT(registry_id,logical_key) DO UPDATE SET instance_id=excluded.instance_id,
            scientific_contract_json=excluded.scientific_contract_json""",
            (owner, item.spec.key, ids[keys[item.spec.key]], json.dumps(item.contract)),
        )
    db.execute(
        """INSERT INTO request_plans VALUES (?,?) ON CONFLICT(request_id)
        DO UPDATE SET payload_json=excluded.payload_json""",
        (request_id, json.dumps(payload)),
    )
    for item in plan.work:
        db.execute(
            "INSERT INTO request_instances VALUES (?,?,?,?)",
            (
                request_id,
                ids[keys[item.spec.key]],
                "target" if item.spec.key in plan.terminals else "dependency",
                "active" if payload.get("demand", True) else "cancelled",
            ),
        )
    registry._normalize_active_request_graph_locked(db)
    return request_id


def prepare_attempt(
    registry, db, instance: dict, metadata: dict, log_dir: Path
) -> tuple[str, ...] | None:
    """Pin actual input generations and route one claimed recipe through context transport.

    Return None when inherited science has changed and local fallback planning
    is required. No attempt should be created in that case.
    """
    from dataclasses import replace

    from nro.engine.io import atomic_write_text
    from nro.orchestration.attempt_entry import encode_payload
    from nro.orchestration.execution_context import ExecutionContext

    catalog = BranchStore(registry.paths.control).read().topology
    owner = catalog.records.get(metadata["branch"])
    if owner is None or owner.retired or owner.registry_id != metadata["registry_id"]:
        return None
    context = ExecutionContext.from_dict(json.loads(metadata["context_json"]))
    sources = json.loads(metadata["binding_sources_json"])
    if len(context.inputs) != len(sources):
        raise ValueError("Execution bindings do not match their registered producers")
    bindings = []
    for binding, source in zip(context.inputs, sources):
        row = db.execute(
            "SELECT current_generation,artifact_fingerprint,artifact_state FROM instances WHERE id=?",
            (source["id"],),
        ).fetchone()
        if (
            row is None
            or row["artifact_state"] != "fresh"
            or row["current_generation"] < 0
            or (source["inherited"] and row["artifact_fingerprint"] != source["contract"])
        ):
            db.execute(
                "UPDATE instances SET artifact_state='stale', artifact_reason=? WHERE id=?",
                ("Inherited input requires local replanning", instance["id"]),
            )
            return None
        bindings.append(replace(binding, generation=row["current_generation"]))
    context = replace(context, inputs=tuple(bindings))
    command = tuple(json.loads(instance["command_json"]))
    # SourceSnapshot.command fixes these positions and verifies the source/site
    # digests before this entry point imports any scientific module.
    if len(command) < 6 or Path(command[1]).name != "source_launcher.py":
        raise ValueError("Branch execution requires a pinned source launcher")
    data, digest = encode_payload(
        module=command[5],
        argv=list(command[6:]),
        runtime_config=Path(instance["runtime_config_path"]),
        configuration=instance["config_fingerprint"],
        context=context,
    )
    path = log_dir / (digest + ".attempt.json")
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Pinned attempt payload changed")
    if not path.exists():
        atomic_write_text(path, data.decode(), mode=0o444, durable=True)
    return (
        *command[:5],
        "nro.orchestration.attempt_entry",
        "--payload",
        str(path),
        "--digest",
        digest,
    )
