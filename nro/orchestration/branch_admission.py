"""Admit authorized branch recipes to the single shared request/attempt index."""

from __future__ import annotations

import getpass
import json
import uuid
from pathlib import Path

from nro.definitions.store import fingerprint
from nro.orchestration.artifact_resolution import scientific_contract_fingerprint
from nro.orchestration.branch_planning import BranchPlan
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.execution_cache import cache_publication
from nro.orchestration.registry import utcnow
from nro.orchestration.request_plans import decode_plan, encode_plan


def _replace_identities(value, mapping: dict[str, str]):
    """Replace exact work-item identities inside retained request data."""
    if isinstance(value, dict):
        return {key: _replace_identities(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_identities(item, mapping) for item in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def _repair_renamed_work_item_identities(db, *, items, registry_id: str) -> int:
    """Repair central keys produced from namespaced rename lineages.

    The stable branch, lineage, participant, entities, and output namespace
    must identify exactly one existing row. The batch repair reads retained
    request plans only once even when a large project graph needs correction.
    """
    existing_keys = {str(row[0]) for row in db.execute("SELECT work_item_key FROM work_items")}
    candidates = {}
    for row in db.execute(
        """SELECT item.id,item.work_item_key,item.module,item.module_lineage_id,
                  item.project,item.participant,item.entities_json,item.output_root,
                  item.output_prefix,execution.logical_key
           FROM work_items item JOIN work_item_execution execution
             ON execution.work_item_id=item.id
           WHERE execution.registry_id=?""",
        (registry_id,),
    ):
        identity = (
            str(row["module"]),
            int(row["module_lineage_id"]),
            str(row["project"]),
            str(row["participant"]),
            json.dumps(json.loads(row["entities_json"]), separators=(",", ":"), sort_keys=True),
            str(row["output_root"]),
            row["output_prefix"],
        )
        candidates.setdefault(identity, []).append(row)
    repairs = []
    replacements = {}
    for spec, logical_key in items:
        if spec.key in existing_keys:
            continue
        identity = (
            spec.module,
            spec.module_lineage_id,
            spec.project,
            spec.participant,
            json.dumps(dict(spec.entities), separators=(",", ":"), sort_keys=True),
            str(spec.output_root),
            spec.output_prefix,
        )
        rows = candidates.get(identity, ())
        if not rows:
            continue
        if len(rows) != 1:
            raise ValueError(
                f"Multiple registered work items match the stable identity of {logical_key}"
            )
        row = rows[0]
        old_stored = str(row["work_item_key"])
        old_logical = str(row["logical_key"])
        repairs.append((int(row["id"]), old_stored, spec.key, old_logical, logical_key))
        replacements[old_stored] = spec.key
        replacements[old_logical] = logical_key
    for work_item_id, old_stored, stored, old_logical, logical in repairs:
        for table in ("compiled_revisions", "branch_work_items"):
            conflict = db.execute(
                f"SELECT * FROM {table} WHERE registry_id=? AND logical_key=?",
                (registry_id, logical),
            ).fetchone()
            if conflict is not None:
                if table == "compiled_revisions":
                    # Request publication records the canonical revision before
                    # admission repairs an identity retained from a project
                    # rename.  Keep that newly compiled record and retire the
                    # superseded key instead of treating the expected overlap
                    # as conflicting work.
                    db.execute(
                        "DELETE FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
                        (registry_id, old_logical),
                    )
                    continue
                if int(conflict["work_item_id"]) == work_item_id:
                    db.execute(
                        "DELETE FROM branch_work_items WHERE registry_id=? AND logical_key=?",
                        (registry_id, old_logical),
                    )
                    continue
                raise ValueError(f"Cannot repair duplicate {table} identity for {logical}")
            db.execute(
                f"UPDATE {table} SET logical_key=? WHERE registry_id=? AND logical_key=?",
                (logical, registry_id, old_logical),
            )
        db.execute(
            "UPDATE work_item_execution SET logical_key=? WHERE work_item_id=?",
            (logical, work_item_id),
        )
        db.execute("UPDATE work_items SET work_item_key=? WHERE id=?", (stored, work_item_id))
    if not repairs:
        return 0
    for request in db.execute("SELECT request_id,payload_json FROM request_plans").fetchall():
        payload = decode_plan(request["payload_json"])
        translated = _replace_identities(payload, replacements)
        if translated != payload:
            db.execute(
                "UPDATE request_plans SET payload_json=? WHERE request_id=?",
                (encode_plan(translated), request["request_id"]),
            )
    return len(repairs)


def _workflow(
    db, payload: dict, owner: str, *, required_lineages: set[int] | None = None
) -> tuple[int, dict[int, int]]:
    """Import a workflow and any auxiliary lineages used by its work-item graph."""
    revision = payload["revision"]
    lineages, bindings, dependencies = (
        payload[key] for key in ("lineages", "bindings", "dependencies")
    )
    needed = {
        *(row["module_lineage_id"] for row in bindings),
        *(required_lineages or ()),
    }
    while True:
        expanded = needed | {
            row["upstream_module_lineage_id"]
            for row in dependencies
            if row["module_lineage_id"] in needed
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
            """INSERT INTO module_lineages
            (configuration_class,config_id,config_fingerprint,lineage_fingerprint,resolved_yaml,directory_label,created_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(configuration_class,lineage_fingerprint) DO UPDATE SET
            config_fingerprint=excluded.config_fingerprint,
            resolved_yaml=excluded.resolved_yaml""",
            (
                row["configuration_class"],
                row["config_id"],
                row["config_fingerprint"],
                signature,
                row["resolved_yaml"],
                row["directory_label"],
                row["created_at"],
            ),
        )
        mapping[row["id"]] = db.execute(
            """SELECT id FROM module_lineages
            WHERE configuration_class=? AND lineage_fingerprint=?""",
            (row["configuration_class"], signature),
        ).fetchone()[0]
    for row in dependencies:
        if row["module_lineage_id"] in needed:
            db.execute(
                "INSERT OR IGNORE INTO module_lineage_dependencies VALUES (?,?,?)",
                (
                    mapping[row["module_lineage_id"]],
                    mapping[row["upstream_module_lineage_id"]],
                    row["role"],
                ),
            )
    name = owner + ":" + revision["workflow_id"]
    current = db.execute(
        """SELECT id FROM workflow_revisions
           WHERE workflow_id=? AND definition_fingerprint=?""",
        (name, revision["definition_fingerprint"]),
    ).fetchone()
    if current is None:
        central_revision = int(
            db.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM workflow_revisions WHERE workflow_id=?",
                (name,),
            ).fetchone()[0]
        )
        revision_id = int(
            db.execute(
                """INSERT INTO workflow_revisions
                (workflow_id,revision,definition_fingerprint,source_path,resolved_yaml,created_at)
                VALUES (?,?,?,?,?,?)""",
                (
                    name,
                    central_revision,
                    revision["definition_fingerprint"],
                    revision["source_path"],
                    revision["resolved_yaml"],
                    revision["created_at"],
                ),
            ).lastrowid
        )
    else:
        revision_id = int(current["id"])
    for row in bindings:
        db.execute(
            "INSERT OR IGNORE INTO workflow_bindings VALUES (?,?,?)",
            (revision_id, row["configuration_class"], mapping[row["module_lineage_id"]]),
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
        recorded = {item.key: item for item in scientific.work_items()}
        for item in plan.work_items:
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
        from nro.site.configuration import protected_site_fingerprint

        payload = dict(
            protocol=1,
            branch=name,
            registry_id=owner,
            project=registry.paths.project,
            context=plan.work_items[0].context.as_dict(),
            specifications=[
                encode_spec(spec)
                for spec in (plan.specifications or tuple(item.spec for item in plan.work_items))
            ],
            revisions={
                spec.key: recorded[spec.key].revision
                for spec in (plan.specifications or tuple(item.spec for item in plan.work_items))
            },
            contracts={item.spec.key: item.contract for item in plan.work_items},
            terminals=list(plan.terminals),
            inherit=plan.inherit,
            workflow=export_workflow(scientific, registered),
            source=dict(root=str(source.root), digest=source.digest),
            site=str(site),
            site_fingerprint=protected_site_fingerprint(),
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
    for item in plan.work_items:
        artifact = item.artifact
        if artifact is None:
            keys[item.spec.key] = item.spec.key if name == "main" else owner + ":" + item.spec.key
            continue
        row = db.execute(
            """SELECT i.* FROM work_items i LEFT JOIN work_item_execution e ON e.work_item_id=i.id
            WHERE COALESCE(e.branch,'main')=? AND COALESCE(e.logical_key,i.work_item_key)=?""",
            (artifact.branch, artifact.key),
        ).fetchone()
        if (
            row is None
            or row["artifact_state"] != "fresh"
            or row["current_generation"] != artifact.generation
            or Path(row["output_root"]).resolve() != artifact.root.resolve()
        ):
            raise ValueError("Inherited artifact changed before admission; resolve it again")
        keys[item.spec.key] = row["work_item_key"]
        external[row["work_item_key"]] = int(row["id"])
    revision_id, lineages = _workflow(
        db,
        payload["workflow"],
        owner,
        required_lineages={item.spec.module_lineage_id for item in plan.work},
    )
    specs = []
    for item in plan.work:
        spec = item.spec
        specs.append(
            spec.evolve(
                key=keys[spec.key],
                module_lineage_id=lineages[spec.module_lineage_id],
                dependencies=tuple(keys[key] for key in spec.dependencies),
                input_paths=tuple(item.context.input_path(path) for path in spec.input_paths),
                output_root=item.context.output_path(spec.output_root),
                expected_outputs=tuple(
                    item.context.output_path(path) for path in spec.expected_outputs
                ),
                command=source.command((str(python), *spec.command[1:]), site=site),
            )
        )
    _repair_renamed_work_item_identities(
        db,
        items=tuple((spec, item.spec.key) for item, spec in zip(plan.work, specs, strict=True)),
        registry_id=owner,
    )
    now = utcnow()
    ids = registry._upsert_work_item_graph_locked(
        db,
        tuple(
            (
                spec,
                spec.as_record(
                    # Main executes the scheduler's own approved scientific
                    # catalog, so publish its canonical contract before demand
                    # becomes claimable. Development contracts remain opaque
                    # to the central scheduler and are assessed by their owner.
                    compiled_contract=(
                        spec.work_item_contract
                        if name == "main"
                        else spec.contract.as_dict(spec.identity)
                    )
                ),
            )
            for spec in specs
        ),
        now=now,
        external_ids=external,
        owner_branch=name,
    )
    for item in plan.work:
        work_item_id = ids[keys[item.spec.key]]
        previous = db.execute(
            "SELECT * FROM work_item_execution WHERE work_item_id=?", (work_item_id,)
        ).fetchone()
        if previous is not None:
            from nro.orchestration import dependency_state

            science_changed = scientific_contract_fingerprint(
                json.loads(previous["scientific_contract_json"])
            ) != scientific_contract_fingerprint(item.contract)
            if science_changed:
                dependency_state.invalidate(
                    db,
                    [work_item_id],
                    now=now,
                    include_roots=True,
                    reason="Resolved scientific contract changed",
                )
            # Execution recipes are operational pins, not scientific inputs.
            # Every admission adopts the requesting checkout's captured source
            # even when its normalized scientific contract is unchanged.
            db.execute(
                "UPDATE work_items SET command_json=?, runtime_config_path=? WHERE id=?",
                (
                    json.dumps(source.command((str(python), *item.spec.command[1:]), site=site)),
                    str(item.spec.runtime_config),
                    work_item_id,
                ),
            )
        sources = []
        for key in dict.fromkeys(item.spec.dependencies):
            upstream = ids[keys[key]]
            contract = db.execute(
                "SELECT artifact_fingerprint FROM work_items WHERE id=?", (upstream,)
            ).fetchone()[0]
            sources.append(dict(id=upstream, contract=contract, inherited=keys[key] in external))
        db.execute(
            """INSERT INTO work_item_execution VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(work_item_id) DO UPDATE SET context_json=excluded.context_json,
            binding_sources_json=excluded.binding_sources_json,
            provenance_json=excluded.provenance_json, scientific_contract_json=excluded.scientific_contract_json""",
            (
                work_item_id,
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
    request_state = (
        ("active" if plan.work else "satisfied") if payload.get("demand", True) else "registered"
    )
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
            request_state,
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
                    "SELECT artifact_fingerprint FROM work_items WHERE id=?",
                    (ids[keys[item.spec.key]],),
                ).fetchone()[0],
            )
            for item in plan.work_items
            if item.artifact is not None
        ],
    )
    db.execute("DELETE FROM request_work_items WHERE request_id=?", (request_id,))
    db.execute("DELETE FROM request_artifacts WHERE request_id=?", (request_id,))
    db.executemany(
        "INSERT OR IGNORE INTO request_artifacts VALUES (?,?)",
        [(request_id, ids[keys[item.spec.key]]) for item in plan.work_items],
    )
    for item in plan.work_items:
        db.execute(
            """INSERT INTO branch_work_items VALUES (?,?,?,?)
            ON CONFLICT(registry_id,logical_key) DO UPDATE SET work_item_id=excluded.work_item_id,
            scientific_contract_json=excluded.scientific_contract_json
            WHERE branch_work_items.work_item_id!=excluded.work_item_id
               OR branch_work_items.scientific_contract_json!=excluded.scientific_contract_json""",
            (owner, item.spec.key, ids[keys[item.spec.key]], json.dumps(item.contract)),
        )
    from nro.orchestration.request_plans import encode_plan, terminal_plan

    encoded_plan = encode_plan(payload) if request_state == "active" else terminal_plan(payload)
    db.execute(
        """INSERT INTO request_plans VALUES (?,?) ON CONFLICT(request_id)
        DO UPDATE SET payload_json=excluded.payload_json""",
        (request_id, encoded_plan),
    )
    for item in plan.work:
        db.execute(
            "INSERT INTO request_work_items VALUES (?,?,?,?)",
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
    registry, db, work_item: dict, metadata: dict, log_dir: Path
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
            "SELECT current_generation,artifact_fingerprint,artifact_state FROM work_items WHERE id=?",
            (source["id"],),
        ).fetchone()
        if (
            row is None
            or row["artifact_state"] != "fresh"
            or row["current_generation"] < 0
            or (source["inherited"] and row["artifact_fingerprint"] != source["contract"])
        ):
            db.execute(
                "UPDATE work_items SET artifact_state='stale', artifact_reason=? WHERE id=?",
                ("Inherited input requires local replanning", work_item["id"]),
            )
            return None
        bindings.append(replace(binding, generation=row["current_generation"]))
    context = replace(context, inputs=tuple(bindings))
    command = tuple(json.loads(work_item["command_json"]))
    # SourceSnapshot.command fixes these positions and verifies the source/site
    # digests before this entry point imports any scientific module.
    if len(command) < 6 or Path(command[1]).name != "source_launcher.py":
        raise ValueError("Branch execution requires a pinned source launcher")
    data, digest = encode_payload(
        module=command[5],
        argv=list(command[6:]),
        runtime_config=Path(work_item["runtime_config_path"]),
        configuration=work_item["config_fingerprint"],
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
