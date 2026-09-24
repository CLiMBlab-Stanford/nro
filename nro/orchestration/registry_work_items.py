"""Register work-item graphs inside a caller-owned registry transaction."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from nro.configuration.store import fingerprint
from nro.orchestration import dependency_state
from nro.orchestration.artifact_resolution import (
    output_contracts_overlap,
    validate_output_ownership,
)

if TYPE_CHECKING:
    from nro.orchestration.contracts import WorkItemSpec


def work_item_relative_directory(work_item: Mapping[str, object]) -> Path:
    """Return the private log directory for one work-item record."""
    project = str(work_item["project"])
    participant = str(work_item["participant"]).removeprefix("sub-")
    raw_entities = work_item["entities_json"]
    entities = json.loads(raw_entities) if isinstance(raw_entities, str) else raw_entities
    if not isinstance(entities, Mapping):
        raise ValueError("Work-item entities must be a mapping")
    preferred = ("ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part", "chunk")
    ordered = [key for key in preferred if key in entities]
    ordered.extend(sorted(set(entities) - set(ordered)))
    name = "_".join([f"sub-{participant}", *(f"{key}-{entities[key]}" for key in ordered)])
    digest = str(work_item["work_item_key"]).rsplit(":", 1)[-1][:16]
    return Path(project) / str(work_item["module"]) / f"sub-{participant}" / name / digest


def upsert_work_item_graph(
    database: sqlite3.Connection,
    work_item_records: Sequence[tuple[WorkItemSpec, dict]],
    *,
    now: str,
    external_ids: Mapping[str, int] | None = None,
    owner_branch: str | None = None,
) -> dict[str, int]:
    """Merge work-item contracts and edges without opening a connection or creating demand."""
    validate_output_ownership(tuple(spec for spec, _record in work_item_records))
    claims_by_root: dict[str, list[dict]] = {}
    output_roots = sorted(
        {str(spec.output_root.expanduser().resolve()) for spec, _record in work_item_records}
    )
    if output_roots:
        placeholders = ",".join("?" for _root in output_roots)
        for row in database.execute(
            f"""SELECT work_item_key, output_root, output_prefix, expected_outputs_json
                FROM work_items WHERE output_root IN ({placeholders})""",
            output_roots,
        ):
            claim = dict(row)
            claims_by_root.setdefault(str(claim["output_root"]), []).append(claim)
    work_item_ids: dict[str, int] = dict(external_ids or {})
    replace_dependencies: dict[int, bool] = {}
    for spec, record in work_item_records:
        root = spec.output_root.expanduser().resolve()
        for claimed in claims_by_root.get(str(root), ()):
            if claimed["work_item_key"] == spec.key:
                continue
            if output_contracts_overlap(
                spec.output_root,
                spec.output_prefix,
                spec.expected_outputs,
                Path(str(claimed["output_root"])),
                claimed["output_prefix"],
                tuple(Path(value) for value in json.loads(claimed["expected_outputs_json"])),
            ):
                raise ValueError(
                    "Work-item output claim conflicts with registered work: "
                    f"{spec.key} and {claimed['work_item_key']}"
                )
        lineage = database.execute(
            "SELECT directory_label FROM module_lineages WHERE id=?",
            (spec.module_lineage_id,),
        ).fetchone()
        if lineage is None:
            raise ValueError(
                f"Work item {spec.key} references unknown module lineage {spec.module_lineage_id}"
            )
        if str(lineage["directory_label"]) != spec.directory_label:
            raise ValueError(
                f"Work item {spec.key} does not match its registered module lineage: "
                f"expected directory {lineage['directory_label']}, received {spec.directory_label}"
            )
        existing = database.execute(
            """SELECT id, module, module_lineage_id, scope, resource_class, revision_fingerprint,
                      artifact_contract_json, artifact_fingerprint,
                      command_json, runtime_config_path, input_paths_json,
                      output_root, output_prefix, expected_outputs_json
               FROM work_items WHERE work_item_key=?""",
            (spec.key,),
        ).fetchone()
        if existing:
            if (
                str(existing["module"]) != spec.module
                or int(existing["module_lineage_id"]) != spec.module_lineage_id
            ):
                raise ValueError(
                    f"Work-item key {spec.key} is already bound to a different module lineage"
                )
            work_item_id = int(existing["id"])
            try:
                recorded_contract = json.loads(existing["artifact_contract_json"])
                if owner_branch in {None, "main"}:
                    from nro.orchestration.catalog import canonical_contract

                    completion = database.execute(
                        """SELECT config_id,config_fingerprint,resolved_yaml
                           FROM completions WHERE work_item_id=?""",
                        (work_item_id,),
                    ).fetchone()
                    configuration = (
                        {
                            "id": completion["config_id"],
                            "fingerprint": completion["config_fingerprint"],
                            "resolved": yaml.safe_load(completion["resolved_yaml"]) or {},
                        }
                        if completion is not None
                        else None
                    )
                    recorded_contract = canonical_contract(recorded_contract, configuration)
                artifact_changed = fingerprint(recorded_contract) != record["artifact_fingerprint"]
            except (ValueError, TypeError, KeyError):
                artifact_changed = True
            replace_dependencies[work_item_id] = artifact_changed
            database.execute(
                """
                UPDATE work_items SET scope=?, resource_class=?,
                    revision_fingerprint=?, artifact_contract_json=?,
                    artifact_fingerprint=?, command_json=?, runtime_config_path=?,
                    memory_gb=MAX(memory_gb, ?), max_memory_gb=MAX(max_memory_gb, ?),
                    input_paths_json=?, output_root=?, output_prefix=?,
                    expected_outputs_json=?,
                    artifact_state=CASE WHEN ? THEN 'stale' ELSE artifact_state END,
                    artifact_reason=CASE WHEN ? THEN 'Work-item contract changed' ELSE artifact_reason END,
                    updated_at=?
                WHERE id=?
                """,
                (
                    record["scope"],
                    record["resource_class"],
                    record["revision_fingerprint"],
                    record["artifact_contract_json"],
                    record["artifact_fingerprint"],
                    record["command_json"],
                    record["runtime_config_path"],
                    record["memory_gb"],
                    record["max_memory_gb"],
                    record["input_paths_json"],
                    record["output_root"],
                    record["output_prefix"],
                    record["expected_outputs_json"],
                    artifact_changed,
                    artifact_changed,
                    now,
                    work_item_id,
                ),
            )
            if artifact_changed:
                dependency_state.invalidate(
                    database,
                    [work_item_id],
                    now=now,
                    reason="Resolved upstream work-item contract changed",
                )
                database.execute(
                    """UPDATE attempts SET state='cancel_requested',
                              error_type='WorkItemGraphChanged',
                              error_message='Work-item contract changed while work was active'
                       WHERE work_item_id=? AND state IN ('queued', 'running')""",
                    (work_item_id,),
                )
        else:
            cursor = database.execute(
                """
                INSERT INTO work_items(
                    work_item_key, module, module_lineage_id, project, participant,
                    entities_json, scope, artifact_state, artifact_reason,
                    resource_class, revision_fingerprint,
                    artifact_contract_json, artifact_fingerprint,
                    memory_gb, max_memory_gb, command_json, runtime_config_path, input_paths_json,
                    output_root, output_prefix, expected_outputs_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'missing', 'Not yet assessed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.key,
                    spec.module,
                    spec.module_lineage_id,
                    spec.project,
                    spec.participant,
                    record["entities_json"],
                    spec.scope,
                    spec.resource_class,
                    record["revision_fingerprint"],
                    record["artifact_contract_json"],
                    record["artifact_fingerprint"],
                    record["memory_gb"],
                    record["max_memory_gb"],
                    record["command_json"],
                    record["runtime_config_path"],
                    record["input_paths_json"],
                    record["output_root"],
                    record["output_prefix"],
                    record["expected_outputs_json"],
                    now,
                    now,
                ),
            )
            work_item_id = int(cursor.lastrowid)
            replace_dependencies[work_item_id] = True
        work_item_ids[spec.key] = work_item_id

    for spec, _record in work_item_records:
        work_item_id = work_item_ids[spec.key]
        if not replace_dependencies[work_item_id]:
            continue
        database.execute("DELETE FROM work_item_dependencies WHERE work_item_id=?", (work_item_id,))
        for dependency in spec.dependencies:
            database.execute(
                """
                INSERT INTO work_item_dependencies(work_item_id, upstream_work_item_id, role, required_generation)
                VALUES (?, ?, ?, NULL)
                """,
                (
                    work_item_id,
                    work_item_ids[dependency],
                    "inherited" if dependency in (external_ids or {}) else "input",
                ),
            )
    dependency_state.synchronize(database, now=now)
    return work_item_ids


def normalize_active_request_graph(database: sqlite3.Connection) -> None:
    """Reconcile active demand with the current dependency graph."""
    parents: dict[int, list[int]] = {}
    for row in database.execute(
        "SELECT work_item_id, upstream_work_item_id FROM work_item_dependencies WHERE role != 'inherited'"
    ):
        parents.setdefault(int(row["work_item_id"]), []).append(int(row["upstream_work_item_id"]))
    for active_request in database.execute(
        "SELECT id FROM requests WHERE state='active'"
    ).fetchall():
        request_id = str(active_request["id"])
        targets = {
            int(row["work_item_id"])
            for row in database.execute(
                """SELECT work_item_id FROM request_work_items
                   WHERE request_id=? AND role='target' AND demand_state='active'""",
                (request_id,),
            )
        }
        required = set(targets)
        pending = list(targets)
        while pending:
            work_item_id = pending.pop()
            for upstream_id in parents.get(work_item_id, ()):
                if upstream_id not in required:
                    required.add(upstream_id)
                    pending.append(upstream_id)
        existing = {
            int(row["work_item_id"]): str(row["demand_state"])
            for row in database.execute(
                "SELECT work_item_id, demand_state FROM request_work_items WHERE request_id=?",
                (request_id,),
            )
        }
        for required_id in required - set(existing):
            database.execute(
                """INSERT INTO request_work_items(request_id, work_item_id, role, demand_state)
                   VALUES (?, ?, 'dependency', 'active')""",
                (request_id, required_id),
            )
        orphaned = {
            work_item_id
            for work_item_id, demand_state in existing.items()
            if demand_state == "active" and work_item_id not in required
        }
        if orphaned:
            placeholders = ",".join("?" for _ in orphaned)
            database.execute(
                f"""UPDATE request_work_items SET demand_state='cancelled'
                    WHERE request_id=? AND work_item_id IN ({placeholders})
                      AND demand_state='active'""",
                (request_id, *tuple(orphaned)),
            )


def cancel_purged_demand(
    database: sqlite3.Connection,
    work_item_ids: Sequence[int],
    *,
    now: str,
) -> int:
    """Withdraw demand that requires explicitly purged work items.

    Purging an upstream item also makes every dependent target ineligible to
    continue under its existing request. Unrelated targets in the same request
    remain active, together with their dependency closures.
    """
    selected = {int(value) for value in work_item_ids}
    if not selected:
        return 0
    dependents: dict[int, set[int]] = {}
    for row in database.execute(
        "SELECT work_item_id,upstream_work_item_id FROM work_item_dependencies"
    ):
        dependents.setdefault(int(row["upstream_work_item_id"]), set()).add(
            int(row["work_item_id"])
        )
    affected = set(selected)
    pending = list(selected)
    while pending:
        for dependent in dependents.get(pending.pop(), ()):
            if dependent not in affected:
                affected.add(dependent)
                pending.append(dependent)

    placeholders = ",".join("?" for _ in affected)
    cursor = database.execute(
        f"""UPDATE request_work_items SET demand_state='cancelled'
            WHERE work_item_id IN ({placeholders}) AND demand_state='active'""",
        tuple(sorted(affected)),
    )
    cancelled = int(cursor.rowcount)
    normalize_active_request_graph(database)
    database.execute(
        """UPDATE requests SET state='cancelled',updated_at=?
           WHERE state='active' AND NOT EXISTS (
               SELECT 1 FROM request_work_items
               WHERE request_id=requests.id AND role='target' AND demand_state='active'
           )""",
        (now,),
    )
    database.execute(
        """UPDATE attempts SET state='cancel_requested',
                  error_type='UserCancelled',
                  error_message='Demand withdrawn because a required artifact was purged'
           WHERE state IN ('queued','running') AND NOT EXISTS (
               SELECT 1 FROM request_work_items links JOIN requests requests
                 ON requests.id=links.request_id
               WHERE links.work_item_id=attempts.work_item_id
                 AND links.demand_state='active' AND requests.state='active'
           )"""
    )
    return cancelled


def forget_purged_work_items(
    database: sqlite3.Connection,
    work_item_ids: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    """Delete unreferenced scheduler records for purged work items.

    A selected item remains when a registered item outside the purge still
    depends on it. Such a row is part of the surviving artifact's saved DAG,
    even though its own artifact is missing and has no demand.
    """
    selected = {int(value) for value in work_item_ids}
    if not selected:
        return 0, ()
    existing = {
        int(row["id"])
        for row in database.execute(
            f"SELECT id FROM work_items WHERE id IN ({','.join('?' for _ in selected)})",
            tuple(sorted(selected)),
        )
    }
    deletable = set(existing)
    while deletable:
        placeholders = ",".join("?" for _ in deletable)
        protected = {
            int(row["upstream_work_item_id"])
            for row in database.execute(
                f"""SELECT upstream_work_item_id FROM work_item_dependencies
                    WHERE upstream_work_item_id IN ({placeholders})
                      AND work_item_id NOT IN ({placeholders})""",
                (*tuple(sorted(deletable)), *tuple(sorted(deletable))),
            )
        }
        if not protected:
            break
        deletable.difference_update(protected)
    if not deletable:
        return 0, tuple(sorted(existing))

    placeholders = ",".join("?" for _ in deletable)
    values = tuple(sorted(deletable))
    attempt_ids = tuple(
        int(row["id"])
        for row in database.execute(
            f"SELECT id FROM attempts WHERE work_item_id IN ({placeholders})", values
        )
    )
    if attempt_ids:
        attempt_placeholders = ",".join("?" for _ in attempt_ids)
        database.execute(
            f"DELETE FROM attempt_dependencies WHERE attempt_id IN ({attempt_placeholders})",
            attempt_ids,
        )
        database.execute(
            f"DELETE FROM attempt_execution WHERE attempt_id IN ({attempt_placeholders})",
            attempt_ids,
        )
    database.execute(
        f"DELETE FROM attempt_dependencies WHERE upstream_work_item_id IN ({placeholders})",
        values,
    )
    database.execute(f"DELETE FROM artifacts WHERE work_item_id IN ({placeholders})", values)
    database.execute(f"DELETE FROM completions WHERE work_item_id IN ({placeholders})", values)
    database.execute(f"DELETE FROM attempts WHERE work_item_id IN ({placeholders})", values)
    database.execute(
        f"DELETE FROM request_work_items WHERE work_item_id IN ({placeholders})", values
    )
    database.execute(
        f"DELETE FROM request_artifacts WHERE work_item_id IN ({placeholders})", values
    )
    database.execute(
        f"DELETE FROM work_item_dependencies WHERE work_item_id IN ({placeholders})", values
    )
    database.execute(
        f"DELETE FROM artifact_mutations WHERE work_item_id IN ({placeholders})", values
    )
    mappings = database.execute(
        f"""SELECT registry_id,logical_key FROM branch_work_items
            WHERE work_item_id IN ({placeholders})""",
        values,
    ).fetchall()
    database.executemany(
        "DELETE FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
        [(row["registry_id"], row["logical_key"]) for row in mappings],
    )
    database.execute(
        f"DELETE FROM branch_work_items WHERE work_item_id IN ({placeholders})", values
    )
    database.execute(
        f"DELETE FROM work_item_execution WHERE work_item_id IN ({placeholders})", values
    )
    database.execute(f"DELETE FROM work_items WHERE id IN ({placeholders})", values)
    return len(deletable), tuple(sorted(existing - deletable))
