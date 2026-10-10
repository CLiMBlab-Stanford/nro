"""Adopt work items created before personal installations used the scheduler."""

from __future__ import annotations

import json

from nro.definitions.store import fingerprint
from nro.orchestration.branch_admission import _import_lineages
from nro.orchestration.branch_reconciliation import candidates_locked
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.planning_context import work_item_key

HOTFIX_ID = "v0319-personal-scheduler-adoption"
SUMMARY = "Adopt unbound work items created by the former personal-installation runtime."


def _legacy_lineage_ids(database) -> set[int]:
    """Return lineages that exactly satisfy the former unscoped identity rule."""
    lineages = {
        int(row["id"]): dict(row) for row in database.execute("SELECT * FROM module_lineages")
    }
    parents: dict[int, list[int]] = {}
    for row in database.execute(
        "SELECT module_lineage_id,upstream_module_lineage_id FROM module_lineage_dependencies"
    ):
        parents.setdefault(int(row["module_lineage_id"]), []).append(
            int(row["upstream_module_lineage_id"])
        )
    accepted: dict[int, bool] = {}
    active: set[int] = set()

    def valid(lineage_id: int) -> bool:
        if lineage_id in accepted:
            return accepted[lineage_id]
        if lineage_id in active or lineage_id not in lineages:
            return False
        active.add(lineage_id)
        upstream_ids = parents.get(lineage_id, [])
        upstream = None
        if len(upstream_ids) > 1:
            result = False
        elif upstream_ids:
            parent_id = upstream_ids[0]
            result = valid(parent_id)
            if result:
                upstream = lineages[parent_id]["lineage_fingerprint"]
        else:
            result = True
        row = lineages[lineage_id]
        if result:
            expected = fingerprint(
                {
                    "module": row["configuration_class"],
                    "config_id": row["config_id"],
                    "upstream": upstream,
                }
            )
            result = row["lineage_fingerprint"] == expected
        active.remove(lineage_id)
        accepted[lineage_id] = result
        return result

    return {lineage_id for lineage_id in lineages if valid(lineage_id)}


def _candidates(database, projects: tuple[str, ...]) -> tuple[dict, ...]:
    legacy = _legacy_lineage_ids(database)
    if not legacy:
        return ()
    placeholders = ",".join("?" for _project in projects)
    rows = database.execute(
        f"""SELECT item.*,lineage.lineage_fingerprint
            FROM work_items item
            JOIN module_lineages lineage ON lineage.id=item.module_lineage_id
            WHERE item.project IN ({placeholders})
              AND NOT EXISTS(
                  SELECT 1 FROM work_item_execution execution
                  WHERE execution.work_item_id=item.id
              )
              AND NOT EXISTS(
                  SELECT 1 FROM branch_work_items binding
                  WHERE binding.work_item_id=item.id
              )
            ORDER BY item.id""",
        projects,
    ).fetchall()
    result = []
    for source in rows:
        row = dict(source)
        if int(row["module_lineage_id"]) not in legacy:
            continue
        expected = work_item_key(
            str(row["project"]),
            str(row["module"]),
            str(row["lineage_fingerprint"]),
            str(row["participant"]),
            json.loads(row["entities_json"]),
        )
        if row["work_item_key"] == expected:
            result.append(row)
    return tuple(result)


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Move exact legacy personal records into the current scheduler namespace."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    topology = BranchStore(registry.paths.control).read().topology
    main = topology.records.get("main")
    if main is None or main.retired:
        raise ValueError("Hotfix requires an active main installation")
    with registry.connection(write=execute) as database:
        candidates = _candidates(database, selected)
        if execute and candidates:
            placeholders = ",".join("?" for _project in selected)
            active = database.execute(
                f"""SELECT COUNT(*) FROM attempts attempt
                    JOIN work_items item ON item.id=attempt.work_item_id
                    WHERE item.project IN ({placeholders})
                      AND attempt.state IN ('queued','running','cancel_requested')""",
                selected,
            ).fetchone()[0]
            if active:
                raise ValueError("Hotfix requires selected-project attempts to be stopped")
            contracts = {
                int(candidate.evidence["work_item_id"]): dict(candidate.contract)
                for project in selected
                for candidate in candidates_locked(database, project, fresh_only=False)
            }
            required = {int(row["module_lineage_id"]) for row in candidates}
            lineage_rows = [dict(row) for row in database.execute("SELECT * FROM module_lineages")]
            dependency_rows = [
                dict(row) for row in database.execute("SELECT * FROM module_lineage_dependencies")
            ]
            mapping = _import_lineages(
                database,
                lineages=lineage_rows,
                dependencies=dependency_rows,
                owner=main.registry_id,
                required_lineages=required,
            )
            for row in candidates:
                work_item_id = int(row["id"])
                contract = contracts.get(work_item_id)
                if contract is None:
                    raise ValueError(
                        f"Cannot reconstruct legacy scientific contract for {row['work_item_key']}"
                    )
                database.execute(
                    "UPDATE work_items SET module_lineage_id=? WHERE id=?",
                    (mapping[int(row["module_lineage_id"])], work_item_id),
                )
                serialized = json.dumps(contract, sort_keys=True, allow_nan=False)
                database.execute(
                    "INSERT INTO branch_work_items VALUES (?,?,?,?)",
                    (main.registry_id, row["work_item_key"], work_item_id, serialized),
                )
                database.execute(
                    "INSERT OR IGNORE INTO compiled_revisions VALUES (?,?,?,?)",
                    (main.registry_id, row["work_item_key"], 1, fingerprint(contract)),
                )
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=(),
        records=len(candidates),
        applied=execute,
    )
