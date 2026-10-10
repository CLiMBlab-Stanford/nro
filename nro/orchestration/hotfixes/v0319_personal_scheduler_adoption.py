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


def _seeded_revision_candidates(
    database,
    projects: tuple[str, ...],
    owner: str,
) -> tuple[dict, ...]:
    """Return revision rows written by the original form of this hotfix."""
    legacy_ids = _legacy_lineage_ids(database)
    if not legacy_ids:
        return ()
    lineages = {
        int(row["id"]): dict(row) for row in database.execute("SELECT * FROM module_lineages")
    }
    legacy_by_scoped = {
        fingerprint(
            {"owner": owner, "lineage": lineages[lineage_id]["lineage_fingerprint"]}
        ): lineages[lineage_id]
        for lineage_id in legacy_ids
    }
    placeholders = ",".join("?" for _project in projects)
    rows = database.execute(
        f"""SELECT item.*,lineage.lineage_fingerprint,lineage.configuration_class,
                   lineage.config_id,lineage.directory_label,binding.scientific_contract_json,
                   revision.revision,revision.fingerprint AS seeded_fingerprint
            FROM work_items item
            JOIN module_lineages lineage ON lineage.id=item.module_lineage_id
            JOIN branch_work_items binding
              ON binding.work_item_id=item.id AND binding.registry_id=?
            JOIN compiled_revisions revision
              ON revision.registry_id=binding.registry_id
             AND revision.logical_key=binding.logical_key
            WHERE item.project IN ({placeholders})
              AND binding.logical_key=item.work_item_key
              AND revision.revision=1
              AND NOT EXISTS(
                  SELECT 1 FROM work_item_execution execution
                  WHERE execution.work_item_id=item.id
              )
            ORDER BY item.id""",
        (owner, *projects),
    ).fetchall()
    result = []
    for source in rows:
        row = dict(source)
        legacy = legacy_by_scoped.get(str(row["lineage_fingerprint"]))
        if legacy is None:
            continue
        if any(
            row[name] != legacy[name]
            for name in ("configuration_class", "config_id", "directory_label")
        ):
            continue
        expected_key = work_item_key(
            str(row["project"]),
            str(row["module"]),
            str(legacy["lineage_fingerprint"]),
            str(row["participant"]),
            json.loads(row["entities_json"]),
        )
        if row["work_item_key"] != expected_key:
            continue
        contract = json.loads(row["scientific_contract_json"])
        if row["seeded_fingerprint"] == fingerprint(contract):
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
        seeded_revisions = _seeded_revision_candidates(database, selected, main.registry_id)
        if execute and (candidates or seeded_revisions):
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
            database.executemany(
                "DELETE FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
                ((main.registry_id, row["work_item_key"]) for row in seeded_revisions),
            )
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=(),
        records=len(candidates) + len(seeded_revisions),
        applied=execute,
    )
