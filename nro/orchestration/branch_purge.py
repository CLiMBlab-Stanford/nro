"""Validate branch-selected deletion paths against central ownership and readers."""

import bisect
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Callable

from nro.configuration.store import fingerprint
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.purge_paths import (
    _entry_location,
    _is_removal_within,
    _purge_attempt_logs,
    _purge_inactive_worker_logs,
    _remove_path,
)
from nro.orchestration.registry import Registry, utcnow


def _receipt_rows(registry, work_item_ids: set[int]) -> dict[int, dict]:
    """Return the location metadata needed to remove ownership receipts."""
    if not work_item_ids:
        return {}
    placeholders = ",".join("?" for _ in work_item_ids)
    with registry.connection() as db:
        rows = db.execute(
            f"""SELECT i.id,i.project,i.module,i.work_item_key,
                       lineage.configuration_class,lineage.directory_label,
                       execution.logical_key,execution.context_json
                FROM work_items i
                JOIN module_lineages lineage ON lineage.id=i.module_lineage_id
                LEFT JOIN work_item_execution execution ON execution.work_item_id=i.id
                WHERE i.id IN ({placeholders})""",
            tuple(sorted(work_item_ids)),
        ).fetchall()
    return {int(row["id"]): dict(row) for row in rows}


def _receipt_location(registry, row: dict) -> tuple[Path, tuple[Path, str, str]]:
    """Resolve one receipt and the lineage root that owns it."""
    from nro.orchestration.ownership import work_item_record_path

    project_root = registry.paths.bids_root / str(row["project"])
    if row.get("context_json"):
        context = ExecutionContext.from_dict(json.loads(row["context_json"]))
        project_root = context.paths.output_project(str(row["project"]))
    configuration_class = str(row["configuration_class"])
    directory_label = str(row["directory_label"])
    path = work_item_record_path(
        project_root,
        configuration_class,
        directory_label,
        str(row["module"]),
        str(row.get("logical_key") or row["work_item_key"]),
    )
    return path, (project_root, configuration_class, directory_label)


def _protected_output_index(paths) -> tuple[frozenset[str], tuple[str, ...]]:
    """Resolve protected outputs once and index them for ancestor queries."""
    exact = frozenset(str(_entry_location(Path(path))) for path in paths)
    return exact, tuple(sorted(exact))


def _contains_protected_output(
    path: Path,
    index: tuple[frozenset[str], tuple[str, ...]],
) -> bool:
    """Return whether deleting path would remove an indexed protected output."""
    exact, ordered = index
    resolved = str(_entry_location(path))
    if resolved in exact:
        return True
    prefix = resolved if resolved == os.sep else resolved + os.sep
    position = bisect.bisect_left(ordered, prefix)
    return position < len(ordered) and ordered[position].startswith(prefix)


def token(row: dict, context_json: str | None) -> str:
    """Bind confirmation to the selected generation, storage paths, and contract."""
    return fingerprint(
        {
            key: row[key]
            for key in (
                "artifact_fingerprint",
                "current_generation",
                "output_root",
                "output_prefix",
            )
        }
        | {"context": context_json}
    )


def snapshot(registry, *, checkout: Path, site_values: dict) -> dict:
    """Return only owned work items; inherited selections are not deletion targets."""
    from nro.orchestration.scheduler_service import status

    topology = BranchStore(registry.paths.control).read().topology
    name = topology.require_checkout(checkout)
    owner = topology.records[name].registry_id
    report = status(registry, checkout=checkout, mode="cached")
    paths = BranchPaths(name, *(Path(site_values[key]) for key in ("bids", "work", "development")))
    if not registry.existing_database_path().is_file():
        return {"rows": [], "branch": name}
    with registry.connection() as db:
        metadata = {
            row["work_item_id"]: dict(row)
            for row in db.execute("SELECT * FROM work_item_execution")
        }
    rows = []
    for row in report["rows"]:
        item = metadata.get(row["id"])
        if (item is None and name != "main") or (item is not None and item["registry_id"] != owner):
            continue
        encoded = item["context_json"] if item else None
        context = (
            ExecutionContext.from_dict(json.loads(encoded))
            if encoded
            else ExecutionContext(paths, row["project"], row["work_item_key"], ())
        )
        rows.append(dict(row, execution_context=context.as_dict(), purge_token=token(row, encoded)))
    return {"rows": rows, "branch": name}


def purge(
    registry,
    *,
    checkout: Path,
    site_values: dict,
    plan: list[dict],
    logs_only: bool,
    dry_run: bool,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Delete confirmed owned paths after fencing all affected readers.

    The branch computes paths using its own module definitions. Central code
    checks ownership, generation, overlap, and shutdown before removing them.
    It never interprets a branch's module catalog.
    """
    view = snapshot(registry, checkout=checkout, site_values=site_values)
    owned = {row["id"]: row for row in view["rows"]}
    ids = {item["id"] for item in plan}
    if len(ids) != len(plan) or not ids <= owned.keys():
        raise ValueError("Purge includes duplicate, inherited, or foreign work items")
    for item in plan:
        if item["token"] != owned[item["id"]]["purge_token"]:
            raise ValueError("Purge targets changed; generate and confirm a new report")
        if logs_only and (item["public"] or item["private"]):
            raise ValueError("Log-only purge cannot delete derivative paths")

    deletable_ids, _retained_ids = registry.purge_record_partition(ids)
    receipt_rows = _receipt_rows(registry, ids | set(deletable_ids))
    receipt_locations = {
        work_item_id: _receipt_location(registry, row)
        for work_item_id, row in receipt_rows.items()
    }
    receipt_paths = {path for path, _root in receipt_locations.values()}

    def report(phase: str, completed: int, total: int) -> None:
        if progress is not None:
            progress(phase, completed, total)

    def validate(db, *, phase: str):
        report(phase, 0, len(plan))
        rows = {
            row["id"]: dict(row)
            for row in db.execute("SELECT * FROM work_items")
            if row["id"] in ids
        }
        executions = {
            row["work_item_id"]: row["context_json"]
            for row in db.execute("SELECT work_item_id,context_json FROM work_item_execution")
            if row["work_item_id"] in ids
        }
        if rows.keys() != ids:
            raise ValueError("Purge targets changed; generate and confirm a new report")
        protected = (
            _protected_output_index(
                path
                for row in db.execute("SELECT id,expected_outputs_json FROM work_items")
                if row["id"] not in ids
                for path in json.loads(row["expected_outputs_json"])
            )
            if not logs_only
            else (frozenset(), ())
        )
        for position, item in enumerate(plan, start=1):
            row = rows[item["id"]]
            if token(row, executions.get(item["id"])) != item["token"]:
                raise ValueError("Purge targets changed; generate and confirm a new report")
            context = ExecutionContext.from_dict(owned[item["id"]]["execution_context"])
            for raw in (*item["public"], *item["private"]):
                path = Path(raw)
                if not path.is_absolute() or ".." in path.parts:
                    raise ValueError("Purge paths must be normalized and absolute")
                context.require_removal(path)
                if _contains_protected_output(path, protected):
                    raise ValueError(
                        "Purge would remove another registered work item; narrow the paths or expand the selection"
                    )
            if position % 100 == 0 or position == len(plan):
                report(phase, position, len(plan))

    with registry.connection() as db:
        validate(db, phase="Validating purge selection")
    counts = {"work_items": len(ids), "derivative_paths": 0, "work_paths": 0}
    if not logs_only:
        from contextlib import nullcontext

        with nullcontext() if dry_run else registry.artifact_mutation(ids):
            with registry.connection() as db:
                validate(db, phase="Revalidating purge selection")
            groups = []
            for kind, counter in (("public", "derivative_paths"), ("private", "work_paths")):
                targets: dict[Path, Path] = {}
                for item in plan:
                    context = ExecutionContext.from_dict(owned[item["id"]]["execution_context"])
                    project = owned[item["id"]]["project"]
                    roots = (
                        (
                            context.paths.output_project(project) / "derivatives",
                            registry.paths.control,
                        )
                        if kind == "public"
                        else (context.paths.private_project(project) / "derivatives",)
                    )
                    for raw in item[kind]:
                        path = Path(raw)
                        if path in receipt_paths:
                            continue
                        targets[path] = next(
                            root for root in roots if _is_removal_within(path, root)
                        )
                groups.append((counter, targets))
            planned_receipts = {
                receipt_locations[work_item_id][0]
                for work_item_id in deletable_ids
                if work_item_id in receipt_locations
                and (
                    receipt_locations[work_item_id][0].exists()
                    or receipt_locations[work_item_id][0].is_symlink()
                )
            }
            total_paths = sum(len(targets) for _counter, targets in groups) + len(
                planned_receipts
            )
            completed_paths = 0
            report("Removing artifact paths", 0, total_paths)
            for counter, targets in groups:
                for path in sorted(targets, key=lambda value: len(value.parts)):
                    counts[counter] += int(
                        _remove_path(
                            path,
                            dry_run=dry_run,
                            prune_root=targets[path],
                        )
                    )
                    completed_paths += 1
                    if completed_paths % 25 == 0 or completed_paths == total_paths:
                        report("Removing artifact paths", completed_paths, total_paths)
            if dry_run:
                counts["derivative_paths"] += len(planned_receipts)
                completed_paths += len(planned_receipts)
                if planned_receipts:
                    report("Removing artifact paths", completed_paths, total_paths)
            else:
                with registry.connection(write=True) as db:
                    db.executemany(
                        "UPDATE work_items SET artifact_state='missing',artifact_reason='Purged by user',updated_at=? WHERE id=?",
                        [(utcnow(), work_item_id) for work_item_id in ids],
                    )
                    from nro.orchestration.registry_work_items import cancel_purged_demand

                    counts["demand_links"] = cancel_purged_demand(db, ids, now=utcnow())
    from nro.orchestration.control_paths import ControlPaths

    scoped = Registry(
        replace(
            registry.paths,
            events=ControlPaths(registry.paths.control).branch(view["branch"]) / "events",
        )
    )
    report("Removing logs", 0, 1)
    counts["attempt_logs"] = _purge_attempt_logs(scoped, work_item_ids=ids, dry_run=dry_run)
    counts["worker_logs"] = _purge_inactive_worker_logs(registry, dry_run=dry_run)
    report("Removing logs", 1, 1)
    if not logs_only and not dry_run:
        from nro.orchestration.ownership import lineage_root, remove_empty_ownership_root

        deleted, retained = registry.forget_purged_work_items_detailed(ids)
        for work_item_id in deleted:
            location = receipt_locations.get(work_item_id)
            if location is None:
                continue
            path, root = location
            counts["derivative_paths"] += int(
                _remove_path(
                    path,
                    dry_run=False,
                    prune_root=lineage_root(*root),
                )
            )
            completed_paths += 1
        for work_item_id in deleted:
            location = receipt_locations.get(work_item_id)
            if location is not None:
                remove_empty_ownership_root(*location[1])
        if planned_receipts:
            report("Removing artifact paths", completed_paths, total_paths)
        counts["scheduler_records"] = len(deleted)
        counts["retained_dependency_records"] = len(retained)
        counts["retained_dependency_modules"] = registry.retained_dependency_modules(retained)
    return counts
