"""Read and project work-item state from an existing registry transaction."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence


def work_item_rows(database: sqlite3.Connection) -> list[dict]:
    """Read work-item facts and attach each item's configuration route."""
    rows = database.execute(
        """
        SELECT t.*, ci.config_id, ci.directory_label, ci.lineage_fingerprint,
               ci.configuration_class,
               EXISTS(
                 SELECT 1 FROM workflow_bindings binding
                 WHERE binding.module_lineage_id=t.module_lineage_id
               ) AS recomputable,
               EXISTS(SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.work_item_id=t.id AND rt.demand_state='active' AND r.state='active') AS demanded,
               (SELECT a.state FROM attempts a WHERE a.work_item_id=t.id ORDER BY a.id DESC LIMIT 1) AS attempt_state,
               (SELECT a.error_type FROM attempts a WHERE a.work_item_id=t.id ORDER BY a.id DESC LIMIT 1) AS error_type,
               (SELECT a.error_message FROM attempts a WHERE a.work_item_id=t.id ORDER BY a.id DESC LIMIT 1) AS error_message,
               (SELECT a.log_path FROM attempts a WHERE a.work_item_id=t.id ORDER BY a.id DESC LIMIT 1) AS log_path,
               (SELECT COUNT(*) FROM attempts a WHERE a.work_item_id=t.id AND a.oom_detected=1) AS oom_count,
               (SELECT a.memory_gb FROM attempts a WHERE a.work_item_id=t.id ORDER BY a.id DESC LIMIT 1) AS attempt_memory_gb,
               EXISTS(
                 SELECT 1 FROM request_work_items retry_rt JOIN requests retry ON retry.id=retry_rt.request_id
                 WHERE retry_rt.work_item_id=t.id AND retry_rt.demand_state='active' AND retry.state='active'
                   AND retry.updated_at > COALESCE(
                     (SELECT CASE WHEN latest.state='cancelled'
                              THEN latest.started_at ELSE latest.completed_at END
                      FROM attempts latest WHERE latest.work_item_id=t.id
                      ORDER BY latest.id DESC LIMIT 1),
                     ''
                   )
               ) AS retry_requested,
               (SELECT GROUP_CONCAT(DISTINCT wr.workflow_id)
                 FROM request_work_items rt
                 JOIN requests r ON r.id=rt.request_id
                 JOIN workflow_revisions wr ON wr.id=r.workflow_revision_id
                 WHERE rt.work_item_id=t.id) AS workflow_ids
        FROM work_items t
        JOIN module_lineages ci ON ci.id=t.module_lineage_id
        ORDER BY t.participant, t.module, t.work_item_key
        """
    ).fetchall()
    lineages = {
        int(row["id"]): {
            "module": str(row["configuration_class"]),
            "config": str(row["config_id"]),
        }
        for row in database.execute(
            "SELECT id, configuration_class, config_id FROM module_lineages"
        )
    }
    parents: dict[int, list[int]] = {}
    for edge in database.execute(
        """SELECT module_lineage_id, upstream_module_lineage_id
           FROM module_lineage_dependencies"""
    ):
        parents.setdefault(int(edge[0]), []).append(int(edge[1]))

    def route(lineage_id: int) -> list[dict[str, str]]:
        ordered: list[dict[str, str]] = []
        visited: set[int] = set()

        def visit(current: int) -> None:
            if current in visited:
                return
            visited.add(current)
            for parent in parents.get(current, ()):
                visit(parent)
            if current in lineages:
                ordered.append(lineages[current])

        visit(lineage_id)
        return ordered

    result = []
    for row in rows:
        item = dict(row)
        item["configuration_route_json"] = json.dumps(
            route(int(item["module_lineage_id"])),
            separators=(",", ":"),
            sort_keys=True,
        )
        result.append(item)
    return result


def work_item_dependencies(database: sqlite3.Connection) -> list[tuple[int, int]]:
    """Read work-item dependency pairs from an existing connection."""
    return [
        (int(row["work_item_id"]), int(row["upstream_work_item_id"]))
        for row in database.execute(
            "SELECT work_item_id, upstream_work_item_id FROM work_item_dependencies"
        )
    ]


def project_work_item_status(
    rows: Sequence[Mapping[str, object]],
    dependencies: Sequence[tuple[int, int]],
    *,
    artifact_states: Mapping[int, tuple[str, str]] | None = None,
) -> list[dict]:
    """Combine artifact, attempt, demand, and dependency facts into display states."""
    projected_rows = [dict(row) for row in rows]
    if artifact_states:
        for row in projected_rows:
            projected = artifact_states.get(int(row["id"]))
            if projected is not None:
                row["artifact_state"], row["artifact_reason"] = projected
    by_id = {int(row["id"]): row for row in projected_rows}
    parents: dict[int, list[int]] = {}
    for work_item_id, upstream_id in dependencies:
        parents.setdefault(work_item_id, []).append(upstream_id)
    root_cache: dict[int, tuple[int, ...]] = {}

    def failure_roots(work_item_id: int) -> tuple[int, ...]:
        if work_item_id in root_cache:
            return root_cache[work_item_id]
        row = by_id[work_item_id]
        roots: set[int] = set()
        if (
            row["artifact_state"] != "fresh"
            and row.get("attempt_state") == "error"
            and not row.get("retry_requested")
            and not (
                row["artifact_state"] == "missing"
                and row.get("artifact_reason") == "Purged by user"
                and not row.get("demanded")
            )
        ):
            roots.add(work_item_id)
        for parent in parents.get(work_item_id, ()):
            if parent in by_id:
                roots.update(failure_roots(parent))
        root_cache[work_item_id] = tuple(sorted(roots))
        return root_cache[work_item_id]

    snapshot: list[dict] = []
    for row in projected_rows:
        item = dict(row)
        work_item_id = int(item["id"])
        roots = failure_roots(work_item_id)
        unfinished_dependencies = tuple(
            sorted(
                parent
                for parent in parents.get(work_item_id, ())
                if parent in by_id and by_id[parent]["artifact_state"] != "fresh"
            )
        )
        attempt = item.get("attempt_state")
        if item["artifact_state"] == "fresh":
            state = "Success"
        elif not item.get("recomputable") and not item.get("demanded"):
            state = "Unavailable"
        elif attempt == "error" and work_item_id in roots:
            state = "Corrupt" if item["artifact_state"] == "corrupt" else "Error"
        elif roots and item.get("demanded"):
            state = "Blocked"
        elif attempt == "cancel_requested":
            state = "Stopping"
        elif attempt == "running":
            state = "Running"
        elif unfinished_dependencies and item.get("demanded"):
            state = "Waiting"
        elif attempt == "queued" or item.get("retry_requested"):
            state = "Queued"
        elif (
            item["artifact_state"] == "missing"
            and item.get("artifact_reason") == "Purged by user"
            and not item.get("demanded")
        ):
            state = "Missing"
        elif attempt == "error":
            state = "Error"
        elif item.get("demanded"):
            state = "Queued"
        elif attempt == "cancelled" and item.get("error_type") == "UserCancelled":
            state = "Stopped"
        elif item["artifact_state"] == "corrupt":
            state = "Corrupt"
        elif item["artifact_state"] == "missing":
            state = "Missing"
        else:
            state = "Stale"
        item["status"] = state
        item["root_failure_ids"] = roots
        item["unfinished_dependency_ids"] = unfinished_dependencies
        snapshot.append(item)
    return snapshot
