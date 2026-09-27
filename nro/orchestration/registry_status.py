"""Read and project work-item state from an existing registry transaction."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence


def work_item_rows(database: sqlite3.Connection) -> list[dict]:
    """Read work-item facts and attach each item's configuration route."""
    rows = database.execute(
        """
        WITH latest_attempt AS (
            SELECT * FROM (
                SELECT attempt.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY attempt.work_item_id ORDER BY attempt.id DESC
                       ) AS history_rank
                FROM attempts attempt
            ) WHERE history_rank=1
        ),
        attempt_summary AS (
            SELECT work_item_id,
                   SUM(CASE WHEN oom_detected=1 THEN 1 ELSE 0 END) AS oom_count
            FROM attempts GROUP BY work_item_id
        ),
        current_resource_step AS (
            SELECT * FROM (
                SELECT task.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY task.work_item_id ORDER BY task.id DESC
                       ) AS history_rank
                FROM resource_step_tasks task
                JOIN work_items current ON current.id=task.work_item_id
                WHERE task.generation=current.current_generation
                  AND task.revision_fingerprint=current.revision_fingerprint
            ) WHERE history_rank=1
        ),
        request_summary AS (
            SELECT link.work_item_id,
                   MAX(CASE WHEN link.demand_state='active' AND request.state='active'
                            THEN 1 ELSE 0 END) AS demanded,
                   MAX(CASE WHEN link.demand_state='active' AND request.state='active'
                                  AND request.updated_at > COALESCE(
                                      CASE WHEN attempt.state='cancelled'
                                           THEN attempt.started_at
                                           ELSE attempt.completed_at END,
                                      '')
                            THEN 1 ELSE 0 END) AS retry_requested,
                   GROUP_CONCAT(DISTINCT workflow.workflow_id) AS workflow_ids
            FROM request_work_items link
            JOIN requests request ON request.id=link.request_id
            JOIN workflow_revisions workflow ON workflow.id=request.workflow_revision_id
            LEFT JOIN latest_attempt attempt ON attempt.work_item_id=link.work_item_id
            GROUP BY link.work_item_id
        )
        SELECT t.*, lineage.config_id, lineage.directory_label,
               lineage.lineage_fingerprint, lineage.configuration_class,
               EXISTS(
                   SELECT 1 FROM workflow_bindings binding
                   WHERE binding.module_lineage_id=t.module_lineage_id
               ) AS recomputable,
               COALESCE(requests.demanded, 0) AS demanded,
               attempt.state AS attempt_state,
               attempt.error_type,
               attempt.error_message,
               attempt.log_path,
               COALESCE(attempts.oom_count, 0) AS oom_count,
               attempt.memory_gb AS attempt_memory_gb,
               resource.state AS resource_step_state,
               resource.resource_class AS waiting_resource_class,
               resource.error_type AS resource_step_error_type,
               resource.error_message AS resource_step_error_message,
               COALESCE(requests.retry_requested, 0) AS retry_requested,
               requests.workflow_ids
        FROM work_items t
        JOIN module_lineages lineage ON lineage.id=t.module_lineage_id
        LEFT JOIN latest_attempt attempt ON attempt.work_item_id=t.id
        LEFT JOIN attempt_summary attempts ON attempts.work_item_id=t.id
        LEFT JOIN current_resource_step resource ON resource.work_item_id=t.id
        LEFT JOIN request_summary requests ON requests.work_item_id=t.id
        ORDER BY t.participant, t.module, t.work_item_key
        """
    ).fetchall()
    elapsed_by_work_item = {
        int(row["work_item_id"]): float(row["elapsed_seconds"])
        for row in database.execute(
            """
            SELECT a.work_item_id,
                   SUM(MAX(0.0, (julianday(COALESCE(a.completed_at, CURRENT_TIMESTAMP))
                                 - julianday(a.started_at)) * 86400.0)) AS elapsed_seconds
            FROM attempts a
            JOIN work_items t ON t.id=a.work_item_id
            WHERE a.started_at IS NOT NULL
              AND a.revision_fingerprint=t.revision_fingerprint
            GROUP BY a.work_item_id
            """
        )
    }
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
        item["elapsed_seconds"] = elapsed_by_work_item.get(int(item["id"]))
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
        resource_step = item.get("resource_step_state")
        if item["artifact_state"] == "fresh":
            state = "Success"
        elif not item.get("recomputable") and not item.get("demanded"):
            state = "Unavailable"
        elif attempt == "error" and work_item_id in roots:
            if item.get("error_type") == "Timeout":
                state = "Timeout"
            else:
                state = "Corrupt" if item["artifact_state"] == "corrupt" else "Error"
        elif roots and item.get("demanded"):
            state = "Blocked"
        elif attempt == "cancel_requested":
            state = "Stopping"
        elif attempt == "running":
            state = "Running"
        elif resource_step == "running" and item.get("demanded"):
            state = "Running"
        elif resource_step == "pending" and item.get("demanded"):
            state = "Queued"
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
            state = "Timeout" if item.get("error_type") == "Timeout" else "Error"
        elif item.get("demanded"):
            state = "Queued"
        elif attempt == "cancelled" and item.get("error_type") == "UserCancelled":
            state = "Stopped"
        elif (
            resource_step == "cancelled" and item.get("resource_step_error_type") == "UserCancelled"
        ):
            state = "Stopped"
            item["error_type"] = item["resource_step_error_type"]
            item["error_message"] = item["resource_step_error_message"]
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
