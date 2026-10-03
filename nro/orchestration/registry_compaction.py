"""Retain current scheduler state without accumulating execution history."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from nro.orchestration.request_plans import compact_terminal_plans, decode_plan

_ACTIVE_SUBMISSIONS = ("prepared", "submitted", "running", "cancel_requested")
_ACTIVE_WORKERS = ("idle", "running", "draining", "shutdown_requested")


@dataclass(frozen=True)
class CompactionReport:
    """Summarize records removed by one current-state retention pass."""

    coalesced_requests: int = 0
    compacted_plans: int = 0
    removed_requests: int = 0
    removed_request_links: int = 0
    removed_attempts: int = 0
    removed_submissions: int = 0
    removed_workers: int = 0
    removed_work_items: int = 0
    expired_scheduler_requests: int = 0
    skipped: bool = False

    @property
    def changed(self) -> bool:
        """Return whether the pass removed or compacted persisted state."""
        return any(
            (
                self.coalesced_requests,
                self.compacted_plans,
                self.removed_requests,
                self.removed_request_links,
                self.removed_attempts,
                self.removed_submissions,
                self.removed_workers,
                self.removed_work_items,
                self.expired_scheduler_requests,
            )
        )


def _coalesce_active_requests(database: sqlite3.Connection) -> int:
    """Supersede older demand records that select the same terminal work."""
    rows = database.execute(
        """SELECT request.id,request.project,request.workflow_revision_id,
                  request.target_module,request.selectors_json,request.created_at,
                  owner.registry_id
           FROM requests request
           JOIN request_owners owner ON owner.request_id=request.id
           WHERE request.state='active'
           ORDER BY request.created_at DESC,request.id DESC"""
    ).fetchall()
    targets_by_request: dict[str, set[int]] = {}
    for target in database.execute(
        """SELECT link.request_id,link.work_item_id FROM request_work_items link
           JOIN requests request ON request.id=link.request_id
           WHERE request.state='active' AND link.role='target'
             AND link.demand_state='active'"""
    ):
        targets_by_request.setdefault(str(target["request_id"]), set()).add(
            int(target["work_item_id"])
        )
    selected: dict[tuple[object, ...], tuple[frozenset[int], str]] = {}
    superseded = []
    for row in rows:
        key = (
            row["registry_id"],
            row["project"],
            row["workflow_revision_id"],
            row["target_module"],
            row["selectors_json"],
        )
        targets = frozenset(targets_by_request.get(str(row["id"]), ()))
        current = selected.get(key)
        if current is None or current[0] != targets:
            selected[key] = (targets, str(row["id"]))
            continue
        superseded.append(str(row["id"]))
    if not superseded:
        return 0
    database.execute("DROP TABLE IF EXISTS temp.compaction_superseded")
    database.execute(
        "CREATE TEMP TABLE compaction_superseded(id TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    database.executemany(
        "INSERT INTO compaction_superseded VALUES (?)",
        ((request_id,) for request_id in superseded),
    )
    database.execute(
        "UPDATE requests SET state='superseded' WHERE id IN (SELECT id FROM compaction_superseded)"
    )
    database.execute(
        """UPDATE request_work_items SET demand_state='cancelled'
           WHERE request_id IN (SELECT id FROM compaction_superseded)"""
    )
    database.execute("DROP TABLE compaction_superseded")
    return len(superseded)


def _remove_terminal_submissions(database: sqlite3.Connection) -> int:
    """Remove finished allocation records after no active worker references them."""
    placeholders = ",".join("?" for _ in _ACTIVE_SUBMISSIONS)
    cursor = database.execute(
        f"""DELETE FROM scheduler_submissions
            WHERE state NOT IN ({placeholders})
              AND NOT EXISTS (
                  SELECT 1 FROM workers
                  WHERE workers.successor_submission_id=scheduler_submissions.id
                    AND workers.state IN ('idle','running','draining','shutdown_requested')
              )""",
        _ACTIVE_SUBMISSIONS,
    )
    return int(cursor.rowcount)


def _compact_terminal_requests(database: sqlite3.Connection) -> tuple[int, int]:
    """Keep publication handles but discard their execution-demand histories."""
    removed_links = int(
        database.execute(
            """DELETE FROM request_work_items
               WHERE request_id IN (SELECT id FROM requests WHERE state!='active')"""
        ).rowcount
    )
    database.execute("DROP TABLE IF EXISTS temp.compaction_terminals")
    database.execute(
        """CREATE TEMP TABLE compaction_terminals(
               request_id TEXT NOT NULL,
               logical_key TEXT NOT NULL,
               PRIMARY KEY(request_id,logical_key)
           ) WITHOUT ROWID"""
    )
    cursor = database.execute(
        """SELECT request.id,plan.payload_json FROM requests request
           LEFT JOIN request_plans plan ON plan.request_id=request.id
           WHERE request.state IN ('satisfied','registered')"""
    )
    while rows := cursor.fetchmany(64):
        terminals = []
        for row in rows:
            payload = {} if row["payload_json"] is None else decode_plan(row["payload_json"])
            terminals.extend(
                (str(row["id"]), str(key)) for key in payload.get("terminals", ())
            )
        database.executemany("INSERT INTO compaction_terminals VALUES (?,?)", terminals)
    cursor = database.execute(
        """DELETE FROM request_artifacts AS artifact
           WHERE artifact.request_id IN (
               SELECT id FROM requests WHERE state IN ('satisfied','registered')
           ) AND NOT EXISTS (
               SELECT 1 FROM compaction_terminals terminal
               JOIN work_items item ON item.id=artifact.work_item_id
               LEFT JOIN work_item_execution execution
                 ON execution.work_item_id=artifact.work_item_id
               WHERE terminal.request_id=artifact.request_id
                 AND terminal.logical_key=COALESCE(execution.logical_key,item.work_item_key)
           )"""
    )
    removed_links += int(cursor.rowcount)
    database.execute("DROP TABLE compaction_terminals")

    database.execute("DROP TABLE IF EXISTS temp.compaction_requests")
    database.execute(
        """CREATE TEMP TABLE compaction_requests(id TEXT PRIMARY KEY) WITHOUT ROWID"""
    )
    database.execute(
        """INSERT INTO compaction_requests
           SELECT request.id FROM requests request
           WHERE request.state IN ('cancelled','superseded')
             AND NOT EXISTS (
                 SELECT 1 FROM scheduler_submissions submission
                 WHERE submission.request_id=request.id
             )"""
    )
    removable = int(
        database.execute("SELECT COUNT(*) FROM compaction_requests").fetchone()[0]
    )
    if not removable:
        database.execute("DROP TABLE compaction_requests")
        return removed_links, 0
    for table in ("request_work_items", "request_artifacts", "request_plans", "request_owners"):
        cursor = database.execute(
            f"DELETE FROM {table} WHERE request_id IN (SELECT id FROM compaction_requests)"
        )
        if table in {"request_work_items", "request_artifacts"}:
            removed_links += int(cursor.rowcount)
    database.execute("DELETE FROM requests WHERE id IN (SELECT id FROM compaction_requests)")
    database.execute("DROP TABLE compaction_requests")
    return removed_links, removable


def _remove_attempt_history(database: sqlite3.Connection) -> int:
    """Retain active, current diagnostic, and current-completion attempts."""
    database.execute(
        """DELETE FROM attempt_dependencies WHERE attempt_id IN (
               SELECT id FROM attempts
               WHERE state NOT IN ('queued','running','cancel_requested')
           )"""
    )
    database.execute(
        """DELETE FROM attempt_execution WHERE attempt_id IN (
               SELECT id FROM attempts
               WHERE state NOT IN ('queued','running','cancel_requested')
           )"""
    )
    database.execute(
        """UPDATE resource_step_tasks SET worker_id=NULL,attempt_id=NULL
           WHERE state NOT IN ('queued','running','cancel_requested')"""
    )
    cursor = database.execute(
        """DELETE FROM attempts AS candidate
           WHERE candidate.state NOT IN ('queued','running','cancel_requested')
             AND NOT EXISTS (
                 SELECT 1 FROM completions WHERE completions.attempt_id=candidate.id
             )
             AND NOT EXISTS (
                 SELECT 1 FROM artifacts WHERE artifacts.attempt_id=candidate.id
             )
             AND NOT EXISTS (
                 SELECT 1 FROM resource_step_tasks
                 WHERE resource_step_tasks.attempt_id=candidate.id
             )
             AND NOT (
                 candidate.state IN ('error','cancelled')
                 AND candidate.id=(
                     SELECT MAX(latest.id) FROM attempts latest
                     WHERE latest.work_item_id=candidate.work_item_id
                 )
             )"""
    )
    return int(cursor.rowcount)


def _remove_unused_work_items(database: sqlite3.Connection) -> int:
    """Remove missing, undemanded work and records that only describe it."""
    database.execute("DROP TABLE IF EXISTS temp.compaction_work_items")
    database.execute(
        """CREATE TEMP TABLE compaction_work_items(id INTEGER PRIMARY KEY) WITHOUT ROWID"""
    )
    database.execute(
        """WITH RECURSIVE retained(id) AS (
               SELECT id FROM work_items WHERE artifact_state!='missing'
               UNION SELECT work_item_id FROM completions
               UNION SELECT work_item_id FROM artifacts
               UNION SELECT work_item_id FROM artifact_mutations
               UNION SELECT work_item_id FROM attempts
               UNION SELECT work_item_id FROM resource_step_tasks
                    WHERE state IN ('queued','running','cancel_requested')
               UNION SELECT link.work_item_id FROM request_work_items link
                    JOIN requests request ON request.id=link.request_id
                    WHERE request.state='active' AND link.demand_state='active'
               UNION SELECT edge.upstream_work_item_id
                    FROM work_item_dependencies edge
                    JOIN retained child ON child.id=edge.work_item_id
           )
           INSERT INTO compaction_work_items
           SELECT id FROM work_items WHERE id NOT IN (SELECT id FROM retained)"""
    )
    removed = int(
        database.execute("SELECT COUNT(*) FROM compaction_work_items").fetchone()[0]
    )
    if not removed:
        database.execute("DROP TABLE compaction_work_items")
        return 0
    database.execute(
        """DELETE FROM attempt_dependencies WHERE attempt_id IN (
               SELECT attempt.id FROM attempts attempt
               JOIN compaction_work_items item ON item.id=attempt.work_item_id
           ) OR upstream_work_item_id IN (SELECT id FROM compaction_work_items)"""
    )
    database.execute(
        """DELETE FROM attempt_execution WHERE attempt_id IN (
               SELECT attempt.id FROM attempts attempt
               JOIN compaction_work_items item ON item.id=attempt.work_item_id
           )"""
    )
    for table in (
        "artifacts",
        "completions",
        "attempts",
        "request_work_items",
        "request_artifacts",
        "artifact_mutations",
    ):
        database.execute(
            f"DELETE FROM {table} WHERE work_item_id IN (SELECT id FROM compaction_work_items)"
        )
    database.execute(
        """DELETE FROM compiled_revisions WHERE EXISTS (
               SELECT 1 FROM branch_work_items binding
               JOIN compaction_work_items item ON item.id=binding.work_item_id
               WHERE binding.registry_id=compiled_revisions.registry_id
                 AND binding.logical_key=compiled_revisions.logical_key
           )"""
    )
    database.execute(
        "DELETE FROM branch_work_items WHERE work_item_id IN (SELECT id FROM compaction_work_items)"
    )
    database.execute(
        "DELETE FROM work_item_execution WHERE work_item_id IN (SELECT id FROM compaction_work_items)"
    )
    database.execute(
        """DELETE FROM work_item_dependencies
           WHERE work_item_id IN (SELECT id FROM compaction_work_items)
              OR upstream_work_item_id IN (SELECT id FROM compaction_work_items)"""
    )
    database.execute("DELETE FROM work_items WHERE id IN (SELECT id FROM compaction_work_items)")
    database.execute("DROP TABLE compaction_work_items")
    return removed


def _remove_terminal_workers(database: sqlite3.Connection) -> int:
    """Remove workers after all retained diagnostics and allocations release them."""
    placeholders = ",".join("?" for _ in _ACTIVE_WORKERS)
    cursor = database.execute(
        f"""DELETE FROM workers
            WHERE state NOT IN ({placeholders})
              AND NOT EXISTS (SELECT 1 FROM attempts WHERE attempts.worker_id=workers.id)
              AND NOT EXISTS (
                  SELECT 1 FROM resource_step_tasks WHERE resource_step_tasks.worker_id=workers.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM scheduler_submissions
                  WHERE scheduler_submissions.predecessor_worker_id=workers.id
              )""",
        _ACTIVE_WORKERS,
    )
    return int(cursor.rowcount)


def _expire_scheduler_requests(database: sqlite3.Connection) -> int:
    """Retain committed transport responses for the bounded retry window."""
    cursor = database.execute(
        """DELETE FROM scheduler_requests
           WHERE state='completed' AND datetime(updated_at) < datetime('now','-1 hour')"""
    )
    return int(cursor.rowcount)


def compact_registry(
    registry,
    *,
    minimum_interval_seconds: float = 3600.0,
    now: float | None = None,
) -> CompactionReport:
    """Apply current-state retention through the scheduler's registry transaction."""
    timestamp = time.time() if now is None else now
    with registry.connection() as database:
        previous = database.execute(
            "SELECT value FROM metadata WHERE key='last_current_state_compaction'"
        ).fetchone()
        if (
            previous is not None
            and minimum_interval_seconds > 0
            and timestamp - float(previous[0]) < minimum_interval_seconds
        ):
            return CompactionReport(skipped=True)

    with registry.connection(write=True) as database:
        coalesced = _coalesce_active_requests(database)
        removed_submissions = _remove_terminal_submissions(database)
        removed_links, removed_requests = _compact_terminal_requests(database)

    with registry.connection(write=True) as database:
        compacted_plans = compact_terminal_plans(database)

    with registry.connection(write=True) as database:
        removed_attempts = _remove_attempt_history(database)

    with registry.connection(write=True) as database:
        removed_work_items = _remove_unused_work_items(database)
        removed_workers = _remove_terminal_workers(database)
        expired = _expire_scheduler_requests(database)
        database.execute(
            """INSERT INTO metadata(key,value)
               VALUES ('last_current_state_compaction',?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (str(timestamp),),
        )

    # Existing schema-24 registries simply retain reusable free pages. Schema
    # 25 and new registries return a bounded number to the filesystem per pass.
    with registry.connection() as database:
        if int(database.execute("PRAGMA auto_vacuum").fetchone()[0]) == 2:
            database.execute("PRAGMA incremental_vacuum(4096)")
    return CompactionReport(
        coalesced_requests=coalesced,
        compacted_plans=compacted_plans,
        removed_requests=removed_requests,
        removed_request_links=removed_links,
        removed_attempts=removed_attempts,
        removed_submissions=removed_submissions,
        removed_workers=removed_workers,
        removed_work_items=removed_work_items,
        expired_scheduler_requests=expired,
    )
