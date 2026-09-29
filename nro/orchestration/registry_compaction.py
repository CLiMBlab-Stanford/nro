"""Retain current scheduler state without accumulating execution history."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from nro.orchestration.registry_work_items import forget_purged_work_items_detailed
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
        targets = frozenset(
            int(item[0])
            for item in database.execute(
                """SELECT work_item_id FROM request_work_items
                   WHERE request_id=? AND role='target' AND demand_state='active'""",
                (row["id"],),
            )
        )
        current = selected.get(key)
        if current is None or current[0] != targets:
            selected[key] = (targets, str(row["id"]))
            continue
        superseded.append(str(row["id"]))
    if not superseded:
        return 0
    placeholders = ",".join("?" for _ in superseded)
    database.execute(
        f"UPDATE requests SET state='superseded' WHERE id IN ({placeholders})",
        superseded,
    )
    database.execute(
        f"""UPDATE request_work_items SET demand_state='cancelled'
            WHERE request_id IN ({placeholders})""",
        superseded,
    )
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


def _terminal_keys(database: sqlite3.Connection, request_id: str) -> set[str]:
    """Return logical terminal identities from a compact or complete request plan."""
    row = database.execute(
        "SELECT payload_json FROM request_plans WHERE request_id=?", (request_id,)
    ).fetchone()
    if row is None:
        return set()
    payload = decode_plan(row[0])
    return {str(value) for value in payload.get("terminals", ())}


def _compact_terminal_requests(database: sqlite3.Connection) -> tuple[int, int]:
    """Keep publication handles but discard their execution-demand histories."""
    removed_links = int(
        database.execute(
            """DELETE FROM request_work_items
               WHERE request_id IN (SELECT id FROM requests WHERE state!='active')"""
        ).rowcount
    )
    for row in database.execute(
        "SELECT id FROM requests WHERE state IN ('satisfied','registered')"
    ).fetchall():
        request_id = str(row["id"])
        terminals = _terminal_keys(database, request_id)
        artifacts = database.execute(
            """SELECT artifact.work_item_id,
                      COALESCE(execution.logical_key,item.work_item_key) AS logical_key
               FROM request_artifacts artifact
               JOIN work_items item ON item.id=artifact.work_item_id
               LEFT JOIN work_item_execution execution
                 ON execution.work_item_id=artifact.work_item_id
               WHERE artifact.request_id=?""",
            (request_id,),
        ).fetchall()
        discard = [
            int(item["work_item_id"]) for item in artifacts if item["logical_key"] not in terminals
        ]
        if discard:
            placeholders = ",".join("?" for _ in discard)
            cursor = database.execute(
                f"""DELETE FROM request_artifacts
                    WHERE request_id=? AND work_item_id IN ({placeholders})""",
                (request_id, *discard),
            )
            removed_links += int(cursor.rowcount)

    removable = [
        str(row[0])
        for row in database.execute(
            """SELECT request.id FROM requests request
               WHERE request.state IN ('cancelled','superseded')
                 AND NOT EXISTS (
                     SELECT 1 FROM scheduler_submissions submission
                     WHERE submission.request_id=request.id
                 )"""
        )
    ]
    if not removable:
        return removed_links, 0
    placeholders = ",".join("?" for _ in removable)
    for table in ("request_work_items", "request_artifacts", "request_plans", "request_owners"):
        cursor = database.execute(
            f"DELETE FROM {table} WHERE request_id IN ({placeholders})", removable
        )
        if table in {"request_work_items", "request_artifacts"}:
            removed_links += int(cursor.rowcount)
    database.execute(f"DELETE FROM requests WHERE id IN ({placeholders})", removable)
    return removed_links, len(removable)


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
    retained = {
        int(row[0])
        for row in database.execute(
            """SELECT id FROM attempts WHERE state IN ('queued','running','cancel_requested')
               UNION SELECT attempt_id FROM completions WHERE attempt_id IS NOT NULL
               UNION SELECT attempt_id FROM artifacts WHERE attempt_id IS NOT NULL
               UNION SELECT attempt_id FROM resource_step_tasks WHERE attempt_id IS NOT NULL
               UNION SELECT id FROM (
                   SELECT id,state,ROW_NUMBER() OVER (
                       PARTITION BY work_item_id ORDER BY id DESC
                   ) AS history_rank FROM attempts
               ) WHERE history_rank=1 AND state IN ('error','cancelled')"""
        )
    }
    discard = [
        int(row[0])
        for row in database.execute("SELECT id FROM attempts")
        if int(row[0]) not in retained
    ]
    if not discard:
        return 0
    placeholders = ",".join("?" for _ in discard)
    database.execute(
        f"DELETE FROM attempt_dependencies WHERE attempt_id IN ({placeholders})", discard
    )
    database.execute(f"DELETE FROM attempt_execution WHERE attempt_id IN ({placeholders})", discard)
    database.execute(f"DELETE FROM attempts WHERE id IN ({placeholders})", discard)
    return len(discard)


def _retained_work_items(database: sqlite3.Connection) -> set[int]:
    """Return current-state roots and their complete dependency closure."""
    retained = {
        int(row[0])
        for row in database.execute(
            """SELECT id FROM work_items WHERE artifact_state!='missing'
               UNION SELECT work_item_id FROM completions
               UNION SELECT work_item_id FROM artifacts
               UNION SELECT work_item_id FROM artifact_mutations
               UNION SELECT work_item_id FROM attempts
               UNION SELECT work_item_id FROM resource_step_tasks
                    WHERE state IN ('queued','running','cancel_requested')
               UNION SELECT link.work_item_id FROM request_work_items link
                    JOIN requests request ON request.id=link.request_id
                    WHERE request.state='active' AND link.demand_state='active'"""
        )
    }
    parents: dict[int, set[int]] = {}
    for row in database.execute(
        "SELECT work_item_id,upstream_work_item_id FROM work_item_dependencies"
    ):
        parents.setdefault(int(row[0]), set()).add(int(row[1]))
    pending = list(retained)
    while pending:
        for parent in parents.get(pending.pop(), ()):
            if parent not in retained:
                retained.add(parent)
                pending.append(parent)
    return retained


def _remove_unused_work_items(database: sqlite3.Connection) -> int:
    """Remove missing, undemanded work and records that only describe it."""
    retained = _retained_work_items(database)
    discard = [
        int(row[0])
        for row in database.execute("SELECT id FROM work_items")
        if int(row[0]) not in retained
    ]
    if not discard:
        return 0
    deleted, protected = forget_purged_work_items_detailed(database, discard)
    if protected:
        raise RuntimeError("Current-state roots did not include the complete dependency closure")
    return len(deleted)


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
    with registry.connection(write=True) as database:
        previous = database.execute(
            "SELECT value FROM metadata WHERE key='last_current_state_compaction'"
        ).fetchone()
        if (
            previous is not None
            and minimum_interval_seconds > 0
            and timestamp - float(previous[0]) < minimum_interval_seconds
        ):
            return CompactionReport(skipped=True)

        coalesced = _coalesce_active_requests(database)
        compacted_plans = compact_terminal_plans(database)
        removed_submissions = _remove_terminal_submissions(database)
        removed_links, removed_requests = _compact_terminal_requests(database)
        removed_attempts = _remove_attempt_history(database)
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
