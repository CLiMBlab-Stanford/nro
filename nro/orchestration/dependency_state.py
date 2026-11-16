"""Apply invalidation to resolved work-item dependencies, independent of their owners."""

import sqlite3
from collections.abc import Iterable

from nro.orchestration.sql_values import JSON_VALUES, json_values

ACTIVE = "('queued','running','cancel_requested')"
WRITE_READY = f"""NOT EXISTS (
    SELECT 1 FROM attempt_dependencies pinned JOIN attempts reader ON reader.id=pinned.attempt_id
    WHERE pinned.upstream_work_item_id=t.id AND reader.state IN {ACTIVE}
) AND NOT EXISTS (
    WITH RECURSIVE ancestors(id) AS (
        SELECT t.id UNION SELECT edge.upstream_work_item_id FROM work_item_dependencies edge
        JOIN ancestors parent ON edge.work_item_id=parent.id
    ) SELECT 1 FROM ancestors JOIN artifact_mutations mutation ON mutation.work_item_id=ancestors.id
)"""

EDGES = f"""SELECT work_item_id, upstream_work_item_id FROM work_item_dependencies
UNION SELECT reader.work_item_id, pinned.upstream_work_item_id
FROM attempt_dependencies pinned JOIN attempts reader ON reader.id=pinned.attempt_id
WHERE reader.state IN {ACTIVE}"""


class AttemptInvalidated(RuntimeError):
    """Completion was superseded while the attempt was computing or being validated."""


def capture_inputs(db: sqlite3.Connection, attempt_id: int, work_item_id: int) -> None:
    """Pin all resolved ancestors and generations under the same lock as claiming work."""
    db.execute(
        """WITH RECURSIVE ancestors(id) AS (
        SELECT upstream_work_item_id FROM work_item_dependencies WHERE work_item_id=?
        UNION SELECT edge.upstream_work_item_id FROM work_item_dependencies edge
              JOIN ancestors parent ON edge.work_item_id=parent.id
    ) INSERT INTO attempt_dependencies(attempt_id, upstream_work_item_id, generation)
      SELECT ?, upstream.id, upstream.current_generation
      FROM ancestors JOIN work_items upstream ON upstream.id=ancestors.id""",
        (work_item_id, attempt_id),
    )


def invalidate(
    db: sqlite3.Connection,
    roots: Iterable[int],
    *,
    now: str,
    reason: str,
    error_type: str = "UpstreamStale",
    include_roots: bool = False,
) -> list[dict]:
    """Invalidate and cancel transitive consumers, preserving demand and captured edges.

    The caller supplies concrete producer IDs, not branch names. Only fresh
    downstream artifacts change state; missing outputs and existing errors keep
    their more specific reasons. Set include_roots when the supplied work items
    themselves need invalidation. Cancellation is not confirmation of shutdown.
    """
    roots = tuple(sorted(set(roots)))
    if not roots:
        return []
    descendants = tuple(
        row[0]
        for row in db.execute(
            f"""WITH RECURSIVE
        edges(work_item_id, upstream_work_item_id) AS ({EDGES}),
        affected(id) AS (
            SELECT work_item_id FROM edges WHERE upstream_work_item_id IN ({JSON_VALUES})
            UNION SELECT edge.work_item_id FROM edges edge JOIN affected parent ON edge.upstream_work_item_id=parent.id
        ) SELECT id FROM affected""",
            (json_values(roots),),
        )
    )
    if include_roots:
        descendants = tuple(sorted(set(descendants).union(roots)))
    return _apply_invalidation(db, descendants, now=now, reason=reason, error_type=error_type)


def _apply_invalidation(
    db: sqlite3.Connection,
    descendants: Iterable[int],
    *,
    now: str,
    reason: str,
    error_type: str,
) -> list[dict]:
    """Mark fresh work items stale and request cancellation of their attempts."""
    descendants = tuple(descendants)
    if not descendants:
        return []
    descendant_values = json_values(descendants)
    db.execute(
        f"""UPDATE work_items SET artifact_state='stale', artifact_reason=?, updated_at=?
                   WHERE id IN ({JSON_VALUES}) AND artifact_state='fresh' """,
        (reason, now, descendant_values),
    )
    rows = [
        dict(row)
        for row in db.execute(
            f"""SELECT a.id AS attempt_id, a.work_item_id, a.worker_id,
        i.work_item_key, i.module FROM attempts a JOIN work_items i ON i.id=a.work_item_id
        WHERE a.work_item_id IN ({JSON_VALUES}) AND a.state IN ('queued','running')""",
            (descendant_values,),
        )
    ]
    db.executemany(
        """UPDATE attempts SET state='cancel_requested', error_type=?, error_message=?, completed_at=NULL
                      WHERE id=?""",
        [(error_type, reason, row["attempt_id"]) for row in rows],
    )
    return rows


def synchronize(db: sqlite3.Connection, *, now: str) -> list[dict]:
    """Propagate current staleness and changed captured generations before scheduling."""
    db.execute(
        """UPDATE work_items SET artifact_state='stale', artifact_reason='Outputs reserved for mutation',
                  updated_at=? WHERE id IN (SELECT work_item_id FROM artifact_mutations) AND artifact_state='fresh'""",
        (now,),
    )
    changed = [
        row[0]
        for row in db.execute("""SELECT edge.work_item_id
        FROM work_item_dependencies edge JOIN work_items upstream ON upstream.id=edge.upstream_work_item_id
        JOIN work_items consumer ON consumer.id=edge.work_item_id
        WHERE consumer.artifact_state='fresh' AND edge.required_generation IS NOT NULL
              AND edge.required_generation!=upstream.current_generation""")
    ]
    cancelled = invalidate(
        db,
        changed,
        now=now,
        include_roots=True,
        reason="A completed artifact used an earlier upstream generation",
    )
    changed_pins = [
        row[0]
        for row in db.execute(f"""SELECT pinned.upstream_work_item_id
        FROM attempt_dependencies pinned JOIN work_items upstream ON upstream.id=pinned.upstream_work_item_id
        JOIN attempts reader ON reader.id=pinned.attempt_id
        WHERE reader.state IN {ACTIVE} AND pinned.generation!=upstream.current_generation""")
    ]
    # Invalidation changes only fresh work items and queued or running
    # attempts. Every non-fresh work item is a root, so any such target is
    # reachable from a root through fresh intermediates alone. Recursing only
    # through fresh work items therefore reaches every target that a full
    # descendant walk would, without revisiting the whole non-fresh graph.
    affected = [
        row[0]
        for row in db.execute(
            f"""WITH RECURSIVE
        changed_pins(id) AS ({JSON_VALUES}),
        target(id) AS (
            SELECT id FROM work_items WHERE artifact_state='fresh'
            UNION SELECT work_item_id FROM attempts WHERE state IN ('queued','running')
        ),
        affected(id) AS (
            SELECT target.id FROM target WHERE EXISTS (
                SELECT 1 FROM work_item_dependencies edge
                JOIN work_items upstream ON upstream.id=edge.upstream_work_item_id
                WHERE edge.work_item_id=target.id
                  AND (upstream.artifact_state!='fresh' OR upstream.id IN changed_pins)
            ) OR EXISTS (
                SELECT 1 FROM attempts reader
                JOIN attempt_dependencies pinned ON pinned.attempt_id=reader.id
                JOIN work_items upstream ON upstream.id=pinned.upstream_work_item_id
                WHERE reader.work_item_id=target.id AND reader.state IN {ACTIVE}
                  AND (upstream.artifact_state!='fresh' OR upstream.id IN changed_pins)
            )
            UNION SELECT edge.work_item_id FROM affected parent
            JOIN work_items through ON through.id=parent.id AND through.artifact_state='fresh'
            JOIN work_item_dependencies edge ON edge.upstream_work_item_id=parent.id
            WHERE edge.work_item_id IN target
            UNION SELECT reader.work_item_id FROM affected parent
            JOIN work_items through ON through.id=parent.id AND through.artifact_state='fresh'
            JOIN attempt_dependencies pinned ON pinned.upstream_work_item_id=parent.id
            JOIN attempts reader ON reader.id=pinned.attempt_id AND reader.state IN {ACTIVE}
            WHERE reader.work_item_id IN target
        ) SELECT id FROM affected ORDER BY id""",
            (json_values(changed_pins),),
        )
    ]
    return cancelled + _apply_invalidation(
        db,
        affected,
        now=now,
        reason="A resolved upstream artifact is missing, stale, or changed generation",
        error_type="UpstreamStale",
    )


def check_completion(db: sqlite3.Connection, work_item: dict, attempt_id: int) -> None:
    """Reject late completion under the publication lock if its inputs or contract changed."""
    row = db.execute(
        """SELECT a.state, a.work_item_id, a.revision_fingerprint AS started_revision,
                       i.revision_fingerprint, i.artifact_fingerprint, i.current_generation
                       FROM attempts a JOIN work_items i ON i.id=a.work_item_id WHERE a.id=?""",
        (attempt_id,),
    ).fetchone()
    if (
        row is None
        or row["state"] != "running"
        or row["work_item_id"] != work_item["id"]
        or row["started_revision"] != row["revision_fingerprint"]
        or row["artifact_fingerprint"] != work_item["artifact_fingerprint"]
        or row["current_generation"] != work_item["current_generation"]
    ):
        raise AttemptInvalidated(
            "Attempt or work item changed before completion could be published"
        )
    if db.execute(
        """SELECT 1 FROM attempt_dependencies pinned
        JOIN work_items upstream ON upstream.id=pinned.upstream_work_item_id
        WHERE pinned.attempt_id=? AND (upstream.artifact_state!='fresh' OR upstream.current_generation!=pinned.generation)
        LIMIT 1""",
        (attempt_id,),
    ).fetchone():
        raise AttemptInvalidated(
            "A captured upstream artifact changed before completion could be published"
        )
    if db.execute(
        """SELECT 1 FROM artifact_mutations WHERE work_item_id=? OR work_item_id IN (
        SELECT upstream_work_item_id FROM attempt_dependencies WHERE attempt_id=?) LIMIT 1""",
        (work_item["id"], attempt_id),
    ).fetchone():
        raise AttemptInvalidated("An input or output is reserved for replacement")
