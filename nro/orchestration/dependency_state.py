"""Apply invalidation to resolved instance dependencies, independent of their owners."""

import sqlite3
from collections.abc import Iterable

ACTIVE = "('queued','running','cancel_requested')"
WRITE_READY = f"""NOT EXISTS (
    SELECT 1 FROM attempt_dependencies pinned JOIN attempts reader ON reader.id=pinned.attempt_id
    WHERE pinned.upstream_instance_id=t.id AND reader.state IN {ACTIVE}
) AND NOT EXISTS (
    WITH RECURSIVE ancestors(id) AS (
        SELECT t.id UNION SELECT edge.upstream_instance_id FROM instance_dependencies edge
        JOIN ancestors parent ON edge.instance_id=parent.id
    ) SELECT 1 FROM ancestors JOIN artifact_mutations mutation ON mutation.instance_id=ancestors.id
)"""

EDGES = f"""SELECT instance_id, upstream_instance_id FROM instance_dependencies
UNION SELECT reader.instance_id, pinned.upstream_instance_id
FROM attempt_dependencies pinned JOIN attempts reader ON reader.id=pinned.attempt_id
WHERE reader.state IN {ACTIVE}"""


class AttemptInvalidated(RuntimeError):
    """Completion was superseded while the attempt was computing or being validated."""


def capture_inputs(db: sqlite3.Connection, attempt_id: int, instance_id: int) -> None:
    """Pin all resolved ancestors and generations under the same lock as claiming work."""
    db.execute(
        """WITH RECURSIVE ancestors(id) AS (
        SELECT upstream_instance_id FROM instance_dependencies WHERE instance_id=?
        UNION SELECT edge.upstream_instance_id FROM instance_dependencies edge
              JOIN ancestors parent ON edge.instance_id=parent.id
    ) INSERT INTO attempt_dependencies(attempt_id, upstream_instance_id, generation)
      SELECT ?, upstream.id, upstream.current_generation
      FROM ancestors JOIN instances upstream ON upstream.id=ancestors.id""",
        (instance_id, attempt_id),
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
    their more specific reasons. Set include_roots when the supplied instances
    themselves need invalidation. Cancellation is not confirmation of shutdown.
    """
    roots = tuple(sorted(set(roots)))
    if not roots:
        return []
    placeholders = ",".join("?" for _ in roots)
    descendants = tuple(
        row[0]
        for row in db.execute(
            f"""WITH RECURSIVE
        edges(instance_id, upstream_instance_id) AS ({EDGES}),
        affected(id) AS (
            SELECT instance_id FROM edges WHERE upstream_instance_id IN ({placeholders})
            UNION SELECT edge.instance_id FROM edges edge JOIN affected parent ON edge.upstream_instance_id=parent.id
        ) SELECT id FROM affected""",
            roots,
        )
    )
    if include_roots:
        descendants = tuple(sorted(set(descendants).union(roots)))
    if not descendants:
        return []
    placeholders = ",".join("?" for _ in descendants)
    db.execute(
        f"""UPDATE instances SET artifact_state='stale', artifact_reason=?, updated_at=?
                   WHERE id IN ({placeholders}) AND artifact_state='fresh' """,
        (reason, now, *descendants),
    )
    rows = [
        dict(row)
        for row in db.execute(
            f"""SELECT a.id AS attempt_id, a.instance_id, a.worker_id,
        i.instance_key, i.module FROM attempts a JOIN instances i ON i.id=a.instance_id
        WHERE a.instance_id IN ({placeholders}) AND a.state IN ('queued','running')""",
            descendants,
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
        """UPDATE instances SET artifact_state='stale', artifact_reason='Outputs reserved for mutation',
                  updated_at=? WHERE id IN (SELECT instance_id FROM artifact_mutations) AND artifact_state='fresh'""",
        (now,),
    )
    roots = {row[0] for row in db.execute("SELECT id FROM instances WHERE artifact_state!='fresh'")}
    changed = [
        row[0]
        for row in db.execute("""SELECT edge.instance_id
        FROM instance_dependencies edge JOIN instances upstream ON upstream.id=edge.upstream_instance_id
        JOIN instances consumer ON consumer.id=edge.instance_id
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
    roots.update(
        row[0]
        for row in db.execute(f"""SELECT pinned.upstream_instance_id
        FROM attempt_dependencies pinned JOIN instances upstream ON upstream.id=pinned.upstream_instance_id
        JOIN attempts reader ON reader.id=pinned.attempt_id
        WHERE reader.state IN {ACTIVE} AND pinned.generation!=upstream.current_generation""")
    )
    return cancelled + invalidate(
        db,
        roots,
        now=now,
        reason="A resolved upstream artifact is missing, stale, or changed generation",
    )


def check_completion(db: sqlite3.Connection, instance: dict, attempt_id: int) -> None:
    """Reject late completion under the publication lock if its inputs or contract changed."""
    row = db.execute(
        """SELECT a.state, a.instance_id, a.revision_fingerprint AS started_revision,
                       i.revision_fingerprint, i.artifact_fingerprint, i.current_generation
                       FROM attempts a JOIN instances i ON i.id=a.instance_id WHERE a.id=?""",
        (attempt_id,),
    ).fetchone()
    if (
        row is None
        or row["state"] != "running"
        or row["instance_id"] != instance["id"]
        or row["started_revision"] != row["revision_fingerprint"]
        or row["artifact_fingerprint"] != instance["artifact_fingerprint"]
        or row["current_generation"] != instance["current_generation"]
    ):
        raise AttemptInvalidated("Attempt or instance changed before completion could be published")
    if db.execute(
        """SELECT 1 FROM attempt_dependencies pinned
        JOIN instances upstream ON upstream.id=pinned.upstream_instance_id
        WHERE pinned.attempt_id=? AND (upstream.artifact_state!='fresh' OR upstream.current_generation!=pinned.generation)
        LIMIT 1""",
        (attempt_id,),
    ).fetchone():
        raise AttemptInvalidated(
            "A captured upstream artifact changed before completion could be published"
        )
    if db.execute(
        """SELECT 1 FROM artifact_mutations WHERE instance_id=? OR instance_id IN (
        SELECT upstream_instance_id FROM attempt_dependencies WHERE attempt_id=?) LIMIT 1""",
        (instance["id"], attempt_id),
    ).fetchone():
        raise AttemptInvalidated("An input or output is reserved for replacement")
