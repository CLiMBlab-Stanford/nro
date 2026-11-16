"""The bounded synchronize walk matches a full descendant walk on random graphs."""

import random
import sqlite3

import pytest

from nro.orchestration import dependency_state

SCHEMA = """
CREATE TABLE work_items(
    id INTEGER PRIMARY KEY, work_item_key TEXT, module TEXT, artifact_state TEXT,
    artifact_reason TEXT, current_generation INTEGER, updated_at TEXT
);
CREATE TABLE work_item_dependencies(
    work_item_id INTEGER, upstream_work_item_id INTEGER, required_generation INTEGER
);
CREATE TABLE attempts(
    id INTEGER PRIMARY KEY, work_item_id INTEGER, worker_id TEXT, state TEXT,
    error_type TEXT, error_message TEXT, completed_at TEXT
);
CREATE TABLE attempt_dependencies(attempt_id INTEGER, upstream_work_item_id INTEGER, generation INTEGER);
CREATE TABLE artifact_mutations(work_item_id INTEGER);
"""


def _reference_synchronize(db: sqlite3.Connection, *, now: str) -> list[dict]:
    """The previous implementation, which walked every non-fresh root's descendants."""
    db.execute(
        """UPDATE work_items SET artifact_state='stale', artifact_reason='Outputs reserved for mutation',
                  updated_at=? WHERE id IN (SELECT work_item_id FROM artifact_mutations) AND artifact_state='fresh'""",
        (now,),
    )
    roots = {
        row[0] for row in db.execute("SELECT id FROM work_items WHERE artifact_state!='fresh'")
    }
    changed = [
        row[0]
        for row in db.execute("""SELECT edge.work_item_id
        FROM work_item_dependencies edge JOIN work_items upstream ON upstream.id=edge.upstream_work_item_id
        JOIN work_items consumer ON consumer.id=edge.work_item_id
        WHERE consumer.artifact_state='fresh' AND edge.required_generation IS NOT NULL
              AND edge.required_generation!=upstream.current_generation""")
    ]
    cancelled = dependency_state.invalidate(
        db,
        changed,
        now=now,
        include_roots=True,
        reason="A completed artifact used an earlier upstream generation",
    )
    roots.update(
        row[0]
        for row in db.execute(f"""SELECT pinned.upstream_work_item_id
        FROM attempt_dependencies pinned JOIN work_items upstream ON upstream.id=pinned.upstream_work_item_id
        JOIN attempts reader ON reader.id=pinned.attempt_id
        WHERE reader.state IN {dependency_state.ACTIVE} AND pinned.generation!=upstream.current_generation""")
    )
    return cancelled + dependency_state.invalidate(
        db,
        roots,
        now=now,
        reason="A resolved upstream artifact is missing, stale, or changed generation",
    )


def _random_registry(seed: int) -> sqlite3.Connection:
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    count = rng.randint(2, 40)
    states = ("fresh", "fresh", "fresh", "stale", "missing", "error")
    for item in range(1, count + 1):
        db.execute(
            "INSERT INTO work_items VALUES (?,?,?,?,?,?,?)",
            (item, f"key-{item}", "anat", rng.choice(states), "", rng.randint(0, 3), "t"),
        )
    for item in range(2, count + 1):
        for upstream in rng.sample(range(1, item), k=min(item - 1, rng.randint(0, 3))):
            required = rng.choice((None, rng.randint(0, 3)))
            db.execute(
                "INSERT INTO work_item_dependencies VALUES (?,?,?)", (item, upstream, required)
            )
    attempt = 0
    for item in range(1, count + 1):
        for _ in range(rng.randint(0, 2)):
            attempt += 1
            state = rng.choice(("queued", "running", "cancel_requested", "success", "error"))
            db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,NULL,NULL,NULL)",
                (attempt, item, f"worker-{attempt}", state),
            )
            for upstream in rng.sample(range(1, count + 1), k=rng.randint(0, 3)):
                if upstream != item:
                    db.execute(
                        "INSERT INTO attempt_dependencies VALUES (?,?,?)",
                        (attempt, upstream, rng.randint(0, 3)),
                    )
    for item in rng.sample(range(1, count + 1), k=rng.randint(0, 2)):
        db.execute("INSERT INTO artifact_mutations VALUES (?)", (item,))
    return db


def _state(db: sqlite3.Connection) -> tuple:
    items = db.execute(
        "SELECT id, artifact_state, artifact_reason, updated_at FROM work_items ORDER BY id"
    ).fetchall()
    attempts = db.execute(
        "SELECT id, state, error_type, error_message FROM attempts ORDER BY id"
    ).fetchall()
    return [tuple(row) for row in items], [tuple(row) for row in attempts]


@pytest.mark.parametrize("seed", range(400))
def test_bounded_synchronize_matches_full_descendant_walk(seed):
    expected_db = _random_registry(seed)
    actual_db = _random_registry(seed)

    expected = _reference_synchronize(expected_db, now="now")
    actual = dependency_state.synchronize(actual_db, now="now")

    assert _state(actual_db) == _state(expected_db)
    key = lambda row: row["attempt_id"]  # noqa: E731
    assert sorted(actual, key=key) == sorted(expected, key=key)


def test_bounded_synchronize_is_a_fixpoint():
    for seed in range(100):
        db = _random_registry(seed)
        dependency_state.synchronize(db, now="first")
        before = _state(db)
        assert dependency_state.synchronize(db, now="second") == []
        assert _state(db) == before
