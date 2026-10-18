"""Validate cached artifact evidence without reading file contents."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from nro.orchestration.registry import utcnow

_QUERY_CHUNK_SIZE = 500
_STAT_WORKERS = 32


def _chunks(values: set[int]) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(sorted(values))
    return tuple(
        ordered[offset : offset + _QUERY_CHUNK_SIZE]
        for offset in range(0, len(ordered), _QUERY_CHUNK_SIZE)
    )


def _stat(path: str) -> tuple[int, int] | None:
    try:
        value = os.stat(path)
    except OSError:
        return None
    return value.st_size, value.st_mtime_ns


def assess_cached_artifacts(registry, work_item_ids: Iterable[int]) -> set[int]:
    """Return reusable artifacts after a parallel size and mtime assessment.

    Candidate contracts have already been matched against the new scientific
    graph. This pass checks cached completion evidence, direct inputs, public
    outputs, upstream generations, and cached upstream states. Private WORK
    files do not determine whether an already published artifact is reusable.
    """
    selected = {int(value) for value in work_item_ids}
    if not selected:
        return set()
    rows: dict[int, dict] = {}
    dependencies: dict[int, list[dict]] = {}
    artifacts: dict[int, list[dict]] = {}
    completed: set[int] = set()
    with registry.connection() as database:
        for chunk in _chunks(selected):
            placeholders = ",".join("?" for _ in chunk)
            for row in database.execute(
                f"""SELECT id,artifact_state,artifact_reason,current_generation
                    FROM work_items WHERE id IN ({placeholders})""",
                chunk,
            ):
                rows[int(row["id"])] = dict(row)
            for row in database.execute(
                f"""SELECT d.work_item_id,d.upstream_work_item_id,d.required_generation,
                           parent.artifact_state AS upstream_state,
                           parent.current_generation AS upstream_generation
                    FROM work_item_dependencies d
                    JOIN work_items parent ON parent.id=d.upstream_work_item_id
                    WHERE d.work_item_id IN ({placeholders})
                    ORDER BY d.work_item_id,d.upstream_work_item_id,d.role""",
                chunk,
            ):
                dependencies.setdefault(int(row["work_item_id"]), []).append(dict(row))
            completed.update(
                int(row[0])
                for row in database.execute(
                    f"SELECT work_item_id FROM completions WHERE work_item_id IN ({placeholders})",
                    chunk,
                )
            )
            for row in database.execute(
                f"""SELECT a.work_item_id,a.direction,a.path,a.size,a.mtime_ns
                    FROM artifacts a
                    JOIN completions c ON c.work_item_id=a.work_item_id
                                      AND a.attempt_id IS c.attempt_id
                    WHERE a.work_item_id IN ({placeholders})
                      AND a.direction IN ('input','output')
                    ORDER BY a.work_item_id,a.direction,a.path""",
                chunk,
            ):
                artifacts.setdefault(int(row["work_item_id"]), []).append(dict(row))

    paths = tuple(sorted({str(item["path"]) for values in artifacts.values() for item in values}))
    with ThreadPoolExecutor(max_workers=min(_STAT_WORKERS, max(1, len(paths)))) as pool:
        current = dict(zip(paths, pool.map(_stat, paths), strict=True))

    states: dict[int, tuple[str, str]] = {}
    remaining = set(rows)
    while remaining:
        progressed = False
        for work_item_id in tuple(remaining):
            parents = dependencies.get(work_item_id, ())
            pending_parent = next(
                (
                    int(parent["upstream_work_item_id"])
                    for parent in parents
                    if int(parent["upstream_work_item_id"]) in remaining
                ),
                None,
            )
            if pending_parent is not None:
                continue
            row = rows[work_item_id]
            if row["artifact_state"] != "fresh":
                state = (str(row["artifact_state"]), str(row["artifact_reason"]))
            elif work_item_id not in completed:
                state = ("stale", "Cached artifact has no completion evidence")
            elif not any(item["direction"] == "output" for item in artifacts.get(work_item_id, ())):
                state = ("stale", "Cached completion records no public outputs")
            else:
                changed = next(
                    (
                        item
                        for item in artifacts.get(work_item_id, ())
                        if item["size"] is None
                        or item["mtime_ns"] is None
                        or current[str(item["path"])] != (int(item["size"]), int(item["mtime_ns"]))
                    ),
                    None,
                )
                if changed is not None:
                    state = (
                        "stale",
                        f"{str(changed['direction']).capitalize()} artifact size or modification "
                        f"time changed: {changed['path']}",
                    )
                else:
                    upstream_error = None
                    for parent in parents:
                        parent_id = int(parent["upstream_work_item_id"])
                        parent_state = states.get(
                            parent_id,
                            (str(parent["upstream_state"]), "Cached upstream state"),
                        )[0]
                        required = parent["required_generation"]
                        if parent_state != "fresh":
                            upstream_error = "An upstream derivative is missing or stale"
                            break
                        if required is None or int(required) != int(parent["upstream_generation"]):
                            upstream_error = "Upstream derivative generation changed"
                            break
                    state = (
                        ("stale", upstream_error)
                        if upstream_error is not None
                        else ("fresh", "Cached artifact metadata validates")
                    )
            states[work_item_id] = state
            remaining.remove(work_item_id)
            progressed = True
        if not progressed:
            for work_item_id in remaining:
                states[work_item_id] = ("stale", "Work-item dependency cycle detected")
            break

    stale = {
        work_item_id: reason
        for work_item_id, (state, reason) in states.items()
        if rows[work_item_id]["artifact_state"] == "fresh" and state != "fresh"
    }
    if stale:
        now = utcnow()
        with registry.connection(write=True) as database:
            for work_item_id, reason in stale.items():
                database.execute(
                    """UPDATE work_items SET artifact_state='stale',artifact_reason=?,updated_at=?
                       WHERE id=? AND artifact_state='fresh' AND current_generation=?""",
                    (reason, now, work_item_id, int(rows[work_item_id]["current_generation"])),
                )
    return {work_item_id for work_item_id, (state, _reason) in states.items() if state == "fresh"}
