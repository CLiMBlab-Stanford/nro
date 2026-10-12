"""Re-attest unchanged artifacts after an erroneous upstream rebuild."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from nro.engine.io import atomic_write_json, sync_directory
from nro.orchestration import dependency_state
from nro.orchestration.artifact_records import inventory
from nro.orchestration.assessment import capture_assessment
from nro.orchestration.branch_purge import snapshot as branch_snapshot
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.manifests import evaluate_assessment
from nro.orchestration.registry import ensure_shared_directory, utcnow

_RECOVERABLE_STALE_REASONS = frozenset(
    {
        "An upstream derivative is missing or stale",
        "Upstream derivative set or generation changed",
    }
)


def _matches(row: dict, selection: dict) -> bool:
    from nro.engine.cli import matches_module_lineage
    from nro.engine.cli import matches_work_item_selectors as matches_selectors

    workflows = set(selection.get("workflows", ()))
    return not (
        selection.get("projects")
        and row["project"] not in selection["projects"]
        or selection.get("participants")
        and row["participant"] not in selection["participants"]
        or selection.get("modules")
        and row["module"] not in selection["modules"]
        or workflows
        and not workflows.intersection(str(row.get("workflow_ids") or "").split(","))
        or not matches_module_lineage(
            str(row["module"]),
            str(row.get("directory_label") or ""),
            tuple(selection.get("lineages", ())),
        )
        or not matches_selectors(json.loads(row["entities_json"]), selection.get("entities", {}))
    )


def _same_file(record: dict) -> tuple[bool, str]:
    path = Path(str(record["path"]))
    if not path.is_file():
        return False, f"missing artifact: {path}"
    stat = path.stat()
    if stat.st_size != int(record["size"]):
        return False, f"artifact size changed: {path}"
    digest = record.get("sha256")
    if digest is not None and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return False, f"artifact content changed: {path}"
    if digest is None and stat.st_mtime_ns != int(record["mtime_ns"]):
        return False, f"artifact modification time changed: {path}"
    return True, ""


def _topological(ids: set[int], dependencies: tuple[dict, ...]) -> tuple[int, ...]:
    parents = {
        work_item_id: {
            int(edge["upstream_work_item_id"])
            for edge in dependencies
            if int(edge["work_item_id"]) == work_item_id
            and int(edge["upstream_work_item_id"]) in ids
        }
        for work_item_id in ids
    }
    ordered: list[int] = []
    remaining = set(ids)
    while remaining:
        ready = sorted(key for key in remaining if not parents[key].intersection(remaining))
        if not ready:
            raise ValueError("Selected work items contain a dependency cycle")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return tuple(ordered)


def _journal_root(registry) -> Path:
    return ControlPaths(registry.paths.control).shared / "freshening"


def _remove_journal(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.parent.is_dir():
        sync_directory(path.parent)


def _recover_journals(registry) -> None:
    """Resolve interrupted timestamp repairs before starting another operation."""
    root = _journal_root(registry)
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid freshen recovery journal: {path}") from error
        if record.get("version") != 1 or not isinstance(record.get("items"), list):
            raise RuntimeError(f"Unsupported freshen recovery journal: {path}")
        token = str(record.get("token") or "")
        if not token:
            raise RuntimeError(f"Invalid freshen recovery token: {path}")
        for item in record["items"]:
            work_item_id = int(item["id"])
            artifacts = item["artifacts"]
            artifact_ids = tuple(int(artifact["id"]) for artifact in artifacts)
            with registry.connection() as db:
                placeholders = ",".join("?" for _ in artifact_ids)
                recorded = {
                    int(row["id"]): int(row["mtime_ns"])
                    for row in db.execute(
                        f"SELECT id,mtime_ns FROM artifacts WHERE id IN ({placeholders})",
                        artifact_ids,
                    )
                }
                state = db.execute(
                    "SELECT artifact_state FROM work_items WHERE id=?", (work_item_id,)
                ).fetchone()
                mutation = db.execute(
                    "SELECT token FROM artifact_mutations WHERE work_item_id=?",
                    (work_item_id,),
                ).fetchone()
            targets = {
                int(artifact["id"]): int(artifact["target_mtime_ns"]) for artifact in artifacts
            }
            committed = (
                state is not None
                and state["artifact_state"] == "fresh"
                and recorded == targets
                and mutation is None
            )
            if committed:
                continue
            if mutation is not None and mutation["token"] != token:
                raise RuntimeError(
                    f"Freshen recovery conflicts with another artifact mutation: {work_item_id}"
                )
            if mutation is None and any(
                recorded.get(int(artifact["id"])) != int(artifact["old_mtime_ns"])
                for artifact in artifacts
            ):
                raise RuntimeError(f"Freshen recovery is ambiguous for work item {work_item_id}")
            for artifact in artifacts:
                artifact_path = Path(str(artifact["path"]))
                if not artifact_path.is_file() or artifact_path.stat().st_size != int(
                    artifact["size"]
                ):
                    raise RuntimeError(
                        f"Freshen recovery cannot restore a changed artifact: {artifact_path}"
                    )
                stat = artifact_path.stat()
                os.utime(
                    artifact_path,
                    ns=(stat.st_atime_ns, int(artifact["old_mtime_ns"])),
                )
            with registry.connection(write=True) as db:
                db.executemany(
                    "UPDATE artifacts SET mtime_ns=? WHERE id=?",
                    [
                        (int(artifact["old_mtime_ns"]), int(artifact["id"]))
                        for artifact in artifacts
                    ],
                )
                db.execute(
                    "DELETE FROM artifact_mutations WHERE work_item_id=? AND token=?",
                    (work_item_id, token),
                )
        _remove_journal(path)


def _artifact_rows(db, work_item_id: int) -> list[dict]:
    return [
        dict(row)
        for row in db.execute(
            """SELECT artifact.id,artifact.direction,artifact.path,artifact.size,
                      artifact.mtime_ns,artifact.digest_algorithm,artifact.digest
               FROM artifacts artifact
               JOIN completions completion ON completion.work_item_id=artifact.work_item_id
                    AND artifact.attempt_id IS completion.attempt_id
               WHERE artifact.work_item_id=? AND artifact.direction IN ('output','private')
               ORDER BY artifact.direction,artifact.path""",
            (work_item_id,),
        )
    ]


def _prepare_journal(registry, chosen: list[dict]) -> tuple[Path, dict]:
    token = uuid.uuid4().hex
    target_public_max: dict[int, int] = {}
    plans = []
    with registry.connection() as db:
        for sequence, item in enumerate(chosen):
            work_item_id = int(item["id"])
            row = db.execute(
                "SELECT input_paths_json FROM work_items WHERE id=?", (work_item_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Freshen work item disappeared: {work_item_id}")
            threshold = max(
                (Path(value).stat().st_mtime_ns for value in json.loads(row["input_paths_json"])),
                default=0,
            )
            parents = db.execute(
                """SELECT parent.id,artifact.path
                   FROM work_item_dependencies edge
                   JOIN work_items parent ON parent.id=edge.upstream_work_item_id
                   LEFT JOIN completions completion ON completion.work_item_id=parent.id
                   LEFT JOIN artifacts artifact ON artifact.work_item_id=parent.id
                        AND artifact.attempt_id IS completion.attempt_id
                        AND artifact.direction='output'
                   WHERE edge.work_item_id=?""",
                (work_item_id,),
            ).fetchall()
            for parent in parents:
                parent_id = int(parent["id"])
                if parent_id in target_public_max:
                    threshold = max(threshold, target_public_max[parent_id])
                elif parent["path"] and Path(str(parent["path"])).is_file():
                    threshold = max(threshold, Path(str(parent["path"])).stat().st_mtime_ns)
            rows = [row for row in _artifact_rows(db, work_item_id) if Path(row["path"]).is_file()]
            if not rows or not any(row["direction"] == "output" for row in rows):
                raise ValueError(f"Freshen work item has no current public output: {work_item_id}")
            oldest = min(int(row["mtime_ns"]) for row in rows)
            delta = max(0, threshold + 1 + sequence - oldest)
            artifacts = [
                {
                    "id": int(row["id"]),
                    "path": str(row["path"]),
                    "size": int(row["size"]),
                    "old_mtime_ns": int(row["mtime_ns"]),
                    "target_mtime_ns": int(row["mtime_ns"]) + delta,
                    "digest_algorithm": row["digest_algorithm"],
                    "digest": row["digest"],
                }
                for row in rows
            ]
            target_public_max[work_item_id] = max(
                artifact["target_mtime_ns"]
                for artifact, row in zip(artifacts, rows, strict=True)
                if row["direction"] == "output"
            )
            parent_generations = {
                int(parent["upstream_work_item_id"]): int(parent["current_generation"])
                for parent in db.execute(
                    """SELECT edge.upstream_work_item_id,parent.current_generation
                       FROM work_item_dependencies edge JOIN work_items parent
                         ON parent.id=edge.upstream_work_item_id
                       WHERE edge.work_item_id=?""",
                    (work_item_id,),
                )
            }
            plans.append(
                {
                    "id": work_item_id,
                    "artifacts": artifacts,
                    "parent_generations": parent_generations,
                }
            )
    record = {"version": 1, "token": token, "state": "prepared", "items": plans}
    root = ensure_shared_directory(_journal_root(registry))
    path = root / f"{token}.json"
    atomic_write_json(path, record, sort_keys=True, mode=0o664, durable=True)
    return path, record


def preview(registry, *, checkout: Path, site_values: dict, selection: dict) -> dict:
    """Return unchanged completions that can adopt current upstream generations."""
    _recover_journals(registry)
    owned = branch_snapshot(registry, checkout=checkout, site_values=site_values)
    selected_rows = [row for row in owned["rows"] if _matches(row, selection)]
    selected = {int(row["id"]) for row in selected_rows}
    if not selected:
        return {"branch": owned["branch"], "items": [], "freshen": 0, "retained": 0, "rejected": 0}

    snapshot = capture_assessment(registry, work_item_ids=selected)
    assessment = evaluate_assessment(snapshot)
    updates = {int(update["id"]): update for update in assessment.updates}
    rows = {int(row["id"]): row for row in snapshot.work_items}
    completions = {
        int(key): value for key, value in completion_records_from_snapshot(snapshot).items()
    }
    parents: dict[int, set[int]] = {key: set() for key in rows}
    for edge in snapshot.dependencies:
        parents[int(edge["work_item_id"])].add(int(edge["upstream_work_item_id"]))
    active = {int(row["work_item_id"]) for row in snapshot.attempts}
    mutating = {int(row["work_item_id"]) for row in snapshot.mutations}
    owned_by_id = {int(row["id"]): row for row in selected_rows}
    available = {
        key
        for key, update in updates.items()
        if update["state"] == "fresh"
        and key not in selected
        and key not in active
        and key not in mutating
    }
    items = []
    order = _topological(selected, snapshot.dependencies)
    for work_item_id in order:
        row = rows[work_item_id]
        update = updates[work_item_id]
        completion = completions.get(work_item_id)
        action = "reject"
        reason = update["reason"]
        if work_item_id in active:
            reason = "An active attempt owns this work item"
        elif work_item_id in mutating:
            reason = "An artifact mutation owns this work item"
        elif update["state"] == "fresh":
            action, reason = "retain", "Artifact already validates as fresh"
        elif update["state"] != "stale":
            reason = f"Artifact state is {update['state']}, not stale"
        elif update["reason"] not in _RECOVERABLE_STALE_REASONS:
            reason = f"Staleness is not limited to an upstream rebuild: {update['reason']}"
        elif update["contract"] is not None:
            reason = "Current scientific contract changed"
        elif update["inputs"] is not None:
            reason = "Selected direct input set changed"
        elif completion is None:
            reason = "Private completion evidence is unavailable"
        elif {int(item["work_item_id"]) for item in completion.get("upstream", ())} != parents[
            work_item_id
        ]:
            reason = "Upstream derivative set changed"
        elif not parents[work_item_id].issubset(available):
            reason = "An upstream artifact cannot be freshened safely"
        else:
            try:
                direct = inventory(json.loads(row["input_paths_json"]))
            except (OSError, ValueError) as error:
                reason = f"Direct input cannot be validated: {error}"
            else:
                if direct != completion.get("inputs", []):
                    reason = "Direct input fingerprint changed"
                else:
                    records = [
                        *completion.get("public_outputs", []),
                        *(
                            item
                            for item in completion.get("private_artifacts", [])
                            if Path(str(item.get("path", ""))).exists()
                        ),
                    ]
                    error = next(
                        (message for okay, message in map(_same_file, records) if not okay), None
                    )
                    if error is not None:
                        reason = error
                    else:
                        action = "freshen"
                        reason = "Contract, direct inputs, and artifacts are unchanged"
        if action in {"retain", "freshen"}:
            available.add(work_item_id)
        owned_row = owned_by_id[work_item_id]
        items.append(
            {
                "id": work_item_id,
                "token": owned_row["purge_token"],
                "project": row["project"],
                "participant": row["participant"],
                "module": row["module"],
                "lineage": owned_row.get("directory_label"),
                "entities": json.loads(row["entities_json"]),
                "action": action,
                "reason": reason,
            }
        )
    return {
        "branch": owned["branch"],
        "items": items,
        "freshen": sum(item["action"] == "freshen" for item in items),
        "retained": sum(item["action"] == "retain" for item in items),
        "rejected": sum(item["action"] == "reject" for item in items),
    }


def completion_records_from_snapshot(snapshot) -> dict[int, dict]:
    """Return detached completion records already captured for assessment."""
    return {int(record["work_item_id"]): record for record in snapshot.completions}


def execute(
    registry,
    *,
    checkout: Path,
    site_values: dict,
    selection: dict,
    approved: dict,
) -> dict:
    """Freshen an unchanged topological selection after revalidating its preview."""
    current = preview(registry, checkout=checkout, site_values=site_values, selection=selection)
    if current != approved:
        raise ValueError("Freshen selection or validation changed; review a new preview")
    chosen = [item for item in current["items"] if item["action"] == "freshen"]
    if not chosen:
        return current | {"executed": 0}
    journal_path, journal = _prepare_journal(registry, chosen)
    token = journal["token"]
    identifiers = tuple(int(item["id"]) for item in journal["items"])
    placeholders = ",".join("?" for _ in identifiers)
    try:
        with registry.connection(write=True) as db:
            if db.execute(
                f"SELECT 1 FROM attempts WHERE work_item_id IN ({placeholders}) "
                f"AND state IN {dependency_state.ACTIVE} LIMIT 1",
                identifiers,
            ).fetchone():
                raise RuntimeError("A selected work item became active after the preview")
            if db.execute(
                f"SELECT 1 FROM artifact_mutations WHERE work_item_id IN ({placeholders}) LIMIT 1",
                identifiers,
            ).fetchone():
                raise RuntimeError("A selected artifact became reserved after the preview")
            for item in journal["items"]:
                current_generations = {
                    int(row["upstream_work_item_id"]): int(row["current_generation"])
                    for row in db.execute(
                        """SELECT edge.upstream_work_item_id,parent.current_generation
                           FROM work_item_dependencies edge JOIN work_items parent
                             ON parent.id=edge.upstream_work_item_id
                           WHERE edge.work_item_id=?""",
                        (int(item["id"]),),
                    )
                }
                expected = {
                    int(key): int(value) for key, value in item["parent_generations"].items()
                }
                if current_generations != expected:
                    raise RuntimeError("An upstream generation changed after the freshen preview")
                for artifact in item["artifacts"]:
                    row = db.execute(
                        "SELECT mtime_ns FROM artifacts WHERE id=?", (int(artifact["id"]),)
                    ).fetchone()
                    if row is None or int(row["mtime_ns"]) != int(artifact["old_mtime_ns"]):
                        raise RuntimeError("Artifact evidence changed after the freshen preview")
            db.executemany(
                "INSERT INTO artifact_mutations(work_item_id,token) VALUES (?,?)",
                [(work_item_id, token) for work_item_id in identifiers],
            )
        journal["state"] = "reserved"
        atomic_write_json(journal_path, journal, sort_keys=True, mode=0o664, durable=True)
        for item in journal["items"]:
            work_item_id = int(item["id"])
            for artifact in item["artifacts"]:
                path = Path(str(artifact["path"]))
                stat = path.stat()
                changed = stat.st_size != int(artifact["size"]) or stat.st_mtime_ns != int(
                    artifact["old_mtime_ns"]
                )
                if (
                    not changed
                    and artifact["digest_algorithm"] == "sha256"
                    and hashlib.sha256(path.read_bytes()).hexdigest() != artifact["digest"]
                ):
                    changed = True
                if changed:
                    raise RuntimeError(f"Artifact changed while freshening: {path}")
                os.utime(path, ns=(stat.st_atime_ns, int(artifact["target_mtime_ns"])))
                artifact["target_mtime_ns"] = path.stat().st_mtime_ns
            atomic_write_json(journal_path, journal, sort_keys=True, mode=0o664, durable=True)
            now = utcnow()
            with registry.connection(write=True) as db:
                mutation = db.execute(
                    "SELECT token FROM artifact_mutations WHERE work_item_id=?",
                    (work_item_id,),
                ).fetchone()
                if mutation is None or mutation["token"] != token:
                    raise RuntimeError("Freshen lost its artifact reservation")
                current_generations = {
                    int(row["upstream_work_item_id"]): int(row["current_generation"])
                    for row in db.execute(
                        """SELECT edge.upstream_work_item_id,parent.current_generation,
                                  parent.artifact_state
                           FROM work_item_dependencies edge JOIN work_items parent
                             ON parent.id=edge.upstream_work_item_id
                           WHERE edge.work_item_id=?""",
                        (work_item_id,),
                    )
                    if row["artifact_state"] == "fresh"
                }
                expected = {
                    int(key): int(value) for key, value in item["parent_generations"].items()
                }
                if current_generations != expected:
                    raise RuntimeError("An upstream generation changed while freshening")
                db.executemany(
                    "UPDATE artifacts SET mtime_ns=? WHERE id=?",
                    [
                        (int(artifact["target_mtime_ns"]), int(artifact["id"]))
                        for artifact in item["artifacts"]
                    ],
                )
                db.execute(
                    """UPDATE work_items SET artifact_state='fresh',
                              artifact_reason='Freshened after unchanged-contract validation',
                              updated_at=? WHERE id=?""",
                    (now, work_item_id),
                )
                db.executemany(
                    """UPDATE work_item_dependencies SET required_generation=?
                       WHERE work_item_id=? AND upstream_work_item_id=?""",
                    [
                        (generation, work_item_id, parent_id)
                        for parent_id, generation in expected.items()
                    ],
                )
                db.execute(
                    "DELETE FROM artifact_mutations WHERE work_item_id=? AND token=?",
                    (work_item_id, token),
                )
                dependency_state.synchronize(db, now=now)
            item["committed"] = True
            atomic_write_json(journal_path, journal, sort_keys=True, mode=0o664, durable=True)
    except BaseException:
        _recover_journals(registry)
        raise
    _remove_journal(journal_path)
    return current | {"executed": len(chosen)}
