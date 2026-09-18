"""Commit successful work-item evidence to the coordinator database.

Publication inventories the public products and available private intermediates,
checks the attempt against its registered work-item contract, and advances the
artifact generation atomically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from nro.orchestration import dependency_state
from nro.orchestration.artifact_records import inventory, is_control_artifact, read_json_mapping
from nro.orchestration.completion_records import completion_record
from nro.orchestration.ownership import write_work_item_ownership
from nro.orchestration.registry import Registry, utcnow

COMPLETION_VISIBILITY_TIMEOUT = 30.0
COMPLETION_VISIBILITY_POLL_INTERVAL = 0.25


def _completion_output_inventory(paths: Iterable[str | Path]) -> list[dict]:
    """Inventory completed outputs after bounded visibility retries.

    A worker publishes files atomically before asking the coordinator to certify
    them. Shared filesystems can expose a replaced directory entry to another
    node after a short delay, so the authoritative coordinator read retries only
    paths that are still absent. Visible non-files fail immediately.
    """
    resolved = tuple(sorted({Path(path).expanduser().resolve() for path in paths}))
    deadline = time.monotonic() + COMPLETION_VISIBILITY_TIMEOUT
    while True:
        try:
            return inventory(resolved)
        except ValueError:
            if any(path.exists() and not path.is_file() for path in resolved):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(COMPLETION_VISIBILITY_POLL_INTERVAL)


def _private_step_artifacts(
    registry: Registry,
    *,
    attempt_id: int,
    output_root: Path,
) -> list[dict]:
    """Inventory absolute, persistent step outputs outside the derivative root.

    Runner ledgers are intentionally best-effort diagnostics, so a malformed
    or relative output entry is simply not eligible for work-item validation.
    Module-level staleness still owns all such details whenever the module is
    invoked.
    """
    with registry.connection() as db:
        row = db.execute("SELECT log_path FROM attempts WHERE id=?", (attempt_id,)).fetchone()
    if row is None or not row["log_path"]:
        return []
    ledger = Path(str(row["log_path"])).parent / "current-steps.json"
    value = read_json_mapping(ledger)
    if value is None:
        return []
    public_root = output_root.resolve()
    paths: set[Path] = set()
    for step in value.values():
        if not isinstance(step, dict):
            continue
        cwd = step.get("cwd")
        for item in step.get("outputs", []):
            if not isinstance(item, str) or not item.strip():
                continue
            path = Path(item).expanduser()
            if not path.is_absolute():
                # Relative output paths are only meaningful when the ledger
                # recorded a concrete working directory.
                if not cwd:
                    continue
                path = Path(str(cwd)).expanduser() / path
            try:
                resolved = path.resolve()
                resolved.relative_to(public_root)
            except ValueError:
                if resolved.is_file() and not is_control_artifact(resolved, registry.paths.control):
                    paths.add(resolved)
    return inventory(paths)


def record_completion(
    registry: Registry,
    *,
    work_item_id: int,
    attempt_id: int,
    outputs: Iterable[str | Path],
) -> dict:
    """Validate filesystem evidence and commit one completed generation atomically."""
    with registry.connection() as db:
        existing = completion_record(db, work_item_id)
        if existing is not None and int(existing["attempt_id"]) == attempt_id:
            return existing
        work_item = dict(
            db.execute(
                """SELECT t.*, ci.config_id, ci.config_fingerprint,
                          ci.lineage_fingerprint, ci.resolved_yaml
                   FROM work_items t JOIN module_lineages ci ON ci.id=t.module_lineage_id
                   WHERE t.id=?""",
                (work_item_id,),
            ).fetchone()
        )
        dependency_state.check_completion(db, work_item, attempt_id)
        parents = [
            dict(row)
            for row in db.execute(
                """
                SELECT t.id, t.current_generation
                FROM work_item_dependencies td JOIN work_items t ON t.id=td.upstream_work_item_id
                WHERE td.work_item_id=? ORDER BY t.id
                """,
                (work_item_id,),
            )
        ]
        execution = db.execute(
            "SELECT provenance_json,command_json FROM attempt_execution WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
    output_records = _completion_output_inventory(outputs)
    if not output_records:
        raise RuntimeError(
            f"Work item produced no discoverable outputs under {work_item['output_root']}"
        )
    private_records = _private_step_artifacts(
        registry,
        attempt_id=attempt_id,
        output_root=Path(work_item["output_root"]),
    )
    input_records = inventory(json.loads(work_item["input_paths_json"]))
    generation = int(work_item["current_generation"]) + 1
    completed_at = utcnow()
    write_work_item_ownership(registry, work_item_id, attempt_id=attempt_id)
    with registry.connection(write=True) as db:
        dependency_state.check_completion(db, work_item, attempt_id)
        db.execute("DELETE FROM completions WHERE work_item_id=?", (work_item_id,))
        db.execute("DELETE FROM artifacts WHERE work_item_id=?", (work_item_id,))
        db.execute(
            """INSERT INTO completions(
                   work_item_id,attempt_id,generation,completed_at,revision_fingerprint,
                   artifact_contract_json,artifact_fingerprint,config_id,config_fingerprint,
                   lineage_fingerprint,resolved_yaml,provenance_json,command_json
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                work_item_id,
                attempt_id,
                generation,
                completed_at,
                work_item["revision_fingerprint"],
                work_item["artifact_contract_json"],
                work_item["artifact_fingerprint"],
                work_item["config_id"],
                work_item["config_fingerprint"],
                work_item["lineage_fingerprint"],
                work_item["resolved_yaml"],
                execution["provenance_json"] if execution is not None else "{}",
                execution["command_json"] if execution is not None else work_item["command_json"],
            ),
        )
        db.execute(
            """
            UPDATE work_items SET artifact_state='fresh', artifact_reason='Completed successfully',
                current_generation=?, updated_at=? WHERE id=?
            """,
            (generation, utcnow(), work_item_id),
        )
        db.executemany(
            """UPDATE work_item_dependencies SET required_generation=?
                          WHERE work_item_id=? AND upstream_work_item_id=?""",
            [(parent["current_generation"], work_item_id, parent["id"]) for parent in parents],
        )
        for direction, records in (
            ("input", input_records),
            ("output", output_records),
            ("private", private_records),
        ):
            for item in records:
                db.execute(
                    """INSERT INTO artifacts(
                           work_item_id,attempt_id,direction,path,size,mtime_ns,
                           digest_algorithm,digest,metadata_json
                       ) VALUES (?,?,?,?,?,?,?,?, '{}')""",
                    (
                        work_item_id,
                        attempt_id,
                        direction,
                        item["path"],
                        item["size"],
                        item["mtime_ns"],
                        "sha256" if "sha256" in item else None,
                        item.get("sha256"),
                    ),
                )
        record = completion_record(db, work_item_id)
        if record is None:
            raise RuntimeError("Committed completion evidence could not be read back")
        return record
