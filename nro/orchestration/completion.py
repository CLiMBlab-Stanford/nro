"""Publish completion certificates for successful work-item attempts.

Publication inventories the public products and available private intermediates,
checks the attempt against its registered work-item contract, and advances the
artifact generation atomically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import yaml

from nro.engine.io import atomic_write_json
from nro.orchestration import dependency_state
from nro.orchestration.manifests import (
    MANIFEST_VERSION,
    _completion_output_inventory,
    _is_orchestration_control_artifact,
    _read_manifest,
    file_record,
    inventory,
)
from nro.orchestration.ownership import write_work_item_ownership
from nro.orchestration.registry import Registry, ensure_shared_directory, utcnow


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
    value = _read_manifest(ledger)
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
                if resolved.is_file() and not _is_orchestration_control_artifact(
                    resolved, registry
                ):
                    paths.add(resolved)
    return inventory(paths)


def record_completion(
    registry: Registry,
    *,
    work_item_id: int,
    attempt_id: int,
    outputs: Iterable[str | Path],
) -> dict:
    """Write a completion manifest last, then advance the registry generation."""
    with registry.connection() as db:
        existing = db.execute(
            "SELECT manifest_path,artifact_state FROM work_items WHERE id=?", (work_item_id,)
        ).fetchone()
    if existing is not None and existing["artifact_state"] == "fresh":
        path = Path(existing["manifest_path"])
        if path.is_file():
            manifest = json.loads(path.read_text())
            if int(manifest.get("attempt_id", -1)) == attempt_id:
                return manifest
    with registry.connection() as db:
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
        execution = db.execute(
            "SELECT provenance_json,command_json FROM attempt_execution WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        parents = [
            dict(row)
            for row in db.execute(
                """
                SELECT t.id, t.current_generation, t.manifest_path
                FROM work_item_dependencies td JOIN work_items t ON t.id=td.upstream_work_item_id
                WHERE td.work_item_id=? ORDER BY t.id
                """,
                (work_item_id,),
            )
        ]
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
    runtime_config = file_record(work_item["runtime_config_path"])
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "work_item_id": work_item_id,
        "work_item_key": work_item["work_item_key"],
        "module": work_item["module"],
        "project": work_item["project"],
        "participant": work_item["participant"],
        "entities": json.loads(work_item["entities_json"]),
        "module_lineage_id": work_item["module_lineage_id"],
        "revision_fingerprint": work_item["revision_fingerprint"],
        "artifact_contract": json.loads(work_item["artifact_contract_json"]),
        "artifact_fingerprint": work_item["artifact_fingerprint"],
        "configuration": {
            "id": work_item["config_id"],
            "fingerprint": work_item["config_fingerprint"],
            "lineage_fingerprint": work_item["lineage_fingerprint"],
            "resolved": yaml.safe_load(work_item["resolved_yaml"]) or {},
        },
        "runtime_config": runtime_config,
        "software": {
            "name": "nro",
            "python": sys.version,
            "executable": sys.executable,
            "command": json.loads(work_item["command_json"]),
        },
        "generation": generation,
        "attempt_id": attempt_id,
        "completed_at": utcnow(),
        "inputs": input_records,
        "upstream": [
            {
                "work_item_id": int(parent["id"]),
                "generation": int(parent["current_generation"]),
                "manifest": parent["manifest_path"],
            }
            for parent in parents
        ],
        "public_outputs": output_records,
        "private_artifacts": private_records,
    }
    path = Path(work_item["manifest_path"])
    if execution is not None:
        manifest["implementation"] = json.loads(execution["provenance_json"])
        manifest["software"]["command"] = json.loads(execution["command_json"])
    ensure_shared_directory(path.parent)
    write_work_item_ownership(registry, work_item_id, attempt_id=attempt_id)
    with registry.connection(write=True) as db:
        dependency_state.check_completion(db, work_item, attempt_id)
        atomic_write_json(path, manifest, sort_keys=True, mode=0o664, durable=True)
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
        for item in input_records:
            db.execute(
                """INSERT INTO artifacts(work_item_id, attempt_id, direction, path, size, mtime_ns,
                   digest_algorithm, digest, metadata_json) VALUES (?, ?, 'input', ?, ?, ?, ?, ?, '{}')""",
                (
                    work_item_id,
                    attempt_id,
                    item["path"],
                    item["size"],
                    item["mtime_ns"],
                    "sha256" if "sha256" in item else None,
                    item.get("sha256"),
                ),
            )
        for item in output_records:
            db.execute(
                """INSERT INTO artifacts(work_item_id, attempt_id, direction, path, size, mtime_ns,
                   digest_algorithm, digest, metadata_json) VALUES (?, ?, 'output', ?, ?, ?, ?, ?, '{}')""",
                (
                    work_item_id,
                    attempt_id,
                    item["path"],
                    item["size"],
                    item["mtime_ns"],
                    "sha256" if "sha256" in item else None,
                    item.get("sha256"),
                ),
            )
    return manifest
