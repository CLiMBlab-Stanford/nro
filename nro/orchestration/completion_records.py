"""Read completed work-item evidence from the coordinator database."""

from __future__ import annotations

import json
import sqlite3

import yaml

_QUERY_CHUNK_SIZE = 500


def _chunks(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Split identifiers below SQLite's conservative parameter ceiling."""
    return tuple(
        values[offset : offset + _QUERY_CHUNK_SIZE]
        for offset in range(0, len(values), _QUERY_CHUNK_SIZE)
    )


def completion_records(
    database: sqlite3.Connection, work_item_ids: tuple[int, ...]
) -> dict[int, dict]:
    """Return available completions using a bounded number of database queries."""
    identifiers = tuple(dict.fromkeys(int(value) for value in work_item_ids))
    if not identifiers:
        return {}

    rows: dict[int, dict] = {}
    for chunk in _chunks(identifiers):
        placeholders = ",".join("?" for _ in chunk)
        for row in database.execute(
            f"""
            SELECT item.work_item_key,item.module,item.project,item.participant,
                   completion.work_item_id,
                   item.entities_json,item.module_lineage_id,
                   completion.attempt_id,completion.generation,completion.completed_at,
                   completion.revision_fingerprint,completion.artifact_contract_json,
                   completion.artifact_fingerprint,
                   completion.config_id,completion.config_fingerprint,
                   completion.lineage_fingerprint,completion.resolved_yaml,
                   completion.provenance_json,completion.command_json
            FROM completions completion
            JOIN work_items item ON item.id=completion.work_item_id
            WHERE completion.work_item_id IN ({placeholders})
            """,
            chunk,
        ):
            rows[int(row["work_item_id"])] = dict(row)
    if not rows:
        return {}

    artifacts: dict[int, dict[str, list[dict]]] = {
        work_item_id: {"input": [], "output": [], "private": []} for work_item_id in rows
    }
    upstream: dict[int, list[dict]] = {work_item_id: [] for work_item_id in rows}
    row_ids = tuple(rows)
    for chunk in _chunks(row_ids):
        placeholders = ",".join("?" for _ in chunk)
        for artifact in database.execute(
            f"""SELECT artifact.work_item_id,artifact.direction,artifact.path,
                       artifact.size,artifact.mtime_ns,artifact.digest_algorithm,
                       artifact.digest
                FROM artifacts artifact
                JOIN completions completion
                  ON completion.work_item_id=artifact.work_item_id
                 AND artifact.attempt_id IS completion.attempt_id
                WHERE artifact.work_item_id IN ({placeholders})
                ORDER BY artifact.work_item_id,artifact.direction,artifact.path""",
            chunk,
        ):
            item = {
                "path": artifact["path"],
                "size": artifact["size"],
                "mtime_ns": artifact["mtime_ns"],
            }
            if artifact["digest_algorithm"] == "sha256" and artifact["digest"] is not None:
                item["sha256"] = artifact["digest"]
            artifacts[int(artifact["work_item_id"])].setdefault(
                str(artifact["direction"]), []
            ).append(item)
        for parent in database.execute(
            f"""SELECT work_item_id,upstream_work_item_id,required_generation
                FROM work_item_dependencies
                WHERE work_item_id IN ({placeholders})
                  AND required_generation IS NOT NULL
                ORDER BY work_item_id,upstream_work_item_id,role""",
            chunk,
        ):
            upstream[int(parent["work_item_id"])].append(
                {
                    "work_item_id": int(parent["upstream_work_item_id"]),
                    "generation": int(parent["required_generation"]),
                }
            )

    records: dict[int, dict] = {}
    for work_item_id, row in rows.items():
        item_artifacts = artifacts[work_item_id]
        record = {
            "work_item_id": work_item_id,
            "work_item_key": row["work_item_key"],
            "module": row["module"],
            "project": row["project"],
            "participant": row["participant"],
            "entities": json.loads(row["entities_json"]),
            "module_lineage_id": row["module_lineage_id"],
            "revision_fingerprint": row["revision_fingerprint"],
            "artifact_contract": json.loads(row["artifact_contract_json"]),
            "artifact_fingerprint": row["artifact_fingerprint"],
            "configuration": {
                "id": row["config_id"],
                "fingerprint": row["config_fingerprint"],
                "lineage_fingerprint": row["lineage_fingerprint"],
                "resolved": yaml.safe_load(row["resolved_yaml"]) or {},
            },
            "generation": int(row["generation"]),
            "attempt_id": (int(row["attempt_id"]) if row["attempt_id"] is not None else None),
            "completed_at": row["completed_at"],
            "inputs": item_artifacts["input"],
            "upstream": upstream[work_item_id],
            "public_outputs": item_artifacts["output"],
            "private_artifacts": item_artifacts["private"],
        }
        if row["command_json"] is not None:
            record["software"] = {
                "name": "nro",
                "command": json.loads(row["command_json"]),
            }
        if row["provenance_json"] is not None:
            record["implementation"] = json.loads(row["provenance_json"])
        records[work_item_id] = record
    return records


def completion_record(database: sqlite3.Connection, work_item_id: int) -> dict | None:
    """Return one completion in the transport shape used by workers and publishers."""
    return completion_records(database, (work_item_id,)).get(work_item_id)
