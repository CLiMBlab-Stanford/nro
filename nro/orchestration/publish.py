"""Freeze a completed nro request into a standalone derivative dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from nro.engine.paths import module_artifact_root
from nro.orchestration.completion_records import completion_record
from nro.orchestration.manifests import assess_registry
from nro.orchestration.registry import Registry, utcnow


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _derivative_root(project_root: Path, work_item: dict) -> Path:
    path = Path(work_item["output_root"]).resolve()
    expected = module_artifact_root(
        project_root,
        str(work_item["module"]),
        str(work_item["directory_label"]),
    ).resolve()
    try:
        path.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(f"Work-item output is outside its derivative dataset: {path}") from error
    return expected


def _completion_digest(record: dict) -> str:
    """Fingerprint one detached database completion record."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _portable_manifest(database, work_item_id: int, seen: set[int]) -> dict:
    """Embed workflow-agnostic recursive provenance without registry IDs/paths."""
    if work_item_id in seen:
        raise RuntimeError(f"Completion dependency cycle at work item {work_item_id}")
    seen.add(work_item_id)
    manifest = completion_record(database, work_item_id)
    if manifest is None:
        raise RuntimeError(f"Work item {work_item_id} has no completion record")
    upstream = [
        _portable_manifest(database, int(item["work_item_id"]), seen)
        for item in manifest.get("upstream", [])
    ]
    seen.remove(work_item_id)
    return {
        "module": manifest["module"],
        "participant": manifest["participant"],
        "entities": manifest.get("entities", {}),
        "generation": manifest["generation"],
        "completed_at": manifest["completed_at"],
        "configuration": manifest.get("configuration", {}),
        "software": manifest.get("software", {}),
        "inputs": manifest.get("inputs", []),
        "upstream": upstream,
    }


def publish(
    registry: Registry,
    *,
    request_id: str,
    destination: Path,
    validate: bool = True,
    compiled: bool = False,
) -> Path:
    """Snapshot fresh terminal artifacts and recursive provenance into a new destination.

    Reassess the registry before and after copying. Validate checksums and
    generations; reject an existing destination or changed source artifacts.
    Run an available BIDS validator unless validate is false.
    """
    assess_registry(registry, projects=(registry.paths.project,), compiled=compiled)
    request, work_items = registry.publication_work_items(request_id)
    if not work_items:
        raise RuntimeError(f"Request has no terminal derivative work items: {request_id}")
    not_fresh = [item for item in work_items if item["artifact_state"] != "fresh"]
    if not_fresh:
        raise RuntimeError(
            "Cannot publish a request with nonfresh terminal derivatives: "
            + ", ".join(str(work_item["id"]) for work_item in not_fresh)
        )
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Publication destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    copied: list[dict] = []
    expected_generations: dict[int, tuple[int, str]] = {}
    embedded_work_items: list[dict] = []
    try:
        for work_item in work_items:
            with registry.connection() as db:
                manifest = completion_record(db, int(work_item["id"]))
                if manifest is None:
                    raise RuntimeError(
                        f"Fresh work item {work_item['id']} has no completion record"
                    )
                portable = _portable_manifest(db, int(work_item["id"]), set())
            expected_generations[int(work_item["id"])] = (
                int(work_item["current_generation"]),
                _completion_digest(manifest),
            )
            with registry.connection() as db:
                execution = db.execute(
                    "SELECT context_json FROM work_item_execution WHERE work_item_id=?",
                    (work_item["id"],),
                ).fetchone()
            project_root = registry.paths.project_root
            if execution:
                from nro.orchestration.execution_context import ExecutionContext

                project_root = ExecutionContext.from_dict(
                    json.loads(execution[0])
                ).paths.output_project(work_item["project"])
            root = _derivative_root(project_root, work_item)
            for item in manifest["public_outputs"]:
                source = Path(item["path"]).resolve()
                relative = source.relative_to(root)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source_digest = _sha256(source)
                shutil.copy2(source, target)
                target_digest = _sha256(target)
                if target_digest != source_digest:
                    raise RuntimeError(f"Checksum mismatch while publishing {source}")
                copied.append(
                    {"source": str(source), "path": str(relative), "sha256": target_digest}
                )
            embedded_work_items.append(portable)
        description = {
            "Name": destination.name,
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "nro"}],
            "SourceDatasets": [{"URL": str(registry.paths.project_root)}],
        }
        (staging / "dataset_description.json").write_text(
            json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = {
            "published_at": utcnow(),
            "project": request["project"],
            "target_module": request["target_module"],
            "work_items": embedded_work_items,
            "files": [
                {key: value for key, value in item.items() if key != "source"} for item in copied
            ],
        }
        (staging / ".nro-publication.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validator = shutil.which("bids-validator") if validate else None
        if validator:
            subprocess.run(
                [validator, str(staging)],
                check=True,
                text=True,
                capture_output=True,
            )
        assess_registry(registry, projects=(registry.paths.project,), compiled=compiled)
        _request_after, work_items_after = registry.publication_work_items(request_id)
        if any(work_item["artifact_state"] != "fresh" for work_item in work_items_after):
            raise RuntimeError("Live derivative changed or became stale during publication")
        current = {int(work_item["id"]): work_item for work_item in work_items_after}
        for work_item_id, (generation, completion_digest) in expected_generations.items():
            work_item = current.get(work_item_id)
            if work_item is None or int(work_item["current_generation"]) != generation:
                raise RuntimeError("Live derivative generation changed during publication")
            with registry.connection() as db:
                current_completion = completion_record(db, work_item_id)
            if (
                current_completion is None
                or _completion_digest(current_completion) != completion_digest
            ):
                raise RuntimeError("Live derivative completion changed during publication")
        for item in copied:
            if _sha256(Path(item["source"])) != item["sha256"]:
                raise RuntimeError("Live derivative output changed during publication")
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
