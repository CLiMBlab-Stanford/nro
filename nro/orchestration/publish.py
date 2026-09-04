"""Freeze a completed nro request into a standalone derivative dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from nro.orchestration.catalog import module_descriptor
from nro.orchestration.manifests import assess_registry
from nro.orchestration.registry import Registry, utcnow


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _derivative_root(project_root: Path, instance: dict) -> Path:
    path = Path(instance["output_root"]).resolve()
    expected = (
        project_root
        / "derivatives"
        / module_descriptor(instance["module"]).configuration_class
    )
    try:
        relative = path.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(f"Instance output is outside its derivative dataset: {path}") from error
    if not relative.parts:
        raise RuntimeError(f"Instance output does not identify a derivative configuration: {path}")
    return expected / relative.parts[0]


def _portable_manifest(path: Path, seen: set[Path], digests: dict[Path, str]) -> dict:
    """Embed workflow-agnostic recursive provenance without registry IDs/paths."""
    path = path.resolve()
    if path in seen:
        raise RuntimeError(f"Completion-manifest dependency cycle: {path}")
    seen.add(path)
    digests[path] = _sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    upstream = [
        _portable_manifest(Path(item["manifest"]), seen, digests)
        for item in manifest.get("upstream", [])
    ]
    seen.remove(path)
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
) -> Path:
    assess_registry(registry, projects=(registry.paths.project,))
    request, instances = registry.publication_instances(request_id)
    if not instances:
        raise RuntimeError(f"Request has no terminal derivative instances: {request_id}")
    not_fresh = [instance for instance in instances if instance["artifact_state"] != "fresh"]
    if not_fresh:
        raise RuntimeError(
            "Cannot publish a request with nonfresh terminal derivatives: "
            + ", ".join(str(instance["id"]) for instance in not_fresh)
        )
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Publication destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    copied: list[dict] = []
    expected_generations: dict[int, tuple[int, str]] = {}
    provenance_manifests: dict[Path, str] = {}
    embedded_instances: list[dict] = []
    try:
        for instance in instances:
            manifest_path = Path(instance["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_generations[int(instance["id"])] = (
                int(instance["current_generation"]),
                _sha256(manifest_path),
            )
            root = _derivative_root(registry.paths.project_root, instance)
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
            embedded_instances.append(
                _portable_manifest(manifest_path, set(), provenance_manifests)
            )
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
            "instances": embedded_instances,
            "files": [{key: value for key, value in item.items() if key != "source"} for item in copied],
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
        assess_registry(registry, projects=(registry.paths.project,))
        _request_after, instances_after = registry.publication_instances(request_id)
        if any(instance["artifact_state"] != "fresh" for instance in instances_after):
            raise RuntimeError("Live derivative changed or became stale during publication")
        current = {int(instance["id"]): instance for instance in instances_after}
        for instance_id, (generation, manifest_digest) in expected_generations.items():
            instance = current.get(instance_id)
            if instance is None or int(instance["current_generation"]) != generation:
                raise RuntimeError("Live derivative generation changed during publication")
            if _sha256(Path(instance["manifest_path"])) != manifest_digest:
                raise RuntimeError("Live derivative manifest changed during publication")
        for item in copied:
            if _sha256(Path(item["source"])) != item["sha256"]:
                raise RuntimeError("Live derivative output changed during publication")
        for manifest_path, digest in provenance_manifests.items():
            if _sha256(manifest_path) != digest:
                raise RuntimeError("Upstream derivative provenance changed during publication")
        os.replace(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
