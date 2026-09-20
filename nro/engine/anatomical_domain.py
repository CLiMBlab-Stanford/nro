"""Resolve the public spatial domain published by an anatomical artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nro.engine.execution import require_existing_path
from nro.engine.io import read_json
from nro.modules.anat.contract import validate_anatomical_manifest


def _path(value: object, *, field: str, manifest: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Anatomical manifest lacks {field}: {manifest}")
    path = Path(raw)
    require_existing_path(path, f"anatomical output {field}")
    return path


@dataclass(frozen=True)
class AnatomicalDomain:
    """Public observed anatomy plus any intact scaffold needed for registration."""

    manifest: Path
    observed_t1w: Path
    registration_t1w: Path
    brain_mask: Path
    subjects_dir: Path
    fs_subject: str
    surfaces: Mapping[str, Path]
    vertex_mappings: Mapping[str, Path]
    lesion_mask: Path | None

    @property
    def lesion_aware(self) -> bool:
        """Return whether the domain excludes automatically identified lesion tissue."""
        return self.lesion_mask is not None


def load_anatomical_domain(manifest: Path) -> tuple[AnatomicalDomain, dict[str, object]]:
    """Load and validate anatomy paths used by downstream spatial operations.

    The observed T1w remains the public anatomical reference. Lesion-aware
    artifacts additionally provide a synthetic intact T1w for registration
    algorithms that assume a closed, undamaged brain.
    """
    document = read_json(manifest)
    validate_anatomical_manifest(document)
    if not bool(document.get("complete")):
        raise ValueError(f"Anatomical manifest is not marked complete: {manifest}")
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"Anatomical manifest lacks outputs: {manifest}")
    observed = _path(outputs.get("acpc_t1w"), field="outputs.acpc_t1w", manifest=manifest)
    brain_mask = _path(outputs.get("brain_mask"), field="outputs.brain_mask", manifest=manifest)
    subjects_dir = _path(
        document.get("freesurfer_subjects_dir"),
        field="freesurfer_subjects_dir",
        manifest=manifest,
    )
    fs_subject = str(document.get("fs_subject") or "").strip()
    if not fs_subject:
        raise ValueError(f"Anatomical manifest lacks fs_subject: {manifest}")

    lesion = document.get("lesion")
    lesion_aware = isinstance(lesion, Mapping) and lesion.get("enabled") is True
    registration = (
        _path(
            outputs.get("inpainted_acpc_t1w"),
            field="outputs.inpainted_acpc_t1w",
            manifest=manifest,
        )
        if lesion_aware
        else observed
    )
    lesion_mask = (
        _path(outputs.get("lesion_mask"), field="outputs.lesion_mask", manifest=manifest)
        if lesion_aware
        else None
    )

    raw_surfaces = outputs.get("surfaces")
    if not isinstance(raw_surfaces, Mapping):
        raise ValueError(f"Anatomical manifest lacks outputs.surfaces: {manifest}")
    surfaces = {
        str(key): _path(value, field=f"outputs.surfaces.{key}", manifest=manifest)
        for key, value in raw_surfaces.items()
    }
    raw_mappings = outputs.get("surface_vertex_mappings", {})
    if not isinstance(raw_mappings, Mapping):
        raise ValueError(
            f"Anatomical manifest outputs.surface_vertex_mappings must be a mapping: {manifest}"
        )
    mappings = {
        str(key): _path(
            value,
            field=f"outputs.surface_vertex_mappings.{key}",
            manifest=manifest,
        )
        for key, value in raw_mappings.items()
    }
    if lesion_aware and set(mappings) != {"lh", "rh"}:
        raise ValueError(
            f"Lesion-aware anatomy must publish left and right surface vertex mappings: {manifest}"
        )
    return (
        AnatomicalDomain(
            manifest=manifest,
            observed_t1w=observed,
            registration_t1w=registration,
            brain_mask=brain_mask,
            subjects_dir=subjects_dir,
            fs_subject=fs_subject,
            surfaces=surfaces,
            vertex_mappings=mappings,
            lesion_mask=lesion_mask,
        ),
        document,
    )
