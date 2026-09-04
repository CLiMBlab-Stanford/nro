"""Discovery of subject-level microparcellation targets for network estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class MicroparcellationTarget:
    domain: str
    space: str
    smoothing_mm: int
    manifest: Path
    microparcels: Path
    connectivity: Path
    source_surfaces: tuple[Path, ...]
    scene_surfaces: tuple[Path, ...]
    label_volume: Path | None


def _output_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def discover_microparcellation_targets(
    manifests: Iterable[Path],
) -> tuple[MicroparcellationTarget, ...]:
    """Load the exact microparcellation manifests derived by the caller."""
    targets = []
    manifest_paths = tuple(Path(path) for path in manifests)
    if not manifest_paths:
        raise FileNotFoundError("No source/config-defined microparcellation targets")
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing microparcellation publication manifest: {manifest_path}"
            )
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        manifest_domain = manifest.get("domain")
        if manifest_domain not in {"surface", "volume"}:
            continue
        manifest_space = str(manifest.get("space") or "").strip()
        manifest_smoothing = manifest.get("smoothing_fwhm_mm")
        if not manifest_space or manifest_smoothing is None:
            raise ValueError(
                "Microparcellation manifest must declare space and smoothing_fwhm_mm: "
                f"{manifest_path}"
            )
        smoothing_mm = int(manifest_smoothing)
        outputs = manifest.get("outputs") or {}
        missing = {name for name in ("microparcels", "connectivity") if not outputs.get(name)}
        if missing:
            raise ValueError(
                f"Microparcellation manifest lacks required outputs {sorted(missing)}: "
                f"{manifest_path}"
            )
        source_surfaces = tuple(
            _output_path(value, manifest_path)
            for value in manifest.get("source_surfaces", ())
        )
        scene_surfaces = tuple(
            _output_path(value, manifest_path)
            for value in outputs.get("scene_surfaces", ())
        )
        label_volume = (
            _output_path(outputs["microparcels_volume"], manifest_path)
            if outputs.get("microparcels_volume")
            else None
        )
        if manifest_domain == "surface" and len(scene_surfaces) != 8:
            raise ValueError(
                f"Surface microparcellation manifest must publish eight scene surfaces: {manifest_path}"
            )
        if manifest_domain == "volume" and label_volume is None:
            raise ValueError(
                f"Volumetric microparcellation manifest lacks outputs.microparcels_volume: {manifest_path}"
            )
        resolved_microparcels = _output_path(outputs["microparcels"], manifest_path)
        resolved_connectivity = _output_path(outputs["connectivity"], manifest_path)
        if not resolved_microparcels.name.endswith(".dlabel.nii"):
            raise ValueError(f"Networks requires a CIFTI dlabel input: {resolved_microparcels}")
        if not resolved_connectivity.name.endswith(".pconn.nii"):
            raise ValueError(f"Networks requires a CIFTI pconn input: {resolved_connectivity}")
        required_paths = (
            resolved_microparcels,
            resolved_connectivity,
            *source_surfaces,
            *scene_surfaces,
            *((label_volume,) if label_volume is not None else ()),
        )
        absent = [str(path) for path in required_paths if not path.is_file()]
        if absent:
            raise FileNotFoundError(
                "Microparcellation manifest records missing output(s): " + ", ".join(absent)
            )
        targets.append(
            MicroparcellationTarget(
                domain=manifest_domain,
                space=manifest_space,
                smoothing_mm=smoothing_mm,
                manifest=manifest_path,
                microparcels=resolved_microparcels,
                connectivity=resolved_connectivity,
                source_surfaces=source_surfaces,
                scene_surfaces=scene_surfaces,
                label_volume=label_volume,
            )
        )
    if not targets:
        raise FileNotFoundError(
            "No microparcellation targets were found in the configured manifest sequence"
        )
    identities = [
        (target.domain, target.space, target.smoothing_mm) for target in targets
    ]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        formatted = ", ".join(
            f"{domain}/space-{space}/smoothing-{smoothing_mm}mm"
            for domain, space, smoothing_mm in duplicates
        )
        raise ValueError(
            "Multiple microparcellation manifests represent the same target: "
            f"{formatted}"
        )
    return tuple(
        sorted(
            targets,
            key=lambda target: (target.domain, target.space, target.smoothing_mm),
        )
    )
