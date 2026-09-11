"""Canonical paths for microparcellation derivatives."""

from __future__ import annotations

from pathlib import Path


def output_paths(directory: Path, prefix: str) -> dict[str, Path]:
    """Return the fixed outputs for one microparcellation target."""
    directory = Path(directory)
    return {
        "microparcels": directory / f"{prefix}_desc-microparcellation_dseg.dlabel.nii",
        "connectivity": directory / f"{prefix}_connectivity.pconn.nii",
        "quality": directory / f"{prefix}_desc-microparcellationQuality_metrics.json",
        "manifest": directory / f"{prefix}_desc-microparcellation_manifest.yaml",
        "index": directory / f"{prefix}_desc-microparcellationIndex_manifest.json",
        "microparcels_volume": directory / f"{prefix}_desc-microparcellation_dseg.nii.gz",
    }
