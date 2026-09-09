"""Canonical public paths for dynamic-connectivity artifacts."""

from pathlib import Path


def output_paths(directory: Path, prefix: str, domain: str) -> dict[str, Path]:
    """Return the fixed outputs for one target."""

    suffix = ".dtseries.nii" if domain == "surface" else ".nii"
    return {
        "timeseries": Path(directory) / f"{prefix}_desc-dynamicConnectivity_bold{suffix}",
        "scene": Path(directory) / f"{prefix}_desc-dynamicConnectivity_scene.scene",
        "manifest": Path(directory) / f"{prefix}_desc-dynamicConnectivity_manifest.yaml",
        "index": Path(directory) / f"{prefix}_desc-dynamicConnectivityIndex_manifest.json",
    }
