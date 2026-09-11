"""Canonical paths and BIDS descriptors for network derivatives."""

from __future__ import annotations

from pathlib import Path

from nro.engine.cifti import indexed_cifti_sidecar


def fixed_output_paths(directory: Path, prefix: str) -> dict[str, Path]:
    """Return the fixed outputs for one individualized-network target."""
    directory = Path(directory)
    paths = {
        "membership": directory / f"{prefix}_desc-networks_stat.dscalar.nii",
        "stability": directory / f"{prefix}_desc-networkStability_stat.dscalar.nii",
        "homeless": directory / f"{prefix}_desc-homelessNetwork_stat.dscalar.nii",
        "overlap": directory / f"{prefix}_desc-networkOverlap_stat.dscalar.nii",
        "network_labels": directory / f"{prefix}_desc-networkLabels_labels.tsv",
        "network_labels_metadata": directory / f"{prefix}_desc-networkLabels_labels.json",
        "manifest": directory / f"{prefix}_desc-networks_manifest.yaml",
        "index": directory / f"{prefix}_desc-networksIndex_manifest.json",
    }
    paths.update(
        {
            f"{name}_metadata": indexed_cifti_sidecar(paths[name])
            for name in ("membership", "stability", "homeless", "overlap")
        }
    )
    return paths
