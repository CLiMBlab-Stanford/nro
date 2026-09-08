"""Canonical paths and BIDS descriptors for network derivatives."""

from __future__ import annotations

import re
from pathlib import Path


def fixed_output_paths(directory: Path, prefix: str) -> dict[str, Path]:
    """Return the fixed outputs for one individualized-network target."""
    directory = Path(directory)
    return {
        "membership": directory / f"{prefix}_desc-networks_stat.dscalar.nii",
        "stability": directory / f"{prefix}_desc-networkStability_stat.dscalar.nii",
        "homeless": directory / f"{prefix}_desc-homelessNetwork_stat.dscalar.nii",
        "overlap": directory / f"{prefix}_desc-networkOverlap_stat.dscalar.nii",
        "network_labels": directory / f"{prefix}_desc-networkLabels_labels.tsv",
        "network_labels_metadata": directory / f"{prefix}_desc-networkLabels_labels.json",
        "manifest": directory / f"{prefix}_desc-networks_manifest.yaml",
        "index": directory / f"{prefix}_desc-networksIndex_manifest.json",
    }


def network_descriptor(candidate: str | None, network: int) -> str:
    """Return one alphanumeric BIDS description for a network map."""
    if candidate:
        value = re.sub(r"[^A-Za-z0-9]", "", candidate).upper()
        if not value:
            raise ValueError(f"Network candidate has no alphanumeric identifier: {candidate!r}")
        return f"network{value}"
    return f"network{network:03d}"


def network_map_path(directory: Path, prefix: str, descriptor: str) -> Path:
    return Path(directory) / f"{prefix}_desc-{descriptor}_stat.dscalar.nii"
