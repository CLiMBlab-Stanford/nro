"""Resolve microparcellation inputs from fixed cleaning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nro.engine.bids import BidsRun
from nro.clean.paths import clean_session_dir, clean_subject_dir
from nro.engine.targets import smoothing_entity_value


@dataclass(frozen=True)
class CleanTarget:
    """Matched clean artifact inputs for one space and smoothing target."""
    domain: str
    space: str
    smoothing_mm: int
    functional: tuple[tuple[Path, ...], ...]
    temporal_masks: tuple[Path, ...]


def expected_clean_target(
    runs: Iterable[BidsRun],
    *,
    space: str,
    smoothing_mm: int,
    project: str,
    clean_id: str,
) -> CleanTarget:
    """Construct one clean target solely from source runs and requested entities."""
    source_runs = tuple(runs)
    if not source_runs:
        raise FileNotFoundError("No source-BIDS runs were selected for microparcellation")
    domain = "surface" if space in {"fsnative", "fsaverage"} else "volume"
    smoothing = smoothing_entity_value(smoothing_mm)
    functionals: list[tuple[Path, ...]] = []
    temporal_masks: list[Path] = []
    for run in source_runs:
        sub_id = f"sub-{run.participant}"
        directory = (
            clean_session_dir(
                sub_id, f"ses-{run.session}", project=project, clean_id=clean_id
            )
            if run.session
            else clean_subject_dir(sub_id, project=project, clean_id=clean_id)
        )
        if domain == "surface":
            functionals.append(
                tuple(
                    directory
                    / (
                        f"{run.stem}_space-{space}_smoothing-{smoothing}_hemi-{hemi}"
                        "_desc-clean_bold.func.gii"
                    )
                    for hemi in ("L", "R")
                )
            )
        else:
            functionals.append(
                (
                    directory
                    / (
                        f"{run.stem}_space-{space}_smoothing-{smoothing}"
                        "_desc-clean_bold.nii.gz"
                    ),
                )
            )
        temporal_masks.append(
            directory
            / (
                f"{run.stem}_space-{space}_smoothing-{smoothing}_"
                "desc-confounds_timeseries.tsv"
            )
        )
    return CleanTarget(
        domain,
        space,
        smoothing_mm,
        tuple(functionals),
        tuple(temporal_masks),
    )
