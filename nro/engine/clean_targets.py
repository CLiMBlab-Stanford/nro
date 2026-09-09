"""Resolve multirun inputs published by the clean module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nro.engine.bids import BidsRun
from nro.engine.paths import clean_session_dir, clean_subject_dir
from nro.engine.targets import smoothing_entity_value
from nro.orchestration.execution_context import ExecutionContext


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
    execution_context: ExecutionContext | None = None,
) -> CleanTarget:
    """Construct one multirun clean target from source runs and requested entities."""

    source_runs = tuple(runs)
    if execution_context is not None and execution_context.project != project:
        raise ValueError("Clean target project differs from its execution context")
    bids_root = None if execution_context is None else execution_context.paths.bids
    if not source_runs:
        raise FileNotFoundError("No source-BIDS runs were selected for the multirun target")
    domain = "surface" if space in {"fsnative", "fsaverage"} else "volume"
    smoothing = smoothing_entity_value(smoothing_mm)
    functionals, temporal_masks = [], []
    for run in source_runs:
        sub_id = f"sub-{run.participant}"
        directory = (
            clean_session_dir(
                sub_id,
                f"ses-{run.session}",
                project=project,
                clean_id=clean_id,
                bids_root=bids_root,
            )
            if run.session
            else clean_subject_dir(sub_id, project=project, clean_id=clean_id, bids_root=bids_root)
        )
        if domain == "surface":
            functionals.append(
                tuple(
                    directory
                    / (
                        f"{run.stem}_space-{space}_smoothing-{smoothing}_"
                        f"hemi-{hemi}_desc-clean_bold.func.gii"
                    )
                    for hemi in ("L", "R")
                )
            )
        else:
            functionals.append(
                (
                    directory
                    / f"{run.stem}_space-{space}_smoothing-{smoothing}_desc-clean_bold.nii.gz",
                )
            )
        temporal_masks.append(
            directory
            / f"{run.stem}_space-{space}_smoothing-{smoothing}_desc-confounds_timeseries.tsv"
        )
    if execution_context is not None:
        functionals = [
            tuple(execution_context.input_path(path) for path in group) for group in functionals
        ]
        temporal_masks = [execution_context.input_path(path) for path in temporal_masks]
    return CleanTarget(domain, space, smoothing_mm, tuple(functionals), tuple(temporal_masks))
