"""Shared consumption contract for cleaned functional time series."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .images import sidecar_json_path


@dataclass(frozen=True)
class CleanedRunMetadata:
    """Run-level cleaning metadata agreed across all files in one run."""

    files: tuple[Path, ...]
    sidecars: tuple[Path, ...]
    cleaning_defined: bool
    undefined_reasons: tuple[str, ...]
    total_frames: int
    retained_frames: int
    censored_fraction: float
    residual_design_dof: int
    algebraic_temporal_rank: int
    participation_effective_rank: float
    dominant_temporal_variance_fraction: float
    temporal_mask_file: Path
    temporal_mask_regex: str


@dataclass(frozen=True)
class CleanedRunInclusionPolicy:
    """Scientific eligibility thresholds applied without loading image data."""

    minimum_retained_frames: int
    minimum_retained_fraction: float
    minimum_residual_design_dof: int
    minimum_participation_effective_rank: float
    maximum_dominant_temporal_variance_fraction: float


def _required(mapping: dict[str, object], key: str, sidecar: Path) -> object:
    if key not in mapping:
        raise ValueError(f"Cleaned sidecar lacks Cleaning.{key}: {sidecar}")
    return mapping[key]


def load_cleaned_run_metadata(files: tuple[Path, ...]) -> CleanedRunMetadata:
    """Read and validate the sidecar contract for one volume or surface pair."""
    if not files:
        raise ValueError("A cleaned run must contain at least one functional file")
    records: list[tuple[Path, dict[str, object], dict[str, object]]] = []
    for file in files:
        sidecar = sidecar_json_path(file)
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"Cleaned functional sidecar is missing or unreadable: {sidecar}"
            ) from error
        cleaning = document.get("Cleaning") or {}
        if not isinstance(cleaning, dict):
            raise ValueError(f"Cleaning metadata must be an object: {sidecar}")
        quality = cleaning.get("QualityControl") or {}
        if not isinstance(quality, dict):
            raise ValueError(f"Cleaning.QualityControl must be an object: {sidecar}")
        records.append((sidecar, cleaning, quality))

    run_keys = (
        "TotalFrames",
        "RetainedFrames",
        "CensoredFraction",
        "ResidualDesignDegreesOfFreedom",
        "AlgebraicTemporalRank",
        "TemporalMaskFile",
        "TemporalMaskRegex",
    )
    first_sidecar, first, _first_quality = records[0]
    values = {
        key: _required(first, key, first_sidecar)
        for key in run_keys
    }
    for sidecar, cleaning, _quality in records[1:]:
        for key, expected in values.items():
            actual = _required(cleaning, key, sidecar)
            if actual != expected:
                raise ValueError(
                    f"Cleaned run sidecars disagree on Cleaning.{key}: "
                    f"{first_sidecar} != {sidecar}"
                )

    defined: list[bool] = []
    reasons: set[str] = set()
    participation: list[float] = []
    dominant: list[float] = []
    for sidecar, cleaning, quality in records:
        is_defined = bool(_required(cleaning, "CleaningDefined", sidecar))
        defined.append(is_defined)
        if not is_defined:
            reasons.add(
                str(cleaning.get("CleaningUndefinedReason") or "unspecified")
            )
        participation.append(
            float(_required(quality, "ParticipationRatioEffectiveTemporalRank", sidecar))
        )
        dominant.append(
            float(_required(quality, "DominantTemporalVarianceFraction", sidecar))
        )

    mask_file = Path(str(values["TemporalMaskFile"]))
    return CleanedRunMetadata(
        files=tuple(Path(path) for path in files),
        sidecars=tuple(sidecar for sidecar, _cleaning, _quality in records),
        cleaning_defined=all(defined),
        undefined_reasons=tuple(sorted(reasons)),
        total_frames=int(values["TotalFrames"]),
        retained_frames=int(values["RetainedFrames"]),
        censored_fraction=float(values["CensoredFraction"]),
        residual_design_dof=int(values["ResidualDesignDegreesOfFreedom"]),
        algebraic_temporal_rank=int(values["AlgebraicTemporalRank"]),
        participation_effective_rank=min(participation),
        dominant_temporal_variance_fraction=max(dominant),
        temporal_mask_file=mask_file,
        temporal_mask_regex=str(values["TemporalMaskRegex"]),
    )


def cleaned_run_exclusions(
    metadata: CleanedRunMetadata,
    policy: CleanedRunInclusionPolicy,
) -> tuple[dict[str, object], ...]:
    """Return machine-readable exclusion records under one downstream policy."""
    exclusions: list[dict[str, object]] = []

    def below(metric: str, observed: int | float, threshold: int | float) -> None:
        if observed < threshold:
            exclusions.append(
                {
                    "reason": f"{metric}_below_minimum",
                    "metric": metric,
                    "observed": observed,
                    "minimum": threshold,
                }
            )

    if not metadata.cleaning_defined:
        exclusions.append(
            {
                "reason": "cleaning_undefined",
                "cleaning_undefined_reasons": list(metadata.undefined_reasons),
            }
        )
        return tuple(exclusions)
    below("retained_frames", metadata.retained_frames, policy.minimum_retained_frames)
    below(
        "retained_fraction",
        1.0 - metadata.censored_fraction,
        policy.minimum_retained_fraction,
    )
    below(
        "residual_design_dof",
        metadata.residual_design_dof,
        policy.minimum_residual_design_dof,
    )
    below(
        "participation_effective_rank",
        metadata.participation_effective_rank,
        policy.minimum_participation_effective_rank,
    )
    if (
        metadata.dominant_temporal_variance_fraction
        > policy.maximum_dominant_temporal_variance_fraction
    ):
        exclusions.append(
            {
                "reason": "dominant_temporal_variance_fraction_above_maximum",
                "metric": "dominant_temporal_variance_fraction",
                "observed": metadata.dominant_temporal_variance_fraction,
                "maximum": policy.maximum_dominant_temporal_variance_fraction,
            }
        )
    return tuple(exclusions)


def load_retained_frame_mask(metadata: CleanedRunMetadata) -> np.ndarray:
    """Load the declared temporal mask and verify it against sidecar summaries."""
    try:
        confounds = pd.read_csv(metadata.temporal_mask_file, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(
            f"Temporal-mask table is missing or unreadable: {metadata.temporal_mask_file}"
        ) from error
    if len(confounds) != metadata.total_frames:
        raise ValueError(
            "Temporal-mask row count disagrees with clean metadata: "
            f"{len(confounds)} != {metadata.total_frames}"
        )
    outliers = confounds.filter(regex=metadata.temporal_mask_regex)
    retained = (
        np.ones(metadata.total_frames, dtype=bool)
        if outliers.empty
        else ~np.any(outliers.fillna(0.0).to_numpy(dtype=np.float64) != 0.0, axis=1)
    )
    if int(retained.sum()) != metadata.retained_frames:
        raise ValueError(
            "Temporal mask disagrees with clean metadata: "
            f"{int(retained.sum())} != {metadata.retained_frames} retained frames"
        )
    return retained
