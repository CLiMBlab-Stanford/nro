"""Common run weighting for covariance and correlation estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .cleaned_timeseries import CleanedRunMetadata


@dataclass(frozen=True)
class ConnectivityRunWeight:
    """Temporal information and normalized covariance weight for one run."""

    effective_dof: int
    normalized_weight: float


def connectivity_run_weights(
    metadata: Sequence[CleanedRunMetadata],
    *,
    weighting: str,
    global_signal_regression: bool = False,
) -> tuple[ConnectivityRunWeight, ...]:
    """Assign equal or effective-DOF weights to admitted cleaned runs.

    Global-signal regression consumes one temporal direction. Precision
    weighting is linear in the remaining rank because covariance sampling
    variance scales inversely with temporal degrees of freedom.
    """

    if weighting not in {"equal", "precision"}:
        raise ValueError(f"Unsupported connectivity weighting: {weighting}")
    if not metadata:
        raise ValueError("Connectivity weighting requires at least one run")
    effective_dof = np.asarray(
        [record.algebraic_temporal_rank - int(global_signal_regression) for record in metadata],
        dtype=np.int64,
    )
    if np.any(effective_dof <= 0):
        raise ValueError("Connectivity weighting requires positive effective temporal DOF")
    raw = effective_dof.astype(np.float64) if weighting == "precision" else np.ones(len(metadata))
    normalized = raw / raw.sum()
    return tuple(
        ConnectivityRunWeight(int(dof), float(weight))
        for dof, weight in zip(effective_dof, normalized)
    )
