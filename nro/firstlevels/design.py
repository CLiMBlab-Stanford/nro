"""Run designs with explicit censoring, estimability and nuisance rank budgets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg

from .models import event_design
from .statistics import UnidentifiableDesignError, nuisance_components


@dataclass
class RunDesign:
    """Independent fit coordinates and the map back to named scientific effects."""

    names: list[str]
    full: np.ndarray
    retained: np.ndarray
    matrix: np.ndarray
    coefficient_map: np.ndarray
    estimability: np.ndarray
    metadata: dict

    def is_estimable(self, weights: np.ndarray) -> bool:
        """Test whether a named linear contrast lies in the estimable row space."""
        weights = np.asarray(weights)
        return bool(np.allclose(weights @ self.estimability, weights, rtol=1e-7, atol=1e-9))


def build_design(node: dict, events: pd.DataFrame, confounds: pd.DataFrame,
                 tr: float, config: dict) -> RunDesign:
    """Fit coordinates from a resolved run node, without reading image intensities.

    Resolve event HRFs and nuisance columns with realize_run_node first.
    The compiled denoising rules must match the supplied module configuration.
    """
    if {"high_pass", "low_pass"} & config.keys() or "Options" in node["Model"]:
        raise ValueError("Firstlevels does not support temporal filtering or Model.Options")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("Repetition time must be positive and finite")
    rules = node["Model"]["Software"]["nro"]
    if any(config[key] != value for key, value in rules["denoising"].items()):
        raise ValueError("Compiled denoising differs from the firstlevels configuration")
    outlier_names = rules["outlier_columns"]
    flags = confounds[outlier_names].to_numpy(float)
    if not np.isfinite(flags).all():
        raise ValueError("Temporal-mask columns contain nonfinite values")
    retained = ~np.any(flags != 0, axis=1)
    if not retained.any():
        raise UnidentifiableDesignError("No retained frames")
    observed_confounds = confounds.copy()
    # Derivatives/FD may have one undefined first observation by convention.
    # No general missing-value imputation is permitted.
    for column in observed_confounds:
        if len(observed_confounds) and pd.isna(observed_confounds[column].iloc[0]):
            observed_confounds.loc[observed_confounds.index[0], column] = 0.0
    full = event_design(node, events, observed_confounds, tr)
    if not np.isfinite(full.to_numpy()).all():
        raise ValueError("Selected design variables contain nonfinite values")
    if any(name.startswith("global_signal") for name in full):
        raise ValueError("Firstlevels does not remove global signal")
    nuisance_names = rules["nuisance_columns"]
    names = [name for name in full if name not in nuisance_names and name not in outlier_names]
    task = full[names].to_numpy()[retained]
    nuisance = full[nuisance_names].to_numpy()[retained]
    u, s, vh = linalg.svd(task, full_matrices=False)
    tolerance = max(task.shape) * np.finfo(float).eps * max(s[0] if len(s) else 0, 1)
    keep = s > tolerance
    # Use independent named columns in the fit/plot. The map retains estimable
    # linear combinations when the original named parameterization is aliased.
    _, _, pivots = linalg.qr(task, mode="economic", pivoting=True)
    selected = pivots[:int(keep.sum())]
    exact = task[:, selected]
    mapping = (vh[keep].T / s[keep]) @ u[:, keep].T @ exact
    projector = vh[keep].T @ vh[keep]
    components, metadata = nuisance_components(
        exact, nuisance, variance_target=config["nuisance_variance_explained"],
        minimum_rank=config["minimum_temporal_rank"], minimum_fraction=config["minimum_temporal_rank_fraction"],
    )
    matrix = np.column_stack((exact, components))
    coefficient_map = np.column_stack((mapping, np.zeros((len(names), components.shape[1]))))
    metadata.update({"Predictors": names, "NuisanceCandidates": nuisance_names,
                     "FitColumns": [names[i] for i in selected] + [f"nuisancePC-{i + 1}" for i in range(components.shape[1])],
                     "TotalFrames": len(full), "RetainedFrameIndices": np.flatnonzero(retained).tolist(),
                     "TemporalMaskColumns": outlier_names, "ObservationDimension": int(retained.sum()),
                     "ResidualDegreesOfFreedom": matrix.shape[0] - matrix.shape[1],
                     "TemporalFiltering": "none", "RepetitionTime": tr,
                     "ResponseScaling": "none", "CoefficientEstimability": projector.tolist(),
                     "StatsModelNode": node,
                     "PredictorSources": {**{n: "events.tsv" for n in rules["event_columns"]},
                                          **{n: "confounds.tsv" for n in nuisance_names + outlier_names},
                                          "intercept": "constant"}})
    return RunDesign(names, full.to_numpy(), retained, matrix, coefficient_map, projector, metadata)
