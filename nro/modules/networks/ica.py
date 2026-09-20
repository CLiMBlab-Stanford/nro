"""ICA network estimation from bounded spatial features."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .config import IcaConfig

LOG = logging.getLogger(__name__)


def ica_membership(
    features: np.ndarray,
    config: IcaConfig,
) -> np.ndarray:
    """Return normalized ICA loadings for each spatial location and network."""
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    features = np.asarray(features, dtype=np.float32)
    n_locations, n_features = features.shape
    if config.n_networks >= min(n_locations, n_features):
        raise ValueError(
            "ica.n_networks must be smaller than both spatial and feature dimensions: "
            f"{config.n_networks} >= min({n_locations}, {n_features})"
        )

    LOG.info(
        "Fitting %d ICA networks from %d locations by %d features",
        config.n_networks,
        n_locations,
        n_features,
    )
    estimator = FastICA(
        n_components=config.n_networks,
        whiten="unit-variance",
        whiten_solver="eigh",
        max_iter=config.max_iterations,
        tol=config.tolerance,
        random_state=config.random_seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        loadings = estimator.fit_transform(features)
    if estimator.n_iter_ >= config.max_iterations:
        LOG.warning(
            "ICA reached its %d-iteration limit before convergence",
            config.max_iterations,
        )

    # A functional network is expected to occupy less than half the spatial
    # domain. Match the sign convention used by climbprep before retaining the
    # positive tail of each component.
    loadings = np.where(
        np.median(loadings, axis=0, keepdims=True) > 0,
        -loadings,
        loadings,
    )
    upper = np.quantile(loadings, config.upper_quantile, axis=0, keepdims=True)
    if np.any(~np.isfinite(upper)) or np.any(upper <= 0):
        raise ValueError("ICA produced a component without a positive loading tail")
    probabilities = np.clip(loadings, 0, upper) / upper
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("ICA produced non-finite normalized loadings")
    return np.asarray(probabilities, dtype=np.float32)
