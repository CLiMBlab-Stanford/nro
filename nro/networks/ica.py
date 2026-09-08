"""ICA network estimation from sparse microparcel connectivity."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import sparse

from .config import IcaConfig


LOG = logging.getLogger(__name__)


def ica_membership(
    lower_adjacency: sparse.spmatrix,
    config: IcaConfig,
) -> np.ndarray:
    """Return normalized ICA loadings for each microparcel and network."""
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.utils.extmath import randomized_svd

    adjacency = sparse.csr_matrix(lower_adjacency, dtype=np.float32)
    adjacency = adjacency + adjacency.T
    n_microparcels = int(adjacency.shape[0])
    if adjacency.shape[1] != n_microparcels:
        raise ValueError("ICA requires a square microparcel adjacency")
    if config.n_networks >= n_microparcels:
        raise ValueError(
            "ica.n_networks must be smaller than the number of microparcels: "
            f"{config.n_networks} >= {n_microparcels}"
        )

    LOG.info(
        "Reducing the %d x %d sparse adjacency to %d ICA dimensions",
        n_microparcels,
        n_microparcels,
        config.n_networks,
    )
    left, singular_values, _right = randomized_svd(
        adjacency,
        n_components=config.n_networks,
        n_oversamples=config.svd_oversamples,
        n_iter=config.svd_power_iterations,
        random_state=config.random_seed,
        flip_sign=False,
    )
    scores = left * singular_values[np.newaxis, :]
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
        loadings = estimator.fit_transform(scores)
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
    upper = np.quantile(
        loadings, config.upper_quantile, axis=0, keepdims=True
    )
    if np.any(~np.isfinite(upper)) or np.any(upper <= 0):
        raise ValueError("ICA produced a component without a positive loading tail")
    probabilities = np.clip(loadings, 0, upper) / upper
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("ICA produced non-finite normalized loadings")
    return np.asarray(probabilities, dtype=np.float32)
