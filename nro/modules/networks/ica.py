"""ICA network estimation from bounded spatial features."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .config import IcaConfig

LOG = logging.getLogger(__name__)

READOUTS = ("loading", "correlation")
# Each ICA parcellation strategy and the readout that turns its fit into maps.
ICA_READOUTS = {"ica": "loading", "ica_correlation": "correlation"}
ICA_STRATEGIES = frozenset(ICA_READOUTS)


def ica_membership(
    features: np.ndarray,
    config: IcaConfig,
    *,
    readout: str = "loading",
) -> np.ndarray:
    """Return normalized ICA maps for each spatial location and network.

    With ``readout="loading"`` each map is the component's spatial loading.
    With ``readout="correlation"`` the same fit is read out as each location's
    correlation with the component's mixing direction, which is the component
    time course when rows of ``features`` are correlation factors. Both
    readouts share the sign convention. Loading maps keep the positive tail,
    scaled to [0, 1] by ``config.upper_quantile``. Correlation maps are the
    unscaled correlations in [-1, 1].
    """
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    if readout not in READOUTS:
        raise ValueError(f"Unsupported ICA readout: {readout!r}")
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
    signs = np.where(np.median(loadings, axis=0) > 0, -1.0, 1.0).astype(np.float32)
    loadings = loadings * signs
    if readout == "correlation":
        # Correlations are published unscaled so each value stays a correlation.
        correlations = _component_correlations(features, estimator.mixing_ * signs, loadings)
        if not np.all(np.isfinite(correlations)):
            raise ValueError("ICA produced non-finite component correlations")
        return np.clip(correlations, -1.0, 1.0).astype(np.float32)
    upper = np.quantile(loadings, config.upper_quantile, axis=0, keepdims=True)
    if np.any(~np.isfinite(upper)) or np.any(upper <= 0):
        raise ValueError("ICA produced a component without a positive loading tail")
    probabilities = np.clip(loadings, 0, upper) / upper
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("ICA produced non-finite normalized loadings")
    return np.asarray(probabilities, dtype=np.float32)


def _component_correlations(
    features: np.ndarray,
    mixing: np.ndarray,
    loadings: np.ndarray,
) -> np.ndarray:
    """Return each location's correlation with each oriented component direction.

    Rows of ``features`` are correlation factors, so the cosine between a row
    and a component's mixing column equals the correlation between that
    location's series and the component's series. Each correlation map is then
    oriented to agree with the sign of its loading map.
    """

    directions = np.asarray(mixing, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=0, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("ICA produced a component without a finite mixing direction")
    directions /= norms
    row_norms = np.linalg.norm(features, axis=1, keepdims=True).astype(np.float64)
    row_norms[row_norms <= 0] = 1.0
    correlations = np.empty(loadings.shape, dtype=np.float32)
    for start in range(0, features.shape[0], _ROW_BLOCK):
        stop = start + _ROW_BLOCK
        block = np.asarray(features[start:stop], dtype=np.float64) / row_norms[start:stop]
        correlations[start:stop] = block @ directions
    agreement = np.einsum("ij,ij->j", correlations, loadings, dtype=np.float64)
    return correlations * np.where(agreement < 0, -1.0, 1.0).astype(np.float32)


_ROW_BLOCK = 65536
