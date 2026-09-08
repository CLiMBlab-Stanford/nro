"""Repeated mini-batch k-means clustering of connectivity profiles."""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from .config import ClusteringConfig


LOG = logging.getLogger(__name__)


def _correlation_assignment(
    reference_counts: np.ndarray,
    labels: np.ndarray,
    n_networks: int,
) -> np.ndarray:
    """Match a hard partition to accumulated memberships by spatial correlation."""
    n_nodes = int(labels.size)
    rows = np.arange(n_nodes, dtype=np.int64)
    indicators = sparse.csr_matrix(
        (np.ones(n_nodes, dtype=np.float32), (rows, labels)),
        shape=(n_nodes, n_networks),
    )
    cross = np.asarray(reference_counts.T @ indicators, dtype=np.float64)
    reference_sum = np.asarray(reference_counts.sum(axis=0), dtype=np.float64)
    reference_square_sum = np.asarray(
        np.square(reference_counts, dtype=np.float64).sum(axis=0),
        dtype=np.float64,
    )
    candidate_sum = np.bincount(labels, minlength=n_networks).astype(np.float64)
    numerator = cross - np.outer(reference_sum, candidate_sum) / n_nodes
    reference_norm = reference_square_sum - np.square(reference_sum) / n_nodes
    candidate_norm = candidate_sum - np.square(candidate_sum) / n_nodes
    denominator = np.sqrt(np.maximum(np.outer(reference_norm, candidate_norm), 0.0))
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    rows_assigned, columns_assigned = linear_sum_assignment(scores, maximize=True)
    if not np.array_equal(rows_assigned, np.arange(n_networks)):
        raise RuntimeError("Clustering alignment did not assign every reference network")
    return columns_assigned


def clustering_membership(
    lower_adjacency: sparse.spmatrix,
    config: ClusteringConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned assignment frequencies and fit inertias.

    This adapts the procedure used by Shain and Fedorenko (2026): locations are
    clustered repeatedly from binarized connectivity profiles, fit labels are
    aligned one-to-one by spatial correlation, and aligned assignments are
    averaged. The sparse representation avoids materializing the dense
    location-by-location connectivity matrix.
    """
    from sklearn.cluster import MiniBatchKMeans

    profiles = sparse.csr_matrix(lower_adjacency, dtype=np.float32)
    profiles = profiles + profiles.T
    profiles.data.fill(1.0)
    profiles.eliminate_zeros()
    n_nodes = int(profiles.shape[0])
    if profiles.shape[1] != n_nodes:
        raise ValueError("Clustering requires a square microparcel adjacency")
    if config.n_networks >= n_nodes:
        raise ValueError(
            "clustering.n_networks must be smaller than the number of microparcels: "
            f"{config.n_networks} >= {n_nodes}"
        )

    fits: list[tuple[float, np.ndarray]] = []
    for repetition in range(config.repetitions):
        seed = (
            None
            if config.random_seed is None
            else int(config.random_seed) + repetition
        )
        LOG.info(
            "Fitting clustering repetition %d/%d",
            repetition + 1,
            config.repetitions,
        )
        estimator = MiniBatchKMeans(
            n_clusters=config.n_networks,
            n_init=config.n_init,
            max_iter=config.max_iterations,
            batch_size=config.batch_size,
            max_no_improvement=config.max_no_improvement,
            reassignment_ratio=config.reassignment_ratio,
            random_state=seed,
        )
        labels = np.asarray(estimator.fit_predict(profiles), dtype=np.int32)
        if np.unique(labels).size != config.n_networks:
            raise ValueError(
                "Mini-batch k-means returned fewer networks than requested"
            )
        fits.append((float(estimator.inertia_), labels))

    # The lowest-inertia fit supplies the initial identities. Subsequent fits
    # are greedily aligned to the accumulated membership maps, matching the
    # reference implementation's default alignment procedure.
    fits.sort(key=lambda item: item[0])
    counts = np.zeros((n_nodes, config.n_networks), dtype=np.float32)
    rows = np.arange(n_nodes, dtype=np.int64)
    for index, (_inertia, labels) in enumerate(fits):
        if index:
            assignment = _correlation_assignment(counts, labels, config.n_networks)
            inverse = np.empty(config.n_networks, dtype=np.int32)
            inverse[assignment] = np.arange(config.n_networks, dtype=np.int32)
            labels = inverse[labels]
        counts[rows, labels] += 1.0
    counts /= float(config.repetitions)
    lower = counts.min(axis=0, keepdims=True)
    span = counts.max(axis=0, keepdims=True) - lower
    if np.any(span <= 0):
        raise ValueError("Clustering produced a spatially constant network membership")
    membership = (counts - lower) / span
    return membership, np.asarray([item[0] for item in fits], dtype=np.float64)
