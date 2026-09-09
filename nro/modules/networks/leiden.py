"""Fit Leiden partitions to sparse parcel-connectivity graphs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import sparse

LOG = logging.getLogger(__name__)


def leiden_partition(
    adjacency: sparse.spmatrix,
    *,
    resolution: float,
    iterations: int,
    seed: int | None,
) -> list[list[int]]:
    """Fit a weighted Leiden partition to the coarse connectivity matrix."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise RuntimeError(
            "Leiden initialization requires the 'igraph' and 'leidenalg' packages. "
            "Install networks with its declared dependencies."
        ) from exc

    lower = sparse.tril(adjacency, k=-1, format="coo")
    n = int(lower.shape[0])
    edges = np.column_stack((lower.row, lower.col)).astype(np.int32, copy=False)
    edge_weights = np.asarray(lower.data, dtype=np.float32)
    LOG.info("Fitting Leiden initialization on %d nodes and %d edges", n, len(edge_weights))

    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = edge_weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=iterations,
        seed=seed,
    )
    groups = [sorted(map(int, group)) for group in partition if group]
    LOG.info("Leiden initialization found %d communities", len(groups))
    return groups


def write_hint(path: Path, groups: list[list[int]]) -> Path:
    """Write an OSLOM -hint file: one whitespace-delimited community per line."""
    with path.open("w", encoding="utf8") as f:
        for group in groups:
            if group:
                f.write(" ".join(map(str, group)))
                f.write("\n")
    return path
