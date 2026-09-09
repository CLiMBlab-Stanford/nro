"""Spatial null partitions for microparcellation quality assessment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class NullPartition:
    """Spatial null labels and parcel-size diagnostics used during variance scoring."""

    labels: np.ndarray
    mean_absolute_size_error: float
    maximum_absolute_size_error: int
    exactly_matched_fraction: float


def _grow_component(
    nodes: np.ndarray,
    adjacency: sparse.csr_matrix,
    target_sizes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow connected regions toward a randomly assigned size quota."""
    region_count = len(target_sizes)
    seeds = rng.choice(nodes, size=region_count, replace=False)
    targets = rng.permutation(target_sizes)
    assigned = np.full(adjacency.shape[0], -1, dtype=np.int64)
    assigned[seeds] = np.arange(region_count, dtype=np.int64)
    frontiers = [set() for _ in range(region_count)]
    for region, seed in enumerate(seeds):
        neighbors = adjacency.indices[adjacency.indptr[seed] : adjacency.indptr[seed + 1]]
        frontiers[region].update(int(node) for node in neighbors if assigned[node] < 0)

    schedule = np.repeat(np.arange(region_count, dtype=np.int64), targets - 1)
    rng.shuffle(schedule)
    for region_value in schedule:
        region = int(region_value)
        frontier = frontiers[region]
        if not frontier:
            continue
        node = int(rng.choice(tuple(frontier)))
        assigned[node] = region
        neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        for neighbor_value in neighbors:
            neighbor = int(neighbor_value)
            neighbor_region = int(assigned[neighbor])
            if neighbor_region >= 0:
                frontiers[neighbor_region].discard(node)
            else:
                frontier.add(neighbor)

    # Competition can occasionally surround a region before it reaches its
    # quota. Attach any remaining nodes to an adjacent grown region. This
    # preserves connectivity, and candidate selection below minimizes the
    # resulting size error.
    queue = deque(int(node) for node in nodes if assigned[node] >= 0)
    while queue:
        node = queue.popleft()
        region = assigned[node]
        neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        for neighbor_value in neighbors:
            neighbor = int(neighbor_value)
            if assigned[neighbor] < 0:
                assigned[neighbor] = region
                queue.append(neighbor)
    if np.any(assigned[nodes] < 0):
        raise RuntimeError("Spatial null region growing left nodes unassigned")
    return assigned[nodes], targets


def spatial_null_partitions(
    labels: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    *,
    count: int,
    seed: int,
    candidate_attempts: int,
) -> tuple[NullPartition, ...]:
    """Create graph-contiguous nulls closely matching fitted parcel sizes."""
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    active_nodes = np.flatnonzero(mask)
    active_labels = labels[mask]
    active_index = np.full(len(labels), -1, dtype=np.int64)
    active_index[active_nodes] = np.arange(len(active_nodes), dtype=np.int64)
    active_edges = active_index[np.asarray(edges, dtype=np.int64)]
    active_edges = active_edges[(active_edges >= 0).all(axis=1)]
    row = np.concatenate((active_edges[:, 0], active_edges[:, 1]))
    column = np.concatenate((active_edges[:, 1], active_edges[:, 0]))
    adjacency = sparse.csr_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, column)),
        shape=(len(active_nodes), len(active_nodes)),
    )
    component_count, component_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    parcel_components = np.full(int(active_labels.max()) + 1, -1, dtype=np.int64)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for component in range(component_count):
        nodes = np.flatnonzero(component_labels == component)
        fitted = active_labels[nodes]
        parcel_ids, target_sizes = np.unique(fitted, return_counts=True)
        if np.any(parcel_components[parcel_ids] >= 0):
            raise ValueError("A fitted microparcel spans disconnected spatial components")
        parcel_components[parcel_ids] = component
        components.append((nodes, target_sizes.astype(np.int64)))

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(count):
        best: NullPartition | None = None
        for _attempt in range(candidate_attempts):
            null_labels = np.full(len(active_nodes), -1, dtype=np.int64)
            target_inventory: list[np.ndarray] = []
            offset = 0
            for nodes, target_sizes in components:
                grown, assigned_targets = _grow_component(nodes, adjacency, target_sizes, rng)
                null_labels[nodes] = grown + offset
                target_inventory.append(assigned_targets)
                offset += len(target_sizes)
            targets = np.concatenate(target_inventory)
            actual = np.bincount(null_labels, minlength=len(targets))
            errors = np.abs(actual - targets)
            candidate = NullPartition(
                labels=null_labels,
                mean_absolute_size_error=float(errors.mean()),
                maximum_absolute_size_error=int(errors.max(initial=0)),
                exactly_matched_fraction=float(np.mean(errors == 0)),
            )
            if best is None or (
                candidate.mean_absolute_size_error,
                candidate.maximum_absolute_size_error,
            ) < (
                best.mean_absolute_size_error,
                best.maximum_absolute_size_error,
            ):
                best = candidate
        assert best is not None
        nulls.append(best)
    return tuple(nulls)
