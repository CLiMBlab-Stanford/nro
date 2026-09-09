"""Match repeated network partitions and derive a consensus partition."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def jaccard(a: set[int], b: set[int]) -> float:
    """Return the Jaccard overlap of two vertex sets."""
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def choose_reference(runs: list[list[set[int]]]) -> int:
    """Choose the replicate with greatest matched overlap to the others."""
    if len(runs) == 1:
        return 0
    scores = np.zeros(len(runs))
    for i, a in enumerate(runs):
        for j, b in enumerate(runs):
            if i == j or not a or not b:
                continue
            cost = -np.array([[jaccard(x, y) for y in b] for x in a])
            rr, cc = linear_sum_assignment(cost)
            scores[i] += float((-cost[rr, cc]).sum()) / max(len(a), len(b))
    return int(np.argmax(scores))


def membership_stability(
    runs: list[list[set[int]]], n_vertices: int, minimum_match: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Estimate matched membership, homeless, and overlap frequencies."""
    ref_idx = choose_reference(runs)
    reference = runs[ref_idx]
    k = len(reference)
    counts = np.zeros((n_vertices, k), dtype=np.float64)
    homeless = np.zeros(n_vertices, dtype=np.float64)
    overlap = np.zeros(n_vertices, dtype=np.float64)
    all_nodes = set(range(n_vertices))
    for groups in runs:
        memberships = np.zeros(n_vertices, dtype=np.int16)
        for g in groups:
            memberships[list(g & all_nodes)] += 1
        homeless += memberships == 0
        overlap += memberships > 1
        if not reference or not groups:
            continue
        score = np.array([[jaccard(a, b) for b in groups] for a in reference])
        rr, cc = linear_sum_assignment(-score)
        for r, c in zip(rr, cc):
            if score[r, c] >= minimum_match:
                counts[list(groups[c] & all_nodes), r] += 1
    scale = float(len(runs))
    return counts / scale, homeless / scale, overlap / scale, ref_idx
