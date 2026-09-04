from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def _laplacian(w: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    degree = np.asarray(w.sum(axis=1)).ravel()
    return (sparse.diags(degree) - w).tocsr(), degree


def _smallest_eigenspace(lap: sparse.csr_matrix, k: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    n = lap.shape[0]
    k = min(max(1, k), n - 1)
    if n <= max(64, 2 * k + 1):
        values, vectors = np.linalg.eigh(lap.toarray())
        return values[:k], vectors[:, :k]
    max_degree = float(np.max(lap.diagonal(), initial=0.0))
    offset = max(2.0 * max_degree, 1.0)
    shifted = offset * sparse.eye(n, format="csr", dtype=np.float32) - lap
    shifted_values, vectors = eigsh(shifted, k=k, which="LM", tol=tol)
    order = np.argsort(offset - shifted_values)
    return (offset - shifted_values)[order], vectors[:, order]


def _initial_basis(lap: sparse.csr_matrix, k: int, tol: float) -> np.ndarray:
    values, vectors = _smallest_eigenspace(lap, k, tol)
    invsqrt = np.zeros_like(values)
    positive = values >= 1e-10
    invsqrt[positive] = values[positive] ** -0.5
    return vectors @ np.diag(invsqrt)


def _variation_edge_costs(
    w: sparse.csr_matrix,
    degree: np.ndarray,
    a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper = sparse.triu(w, k=1).tocoo()
    edges = np.column_stack((upper.row, upper.col)).astype(np.int64)
    # Exact vectorization of Loukas's two-node expression:
    # ||(P A_e)' L_e (P A_e)||_F
    #   = 0.5 (degree_i + degree_j) ||A_i - A_j||_2^2.
    delta = a[edges[:, 0]] - a[edges[:, 1]]
    costs = 0.5 * (degree[edges[:, 0]] + degree[edges[:, 1]]) * np.einsum("ij,ij->i", delta, delta)
    return edges, costs


def _greedy_minimum_matching(edges: np.ndarray, costs: np.ndarray, merges_needed: int, n: int) -> list[tuple[int, int]]:
    marked = np.zeros(n, dtype=bool)
    selected: list[tuple[int, int]] = []
    for idx in np.argsort(costs, kind="stable"):
        i, j = map(int, edges[idx])
        if marked[i] or marked[j]:
            continue
        marked[i] = marked[j] = True
        selected.append((i, j))
        if len(selected) == merges_needed:
            break
    return selected


def _coarsening_matrix(n: int, matching: list[tuple[int, int]]) -> sparse.csr_matrix:
    mate = np.full(n, -1, dtype=np.int64)
    for i, j in matching:
        mate[i], mate[j] = j, i
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    coarse = 0
    for i in range(n):
        if mate[i] >= 0 and mate[i] < i:
            continue
        if mate[i] >= 0:
            rows.extend((coarse, coarse))
            cols.extend((i, int(mate[i])))
            data.extend((2 ** -0.5, 2 ** -0.5))
        else:
            rows.append(coarse)
            cols.append(i)
            data.append(1.0)
        coarse += 1
    return sparse.csr_matrix((data, (rows, cols)), shape=(coarse, n), dtype=np.float32)


def _coarsen_weights(w: sparse.csr_matrix, c: sparse.csr_matrix) -> sparse.csr_matrix:
    normalized = c.tocsc()
    column_sums = np.asarray(normalized.sum(axis=0)).ravel()
    pinv = (normalized @ sparse.diags(np.divide(1.0, column_sums, out=np.zeros_like(column_sums), where=column_sums != 0))).T
    wc = (pinv.T @ w @ pinv).tocsr()
    wc.setdiag(0.0)
    wc.eliminate_zeros()
    return ((wc + wc.T) * 0.5).tocsr()


def loukas_variation_edges(
    n_vertices: int,
    edges: np.ndarray,
    similarity: np.ndarray,
    mask: np.ndarray,
    target: int,
    *,
    k: int,
    max_levels: int,
    eigensolver_tolerance: float,
) -> np.ndarray:
    """Loukas local-variation coarsening using the edge-based family.

    The graph contains only spatial-neighbor edges. Every contraction and every
    lifted supervertex is therefore spatially connected. This follows the
    multilevel ``variation_edges`` algorithm from Loukas (JMLR 2019), with its
    greedy matching option.
    """
    active = np.flatnonzero(mask)
    if not 1 <= target <= len(active):
        raise ValueError("target must be between 1 and the number of active vertices")
    if target == len(active):
        result = np.full(n_vertices, -1, dtype=np.int64)
        result[active] = np.arange(len(active))
        return result
    compact = np.full(n_vertices, -1, dtype=np.int64)
    compact[active] = np.arange(len(active))
    keep = mask[edges[:, 0]] & mask[edges[:, 1]]
    compact_edges = compact[edges[keep]]
    values = np.asarray(similarity[keep], dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Loukas similarity weights must be finite and positive")
    w = sparse.coo_matrix(
        (np.concatenate((values, values)),
         (np.concatenate((compact_edges[:, 0], compact_edges[:, 1])),
          np.concatenate((compact_edges[:, 1], compact_edges[:, 0])))),
        shape=(len(active), len(active)),
    ).tocsr()
    w.sum_duplicates()
    w.setdiag(0.0)
    w.eliminate_zeros()
    lap, _ = _laplacian(w)
    basis = _initial_basis(lap, k, eigensolver_tolerance)
    total_c = sparse.eye(len(active), format="csr", dtype=np.float32)

    for level in range(max_levels):
        n = w.shape[0]
        if n == target:
            break
        lap, degree = _laplacian(w)
        if level == 0:
            a = basis
        else:
            reduced_basis = level_c @ basis
            d, v = np.linalg.eigh(reduced_basis.T @ (lap @ reduced_basis))
            invsqrt = np.zeros_like(d)
            positive = d >= 1e-10
            invsqrt[positive] = d[positive] ** -0.5
            a = reduced_basis @ np.diag(invsqrt) @ v
            basis = reduced_basis
        candidate_edges, costs = _variation_edge_costs(w, degree, a)
        if not len(candidate_edges):
            break
        merges = min(n - target, n // 2)
        matching = _greedy_minimum_matching(candidate_edges, costs, merges, n)
        if not matching:
            break
        level_c = _coarsening_matrix(n, matching)
        total_c = level_c @ total_c
        w = _coarsen_weights(w, level_c)
    if w.shape[0] != target:
        raise ValueError(
            f"Loukas coarsening stopped at {w.shape[0]} vertices rather than {target}; "
            "increase max_levels or ensure the spatial graph is sufficiently connected"
        )
    membership = total_c.tocsc()
    labels_active = np.empty(len(active), dtype=np.int64)
    for fine in range(len(active)):
        labels_active[fine] = int(membership[:, fine].indices[0])
    result = np.full(n_vertices, -1, dtype=np.int64)
    result[active] = labels_active
    return result
