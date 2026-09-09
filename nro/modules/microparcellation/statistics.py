"""Estimate local similarity, parcel connectivity, and quality statistics."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from nro.engine.images import load_surface_timeseries as load_functional

LOG = logging.getLogger(__name__)
FLOAT32_TINY = np.finfo(np.float32).tiny
CONNECTOME_POWER_INITIALIZATION_SEED = 0


@dataclass(frozen=True)
class LocalCorrelationResult:
    """Local edge correlations with spatial support and run-quality bookkeeping."""

    correlations: np.ndarray
    included_runs: tuple[tuple[Path, ...], ...]


@dataclass(frozen=True)
class ParcelCorrelationResult:
    """Parcel connectivity, validity, reliability, and accumulated quality statistics."""

    correlations: np.ndarray
    variance_preserved: float
    residual_sum_squares: float
    total_sum_squares: float
    null_variance_preserved: tuple[float, ...]
    null_residual_sum_squares: tuple[float, ...]
    parcel_reliability_mean: np.ndarray
    parcel_reliability_minimum: np.ndarray
    parcel_reliability_maximum: np.ndarray
    parcel_supporting_runs: np.ndarray
    parcel_effective_runs: np.ndarray
    run_contributions: tuple[dict[str, float | int], ...]
    split_half: dict[str, float | int | str | list[int] | None]
    connectome: dict[str, object]


def _log_run_progress(phase: str, completed: int, total: int) -> None:
    width = 20
    filled = round(width * completed / total) if total else width
    bar = "#" * filled + "." * (width - filled)
    message = f"{phase} progress [{bar}] {completed}/{total} runs"
    if sys.stderr.isatty():
        print(f"\r{message}", end="\n" if completed == total else "", file=sys.stderr, flush=True)
    else:
        LOG.info(message)


def _combine_vertex_moments(
    n: int, mean: np.ndarray, m2: np.ndarray, block: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """Combine temporal blocks using float64 moments to limit chunk-order error."""
    bn = block.shape[0]
    block = np.asarray(block, dtype=np.float64)
    bmean = block.mean(axis=0)
    centered = block - bmean
    bm2 = np.einsum("tv,tv->v", centered, centered)
    if n == 0:
        return bn, bmean, bm2
    new_n = n + bn
    delta = bmean - mean
    factor = n * bn / new_n
    return new_n, mean + delta * (bn / new_n), m2 + bm2 + delta * delta * factor


def _moments(data: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(data.shape[1], dtype=np.float64)
    m2 = np.zeros(data.shape[1], dtype=np.float64)
    n = 0
    for start in range(0, len(data), block_size):
        n, mean, m2 = _combine_vertex_moments(n, mean, m2, data[start : start + block_size])
    return mean, m2


def _valid_variance(m2: np.ndarray, sample_count: int) -> np.ndarray:
    """Reject zero, non-finite, and subnormal float32 sample variances."""
    threshold = np.float32(sample_count - 1) * FLOAT32_TINY
    return np.isfinite(m2) & (m2 > threshold)


def _split_half_connectome_summary(
    first: np.ndarray,
    second: np.ndarray,
    first_sum_vector: np.ndarray,
    second_sum_vector: np.ndarray,
    first_frames: int,
    second_frames: int,
) -> dict[str, float | int]:
    """Compare unique off-diagonal correlations without constructing them."""
    if first_frames:
        first_diagonal = np.maximum(
            np.diag(first) - np.square(first_sum_vector, dtype=np.float64) / first_frames,
            0.0,
        )
    else:
        first_diagonal = np.zeros(first.shape[0], dtype=np.float64)
    if second_frames:
        second_diagonal = np.maximum(
            np.diag(second) - np.square(second_sum_vector, dtype=np.float64) / second_frames,
            0.0,
        )
    else:
        second_diagonal = np.zeros(second.shape[0], dtype=np.float64)
    valid = (first_diagonal > 0) & (second_diagonal > 0)
    first_scale = np.sqrt(first_diagonal)
    second_scale = np.sqrt(second_diagonal)
    count = 0
    first_sum = second_sum = first_square = second_square = cross = 0.0
    absolute_difference = squared_difference = 0.0
    for row in range(1, first.shape[0]):
        keep = valid[:row] & valid[row]
        if not np.any(keep):
            continue
        first_cross = first[row, :row][keep] - (
            first_sum_vector[row] * first_sum_vector[:row][keep] / first_frames
        )
        second_cross = second[row, :row][keep] - (
            second_sum_vector[row] * second_sum_vector[:row][keep] / second_frames
        )
        first_values = first_cross / (first_scale[row] * first_scale[:row][keep])
        second_values = second_cross / (second_scale[row] * second_scale[:row][keep])
        difference = first_values - second_values
        count += int(first_values.size)
        first_sum += float(first_values.sum(dtype=np.float64))
        second_sum += float(second_values.sum(dtype=np.float64))
        first_square += float(np.square(first_values, dtype=np.float64).sum())
        second_square += float(np.square(second_values, dtype=np.float64).sum())
        cross += float(np.multiply(first_values, second_values, dtype=np.float64).sum())
        absolute_difference += float(np.abs(difference).sum(dtype=np.float64))
        squared_difference += float(np.square(difference, dtype=np.float64).sum())
    if count < 2:
        correlation = 0.0
    else:
        covariance = cross - first_sum * second_sum / count
        first_variance = first_square - first_sum * first_sum / count
        second_variance = second_square - second_sum * second_sum / count
        denominator = np.sqrt(max(first_variance * second_variance, 0.0))
        correlation = float(covariance / denominator) if denominator > 0 else 0.0
        correlation = float(np.clip(correlation, -1.0, 1.0))
    spearman_brown = 2.0 * correlation / (1.0 + correlation) if correlation > -1.0 else -1.0
    return {
        "comparable_edges": count,
        "edge_correlation": correlation,
        "spearman_brown_reliability": float(spearman_brown),
        "mean_absolute_difference": absolute_difference / count if count else 0.0,
        "root_mean_square_difference": float(np.sqrt(squared_difference / count)) if count else 0.0,
    }


def _histogram_quantile(counts: np.ndarray, probability: float) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    rank = int(np.ceil(probability * total) - 1)
    index = int(np.searchsorted(np.cumsum(counts), max(rank, 0), side="right"))
    return float((index - 127) / 127.0)


def _connectome_summary(
    correlations: np.ndarray,
    valid_diagonal: np.ndarray,
    *,
    power_iterations: int,
) -> dict[str, object]:
    """Summarize a dense correlation matrix with vectors and scalar accumulators."""
    histogram = np.zeros(255, dtype=np.int64)
    count = 0
    value_sum = value_square_sum = 0.0
    maximum_asymmetry = 0.0
    for row in range(1, correlations.shape[0]):
        values = correlations[row, :row]
        codes = np.rint(np.clip(values, -1.0, 1.0) * 127.0).astype(np.int16)
        histogram += np.bincount(codes + 127, minlength=255)
        count += int(values.size)
        value_sum += float(values.sum(dtype=np.float64))
        value_square_sum += float(np.square(values, dtype=np.float64).sum())
    for start in range(0, correlations.shape[0], 512):
        stop = min(start + 512, correlations.shape[0])
        maximum_asymmetry = max(
            maximum_asymmetry,
            float(
                np.max(
                    np.abs(correlations[start:stop] - correlations[:, start:stop].T),
                    initial=0.0,
                )
            ),
        )
    mean = value_sum / count if count else 0.0
    variance = max(value_square_sum / count - mean * mean, 0.0) if count else 0.0
    valid_count = int(np.count_nonzero(valid_diagonal))
    frobenius_square = float(valid_count + 2.0 * value_square_sum)
    participation_rank = (
        float(valid_count * valid_count / frobenius_square) if frobenius_square > 0 else 0.0
    )

    dominant_fraction = 0.0
    if valid_count:
        vector = np.zeros(correlations.shape[0], dtype=np.float64)
        vector[valid_diagonal] = np.random.default_rng(CONNECTOME_POWER_INITIALIZATION_SEED).normal(
            size=valid_count
        )
        vector /= np.linalg.norm(vector)
        for _ in range(power_iterations):
            product = correlations @ vector
            product[valid_diagonal] += vector[valid_diagonal]
            norm = np.linalg.norm(product)
            if not np.isfinite(norm) or norm == 0:
                break
            vector = product / norm
        product = correlations @ vector
        product[valid_diagonal] += vector[valid_diagonal]
        dominant_fraction = float(np.clip(max(vector @ product, 0.0) / valid_count, 0.0, 1.0))

    quantiles = {
        f"p{int(probability * 100):02d}": _histogram_quantile(histogram, probability)
        for probability in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    }
    return {
        "unique_edges": count,
        "off_diagonal_mean": mean,
        "off_diagonal_standard_deviation": float(np.sqrt(variance)),
        "encoded_quantiles": quantiles,
        "encoded_positive_fraction": float(histogram[128:].sum() / count) if count else 0.0,
        "encoded_negative_fraction": float(histogram[:127].sum() / count) if count else 0.0,
        "encoded_zero_fraction": float(histogram[127] / count) if count else 0.0,
        "encoded_saturation_fraction": float((histogram[0] + histogram[-1]) / count)
        if count
        else 0.0,
        "encoded_histogram": {
            "codes": list(range(-127, 128)),
            "counts": histogram.tolist(),
        },
        "maximum_asymmetry": maximum_asymmetry,
        "participation_ratio_rank": participation_rank,
        "dominant_eigenvalue_fraction": dominant_fraction,
        "power_iterations": power_iterations,
        "power_initialization_seed": CONNECTOME_POWER_INITIALIZATION_SEED,
    }


def _standardization_parameters(
    data: np.ndarray,
    active: np.ndarray,
    block_size: int,
    *,
    global_signal_regression: bool,
    node_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Fit optional runwise GSR, then return post-regression z-score parameters."""
    mean, raw_m2 = _moments(data, block_size)
    beta = np.zeros(data.shape[1], dtype=np.float32)
    global_signal = None
    residual_m2 = raw_m2
    if global_signal_regression:
        if not np.any(active):
            raise ValueError("Global signal regression requires at least one active vertex")
        if node_weights is None:
            global_signal = np.asarray(data[:, active].mean(axis=1), dtype=np.float32)
        else:
            node_weights = np.asarray(node_weights, dtype=np.float32)
            if node_weights.shape != (data.shape[1],):
                raise ValueError("Global-signal node weights do not match the spatial node count")
            active_weights = node_weights[active]
            weight_sum = float(active_weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                raise ValueError("Global-signal node weights must have positive finite mass")
            global_signal = np.asarray(
                data[:, active] @ (active_weights / np.float32(weight_sum)),
                dtype=np.float32,
            )
        global_signal -= global_signal.mean()
        global_ss = np.dot(global_signal, global_signal)
        if np.isfinite(global_ss) and global_ss > FLOAT32_TINY:
            covariance = np.zeros(data.shape[1], dtype=np.float64)
            for start in range(0, len(data), block_size):
                stop = min(start + block_size, len(data))
                covariance += (data[start:stop] - mean).T @ global_signal[start:stop]
            beta = covariance / global_ss
            residual_m2 = np.maximum(raw_m2 - covariance * beta, 0.0)
    valid = _valid_variance(residual_m2, len(data))
    inv_sd = np.zeros(data.shape[1], dtype=np.float32)
    inv_sd[valid] = np.sqrt(np.float32(len(data) - 1) / residual_m2[valid])
    return mean, beta, global_signal, inv_sd


def _standardized_block(
    block: np.ndarray,
    mean: np.ndarray,
    beta: np.ndarray,
    global_signal: np.ndarray | None,
    inv_sd: np.ndarray,
) -> np.ndarray:
    valid = inv_sd > 0
    standardized = np.zeros(block.shape, dtype=np.float32)
    if not np.any(valid):
        return standardized
    residual = block[:, valid] - mean[valid]
    if global_signal is not None:
        residual -= global_signal[:, None] * beta[valid]
    standardized[:, valid] = residual * inv_sd[valid]
    return standardized


def _quarter_standardized(
    data: np.ndarray,
    *,
    global_signal_regression: bool,
    node_weights: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    quarters = np.array_split(data, 4, axis=0)
    if any(len(quarter) < 2 for quarter in quarters):
        raise ValueError("Split-half reliability requires at least eight timepoints per run")
    standardized = []
    valid_quarters = []
    for quarter in quarters:
        mean, beta, global_signal, inv_sd = _standardization_parameters(
            quarter,
            np.ones(quarter.shape[1], dtype=bool),
            len(quarter),
            global_signal_regression=global_signal_regression,
            node_weights=node_weights,
        )
        z = _standardized_block(quarter, mean, beta, global_signal, inv_sd)
        valid = inv_sd > 0
        standardized.append(z)
        valid_quarters.append(valid)
    return standardized, valid_quarters


def _profile_reliability(
    first: np.ndarray,
    second: np.ndarray,
    *,
    vertex_block_size: int,
) -> np.ndarray:
    """Exact row-wise correlation of two connectomes via temporal dual matrices."""
    n_nodes = first.shape[1]
    if second.shape[1] != n_nodes:
        raise ValueError("Split connectomes have different node counts")
    first_ss = np.einsum("ti,ti->i", first, first)
    second_ss = np.einsum("ti,ti->i", second, second)
    common_valid = (
        np.isfinite(first_ss)
        & np.isfinite(second_ss)
        & (first_ss > FLOAT32_TINY)
        & (second_ss > FLOAT32_TINY)
    )
    valid_count = int(common_valid.sum())
    if valid_count < 3:
        return np.zeros(n_nodes, dtype=np.float32)
    first_u = np.divide(
        first,
        np.sqrt(first_ss)[None, :],
        out=np.zeros_like(first),
        where=first_ss[None, :] > 0,
    )
    second_u = np.divide(
        second,
        np.sqrt(second_ss)[None, :],
        out=np.zeros_like(second),
        where=second_ss[None, :] > 0,
    )
    first_u[:, ~common_valid] = 0.0
    second_u[:, ~common_valid] = 0.0

    first_sum = first_u.T @ first_u.sum(axis=1)
    second_sum = second_u.T @ second_u.sum(axis=1)
    first_kernel = first_u @ first_u.T
    second_kernel = second_u @ second_u.T
    cross_kernel = first_u @ second_u.T
    first_square = np.empty(n_nodes, dtype=np.float32)
    second_square = np.empty(n_nodes, dtype=np.float32)
    cross = np.empty(n_nodes, dtype=np.float32)
    for start in range(0, n_nodes, vertex_block_size):
        stop = min(start + vertex_block_size, n_nodes)
        first_block = first_u[:, start:stop]
        second_block = second_u[:, start:stop]
        first_square[start:stop] = np.einsum("ti,ti->i", first_block, first_kernel @ first_block)
        second_square[start:stop] = np.einsum(
            "ti,ti->i", second_block, second_kernel @ second_block
        )
        cross[start:stop] = np.einsum("ti,ti->i", first_block, cross_kernel @ second_block)

    first_diagonal = common_valid.astype(np.float32)
    second_diagonal = common_valid.astype(np.float32)
    first_sum -= first_diagonal
    second_sum -= second_diagonal
    first_square -= first_diagonal
    second_square -= second_diagonal
    cross -= first_diagonal * second_diagonal
    profile_size = np.float32(valid_count - 1)
    covariance = cross - first_sum * second_sum / profile_size
    first_variance = np.maximum(first_square - first_sum * first_sum / profile_size, 0.0)
    second_variance = np.maximum(second_square - second_sum * second_sum / profile_size, 0.0)
    denominator = np.sqrt(first_variance * second_variance)
    quality = np.divide(
        covariance,
        denominator,
        out=np.zeros(n_nodes, dtype=np.float32),
        where=denominator > 0,
    )
    quality[~common_valid] = 0.0
    return np.nan_to_num(np.clip(quality, 0.0, 1.0), copy=False)


def _vertex_reliability(
    data: np.ndarray,
    active: np.ndarray,
    *,
    global_signal_regression: bool,
    vertex_block_size: int,
    node_weights: np.ndarray | None = None,
) -> np.ndarray:
    quarters, valid_quarters = _quarter_standardized(
        data[:, active],
        global_signal_regression=global_signal_regression,
        node_weights=None if node_weights is None else node_weights[active],
    )
    first = np.concatenate((quarters[0], quarters[3]), axis=0)
    second = np.concatenate((quarters[1], quarters[2]), axis=0)
    reliable_vertices = np.logical_and.reduce(valid_quarters)
    first[:, ~reliable_vertices] = 0.0
    second[:, ~reliable_vertices] = 0.0
    active_quality = _profile_reliability(first, second, vertex_block_size=vertex_block_size)
    quality = np.zeros(data.shape[1], dtype=np.float32)
    quality[active] = active_quality
    return quality


def _parcel_membership(
    labels: np.ndarray, valid: np.ndarray, count: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    valid_labels = labels[valid]
    counts = np.bincount(valid_labels, minlength=count).astype(np.float32)
    membership = sparse.csr_matrix(
        (1.0 / counts[valid_labels], (np.flatnonzero(valid), valid_labels)),
        shape=(len(labels), count),
        dtype=np.float32,
    )
    return membership, counts > 0


def make_parcel_mean_loader(
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    load_run=None,
):
    """Return a run loader that averages original samples into current parcels.

    The sparse membership is retained in memory, while each run's parcel means
    are computed on demand and discarded by the caller after its statistics
    have been accumulated.
    """
    load_run = load_functional if load_run is None else load_run
    active_labels = labels[mask]
    count = int(active_labels.max()) + 1
    membership, valid_parcels = _parcel_membership(labels, mask, count)
    if not np.all(valid_parcels):
        raise ValueError("Parcel labels must be contiguous and nonempty")
    masses = np.bincount(active_labels, minlength=count).astype(np.float32)

    def load_parcel_run(path: tuple[Path, ...]) -> np.ndarray:
        data = load_run(path)
        if data.shape[1] != len(labels):
            raise ValueError(f"Spatial node count mismatch in {path}")
        return np.asarray((membership.T @ data.T).T, dtype=np.float32)

    return load_parcel_run, masses


def _parcel_reliability(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    global_signal_regression: bool,
    vertex_block_size: int,
) -> np.ndarray:
    count = int(labels.max()) + 1
    quarters, valid_quarters = _quarter_standardized(
        data, global_signal_regression=global_signal_regression
    )
    parcel_quarters = []
    valid_parcels = []
    for quarter, valid in zip(quarters, valid_quarters):
        membership, valid_parcel = _parcel_membership(labels, valid, count)
        parcel_quarters.append(np.asarray((membership.T @ quarter.T).T, dtype=np.float32))
        valid_parcels.append(valid_parcel)
    first = np.concatenate((parcel_quarters[0], parcel_quarters[3]), axis=0)
    second = np.concatenate((parcel_quarters[1], parcel_quarters[2]), axis=0)
    reliable_parcels = np.logical_and.reduce(valid_parcels)
    first[:, ~reliable_parcels] = 0.0
    second[:, ~reliable_parcels] = 0.0
    quality = _profile_reliability(first, second, vertex_block_size=vertex_block_size)
    return quality


def local_edge_correlations(
    files: tuple[tuple[Path, ...], ...],
    edges: np.ndarray,
    n_vertices: int,
    block_size: int,
    *,
    global_signal_regression: bool,
    reliability_vertex_block_size: int,
    reliability_weighting: bool = True,
    mask: np.ndarray | None = None,
    load_run=None,
    node_weights: np.ndarray | None = None,
    progress_label: str = "Streaming pass 1 (microparcellation)",
) -> LocalCorrelationResult:
    """Stream correlations for spatial graph edges."""
    load_run = load_functional if load_run is None else load_run
    edge_gram = np.zeros(len(edges), dtype=np.float64)
    vertex_diagonal = np.zeros(n_vertices, dtype=np.float64)
    active = np.ones(n_vertices, dtype=bool) if mask is None else mask
    total_runs = len(files)
    included_runs = []
    for run_index, path in enumerate(files, start=1):
        data = load_run(path)
        if len(data) < 2:
            raise ValueError(
                "Sidecar-qualified functional run has fewer than two retained "
                f"frames after masking: {path}"
            )
        if data.shape[1] != n_vertices:
            raise ValueError(f"Spatial node count mismatch in {path}")
        included_runs.append(path)
        mean, beta, global_signal, inv_sd = _standardization_parameters(
            data,
            active,
            block_size,
            global_signal_regression=global_signal_regression,
            node_weights=node_weights,
        )
        valid = inv_sd > 0
        valid_active = active & valid
        if reliability_weighting:
            quality = _vertex_reliability(
                data,
                active,
                global_signal_regression=global_signal_regression,
                vertex_block_size=reliability_vertex_block_size,
                node_weights=node_weights,
            )
        else:
            quality = np.ones(n_vertices, dtype=np.float32)
        quality[~valid_active] = 0.0
        edge_cross = np.zeros(len(edges), dtype=np.float64)
        vertex_ss = np.zeros(n_vertices, dtype=np.float64)
        for start in range(0, len(data), block_size):
            stop = min(start + block_size, len(data))
            standardized = _standardized_block(
                data[start:stop],
                mean,
                beta,
                None if global_signal is None else global_signal[start:stop],
                inv_sd,
            )
            edge_cross += np.einsum(
                "te,te->e",
                standardized[:, edges[:, 0]],
                standardized[:, edges[:, 1]],
                dtype=np.float64,
            )
            vertex_ss += np.einsum("tv,tv->v", standardized, standardized, dtype=np.float64)
        root_quality = np.sqrt(quality)
        edge_gram += root_quality[edges[:, 0]] * root_quality[edges[:, 1]] * edge_cross
        vertex_diagonal += quality * vertex_ss
        _log_run_progress(progress_label, run_index, total_runs)
    denominator = np.sqrt(vertex_diagonal[edges[:, 0]] * vertex_diagonal[edges[:, 1]])
    if not included_runs:
        raise ValueError("No sidecar-qualified functional runs were supplied")
    if not np.any(denominator > 0):
        raise ValueError("No reliable mesh-edge variance was observed")
    correlations = np.divide(
        edge_gram, denominator, out=np.zeros_like(edge_gram), where=denominator > 0
    )
    return LocalCorrelationResult(
        correlations=correlations.astype(np.float32),
        included_runs=tuple(included_runs),
    )


def parcel_correlations(
    files: tuple[tuple[Path, ...], ...],
    labels: np.ndarray,
    mask: np.ndarray,
    block_size: int,
    *,
    split_half_block_frames: int,
    global_signal_regression: bool,
    reliability_vertex_block_size: int,
    reliability_weighting: bool = True,
    connectome_power_iterations: int = 20,
    load_run=None,
    null_partitions: tuple[np.ndarray, ...] = (),
) -> ParcelCorrelationResult:
    """Stream connectivity and quality with an independent scientific split size.

    Single-run split blocks count retained frames, capped at half the run.
    Processing block sizes control memory use, not the split allocation.
    """
    if split_half_block_frames < 1:
        raise ValueError("split_half_block_frames must be positive")
    load_run = load_functional if load_run is None else load_run
    active_labels = labels[mask]
    count = int(active_labels.max()) + 1
    half_grams = [
        np.zeros((count, count), dtype=np.float32),
        np.zeros((count, count), dtype=np.float32),
    ]
    half_frames = [0, 0]
    half_sums = [
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
    ]
    half_runs: list[list[int]] = [[], []]
    split_block_size_used: int | None = None
    reliability_sum = np.zeros(count, dtype=np.float64)
    reliability_square_sum = np.zeros(count, dtype=np.float64)
    reliability_minimum = np.full(count, np.inf, dtype=np.float32)
    reliability_maximum = np.zeros(count, dtype=np.float32)
    supporting_runs = np.zeros(count, dtype=np.int32)
    run_contributions: list[dict[str, float | int]] = []
    null_labels = tuple(np.asarray(value, dtype=np.int64) for value in null_partitions)
    if any(value.shape != active_labels.shape for value in null_labels):
        raise ValueError("Null partition size does not match the active source nodes")
    if any(not np.array_equal(np.unique(value), np.arange(count)) for value in null_labels):
        raise ValueError("Null partitions must contain the fitted number of parcels")
    total_sum_squares = 0.0
    residual_sum_squares = 0.0
    null_residual_sum_squares = np.zeros(len(null_labels), dtype=np.float64)
    total_runs = len(files)
    for run_index, path in enumerate(files, start=1):
        data = load_run(path)
        if data.shape[1] != mask.size:
            raise ValueError(f"Spatial node count mismatch in {path}")
        if len(data) < 2:
            continue
        active_data = data[:, mask]
        active = np.ones(active_data.shape[1], dtype=bool)
        mean, beta, global_signal, inv_sd = _standardization_parameters(
            active_data,
            active,
            block_size,
            global_signal_regression=global_signal_regression,
        )
        valid_vertices = inv_sd > 0
        membership, valid_parcels = _parcel_membership(active_labels, valid_vertices, count)
        valid_counts = np.bincount(active_labels[valid_vertices], minlength=count).astype(
            np.float32
        )
        null_memberships = []
        for null in null_labels:
            null_membership, _null_valid_parcels = _parcel_membership(null, valid_vertices, count)
            null_memberships.append(
                (
                    null_membership,
                    np.bincount(null[valid_vertices], minlength=count).astype(np.float32),
                )
            )
        if reliability_weighting:
            quality = _parcel_reliability(
                active_data,
                active_labels,
                global_signal_regression=global_signal_regression,
                vertex_block_size=reliability_vertex_block_size,
            )
        else:
            quality = np.ones(count, dtype=np.float32)
        quality[~valid_parcels] = 0.0
        parcel_timecourses = np.empty((len(data), count), dtype=np.float32)
        for start in range(0, len(data), block_size):
            stop = min(start + block_size, len(data))
            standardized = _standardized_block(
                active_data[start:stop],
                mean,
                beta,
                None if global_signal is None else global_signal[start:stop],
                inv_sd,
            )
            parcel_block = np.asarray((membership.T @ standardized.T).T, dtype=np.float32)
            parcel_timecourses[start:stop] = parcel_block
            block_sum_squares = float(
                np.einsum("tv,tv->", standardized, standardized, dtype=np.float64)
            )
            preserved_sum_squares = float(
                np.einsum("tp,p,tp->", parcel_block, valid_counts, parcel_block, dtype=np.float64)
            )
            total_sum_squares += block_sum_squares
            residual_sum_squares += max(0.0, block_sum_squares - preserved_sum_squares)
            for null_index, (null_membership, null_valid_counts) in enumerate(null_memberships):
                null_block = np.asarray(
                    (null_membership.T @ standardized.T).T,
                    dtype=np.float32,
                )
                null_preserved_sum_squares = float(
                    np.einsum(
                        "tp,p,tp->",
                        null_block,
                        null_valid_counts,
                        null_block,
                        dtype=np.float64,
                    )
                )
                null_residual_sum_squares[null_index] += max(
                    0.0,
                    block_sum_squares - null_preserved_sum_squares,
                )
        parcel_timecourses -= parcel_timecourses.mean(axis=0)
        parcel_ss = np.einsum("tp,tp->p", parcel_timecourses, parcel_timecourses)
        valid_parcel_variance = _valid_variance(parcel_ss, len(data))
        quality[~valid_parcel_variance] = 0.0
        supported = valid_parcels & valid_parcel_variance
        reliability_sum += quality
        reliability_square_sum += np.square(quality, dtype=np.float64)
        reliability_minimum[supported] = np.minimum(
            reliability_minimum[supported], quality[supported]
        )
        reliability_maximum = np.maximum(reliability_maximum, quality)
        supporting_runs += supported.astype(np.int32)
        parcel_scale = np.zeros(count, dtype=np.float32)
        parcel_scale[valid_parcel_variance] = np.sqrt(
            np.float32(len(data) - 1) / parcel_ss[valid_parcel_variance]
        )
        parcel_timecourses *= parcel_scale[None, :]
        parcel_timecourses *= np.sqrt(quality)[None, :]
        parcel_timecourses[:, quality <= 0] = 0.0
        if not np.all(np.isfinite(parcel_timecourses)):
            raise ValueError(f"Non-finite weighted parcel timecourses in {path}")
        run_gram = parcel_timecourses.T @ parcel_timecourses
        run_trace = float(np.trace(run_gram, dtype=np.float64))
        run_frobenius = float(np.linalg.norm(run_gram))
        run_contributions.append(
            {
                "run": run_index,
                "retained_frames": len(data),
                "mean_parcel_reliability": float(quality.mean()),
                "median_parcel_reliability": float(np.median(quality)),
                "minimum_parcel_reliability": float(quality.min()),
                "maximum_parcel_reliability": float(quality.max()),
                "gram_trace": run_trace,
                "gram_frobenius_norm": run_frobenius,
            }
        )
        if total_runs > 1:
            half = 0 if half_frames[0] <= half_frames[1] else 1
            half_grams[half] += run_gram
            half_sums[half] += parcel_timecourses.sum(axis=0, dtype=np.float64)
            half_frames[half] += len(data)
            half_runs[half].append(run_index)
        else:
            split_block_size = min(split_half_block_frames, max(1, len(data) // 2))
            split_block_size_used = split_block_size
            for split_index, start in enumerate(range(0, len(data), split_block_size)):
                stop = min(start + split_block_size, len(data))
                half = split_index % 2
                block = parcel_timecourses[start:stop]
                half_grams[half] += block.T @ block
                half_sums[half] += block.sum(axis=0, dtype=np.float64)
                half_frames[half] += len(block)
        del run_gram
        _log_run_progress("Streaming pass 2 (connectivity and quality)", run_index, total_runs)
    split_half = _split_half_connectome_summary(
        *half_grams,
        *half_sums,
        *half_frames,
    )
    split_half.update(
        {
            "method": "whole runs" if total_runs > 1 else "alternating temporal blocks",
            "allocation": (
                "greedy retained-frame balance in input order"
                if total_runs > 1
                else "alternating contiguous retained-frame blocks"
            ),
            "temporal_block_size": split_block_size_used,
            "first_half_runs": half_runs[0],
            "second_half_runs": half_runs[1],
            "first_half_retained_frames": half_frames[0],
            "second_half_retained_frames": half_frames[1],
        }
    )
    gram = half_grams[0]
    second_gram = half_grams[1]
    gram += second_gram
    half_grams.clear()
    del second_gram
    if not np.all(np.isfinite(gram)):
        raise ValueError("Non-finite values accumulated in the parcel Gram matrix")
    diagonal = np.maximum(np.diag(gram), 0.0)
    valid_diagonal = diagonal > 0
    if not np.any(valid_diagonal):
        raise ValueError("No reliable parcel variance was observed")
    inverse_scale = np.zeros(count, dtype=np.float32)
    inverse_scale[valid_diagonal] = 1.0 / np.sqrt(diagonal[valid_diagonal])
    correlations = gram
    for start in range(0, count, 512):
        stop = min(start + 512, count)
        correlations[start:stop] *= inverse_scale[start:stop, None]
        correlations[start:stop] *= inverse_scale[None, :]
    np.fill_diagonal(correlations, 0.0)
    connectome = _connectome_summary(
        correlations,
        valid_diagonal,
        power_iterations=connectome_power_iterations,
    )
    if not np.isfinite(total_sum_squares) or total_sum_squares <= 0:
        raise ValueError("No finite source-resolution variance was observed")
    variance_preserved = 1.0 - residual_sum_squares / total_sum_squares
    null_scores = tuple(
        float(np.clip(1.0 - residual / total_sum_squares, 0.0, 1.0))
        for residual in null_residual_sum_squares
    )
    null_residuals = tuple(float(residual) for residual in null_residual_sum_squares)
    parcel_reliability_mean = np.divide(
        reliability_sum,
        supporting_runs,
        out=np.zeros(count, dtype=np.float64),
        where=supporting_runs > 0,
    ).astype(np.float32)
    reliability_minimum[~np.isfinite(reliability_minimum)] = 0.0
    parcel_effective_runs = np.divide(
        np.square(reliability_sum),
        reliability_square_sum,
        out=np.zeros(count, dtype=np.float64),
        where=reliability_square_sum > 0,
    ).astype(np.float32)
    total_trace = sum(float(record["gram_trace"]) for record in run_contributions)
    for record in run_contributions:
        record["diagonal_weight_fraction"] = (
            float(record["gram_trace"]) / total_trace if total_trace > 0 else 0.0
        )
    return ParcelCorrelationResult(
        correlations=correlations,
        variance_preserved=float(np.clip(variance_preserved, 0.0, 1.0)),
        residual_sum_squares=residual_sum_squares,
        total_sum_squares=total_sum_squares,
        null_variance_preserved=null_scores,
        null_residual_sum_squares=null_residuals,
        parcel_reliability_mean=parcel_reliability_mean,
        parcel_reliability_minimum=reliability_minimum,
        parcel_reliability_maximum=reliability_maximum,
        parcel_supporting_runs=supporting_runs,
        parcel_effective_runs=parcel_effective_runs,
        run_contributions=tuple(run_contributions),
        split_half=split_half,
        connectome=connectome,
    )
