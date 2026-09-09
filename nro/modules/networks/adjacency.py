"""Build sparse weighted graphs from microparcel connectivity."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from nro.engine.cifti import PCONN_INT8_SCALE

LOG = logging.getLogger(__name__)


def transform_weights(weights: np.ndarray, method: str) -> np.ndarray:
    """Apply the configured nonnegative connectivity-weight transform."""
    if method == "clip_positive":
        return np.maximum(weights, 0.0)
    if method == "absolute":
        return np.abs(weights)
    if method == "square":
        return np.square(weights)
    raise ValueError(f"Unsupported connectivity transform: {method}")


def lower_triangular_adjacency(
    weights: np.ndarray,
    minimum_weight: float,
    percentile_cutoff: float | None,
) -> tuple[sparse.csr_matrix, float | None]:
    """Threshold a symmetric dense matrix in place and return its sparse lower triangle."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix")
    if percentile_cutoff is not None and not 0 <= percentile_cutoff <= 100:
        raise ValueError("percentile_cutoff must lie in [0, 100]")

    percentile_weight = None
    if percentile_cutoff is not None:
        diagonal = np.diag_indices_from(weights)
        weights[diagonal] = np.nan
        percentile_weight = float(np.nanpercentile(weights, percentile_cutoff))
        weights[diagonal] = 0.0

    for row_index in range(weights.shape[0]):
        row = weights[row_index, :row_index]
        keep = np.isfinite(row) & (row > minimum_weight)
        if percentile_weight is not None:
            keep &= row >= percentile_weight
        row[~keep] = 0.0
        weights[row_index, row_index:] = 0.0

    return sparse.csr_matrix(weights, dtype=np.float32), percentile_weight


def _histogram_percentile(
    values: np.ndarray,
    counts: np.ndarray,
    percentile: float,
) -> float:
    """Match NumPy's linear percentile using a compact value histogram."""
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values[order], dtype=np.float64)
    ordered_counts = np.asarray(counts[order], dtype=np.int64)
    keep = ordered_counts > 0
    ordered_values = ordered_values[keep]
    ordered_counts = ordered_counts[keep]
    if not len(ordered_values):
        raise ValueError("Cannot calculate a percentile from an empty pconn")
    total = int(ordered_counts.sum())
    rank = (total - 1) * percentile / 100.0
    lower_rank = math.floor(rank)
    upper_rank = math.ceil(rank)
    cumulative = np.cumsum(ordered_counts)
    lower_value = ordered_values[np.searchsorted(cumulative, lower_rank, side="right")]
    upper_value = ordered_values[np.searchsorted(cumulative, upper_rank, side="right")]
    return float(lower_value + (rank - lower_rank) * (upper_value - lower_value))


def _transformed_codes(method: str, *, slope: float, intercept: float) -> np.ndarray:
    values = np.arange(-128, 128, dtype=np.float64) * slope + intercept
    return np.asarray(transform_weights(values, method), dtype=np.float32)


def pconn_to_adjacency(
    path: Path,
    *,
    transform: str,
    minimum_weight: float,
    percentile_cutoff: float | None,
    block_size: int = 512,
) -> tuple[sparse.csc_matrix, float | None]:
    """Stream a quantized microparcel pconn into a sparse lower triangle.

    The first pass builds exact per-column histograms, allowing percentile
    selection without materializing or sorting the dense matrix. The second
    pass writes retained edges directly into CSC storage.
    """
    import nibabel as nib

    image = nib.load(str(path))
    axes = (image.header.get_axis(0), image.header.get_axis(1))
    if not all(isinstance(axis, nib.cifti2.ParcelsAxis) for axis in axes):
        raise ValueError(f"Expected parcel-by-parcel CIFTI: {path}")
    if axes[0] != axes[1] or image.shape[0] != image.shape[1]:
        raise ValueError(f"Microparcel connectivity axes differ or are not square: {path}")
    proxy = image.dataobj
    if np.dtype(image.get_data_dtype()) != np.dtype(np.int8):
        raise ValueError(
            f"Networks requires the quantized int8 pconn written by microparcellation: {path}"
        )
    slope = float(proxy.slope if proxy.slope is not None else 1.0)
    intercept = float(proxy.inter if proxy.inter is not None else 0.0)
    if not np.isclose(slope, 1.0 / PCONN_INT8_SCALE) or not np.isclose(intercept, 0.0):
        raise ValueError(
            f"Unexpected pconn scaling in {path}: slope={slope}, intercept={intercept}"
        )
    raw = proxy.get_unscaled()
    size = int(image.shape[0])
    column_histograms = np.zeros((size, 256), dtype=np.uint32)
    blocks = math.ceil(size / block_size)
    LOG.info(
        "Scanning quantized %d x %d pconn in %d column blocks for the edge threshold",
        size,
        size,
        blocks,
    )
    for block_index, start in enumerate(range(0, size, block_size), start=1):
        stop = min(start + block_size, size)
        block = np.asarray(raw[:, start:stop], dtype=np.int8)
        for local, column in enumerate(range(start, stop)):
            codes = block[column + 1 :, local].astype(np.int16) + 128
            column_histograms[column] = np.bincount(codes, minlength=256)
        if block_index == 1 or block_index == blocks or block_index % max(1, blocks // 10) == 0:
            LOG.info("Connectivity threshold pass: %d/%d blocks", block_index, blocks)

    # The axes and microparcellation writer establish symmetry. Sample it
    # independently so corruption is still caught without rereading both full
    # triangles.
    sample_columns = np.unique(np.linspace(0, max(0, size - 2), min(64, size), dtype=int))
    for column in sample_columns:
        below = np.asarray(raw[column + 1 :, column], dtype=np.int8)
        across = np.asarray(raw[column, column + 1 :], dtype=np.int8)
        if not np.array_equal(below, across):
            raise ValueError(
                f"Microparcel connectivity is not symmetric near column {column}: {path}"
            )

    transformed = _transformed_codes(transform, slope=slope, intercept=intercept)
    raw_counts = column_histograms.sum(axis=0, dtype=np.int64) * 2
    percentile_weight = None
    if percentile_cutoff is not None:
        percentile_weight = _histogram_percentile(transformed, raw_counts, percentile_cutoff)
    retained_codes = transformed > minimum_weight
    if percentile_weight is not None:
        retained_codes &= transformed >= percentile_weight
    counts_per_column = column_histograms[:, retained_codes].sum(axis=1, dtype=np.int64)
    indptr = np.empty(size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts_per_column, out=indptr[1:])
    edge_count = int(indptr[-1])
    LOG.info(
        "Allocating sparse adjacency for %d retained edges (%.3f%% density)",
        edge_count,
        100.0 * edge_count / (size * (size - 1) // 2) if size > 1 else 0.0,
    )
    indices = np.empty(edge_count, dtype=np.int32)
    data = np.empty(edge_count, dtype=np.float32)
    offsets = indptr[:-1].copy()
    for block_index, start in enumerate(range(0, size, block_size), start=1):
        stop = min(start + block_size, size)
        block = np.asarray(raw[:, start:stop], dtype=np.int8)
        for local, column in enumerate(range(start, stop)):
            raw_values = block[column + 1 :, local]
            codes = raw_values.astype(np.int16) + 128
            keep = retained_codes[codes]
            count = int(np.count_nonzero(keep))
            offset = int(offsets[column])
            indices[offset : offset + count] = np.flatnonzero(keep) + column + 1
            data[offset : offset + count] = transformed[codes[keep]]
            offsets[column] += count
        if block_index == 1 or block_index == blocks or block_index % max(1, blocks // 10) == 0:
            LOG.info("Sparse adjacency pass: %d/%d blocks", block_index, blocks)
    return sparse.csc_matrix((data, indices, indptr), shape=(size, size)), percentile_weight


def save_adjacency(path: Path, adjacency: sparse.spmatrix) -> Path:
    """Save a sparse adjacency in canonical strict-lower-triangle form."""
    path = Path(path)
    sparse.save_npz(path, sparse.tril(adjacency, k=-1, format="csc"), compressed=True)
    return path


def load_adjacency(path: Path) -> sparse.csc_matrix:
    """Load and validate a canonical sparse adjacency."""
    adjacency = sparse.load_npz(path).tocsc()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Saved adjacency is not square: {path}")
    if sparse.triu(adjacency, k=0).nnz:
        raise ValueError(f"Saved adjacency is not strictly lower triangular: {path}")
    return adjacency
