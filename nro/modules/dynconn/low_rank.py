"""Stream a spectral approximation of a location-wise correlation matrix."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

BlockFactory = Callable[[], Iterator[np.ndarray]]
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LowRankFit:
    """Factors and diagnostics for a rank-limited correlation approximation."""

    spatial_basis: np.ndarray
    component_transform: np.ndarray
    eigenvalues: np.ndarray
    input_frames: int
    valid_locations: int
    requested_dimensions: int
    realized_rank: int
    retained_variance_fraction: float
    random_seed: int

    @property
    def dimensions(self) -> int:
        """Return the requested output dimension represented by the factors."""

        return int(self.eigenvalues.size)

    @property
    def synthetic_frames(self) -> int:
        """Return the minimum centered sample count for these dimensions."""

        return self.dimensions + 1


def helmert_basis(dimensions: int) -> np.ndarray:
    """Return orthonormal, zero-mean contrasts with ``dimensions + 1`` columns."""

    basis = np.zeros((dimensions, dimensions + 1), dtype=np.float64)
    for row in range(dimensions):
        denominator = np.sqrt((row + 1) * (row + 2))
        basis[row, : row + 1] = 1.0 / denominator
        basis[row, row + 1] = -(row + 1) / denominator
    return basis


def _moments(
    blocks: BlockFactory,
    *,
    n_locations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    mean = np.zeros(n_locations, dtype=np.float64)
    sum_squares = np.zeros(n_locations, dtype=np.float64)
    count = 0
    for block in blocks():
        if block.ndim != 2 or block.shape[1] != n_locations:
            raise ValueError("Low-rank input blocks must share one spatial width")
        if not np.all(np.isfinite(block)):
            raise ValueError("Low-rank input contains non-finite values")
        block_count = int(block.shape[0])
        if not block_count:
            continue
        block_mean = np.mean(block, axis=0, dtype=np.float64)
        block_variance = np.var(block, axis=0, dtype=np.float64)
        if not count:
            mean[:] = block_mean
            sum_squares[:] = block_variance * block_count
            count = block_count
            continue
        updated_count = count + block_count
        difference = block_mean - mean
        sum_squares += (
            block_variance * block_count
            + difference * difference * count * block_count / updated_count
        )
        mean += difference * block_count / updated_count
        count = updated_count
    if count < 2:
        raise ValueError("Low-rank correlation requires at least two retained frames")
    return mean.astype(np.float32), np.sqrt(np.maximum(sum_squares, 0.0)).astype(np.float32), count


def _standardized_blocks(
    blocks: BlockFactory,
    mean: np.ndarray,
    norm: np.ndarray,
) -> Iterator[np.ndarray]:
    inverse = np.zeros_like(norm)
    valid = norm > 0
    inverse[valid] = 1.0 / norm[valid]
    for block in blocks():
        standardized = np.asarray(block, dtype=np.float32) - mean
        standardized *= inverse
        yield standardized


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return np.asarray(basis, dtype=np.float32)


def fit_low_rank_correlation(
    blocks: BlockFactory,
    *,
    n_locations: int,
    dimensions: int,
    oversampling: int,
    power_iterations: int,
    random_seed: int | None = None,
) -> LowRankFit:
    """Approximate a correlation matrix without constructing it.

    ``blocks`` must return a fresh iterator over the same retained frames on
    every call. Each yielded array has frames on rows and spatial locations on
    columns.
    """

    total_passes = power_iterations + 3
    LOG.info("Low-rank pass 1/%d: estimating spatial means and variances", total_passes)
    mean, norm, n_frames = _moments(blocks, n_locations=n_locations)
    valid_locations = int(np.count_nonzero(norm > 0))
    if not valid_locations:
        raise ValueError("Low-rank correlation has no spatial locations with temporal variance")
    maximum_rank = min(valid_locations, n_frames - 1)
    fitted_dimensions = min(dimensions, maximum_rank)
    if fitted_dimensions < dimensions:
        LOG.info(
            "Low-rank dimension ceiling reduced from %d to the maximum centered rank %d",
            dimensions,
            fitted_dimensions,
        )
    subspace_size = min(maximum_rank, fitted_dimensions + oversampling)
    seed_sequence = (
        np.random.SeedSequence() if random_seed is None else np.random.SeedSequence(random_seed)
    )
    effective_seed = int(seed_sequence.entropy)
    random = np.random.default_rng(seed_sequence)
    projection = np.zeros((n_locations, subspace_size), dtype=np.float32)
    LOG.info("Low-rank pass 2/%d: constructing the randomized spatial subspace", total_passes)
    for block in _standardized_blocks(blocks, mean, norm):
        weights = random.standard_normal((block.shape[0], subspace_size), dtype=np.float32)
        projection += block.T @ weights
    basis = _orthonormalize(projection)

    for iteration in range(power_iterations):
        LOG.info(
            "Low-rank pass %d/%d: refining the spatial subspace",
            iteration + 3,
            total_passes,
        )
        projection.fill(0)
        for block in _standardized_blocks(blocks, mean, norm):
            temporal_projection = block @ basis
            projection += block.T @ temporal_projection
        basis = _orthonormalize(projection)

    core = np.zeros((subspace_size, subspace_size), dtype=np.float64)
    LOG.info("Low-rank pass %d/%d: estimating the projected spectrum", total_passes, total_passes)
    for block in _standardized_blocks(blocks, mean, norm):
        temporal_projection = np.asarray(block @ basis, dtype=np.float64)
        core += temporal_projection.T @ temporal_projection
    eigenvalues, eigenvectors = np.linalg.eigh(core)
    order = np.argsort(eigenvalues)[::-1][:fitted_dimensions]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    transform = np.asarray(eigenvectors[:, order], dtype=np.float32)
    tolerance = (
        float(eigenvalues[0]) * max(n_locations, n_frames) * np.finfo(np.float32).eps
        if eigenvalues.size
        else 0.0
    )
    retained = float(np.clip(np.sum(eigenvalues) / valid_locations, 0.0, 1.0))
    return LowRankFit(
        spatial_basis=basis,
        component_transform=transform,
        eigenvalues=eigenvalues,
        input_frames=n_frames,
        valid_locations=valid_locations,
        requested_dimensions=int(dimensions),
        realized_rank=int(np.count_nonzero(eigenvalues > tolerance)),
        retained_variance_fraction=retained,
        random_seed=effective_seed,
    )


def pseudo_timeseries_block(
    fit: LowRankFit,
    start: int,
    stop: int,
) -> np.ndarray:
    """Construct synthetic samples for a contiguous block of locations."""

    factors = fit.spatial_basis[start:stop] @ fit.component_transform
    factors *= np.sqrt(fit.eigenvalues).astype(np.float32)
    samples = np.sqrt(fit.dimensions) * factors @ helmert_basis(fit.dimensions)
    return np.asarray(samples.T, dtype=np.float32)
