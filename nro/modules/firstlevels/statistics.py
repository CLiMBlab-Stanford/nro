"""Compact GLS fits and linear summaries retaining original-run covariance."""

from dataclasses import dataclass

import numpy as np
from scipy import linalg


class UnidentifiableDesignError(ValueError):
    """The selected frames cannot identify the temporal model with residual DOF."""


@dataclass
class RunFit:
    """Coefficient estimates with shared/grouped normalized covariance.

    Columns of beta index spatial locations. Each location selects a covariance
    matrix through groups and multiplies it by its residual variance. DOF are
    conditional on the selected temporal correlation model.
    """

    beta: np.ndarray
    residual_variance: np.ndarray
    groups: np.ndarray
    covariance: np.ndarray
    dof: float
    ar_coefficients: np.ndarray

    def variance(self, coefficients: np.ndarray) -> np.ndarray:
        """Evaluate a linear combination's variance at each spatial location."""
        weights = np.asarray(coefficients, dtype=np.float64)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights[:, None], self.beta.shape)
        if weights.shape != self.beta.shape:
            raise ValueError("Contrast coefficient dimensions differ from run estimates")
        result = np.zeros(self.beta.shape[1], dtype=np.float64)
        for group, covariance in enumerate(self.covariance):
            selected = self.groups == group
            w = weights[:, selected]
            result[selected] = np.einsum("iv,ij,jv->v", w, covariance, w)
        return np.maximum(result, 0) * self.residual_variance


@dataclass
class Estimate:
    """A linear effect represented on independent original-run coefficients.

    Keys identify source fits, not intermediate sessions. Combining coefficients
    before evaluating variance preserves covariance through nested summaries.
    """

    coefficients: dict[str, np.ndarray]

    def evaluate(self, fits: dict[str, RunFit]) -> dict[str, np.ndarray]:
        """Return effect, variance, t and Satterthwaite DOF arrays.

        Undefined zero-variance locations receive NaN t/DOF in published maps.
        Noise groups and precision weights are treated as fixed.
        """
        if not self.coefficients:
            raise ValueError("An estimate needs at least one source run")
        n_locations = fits[next(iter(self.coefficients))].beta.shape[1]
        effect = np.zeros(n_locations)
        variance = np.zeros(n_locations)
        denominator = np.zeros(n_locations)
        for key, weights in self.coefficients.items():
            fit = fits[key]
            weights = np.asarray(weights)
            if weights.ndim == 1:
                weights = weights[:, None]
            effect += np.sum(weights * fit.beta, axis=0)
            contribution = fit.variance(weights if weights.shape[1] != 1 else weights[:, 0])
            variance += contribution
            denominator += contribution**2 / fit.dof
        valid = (variance > 0) & (denominator > 0) & np.isfinite(effect)
        dof = np.divide(variance**2, denominator, out=np.full(n_locations, np.nan), where=valid)
        t = np.divide(effect, np.sqrt(variance), out=np.full(n_locations, np.nan), where=valid)
        return {"effect": effect, "variance": variance, "t": t, "dof": dof}


def linear_combination(estimates: list[Estimate], weights: list[float | np.ndarray]) -> Estimate:
    """Combine effects, adding coefficients from shared runs before uncertainty."""
    if len(estimates) != len(weights) or not estimates:
        raise ValueError("Provide matching nonempty estimates and weights")
    combined: dict[str, np.ndarray] = {}
    for estimate, weight in zip(estimates, weights):
        for key, coefficients in estimate.coefficients.items():
            values = np.asarray(coefficients)
            if values.ndim == 1:
                values = values[:, None]
            contribution = values * np.asarray(weight)
            combined[key] = combined.get(key, 0) + contribution
    return Estimate(combined)


def aggregate(
    estimates: list[Estimate], fits: dict[str, RunFit], *, weighting: str = "equal"
) -> Estimate:
    """Pool available effects, rejecting overlapping source runs within a pool.

    Equal weights define an arithmetic mean. Precision weights use marginal
    estimated variances; subsequent inference conditions on those weights.
    """
    if not estimates:
        raise ValueError("Cannot aggregate an absent condition")
    keys = [key for estimate in estimates for key in estimate.coefficients]
    if len(set(keys)) != len(keys):
        raise ValueError("Aggregation would count a source run more than once")
    if weighting == "equal":
        weights = [1 / len(estimates)] * len(estimates)
    elif weighting == "precision":
        variances = np.array([estimate.evaluate(fits)["variance"] for estimate in estimates])
        valid = np.isfinite(variances) & (variances > 0)
        precision = np.divide(1, variances, out=np.zeros_like(variances), where=valid)
        total = precision.sum(axis=0)
        weights = list(np.divide(precision, total, out=np.zeros_like(precision), where=total > 0))
    else:
        raise ValueError(f"Unknown aggregation weighting: {weighting}")
    return linear_combination(estimates, weights)


def nuisance_components(
    task: np.ndarray,
    nuisance: np.ndarray,
    *,
    variance_target: float,
    minimum_rank: int,
    minimum_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Select nuisance PCs within the residual-rank budget of exact predictors.

    PCs are fitted jointly with task regressors, not made task-orthogonal. This
    retains adjusted task coefficients in their original model parameterization.
    """
    n = task.shape[0]
    exact_rank = int(np.linalg.matrix_rank(task)) if task.shape[1] else 0
    remaining = n - exact_rank
    if remaining <= 0:
        raise UnidentifiableDesignError("Task model leaves no residual temporal dimensions")
    if not 0 < variance_target <= 1 or minimum_rank < 0 or not 0 <= minimum_fraction <= 1:
        raise ValueError("Invalid nuisance rank protection settings")
    cap = max(0, remaining - max(minimum_rank, int(np.ceil(minimum_fraction * remaining))))
    scales = np.sqrt(np.mean(nuisance**2, axis=0))
    selected = scales > np.finfo(float).eps
    scaled = nuisance[:, selected] / scales[selected]
    if scaled.shape[1]:
        u, s, _ = linalg.svd(scaled, full_matrices=False)
        nonzero = s > max(scaled.shape) * np.finfo(float).eps * s[0]
        u, s = u[:, nonzero], s[nonzero]
        cumulative = np.cumsum(s**2) / np.sum(s**2)
        requested = int(np.searchsorted(cumulative, variance_target) + 1)
        count = min(requested, cap)
        components = u[:, :count]
        # Remove PCs redundant with the exact model; never perturb task columns.
        kept = []
        rank = exact_rank
        for column in components.T:
            candidate = np.column_stack((task, *kept, column))
            new_rank = np.linalg.matrix_rank(candidate)
            if new_rank > rank:
                kept.append(column)
                rank = new_rank
        components = np.column_stack(kept) if kept else np.empty((n, 0))
        explained = float(cumulative[count - 1]) if count else 0.0
    else:
        components = np.empty((n, 0))
        requested = 0
        explained = 0.0
    return components, {
        "ExactRank": int(exact_rank),
        "PostExactRank": remaining,
        "NuisancePCs": components.shape[1],
        "RequestedNuisancePCs": requested,
        "NuisanceVarianceExplained": explained,
        "NuisanceRankCap": cap,
    }


def fit_glm(
    data: np.ndarray,
    design: np.ndarray,
    *,
    retained: np.ndarray,
    ar_grid: np.ndarray,
    block_size: int = 2048,
) -> RunFit:
    """Fit OLS or finite-grid AR(1) REML on uncensored acquisition frames.

    Data have full-time rows and spatial columns; design contains retained rows.
    No temporal filter is applied. Censor gaps enter the AR covariance at their original
    integer lags. A singleton zero AR grid gives OLS. Inference is conditional on
    the estimated group; it is not an exact small-sample t law for estimated AR.
    """
    data, design = np.asarray(data), np.asarray(design, dtype=np.float64)
    retained = np.asarray(retained, dtype=bool)
    grid = np.asarray(ar_grid, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(retained) or not np.all(np.isfinite(data[retained])):
        raise ValueError("GLM requires finite, aligned retained data")
    if design.shape[0] != retained.sum():
        raise ValueError("Design and retained frames disagree")
    if grid.ndim != 1 or not len(grid) or np.any(~np.isfinite(grid) | (np.abs(grid) >= 1)):
        raise ValueError("AR(1) coefficients must lie strictly between -1 and 1")
    if design.shape[1] and np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Fit design must contain independent columns")
    dof = design.shape[0] - design.shape[1]
    if dof <= 0 or block_size < 1:
        raise ValueError("Fit needs positive residual DOF and block size")
    indices = np.flatnonzero(retained)
    lags = np.abs(indices[:, None] - indices[None, :])
    kernels, covariances = [], []
    for rho in grid:
        correlation = rho**lags
        chol = linalg.cholesky(correlation, lower=True)
        white_design = linalg.solve_triangular(chol, design, lower=True)
        q, r = linalg.qr(white_design, mode="economic")
        inverse_r = linalg.solve_triangular(r, np.eye(design.shape[1]))
        covariance = inverse_r @ inverse_r.T
        operator = inverse_r @ q.T
        penalty = 2 * (np.log(np.diag(chol)).sum() + np.log(np.abs(np.diag(r))).sum())
        kernels.append((chol, white_design, operator, penalty))
        covariances.append(covariance)
    beta = np.empty((design.shape[1], data.shape[1]), dtype=np.float32)
    residual_variance = np.empty(data.shape[1], dtype=np.float32)
    groups = np.empty(data.shape[1], dtype=np.int32)
    for start in range(0, data.shape[1], block_size):
        stop = min(start + block_size, data.shape[1])
        values = np.asarray(data[retained, start:stop], dtype=np.float64)
        best = np.full(stop - start, np.inf)
        for group, (chol, white_design, operator, penalty) in enumerate(kernels):
            white = linalg.solve_triangular(chol, values, lower=True)
            coefficients = operator @ white
            residual = white - white_design @ coefficients
            scale = np.sum(residual**2, axis=0) / dof
            objective = dof * np.log(np.maximum(scale, np.finfo(float).tiny)) + penalty
            take = objective < best
            best[take] = objective[take]
            beta[:, start:stop][:, take] = coefficients[:, take]
            residual_variance[start:stop][take] = scale[take]
            groups[start:stop][take] = group
    return RunFit(beta, residual_variance, groups, np.asarray(covariances), float(dof), grid)
