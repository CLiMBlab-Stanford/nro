"""Construct and apply the run-level cleaning model."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from nro.engine.bids import bids_entity, replace_bids_entity_token
from nro.engine.execution import new_step_counter
from nro.engine.images import (
    gifti_vertex_count,
    load_gifti_timeseries,
    save_gifti_timeseries,
    sidecar_json_path,
)
from nro.engine.io import read_json, write_json
from nro.engine.targets import is_fsaverage_space, is_surface_space
from nro.engine.templates import find_fsaverage_surface

next_step = new_step_counter()


def _smoothed_desc_label(input_desc: str) -> str:
    return (
        "desc-smoothedPreCleanNoAROMA"
        if input_desc == "desc-preprocNoAROMA"
        else "desc-smoothedPreClean"
    )


def _space_name(path: Path) -> str:
    return bids_entity(path, "space", default="unknown") or "unknown"


def _resolve_surface_for_metric(
    *,
    metric_path: Path,
    metric_sidecar: dict[str, Any],
    anat_outputs: dict[str, Any],
    templateflow_root: Path,
    fsaverage_cache: dict[tuple[str, int], Path],
) -> Path:
    hemi_match = re.search(r"_hemi-([LR])_", metric_path.name)
    if hemi_match is None:
        raise SystemExit(f"Could not infer hemisphere from surface metric path: {metric_path}")
    hemi = hemi_match.group(1)
    space = _space_name(metric_path)
    if space == "fsnative":
        manifest_key = "lh.midthickness" if hemi == "L" else "rh.midthickness"
        raw = str((anat_outputs.get("surfaces") or {}).get(manifest_key) or "").strip()
        surf = Path(raw)
        if not surf.exists():
            raise SystemExit(
                f"Missing fsnative midthickness surface in anatomical manifest: surfaces.{manifest_key}"
            )
        return surf
    if is_fsaverage_space(space):
        n_vertices = gifti_vertex_count(metric_path)
        key = (hemi, n_vertices)
        if key not in fsaverage_cache:
            try:
                fsaverage_cache[key] = find_fsaverage_surface(
                    hemi=hemi,
                    surface="midthickness",
                    n_vertices=n_vertices,
                    roots=(templateflow_root,),
                )
            except FileNotFoundError as error:
                raise SystemExit(str(error)) from error
        return fsaverage_cache[key]
    sources = metric_sidecar.get("Sources") or []
    for source in sources:
        source_path = Path(str(source))
        if source_path.name.endswith("_midthickness.surf.gii") and source_path.exists():
            return source_path
    raise SystemExit(f"Unsupported surface space for smoothing: {metric_path}")


def _build_task_regressors(
    *,
    events_path: Path,
    n_scans: int,
    tr: float,
    start_time: float,
) -> tuple[list[pd.DataFrame], dict[str, dict[str, Any]]]:
    try:
        from nilearn import glm
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency for event regression: nilearn ({e})")

    if not events_path.exists():
        return [], {}
    events = pd.read_csv(events_path, sep="\t")
    required = {"onset", "duration"}
    if not required.issubset(events.columns):
        return [], {}
    if "trial_type" not in events.columns:
        events["trial_type"] = "task"
    events = events[["trial_type", "onset", "duration"]].copy()
    dummies = pd.get_dummies(events["trial_type"], prefix="task", prefix_sep=".")
    events = pd.concat([events, dummies], axis=1)
    frame_times = np.arange(n_scans, dtype=np.float64) * float(tr) + float(start_time)
    out: list[pd.DataFrame] = []
    sidecar: dict[str, dict[str, Any]] = {}
    for col in dummies.columns:
        vals, _names = glm.first_level.compute_regressor(
            (events["onset"].to_numpy(), events["duration"].to_numpy(), events[col].to_numpy()),
            "spm",
            frame_times,
        )
        out.append(pd.DataFrame(vals, columns=[col]))
        sidecar[col] = {"Description": f"Task regressor for trial type {col}"}
    return out, sidecar


def _filter_confounds(
    *,
    confounds_tsv: Path,
    confounds_json: Path,
    regex: str,
    events_path: Path,
    n_scans: int,
    tr: float,
    start_time: float,
    regress_out_task: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    confounds = pd.read_csv(confounds_tsv, sep="\t")
    sidecar = read_json(confounds_json) if confounds_json.exists() else {}
    selected = confounds.filter(regex=regex).fillna(0.0)
    selected_sidecar = {k: v for k, v in sidecar.items() if k in selected.columns}
    task = pd.DataFrame(index=np.arange(n_scans))
    if regress_out_task:
        task_regs, task_sidecar = _build_task_regressors(
            events_path=events_path,
            n_scans=n_scans,
            tr=tr,
            start_time=start_time,
        )
        if task_regs:
            task = pd.concat(task_regs, axis=1)
            selected_sidecar.update(task_sidecar)
    return selected, task.fillna(0.0), selected_sidecar


def _select_outlier_columns(
    *,
    confounds_tsv: Path,
    regex: str,
    n_scans: int,
) -> pd.DataFrame:
    confounds = pd.read_csv(confounds_tsv, sep="\t")
    if len(confounds) != int(n_scans):
        raise ValueError(
            f"Confounds rows do not match functional timepoints: "
            f"{len(confounds)} != {n_scans} ({confounds_tsv})"
        )
    return confounds.filter(regex=regex).fillna(0.0)


def _fourier_passband_basis(
    *,
    n_scans: int,
    tr: float,
    high_pass: Optional[float],
    low_pass: Optional[float],
) -> tuple[np.ndarray, list[str], list[float]]:
    """Build a real Fourier basis for frequencies inside the passband."""
    if n_scans < 2:
        return np.empty((n_scans, 0), dtype=np.float64), [], []
    nyquist = 0.5 / float(tr)
    if high_pass is not None and not 0.0 <= float(high_pass) < nyquist:
        raise ValueError(f"high_pass must be in [0, {nyquist:g}), got {high_pass}")
    if low_pass is not None and not 0.0 < float(low_pass) <= nyquist:
        raise ValueError(f"low_pass must be in (0, {nyquist:g}], got {low_pass}")
    if high_pass is not None and low_pass is not None and float(high_pass) >= float(low_pass):
        raise ValueError(f"high_pass must be below low_pass: {high_pass} >= {low_pass}")
    if high_pass is None and low_pass is None:
        return np.empty((n_scans, 0), dtype=np.float64), [], []

    times = np.arange(n_scans, dtype=np.float64) * float(tr)
    frequencies = np.fft.rfftfreq(n_scans, d=float(tr))
    columns: list[np.ndarray] = []
    names: list[str] = []
    included_frequencies: list[float] = []
    for index, frequency in enumerate(frequencies):
        include = (high_pass is None or frequency >= float(high_pass)) and (
            low_pass is None or frequency <= float(low_pass)
        )
        if not include:
            continue
        if index == 0:
            columns.append(np.ones(n_scans, dtype=np.float64))
            names.append("passband_dc")
            included_frequencies.append(float(frequency))
            continue
        angle = 2.0 * np.pi * frequency * times
        cosine = np.cos(angle)
        columns.append(cosine)
        names.append(f"passband_cos_{frequency:.12g}Hz")
        included_frequencies.append(float(frequency))
        sine = np.sin(angle)
        if np.linalg.norm(sine) > np.finfo(np.float64).eps * n_scans:
            columns.append(sine)
            names.append(f"passband_sin_{frequency:.12g}Hz")
    if not columns:
        return np.empty((n_scans, 0), dtype=np.float64), [], []
    return np.column_stack(columns), names, included_frequencies


@dataclass(frozen=True)
class _CleaningProjection:
    cleaning_defined: bool
    undefined_reason: str | None
    final_basis: np.ndarray | None
    retained: np.ndarray
    design: np.ndarray
    design_pseudoinverse: np.ndarray
    passband_dimension: int
    passband_rank: int
    passband_condition_number: float | None
    exact_design_rank: int
    post_exact_temporal_rank: int
    regression_design_rank: int
    nuisance_input_columns: int
    nuisance_usable_columns: int
    nuisance_pca_components: int
    nuisance_variance_target: float
    nuisance_variance_explained: float
    nuisance_variance_target_reached: bool
    nuisance_selection_limited_by_rank: bool
    minimum_temporal_rank: int
    minimum_temporal_rank_fraction: float
    protected_temporal_rank: int
    temporal_rank_floor_satisfied: bool
    algebraic_temporal_rank: int
    standardize: bool

    def transform(self, data: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
        """Fit retained rows and evaluate the clean temporal model at all rows."""
        values = np.asarray(data)
        n_scans = len(self.retained)
        if values.ndim != 2 or values.shape[0] != n_scans:
            raise ValueError(
                f"Expected time-by-feature data with {n_scans} rows, got {values.shape}"
            )
        output = np.empty(values.shape, dtype=np.float32)
        if not self.cleaning_defined:
            output.fill(0.0)
            return output
        for start in range(0, values.shape[1], int(chunk_size)):
            stop = min(start + int(chunk_size), values.shape[1])
            chunk = np.asarray(values[:, start:stop], dtype=np.float64)
            if not np.all(np.isfinite(chunk)):
                raise ValueError("Functional samples contain non-finite values")
            retained_chunk = chunk[self.retained, :]
            if self.final_basis is None:
                coefficients = self.design_pseudoinverse @ retained_chunk
                cleaned = chunk - self.design @ coefficients
            else:
                retained_basis = self.final_basis[self.retained, :]
                cleaned = self.final_basis @ (retained_basis.T @ retained_chunk)
            if self.standardize:
                retained_cleaned = cleaned[self.retained, :]
                means = retained_cleaned.mean(axis=0)
                scales = retained_cleaned.std(axis=0, ddof=0)
                scales[scales <= np.finfo(np.float64).eps] = 1.0
                cleaned = (cleaned - means) / scales
            output[:, start:stop] = cleaned.astype(np.float32)
        return output

    def metadata(self, *, tr: float, outlier_columns: int) -> dict[str, object]:
        """Summarize the fitted cleaning projection for its output sidecar."""
        censored = ~self.retained
        longest = 0
        current = 0
        for value in censored:
            current = current + 1 if value else 0
            longest = max(longest, current)
        return {
            "CleaningDefined": bool(self.cleaning_defined),
            "CleaningUndefinedReason": self.undefined_reason,
            "OutputDataStatus": (
                "cleaned_time_series" if self.cleaning_defined else "all_zero_undefined_sentinel"
            ),
            "TotalFrames": int(len(self.retained)),
            "RetainedFrames": int(self.retained.sum()),
            "CensoredFrames": int(censored.sum()),
            "CensoredFraction": float(censored.mean()),
            "RetainedDurationSeconds": float(self.retained.sum() * tr),
            "LongestCensoredIntervalFrames": int(longest),
            "LongestCensoredIntervalSeconds": float(longest * tr),
            "TemporalMaskColumnCount": int(outlier_columns),
            "PassbandBasisDimension": int(self.passband_dimension),
            "PassbandBasisRank": int(self.passband_rank),
            "PassbandBasisConditionNumber": self.passband_condition_number,
            "ExactDesignRank": int(self.exact_design_rank),
            "PostExactTemporalRank": int(self.post_exact_temporal_rank),
            "RegressionDesignRank": int(self.regression_design_rank),
            "ResidualDesignDegreesOfFreedom": int(
                self.retained.sum() - self.regression_design_rank
            ),
            "NuisanceInputColumnCount": int(self.nuisance_input_columns),
            "NuisanceUsableColumnCount": int(self.nuisance_usable_columns),
            "NuisancePCAComponentCount": int(self.nuisance_pca_components),
            "NuisancePCAVarianceTarget": float(self.nuisance_variance_target),
            "NuisancePCAVarianceExplained": float(self.nuisance_variance_explained),
            "NuisancePCAVarianceTargetReached": bool(self.nuisance_variance_target_reached),
            "NuisanceSelectionLimitedByTemporalRank": bool(self.nuisance_selection_limited_by_rank),
            "MinimumTemporalRank": int(self.minimum_temporal_rank),
            "MinimumTemporalRankFraction": float(self.minimum_temporal_rank_fraction),
            "ProtectedTemporalRank": int(self.protected_temporal_rank),
            "TemporalRankFloorSatisfied": bool(self.temporal_rank_floor_satisfied),
            "AlgebraicTemporalRank": int(self.algebraic_temporal_rank),
        }


def _build_cleaning_projection(
    *,
    confounds: pd.DataFrame,
    task: pd.DataFrame | None = None,
    outliers: pd.DataFrame,
    tr: float,
    detrend: bool,
    standardize: bool,
    high_pass: Optional[float],
    low_pass: Optional[float],
    nuisance_variance_explained: float = 0.99,
    minimum_temporal_rank: int = 30,
    minimum_temporal_rank_fraction: float = 0.5,
) -> _CleaningProjection:
    n_scans = len(confounds)
    if len(outliers) != n_scans:
        raise ValueError(f"Outlier rows do not match confound rows: {len(outliers)} != {n_scans}")
    if outliers.empty:
        retained = np.ones(n_scans, dtype=bool)
    else:
        retained = ~np.any(outliers.to_numpy(dtype=np.float64) != 0.0, axis=1)
    n_retained = int(retained.sum())
    if n_retained == 0:
        raise ValueError("Temporal mask censors every frame")
    target = float(nuisance_variance_explained)
    if not 0.0 < target <= 1.0:
        raise ValueError(
            f"nuisance_variance_explained must be in (0, 1], got {nuisance_variance_explained}"
        )
    minimum_rank = int(minimum_temporal_rank)
    minimum_fraction = float(minimum_temporal_rank_fraction)
    if minimum_rank < 0:
        raise ValueError(f"minimum_temporal_rank must be nonnegative, got {minimum_rank}")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError(
            f"minimum_temporal_rank_fraction must be in [0, 1], got {minimum_fraction}"
        )
    task = task if task is not None else pd.DataFrame(index=np.arange(n_scans))
    if len(task) != n_scans:
        raise ValueError(f"Task rows do not match confound rows: {len(task)} != {n_scans}")

    exact_candidates: list[np.ndarray] = [np.ones(n_scans, dtype=np.float64)]
    if detrend:
        exact_candidates.append(np.linspace(-1.0, 1.0, n_scans, dtype=np.float64))
    exact_candidates.extend(task[column].to_numpy(dtype=np.float64) for column in task.columns)

    passband, _passband_names, _frequencies = _fourier_passband_basis(
        n_scans=n_scans,
        tr=tr,
        high_pass=high_pass,
        low_pass=low_pass,
    )

    def undefined_projection(
        reason: str,
        *,
        passband_rank: int,
        condition: float | None,
        exact_rank: int = 0,
    ) -> _CleaningProjection:
        empty_design = np.empty((n_scans, 0), dtype=np.float64)
        return _CleaningProjection(
            cleaning_defined=False,
            undefined_reason=reason,
            final_basis=np.empty((n_scans, 0), dtype=np.float64),
            retained=retained,
            design=empty_design,
            design_pseudoinverse=np.empty((0, n_retained), dtype=np.float64),
            passband_dimension=(int(passband.shape[1]) if filtered else n_scans),
            passband_rank=int(passband_rank),
            passband_condition_number=condition,
            exact_design_rank=int(exact_rank),
            post_exact_temporal_rank=0,
            regression_design_rank=int(exact_rank),
            nuisance_input_columns=int(confounds.shape[1]),
            nuisance_usable_columns=0,
            nuisance_pca_components=0,
            nuisance_variance_target=target,
            nuisance_variance_explained=0.0,
            nuisance_variance_target_reached=False,
            nuisance_selection_limited_by_rank=False,
            minimum_temporal_rank=minimum_rank,
            minimum_temporal_rank_fraction=minimum_fraction,
            protected_temporal_rank=minimum_rank,
            temporal_rank_floor_satisfied=False,
            algebraic_temporal_rank=0,
            standardize=bool(standardize),
        )

    # With no requested temporal filter, preserve the prior full-length residual
    # behavior.  A censored full-frequency signal cannot be reconstructed at the
    # omitted frames without imposing an interpolation model.
    filtered = high_pass is not None or low_pass is not None
    if filtered:
        retained_passband = passband[retained, :]
        if retained_passband.shape[1] == 0:
            return undefined_projection(
                "passband_contains_no_fourier_components",
                passband_rank=0,
                condition=None,
            )
        u, singular_values, vh = np.linalg.svd(retained_passband, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] == 0:
            return undefined_projection(
                "passband_basis_not_identifiable_from_retained_frames",
                passband_rank=0,
                condition=None,
            )
        tolerance = max(retained_passband.shape) * np.finfo(np.float64).eps * singular_values[0]
        passband_rank = int(np.sum(singular_values > tolerance))
        if passband_rank != retained_passband.shape[1]:
            condition = (
                float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None
            )
            return undefined_projection(
                "passband_basis_not_identifiable_from_retained_frames",
                passband_rank=passband_rank,
                condition=condition,
            )
        condition = float(singular_values[0] / singular_values[-1])
        # U is an orthonormal passband basis on retained frames.  Transform the
        # corresponding full-length basis so its retained rows are exactly U.
        full_orthogonal_passband = passband @ (vh.T / singular_values)
        bandpass_pseudoinverse = (vh.T / singular_values) @ u.T

        def bandlimit(values: np.ndarray) -> np.ndarray:
            return passband @ (bandpass_pseudoinverse @ values[retained, :])

        exact = bandlimit(np.column_stack(exact_candidates))
        nuisance = bandlimit(confounds.to_numpy(dtype=np.float64))
    else:
        passband_rank = n_retained
        condition = 1.0
        full_orthogonal_passband = np.empty((n_scans, 0), dtype=np.float64)
        exact = np.column_stack(exact_candidates)
        nuisance = confounds.to_numpy(dtype=np.float64)

    def independent_columns(values: np.ndarray) -> tuple[np.ndarray, int]:
        if values.shape[1] == 0:
            return values, 0
        retained_values = values[retained, :]
        u_design, s_design, _vh_design = np.linalg.svd(retained_values, full_matrices=False)
        if s_design.size == 0 or s_design[0] == 0:
            return np.empty((n_scans, 0), dtype=np.float64), 0
        tol = max(retained_values.shape) * np.finfo(np.float64).eps * s_design[0]
        rank = int(np.sum(s_design > tol))
        # Project the retained left-singular directions back through the same
        # column combinations to obtain their full-length evaluations.
        coefficients = np.linalg.pinv(retained_values, rcond=tol / s_design[0]) @ u_design[:, :rank]
        return values @ coefficients, rank

    exact, exact_rank = independent_columns(exact)
    post_exact_rank = int((passband_rank if filtered else n_retained) - exact_rank)
    if post_exact_rank <= 0:
        return undefined_projection(
            "exact_design_consumes_estimable_passband",
            passband_rank=passband_rank,
            condition=condition,
            exact_rank=exact_rank,
        )
    protected_rank = max(
        minimum_rank,
        int(np.ceil(minimum_fraction * post_exact_rank)),
    )
    rank_floor_satisfied = post_exact_rank >= minimum_rank
    maximum_nuisance_components = (
        max(0, post_exact_rank - protected_rank) if rank_floor_satisfied else 0
    )
    if exact_rank:
        nuisance = nuisance - exact @ (exact[retained, :].T @ nuisance[retained, :])

    usable_nuisance: list[np.ndarray] = []
    for column in nuisance.T:
        # The exact design already removes the estimable intercept.  Subtracting
        # a full-length constant here could move a censored-fit regressor outside
        # the passband, so scale by retained RMS without recentering.
        scale = float(np.sqrt(np.mean(np.square(column[retained]))))
        if np.isfinite(scale) and scale > np.finfo(np.float64).eps:
            usable_nuisance.append(column / scale)
    nuisance_usable = len(usable_nuisance)
    nuisance_components = np.empty((n_scans, 0), dtype=np.float64)
    component_count = 0
    explained = 0.0
    target_reached = False
    limited_by_rank = False
    if usable_nuisance:
        standardized = np.column_stack(usable_nuisance)
        _u_x, s_x, vh_x = np.linalg.svd(standardized[retained, :], full_matrices=False)
        variance = np.square(s_x)
        positive = variance > np.finfo(np.float64).eps * variance[0]
        variance = variance[positive]
        vh_x = vh_x[positive, :]
        if variance.size:
            cumulative = np.cumsum(variance) / variance.sum()
            target_count = int(np.searchsorted(cumulative, target, side="left") + 1)
            component_count = min(target_count, maximum_nuisance_components)
            limited_by_rank = component_count < target_count
            if component_count:
                explained = float(cumulative[component_count - 1])
                target_reached = explained >= target
                nuisance_components = standardized @ vh_x[:component_count, :].T
                nuisance_components, component_count = independent_columns(nuisance_components)

    design = np.column_stack((exact, nuisance_components))
    design, design_rank = independent_columns(design)
    retained_design = design[retained, :]
    design_pseudoinverse = retained_design.T  # independent_columns makes it orthonormal on R

    if filtered:
        # Express the unwanted design inside the orthonormal passband coordinates,
        # then retain its orthogonal complement as the final admissible basis.
        coordinates = u.T @ retained_design
        if design_rank:
            _u_d, s_d, vh_d = np.linalg.svd(coordinates.T, full_matrices=True)
            tol_d = max(coordinates.T.shape) * np.finfo(np.float64).eps * s_d[0]
            coordinate_rank = int(np.sum(s_d > tol_d))
            complement = vh_d[coordinate_rank:, :].T
        else:
            complement = np.eye(passband_rank, dtype=np.float64)
        final_basis = full_orthogonal_passband @ complement
        algebraic_rank = int(final_basis.shape[1])
        if algebraic_rank <= 0:
            raise ValueError(
                "Task and nuisance regressors consume the complete estimable passband: "
                f"passband rank {passband_rank}, design rank {design_rank}."
            )
    else:
        final_basis = None
        algebraic_rank = int(n_retained - design_rank)

    return _CleaningProjection(
        cleaning_defined=True,
        undefined_reason=None,
        final_basis=final_basis,
        retained=retained,
        design=design,
        design_pseudoinverse=design_pseudoinverse,
        passband_dimension=int(passband.shape[1]) if filtered else n_scans,
        passband_rank=passband_rank,
        passband_condition_number=condition,
        exact_design_rank=exact_rank,
        post_exact_temporal_rank=post_exact_rank,
        regression_design_rank=design_rank,
        nuisance_input_columns=int(confounds.shape[1]),
        nuisance_usable_columns=nuisance_usable,
        nuisance_pca_components=component_count,
        nuisance_variance_target=target,
        nuisance_variance_explained=explained,
        nuisance_variance_target_reached=target_reached,
        nuisance_selection_limited_by_rank=limited_by_rank,
        minimum_temporal_rank=minimum_rank,
        minimum_temporal_rank_fraction=minimum_fraction,
        protected_temporal_rank=protected_rank,
        temporal_rank_floor_satisfied=rank_floor_satisfied,
        algebraic_temporal_rank=algebraic_rank,
        standardize=bool(standardize),
    )


def _clean_volume_data(
    *,
    in_img: Path,
    out_img: Path,
    mask_img: Path,
    projection: _CleaningProjection,
    quality_path: Path,
) -> None:
    try:
        from nilearn import maskers
    except Exception as error:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency for volumetric cleaning: nilearn ({error})"
        ) from error
    masker = maskers.NiftiMasker(
        mask_img=str(mask_img),
        standardize=False,
        detrend=False,
    )
    data = masker.fit_transform(str(in_img))
    cleaned = projection.transform(data)
    masker.inverse_transform(cleaned).to_filename(str(out_img))
    write_json(quality_path, _cleaned_timecourse_quality(cleaned, projection=projection))


def _clean_surface_data(
    *,
    in_img: Path,
    out_img: Path,
    projection: _CleaningProjection,
    n_scans: int,
    quality_path: Path,
) -> None:
    data = load_gifti_timeseries(in_img)
    if int(data.shape[0]) != int(n_scans):
        raise ValueError(
            "Surface timepoint count changed during cleaning: "
            f"{data.shape[0]} != {n_scans} ({in_img})"
        )
    cleaned = projection.transform(data)
    save_gifti_timeseries(
        in_img,
        np.asarray(cleaned, dtype=np.float32),
        out_img,
    )
    write_json(quality_path, _cleaned_timecourse_quality(cleaned, projection=projection))


def _cleaned_timecourse_quality(
    data: np.ndarray,
    *,
    projection: _CleaningProjection,
    maximum_sampled_locations: int = 1024,
) -> dict[str, object]:
    """Summarize observed temporal dimensionality without another image read."""
    source_dtype = np.asarray(data).dtype
    retained_values = np.asarray(data[projection.retained, :], dtype=np.float64)
    variances = retained_values.var(axis=0, ddof=0)
    usable = np.flatnonzero(np.isfinite(variances) & (variances > np.finfo(np.float32).eps))
    if usable.size > maximum_sampled_locations:
        positions = np.linspace(0, usable.size - 1, maximum_sampled_locations, dtype=np.int64)
        sampled = usable[positions]
    else:
        sampled = usable
    if sampled.size:
        values = retained_values[:, sampled]
        values = values - values.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(values, compute_uv=False)
        if singular.size and singular[0] > 0:
            precision = (
                np.finfo(np.float32).eps
                if np.issubdtype(source_dtype, np.floating) and source_dtype.itemsize <= 4
                else np.finfo(np.float64).eps
            )
            tolerance = max(values.shape) * precision * singular[0]
            observed_rank = int(np.sum(singular > tolerance))
            spectrum = np.square(singular[singular > tolerance])
            proportions = spectrum / spectrum.sum()
            entropy_rank = float(np.exp(-np.sum(proportions * np.log(proportions))))
            participation_rank = float(1.0 / np.sum(np.square(proportions)))
            dominant_fraction = float(proportions[0])
        else:
            observed_rank = 0
            entropy_rank = 0.0
            participation_rank = 0.0
            dominant_fraction = 0.0
    else:
        observed_rank = 0
        entropy_rank = 0.0
        participation_rank = 0.0
        dominant_fraction = 0.0
    return {
        "AlgebraicTemporalRank": int(projection.algebraic_temporal_rank),
        "ObservedTemporalRank": observed_rank,
        "EntropyEffectiveTemporalRank": entropy_rank,
        "ParticipationRatioEffectiveTemporalRank": participation_rank,
        "DominantTemporalVarianceFraction": dominant_fraction,
        "EffectiveRankLocationSampleCount": int(sampled.size),
        "EffectiveRankLocationSampling": "evenly spaced among nonconstant locations",
        "NonconstantLocationCount": int(usable.size),
        "ConstantLocationCount": int(retained_values.shape[1] - usable.size),
        "Definitions": {
            "AlgebraicTemporalRank": (
                "Dimension of the clean passband after exact task removal and nuisance-PC removal."
            ),
            "ObservedTemporalRank": (
                "Numerical matrix rank of retained, temporally centered cleaned "
                "samples at the reported location sample."
            ),
            "EntropyEffectiveTemporalRank": (
                "Exponential Shannon entropy of the normalized temporal variance spectrum."
            ),
            "ParticipationRatioEffectiveTemporalRank": (
                "Inverse sum of squared normalized temporal-variance eigenvalues."
            ),
            "DominantTemporalVarianceFraction": (
                "Fraction of sampled cleaned temporal variance represented by the "
                "largest singular component."
            ),
        },
    }


def _write_volume_gm_mask(
    *,
    source_mask: Path,
    target_img: Path,
    out_mask: Path,
    threshold: float,
) -> None:
    try:
        from nilearn import image
    except Exception as error:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency for GM mask preparation: nilearn ({error})"
        ) from error
    source = image.load_img(str(source_mask))
    target = image.load_img(str(target_img))
    resampled = image.resample_to_img(
        source,
        target,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    mask = image.math_img(f"img > {float(threshold):.8g}", img=resampled)
    mask.to_filename(str(out_mask))


def _volume_gm_mask_path(
    *,
    masks_by_space: dict[str, Path],
    target_img: Path,
    work_dir: Path,
    input_desc: str,
) -> tuple[Path, bool]:
    """Resolve one shared spatial-mask path and whether it is newly declared."""
    space = _space_name(target_img)
    existing = masks_by_space.get(space)
    if existing is not None:
        return existing, False
    path = (
        work_dir
        / replace_bids_entity_token(
            target_img,
            input_desc,
            "desc-grayMatterMask",
        ).name
    )
    masks_by_space[space] = path
    return path, True


def _expected_input_groups(
    *,
    func_dir: Path,
    run_stem: str,
    output_space: str,
    clean_ica_aroma: bool,
) -> list[dict[str, object]]:
    """Construct the immutable cleaning inputs from source identity and config.

    The functional publication manifest is completion/provenance evidence.  It
    must never be allowed to add, remove, or rename nodes in cleaning's DAG.
    """
    space = str(output_space)
    variants = [
        {"input_desc": "desc-preproc", "output_desc": "desc-clean", "label": "preproc"},
    ]
    if clean_ica_aroma:
        variants.append(
            {
                "input_desc": "desc-preprocNoAROMA",
                "output_desc": "desc-cleanNoAROMA",
                "label": "preprocNoAROMA",
            }
        )
    for variant in variants:
        desc = str(variant["input_desc"])
        is_surface = is_surface_space(space)
        volumes = [] if is_surface else [func_dir / f"{run_stem}_space-{space}_{desc}_bold.nii.gz"]
        surfaces = (
            [func_dir / f"{run_stem}_space-{space}_hemi-L_{desc}_bold.func.gii"]
            if is_surface
            else []
        )
        variant["vols"] = volumes
        variant["surfs"] = surfaces
    return variants


def _main_sidecar_path(vols: list[Path], surfs: list[Path]) -> Path:
    for p in vols:
        if "_space-T1w_" in p.name:
            return sidecar_json_path(p)
    if vols:
        return sidecar_json_path(vols[0])
    if surfs:
        return sidecar_json_path(surfs[0])
    raise SystemExit("No matching desc-preproc inputs were found.")
