#!/usr/bin/env python3
"""Clean one preprocessed BOLD run in one space at one smoothing level.

Nuisance and Fourier stopband coefficients are estimated from temporally
retained frames, then evaluated across the complete original time axis.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from nro.orchestration.runtime import selected_configuration_fingerprint
from nro.configuration.runtime import SETTINGS
from nro.engine.bids import bids_entity, replace_bids_entity_token
from nro.engine.execution import new_step_counter
from nro.engine.images import (
    gifti_vertex_count,
    load_gifti_timeseries,
    save_gifti_timeseries,
    sidecar_json_path,
)
from nro.engine.io import read_json, write_json
from nro.engine.templates import find_fsaverage_surface
from nro.engine.paths import (
    is_bids_session_id,
    anatomical_manifest_path,
    functional_manifest_path as preprocessing_functional_manifest_path,
    preprocess_subject_func_dir,
    preprocess_session_func_dir,
    project_data_root,
    resolve_cwd_path,
)
from nro.orchestration.runner import (
    ContainerSpec,
    Runner,
    write_completion_breadcrumb,
)
from nro.orchestration.runner_graph import Step
from nro.clean.paths import (
    clean_manifest_path,
    clean_session_dir,
    clean_session_work_dir,
    clean_subject_dir,
    clean_subject_work_dir,
)
from nro.engine.targets import add_smoothing_entity


LOG = logging.getLogger("clean")
next_step = new_step_counter()


def _smoothed_desc_label(input_desc: str) -> str:
    return "desc-smoothedPreCleanNoAROMA" if input_desc == "desc-preprocNoAROMA" else "desc-smoothedPreClean"


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
            raise SystemExit(f"Missing fsnative midthickness surface in anatomical manifest: surfaces.{manifest_key}")
        return surf
    if space == "fsaverage":
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    confounds = pd.read_csv(confounds_tsv, sep="\t")
    sidecar = read_json(confounds_json) if confounds_json.exists() else {}
    selected = confounds.filter(regex=regex)
    selected_sidecar = {k: v for k, v in sidecar.items() if k in selected.columns}
    if regress_out_task:
        task_regs, task_sidecar = _build_task_regressors(
            events_path=events_path,
            n_scans=n_scans,
            tr=tr,
            start_time=start_time,
        )
        if task_regs:
            selected = pd.concat(task_regs + [selected], axis=1)
            selected_sidecar.update(task_sidecar)
    return selected.fillna(0.0), selected_sidecar


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


def _fourier_stopband_basis(
    *,
    n_scans: int,
    tr: float,
    high_pass: Optional[float],
    low_pass: Optional[float],
) -> tuple[np.ndarray, list[str]]:
    """Build real Fourier regressors for frequencies outside the passband."""
    if n_scans < 2:
        return np.empty((n_scans, 0), dtype=np.float64), []
    nyquist = 0.5 / float(tr)
    if high_pass is not None and not 0.0 <= float(high_pass) < nyquist:
        raise ValueError(f"high_pass must be in [0, {nyquist:g}), got {high_pass}")
    if low_pass is not None and not 0.0 < float(low_pass) <= nyquist:
        raise ValueError(f"low_pass must be in (0, {nyquist:g}], got {low_pass}")
    if high_pass is not None and low_pass is not None and float(high_pass) >= float(low_pass):
        raise ValueError(f"high_pass must be below low_pass: {high_pass} >= {low_pass}")
    if high_pass is None and low_pass is None:
        return np.empty((n_scans, 0), dtype=np.float64), []

    times = np.arange(n_scans, dtype=np.float64) * float(tr)
    frequencies = np.fft.rfftfreq(n_scans, d=float(tr))
    columns: list[np.ndarray] = []
    names: list[str] = []
    for index, frequency in enumerate(frequencies):
        if index == 0:
            continue  # The intercept handles DC.
        remove = (
            (high_pass is not None and frequency < float(high_pass))
            or (low_pass is not None and frequency > float(low_pass))
        )
        if not remove:
            continue
        angle = 2.0 * np.pi * frequency * times
        cosine = np.cos(angle)
        columns.append(cosine)
        names.append(f"stopband_cos_{frequency:.12g}Hz")
        sine = np.sin(angle)
        if np.linalg.norm(sine) > np.finfo(np.float64).eps * n_scans:
            columns.append(sine)
            names.append(f"stopband_sin_{frequency:.12g}Hz")
    if not columns:
        return np.empty((n_scans, 0), dtype=np.float64), []
    return np.column_stack(columns), names


@dataclass(frozen=True)
class _CleaningProjection:
    design: np.ndarray
    retained: np.ndarray
    pseudoinverse: np.ndarray
    design_columns: tuple[str, ...]
    rank: int
    residual_dof: int
    standardize: bool

    def transform(self, data: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
        """Fit on retained rows and evaluate residuals at every original row."""
        values = np.asarray(data)
        if values.ndim != 2 or values.shape[0] != self.design.shape[0]:
            raise ValueError(
                f"Expected time-by-feature data with {self.design.shape[0]} rows, "
                f"got {values.shape}"
            )
        output = np.empty(values.shape, dtype=np.float32)
        for start in range(0, values.shape[1], int(chunk_size)):
            stop = min(start + int(chunk_size), values.shape[1])
            chunk = np.asarray(values[:, start:stop], dtype=np.float64)
            if not np.all(np.isfinite(chunk)):
                raise ValueError("Functional samples contain non-finite values")
            retained_chunk = chunk[self.retained, :]
            coefficients = self.pseudoinverse @ retained_chunk
            residuals = chunk - self.design @ coefficients
            if self.standardize:
                retained_residuals = residuals[self.retained, :]
                means = retained_residuals.mean(axis=0)
                scales = retained_residuals.std(axis=0, ddof=0)
                scales[scales <= np.finfo(np.float64).eps] = 1.0
                residuals = (residuals - means) / scales
            output[:, start:stop] = residuals.astype(np.float32)
        return output


def _build_cleaning_projection(
    *,
    confounds: pd.DataFrame,
    outliers: pd.DataFrame,
    tr: float,
    detrend: bool,
    standardize: bool,
    high_pass: Optional[float],
    low_pass: Optional[float],
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

    candidates: list[np.ndarray] = [np.ones(n_scans, dtype=np.float64)]
    names = ["intercept"]
    if detrend:
        candidates.append(np.linspace(-1.0, 1.0, n_scans, dtype=np.float64))
        names.append("linear_trend")
    for column in confounds.columns:
        candidates.append(confounds[column].to_numpy(dtype=np.float64))
        names.append(str(column))
    stopband, stopband_names = _fourier_stopband_basis(
        n_scans=n_scans,
        tr=tr,
        high_pass=high_pass,
        low_pass=low_pass,
    )
    for index, name in enumerate(stopband_names):
        candidates.append(stopband[:, index])
        names.append(name)

    normalized: list[np.ndarray] = []
    retained_names: list[str] = []
    for name, candidate in zip(names, candidates):
        values = np.asarray(candidate, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Cleaning design column contains non-finite values: {name}")
        if name != "intercept":
            values = values - values[retained].mean()
        norm = float(np.linalg.norm(values[retained]))
        if norm <= np.finfo(np.float64).eps * max(1, n_retained):
            continue
        normalized.append(values / norm)
        retained_names.append(name)
    design = np.column_stack(normalized)
    retained_design = design[retained, :]
    singular_values = np.linalg.svd(retained_design, compute_uv=False)
    tolerance = (
        max(retained_design.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    rank = int(np.sum(singular_values > tolerance))
    residual_dof = n_retained - rank
    if residual_dof <= 0:
        raise ValueError(
            "Cleaning design is saturated after temporal masking: "
            f"{n_retained} retained frames, rank {rank}, residual DOF {residual_dof}. "
            "The run cannot support the requested nuisance model and temporal passband."
        )
    pseudoinverse = np.linalg.pinv(retained_design, rcond=tolerance / singular_values[0])
    return _CleaningProjection(
        design=design,
        retained=retained,
        pseudoinverse=pseudoinverse,
        design_columns=tuple(retained_names),
        rank=rank,
        residual_dof=residual_dof,
        standardize=bool(standardize),
    )


def _clean_volume_data(
    *,
    in_img: Path,
    out_img: Path,
    mask_img: Path,
    projection: _CleaningProjection,
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


def _clean_surface_data(
    *,
    in_img: Path,
    out_img: Path,
    projection: _CleaningProjection,
    n_scans: int,
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
    path = work_dir / replace_bids_entity_token(
        target_img,
        input_desc,
        "desc-grayMatterMask",
    ).name
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
        is_surface = space in {"fsnative", "fsaverage"}
        volumes = (
            [] if is_surface else [func_dir / f"{run_stem}_space-{space}_{desc}_bold.nii.gz"]
        )
        surfaces = (
            [func_dir / f"{run_stem}_space-{space}_hemi-L_{desc}_bold.func.gii"]
            if is_surface else []
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


def build_module(
    argv: Optional[Sequence[str]],
) -> tuple[Runner, bool, int]:
    """Resolve BIDS inputs and construct the complete cleaning DAG."""
    cfg = SETTINGS.clean
    ap = argparse.ArgumentParser(prog="clean.py", description=__doc__)
    ap.add_argument("--project", default=SETTINGS.common.project)
    ap.add_argument("--preprocessing-id", default=SETTINGS.common.preprocessing_id)
    ap.add_argument("--clean-id", default=SETTINGS.common.clean_id)
    ap.add_argument("--sub-id", required=True)
    ap.add_argument("--ses-id", default=None)
    ap.add_argument("--run-stem", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--smoothing", required=True, type=int, metavar="MM")
    ap.add_argument("--functional-ica-aroma", action="store_true")
    ap.add_argument("--min-trs", type=int, default=int(cfg.min_trs))
    ap.add_argument(
        "--gm-mask-threshold",
        type=float,
        default=float(cfg.gm_mask_threshold),
    )
    ap.add_argument("--confounds-regex", default=str(cfg.confounds_regex))
    ap.add_argument("--temporal-mask-regex", default=str(cfg.temporal_mask_regex))
    ap.add_argument(
        "--standardize", action="store_true", default=bool(cfg.standardize)
    )
    ap.add_argument("--no-standardize", action="store_false", dest="standardize")
    ap.add_argument("--detrend", action="store_true", default=bool(cfg.detrend))
    ap.add_argument("--no-detrend", action="store_false", dest="detrend")
    ap.add_argument(
        "--regress-out-task",
        action="store_true",
        default=bool(cfg.regress_out_task),
    )
    ap.add_argument(
        "--no-regress-out-task", action="store_false", dest="regress_out_task"
    )
    ap.add_argument(
        "--low-pass",
        type=float,
        default=None if cfg.low_pass is None else float(cfg.low_pass),
    )
    ap.add_argument(
        "--high-pass",
        type=float,
        default=None if cfg.high_pass is None else float(cfg.high_pass),
    )
    ap.add_argument("--force", action="store_true", default=bool(cfg.force))
    ap.add_argument("--verbose", action="store_true", default=bool(cfg.verbose))
    ap.add_argument("--container", type=Path, default=Path(str(cfg.container)))
    ap.add_argument("--no-container", action="store_true", default=bool(cfg.no_container))
    ap.add_argument("--container-engine", default=str(cfg.container_engine))
    ap.add_argument(
        "--container-no-cleanenv",
        action="store_true",
        default=not bool(cfg.container_cleanenv),
    )
    ap.add_argument("--container-bind", action="append", default=list(cfg.container_bind))
    ap.add_argument("--container-home", type=Path, default=cfg.container_home)
    ap.add_argument("--container-inner-setup", default=str(cfg.container_inner_setup))
    args = ap.parse_args(argv)

    if args.smoothing < 0:
        raise SystemExit("--smoothing must be a nonnegative integer FWHM in mm")
    smoothing_mm = int(args.smoothing)
    ses_id = str(args.ses_id).strip() if args.ses_id is not None else None
    if ses_id == "":
        ses_id = None
    if ses_id is not None and not is_bids_session_id(ses_id):
        raise SystemExit(
            f"--ses-id must look like a BIDS session ID (ses-*), got {ses_id!r}"
        )

    if ses_id is None:
        func_dir = preprocess_subject_func_dir(
            args.sub_id,
            project=args.project,
            preprocessing_id=args.preprocessing_id,
        )
        clean_dir = clean_subject_dir(
            args.sub_id, project=args.project, clean_id=args.clean_id
        )
        work_dir = (
            clean_subject_work_dir(
                args.sub_id, project=args.project, clean_id=args.clean_id
            )
            / args.run_stem
            / f"space-{args.space}_smoothing-{smoothing_mm}mm"
        )
    else:
        func_dir = preprocess_session_func_dir(
            args.sub_id,
            ses_id,
            project=args.project,
            preprocessing_id=args.preprocessing_id,
        )
        clean_dir = clean_session_dir(
            args.sub_id, ses_id, project=args.project, clean_id=args.clean_id
        )
        work_dir = (
            clean_session_work_dir(
                args.sub_id,
                ses_id,
                project=args.project,
                clean_id=args.clean_id,
            )
            / args.run_stem
            / f"space-{args.space}_smoothing-{smoothing_mm}mm"
        )

    functional_manifest = preprocessing_functional_manifest_path(
        args.sub_id,
        args.run_stem,
        project=args.project,
        preprocessing_id=args.preprocessing_id,
        ses_id=ses_id,
    )
    if not functional_manifest.is_file():
        raise SystemExit(
            "Functional preprocessing did not publish its fixed run manifest: "
            f"{functional_manifest}"
        )
    functional_contract = read_json(functional_manifest)
    if functional_contract.get("run_stem") != args.run_stem:
        raise SystemExit(f"Functional manifest run identity mismatch: {functional_manifest}")

    variants = _expected_input_groups(
        func_dir=func_dir,
        run_stem=args.run_stem,
        output_space=args.space,
        clean_ica_aroma=bool(args.functional_ica_aroma),
    )
    recorded_clean_inputs = (
        (functional_contract.get("public_outputs") or {}).get("clean_inputs") or {}
    )
    recorded_paths = json.dumps(recorded_clean_inputs, sort_keys=True)
    selected_paths = [
        str(path)
        for variant in variants
        for path in (*variant["vols"], *variant["surfs"])
    ]
    selected_paths.extend(
        str(path).replace("_hemi-L_", "_hemi-R_")
        for path in map(Path, tuple(selected_paths))
        if "_hemi-L_" in path.name
    )
    if any(path not in recorded_paths for path in selected_paths):
        raise SystemExit(
            "Functional publication manifest does not match the immutable "
            f"preprocessing-config output contract: {functional_manifest}"
        )
    missing = [
        str(path)
        for variant in variants
        for left in (*variant["vols"], *variant["surfs"])
        for path in (
            (left, Path(str(left).replace("_hemi-L_", "_hemi-R_")))
            if "_hemi-L_" in left.name
            else (left,)
        )
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "Missing functional clean input(s) required by the selected workflow: "
            + ", ".join(missing)
        )

    primary = next(
        (variant for variant in variants if variant["vols"] or variant["surfs"]),
        None,
    )
    if primary is None:
        raise SystemExit(f"No cleanable inputs were found for run: {args.run_stem}")
    primary_volumes = list(primary["vols"])
    primary_surfaces = list(primary["surfs"])
    main_sidecar = read_json(
        _main_sidecar_path(primary_volumes, primary_surfaces)
    )
    tr = float(main_sidecar.get("RepetitionTime", 0) or 0)
    if tr <= 0:
        raise SystemExit("RepetitionTime was not found in the preprocessed sidecar.")
    start_time = float(main_sidecar.get("StartTime", 0.0) or 0.0)

    anatomical_path = anatomical_manifest_path(
        args.sub_id,
        project=args.project,
        preprocessing_id=args.preprocessing_id,
    )
    anatomical = read_json(anatomical_path)
    anat_outputs = anatomical.get("outputs") or {}
    anat_gray_matter = Path(str(anat_outputs.get("gray_matter_mask") or "").strip())
    if not anat_gray_matter.exists():
        raise SystemExit(
            "Missing anatomical gray matter mask in anatomical manifest: "
            f"{anat_gray_matter}"
        )
    mni_template = Path(str(anatomical.get("mni_template", "")).strip())
    if not mni_template.exists() and any(
        _space_name(path).startswith("MNI") for path in primary_volumes
    ):
        raise SystemExit(f"Missing MNI template in anatomical manifest: {mni_template}")
    mni_gray_matter = Path(
        str(mni_template).replace("_T1w.nii.gz", "_label-GM_probseg.nii.gz")
    )
    if not mni_gray_matter.exists() and any(
        _space_name(path).startswith("MNI") for path in primary_volumes
    ):
        raise SystemExit(
            f"Missing MNI gray matter probability template: {mni_gray_matter}"
        )
    templateflow_root = mni_template.parent.parent
    fsaverage_surface_cache: dict[tuple[str, int], Path] = {}

    confounds_tsv = func_dir / f"{args.run_stem}_desc-confounds_timeseries.tsv"
    confounds_json = func_dir / f"{args.run_stem}_desc-confounds_timeseries.json"
    if not confounds_tsv.exists():
        raise SystemExit(f"Missing confounds TSV: {confounds_tsv}")
    source_root = project_data_root(args.project) / args.sub_id
    if ses_id is not None:
        source_root /= ses_id
    events_path = source_root / "func" / f"{args.run_stem}_events.tsv"

    publication_manifest = clean_manifest_path(
        args.sub_id,
        args.run_stem,
        project=args.project,
        clean_id=args.clean_id,
        space=args.space,
        smoothing_mm=smoothing_mm,
        ses_id=ses_id,
    )
    confounds_basename = (
        f"{args.run_stem}_space-{args.space}_scale-{smoothing_mm}mm_"
        "desc-confounds_timeseries"
    )
    confounds_out = clean_dir / f"{confounds_basename}.tsv"
    confounds_out_json = confounds_out.with_suffix(".json")

    source_inputs: list[Path] = [
        functional_manifest,
        confounds_tsv,
        confounds_json,
        anatomical_path,
    ]
    if events_path.exists():
        source_inputs.append(events_path)
    functional_paths: list[Path] = []
    for variant in variants:
        for volume in variant["vols"]:
            functional_paths.append(volume)
            source_inputs.extend((volume, sidecar_json_path(volume)))
        for left in variant["surfs"]:
            right = Path(str(left).replace("_hemi-L_", "_hemi-R_"))
            functional_paths.extend((left, right))
            source_inputs.extend(
                (
                    left,
                    sidecar_json_path(left),
                    right,
                    sidecar_json_path(right),
                )
            )

    import nibabel as nib

    def functional_trs(path: Path) -> int:
        image = nib.load(str(path))
        if path.name.endswith(".func.gii"):
            return len(image.darrays)
        if len(image.shape) != 4:
            raise SystemExit(f"Expected 4D functional input: {path}")
        return int(image.shape[3])

    counts = {path: functional_trs(path) for path in functional_paths}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise SystemExit(
            "Mismatched number of TRs across clean inputs: "
            + ", ".join(f"{path}={value}" for path, value in counts.items())
        )
    sample_count = unique_counts.pop()
    if sample_count < int(args.min_trs):
        raise SystemExit(
            f"Run has fewer than the required minimum TRs ({sample_count} < "
            f"{int(args.min_trs)}): {args.run_stem}"
        )

    configuration = {
        "clean_id": str(args.clean_id),
        "preprocessing_id": str(args.preprocessing_id),
        "min_trs": int(args.min_trs),
        "gray_matter_mask_threshold": float(args.gm_mask_threshold),
        "space": str(args.space),
        "smoothing_fwhm_mm": smoothing_mm,
        "confounds_regex": str(args.confounds_regex),
        "temporal_mask_regex": str(args.temporal_mask_regex),
        "standardize": bool(args.standardize),
        "detrend": bool(args.detrend),
        "regress_out_task": bool(args.regress_out_task and events_path.exists()),
        "low_pass_hz": args.low_pass,
        "high_pass_hz": args.high_pass,
        "configuration_fingerprint": selected_configuration_fingerprint(),
    }

    container_home = (
        resolve_cwd_path(args.container_home)
        if args.container_home is not None
        else work_dir / SETTINGS.common.qunex_home_dirname
    )
    container = None if args.no_container else ContainerSpec(
        image=Path(resolve_cwd_path(args.container) or args.container),
        engine=str(args.container_engine),
        cleanenv=not bool(args.container_no_cleanenv),
        extra_binds=tuple(str(value) for value in (args.container_bind or [])),
        home_dir=container_home,
        inner_setup=str(args.container_inner_setup or ""),
    )
    runner = Runner(
        module_name="Cleaning Module",
        container=container,
        binds=(),
        logger=LOG,
        next_step=next_step,
    )
    initialized = work_dir / "initialized.complete"

    def initialize() -> None:
        clean_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        if container_home is not None:
            container_home.mkdir(parents=True, exist_ok=True)
        write_completion_breadcrumb(initialized, "Cleaning output initialized\n")

    runner.add_step(
        Step.python(
            name="Initialize Cleaning Outputs",
            outputs=(initialized,),
            action=initialize,
        )
    )

    configuration_snapshot = work_dir / "configuration.json"

    def validate_configuration() -> tuple[bool, str]:
        try:
            current = read_json(configuration_snapshot)
        except (OSError, ValueError, TypeError):
            return False, "Cleaning configuration snapshot is missing or unreadable."
        if current != configuration:
            return False, "Cleaning configuration changed."
        return True, "Cleaning configuration is unchanged."

    runner.add_step(
        Step.python(
            name="Write Cleaning Configuration",
            outputs=(configuration_snapshot,),
            inputs=(initialized,),
            force=bool(args.force),
            action=lambda: write_json(configuration_snapshot, configuration),
            validate=validate_configuration,
        )
    )

    def configured(step: Step) -> Step:
        """Return a scientific step tied to the configuration snapshot."""
        return replace(step, inputs=(configuration_snapshot, *step.inputs))

    selected_confounds_path = work_dir / "selected_confounds.tsv"
    selected_confounds_json = work_dir / "selected_confounds.json"
    outlier_confounds_path = work_dir / "outlier_confounds.tsv"

    def prepare_confounds() -> None:
        selected, sidecar = _filter_confounds(
            confounds_tsv=confounds_tsv,
            confounds_json=confounds_json,
            regex=str(args.confounds_regex),
            events_path=events_path,
            n_scans=sample_count,
            tr=tr,
            start_time=start_time,
            regress_out_task=bool(args.regress_out_task),
        )
        selected.to_csv(selected_confounds_path, sep="\t", index=False)
        outliers = _select_outlier_columns(
            confounds_tsv=confounds_tsv,
            regex=str(args.temporal_mask_regex),
            n_scans=sample_count,
        )
        outliers.to_csv(outlier_confounds_path, sep="\t", index=False)
        write_json(selected_confounds_json, sidecar)

    runner.add_step(configured(
        Step.python(
            name="Prepare Confounds",
            outputs=(
                selected_confounds_path,
                selected_confounds_json,
                outlier_confounds_path,
            ),
            inputs=(
                confounds_tsv,
                confounds_json,
                initialized,
                *((events_path,) if events_path.exists() else ()),
            ),
            force=bool(args.force),
            action=prepare_confounds,
        )
    ))

    projection_cache: dict[str, object] = {}

    def projection_state() -> tuple[_CleaningProjection, dict[str, object]]:
        if "projection" in projection_cache:
            return projection_cache["projection"], projection_cache["metadata"]
        selected = pd.read_csv(selected_confounds_path, sep="\t").fillna(0.0)
        try:
            outliers = pd.read_csv(outlier_confounds_path, sep="\t").fillna(0.0)
        except pd.errors.EmptyDataError:
            outliers = pd.DataFrame(index=np.arange(sample_count))
        projection = _build_cleaning_projection(
            confounds=selected,
            outliers=outliers,
            tr=tr,
            detrend=bool(args.detrend),
            standardize=bool(args.standardize),
            low_pass=args.low_pass,
            high_pass=args.high_pass,
        )
        metadata: dict[str, object] = {
            "TemporalMaskRegex": str(args.temporal_mask_regex),
            "RetainedFrames": int(projection.retained.sum()),
            "CensoredFrames": int((~projection.retained).sum()),
            "ProjectionRank": int(projection.rank),
            "ResidualDegreesOfFreedom": int(projection.residual_dof),
            "TemporalFilterMethod": "masked-fit full-length Fourier projection",
        }
        projection_cache["projection"] = projection
        projection_cache["metadata"] = metadata
        return projection, metadata

    projection_inputs = (selected_confounds_path, outlier_confounds_path)
    expected_outputs: list[Path] = []
    cleaned_outputs: list[Path] = []
    masks_by_space: dict[str, Path] = {}

    def cleaning_metadata(
        *,
        source: Path,
        clean_input: Path,
        input_desc: str,
        gm_mask: Path | None,
    ) -> dict[str, object]:
        sidecar = dict(read_json(sidecar_json_path(source)))
        _projection, projection_metadata = projection_state()
        sidecar["Description"] = "Cleaned BOLD timeseries for connectivity analysis."
        sidecar["Sources"] = [str(source), str(confounds_tsv)] + (
            [str(events_path)]
            if args.regress_out_task and events_path.exists()
            else []
        )
        sidecar["Cleaning"] = {
            "CleanID": str(args.clean_id),
            "PreprocessingID": str(args.preprocessing_id),
            "Space": str(args.space),
            "InputDescription": input_desc,
            "MinTRs": int(args.min_trs),
            **(
                {
                    "GrayMatterMask": str(gm_mask),
                    "GrayMatterMaskThreshold": float(args.gm_mask_threshold),
                }
                if gm_mask is not None
                else {}
            ),
            "SmoothedInput": str(clean_input) if clean_input != source else None,
            "SmoothingFWHMMM": float(args.smoothing),
            "ConfoundsRegex": str(args.confounds_regex),
            **projection_metadata,
            "Standardize": bool(args.standardize),
            "Detrend": bool(args.detrend),
            "RegressOutTask": bool(
                args.regress_out_task and events_path.exists()
            ),
            "LowPassHz": args.low_pass,
            "HighPassHz": args.high_pass,
            "ConfigurationFingerprint": selected_configuration_fingerprint(),
        }
        return sidecar

    for variant in variants:
        input_desc = str(variant["input_desc"])
        output_desc = str(variant["output_desc"])
        for volume in variant["vols"]:
            out_path = clean_dir / add_smoothing_entity(
                replace_bids_entity_token(volume, input_desc, output_desc),
                smoothing_mm,
            ).name
            gm_source = (
                mni_gray_matter
                if _space_name(volume).startswith("MNI")
                else anat_gray_matter
            )
            gm_mask, new_mask = _volume_gm_mask_path(
                masks_by_space=masks_by_space,
                target_img=volume,
                work_dir=work_dir,
                input_desc=input_desc,
            )
            if new_mask:
                runner.add_step(configured(
                    Step.python(
                        name=f"Prepare Gray Matter Mask: {_space_name(volume)}",
                        outputs=(gm_mask,),
                        inputs=(gm_source, volume, initialized),
                        force=bool(args.force),
                        action=lambda source=gm_source, target=volume, output=gm_mask: (
                            _write_volume_gm_mask(
                                source_mask=source,
                                target_img=target,
                                out_mask=output,
                                threshold=float(args.gm_mask_threshold),
                            )
                        ),
                                )
                ))

            clean_input = volume
            if smoothing_mm > 0:
                clean_input = work_dir / replace_bids_entity_token(
                    volume,
                    input_desc,
                    _smoothed_desc_label(input_desc),
                ).name
                runner.add_step(configured(
                    Step.command_step(
                        [
                            str(SETTINGS.common.wb_command),
                            "-volume-smoothing",
                            str(volume),
                            f"{float(smoothing_mm):g}",
                            str(clean_input),
                            "-fwhm",
                            "-roi",
                            str(gm_mask),
                        ],
                        name=f"Smooth Volume: {_space_name(volume)}",
                        outputs=(clean_input,),
                        inputs=(volume, gm_mask),
                        force=bool(args.force),
                        prepare=lambda path=clean_input: path.parent.mkdir(
                            parents=True, exist_ok=True
                        ),
                                )
                ))

            runner.add_step(configured(
                Step.python(
                    name=f"Clean Volume: {_space_name(volume)}",
                    outputs=(out_path,),
                    inputs=(clean_input, gm_mask, *projection_inputs),
                    force=bool(args.force),
                    action=lambda source=clean_input, output=out_path, mask_path=gm_mask: (
                        _clean_volume_data(
                            in_img=source,
                            out_img=output,
                            mask_img=mask_path,
                            projection=projection_state()[0],
                        )
                    ),
                        )
            ))
            metadata_path = sidecar_json_path(out_path)
            runner.add_step(configured(
                Step.python(
                    name=f"Write Cleaned Volume Metadata: {out_path.name}",
                    outputs=(metadata_path,),
                    inputs=(
                        out_path,
                        sidecar_json_path(volume),
                        confounds_tsv,
                        *projection_inputs,
                        *((events_path,) if events_path.exists() else ()),
                    ),
                    force=bool(args.force),
                    action=lambda path=metadata_path, source=volume, cleaned=clean_input, desc=input_desc, mask_path=gm_mask: write_json(
                        path,
                        cleaning_metadata(
                            source=source,
                            clean_input=cleaned,
                            input_desc=desc,
                            gm_mask=mask_path,
                        ),
                    ),
                        )
            ))
            cleaned_outputs.append(out_path)
            expected_outputs.extend((out_path, metadata_path))

        for left in variant["surfs"]:
            right = Path(str(left).replace("_hemi-L_", "_hemi-R_"))
            if not right.exists():
                raise SystemExit(
                    f"Missing right-hemisphere pair for {left.name}: {right}"
                )
            for surface in (left, right):
                source_sidecar = read_json(sidecar_json_path(surface))
                clean_input = surface
                if smoothing_mm > 0:
                    geometry = _resolve_surface_for_metric(
                        metric_path=surface,
                        metric_sidecar=source_sidecar,
                        anat_outputs=anat_outputs,
                        templateflow_root=templateflow_root,
                        fsaverage_cache=fsaverage_surface_cache,
                    )
                    clean_input = work_dir / replace_bids_entity_token(
                        surface,
                        input_desc,
                        _smoothed_desc_label(input_desc),
                    ).name
                    runner.add_step(configured(
                        Step.command_step(
                            [
                                str(SETTINGS.common.wb_command),
                                "-metric-smoothing",
                                str(geometry),
                                str(surface),
                                f"{float(smoothing_mm):g}",
                                str(clean_input),
                                "-fwhm",
                            ],
                            name=f"Smooth Surface: {_space_name(surface)}",
                            outputs=(clean_input,),
                            inputs=(surface, geometry),
                            force=bool(args.force),
                            prepare=lambda path=clean_input: path.parent.mkdir(
                                parents=True, exist_ok=True
                            ),
                                        )
                    ))

                out_path = clean_dir / add_smoothing_entity(
                    replace_bids_entity_token(surface, input_desc, output_desc),
                    smoothing_mm,
                ).name
                runner.add_step(configured(
                    Step.python(
                        name=f"Clean Surface: {_space_name(surface)}",
                        outputs=(out_path,),
                        inputs=(clean_input, *projection_inputs),
                        force=bool(args.force),
                        action=lambda source=clean_input, output=out_path, scans=counts[surface]: (
                            _clean_surface_data(
                                in_img=source,
                                out_img=output,
                                projection=projection_state()[0],
                                n_scans=scans,
                            )
                        ),
                                )
                ))
                metadata_path = sidecar_json_path(out_path)
                runner.add_step(configured(
                    Step.python(
                        name=f"Write Cleaned Surface Metadata: {out_path.name}",
                        outputs=(metadata_path,),
                        inputs=(
                            out_path,
                            sidecar_json_path(surface),
                            confounds_tsv,
                            *projection_inputs,
                            *((events_path,) if events_path.exists() else ()),
                        ),
                        force=bool(args.force),
                        action=lambda path=metadata_path, source=surface, cleaned=clean_input, desc=input_desc: write_json(
                            path,
                            cleaning_metadata(
                                source=source,
                                clean_input=cleaned,
                                input_desc=desc,
                                gm_mask=None,
                            ),
                        ),
                                )
                ))
                cleaned_outputs.append(out_path)
                expected_outputs.extend((out_path, metadata_path))

    def finalize_confounds() -> None:
        selected = pd.read_csv(selected_confounds_path, sep="\t").fillna(0.0)
        try:
            outliers = pd.read_csv(outlier_confounds_path, sep="\t").fillna(0.0)
        except pd.errors.EmptyDataError:
            outliers = pd.DataFrame(index=np.arange(sample_count))
        published = pd.concat([selected, outliers], axis=1)
        published.to_csv(confounds_out, sep="\t", index=False)
        sidecar = read_json(selected_confounds_json)
        sidecar["TemporalMask"] = projection_state()[1]
        sidecar["Columns"] = list(published.columns)
        write_json(confounds_out_json, sidecar)

    runner.add_step(configured(
        Step.python(
            name="Finalize Confounds",
            outputs=(confounds_out, confounds_out_json),
            inputs=(
                selected_confounds_path,
                selected_confounds_json,
                outlier_confounds_path,
            ),
            force=bool(args.force),
            action=finalize_confounds,
        )
    ))
    expected_outputs.extend((confounds_out, confounds_out_json))

    clean_targets: list[dict[str, object]] = []
    for variant in variants:
        input_desc = str(variant["input_desc"])
        output_desc = str(variant["output_desc"])
        variant_name = output_desc.removeprefix("desc-")
        for volume in variant["vols"]:
            output = clean_dir / add_smoothing_entity(
                replace_bids_entity_token(volume, input_desc, output_desc),
                smoothing_mm,
            ).name
            clean_targets.append(
                {
                    "variant": variant_name,
                    "domain": "volume",
                    "space": _space_name(output),
                    "smoothing_fwhm_mm": smoothing_mm,
                    "functional": [str(output)],
                }
            )
        for left in variant["surfs"]:
            output_left = clean_dir / add_smoothing_entity(
                replace_bids_entity_token(left, input_desc, output_desc),
                smoothing_mm,
            ).name
            output_right = clean_dir / add_smoothing_entity(
                replace_bids_entity_token(
                    Path(str(left).replace("_hemi-L_", "_hemi-R_")),
                    input_desc,
                    output_desc,
                ),
                smoothing_mm,
            ).name
            clean_targets.append(
                {
                    "variant": variant_name,
                    "domain": "surface",
                    "space": _space_name(output_left),
                    "smoothing_fwhm_mm": smoothing_mm,
                    "functional": [str(output_left), str(output_right)],
                }
            )

    clean_contract = {
        "manifest_version": 1,
        "module": "clean",
        "run_stem": args.run_stem,
        "space": str(args.space),
        "smoothing_fwhm_mm": smoothing_mm,
        "source_bold": str(
            (functional_contract.get("inputs") or {}).get("epi") or ""
        ),
        "targets": clean_targets,
        "public_outputs": [str(path) for path in expected_outputs],
        "configuration": configuration,
        "configuration_fingerprint": selected_configuration_fingerprint(),
        "complete": True,
    }

    def validate_publication() -> tuple[bool, str]:
        try:
            actual = read_json(publication_manifest)
        except (OSError, ValueError, TypeError):
            return False, f"Cleaning publication manifest is unreadable: {publication_manifest}"
        if actual != clean_contract:
            return False, "Cleaning publication manifest differs from the requested module."
        missing_public = [
            str(path)
            for path in expected_outputs
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing_public:
            return False, "Cleaning publication is missing outputs: " + ", ".join(missing_public)
        return True, "Cleaning publication is complete and current."

    runner.add_step(configured(
        Step.python(
            name="Write Cleaning Publication Manifest",
            outputs=(publication_manifest,),
            inputs=tuple(expected_outputs) + tuple(source_inputs),
            force=bool(args.force),
            action=lambda: write_json(publication_manifest, clean_contract),
            validate=validate_publication,
            completion_boundary=True,
        )
    ))

    LOG.info(
        "Constructed cleaning module for %s in space-%s at %dmm smoothing.",
        args.run_stem,
        args.space,
        smoothing_mm,
    )
    return runner, bool(args.verbose), len(cleaned_outputs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    runner_started = time.perf_counter()
    requested_args = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=(
            logging.DEBUG
            if "--verbose" in requested_args or bool(SETTINGS.clean.verbose)
            else logging.INFO
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runner, verbose, output_count = build_module(argv)
    with runner.run_context(started_at=runner_started):
        runner.execute()
    if verbose:
        print(f"Cleaned {output_count} functional outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
