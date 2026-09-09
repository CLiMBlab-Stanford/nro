#!/usr/bin/env python3
"""Clean one preprocessed BOLD run in one space at one smoothing level.

The cleaner fits each run to the portion of its Fourier basis inside the
requested passband after removing task and nuisance directions.  Censored
frames never influence a fit, but the fitted basis is evaluated over the
complete original time axis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from nro.configuration.runtime import SETTINGS
from nro.engine.bids import replace_bids_entity_token
from nro.engine.images import (
    sidecar_json_path,
)
from nro.engine.io import read_json, write_json
from nro.engine.paths import (
    anatomical_manifest_path,
    clean_manifest_path,
    clean_session_dir,
    clean_session_work_dir,
    clean_subject_dir,
    clean_subject_work_dir,
    is_bids_session_id,
    project_data_root,
    resolve_cwd_path,
)
from nro.engine.paths import (
    functional_manifest_path as preprocessing_functional_manifest_path,
)
from nro.engine.targets import add_smoothing_entity
from nro.modules.clean.contract import (
    clean_output_contract,
    validate_clean_manifest,
    validate_clean_sidecar,
)
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import (
    ContainerSpec,
    Runner,
    write_completion_breadcrumb,
)
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import selected_configuration_fingerprint

from .cleaning import (
    _build_cleaning_projection,
    _clean_surface_data,
    _clean_volume_data,
    _CleaningProjection,
    _expected_input_groups,
    _filter_confounds,
    _main_sidecar_path,
    _resolve_surface_for_metric,
    _select_outlier_columns,
    _smoothed_desc_label,
    _space_name,
    _volume_gm_mask_path,
    _write_volume_gm_mask,
    next_step,
)

LOG = logging.getLogger("clean")


def build_module(
    argv: Optional[Sequence[str]],
    *,
    execution_context: ExecutionContext | None = None,
) -> tuple[Runner, bool, int]:
    """Construct cleaning with fixed input owners and an optional output-owner context."""
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
    ap.add_argument(
        "--nuisance-variance-explained",
        type=float,
        default=float(cfg.nuisance_variance_explained),
        metavar="P",
    )
    ap.add_argument(
        "--minimum-temporal-rank",
        type=int,
        default=int(cfg.minimum_temporal_rank),
        metavar="N",
    )
    ap.add_argument(
        "--minimum-temporal-rank-fraction",
        type=float,
        default=float(cfg.minimum_temporal_rank_fraction),
        metavar="F",
    )
    ap.add_argument("--temporal-mask-regex", default=str(cfg.temporal_mask_regex))
    ap.add_argument("--standardize", action="store_true", default=bool(cfg.standardize))
    ap.add_argument("--no-standardize", action="store_false", dest="standardize")
    ap.add_argument("--detrend", action="store_true", default=bool(cfg.detrend))
    ap.add_argument("--no-detrend", action="store_false", dest="detrend")
    ap.add_argument(
        "--regress-out-task",
        action="store_true",
        default=bool(cfg.regress_out_task),
    )
    ap.add_argument("--no-regress-out-task", action="store_false", dest="regress_out_task")
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
    if execution_context is not None and execution_context.project != args.project:
        raise ValueError("Cleaning project differs from its execution context")
    bids_root = None if execution_context is None else execution_context.paths.bids
    work_root = None if execution_context is None else execution_context.paths.work

    if args.smoothing < 0:
        raise SystemExit("--smoothing must be a nonnegative integer FWHM in mm")
    if not 0.0 < args.nuisance_variance_explained <= 1.0:
        raise SystemExit("--nuisance-variance-explained must be in (0, 1]")
    if args.minimum_temporal_rank < 0:
        raise SystemExit("--minimum-temporal-rank must be nonnegative")
    if not 0.0 <= args.minimum_temporal_rank_fraction <= 1.0:
        raise SystemExit("--minimum-temporal-rank-fraction must be in [0, 1]")
    smoothing_mm = int(args.smoothing)
    ses_id = str(args.ses_id).strip() if args.ses_id is not None else None
    if ses_id == "":
        ses_id = None
    if ses_id is not None and not is_bids_session_id(ses_id):
        raise SystemExit(f"--ses-id must look like a BIDS session ID (ses-*), got {ses_id!r}")

    if ses_id is None:
        clean_dir = clean_subject_dir(
            args.sub_id, project=args.project, clean_id=args.clean_id, bids_root=bids_root
        )
        work_dir = (
            clean_subject_work_dir(
                args.sub_id, project=args.project, clean_id=args.clean_id, work_root=work_root
            )
            / args.run_stem
            / f"space-{args.space}_smoothing-{smoothing_mm}mm"
        )
    else:
        clean_dir = clean_session_dir(
            args.sub_id, ses_id, project=args.project, clean_id=args.clean_id, bids_root=bids_root
        )
        work_dir = (
            clean_session_work_dir(
                args.sub_id,
                ses_id,
                project=args.project,
                clean_id=args.clean_id,
                work_root=work_root,
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
        bids_root=bids_root,
    )
    if execution_context is not None:
        functional_manifest = execution_context.input_path(functional_manifest)
        clean_dir = execution_context.output_path(clean_dir)
        work_dir = execution_context.output_path(work_dir, private=True)
    func_dir = functional_manifest.parent
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
    recorded_clean_inputs = (functional_contract.get("public_outputs") or {}).get(
        "clean_inputs"
    ) or {}
    recorded_paths = json.dumps(recorded_clean_inputs, sort_keys=True)
    selected_paths = [
        str(path) for variant in variants for path in (*variant["vols"], *variant["surfs"])
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
    main_sidecar = read_json(_main_sidecar_path(primary_volumes, primary_surfaces))
    tr = float(main_sidecar.get("RepetitionTime", 0) or 0)
    if tr <= 0:
        raise SystemExit("RepetitionTime was not found in the preprocessed sidecar.")
    start_time = float(main_sidecar.get("StartTime", 0.0) or 0.0)

    anatomical_path = anatomical_manifest_path(
        args.sub_id,
        project=args.project,
        preprocessing_id=args.preprocessing_id,
        bids_root=bids_root,
    )
    if execution_context is not None:
        anatomical_path = execution_context.input_path(anatomical_path)
    anatomical = read_json(anatomical_path)
    anat_outputs = anatomical.get("outputs") or {}
    anat_gray_matter = Path(str(anat_outputs.get("gray_matter_mask") or "").strip())
    if not anat_gray_matter.exists():
        raise SystemExit(
            f"Missing anatomical gray matter mask in anatomical manifest: {anat_gray_matter}"
        )
    mni_template = Path(str(anatomical.get("mni_template", "")).strip())
    if not mni_template.exists() and any(
        _space_name(path).startswith("MNI") for path in primary_volumes
    ):
        raise SystemExit(f"Missing MNI template in anatomical manifest: {mni_template}")
    mni_gray_matter = Path(str(mni_template).replace("_T1w.nii.gz", "_label-GM_probseg.nii.gz"))
    if not mni_gray_matter.exists() and any(
        _space_name(path).startswith("MNI") for path in primary_volumes
    ):
        raise SystemExit(f"Missing MNI gray matter probability template: {mni_gray_matter}")
    templateflow_root = mni_template.parent.parent
    fsaverage_surface_cache: dict[tuple[str, int], Path] = {}

    confounds_tsv = func_dir / f"{args.run_stem}_desc-confounds_timeseries.tsv"
    confounds_json = func_dir / f"{args.run_stem}_desc-confounds_timeseries.json"
    if not confounds_tsv.exists():
        raise SystemExit(f"Missing confounds TSV: {confounds_tsv}")
    source_root = project_data_root(args.project, bids_root=bids_root) / args.sub_id
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
        bids_root=bids_root,
    )
    if execution_context is not None:
        publication_manifest = execution_context.output_path(publication_manifest)
    confounds_basename = (
        f"{args.run_stem}_space-{args.space}_smoothing-{smoothing_mm}mm_desc-confounds_timeseries"
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
        "nuisance_variance_explained": float(args.nuisance_variance_explained),
        "minimum_temporal_rank": int(args.minimum_temporal_rank),
        "minimum_temporal_rank_fraction": float(args.minimum_temporal_rank_fraction),
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
    if execution_context is not None and container_home is not None:
        container_home = execution_context.output_path(container_home, private=True)
    container = (
        None
        if args.no_container
        else ContainerSpec(
            image=Path(resolve_cwd_path(args.container) or args.container),
            engine=str(args.container_engine),
            cleanenv=not bool(args.container_no_cleanenv),
            extra_binds=tuple(str(value) for value in (args.container_bind or [])),
            home_dir=container_home,
            inner_setup=str(args.container_inner_setup or ""),
        )
    )
    runner = Runner(
        module_name="Cleaning Module",
        container=container,
        binds=(),
        logger=LOG,
        next_step=next_step,
        execution_context=execution_context,
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
    task_regressors_path = work_dir / "task_regressors.tsv"
    selected_confounds_json = work_dir / "selected_confounds.json"
    outlier_confounds_path = work_dir / "outlier_confounds.tsv"

    def prepare_confounds() -> None:
        selected, task, sidecar = _filter_confounds(
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
        task.to_csv(task_regressors_path, sep="\t", index=False)
        outliers = _select_outlier_columns(
            confounds_tsv=confounds_tsv,
            regex=str(args.temporal_mask_regex),
            n_scans=sample_count,
        )
        outliers.to_csv(outlier_confounds_path, sep="\t", index=False)
        write_json(selected_confounds_json, sidecar)

    runner.add_step(
        configured(
            Step.python(
                name="Prepare Confounds",
                outputs=(
                    selected_confounds_path,
                    task_regressors_path,
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
        )
    )

    projection_cache: dict[str, object] = {}

    def projection_state() -> tuple[_CleaningProjection, dict[str, object]]:
        if "projection" in projection_cache:
            return projection_cache["projection"], projection_cache["metadata"]
        selected = pd.read_csv(selected_confounds_path, sep="\t").fillna(0.0)
        try:
            task = pd.read_csv(task_regressors_path, sep="\t").fillna(0.0)
        except pd.errors.EmptyDataError:
            task = pd.DataFrame(index=np.arange(sample_count))
        try:
            outliers = pd.read_csv(outlier_confounds_path, sep="\t").fillna(0.0)
        except pd.errors.EmptyDataError:
            outliers = pd.DataFrame(index=np.arange(sample_count))
        projection = _build_cleaning_projection(
            confounds=selected,
            task=task,
            outliers=outliers,
            tr=tr,
            detrend=bool(args.detrend),
            standardize=bool(args.standardize),
            low_pass=args.low_pass,
            high_pass=args.high_pass,
            nuisance_variance_explained=float(args.nuisance_variance_explained),
            minimum_temporal_rank=int(args.minimum_temporal_rank),
            minimum_temporal_rank_fraction=float(args.minimum_temporal_rank_fraction),
        )
        metadata: dict[str, object] = {
            "TemporalMaskRegex": str(args.temporal_mask_regex),
            "TemporalMaskFile": str(confounds_out),
            **projection.metadata(tr=tr, outlier_columns=len(outliers.columns)),
            "TemporalFilterMethod": "censored-fit clean-passband reconstruction",
        }
        projection_cache["projection"] = projection
        projection_cache["metadata"] = metadata
        return projection, metadata

    projection_inputs = (
        selected_confounds_path,
        task_regressors_path,
        outlier_confounds_path,
    )
    expected_outputs: list[Path] = []
    cleaned_outputs: list[Path] = []
    masks_by_space: dict[str, Path] = {}

    def cleaning_metadata(
        *,
        source: Path,
        clean_input: Path,
        input_desc: str,
        gm_mask: Path | None,
        quality_path: Path,
    ) -> dict[str, object]:
        sidecar = dict(read_json(sidecar_json_path(source)))
        projection, projection_metadata = projection_state()
        sidecar["Description"] = (
            "Cleaned BOLD timeseries for connectivity analysis."
            if projection.cleaning_defined
            else (
                "All-zero sentinel: cleaning is mathematically undefined for this "
                "run and configuration; see Cleaning.CleaningUndefinedReason."
            )
        )
        sidecar["Sources"] = [str(source), str(confounds_tsv)] + (
            [str(events_path)] if args.regress_out_task and events_path.exists() else []
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
            "RegressOutTask": bool(args.regress_out_task and events_path.exists()),
            "LowPassHz": args.low_pass,
            "HighPassHz": args.high_pass,
            "ConfigurationFingerprint": selected_configuration_fingerprint(),
            "QualityControl": read_json(quality_path),
        }
        validate_clean_sidecar(sidecar, volume=gm_mask is not None)
        return sidecar

    def validate_cleaning_metadata(
        path: Path,
        *,
        volume: bool,
    ) -> tuple[bool, str]:
        try:
            validate_clean_sidecar(read_json(path), volume=volume)
        except (OSError, TypeError, ValueError) as error:
            return False, str(error)
        return True, "Cleaned sidecar satisfies its artifact metadata contract."

    for variant in variants:
        input_desc = str(variant["input_desc"])
        output_desc = str(variant["output_desc"])
        for volume in variant["vols"]:
            out_path = (
                clean_dir
                / add_smoothing_entity(
                    replace_bids_entity_token(volume, input_desc, output_desc),
                    smoothing_mm,
                ).name
            )
            gm_source = (
                mni_gray_matter if _space_name(volume).startswith("MNI") else anat_gray_matter
            )
            gm_mask, new_mask = _volume_gm_mask_path(
                masks_by_space=masks_by_space,
                target_img=volume,
                work_dir=work_dir,
                input_desc=input_desc,
            )
            if new_mask:
                runner.add_step(
                    configured(
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
                    )
                )

            clean_input = volume
            if smoothing_mm > 0:
                clean_input = (
                    work_dir
                    / replace_bids_entity_token(
                        volume,
                        input_desc,
                        _smoothed_desc_label(input_desc),
                    ).name
                )
                runner.add_step(
                    configured(
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
                    )
                )

            quality_path = work_dir / f"{out_path.name}.quality.json"
            runner.add_step(
                configured(
                    # The QC summary is written while the cleaned array is already
                    # resident, avoiding a second read of a potentially large image.
                    Step.python(
                        name=f"Clean Volume: {_space_name(volume)}",
                        outputs=(
                            out_path,
                            quality_path,
                        ),
                        inputs=(clean_input, gm_mask, *projection_inputs),
                        force=bool(args.force),
                        action=lambda source=clean_input, output=out_path, mask_path=gm_mask, quality=quality_path: (
                            _clean_volume_data(
                                in_img=source,
                                out_img=output,
                                mask_img=mask_path,
                                projection=projection_state()[0],
                                quality_path=quality,
                            )
                        ),
                    )
                )
            )
            metadata_path = sidecar_json_path(out_path)
            runner.add_step(
                configured(
                    Step.python(
                        name=f"Write Cleaned Volume Metadata: {out_path.name}",
                        outputs=(metadata_path,),
                        inputs=(
                            out_path,
                            quality_path,
                            sidecar_json_path(volume),
                            confounds_tsv,
                            *projection_inputs,
                            *((events_path,) if events_path.exists() else ()),
                        ),
                        force=bool(args.force),
                        action=lambda path=metadata_path, source=volume, cleaned=clean_input, desc=input_desc, mask_path=gm_mask, quality=quality_path: (
                            write_json(
                                path,
                                cleaning_metadata(
                                    source=source,
                                    clean_input=cleaned,
                                    input_desc=desc,
                                    gm_mask=mask_path,
                                    quality_path=quality,
                                ),
                            )
                        ),
                        validate=lambda path=metadata_path: validate_cleaning_metadata(
                            path, volume=True
                        ),
                    )
                )
            )
            cleaned_outputs.append(out_path)
            expected_outputs.extend((out_path, metadata_path))

        for left in variant["surfs"]:
            right = Path(str(left).replace("_hemi-L_", "_hemi-R_"))
            if not right.exists():
                raise SystemExit(f"Missing right-hemisphere pair for {left.name}: {right}")
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
                    clean_input = (
                        work_dir
                        / replace_bids_entity_token(
                            surface,
                            input_desc,
                            _smoothed_desc_label(input_desc),
                        ).name
                    )
                    runner.add_step(
                        configured(
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
                        )
                    )

                out_path = (
                    clean_dir
                    / add_smoothing_entity(
                        replace_bids_entity_token(surface, input_desc, output_desc),
                        smoothing_mm,
                    ).name
                )
                quality_path = work_dir / f"{out_path.name}.quality.json"
                runner.add_step(
                    configured(
                        Step.python(
                            name=f"Clean Surface: {_space_name(surface)}",
                            outputs=(
                                out_path,
                                quality_path,
                            ),
                            inputs=(clean_input, *projection_inputs),
                            force=bool(args.force),
                            action=lambda source=clean_input, output=out_path, scans=counts[surface], quality=quality_path: (
                                _clean_surface_data(
                                    in_img=source,
                                    out_img=output,
                                    projection=projection_state()[0],
                                    n_scans=scans,
                                    quality_path=quality,
                                )
                            ),
                        )
                    )
                )
                metadata_path = sidecar_json_path(out_path)
                runner.add_step(
                    configured(
                        Step.python(
                            name=f"Write Cleaned Surface Metadata: {out_path.name}",
                            outputs=(metadata_path,),
                            inputs=(
                                out_path,
                                quality_path,
                                sidecar_json_path(surface),
                                confounds_tsv,
                                *projection_inputs,
                                *((events_path,) if events_path.exists() else ()),
                            ),
                            force=bool(args.force),
                            action=lambda path=metadata_path, source=surface, cleaned=clean_input, desc=input_desc, quality=quality_path: (
                                write_json(
                                    path,
                                    cleaning_metadata(
                                        source=source,
                                        clean_input=cleaned,
                                        input_desc=desc,
                                        gm_mask=None,
                                        quality_path=quality,
                                    ),
                                )
                            ),
                            validate=lambda path=metadata_path: validate_cleaning_metadata(
                                path, volume=False
                            ),
                        )
                    )
                )
                cleaned_outputs.append(out_path)
                expected_outputs.extend((out_path, metadata_path))

    def finalize_confounds() -> None:
        selected = pd.read_csv(selected_confounds_path, sep="\t").fillna(0.0)
        try:
            task = pd.read_csv(task_regressors_path, sep="\t").fillna(0.0)
        except pd.errors.EmptyDataError:
            task = pd.DataFrame(index=np.arange(sample_count))
        try:
            outliers = pd.read_csv(outlier_confounds_path, sep="\t").fillna(0.0)
        except pd.errors.EmptyDataError:
            outliers = pd.DataFrame(index=np.arange(sample_count))
        published = pd.concat([task, selected, outliers], axis=1)
        published.to_csv(confounds_out, sep="\t", index=False)
        sidecar = read_json(selected_confounds_json)
        sidecar["TemporalMask"] = projection_state()[1]
        sidecar["Columns"] = list(published.columns)
        write_json(confounds_out_json, sidecar)

    runner.add_step(
        configured(
            Step.python(
                name="Finalize Confounds",
                outputs=(confounds_out, confounds_out_json),
                inputs=(
                    selected_confounds_path,
                    task_regressors_path,
                    selected_confounds_json,
                    outlier_confounds_path,
                ),
                force=bool(args.force),
                action=finalize_confounds,
            )
        )
    )
    expected_outputs.extend((confounds_out, confounds_out_json))

    clean_targets: list[dict[str, object]] = []
    for variant in variants:
        input_desc = str(variant["input_desc"])
        output_desc = str(variant["output_desc"])
        variant_name = output_desc.removeprefix("desc-")
        for volume in variant["vols"]:
            output = (
                clean_dir
                / add_smoothing_entity(
                    replace_bids_entity_token(volume, input_desc, output_desc),
                    smoothing_mm,
                ).name
            )
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
            output_left = (
                clean_dir
                / add_smoothing_entity(
                    replace_bids_entity_token(left, input_desc, output_desc),
                    smoothing_mm,
                ).name
            )
            output_right = (
                clean_dir
                / add_smoothing_entity(
                    replace_bids_entity_token(
                        Path(str(left).replace("_hemi-L_", "_hemi-R_")),
                        input_desc,
                        output_desc,
                    ),
                    smoothing_mm,
                ).name
            )
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
        "source_bold": str((functional_contract.get("inputs") or {}).get("epi") or ""),
        "targets": clean_targets,
        "public_outputs": [str(path) for path in expected_outputs],
        "output_metadata_contract": clean_output_contract(),
        "configuration": configuration,
        "configuration_fingerprint": selected_configuration_fingerprint(),
        "complete": True,
    }

    def validate_publication() -> tuple[bool, str]:
        try:
            actual = read_json(publication_manifest)
            validate_clean_manifest(actual)
        except (OSError, ValueError, TypeError):
            return False, f"Cleaning publication manifest is unreadable: {publication_manifest}"
        if actual != clean_contract:
            return False, "Cleaning publication manifest differs from the requested module."
        missing_public = [
            str(path) for path in expected_outputs if not path.is_file() or path.stat().st_size == 0
        ]
        if missing_public:
            return False, "Cleaning publication is missing outputs: " + ", ".join(missing_public)
        return True, "Cleaning publication is complete and current."

    def publish_clean_manifest() -> None:
        validate_clean_manifest(clean_contract)
        write_json(publication_manifest, clean_contract)

    runner.add_step(
        configured(
            Step.python(
                name="Write Cleaning Publication Manifest",
                outputs=(publication_manifest,),
                inputs=tuple(expected_outputs) + tuple(source_inputs),
                force=bool(args.force),
                action=publish_clean_manifest,
                validate=validate_publication,
                completion_boundary=True,
            )
        )
    )

    LOG.info(
        "Constructed cleaning module for %s in space-%s at %dmm smoothing.",
        args.run_stem,
        args.space,
        smoothing_mm,
    )
    return runner, bool(args.verbose), len(cleaned_outputs)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    execution_context: ExecutionContext | None = None,
) -> int:
    """Build and execute one cleaning request with optional authorized branch bindings."""
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
    runner, verbose, output_count = build_module(argv, execution_context=execution_context)
    with runner.run_context(started_at=runner_started):
        runner.execute()
    if verbose:
        print(f"Cleaned {output_count} functional outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
