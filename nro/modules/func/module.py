"""Build the preprocessing graph for one BOLD run.

- Susceptibility distortion correction (SDC) from a blip-reversed spin-echo EPI pair using FSL topup
  with explicit warp outputs (topup --dfout/--jacout).
- Construct a robust BOLD reference with provisional motion correction, then estimate the final
  MCFLIRT transforms directly from the raw BOLD to that reference.
- Optionally diagnose and remove signal shared by simultaneous multiband slices in native space
  with the official MARSS implementation before the ordinary preprocessing graph.
- Qualify an available SBRef against the robust BOLD reference; use the robust reference when the
  SBRef is absent or fails metadata, geometry, transform, overlap, or similarity checks.
- Derive a target-acquisition warp from TOPUP's Hz field using the functional readout time, keep
  TOPUP in a stable fieldmap frame, and refine the corrected functional-reference pose against
  the corrected spin-echo median before conjugating the complete SDC chain into reference space.
- Boundary-based registration (FreeSurfer bbregister) on the selected undistorted reference.
- Final ANTs SyN refinement is always estimated before the final BOLD-to-T1 resampling.
- Single-interpolation 4D resampling with the complete spatial warp and one
  MCFLIRT transform per volume using AFNI 3dNwarpApply.
- Optional ICA component classification with ICA-AROMA or CICADA, followed by
  matched component regression (ICA-AROMA by default).
- Confounds TSV/JSON generation from the registered or ICA-AROMA-cleaned BOLD.

Requirements:
- FSL must support `topup --dfout` and `topup --jacout`.
- Any selected SBRef must have phase-encoding and readout metadata compatible with the BOLD run.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

from nro.configuration.hardware import resolve_gradient_unwarping
from nro.configuration.runtime import SETTINGS
from nro.configuration.site import settings as site_settings
from nro.engine.anatomical_domain import load_anatomical_domain
from nro.engine.bids import (
    bids_entity,
    bids_readout_time,
)
from nro.engine.container import (
    ContainerSettings,
    add_container_arguments,
    bind_container_to_execution,
    build_container,
)
from nro.engine.execution import (
    collect_bind_directories,
    create_copy_file_step,
    ensure_directory,
    neuroimaging_environment,
    require_existing_path,
)
from nro.engine.image_paths import nifti_stem, sidecar_json_path, uncompressed_nifti_path
from nro.engine.images import (
    nifti_spatial_shape,
    nifti_volume_count,
)
from nro.engine.io import read_json, write_json
from nro.engine.manifests import (
    create_json_step,
    require_nested_manifest_output,
)
from nro.engine.neuroimaging import (
    create_copy_nifti_step,
    create_flirt_transform_step,
    create_identity_transform_step,
    create_image_support_mask_step,
    create_mask_resampling_step,
    create_n4_bias_correction_step,
    create_native_overlap_mask_step,
    create_nifti_volume_extraction_step,
)
from nro.engine.paths import (
    anatomical_manifest_path,
    func_session_dir,
    func_session_work_dir,
    func_subject_dir,
    func_subject_work_dir,
    functional_manifest_path,
    is_bids_session_id,
    module_derivatives_root,
    resolve_cwd_path,
    resolve_project_path,
    resolve_project_work_path,
)
from nro.engine.targets import supported_output_spaces
from nro.engine.templates import find_fsaverage_template_surface
from nro.modules.func.contract import (
    final_resampling_metadata,
    functional_output_contract,
    validate_functional_image_sidecar,
    validate_functional_manifest,
)
from nro.modules.func.resolver import (
    ResolvedFuncRun,
    resolve_func_run_request,
)
from nro.modules.func.synbold_disco import create_synthetic_reference_step, ensure_image
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

from .config import Inputs, Options, normalize_output_spaces
from .constants import (
    _FIELDMAP_TRANSFER_POLICY_VERSION,
    _ICA_AROMA_MELODIC_MASK_DILATION_MM,
    _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
)
from .construction import (
    plan_initialization,
    plan_input_preparation,
    plan_reference_preparation,
)
from .denoising_steps import (
    _create_cicada_classification_step,
    _create_cicada_melodic_step,
    _create_cicada_motion_metrics_step,
    _create_confounds_step,
    _create_dilated_anatomical_mask_step,
    _create_epi_support_step,
    _create_ica_aroma_workflow_step,
    _create_melodic_smoothing_step,
    _create_shared_aroma_regression_step,
    _ica_aroma_policy_payload,
    _ica_aroma_shared_regression_policy_payload,
)
from .registration_steps import (
    _ants_pe_aligned_frame,
    _create_ants_composite_to_itk_warp_step,
    _create_ants_registration_step,
    _create_ants_synboldaff_step,
    _create_bbregister_step,
    _create_bold_ref_to_topup_transform_step,
    _create_concat_mats_step,
    _create_flirt_registration_step,
    _create_invert_mat_step,
    _create_local_rigid_refinement_step,
    _create_mni_2mm_target_step,
    _create_mris_convert_step,
    _create_nifti_in_ants_frame_step,
    _create_restore_ants_warp_step,
    _create_synbold_rigid_registration_step,
    _create_t1_epi_vox_target_step,
    _create_t1_native_target_step,
    _create_target_readout_warp_step,
    _create_target_shift_step,
    _create_tkregister2_regheader_fslmat_step,
    _create_wb_convert_itk_warp_to_fnirt_step,
)
from .resampling_steps import (
    _create_afni_bold_resampling_step,
    _create_afni_motion_affines_step,
    _create_afni_warp_step,
    _create_applywarp_step,
    _create_temporal_mean_step,
    _create_world_warp_step,
)
from .sdc_steps import (
    TopupDfOutputs,
    _canonical_fieldmap_order,
    _create_average_jacobian_step,
    _create_convertwarp_conjugate_affine_step,
    _create_convertwarp_merge_warps_step,
    _create_convertwarp_postmat_step,
    _create_convertwarp_premat_and_warp_step,
    _create_convertwarp_premat_step,
    _create_topup_dfout_step,
    _create_warp_jacobian_step,
    _normalized_topup_matrix,
    _pe_to_fsl_shift_direction,
)
from .step_support import (
    _ica_aroma_output_label,
    _resolve_fieldmapless_sdc_method,
    _resolve_sdc_reference_policy,
    _with_suffix,
)
from .surface_steps import (
    _create_wb_metric_mask_step,
    _create_wb_metric_resample_step,
    _create_wb_volume_to_surface_mapping_step,
)

LOG = logging.getLogger("func")


def build_module(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> Runner:
    """Construct the functional DAG using fixed anatomy and output-owner selections.

    The caller authorizes an optional execution context. Anatomical paths come
    from the selected producer's manifest; all new outputs belong to this attempt.
    """
    source_inputs = inputs
    if execution_context is not None:
        if execution_context.project != opts.project:
            raise ValueError("Functional project differs from its execution context")
        opts = replace(
            opts,
            out_dir=execution_context.output_path(opts.out_dir),
            work_dir=execution_context.output_path(opts.work_dir, private=True),
        )
    classifier = str(opts.ica_classifier).strip().lower()
    if classifier not in {"none", "ica_aroma", "cicada"}:
        raise SystemExit(f"Unsupported ICA classifier: {classifier!r}")
    denoise_type = {
        "aggressive": "aggr",
        "nonaggressive": "nonaggr",
    }.get(str(opts.ica_regression).strip().lower())
    if denoise_type is None:
        raise SystemExit(f"Unsupported ICA regression policy: {opts.ica_regression!r}")
    ica_enabled = classifier != "none"
    require_existing_path(inputs.epi, "epi")
    require_existing_path(inputs.epi_json, "epi-json")
    assert inputs.epi_json is not None
    epi_metadata_sources = inputs.epi_metadata_sources or (inputs.epi_json,)
    for source in epi_metadata_sources:
        require_existing_path(source, "EPI metadata source")
    epi_input_meta = (
        dict(inputs.epi_metadata) if inputs.epi_metadata is not None else read_json(inputs.epi_json)
    )
    sbref_metadata_sources = inputs.sbref_metadata_sources or (
        (inputs.sbref_json,) if inputs.sbref_json is not None else ()
    )
    se1_metadata_sources = inputs.se1_metadata_sources or (
        (inputs.se1_json,) if inputs.se1_json is not None else ()
    )
    se2_metadata_sources = inputs.se2_metadata_sources or (
        (inputs.se2_json,) if inputs.se2_json is not None else ()
    )

    def acquisition_metadata(
        path: Path | None,
        explicit: dict[str, Any] | None,
        sidecar: Path | None,
    ) -> dict[str, Any]:
        if path is None:
            return {}
        if explicit is not None:
            return dict(explicit)
        return read_json(sidecar) if sidecar is not None and sidecar.is_file() else {}

    gradient_resolutions = {
        "epi": resolve_gradient_unwarping(epi_input_meta, mode=opts.gradient_unwarping),
        "sbref": resolve_gradient_unwarping(
            acquisition_metadata(inputs.sbref, inputs.sbref_metadata, inputs.sbref_json),
            mode=opts.gradient_unwarping,
        ),
        "se1": resolve_gradient_unwarping(
            acquisition_metadata(inputs.se1, inputs.se1_metadata, inputs.se1_json),
            mode=opts.gradient_unwarping,
        ),
        "se2": resolve_gradient_unwarping(
            acquisition_metadata(inputs.se2, inputs.se2_metadata, inputs.se2_json),
            mode=opts.gradient_unwarping,
        ),
    }
    gradient_resolution = gradient_resolutions["epi"]
    if any(resolution.applied for resolution in gradient_resolutions.values()):
        require_existing_path(opts.gradient_unwarp_image, "gradient-unwarping image")
    anat_manifest = anatomical_manifest_path(
        opts.sub_id,
        project=opts.project,
        anat_id=opts.anat_id,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        anat_manifest = execution_context.input_path(anat_manifest)
    if not anat_manifest.exists():
        raise SystemExit(
            "Anatomical preprocessing must be completed before functional preprocessing.\n"
            f"Missing anatomical manifest: {anat_manifest}"
        )
    try:
        anatomical_domain, anat_info = load_anatomical_domain(anat_manifest)
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(str(error)) from error
    subjects_dir = anatomical_domain.subjects_dir
    fs_subject = anatomical_domain.fs_subject
    anat_t1 = anatomical_domain.observed_t1w
    registration_t1 = anatomical_domain.registration_t1w
    anat_brain_mask = anatomical_domain.brain_mask
    anat_mni_template = Path(str(anat_info.get("mni_template", "")).strip())
    require_existing_path(anat_mni_template, "MNI template from anatomical manifest")
    t1_to_mni_xfm = require_nested_manifest_output(
        anat_info,
        "xfms",
        "acpc_to_mni",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    require_nested_manifest_output(
        anat_info,
        "xfms",
        "mni_to_acpc",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    fsnative_surfaces = {
        f"{hemi}.{surface}": anatomical_domain.surfaces[f"{source_hemi}.{surface}"]
        for hemi, source_hemi in (("L", "lh"), ("R", "rh"))
        for surface in ("white", "pial", "midthickness")
    }
    env = neuroimaging_environment(subjects_dir=subjects_dir)
    raw_repetition_time = epi_input_meta.get("RepetitionTime")
    try:
        repetition_time = float(raw_repetition_time) if raw_repetition_time is not None else None
    except (TypeError, ValueError):
        repetition_time = None
    if repetition_time is not None and repetition_time <= 0:
        repetition_time = None
    binds = collect_bind_directories(
        [
            inputs.sbref,
            inputs.epi,
            inputs.se1,
            inputs.se2,
            inputs.se1_json,
            inputs.se2_json,
            inputs.epi_json,
            inputs.sbref_json,
            *epi_metadata_sources,
            *sbref_metadata_sources,
            *se1_metadata_sources,
            *se2_metadata_sources,
            anat_t1,
            registration_t1,
            anat_brain_mask,
            opts.ica_aroma_cmd,
            opts.cicada_cmd if classifier == "cicada" else None,
            (subjects_dir if subjects_dir.exists() else None),
            Path(env["FS_LICENSE"]) if Path(env["FS_LICENSE"]).is_file() else None,
            opts.out_dir,
            opts.work_dir,
            opts.gradient_unwarp_image if gradient_resolution.applied else None,
            *(
                resolution.coefficients
                for resolution in gradient_resolutions.values()
                if resolution.applied
            ),
        ]
    )
    runner = Runner(
        module_name="Functional Module",
        container=bind_container_to_execution(opts.container, execution_context),
        binds=binds,
        logger=LOG,
        execution_context=execution_context,
    )
    derivative_root = module_derivatives_root(
        "func",
        opts.func_id,
        project=opts.project,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        derivative_root = execution_context.output_path(derivative_root)
    fieldmap_pair_available = all(
        p is not None
        for p in (
            inputs.se1,
            inputs.se2,
            inputs.se1_json,
            inputs.se2_json,
        )
    )
    requested_sdc_method = opts.sdc_method.strip().lower()
    if requested_sdc_method not in {"syn", "synbold_disco"}:
        raise SystemExit(f"Unknown SDC method {opts.sdc_method!r}; expected syn or synbold_disco.")
    use_fieldmap_sdc = fieldmap_pair_available
    sdc_method, sdc_fallback_reason = _resolve_fieldmapless_sdc_method(
        requested_sdc_method,
        fieldmap_pair_available=use_fieldmap_sdc,
        bold_metadata=epi_input_meta,
    )
    if sdc_fallback_reason is not None:
        LOG.warning("Susceptibility distortion correction fallback: %s", sdc_fallback_reason)
    use_synbold_disco = sdc_method == "synbold_disco"
    use_synbold_topup = use_synbold_disco and not use_fieldmap_sdc
    use_syn_fallback = sdc_method == "syn" and not use_fieldmap_sdc
    use_synbold_reference, fieldmap_refinement_target = _resolve_sdc_reference_policy(
        requested_sdc_method=sdc_method,
        fieldmap_pair_available=fieldmap_pair_available,
        fieldmap_syn_refine=opts.fieldmap_syn_refine,
    )
    do_refinement = fieldmap_refinement_target is not None
    resolved_sdc_method = "topup_sbref_pe" if use_fieldmap_sdc else sdc_method
    base_cmds = [
        "fslmerge",
        "fslmaths",
        "fslroi",
        "mcflirt",
        "flirt",
        "convert_xfm",
        "mri_convert",
        "mris_convert",
        "antsRegistration",
        "antsApplyTransforms",
        "convertwarp",
        "applywarp",
        "wb_command",
        "N4BiasFieldCorrection",
        "3dNwarpApply",
        "fslcpgeom",
    ]
    if ica_enabled:
        base_cmds.extend(("bet", "fsl_regfilt", "fslinfo", "fslstats", "melodic"))
    initialization = plan_initialization(
        opts=opts,
        derivative_root=derivative_root,
        gradient_unwarping=gradient_resolution,
        required_commands=(
            base_cmds if use_syn_fallback else [*base_cmds, "topup", "bbregister", "tkregister2"]
        ),
        require_commands=runner.require_cmds,
    )
    runner.add_steps(initialization.steps)
    configuration = initialization.products.configuration
    LOG.info(
        "Susceptibility distortion correction: base=%s refinement=%s",
        "fieldmap" if use_fieldmap_sdc else sdc_method,
        fieldmap_refinement_target or "none",
    )
    LOG.info(
        "SynBOLD-DisCo synthetic reference: %s",
        "enabled (fieldmapless fallback)" if use_synbold_reference else "disabled",
    )
    run_stem = nifti_stem(inputs.epi)
    run_base = run_stem[: -len("_bold")] if run_stem.endswith("_bold") else run_stem
    run_prefix = f"{run_base}_space-ACPC"
    source_volume_count = nifti_volume_count(inputs.epi)
    source_spatial_shape = nifti_spatial_shape(inputs.epi)
    func_dir = opts.out_dir
    fmap_dir = func_dir.parent / "fmap"
    sdc_dir = opts.work_dir / "sdc"
    reg_dir = opts.work_dir / "reg"
    mc_dir = opts.work_dir / "mc"
    surf_dir = opts.work_dir / "surf"
    qc_dir = opts.work_dir / "qc"
    preparation = plan_input_preparation(
        opts=opts,
        inputs=inputs,
        epi_metadata=epi_input_meta,
        epi_metadata_sources=epi_metadata_sources,
        gradient_resolutions=gradient_resolutions,
        environment=env,
        run_stem=run_stem,
        run_base=run_base,
        source_volume_count=source_volume_count,
        run_child=runner.run_child,
    )
    runner.add_steps(preparation.steps)
    inputs = preparation.products.inputs
    epi_for_proc = preparation.products.processing_bold
    final_resampling_source = preparation.products.resampling_source
    processing_volume_count = preparation.products.processing_volume_count
    gradient_relative_warp = preparation.products.gradient_relative_warp
    gradient_dir = preparation.products.gradient_directory
    marss_outputs = preparation.products.marss

    reference = plan_reference_preparation(
        opts=opts,
        inputs=inputs,
        processing_bold=epi_for_proc,
        processing_volume_count=processing_volume_count,
        run_stem=run_stem,
        motion_directory=mc_dir,
        environment=env,
        epi_metadata=epi_input_meta,
        epi_metadata_sources=epi_metadata_sources,
        sbref_metadata_sources=sbref_metadata_sources,
        gradient_relative_warp=gradient_relative_warp,
        gradient_directory=gradient_dir,
        run_child=runner.run_child,
    )
    runner.add_steps(reference.steps)
    robust_ref = reference.products.robust_reference
    robust_reference_metadata = reference.products.robust_metadata
    selected_reference_image = reference.products.selected_reference
    epi_to_reference = reference.products.epi_to_reference
    selected_reference_metadata = reference.products.selected_metadata
    gradient_afni_warp = reference.products.gradient_afni_warp
    reg_ref_tag = "regRef"
    reg_ref_space = "FunctionalReference"
    requested_spaces = set(opts.output_spaces)
    want_t1 = "ACPC" in requested_spaces
    want_mni = "MNI152NLin2009cAsym" in requested_spaces
    if classifier == "cicada" and not want_mni:
        raise SystemExit(
            "CICADA classification currently requires MNI152NLin2009cAsym among func output spaces."
        )
    want_fsnative = "fsnative" in requested_spaces
    requested_fsaverage = sorted(
        space for space in requested_spaces if space.startswith("fsaverage")
    )
    if requested_fsaverage and requested_fsaverage != [opts.fsaverage_template]:
        raise SystemExit(
            "Functional output spaces request "
            + ", ".join(requested_fsaverage)
            + f", but anatomy selects {opts.fsaverage_template}."
        )
    fsaverage_space = opts.fsaverage_template
    want_fsaverage = fsaverage_space in requested_spaces
    compute_t1 = True
    need_surface_outputs = want_fsnative or want_fsaverage
    LOG.info("Output spaces requested: %s", ", ".join(opts.output_spaces))
    fsnative_to_fsaverage_spheres: dict[str, Path] = {}
    if want_fsaverage:
        sphere_work = surf_dir / "fsaverage_spheres"
        for hemi, hemi_label in (("lh", "L"), ("rh", "R")):
            subject_sphere = require_nested_manifest_output(
                anat_info,
                "surfaces",
                f"{hemi}.sphere.reg",
                manifest_path=anat_manifest,
                manifest_name="Anatomical",
            )
            fsaverage_sphere = find_fsaverage_template_surface(
                template=fsaverage_space,
                hemi=hemi_label,
                surface="sphere",
            )
            subject_output = sphere_work / (
                f"subject_hemi-{hemi_label}_from-fsnative_to-{fsaverage_space}_sphere.surf.gii"
            )
            fsaverage_output = sphere_work / f"{fsaverage_space}_hemi-{hemi_label}_sphere.surf.gii"
            runner.add_step(
                _create_mris_convert_step(
                    source=subject_sphere,
                    output=subject_output,
                    env=env,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                create_copy_file_step(
                    src=fsaverage_sphere,
                    dst=fsaverage_output,
                    force=opts.overwrite,
                    step_name=f"Stage {fsaverage_space} Hemisphere {hemi_label} Sphere",
                )
            )
            fsnative_to_fsaverage_spheres[f"{hemi_label}.current"] = subject_output
            fsnative_to_fsaverage_spheres[f"{hemi_label}.new"] = fsaverage_output

    preproc_t1_name = _with_suffix(run_prefix, "_desc-preproc_bold.nii.gz")
    preproc_t1 = func_dir / preproc_t1_name
    preproc_mni = func_dir / _with_suffix(
        f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_bold.nii.gz"
    )
    preproc_fsnative = {
        "L": func_dir
        / _with_suffix(f"{run_base}_space-fsnative_hemi-L", "_desc-preproc_bold.func.gii"),
        "R": func_dir
        / _with_suffix(f"{run_base}_space-fsnative_hemi-R", "_desc-preproc_bold.func.gii"),
    }
    preproc_fsaverage = {
        "L": func_dir
        / _with_suffix(f"{run_base}_space-{fsaverage_space}_hemi-L", "_desc-preproc_bold.func.gii"),
        "R": func_dir
        / _with_suffix(f"{run_base}_space-{fsaverage_space}_hemi-R", "_desc-preproc_bold.func.gii"),
    }
    epi_t1 = uncompressed_nifti_path(
        reg_dir / _with_suffix(run_prefix, "_desc-preproc_bold.nii.gz")
    )
    epi_mni = uncompressed_nifti_path(
        reg_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_bold.nii.gz")
    )
    epi_mean_t1 = qc_dir / _with_suffix(run_prefix, "_desc-preproc_mean.nii.gz")
    epi_mean_mni = qc_dir / _with_suffix(
        f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_mean.nii.gz"
    )
    boldref_t1_out = func_dir / _with_suffix(run_prefix, "_boldref.nii.gz")
    reg_prenonlinear_qc_out = func_dir / _with_suffix(
        run_prefix, "_desc-preNonlinearReg_boldref.nii.gz"
    )
    reg_base_qc_out = func_dir / _with_suffix(run_prefix, "_desc-baseReg_boldref.nii.gz")
    reg_refine_qc_out = func_dir / _with_suffix(run_prefix, "_desc-refineReg_boldref.nii.gz")
    reg_mni_qc_out = func_dir / _with_suffix(
        f"{run_base}_space-MNI152NLin2009cAsym", "_boldref.nii.gz"
    )
    melodic_ic_t1_out = func_dir / _with_suffix(run_prefix, "_desc-melodicIC_bold.nii.gz")
    fieldmap_hz_in_t1_out = fmap_dir / _with_suffix(run_prefix, "_desc-fieldmapHz_fieldmap.nii.gz")
    fmap_field_hz_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-fieldmapHz_fieldmap.nii.gz"
    )
    fmap_sdc_warp_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-sdcwarp_fieldmap.nii.gz"
    )
    fmap_bold_sdc_warp_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-boldSDCwarp_fieldmap.nii.gz"
    )
    fmap_jacobian_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-jacobian_fieldmap.nii.gz"
    )
    fmap_topup_coeff_out = fmap_dir / _with_suffix(run_base, "_desc-topupcoeff_fieldmap.nii.gz")
    fmap_synbold_ref_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-synbold_boldref.nii.gz"
    )
    fmap_synbold_rigid_out = fmap_dir / _with_suffix(
        f"{run_base}_space-ACPC", "_desc-synboldRigid_boldref.nii.gz"
    )
    anat_brain_mask_in_t1 = func_dir / _with_suffix(run_prefix, "_desc-brain_mask.nii.gz")
    anat_brain_mask_in_mni = qc_dir / _with_suffix(
        f"{run_base}_space-MNI152NLin2009cAsym", "_desc-brain_mask.nii.gz"
    )
    aroma_dir = opts.work_dir / ("ica_cicada" if classifier == "cicada" else "ica_aroma")
    aroma_label = (
        ("cicadaAgg" if denoise_type == "aggr" else "cicadaNonAgg")
        if classifier == "cicada"
        else _ica_aroma_output_label(denoise_type)
    )
    aroma_t1_dir = aroma_dir / "space-ACPC"
    aroma_mni_dir = aroma_dir / "space-MNI152NLin2009cAsym"
    aroma_clean = uncompressed_nifti_path(
        aroma_t1_dir / _with_suffix(run_prefix, f"_desc-{aroma_label}_bold.nii.gz")
    )
    aroma_clean_mean = aroma_t1_dir / _with_suffix(run_prefix, f"_desc-{aroma_label}_mean.nii.gz")
    aroma_clean_mni = uncompressed_nifti_path(
        aroma_mni_dir
        / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", f"_desc-{aroma_label}_bold.nii.gz")
    )
    aroma_clean_mean_mni = aroma_mni_dir / _with_suffix(
        f"{run_base}_space-MNI152NLin2009cAsym", f"_desc-{aroma_label}_mean.nii.gz"
    )
    confounds_tsv = func_dir / _with_suffix(run_stem, "_desc-confounds_timeseries.tsv")
    confounds_json = func_dir / _with_suffix(run_stem, "_desc-confounds_timeseries.json")
    publication_manifest = functional_manifest_path(
        opts.sub_id,
        run_base,
        project=opts.project,
        func_id=opts.func_id,
        ses_id=opts.ses_id,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        publication_manifest = execution_context.output_path(publication_manifest)
    t1_to_mni_itk_warp = reg_dir / f"{run_stem}_T1ToMNI_itk_warp.nii.gz"
    t1_to_mni_fnirt_warp = reg_dir / f"{run_stem}_T1ToMNI_fnirt_warp.nii.gz"
    warp_epi2mni = reg_dir / f"{run_stem}_EPIToMNI_warp.nii.gz"
    warp_regref2mni = reg_dir / f"{run_stem}_RegRefToMNI_warp.nii.gz"

    t1_ref = (
        reg_dir / f"{run_stem}_t1_grid_epi_vox.nii.gz"
        if opts.output_grid == "t1_epi_vox"
        else reg_dir / f"{run_stem}_t1_native.nii.gz"
    )
    mni_ref = reg_dir / f"{run_stem}_mni_2mm.nii.gz"
    bbr_t1_ref = (
        reg_dir / f"{run_stem}_fs_t1_grid_epi_vox.nii.gz"
        if opts.output_grid == "t1_epi_vox"
        else reg_dir / f"{run_stem}_fs_t1_native.nii.gz"
    )
    epi_mean_to_t1_fallback_mat = reg_dir / f"{run_stem}_epiMeanAffineToT1.mat"
    reg_ref_in_t1_affine = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_affine.nii.gz"
    fs_t1_to_t1w_mat = reg_dir / f"{run_stem}_fsT1ToT1w.mat"
    reg_ref_to_t1w_mat = reg_dir / f"{run_stem}_{reg_ref_tag}2t1w.mat"
    epi_mean_nodc_t1_affine = reg_dir / f"{run_stem}_epi_mean_nodc_t1_affine.nii.gz"
    reg_ref_in_t1_linear = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_linear.nii.gz"
    reg_ref_dc_ref_n4 = sdc_dir / f"{reg_ref_tag}_dc_ref_n4.nii.gz"
    reg_ref_in_t1_base_n4 = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_base_n4.nii.gz"
    reg_ref_in_t1_refine_n4 = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_refine_n4.nii.gz"
    epi_to_t1_warp_planned = reg_dir / f"{run_stem}_EPIToT1w_warp.nii.gz"
    fallback_affine_warp_planned = reg_dir / f"{run_stem}_SyNAffine_warp.nii.gz"
    fallback_base_itk_planned = reg_dir / f"{run_stem}_SyNBoldAff_itk_warp.nii.gz"
    fallback_base_fnirt_planned = reg_dir / f"{run_stem}_SyNBoldAff_fnirt_warp.nii.gz"
    fallback_base_warp_planned = reg_dir / f"{run_stem}_SyNBoldAffBase_warp.nii.gz"
    warp_regref2t1_planned = reg_dir / f"{run_stem}_{reg_ref_tag}ToT1wBase_warp.nii.gz"
    syn_refine_fnirt_planned = reg_dir / f"{run_stem}_SyNBoldAffRefine_fnirt_warp.nii.gz"
    warp_sbref2t1_refined_planned = epi_to_t1_warp_planned
    warp_regref2t1_refined_planned = reg_dir / f"{run_stem}_{reg_ref_tag}ToT1wRefined_warp.nii.gz"
    field_hz_regref = sdc_dir / f"{run_stem}_fieldmap_Hz_{reg_ref_tag}Space.nii.gz"
    if use_syn_fallback:
        registration_method = "ants_synboldaff"
    elif use_synbold_topup:
        registration_method = "synbold_disco_topup_create_bbregister_step"
    else:
        registration_method = (
            "topup_sbref_pe_create_bbregister_step_then_ants_syn"
            if do_refinement
            else "topup_sbref_pe_create_bbregister_step"
        )
    topup_native: TopupDfOutputs | None = None
    warp_sbref: Optional[Path] = None
    warp_sbref2t1: Optional[Path] = None
    se2sbref_mat: Optional[Path] = None
    sbref2se_mat: Optional[Path] = None
    reg_ref_to_topup_mat: Optional[Path] = None
    topup_to_reg_ref_refined_mat: Optional[Path] = None
    postdc_refine_mat: Optional[Path] = None
    postdc_refine_qc_path: Optional[Path] = None
    postdc_refine_qc: Optional[Path] = None
    pe_residual_fnirt: Optional[Path] = None
    pe_residual_details: Optional[dict[str, object]] = None
    warp_bold_to_reg_ref: Optional[Path] = None
    fieldmap_transfer_details: Optional[dict[str, object]] = None
    fieldmap_hz_in_t1: Optional[Path] = None
    synthetic_ref: Optional[Path] = None
    synbold_rigid_qc: Optional[Path] = None
    ants_forward_xfm: Optional[Path] = None
    epi_mc_ref: Optional[Path] = robust_ref
    reg_ref_dist_ref: Optional[Path] = selected_reference_image
    reg_ref_dc_ref: Optional[Path] = None
    pre_nonlinear_ref_in_t1: Optional[Path] = None
    fsl_mat: Optional[Path] = None
    boldref_t1: Path
    mc_mat_dir = mc_dir / f"{run_stem}_mc.nii.gz.mat"

    if use_syn_fallback:
        reg_ref_dist_ref = selected_reference_image
    else:
        require_existing_path(subjects_dir, "FreeSurfer SUBJECTS_DIR from anatomical manifest")
        reg_ref_dist_ref = selected_reference_image

        topup_label = "fieldmap" if use_fieldmap_sdc else "synbold_disco"
        topup_work_dir = opts.work_dir / f"topup_{topup_label}_native"
        reg_dat = reg_dir / f"{run_stem}_{reg_ref_tag}2t1.dat"
        fsl_mat = reg_dir / f"{run_stem}_{reg_ref_tag}2t1.mat"
        raw_epi_ped = epi_input_meta.get("PhaseEncodingDirection")
        epi_ped = str(raw_epi_ped).strip() if raw_epi_ped is not None else ""
        if not epi_ped:
            raise SystemExit("Missing PhaseEncodingDirection in effective EPI metadata.")
        reg_ref_ped = epi_ped

        assert reg_ref_dist_ref is not None
        if use_synbold_reference:
            try:
                readout = float(bids_readout_time(epi_input_meta))
            except (KeyError, TypeError, ValueError):
                readout = None
            if readout is None and use_synbold_topup:
                raise SystemExit(
                    "SynBOLD-DisCo requires TotalReadoutTime, or EffectiveEchoSpacing with "
                    "a usable phase-encoding matrix size in the effective BIDS metadata."
                )
            synbold_work = opts.work_dir / "synbold_disco"
            t1_brain = synbold_work / "T1_brain.nii.gz"
            t1_brain_cmd = [
                "fslmaths",
                str(registration_t1),
                "-mas",
                str(anat_brain_mask),
                str(t1_brain),
            ]
            runner.add_step(
                Step.command_step(
                    t1_brain_cmd,
                    outputs=(t1_brain,),
                    inputs=(registration_t1, anat_brain_mask),
                    force=opts.overwrite,
                    env=env,
                    name="Prepare SynBOLD-DisCo T1",
                    prepare=lambda: ensure_directory(t1_brain.parent),
                )
            )
            synbold_image = ensure_image(
                image=opts.synbold_disco_image,
            )
            synbold_rigid_step = _create_synbold_rigid_registration_step(
                run_child=runner.run_child,
                distorted_reference=reg_ref_dist_ref,
                anatomical_t1=registration_t1,
                anatomical_mask=anat_brain_mask,
                work_dir=synbold_work / "rigid_registration",
                env=env,
                max_translation_mm=opts.synbold_max_rigid_translation_mm,
                max_rotation_degrees=opts.synbold_max_rigid_rotation_degrees,
                force=opts.overwrite,
            )
            runner.add_step(synbold_rigid_step.step)
            synbold_rigid_mat = synbold_rigid_step.matrix
            synbold_rigid_qc = synbold_rigid_step.registered
            synthetic_step = create_synthetic_reference_step(
                run_child=runner.run_child,
                distorted_reference=reg_ref_dist_ref,
                skull_stripped_t1=t1_brain,
                epi_to_t1_mat=synbold_rigid_mat,
                image=synbold_image,
                license_file=opts.synbold_disco_license,
                engine=opts.synbold_disco_engine,
                work_dir=synbold_work,
                force=opts.overwrite,
            )
            runner.add_step(synthetic_step)
            synthetic_ref = synthetic_step.outputs[0]
        if use_synbold_topup:
            assert synthetic_ref is not None
            topup_native = _create_topup_dfout_step(
                run_child=runner.run_child,
                se_a=reg_ref_dist_ref,
                se_b=synthetic_ref,
                ped_a=reg_ref_ped,
                ped_b=reg_ref_ped,
                readout_time=float(readout),
                readout_time_b=0.0,
                topup_dir=topup_work_dir,
                topup_config=opts.topup_config,
                env=env,
                force=opts.overwrite,
                spatial_shape=source_spatial_shape,
                volumes_a=1,
                volumes_b=1,
            )
            runner.add_step(topup_native.step)
            se2sbref_mat = opts.work_dir / "pose" / f"se2{reg_ref_tag}_6dof.mat"
            runner.add_step(create_identity_transform_step(se2sbref_mat))
            sbref2se_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2se_6dof.mat"
            runner.add_step(create_identity_transform_step(sbref2se_mat))
            warp_sbref = sdc_dir / f"WarpField_{reg_ref_tag}Space.nii.gz"
            runner.add_step(
                create_copy_nifti_step(
                    src=topup_native.dfout,
                    dst=warp_sbref,
                    force=opts.overwrite,
                    step_name="Place SynBOLD-DisCo TOPUP Warp",
                )
            )
        else:
            require_existing_path(inputs.se1_json, "se1-json")
            require_existing_path(inputs.se2_json, "se2-json")
            se1_meta = (
                dict(inputs.se1_metadata)
                if inputs.se1_metadata is not None
                else read_json(inputs.se1_json)  # type: ignore[arg-type]
            )
            se2_meta = (
                dict(inputs.se2_metadata)
                if inputs.se2_metadata is not None
                else read_json(inputs.se2_json)  # type: ignore[arg-type]
            )
            se1_ped = str(se1_meta.get("PhaseEncodingDirection", "")).strip()
            se2_ped = str(se2_meta.get("PhaseEncodingDirection", "")).strip()
            if not se1_ped or not se2_ped:
                raise SystemExit("Missing PhaseEncodingDirection in SE JSON sidecars.")
            if reg_ref_ped not in (se1_ped, se2_ped):
                raise SystemExit(
                    f"Functional reference PhaseEncodingDirection {reg_ref_ped!r} does not match se1/se2 "
                    f"({se1_ped!r}, {se2_ped!r})."
                )
            assert inputs.se1 is not None and inputs.se2 is not None
            try:
                float(bids_readout_time(se1_meta))
                float(bids_readout_time(se2_meta))
            except Exception as error:
                raise SystemExit(
                    "Both SE fieldmaps require TotalReadoutTime, or EffectiveEchoSpacing "
                    "with a usable phase-encoding matrix size."
                ) from error
            try:
                bold_readout = float(bids_readout_time(epi_input_meta))
            except (KeyError, TypeError, ValueError):
                raise SystemExit(
                    "Fieldmap SDC requires TotalReadoutTime, or EffectiveEchoSpacing with "
                    "a usable phase-encoding matrix size, for the BOLD acquisition."
                )
            registration_readout = bold_readout
            if registration_readout is None:
                raise SystemExit(
                    "Fieldmap SDC requires TotalReadoutTime, or EffectiveEchoSpacing with "
                    "a usable phase-encoding matrix size, for the selected functional reference."
                )

            (se_a, meta_a), (se_b, meta_b) = _canonical_fieldmap_order(
                (inputs.se1, se1_meta),
                (inputs.se2, se2_meta),
            )
            ped_a = str(meta_a.get("PhaseEncodingDirection", "")).strip()
            ped_b = str(meta_b.get("PhaseEncodingDirection", "")).strip()
            readout_a = float(bids_readout_time(meta_a))
            readout_b = float(bids_readout_time(meta_b))
            topup_native = _create_topup_dfout_step(
                run_child=runner.run_child,
                se_a=se_a,
                se_b=se_b,
                ped_a=ped_a,
                ped_b=ped_b,
                readout_time=readout_a,
                readout_time_b=readout_b,
                topup_dir=topup_work_dir,
                topup_config=opts.topup_config,
                env=env,
                force=opts.overwrite,
                spatial_shape=nifti_spatial_shape(se_a),
                volumes_a=nifti_volume_count(se_a),
                volumes_b=nifti_volume_count(se_b),
            )
            runner.add_step(topup_native.step)
            if reg_ref_ped == ped_a:
                matching_se = se_a
                matching_group = "A"
                matching_index_1based = 1
            elif reg_ref_ped == ped_b:
                matching_se = se_b
                matching_group = "B"
                matching_index_1based = int(topup_native.a_nvols) + 1
            else:
                raise SystemExit(
                    f"Functional reference PE direction {reg_ref_ped!r} does not match "
                    f"the canonical fieldmap pair ({ped_a!r}, {ped_b!r})."
                )
            se_match_ref = opts.work_dir / "pose" / "se_match_ref.nii.gz"
            runner.add_step(
                create_nifti_volume_extraction_step(
                    img=matching_se,
                    index_zero_based=0,
                    out_3d=se_match_ref,
                    env=env,
                    force=opts.overwrite,
                    label="Extract Matching Distorted SE Reference",
                )
            )
            se2sbref_mat = opts.work_dir / "pose" / f"se2{reg_ref_tag}_6dof.mat"
            sbref2se_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2se_6dof.mat"
            pose_work = se2sbref_mat.parent / f"{se2sbref_mat.stem}_qc"
            pose_mask = pose_work / "fixed_support_mask.nii.gz"
            runner.add_step(
                create_image_support_mask_step(
                    image=reg_ref_dist_ref,
                    out_mask=pose_mask,
                    erosion_voxels=1,
                    minimum_voxels=100,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_flirt_registration_step(
                    run_child=runner.run_child,
                    in_img=se_match_ref,
                    ref_img=reg_ref_dist_ref,
                    out_mat=se2sbref_mat,
                    work_dir=pose_work,
                    fixed_mask=pose_mask,
                    env=env,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_invert_mat_step(
                    mat=se2sbref_mat,
                    out_mat=sbref2se_mat,
                    env=env,
                    force=opts.overwrite,
                )
            )

            matching_motion_mat = _normalized_topup_matrix(
                topup_native.rbmout,
                index_1based=matching_index_1based,
            )
            reg_ref_to_topup_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2topup_6dof.mat"
            runner.add_step(
                _create_concat_mats_step(
                    first=matching_motion_mat,
                    second=sbref2se_mat,
                    out_mat=reg_ref_to_topup_mat,
                    env=env,
                    force=opts.overwrite,
                )
            )
            topup_to_reg_ref_mat = opts.work_dir / "pose" / f"topup2{reg_ref_tag}_6dof.mat"
            runner.add_step(
                _create_invert_mat_step(
                    mat=reg_ref_to_topup_mat,
                    out_mat=topup_to_reg_ref_mat,
                    env=env,
                    force=opts.overwrite,
                )
            )

            corrected_se_median = topup_work_dir / "se_unwarped_median.nii.gz"
            corrected_se_cmd = [
                "fslmaths",
                str(topup_native.iout),
                "-Tmedian",
                str(corrected_se_median),
            ]
            runner.add_step(
                Step.command_step(
                    corrected_se_cmd,
                    env=env,
                    outputs=(corrected_se_median,),
                    inputs=(topup_native.iout,),
                    force=opts.overwrite,
                    name="Build Corrected SE Reference",
                )
            )

            registration_shift_vox = (
                topup_work_dir / f"{reg_ref_tag}_registrationReadout_shift_vox.nii.gz"
            )
            registration_warp_topup = (
                topup_work_dir / f"WarpField_{reg_ref_tag}_registrationReadout.nii.gz"
            )
            runner.add_step(
                _create_target_shift_step(
                    field_hz=topup_native.field_hz,
                    reference=corrected_se_median,
                    readout_time=float(registration_readout),
                    shift_vox=registration_shift_vox,
                    env=env,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_target_readout_warp_step(
                    field_hz=topup_native.field_hz,
                    reference=corrected_se_median,
                    phase_encoding_direction=reg_ref_ped,
                    shift_vox=registration_shift_vox,
                    out_warp=registration_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )

            initial_dc_topup = topup_work_dir / f"{reg_ref_tag}_dc_initial_topupSpace.nii.gz"
            runner.add_step(
                _create_applywarp_step(
                    in_img=reg_ref_dist_ref,
                    ref_img=corrected_se_median,
                    warp=registration_warp_topup,
                    premat=reg_ref_to_topup_mat,
                    out_img=initial_dc_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            postdc_refine_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2topup_postdc_6dof.mat"
            postdc_registered = topup_work_dir / f"{reg_ref_tag}_dc_postRigid_topupSpace.nii.gz"
            postdc_refine_qc_path = (
                opts.work_dir / "pose" / f"{reg_ref_tag}2topup_postdc_6dof_qc.json"
            )
            runner.add_step(
                _create_local_rigid_refinement_step(
                    run_child=runner.run_child,
                    moving=initial_dc_topup,
                    fixed=corrected_se_median,
                    out_mat=postdc_refine_mat,
                    out_registered=postdc_registered,
                    qc_json=postdc_refine_qc_path,
                    env=env,
                    force=opts.overwrite,
                )
            )
            postdc_refine_qc = postdc_refine_qc_path
            topup_to_reg_ref_refined_mat = (
                opts.work_dir / "pose" / f"topup2{reg_ref_tag}_postdcRefined_6dof.mat"
            )
            runner.add_step(
                _create_concat_mats_step(
                    first=topup_to_reg_ref_mat,
                    second=postdc_refine_mat,
                    out_mat=topup_to_reg_ref_refined_mat,
                    env=env,
                    force=opts.overwrite,
                )
            )

            pe_work = topup_work_dir / "pe_residual_to_corrected_se"
            pe_overlap = pe_work / "overlap_mask.nii.gz"
            runner.add_step(
                create_native_overlap_mask_step(
                    images=[postdc_registered, corrected_se_median],
                    out_mask=pe_overlap,
                    erosion_voxels=opts.synbold_overlap_erosion_voxels,
                    minimum_voxels=opts.synbold_min_overlap_voxels,
                    force=opts.overwrite,
                )
            )
            pe_frame = _ants_pe_aligned_frame(inputs.epi, reg_ref_ped)
            pe_aligned_dir = pe_work / "pe_aligned"
            pe_moving = pe_aligned_dir / "moving_reference.nii.gz"
            pe_fixed = pe_aligned_dir / "fixed_corrected_se.nii.gz"
            pe_mask = pe_aligned_dir / "overlap_mask.nii.gz"
            runner.add_step(
                _create_nifti_in_ants_frame_step(
                    source=postdc_registered,
                    out_image=pe_moving,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_nifti_in_ants_frame_step(
                    source=corrected_se_median,
                    out_image=pe_fixed,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_nifti_in_ants_frame_step(
                    source=pe_overlap,
                    out_image=pe_mask,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            pe_registration = _create_ants_registration_step(
                run_child=runner.run_child,
                moving_img=pe_moving,
                fixed_img=pe_fixed,
                work_dir=pe_aligned_dir / "ants",
                out_prefix="PEResidual_",
                env=env,
                force=opts.overwrite,
                include_linear=False,
                write_composite=False,
                fixed_mask=pe_mask,
                moving_mask=pe_mask,
                syn_transform=opts.syn_refine_transform,
                syn_convergence=opts.syn_refine_convergence,
                syn_shrink_factors=opts.syn_refine_shrink_factors,
                syn_smoothing_sigmas=opts.syn_refine_smoothing_sigmas,
                restrict_deformation=pe_frame.restriction,
            )
            runner.add_step(pe_registration.step)
            pe_residual_itk = pe_work / "residual_itk_warp.nii.gz"
            runner.add_step(
                _create_restore_ants_warp_step(
                    aligned_warp=pe_registration.forward_transform,
                    original_reference=postdc_registered,
                    out_warp=pe_residual_itk,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            pe_residual_fnirt = pe_work / "residual_fnirt_warp.nii.gz"
            runner.add_step(
                _create_wb_convert_itk_warp_to_fnirt_step(
                    itk_warp=pe_residual_itk,
                    src_space_ref=postdc_registered,
                    out_warp=pe_residual_fnirt,
                    env=env,
                    force=opts.overwrite,
                )
            )
            pe_residual_details = {
                "Method": "PEConstrainedSyNToCorrectedSE",
                "MovingReference": str(postdc_registered),
                "FixedReference": str(corrected_se_median),
                "PhaseEncodingDirection": reg_ref_ped,
                "ANTsRestriction": pe_frame.restriction,
                "HeaderOnlyPEFrame": True,
                "Transform": opts.syn_refine_transform,
                "Convergence": opts.syn_refine_convergence,
                "ShrinkFactors": opts.syn_refine_shrink_factors,
                "SmoothingSigmas": opts.syn_refine_smoothing_sigmas,
                "OverlapErosionVoxels": int(opts.synbold_overlap_erosion_voxels),
                "MinimumOverlapVoxels": int(opts.synbold_min_overlap_voxels),
                "ITKWarp": str(pe_residual_itk),
                "FNIRTWarp": str(pe_residual_fnirt),
                "UsesOtherFunctionalRuns": False,
            }

            registration_rigid_warp_topup = (
                topup_work_dir / f"WarpField_{reg_ref_tag}_registrationReadout_postRigid.nii.gz"
            )
            runner.add_step(
                _create_convertwarp_postmat_step(
                    ref=corrected_se_median,
                    warp1=registration_warp_topup,
                    postmat=postdc_refine_mat,
                    out_warp=registration_rigid_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            registration_complete_warp_topup = (
                topup_work_dir / f"WarpField_{reg_ref_tag}_registrationReadout_complete.nii.gz"
            )
            runner.add_step(
                _create_convertwarp_merge_warps_step(
                    ref=corrected_se_median,
                    warp1=registration_rigid_warp_topup,
                    warp2=pe_residual_fnirt,
                    out_warp=registration_complete_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            warp_sbref = sdc_dir / f"WarpField_{reg_ref_tag}Space.nii.gz"
            runner.add_step(
                _create_convertwarp_conjugate_affine_step(
                    ref=reg_ref_dist_ref,
                    warp1=registration_complete_warp_topup,
                    premat=reg_ref_to_topup_mat,
                    postmat=topup_to_reg_ref_mat,
                    out_warp=warp_sbref,
                    env=env,
                    force=opts.overwrite,
                )
            )

            bold_ref_to_topup_step = _create_bold_ref_to_topup_transform_step(
                reg_ref_to_topup_mat=reg_ref_to_topup_mat,
                epi_ref_to_reg_ref_mat=epi_to_reference,
                work_dir=opts.work_dir,
                env=env,
                force=opts.overwrite,
            )
            runner.add_step(bold_ref_to_topup_step)
            bold_ref_to_topup_mat = bold_ref_to_topup_step.outputs[0]
            bold_shift_vox = topup_work_dir / "bold_target_shift_vox.nii.gz"
            bold_warp_topup = topup_work_dir / "WarpField_bold_targetReadout.nii.gz"
            runner.add_step(
                _create_target_shift_step(
                    field_hz=topup_native.field_hz,
                    reference=corrected_se_median,
                    readout_time=float(bold_readout),
                    shift_vox=bold_shift_vox,
                    env=env,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_target_readout_warp_step(
                    field_hz=topup_native.field_hz,
                    reference=corrected_se_median,
                    phase_encoding_direction=epi_ped,
                    shift_vox=bold_shift_vox,
                    out_warp=bold_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            bold_rigid_warp_topup = topup_work_dir / "WarpField_bold_postRigid.nii.gz"
            runner.add_step(
                _create_convertwarp_postmat_step(
                    ref=corrected_se_median,
                    warp1=bold_warp_topup,
                    postmat=postdc_refine_mat,
                    out_warp=bold_rigid_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            bold_complete_warp_topup = topup_work_dir / "WarpField_bold_complete.nii.gz"
            runner.add_step(
                _create_convertwarp_merge_warps_step(
                    ref=corrected_se_median,
                    warp1=bold_rigid_warp_topup,
                    warp2=pe_residual_fnirt,
                    out_warp=bold_complete_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            warp_bold_to_reg_ref = sdc_dir / "WarpField_boldToRegRef.nii.gz"
            runner.add_step(
                _create_convertwarp_conjugate_affine_step(
                    ref=reg_ref_dist_ref,
                    warp1=bold_complete_warp_topup,
                    premat=bold_ref_to_topup_mat,
                    postmat=topup_to_reg_ref_mat,
                    out_warp=warp_bold_to_reg_ref,
                    env=env,
                    force=opts.overwrite,
                )
            )
            fieldmap_transfer_details = {
                "PolicyVersion": _FIELDMAP_TRANSFER_POLICY_VERSION,
                "CanonicalPhaseEncodingDirections": [ped_a, ped_b],
                "CanonicalInputs": [str(se_a), str(se_b)],
                "CanonicalReadoutTimes": [readout_a, readout_b],
                "MatchingPhaseEncodingDirection": reg_ref_ped,
                "MatchingInputGroup": matching_group,
                "MatchingTopupVolumeIndex": int(matching_index_1based),
                "MatchingTopupMotionMatrix": str(matching_motion_mat),
                "RegistrationReferenceReadoutTime": float(registration_readout),
                "BOLDReadoutTime": float(bold_readout),
                "SeparateBOLDReadoutWarp": True,
                "RegistrationShiftDirection": _pe_to_fsl_shift_direction(reg_ref_ped),
                "BOLDShiftDirection": _pe_to_fsl_shift_direction(epi_ped),
                "RegistrationWarpInTopupSpace": str(registration_warp_topup),
                "BOLDWarpInTopupSpace": str(bold_warp_topup),
                "CorrectedSEReference": str(corrected_se_median),
                "InitialReferenceToTopupTransform": str(reg_ref_to_topup_mat),
                "InitialBOLDReferenceToTopupTransform": str(bold_ref_to_topup_mat),
                "PostSDCRigidTransform": str(postdc_refine_mat),
                "PostSDCRigidQCFile": str(postdc_refine_qc_path),
                "PostSDCRigidQC": str(postdc_refine_qc),
                "PEResidualReference": "SelectedFunctionalReference",
                "PEResidualRefinement": pe_residual_details,
                "BOLDToRegistrationReferenceWarp": str(warp_bold_to_reg_ref),
                "FinalWarpIncludesPostSDCRigidRefinement": True,
                "FinalWarpIncludesPEResidualRefinement": True,
                "PostFieldmapRefinementTarget": fieldmap_refinement_target,
                "PostFieldmapRefinementConstraint": ("Unrestricted" if do_refinement else None),
                "FinalInterpolationCount": 1,
            }

        reg_ref_dc = sdc_dir / f"{reg_ref_tag}_dc.nii.gz"
        runner.add_step(
            _create_applywarp_step(
                in_img=reg_ref_dist_ref,
                ref_img=reg_ref_dist_ref,
                warp=warp_sbref,
                out_img=reg_ref_dc,
                env=env,
                force=opts.overwrite,
            )
        )
        if opts.use_jacobian:
            jac_sbref = sdc_dir / f"Jacobian_{reg_ref_tag}Space.nii.gz"
            jacobian_temporary = jac_sbref.parent / (
                jac_sbref.name.replace(".nii.gz", "") + "_tmp8.nii.gz"
            )
            jacobian_junk = jac_sbref.parent / "junk_warp.nii.gz"
            runner.add_step(
                _create_warp_jacobian_step(
                    warp=warp_sbref,
                    ref=reg_ref_dist_ref,
                    temporary=jacobian_temporary,
                    junk=jacobian_junk,
                    env=env,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _create_average_jacobian_step(
                    temporary=jacobian_temporary,
                    junk=jacobian_junk,
                    warp=warp_sbref,
                    ref=reg_ref_dist_ref,
                    out_jac=jac_sbref,
                    env=env,
                    force=opts.overwrite,
                )
            )
            reg_ref_dc_jac = sdc_dir / f"{reg_ref_tag}_dc_jac.nii.gz"
            jac_cmd = ["fslmaths", str(reg_ref_dc), "-mul", str(jac_sbref), str(reg_ref_dc_jac)]
            runner.add_step(
                Step.command_step(
                    jac_cmd,
                    outputs=(reg_ref_dc_jac,),
                    inputs=(reg_ref_dc, jac_sbref),
                    force=opts.overwrite,
                    env=env,
                )
            )
            reg_ref_dc = reg_ref_dc_jac

        reg_ref_dc_ref = reg_ref_dc
        runner.add_step(
            create_n4_bias_correction_step(
                in_img=reg_ref_dc_ref,
                out_img=reg_ref_dc_ref_n4,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_bbregister_step(
                sbref=reg_ref_dc_ref_n4,
                fs_subject=fs_subject,
                reg_dat=reg_dat,
                fsl_mat=fsl_mat,
                surf=opts.bbregister_surf,
                init=opts.bbregister_init,
                dof=opts.bbregister_dof,
                env=env,
                force=opts.overwrite,
            )
        )
    # Output grid
    if opts.output_grid == "t1_epi_vox":
        t1_ref = reg_dir / f"{run_stem}_t1_grid_epi_vox.nii.gz"
        runner.add_step(
            _create_t1_epi_vox_target_step(
                t1_image=registration_t1,
                source_epi=inputs.epi,
                out_target=t1_ref,
                env=env,
                force=opts.overwrite,
            )
        )
    else:
        t1_ref = reg_dir / f"{run_stem}_t1_native.nii.gz"
        runner.add_step(
            _create_t1_native_target_step(
                t1_image=registration_t1,
                out_target=t1_ref,
                env=env,
                force=opts.overwrite,
            )
        )
    if not use_syn_fallback:
        fs_t1_image = subjects_dir / fs_subject / "mri" / "T1.mgz"
        if opts.output_grid == "t1_epi_vox":
            runner.add_step(
                _create_t1_epi_vox_target_step(
                    t1_image=fs_t1_image,
                    source_epi=inputs.epi,
                    out_target=bbr_t1_ref,
                    env=env,
                    force=opts.overwrite,
                )
            )
        else:
            runner.add_step(
                _create_t1_native_target_step(
                    t1_image=fs_t1_image,
                    out_target=bbr_t1_ref,
                    env=env,
                    force=opts.overwrite,
                )
            )
        runner.add_step(
            _create_tkregister2_regheader_fslmat_step(
                mov_img=bbr_t1_ref,
                targ_img=t1_ref,
                out_mat=fs_t1_to_t1w_mat,
                env=env,
                force=opts.overwrite,
            )
        )
        assert fsl_mat is not None
        runner.add_step(
            _create_concat_mats_step(
                first=fs_t1_to_t1w_mat,
                second=fsl_mat,
                out_mat=reg_ref_to_t1w_mat,
                env=env,
                force=opts.overwrite,
            )
        )
    syn_fixed_mask = reg_dir / f"{run_stem}_t1_brain_mask.nii.gz"
    runner.add_step(
        create_mask_resampling_step(
            src_mask=anat_brain_mask,
            ref_img=t1_ref,
            out_mask=syn_fixed_mask,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _create_mni_2mm_target_step(
            mni_template=anat_mni_template,
            out_target=mni_ref,
            env=env,
            force=opts.overwrite,
        )
    )
    syn_refine_fnirt_warp: Optional[Path] = None
    warp_regref2t1_refined: Optional[Path] = None
    warp_sbref2t1_refined: Optional[Path] = None
    base_static_warp: Optional[Path] = None
    base_ref_in_t1: Optional[Path] = None
    if use_syn_fallback:
        assert reg_ref_dist_ref is not None and epi_mc_ref is not None
        syn_moving = reg_ref_dist_ref
        syn_work = reg_dir / "ants_syn"
        fallback_rigid_step = _create_synbold_rigid_registration_step(
            run_child=runner.run_child,
            distorted_reference=syn_moving,
            anatomical_t1=t1_ref,
            anatomical_mask=syn_fixed_mask,
            work_dir=reg_dir / "whole_brain_rigid",
            env=env,
            max_translation_mm=opts.synbold_max_rigid_translation_mm,
            max_rotation_degrees=opts.synbold_max_rigid_rotation_degrees,
            force=opts.overwrite,
            rigid_mat_out=epi_mean_to_t1_fallback_mat,
            rigid_qc_out=epi_mean_nodc_t1_affine,
            registration_label="fieldmapless EPI-to-T1",
        )
        runner.add_step(fallback_rigid_step.step)
        pre_nonlinear_ref_in_t1 = epi_mean_nodc_t1_affine
        runner.add_step(
            create_n4_bias_correction_step(
                in_img=epi_mean_nodc_t1_affine,
                out_img=reg_ref_in_t1_base_n4,
                env=env,
                force=opts.overwrite,
                mask=syn_fixed_mask,
            )
        )
        fallback_registration = _create_ants_synboldaff_step(
            moving_img=reg_ref_in_t1_base_n4,
            fixed_img=t1_ref,
            work_dir=syn_work,
            out_prefix=f"{run_stem}_SyNBoldAff_",
            env=env,
            force=opts.overwrite,
            fixed_mask=syn_fixed_mask,
            moving_mask=syn_fixed_mask,
            syn_transform=opts.syn_base_transform,
            syn_convergence=opts.syn_base_convergence,
            syn_shrink_factors=opts.syn_base_shrink_factors,
            syn_smoothing_sigmas=opts.syn_base_smoothing_sigmas,
        )
        runner.add_step(fallback_registration.step)
        fallback_composite_xfm = fallback_registration.forward_transform
        ants_forward_xfm = fallback_composite_xfm
        fallback_base_itk_warp = fallback_base_itk_planned
        runner.add_step(
            _create_ants_composite_to_itk_warp_step(
                composite_xfm=fallback_composite_xfm,
                ref_img=t1_ref,
                out_warp=fallback_base_itk_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        fallback_base_fnirt_warp = fallback_base_fnirt_planned
        runner.add_step(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=fallback_base_itk_warp,
                src_space_ref=reg_ref_in_t1_base_n4,
                out_warp=fallback_base_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        syn_refine_fnirt_warp = fallback_base_fnirt_warp
        fallback_affine_warp = fallback_affine_warp_planned
        runner.add_step(
            _create_convertwarp_premat_step(
                ref=t1_ref,
                premat=epi_mean_to_t1_fallback_mat,
                out_warp=fallback_affine_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        fallback_base_warp = fallback_base_warp_planned
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=t1_ref,
                warp1=fallback_affine_warp,
                warp2=fallback_base_fnirt_warp,
                out_warp=fallback_base_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        warp_regref2t1_refined = fallback_base_warp
        runner.add_step(
            _create_convertwarp_premat_and_warp_step(
                ref=t1_ref,
                premat=epi_to_reference,
                warp1=fallback_base_warp,
                out_warp=epi_to_t1_warp_planned,
                env=env,
                force=opts.overwrite,
            )
        )
        base_static_warp = epi_to_t1_warp_planned
        runner.add_step(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=fallback_base_warp,
                out_img=reg_ref_in_t1_affine,
                env=env,
                force=opts.overwrite,
            )
        )
        base_ref_in_t1 = reg_ref_in_t1_affine
        boldref_t1 = boldref_t1_out
        runner.add_step(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=fallback_base_warp,
                out_img=boldref_t1,
                env=env,
                force=opts.overwrite,
            )
        )
    else:
        assert (
            reg_ref_dc_ref is not None
            and reg_ref_dist_ref is not None
            and reg_ref_to_t1w_mat is not None
            and epi_mc_ref is not None
        )
        warp_sbref2t1 = warp_regref2t1_planned
        assert warp_sbref is not None
        runner.add_step(
            create_flirt_transform_step(
                in_img=reg_ref_dist_ref,
                ref_img=t1_ref,
                mat=reg_ref_to_t1w_mat,
                out_img=reg_ref_in_t1_linear,
                env=env,
                force=opts.overwrite,
            )
        )
        pre_nonlinear_ref_in_t1 = reg_ref_in_t1_linear
        runner.add_step(
            _create_convertwarp_postmat_step(
                ref=t1_ref,
                warp1=warp_sbref,
                postmat=reg_ref_to_t1w_mat,
                out_warp=warp_sbref2t1,
                env=env,
                force=opts.overwrite,
            )
        )
        syn_moving = reg_ref_dist_ref
        runner.add_step(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=warp_sbref2t1,
                out_img=reg_ref_in_t1_affine,
                env=env,
                force=opts.overwrite,
            )
        )
        if use_fieldmap_sdc:
            assert warp_bold_to_reg_ref is not None
            runner.add_step(
                _create_convertwarp_postmat_step(
                    ref=t1_ref,
                    warp1=warp_bold_to_reg_ref,
                    postmat=reg_ref_to_t1w_mat,
                    out_warp=epi_to_t1_warp_planned,
                    env=env,
                    force=opts.overwrite,
                )
            )
            base_static_warp = epi_to_t1_warp_planned
        else:
            runner.add_step(
                _create_convertwarp_premat_and_warp_step(
                    ref=t1_ref,
                    premat=epi_to_reference,
                    warp1=warp_sbref2t1,
                    out_warp=epi_to_t1_warp_planned,
                    env=env,
                    force=opts.overwrite,
                )
            )
            base_static_warp = epi_to_t1_warp_planned
        base_ref_in_t1 = reg_ref_in_t1_affine
        warp_regref2t1_refined = warp_sbref2t1
        boldref_t1 = boldref_t1_out
        runner.add_step(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=warp_sbref2t1,
                out_img=boldref_t1,
                env=env,
                force=opts.overwrite,
            )
        )
        assert topup_native is not None and se2sbref_mat is not None
        field_to_reg_ref_mat = topup_to_reg_ref_refined_mat if use_fieldmap_sdc else se2sbref_mat
        assert field_to_reg_ref_mat is not None
        runner.add_step(
            create_flirt_transform_step(
                in_img=topup_native.field_hz,
                ref_img=reg_ref_dist_ref,
                mat=field_to_reg_ref_mat,
                out_img=field_hz_regref,
                env=env,
                force=opts.overwrite,
            )
        )

    assert (
        base_static_warp is not None
        and base_ref_in_t1 is not None
        and warp_regref2t1_refined is not None
    )
    if do_refinement and use_synbold_reference:
        assert synthetic_ref is not None
        assert reg_ref_dc_ref is not None
        assert reg_ref_dist_ref is not None
        assert warp_sbref is not None
        assert reg_ref_to_t1w_mat is not None
        native_refine_work = reg_dir / "synbold_native_refine"
        native_overlap_mask = (
            native_refine_work
            / f"overlap_erode-{int(opts.synbold_overlap_erosion_voxels)}_mask.nii.gz"
        )
        runner.add_step(
            create_native_overlap_mask_step(
                images=[reg_ref_dc_ref, synthetic_ref],
                out_mask=native_overlap_mask,
                erosion_voxels=opts.synbold_overlap_erosion_voxels,
                minimum_voxels=opts.synbold_min_overlap_voxels,
                force=opts.overwrite,
            )
        )
        pe_frame = _ants_pe_aligned_frame(inputs.epi, reg_ref_ped)
        aligned_work = native_refine_work / f"pe-{reg_ref_ped.rstrip('-')}_aligned"
        aligned_moving = aligned_work / "moving_epi_reference.nii.gz"
        runner.add_step(
            _create_nifti_in_ants_frame_step(
                source=reg_ref_dc_ref,
                out_image=aligned_moving,
                frame=pe_frame,
                force=opts.overwrite,
            )
        )
        aligned_fixed = aligned_work / "fixed_synthetic_reference.nii.gz"
        runner.add_step(
            _create_nifti_in_ants_frame_step(
                source=synthetic_ref,
                out_image=aligned_fixed,
                frame=pe_frame,
                force=opts.overwrite,
            )
        )
        aligned_mask = aligned_work / "overlap_mask.nii.gz"
        runner.add_step(
            _create_nifti_in_ants_frame_step(
                source=native_overlap_mask,
                out_image=aligned_mask,
                frame=pe_frame,
                force=opts.overwrite,
            )
        )
        LOG.info(
            "Restricting SynBOLD residual deformation to ANTs component %s after a "
            "header-only PE-frame rotation (voxel axis=%d, physical PE direction RAS=%s)",
            pe_frame.restriction,
            pe_frame.voxel_axis,
            ",".join(f"{value:.6f}" for value in pe_frame.pe_direction_ras),
        )
        aligned_refinement = _create_ants_registration_step(
            run_child=runner.run_child,
            moving_img=aligned_moving,
            fixed_img=aligned_fixed,
            work_dir=aligned_work / "ants",
            out_prefix=f"{run_stem}_SynBOLDResidual_",
            env=env,
            force=opts.overwrite,
            include_linear=False,
            write_composite=False,
            fixed_mask=aligned_mask,
            moving_mask=aligned_mask,
            syn_transform=opts.syn_refine_transform,
            syn_convergence=opts.syn_refine_convergence,
            syn_shrink_factors=opts.syn_refine_shrink_factors,
            syn_smoothing_sigmas=opts.syn_refine_smoothing_sigmas,
            restrict_deformation=pe_frame.restriction,
        )
        runner.add_step(aligned_refinement.step)
        aligned_refine_warp = aligned_refinement.forward_transform
        refine_warp = native_refine_work / "residual_itk_warp_original_frame.nii.gz"
        runner.add_step(
            _create_restore_ants_warp_step(
                aligned_warp=aligned_refine_warp,
                original_reference=reg_ref_dc_ref,
                out_warp=refine_warp,
                frame=pe_frame,
                force=opts.overwrite,
            )
        )
        ants_forward_xfm = refine_warp
        syn_refine_fnirt_warp = native_refine_work / "residual_fnirt_warp.nii.gz"
        runner.add_step(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=refine_warp,
                src_space_ref=reg_ref_dc_ref,
                out_warp=syn_refine_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        combined_native_warp = native_refine_work / "fieldmap_plus_residual_warp.nii.gz"
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=reg_ref_dist_ref,
                warp1=warp_sbref,
                warp2=syn_refine_fnirt_warp,
                out_warp=combined_native_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        warp_regref2t1_refined = warp_regref2t1_refined_planned
        runner.add_step(
            _create_convertwarp_postmat_step(
                ref=t1_ref,
                warp1=combined_native_warp,
                postmat=reg_ref_to_t1w_mat,
                out_warp=warp_regref2t1_refined,
                env=env,
                force=opts.overwrite,
            )
        )
        warp_sbref2t1_refined = warp_sbref2t1_refined_planned
        assert use_fieldmap_sdc and warp_bold_to_reg_ref is not None
        bold_combined_native_warp = native_refine_work / "bold_fieldmap_plus_residual_warp.nii.gz"
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=reg_ref_dist_ref,
                warp1=warp_bold_to_reg_ref,
                warp2=syn_refine_fnirt_warp,
                out_warp=bold_combined_native_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_convertwarp_postmat_step(
                ref=t1_ref,
                warp1=bold_combined_native_warp,
                postmat=reg_ref_to_t1w_mat,
                out_warp=warp_sbref2t1_refined,
                env=env,
                force=opts.overwrite,
            )
        )
    elif do_refinement:
        syn_work = reg_dir / "ants_syn_refine"
        runner.add_step(
            create_n4_bias_correction_step(
                in_img=base_ref_in_t1,
                out_img=reg_ref_in_t1_refine_n4,
                env=env,
                force=opts.overwrite,
                mask=syn_fixed_mask,
            )
        )
        refinement = _create_ants_registration_step(
            run_child=runner.run_child,
            moving_img=reg_ref_in_t1_refine_n4,
            fixed_img=t1_ref,
            work_dir=syn_work,
            out_prefix=f"{run_stem}_SyNRefine_",
            env=env,
            force=opts.overwrite,
            include_linear=False,
            write_composite=False,
            fixed_mask=syn_fixed_mask,
            moving_mask=syn_fixed_mask,
            syn_transform=opts.syn_refine_transform,
            syn_convergence=opts.syn_refine_convergence,
            syn_shrink_factors=opts.syn_refine_shrink_factors,
            syn_smoothing_sigmas=opts.syn_refine_smoothing_sigmas,
        )
        runner.add_step(refinement.step)
        refine_warp = refinement.forward_transform
        ants_forward_xfm = refine_warp if ants_forward_xfm is None else ants_forward_xfm
        syn_refine_fnirt_warp = syn_refine_fnirt_planned
        runner.add_step(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=refine_warp,
                src_space_ref=reg_ref_in_t1_refine_n4,
                out_warp=syn_refine_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        assert warp_regref2t1_refined is not None
        base_regref_warp = warp_regref2t1_refined
        warp_regref2t1_refined = warp_regref2t1_refined_planned
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=t1_ref,
                warp1=base_regref_warp,
                warp2=syn_refine_fnirt_warp,
                out_warp=warp_regref2t1_refined,
                env=env,
                force=opts.overwrite,
            )
        )
        warp_sbref2t1_refined = warp_sbref2t1_refined_planned
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=t1_ref,
                warp1=base_static_warp,
                warp2=syn_refine_fnirt_warp,
                out_warp=warp_sbref2t1_refined,
                env=env,
                force=opts.overwrite,
            )
        )
    else:
        # Without refinement, the selected base warp is the final warp.
        warp_sbref2t1_refined = base_static_warp

    assert (
        pre_nonlinear_ref_in_t1 is not None
        and warp_sbref2t1_refined is not None
        and warp_regref2t1_refined is not None
    )
    if (not use_syn_fallback) and topup_native is not None:
        # The Hz field lives in registration-reference space, so its ACPC
        # derivative must follow the final registration-reference-to-ACPC warp.
        # In particular, defer this until after anatomical SyN refinement; the
        # base warp above is not the final spatial mapping in that pathway.
        fieldmap_hz_in_t1 = fieldmap_hz_in_t1_out
        runner.add_step(
            _create_applywarp_step(
                in_img=field_hz_regref,
                ref_img=t1_ref,
                warp=warp_regref2t1_refined,
                out_img=fieldmap_hz_in_t1,
                env=env,
                force=opts.overwrite,
            )
        )
    runner.add_step(
        create_copy_nifti_step(
            src=pre_nonlinear_ref_in_t1,
            dst=reg_prenonlinear_qc_out,
            force=opts.overwrite,
            step_name="Finalize Registration QC",
        )
    )
    runner.add_step(
        create_copy_nifti_step(
            src=base_ref_in_t1,
            dst=reg_base_qc_out,
            force=opts.overwrite,
            step_name="Finalize Registration QC",
        )
    )
    if do_refinement:
        runner.add_step(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=warp_regref2t1_refined,
                out_img=reg_refine_qc_out,
                env=env,
                force=opts.overwrite,
            )
        )
    else:
        runner.add_step(
            create_copy_nifti_step(
                src=reg_base_qc_out,
                dst=reg_refine_qc_out,
                force=opts.overwrite,
                step_name="Finalize Registration QC",
            )
        )
    if want_mni:
        runner.add_step(
            _create_ants_composite_to_itk_warp_step(
                composite_xfm=t1_to_mni_xfm,
                ref_img=mni_ref,
                out_warp=t1_to_mni_itk_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=t1_to_mni_itk_warp,
                src_space_ref=anat_t1,
                out_warp=t1_to_mni_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=mni_ref,
                warp1=warp_sbref2t1_refined,
                warp2=t1_to_mni_fnirt_warp,
                out_warp=warp_epi2mni,
                env=env,
                force=opts.overwrite,
            )
        )

    final_sources_4d: dict[str, Path] = {}
    final_sources_mean: dict[str, Path] = {}
    final_masks: dict[str, Path] = {}
    afni_motion_affines = mc_dir / f"{run_stem}_mc_pull.aff12.1D"

    space_sequence: list[str] = []
    if compute_t1:
        space_sequence.append("ACPC")
    if want_mni:
        space_sequence.append("MNI152NLin2009cAsym")
    log_space_sequence = []
    if want_t1:
        log_space_sequence.append("ACPC")
    elif compute_t1:
        log_space_sequence.append("ACPC(required)")
    if want_fsnative:
        log_space_sequence.append("fsnative")
    if want_fsaverage:
        log_space_sequence.append(fsaverage_space)
    if want_mni:
        log_space_sequence.append("MNI152NLin2009cAsym")
    LOG.info("Output spaces execution order: %s", ", ".join(log_space_sequence))

    if want_mni:
        runner.add_step(
            _create_convertwarp_merge_warps_step(
                ref=mni_ref,
                warp1=warp_regref2t1_refined,
                warp2=t1_to_mni_fnirt_warp,
                out_warp=warp_regref2mni,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=mni_ref,
                warp=warp_regref2mni,
                out_img=reg_mni_qc_out,
                env=env,
                force=opts.overwrite,
            )
        )

    runner.add_step(
        _create_afni_motion_affines_step(
            in_4d=epi_for_proc,
            motion_ref_3d=robust_ref,
            mc_mat_dir=mc_mat_dir,
            out_affines=afni_motion_affines,
            force=opts.overwrite,
        )
    )

    for space in space_sequence:
        if space == "ACPC":
            raw_4d, mean_3d, mask_3d = epi_t1, epi_mean_t1, anat_brain_mask_in_t1
            ref_img, warp_img = t1_ref, warp_sbref2t1_refined
            aroma_out_4d, aroma_out_mean, aroma_work = aroma_clean, aroma_clean_mean, aroma_t1_dir
            final_dst = preproc_t1
            resampling_work = opts.work_dir / "resampling_t1w"
        else:
            raw_4d, mean_3d, mask_3d = epi_mni, epi_mean_mni, anat_brain_mask_in_mni
            ref_img, warp_img = mni_ref, warp_epi2mni
            aroma_out_4d, aroma_out_mean, aroma_work = (
                aroma_clean_mni,
                aroma_clean_mean_mni,
                aroma_mni_dir,
            )
            final_dst = preproc_mni
            resampling_work = opts.work_dir / "resampling_mni"

        space_name = (
            f"Output Space: {space}"
            if (space != "ACPC" or want_t1)
            else "Preparing Required ACPC Source"
        )
        LOG.info("Constructing %s", space_name)
        world_warp = resampling_work / "warp_world.nii.gz"
        afni_warp = resampling_work / "warp_afni_lps.nii.gz"
        runner.add_step(
            _create_world_warp_step(
                run_child=runner.run_child,
                motion_ref_3d=robust_ref,
                ref_3d=ref_img,
                fnirt_warp=warp_img,
                world_warp=world_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_afni_warp_step(
                world_warp=world_warp,
                motion_ref_3d=robust_ref,
                ref_3d=ref_img,
                afni_warp=afni_warp,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_afni_bold_resampling_step(
                run_child=runner.run_child,
                in_4d=final_resampling_source,
                motion_ref_3d=robust_ref,
                ref_3d=ref_img,
                afni_warp=afni_warp,
                motion_affines=afni_motion_affines,
                gradient_warp=gradient_afni_warp,
                out_4d=raw_4d,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _create_temporal_mean_step(
                in_4d=raw_4d,
                out_3d=mean_3d,
                env=env,
                force=opts.overwrite,
                chunk_vols=opts.io_chunk_vols,
            )
        )
        runner.add_step(
            create_mask_resampling_step(
                src_mask=anat_brain_mask,
                ref_img=mean_3d,
                out_mask=mask_3d,
                force=opts.overwrite,
            )
        )

        final_4d = raw_4d
        final_mean = mean_3d
        if ica_enabled:
            support_brain = aroma_work / "epi_support_bet.nii.gz"
            support_mask = aroma_work / "epi_support_bet_mask.nii.gz"
            regression_mask = aroma_work / "regression_mask.nii.gz"
            policy_path = aroma_work / f"{classifier}_policy.json"
            runner.add_step(
                _create_epi_support_step(
                    epi_mean=mean_3d,
                    support_brain=support_brain,
                    support_mask=support_mask,
                    work_dir=aroma_work,
                    env=env,
                    force=opts.overwrite,
                )
            )
            if space == "ACPC":
                melodic_mask = aroma_work / "melodic_mask.nii.gz"
                melodic_input = aroma_work / "melodic_input_smooth6mm.nii.gz"
                runner.add_step(
                    _create_dilated_anatomical_mask_step(
                        anatomical_mask=mask_3d,
                        support_mask=support_mask,
                        output=melodic_mask,
                        dilation_mm=_ICA_AROMA_MELODIC_MASK_DILATION_MM,
                        role="MELODIC Estimation",
                        force=opts.overwrite,
                    )
                )
                runner.add_step(
                    _create_dilated_anatomical_mask_step(
                        anatomical_mask=mask_3d,
                        support_mask=support_mask,
                        output=regression_mask,
                        dilation_mm=_ICA_AROMA_REGRESSION_MASK_DILATION_MM,
                        role="ICA Regression",
                        force=opts.overwrite,
                    )
                )
                runner.add_step(
                    _create_melodic_smoothing_step(
                        epi=raw_4d,
                        output=melodic_input,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                aroma_anat = aroma_work / "anat"
                t1_to_mni_itk = aroma_anat / "t1_to_mni_itk_warp.nii.gz"
                t1_to_mni_warp = aroma_anat / "t1_to_mni_warp.nii.gz"
                runner.add_step(
                    _create_ants_composite_to_itk_warp_step(
                        composite_xfm=t1_to_mni_xfm,
                        ref_img=anat_mni_template,
                        out_warp=t1_to_mni_itk,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                runner.add_step(
                    _create_wb_convert_itk_warp_to_fnirt_step(
                        itk_warp=t1_to_mni_itk,
                        src_space_ref=anat_t1,
                        out_warp=t1_to_mni_warp,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                identity_transform = aroma_work / "epi_in_t1_identity.mat"
                runner.add_step(create_identity_transform_step(identity_transform))
                aroma_dir = aroma_work / "aroma"
                melodic_dir = aroma_dir / "melodic.ica"
                melodic_products = (
                    melodic_dir / "melodic_IC.nii.gz",
                    melodic_dir / "melodic_mix",
                    melodic_dir / "melodic_FTmix",
                    aroma_dir / "melodic.complete",
                )
                if classifier == "ica_aroma":
                    aroma_outputs: list[Path] = [aroma_dir / "classification_overview.txt"]
                    aroma_outputs.append(
                        aroma_dir
                        / (
                            "denoised_func_data_aggr.nii.gz"
                            if denoise_type == "aggr"
                            else "denoised_func_data_nonaggr.nii.gz"
                        )
                    )
                    runner.add_step(
                        _create_ica_aroma_workflow_step(
                            runner=runner,
                            epi=raw_4d,
                            melodic_input=melodic_input,
                            motion_parameters=mc_dir / "motion.par",
                            melodic_mask=melodic_mask,
                            regression_mask=regression_mask,
                            aroma_dir=aroma_dir,
                            outputs=tuple((*aroma_outputs, *melodic_products)),
                            melodic_products=melodic_products,
                            input_is_mni=False,
                            identity_transform=identity_transform,
                            t1_to_mni_warp=t1_to_mni_warp,
                            mni_reference=mni_ref,
                            configured_command=opts.ica_aroma_cmd,
                            repetition_time=repetition_time,
                            denoise_type=denoise_type,
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                else:
                    if repetition_time is None or repetition_time <= 0:
                        raise SystemExit("CICADA requires a positive RepetitionTime.")
                    runner.add_step(
                        _create_cicada_melodic_step(
                            runner=runner,
                            melodic_input=melodic_input,
                            melodic_mask=melodic_mask,
                            output_directory=aroma_dir,
                            outputs=(
                                *melodic_products,
                                aroma_dir / "melodic_IC_thr_MNI2mm.nii.gz",
                            ),
                            identity_transform=identity_transform,
                            t1_to_mni_warp=t1_to_mni_warp,
                            mni_reference=mni_ref,
                            repetition_time=float(repetition_time),
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                classified_components = aroma_dir / "classified_motion_ICs.txt"
                classifier_policy = policy_path
                if classifier == "cicada":
                    if opts.container is None:
                        raise SystemExit("CICADA requires the configured FSL container.")
                    cicada_root = aroma_work / "cicada"
                    cicada_metrics = cicada_root / "motion_metrics.tsv"
                    cicada_task = cicada_root / "classification"
                    cicada_adapter = cicada_root / "melodic_mni"
                    cicada_manifest = cicada_root / "classification.json"
                    runner.add_step(
                        _create_cicada_motion_metrics_step(
                            epi=epi_mni,
                            mask=anat_brain_mask_in_mni,
                            motion_parameters=mc_dir / "motion.par",
                            output=cicada_metrics,
                            fd_radius_mm=float(SETTINGS.func_confounds.fd_radius_mm),
                            dvars_statistical_alpha=float(
                                SETTINGS.func_confounds.dvars_statistical_alpha
                            ),
                            dvars_practical_threshold_percent=float(
                                SETTINGS.func_confounds.dvars_practical_threshold_percent
                            ),
                            dvars_power=float(SETTINGS.func_confounds.dvars_power),
                            force=opts.overwrite,
                        )
                    )
                    runner.add_step(
                        _create_cicada_classification_step(
                            runner=runner,
                            executable=opts.cicada_cmd,
                            epi_mni=epi_mni,
                            mask_mni=anat_brain_mask_in_mni,
                            confounds=cicada_metrics,
                            source_melodic=melodic_dir,
                            melodic_complete=aroma_dir / "melodic.complete",
                            thresholded_components_mni=aroma_dir / "melodic_IC_thr_MNI2mm.nii.gz",
                            t1_to_mni_warp=t1_to_mni_warp,
                            identity_transform=identity_transform,
                            task_directory=cicada_task,
                            adapter_directory=cicada_adapter,
                            result_manifest=cicada_manifest,
                            tolerance=opts.cicada_tolerance,
                            smoothing_retention_mode=opts.cicada_smoothing_retention_mode,
                            repetition_time=repetition_time,
                            fsl_image=opts.container.image,
                            fsl_runtime=opts.container.engine,
                            fsl_binds=opts.container.extra_binds,
                            fsl_setup=opts.container.inner_setup,
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                    classified_components = cicada_task / "cicada_python/noise_components.txt"
                    classifier_policy = cicada_manifest
                    regression_dir = aroma_work / "regression"
                    denoised_outputs = (
                        regression_dir
                        / (
                            "denoised_func_data_aggr.nii.gz"
                            if denoise_type == "aggr"
                            else "denoised_func_data_nonaggr.nii.gz"
                        ),
                    )
                    runner.add_step(
                        _create_shared_aroma_regression_step(
                            runner=runner,
                            epi=raw_4d,
                            input_space="ACPC",
                            regression_mask=regression_mask,
                            mixing_matrix=melodic_dir / "melodic_mix",
                            classified_components=classified_components,
                            shared_policy=classifier_policy,
                            aroma_dir=regression_dir,
                            outputs=denoised_outputs,
                            denoise_type=denoise_type,
                            classifier_name="CICADA",
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                cleaned_epi = (
                    aroma_work / "regression" if classifier == "cicada" else aroma_dir
                ) / (
                    "denoised_func_data_aggr.nii.gz"
                    if denoise_type == "aggr"
                    else "denoised_func_data_nonaggr.nii.gz"
                )
                runner.add_step(
                    create_copy_nifti_step(
                        src=cleaned_epi,
                        dst=aroma_out_4d,
                        force=opts.overwrite,
                        step_name=f"Install {classifier} Denoised BOLD",
                    )
                )
                runner.add_step(
                    _create_temporal_mean_step(
                        in_4d=aroma_out_4d,
                        out_3d=aroma_out_mean,
                        env=env,
                        force=opts.overwrite,
                        chunk_vols=opts.io_chunk_vols,
                    )
                )
                if classifier == "ica_aroma":
                    runner.add_step(
                        create_json_step(
                            step_name="Write ICA-AROMA Estimation Policy",
                            path=policy_path,
                            payload=_ica_aroma_policy_payload(
                                input_is_mni=False,
                                denoise_type=denoise_type,
                                repetition_time=repetition_time,
                                external_aroma=opts.ica_aroma_cmd is not None,
                            ),
                            inputs=(aroma_out_4d, aroma_out_mean),
                            force=opts.overwrite,
                        )
                    )
                final_4d = aroma_out_4d
                final_mean = aroma_out_mean
            else:
                shared_aroma_dir = aroma_t1_dir / "aroma"
                shared_mixing = shared_aroma_dir / "melodic.ica" / "melodic_mix"
                shared_classified = (
                    shared_aroma_dir / "classified_motion_ICs.txt"
                    if classifier == "ica_aroma"
                    else aroma_t1_dir / "cicada/classification/cicada_python/noise_components.txt"
                )
                shared_policy = (
                    aroma_t1_dir / "ica_aroma_policy.json"
                    if classifier == "ica_aroma"
                    else aroma_t1_dir / "cicada/classification.json"
                )
                runner.add_step(
                    _create_dilated_anatomical_mask_step(
                        anatomical_mask=mask_3d,
                        support_mask=support_mask,
                        output=regression_mask,
                        dilation_mm=_ICA_AROMA_REGRESSION_MASK_DILATION_MM,
                        role=f"{space} ICA Regression",
                        force=opts.overwrite,
                    )
                )
                aroma_dir = aroma_work / "aroma"
                denoised_outputs = [
                    aroma_dir
                    / (
                        "denoised_func_data_aggr.nii.gz"
                        if denoise_type == "aggr"
                        else "denoised_func_data_nonaggr.nii.gz"
                    )
                ]
                runner.add_step(
                    _create_shared_aroma_regression_step(
                        runner=runner,
                        epi=raw_4d,
                        input_space=space,
                        regression_mask=regression_mask,
                        mixing_matrix=shared_mixing,
                        classified_components=shared_classified,
                        shared_policy=shared_policy,
                        aroma_dir=aroma_dir,
                        outputs=tuple(denoised_outputs),
                        denoise_type=denoise_type,
                        classifier_name="ICA-AROMA" if classifier == "ica_aroma" else "CICADA",
                        env=env,
                        force=opts.overwrite,
                    )
                )
                cleaned_epi = aroma_dir / (
                    "denoised_func_data_aggr.nii.gz"
                    if denoise_type == "aggr"
                    else "denoised_func_data_nonaggr.nii.gz"
                )
                runner.add_step(
                    create_copy_nifti_step(
                        src=cleaned_epi,
                        dst=aroma_out_4d,
                        force=opts.overwrite,
                        step_name=f"Install {space} {classifier} Denoised BOLD",
                    )
                )
                runner.add_step(
                    _create_temporal_mean_step(
                        in_4d=aroma_out_4d,
                        out_3d=aroma_out_mean,
                        env=env,
                        force=opts.overwrite,
                        chunk_vols=opts.io_chunk_vols,
                    )
                )
                if classifier == "ica_aroma":
                    runner.add_step(
                        create_json_step(
                            step_name=f"Write {space} ICA-AROMA Regression Policy",
                            path=policy_path,
                            payload=_ica_aroma_shared_regression_policy_payload(
                                input_space=space,
                                denoise_type=denoise_type,
                                repetition_time=repetition_time,
                                shared_work_dir=aroma_t1_dir,
                            ),
                            inputs=(aroma_out_4d, aroma_out_mean),
                            force=opts.overwrite,
                        )
                    )
                final_4d = aroma_out_4d
                final_mean = aroma_out_mean

        runner.add_step(
            create_copy_nifti_step(
                src=final_4d,
                dst=final_dst,
                force=opts.overwrite,
            )
        )
        final_output_4d = final_dst

        final_sources_4d[space] = final_output_4d
        final_sources_mean[space] = final_mean
        final_masks[space] = mask_3d

    confounds_space = "ACPC" if "ACPC" in final_sources_4d else "MNI152NLin2009cAsym"
    confounds_source_4d = final_sources_4d[confounds_space]
    confounds_source_mean = final_sources_mean[confounds_space]
    confounds_mask = final_masks[confounds_space]

    runner.add_step(
        _create_confounds_step(
            epi_4d=confounds_source_4d,
            epi_mean_3d=confounds_source_mean,
            mc_dir=mc_dir,
            subjects_dir=subjects_dir,
            fs_subject=fs_subject,
            brain_mask_in_epi=confounds_mask,
            out_tsv=confounds_tsv,
            out_json=confounds_json,
            force=opts.overwrite,
        )
    )

    if (not use_syn_fallback) and topup_native is not None and warp_sbref is not None:
        runner.add_step(
            create_copy_nifti_step(
                src=field_hz_regref,
                dst=fmap_field_hz_out,
                force=opts.overwrite,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        runner.add_step(
            create_copy_nifti_step(
                src=topup_native.out_prefix.with_name("topup_results_fieldcoef.nii.gz"),
                dst=fmap_topup_coeff_out,
                force=opts.overwrite,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        runner.add_step(
            create_copy_nifti_step(
                src=warp_sbref,
                dst=fmap_sdc_warp_out,
                force=opts.overwrite,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        if warp_bold_to_reg_ref is not None:
            runner.add_step(
                create_copy_nifti_step(
                    src=warp_bold_to_reg_ref,
                    dst=fmap_bold_sdc_warp_out,
                    force=opts.overwrite,
                    step_name="Finalize BOLD-Readout SDC Warp",
                )
            )
        jac_sbref = sdc_dir / f"Jacobian_{reg_ref_tag}Space.nii.gz"
        if opts.use_jacobian:
            runner.add_step(
                create_copy_nifti_step(
                    src=jac_sbref,
                    dst=fmap_jacobian_out,
                    force=opts.overwrite,
                    step_name="Finalize Fieldmap Derivative",
                )
            )
        if synthetic_ref is not None:
            runner.add_step(
                create_copy_nifti_step(
                    src=synthetic_ref,
                    dst=fmap_synbold_ref_out,
                    force=opts.overwrite,
                    step_name="Finalize SynBOLD-DisCo Reference",
                )
            )
            assert synbold_rigid_qc is not None
            runner.add_step(
                create_copy_nifti_step(
                    src=synbold_rigid_qc,
                    dst=fmap_synbold_rigid_out,
                    force=opts.overwrite,
                    step_name="Finalize SynBOLD Rigid-Registration QC",
                )
            )

    if ica_enabled:
        runner.add_step(
            create_copy_nifti_step(
                src=aroma_t1_dir / "aroma" / "melodic.ica" / "melodic_IC.nii.gz",
                dst=melodic_ic_t1_out,
                force=opts.overwrite,
                step_name="Finalize ICA-AROMA Derivative",
            )
        )

    final_preproc_source = final_sources_4d.get("ACPC", confounds_source_4d)
    final_t1_surface_source_4d = final_sources_4d.get("ACPC", final_preproc_source)
    fsnative_metric_outputs = (
        preproc_fsnative
        if want_fsnative
        else {
            "L": surf_dir
            / _with_suffix(f"{run_base}_space-fsnative_hemi-L", "_desc-preproc_internal.func.gii"),
            "R": surf_dir
            / _with_suffix(f"{run_base}_space-fsnative_hemi-R", "_desc-preproc_internal.func.gii"),
        }
    )
    if need_surface_outputs:
        for hemi in ("L", "R"):
            runner.add_step(
                _create_wb_volume_to_surface_mapping_step(
                    volume=final_t1_surface_source_4d,
                    midthickness=fsnative_surfaces[f"{hemi}.midthickness"],
                    white=fsnative_surfaces[f"{hemi}.white"],
                    pial=fsnative_surfaces[f"{hemi}.pial"],
                    out_metric=fsnative_metric_outputs[hemi],
                    env=env,
                    force=opts.overwrite,
                )
            )
            if want_fsaverage:
                if anatomical_domain.lesion_aware:
                    resampled = surf_dir / _with_suffix(
                        f"{run_base}_space-{fsaverage_space}_hemi-{hemi}",
                        "_desc-preproc_unmasked.func.gii",
                    )
                    valid_roi = surf_dir / _with_suffix(
                        f"{run_base}_space-{fsaverage_space}_hemi-{hemi}",
                        "_desc-validAnatomicalDomain_roi.shape.gii",
                    )
                    runner.add_step(
                        _create_wb_metric_resample_step(
                            in_metric=fsnative_metric_outputs[hemi],
                            current_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.current"],
                            new_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.new"],
                            out_metric=resampled,
                            valid_roi_out=valid_roi,
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                    runner.add_step(
                        _create_wb_metric_mask_step(
                            metric=resampled,
                            roi=valid_roi,
                            output=preproc_fsaverage[hemi],
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                else:
                    runner.add_step(
                        _create_wb_metric_resample_step(
                            in_metric=fsnative_metric_outputs[hemi],
                            current_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.current"],
                            new_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.new"],
                            out_metric=preproc_fsaverage[hemi],
                            env=env,
                            force=opts.overwrite,
                        )
                    )

    clean_inputs: dict[str, object] = {
        "desc-preproc": {
            "volumes": [str(preproc_t1), *([str(preproc_mni)] if want_mni else [])],
            "surfaces": {
                **(
                    {"fsnative": {hemi: str(path) for hemi, path in preproc_fsnative.items()}}
                    if want_fsnative
                    else {}
                ),
                **(
                    {fsaverage_space: {hemi: str(path) for hemi, path in preproc_fsaverage.items()}}
                    if want_fsaverage
                    else {}
                ),
            },
        }
    }
    public_images = [
        preproc_t1,
        *([preproc_mni] if want_mni else []),
        *(list(preproc_fsnative.values()) if want_fsnative else []),
        *(list(preproc_fsaverage.values()) if want_fsaverage else []),
        boldref_t1_out,
        reg_prenonlinear_qc_out,
        reg_base_qc_out,
        reg_refine_qc_out,
        anat_brain_mask_in_t1,
        *([reg_mni_qc_out] if want_mni else []),
        *([melodic_ic_t1_out] if ica_enabled else []),
        *(
            [
                marss_outputs.loadings,
                marss_outputs.mean_absolute_artifact,
                marss_outputs.slice_score_map,
            ]
            if marss_outputs is not None
            else []
        ),
    ]
    if not use_syn_fallback:
        public_images.extend(
            [fieldmap_hz_in_t1_out, fmap_field_hz_out, fmap_sdc_warp_out, fmap_topup_coeff_out]
        )
        if use_fieldmap_sdc:
            public_images.append(fmap_bold_sdc_warp_out)
        if opts.use_jacobian:
            public_images.append(fmap_jacobian_out)
        if use_synbold_reference:
            public_images.extend((fmap_synbold_ref_out, fmap_synbold_rigid_out))
    public_images = list(dict.fromkeys(public_images))
    metadata_outputs = tuple(sidecar_json_path(path) for path in public_images)
    marss_public_files = (
        (
            marss_outputs.metadata,
            marss_outputs.timecourses,
            marss_outputs.correlations_before,
            marss_outputs.correlations_after,
            marss_outputs.heatmap,
        )
        if marss_outputs is not None
        else ()
    )
    public_files = tuple(public_images) + (confounds_tsv, confounds_json, *marss_public_files)

    publication_identity = {
        "manifest_version": 2,
        "module": "func",
        "run_stem": run_base,
        "complete": True,
        "inputs": {
            "epi": str(source_inputs.epi),
            "epi_metadata": [str(path) for path in epi_metadata_sources],
            "sbref": str(source_inputs.sbref) if source_inputs.sbref is not None else None,
            "se1": str(source_inputs.se1) if source_inputs.se1 is not None else None,
            "se2": str(source_inputs.se2) if source_inputs.se2 is not None else None,
            "anatomical_manifest": str(anat_manifest),
        },
        "options": configuration,
        "public_outputs": {
            "clean_inputs": clean_inputs,
            "confounds_tsv": str(confounds_tsv),
            "confounds_json": str(confounds_json),
            "files": [str(path) for path in public_files],
        },
        "output_metadata_contract": functional_output_contract(),
    }
    if anatomical_domain.lesion_aware:
        publication_inputs = publication_identity["inputs"]
        assert isinstance(publication_inputs, dict)
        publication_inputs["anatomical_registration_image"] = str(registration_t1)
        publication_inputs["anatomical_lesion_mask"] = str(anatomical_domain.lesion_mask)

    def publication_payload() -> dict[str, object]:
        selection = read_json(selected_reference_metadata)
        multiband_artifact = (
            read_json(marss_outputs.metadata) if marss_outputs is not None else None
        )
        motion_indices: list[int] = []
        classified = (
            aroma_t1_dir / "aroma/classified_motion_ICs.txt"
            if classifier == "ica_aroma"
            else aroma_t1_dir / "cicada/classification/cicada_python/noise_components.txt"
        )
        if ica_enabled and classified.is_file():
            try:
                motion_indices = [
                    int(value)
                    for value in classified.read_text(encoding="utf-8").strip().split(",")
                    if value.strip()
                ]
            except ValueError:
                motion_indices = []
        return {
            **publication_identity,
            "registration": {
                "method": registration_method,
                "requested_sdc_method": requested_sdc_method,
                "sdc_method": resolved_sdc_method,
                "sdc_fallback_reason": sdc_fallback_reason,
                "reference_selection": selection,
                "static_warp": str(warp_sbref2t1_refined),
                "final_resampling": final_resampling_metadata(
                    gradient_unwarping=gradient_resolution.applied
                ),
                "gradient_unwarping": {
                    key: resolution.scientific_record()
                    for key, resolution in gradient_resolutions.items()
                },
                "fieldmap_transfer": fieldmap_transfer_details,
                "pe_residual_refinement": pe_residual_details,
            },
            "denoising": {
                "applied": ica_enabled,
                "method": ("ICA-AROMA" if classifier == "ica_aroma" else "CICADA")
                if ica_enabled
                else None,
                "mode": denoise_type if ica_enabled else None,
                "removed_noise_ic_indices": motion_indices,
                **(
                    {"simultaneous_slice_artifact": multiband_artifact}
                    if multiband_artifact is not None
                    else {}
                ),
            },
        }

    def publish() -> None:
        payload = publication_payload()
        validate_functional_manifest(payload)
        for image_path, metadata_path in zip(public_images, metadata_outputs):
            space = bids_entity(image_path, "space", default=None)
            metadata = {
                **epi_input_meta,
                "Description": "nro functional preprocessing derivative.",
                "Sources": [str(source_inputs.epi), str(anat_manifest)],
                "SpatialReference": space,
                "Registration": payload["registration"],
                "Denoising": payload["denoising"],
                "Configuration": configuration["configuration"],
                "ConfigurationFingerprint": configuration["configuration_fingerprint"],
            }
            validate_functional_image_sidecar(metadata)
            write_json(metadata_path, metadata)
        write_json(publication_manifest, payload)

    def validate_publication() -> tuple[bool, str]:
        try:
            current = read_json(publication_manifest)
            validate_functional_manifest(current)
        except (OSError, ValueError, TypeError):
            return False, "Functional publication manifest is missing or unreadable."
        for key, expected in publication_identity.items():
            if current.get(key) != expected:
                return False, "Functional publication manifest differs from the requested module."
        registration = current.get("registration")
        if not isinstance(registration, dict) or any(
            registration.get(key) != expected
            for key, expected in {
                "method": registration_method,
                "requested_sdc_method": requested_sdc_method,
                "sdc_method": resolved_sdc_method,
                "sdc_fallback_reason": sdc_fallback_reason,
                "static_warp": str(warp_sbref2t1_refined),
                "final_resampling": final_resampling_metadata(
                    gradient_unwarping=gradient_resolution.applied
                ),
                "gradient_unwarping": {
                    key: resolution.scientific_record()
                    for key, resolution in gradient_resolutions.items()
                },
                "fieldmap_transfer": fieldmap_transfer_details,
                "pe_residual_refinement": pe_residual_details,
            }.items()
        ):
            return False, "Functional registration publication differs from the requested module."
        denoising = current.get("denoising")
        if not isinstance(denoising, dict) or any(
            denoising.get(key) != expected
            for key, expected in {
                "applied": ica_enabled,
                "method": ("ICA-AROMA" if classifier == "ica_aroma" else "CICADA")
                if ica_enabled
                else None,
                "mode": denoise_type if ica_enabled else None,
            }.items()
        ):
            return False, "Functional denoising publication differs from the requested module."
        try:
            for metadata_path in metadata_outputs:
                validate_functional_image_sidecar(read_json(metadata_path))
        except (OSError, TypeError, ValueError):
            return False, "A functional image sidecar violates its metadata contract."
        missing = [
            str(path)
            for path in (*public_files, *metadata_outputs)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "Functional publication is missing outputs: " + ", ".join(missing)
        return True, "Functional publication is complete and current."

    runner.add_step(
        Step.python(
            name="Publish Functional Derivatives",
            outputs=(*metadata_outputs, publication_manifest),
            inputs=(*public_files, selected_reference_metadata, robust_reference_metadata),
            force=opts.overwrite,
            action=publish,
            validate=validate_publication,
            completion_boundary=True,
        )
    )
    return runner


def run(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> None:
    """Construct the module graph, execute it through the shared runner, and publish outputs.

    Freshness is evaluated after graph construction. Processing and validation
    errors propagate to the caller; partial private outputs can support resumption.
    """
    runner_started = time.perf_counter()
    runner = build_module(inputs, opts, execution_context=execution_context)
    with runner.run_context(started_at=runner_started):
        runner.execute()


def _build_argparser() -> argparse.ArgumentParser:
    cfg = SETTINGS.func
    p = argparse.ArgumentParser(
        prog="nro.modules.func.module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--sbref", type=Path)
    p.add_argument("--epi", type=Path)
    p.add_argument("--se1", type=Path)
    p.add_argument("--se2", type=Path)
    p.add_argument("--se1-json", type=Path)
    p.add_argument("--se2-json", type=Path)
    p.add_argument("--epi-json", type=Path)
    p.add_argument("--sbref-json", type=Path)
    p.add_argument(
        "--run-stem",
        type=str,
        default=None,
        help="Minimal-mode BIDS run identifier, e.g. sub-c001_ses-ex123_task-langloc_run-01",
    )
    p.add_argument(
        "--sdc-from-sbref-pair",
        action="store_true",
        default=bool(cfg.sdc_from_sbref_pair),
        help="In minimal mode, resolve the topup pair from opposite-PE SBRefs instead of fmap/ SE fieldmaps.",
    )

    p.add_argument("--bbregister-surf", choices=["white", "pial"], default=cfg.bbregister_surf)
    p.add_argument(
        "--bbregister-init", choices=["coreg", "fsl", "header", "rr"], default=cfg.bbregister_init
    )
    p.add_argument("--bbregister-dof", type=int, choices=[6, 9, 12], default=cfg.bbregister_dof)
    p.add_argument("--output-grid", choices=["t1_native", "t1_epi_vox"], default=cfg.output_grid)

    p.add_argument(
        "--sdc-method",
        choices=["syn", "synbold_disco"],
        default=str(cfg.sdc_method),
        help=(
            "Fieldmapless SDC fallback. Reverse-PE fieldmaps, when available, "
            "instead use TOPUP, readout-aware field transfer, SBRef PE refinement, "
            "and BBR."
        ),
    )
    p.add_argument("--synbold-disco-image", type=Path, default=Path(cfg.synbold_disco_image))
    p.add_argument("--synbold-disco-license", type=Path, default=Path(cfg.synbold_disco_license))
    p.add_argument(
        "--synbold-overlap-erosion-voxels",
        type=int,
        default=int(cfg.synbold_overlap_erosion_voxels),
    )
    p.add_argument(
        "--synbold-min-overlap-voxels", type=int, default=int(cfg.synbold_min_overlap_voxels)
    )
    p.add_argument(
        "--synbold-max-rigid-translation-mm",
        type=float,
        default=float(cfg.synbold_max_rigid_translation_mm),
    )
    p.add_argument(
        "--synbold-max-rigid-rotation-degrees",
        type=float,
        default=float(cfg.synbold_max_rigid_rotation_degrees),
    )
    p.add_argument(
        "--sbref-max-rigid-displacement-mm",
        type=float,
        default=float(cfg.sbref_max_rigid_displacement_mm),
    )
    p.add_argument(
        "--sbref-max-rigid-rotation-degrees",
        type=float,
        default=float(cfg.sbref_max_rigid_rotation_degrees),
    )
    p.add_argument(
        "--sbref-min-support-overlap", type=float, default=float(cfg.sbref_min_support_overlap)
    )
    p.add_argument(
        "--sbref-min-intensity-correlation",
        type=float,
        default=float(cfg.sbref_min_intensity_correlation),
    )
    p.add_argument("--topup-config", type=str, default=cfg.topup_config)
    p.add_argument("--ica-aroma-cmd", type=Path, default=cfg.ica_aroma_cmd)
    p.add_argument("--cicada-cmd", type=Path, default=cfg.cicada_cmd)
    p.add_argument(
        "--ica-classifier",
        choices=["none", "ica_aroma", "cicada"],
        default=cfg.ica_classifier,
    )
    p.add_argument(
        "--ica-regression",
        choices=["aggressive", "nonaggressive"],
        default=cfg.ica_regression,
    )
    p.add_argument("--cicada-tolerance", type=int, default=int(cfg.cicada_tolerance))
    p.add_argument(
        "--cicada-smoothing-retention-mode",
        choices=["revised", "historical"],
        default=cfg.cicada_smoothing_retention_mode,
    )
    p.add_argument(
        "--use-jacobian",
        action="store_true",
        default=cfg.use_jacobian,
        help=(
            "Experimentally Jacobian-modulate the corrected registration reference. "
            "Disabled by default; this does not modulate the final BOLD timeseries."
        ),
    )
    p.add_argument(
        "--fieldmap-syn-refine",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.fieldmap_syn_refine),
        help=(
            "Experimentally apply unrestricted anatomical SyN after reverse-PE "
            "fieldmap correction, SBRef PE refinement, and BBR. Disabled by default."
        ),
    )
    p.add_argument("--syn-base-transform", type=str, default=str(cfg.syn_base_transform))
    p.add_argument("--syn-base-convergence", type=str, default=str(cfg.syn_base_convergence))
    p.add_argument("--syn-base-shrink-factors", type=str, default=str(cfg.syn_base_shrink_factors))
    p.add_argument(
        "--syn-base-smoothing-sigmas", type=str, default=str(cfg.syn_base_smoothing_sigmas)
    )
    p.add_argument("--syn-refine-transform", type=str, default=str(cfg.syn_refine_transform))
    p.add_argument("--syn-refine-convergence", type=str, default=str(cfg.syn_refine_convergence))
    p.add_argument(
        "--syn-refine-shrink-factors", type=str, default=str(cfg.syn_refine_shrink_factors)
    )
    p.add_argument(
        "--syn-refine-smoothing-sigmas", type=str, default=str(cfg.syn_refine_smoothing_sigmas)
    )
    p.add_argument(
        "--marss-mode",
        choices=["off", "diagnose", "auto"],
        default=cfg.marss_mode,
        help="Diagnose or correct the native simultaneous-slice artifact before resampling.",
    )
    p.add_argument(
        "--marss-min-multiband-factor",
        type=int,
        default=int(cfg.marss_min_multiband_factor),
        help="Minimum multiband factor corrected by MARSS auto mode.",
    )
    p.add_argument(
        "--gradient-unwarping",
        choices=["auto", "off"],
        default=cfg.gradient_unwarping,
        help="Apply a matching site hardware profile, or disable gradient unwarping.",
    )

    p.add_argument(
        "--project",
        default=SETTINGS.common.project,
        help="BIDS project name under the configured top-level data directory.",
    )
    p.add_argument(
        "--func-id",
        default=SETTINGS.common.func_id,
        help="Functional module configuration ID.",
    )
    p.add_argument(
        "--anat-id",
        default=SETTINGS.common.anat_id,
        help="Anatomical module configuration ID.",
    )
    p.add_argument("--fsaverage-template", default=cfg.fsaverage_template)
    p.add_argument("--sub-id", required=True, help="Subject identifier, e.g. sub-c001")
    p.add_argument("--ses-id", default=None, help="Optional session identifier, e.g. ses-ex31524")
    p.add_argument("--work-dir", type=Path, default=cfg.work_dir)

    p.add_argument(
        "--debug-first-nvols",
        type=int,
        default=cfg.debug_first_nvols,
        help="If >0, run on only the first N volumes of the EPI (faster debug runs).",
    )
    p.add_argument(
        "--io-chunk-vols",
        type=int,
        default=int(cfg.io_chunk_vols),
        help="Number of volumes to read at a time for chunked nibabel-based I/O operations.",
    )
    p.add_argument("--overwrite", action="store_true", default=cfg.overwrite)
    p.add_argument(
        "--output-spaces",
        nargs="+",
        default=list(supported_output_spaces(str(cfg.fsaverage_template))),
        help="Supported output spaces to generate.",
    )
    p.add_argument("--verbose", action="store_true", default=cfg.verbose)

    add_container_arguments(p, cfg.container)
    return p


def main(
    argv: Optional[Sequence[str]] = None, *, execution_context: ExecutionContext | None = None
) -> None:
    """Parse CLI arguments and run the functional module."""
    args = _build_argparser().parse_args(argv)
    if int(args.marss_min_multiband_factor) < 2:
        raise SystemExit("--marss-min-multiband-factor must be at least 2")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project = str(args.project)
    explicit_mode = args.epi is not None or args.epi_json is not None
    minimal_mode = bool(str(args.run_stem or "").strip())
    if explicit_mode and minimal_mode:
        raise SystemExit(
            "Use either explicit input paths (--epi/--epi-json/...) or minimal mode (--run-stem), not both."
        )
    if (not explicit_mode) and (not minimal_mode):
        raise SystemExit(
            "Either explicit input paths (--epi and --epi-json) or minimal mode (--run-stem) is required."
        )

    for key in (
        "sbref",
        "epi",
        "se1",
        "se2",
        "se1_json",
        "se2_json",
        "epi_json",
        "sbref_json",
    ):
        setattr(args, key, resolve_project_path(getattr(args, key), project=project))
    args.ica_aroma_cmd = resolve_cwd_path(args.ica_aroma_cmd)
    args.cicada_cmd = resolve_cwd_path(args.cicada_cmd)
    args.synbold_disco_image = resolve_cwd_path(args.synbold_disco_image)
    args.synbold_disco_license = resolve_cwd_path(args.synbold_disco_license)

    ses_id = str(args.ses_id).strip() if args.ses_id is not None else None
    if ses_id == "":
        ses_id = None
    if ses_id is not None and (not is_bids_session_id(ses_id)):
        raise SystemExit(f"--ses-id must look like a BIDS session ID (ses-*), got {ses_id!r}")

    resolved: Optional[ResolvedFuncRun] = None
    if minimal_mode:
        resolved = resolve_func_run_request(
            project=project,
            sub_id=str(args.sub_id),
            ses_id=ses_id,
            run_stem=str(args.run_stem).strip(),
            sdc_from_sbref_pair=bool(args.sdc_from_sbref_pair),
        )
        args.epi = resolved.bold.img
        args.epi_json = resolved.bold.metadata_path
        args.sbref = resolved.sbref.img if resolved.sbref is not None else None
        args.sbref_json = resolved.sbref.metadata_path if resolved.sbref is not None else None
        args.se1 = resolved.pair.se1.img if resolved.pair is not None else None
        args.se2 = resolved.pair.se2.img if resolved.pair is not None else None
        args.se1_json = resolved.pair.se1.metadata_path if resolved.pair is not None else None
        args.se2_json = resolved.pair.se2.metadata_path if resolved.pair is not None else None
        if resolved.selection_warning:
            LOG.info(
                "Resolved %s -> %s (%s)",
                resolved.requested_run_stem,
                resolved.resolved_run_stem,
                resolved.selection_warning,
            )
        else:
            LOG.info("Resolved %s -> %s", resolved.requested_run_stem, resolved.resolved_run_stem)
    else:
        if args.epi is None or args.epi_json is None:
            raise SystemExit("Explicit mode requires both --epi and --epi-json.")
        if (args.se1 is None) != (args.se1_json is None):
            raise SystemExit("--se1 and --se1-json must be provided together.")
        if (args.se2 is None) != (args.se2_json is None):
            raise SystemExit("--se2 and --se2-json must be provided together.")
        if (args.sbref is None) != (args.sbref_json is None):
            raise SystemExit("--sbref and --sbref-json must be provided together.")

    if ses_id is None:
        out_dir = func_subject_dir(str(args.sub_id), project=project, func_id=str(args.func_id))
        work_session_dir = func_subject_work_dir(
            str(args.sub_id), project=project, func_id=str(args.func_id)
        )
    else:
        out_dir = func_session_dir(
            str(args.sub_id), ses_id, project=project, func_id=str(args.func_id)
        )
        work_session_dir = func_session_work_dir(
            str(args.sub_id), ses_id, project=project, func_id=str(args.func_id)
        )
    work_dir: Path = resolve_project_work_path(
        args.work_dir or (work_session_dir / nifti_stem(Path(args.epi))),
        project=project,
    )
    epi_metadata = None
    epi_metadata_sources: tuple[Path, ...] = ()
    sbref_metadata = None
    sbref_metadata_sources: tuple[Path, ...] = ()
    se1_metadata = None
    se1_metadata_sources: tuple[Path, ...] = ()
    se2_metadata = None
    se2_metadata_sources: tuple[Path, ...] = ()
    sbref_metadata_inheritance = None
    if resolved is not None:
        epi_metadata = dict(resolved.bold.metadata)
        epi_metadata_sources = resolved.bold.metadata_sources
        if resolved.sbref is not None:
            sbref_metadata = dict(resolved.sbref.metadata)
            sbref_metadata_sources = resolved.sbref.metadata_sources
        if resolved.pair is not None:
            se1_metadata = dict(resolved.pair.se1.metadata)
            se1_metadata_sources = resolved.pair.se1.metadata_sources
            se2_metadata = dict(resolved.pair.se2.metadata)
            se2_metadata_sources = resolved.pair.se2.metadata_sources
        if resolved.sbref is not None and resolved.sbref.metadata_inheritance is not None:
            sbref_metadata_inheritance = dict(resolved.sbref.metadata_inheritance)
            LOG.info(
                "Using SBRef %s with metadata inherited from exact-run BOLD metadata %s.",
                resolved.sbref.img,
                ", ".join(str(path) for path in resolved.bold.metadata_sources),
            )

    container = build_container(
        ContainerSettings.from_args(args),
        work_directory=work_dir,
        execution_context=execution_context,
    )
    site, _ = site_settings()

    inputs = Inputs(
        sbref=args.sbref,
        epi=args.epi,
        se1=args.se1,
        se2=args.se2,
        se1_json=args.se1_json,
        se2_json=args.se2_json,
        epi_json=args.epi_json,
        sbref_json=args.sbref_json,
        epi_metadata=epi_metadata,
        epi_metadata_sources=epi_metadata_sources,
        sbref_metadata=sbref_metadata,
        sbref_metadata_sources=sbref_metadata_sources,
        se1_metadata=se1_metadata,
        se1_metadata_sources=se1_metadata_sources,
        se2_metadata=se2_metadata,
        se2_metadata_sources=se2_metadata_sources,
        sbref_metadata_inheritance=sbref_metadata_inheritance,
    )

    opts = Options(
        out_dir=out_dir,
        work_dir=work_dir,
        project=project,
        func_id=str(args.func_id),
        anat_id=str(args.anat_id),
        sub_id=str(args.sub_id),
        ses_id=ses_id,
        overwrite=bool(args.overwrite),
        output_grid=str(args.output_grid),
        topup_config=str(args.topup_config),
        ica_aroma_cmd=(Path(args.ica_aroma_cmd) if args.ica_aroma_cmd is not None else None),
        cicada_cmd=Path(args.cicada_cmd),
        ica_classifier=str(args.ica_classifier),
        ica_regression=str(args.ica_regression),
        cicada_tolerance=int(args.cicada_tolerance),
        cicada_smoothing_retention_mode=str(args.cicada_smoothing_retention_mode),
        use_jacobian=bool(args.use_jacobian),
        fieldmap_syn_refine=bool(args.fieldmap_syn_refine),
        syn_base_transform=str(args.syn_base_transform),
        syn_base_convergence=str(args.syn_base_convergence),
        syn_base_shrink_factors=str(args.syn_base_shrink_factors),
        syn_base_smoothing_sigmas=str(args.syn_base_smoothing_sigmas),
        syn_refine_transform=str(args.syn_refine_transform),
        syn_refine_convergence=str(args.syn_refine_convergence),
        syn_refine_shrink_factors=str(args.syn_refine_shrink_factors),
        syn_refine_smoothing_sigmas=str(args.syn_refine_smoothing_sigmas),
        marss_mode=str(args.marss_mode),
        marss_min_multiband_factor=int(args.marss_min_multiband_factor),
        gradient_unwarping=str(args.gradient_unwarping),
        gradient_unwarp_image=Path(site["gradient_unwarp"]),
        gradient_unwarp_runtime=str(site["runtime"]),
        fsaverage_template=str(args.fsaverage_template),
        bbregister_surf=str(args.bbregister_surf),
        bbregister_init=str(args.bbregister_init),
        bbregister_dof=int(args.bbregister_dof),
        container=container,
        debug_first_nvols=int(args.debug_first_nvols),
        output_spaces=normalize_output_spaces(args.output_spaces),
        io_chunk_vols=max(1, int(args.io_chunk_vols)),
        sdc_method=str(args.sdc_method),
        synbold_disco_image=Path(args.synbold_disco_image),
        synbold_disco_license=Path(args.synbold_disco_license),
        synbold_disco_engine=str(args.container_engine),
        synbold_overlap_erosion_voxels=max(0, int(args.synbold_overlap_erosion_voxels)),
        synbold_min_overlap_voxels=max(1, int(args.synbold_min_overlap_voxels)),
        synbold_max_rigid_translation_mm=float(args.synbold_max_rigid_translation_mm),
        synbold_max_rigid_rotation_degrees=float(args.synbold_max_rigid_rotation_degrees),
        sbref_max_rigid_displacement_mm=float(args.sbref_max_rigid_displacement_mm),
        sbref_max_rigid_rotation_degrees=float(args.sbref_max_rigid_rotation_degrees),
        sbref_min_support_overlap=float(args.sbref_min_support_overlap),
        sbref_min_intensity_correlation=float(args.sbref_min_intensity_correlation),
    )

    run(inputs, opts, execution_context=execution_context)


if __name__ == "__main__":
    main()
