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
- When a fieldmap pair is available and `fieldmap_syn_refine` is set, an ANTs SyN
  refinement is estimated before the final BOLD-to-T1 resampling.
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
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

from nro.definitions.hardware import resolve_gradient_unwarping
from nro.definitions.runtime import SETTINGS
from nro.engine.anatomical_domain import load_anatomical_domain
from nro.engine.container import (
    ContainerSettings,
    add_container_arguments,
    bind_container_to_execution,
    build_container,
)
from nro.engine.execution import (
    collect_bind_directories,
    create_copy_file_step,
    neuroimaging_environment,
    require_existing_path,
)
from nro.engine.image_paths import nifti_stem, uncompressed_nifti_path
from nro.engine.images import (
    nifti_spatial_shape,
    nifti_volume_count,
)
from nro.engine.io import read_json
from nro.engine.manifests import (
    require_nested_manifest_output,
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
from nro.modules.func.resolver import (
    ResolvedFuncRun,
    resolve_func_run_request,
)
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.site.configuration import settings as site_settings

from .config import Inputs, Options, normalize_output_spaces
from .construction import (
    plan_anatomical_refinement,
    plan_confounds_and_ica,
    plan_distortion_correction,
    plan_initialization,
    plan_input_preparation,
    plan_publication,
    plan_reference_preparation,
    plan_registration,
    plan_registration_qc,
    plan_space_outputs,
    plan_surface_outputs,
    plan_t1_reference,
)
from .graph_policy import FuncGraphPolicy
from .layout import FuncLayout
from .registration_steps import (
    _create_mris_convert_step,
)
from .step_support import (
    _ica_aroma_output_label,
    _resolve_fieldmapless_sdc_method,
    _resolve_sdc_reference_policy,
    _with_suffix,
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
        "t1w_to_mni",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    require_nested_manifest_output(
        anat_info,
        "xfms",
        "mni_to_t1w",
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
    if repetition_time is not None and (not math.isfinite(repetition_time) or repetition_time <= 0):
        repetition_time = None
    if repetition_time is None:
        raise SystemExit("BOLD metadata require a finite positive RepetitionTime.")
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
        "3drefit",
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
    layout = FuncLayout(opts=opts, run_stem=run_stem, classifier=classifier)
    source_volume_count = nifti_volume_count(inputs.epi)
    source_spatial_shape = nifti_spatial_shape(inputs.epi)
    source_fieldmap_geometry = (
        {
            kind: (nifti_spatial_shape(path), nifti_volume_count(path))
            for kind, path in (("se1", inputs.se1), ("se2", inputs.se2))
            if path is not None
        }
        if inputs.se1 is not None and inputs.se2 is not None
        else {}
    )
    source_inputs = inputs
    preparation = plan_input_preparation(
        opts=opts,
        inputs=inputs,
        epi_metadata=epi_input_meta,
        gradient_resolutions=gradient_resolutions,
        environment=env,
        run_stem=run_stem,
        run_base=layout.run_base,
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
        motion_directory=layout.mc_dir,
        environment=env,
        epi_metadata=epi_input_meta,
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
    requested_spaces = set(opts.output_spaces)
    want_t1 = "T1w" in requested_spaces
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
    policy = FuncGraphPolicy(
        classifier=classifier,
        denoise_type=denoise_type,
        ica_enabled=ica_enabled,
        epi_metadata_sources=epi_metadata_sources,
        epi_input_meta=epi_input_meta,
        gradient_resolutions=gradient_resolutions,
        gradient_resolution=gradient_resolution,
        subjects_dir=subjects_dir,
        fs_subject=fs_subject,
        anat_t1=anat_t1,
        registration_t1=registration_t1,
        anat_brain_mask=anat_brain_mask,
        anat_mni_template=anat_mni_template,
        t1_to_mni_xfm=t1_to_mni_xfm,
        fsnative_surfaces=fsnative_surfaces,
        requested_sdc_method=requested_sdc_method,
        use_fieldmap_sdc=use_fieldmap_sdc,
        sdc_fallback_reason=sdc_fallback_reason,
        use_synbold_topup=use_synbold_topup,
        use_syn_fallback=use_syn_fallback,
        use_synbold_reference=use_synbold_reference,
        fieldmap_refinement_target=fieldmap_refinement_target,
        do_refinement=do_refinement,
        resolved_sdc_method=resolved_sdc_method,
        want_t1=want_t1,
        want_mni=want_mni,
        want_fsnative=want_fsnative,
        fsaverage_space=fsaverage_space,
        want_fsaverage=want_fsaverage,
        need_surface_outputs=need_surface_outputs,
    )
    LOG.info("Output spaces requested: %s", ", ".join(opts.output_spaces))
    fsnative_to_fsaverage_spheres: dict[str, Path] = {}
    if policy.want_fsaverage:
        sphere_work = layout.surf_dir / "fsaverage_spheres"
        for hemi, hemi_label in (("lh", "L"), ("rh", "R")):
            subject_sphere = require_nested_manifest_output(
                anat_info,
                "surfaces",
                f"{hemi}.sphere.reg",
                manifest_path=anat_manifest,
                manifest_name="Anatomical",
            )
            fsaverage_sphere = find_fsaverage_template_surface(
                template=policy.fsaverage_space,
                hemi=hemi_label,
                surface="sphere",
            )
            subject_output = sphere_work / (
                f"subject_hemi-{hemi_label}_from-fsnative_to-{policy.fsaverage_space}_sphere.surf.gii"
            )
            fsaverage_output = (
                sphere_work / f"{policy.fsaverage_space}_hemi-{hemi_label}_sphere.surf.gii"
            )
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
                    step_name=f"Stage {policy.fsaverage_space} Hemisphere {hemi_label} Sphere",
                )
            )
            fsnative_to_fsaverage_spheres[f"{hemi_label}.current"] = subject_output
            fsnative_to_fsaverage_spheres[f"{hemi_label}.new"] = fsaverage_output

    preproc_fsaverage = {
        "L": layout.func_dir
        / _with_suffix(
            f"{layout.run_base}_space-{policy.fsaverage_space}_hemi-L",
            "_desc-preproc_bold.func.gii",
        ),
        "R": layout.func_dir
        / _with_suffix(
            f"{layout.run_base}_space-{policy.fsaverage_space}_hemi-R",
            "_desc-preproc_bold.func.gii",
        ),
    }
    epi_t1 = uncompressed_nifti_path(
        layout.reg_dir / _with_suffix(layout.run_prefix, "_desc-preproc_bold.nii.gz")
    )
    epi_mni = uncompressed_nifti_path(
        layout.reg_dir
        / _with_suffix(f"{layout.run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_bold.nii.gz")
    )
    aroma_dir = opts.work_dir / ("ica_cicada" if policy.classifier == "cicada" else "ica_aroma")
    aroma_label = (
        ("cicadaAgg" if policy.denoise_type == "aggr" else "cicadaNonAgg")
        if policy.classifier == "cicada"
        else _ica_aroma_output_label(policy.denoise_type)
    )
    aroma_t1_dir = aroma_dir / "space-T1w"
    aroma_mni_dir = aroma_dir / "space-MNI152NLin2009cAsym"
    aroma_clean = uncompressed_nifti_path(
        aroma_t1_dir / _with_suffix(layout.run_prefix, f"_desc-{aroma_label}_bold.nii.gz")
    )
    aroma_clean_mean = aroma_t1_dir / _with_suffix(
        layout.run_prefix, f"_desc-{aroma_label}_mean.nii.gz"
    )
    aroma_clean_mni = uncompressed_nifti_path(
        aroma_mni_dir
        / _with_suffix(
            f"{layout.run_base}_space-MNI152NLin2009cAsym", f"_desc-{aroma_label}_bold.nii.gz"
        )
    )
    aroma_clean_mean_mni = aroma_mni_dir / _with_suffix(
        f"{layout.run_base}_space-MNI152NLin2009cAsym", f"_desc-{aroma_label}_mean.nii.gz"
    )
    publication_manifest = functional_manifest_path(
        opts.sub_id,
        layout.run_base,
        project=opts.project,
        func_id=opts.func_id,
        ses_id=opts.ses_id,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        publication_manifest = execution_context.output_path(publication_manifest)

    t1_ref = (
        layout.reg_dir / f"{run_stem}_t1_grid_epi_vox.nii.gz"
        if opts.output_grid == "t1_epi_vox"
        else layout.reg_dir / f"{run_stem}_t1_native.nii.gz"
    )
    warp_sbref2t1_refined_planned = layout.epi_to_t1_warp_planned
    if policy.use_syn_fallback:
        registration_method = "ants_synboldaff"
    elif policy.use_synbold_topup:
        registration_method = "synbold_disco_topup_create_bbregister_step"
    else:
        registration_method = (
            "topup_sbref_pe_create_bbregister_step_then_ants_syn"
            if policy.do_refinement
            else "topup_sbref_pe_create_bbregister_step"
        )
    distortion_correction = plan_distortion_correction(
        env=env,
        epi_to_reference=epi_to_reference,
        inputs=inputs,
        layout=layout,
        opts=opts,
        policy=policy,
        reg_ref_tag=reg_ref_tag,
        robust_ref=robust_ref,
        run_stem=run_stem,
        selected_reference_image=selected_reference_image,
        source_fieldmap_geometry=source_fieldmap_geometry,
        source_spatial_shape=source_spatial_shape,
        run_child=runner.run_child,
    )
    runner.add_steps(distortion_correction.steps)
    ants_forward_xfm = distortion_correction.products.ants_forward_xfm
    epi_mc_ref = distortion_correction.products.epi_mc_ref
    fieldmap_hz_in_t1 = distortion_correction.products.fieldmap_hz_in_t1
    fieldmap_transfer_details = distortion_correction.products.fieldmap_transfer_details
    fsl_mat = distortion_correction.products.fsl_mat
    jac_sbref = distortion_correction.products.jac_sbref
    pe_residual_details = distortion_correction.products.pe_residual_details
    reg_ref_dc_ref = distortion_correction.products.reg_ref_dc_ref
    reg_ref_dist_ref = distortion_correction.products.reg_ref_dist_ref
    se2sbref_mat = distortion_correction.products.se2sbref_mat
    synbold_rigid_qc = distortion_correction.products.synbold_rigid_qc
    synthetic_ref = distortion_correction.products.synthetic_ref
    topup_native = distortion_correction.products.topup_native
    topup_to_reg_ref_refined_mat = distortion_correction.products.topup_to_reg_ref_refined_mat
    warp_bold_to_reg_ref = distortion_correction.products.warp_bold_to_reg_ref
    warp_sbref = distortion_correction.products.warp_sbref
    warp_sbref2t1 = distortion_correction.products.warp_sbref2t1
    # Output grid
    t1_reference = plan_t1_reference(
        env=env,
        fsl_mat=fsl_mat,
        inputs=inputs,
        layout=layout,
        opts=opts,
        policy=policy,
        run_stem=run_stem,
        t1_ref=t1_ref,
    )
    runner.add_steps(t1_reference.steps)
    t1_ref = t1_reference.products.t1_ref
    registration = plan_registration(
        ants_forward_xfm=ants_forward_xfm,
        env=env,
        epi_mc_ref=epi_mc_ref,
        epi_to_reference=epi_to_reference,
        layout=layout,
        opts=opts,
        policy=policy,
        reg_ref_dc_ref=reg_ref_dc_ref,
        reg_ref_dist_ref=reg_ref_dist_ref,
        run_stem=run_stem,
        se2sbref_mat=se2sbref_mat,
        t1_ref=t1_ref,
        topup_native=topup_native,
        topup_to_reg_ref_refined_mat=topup_to_reg_ref_refined_mat,
        warp_bold_to_reg_ref=warp_bold_to_reg_ref,
        warp_sbref=warp_sbref,
        warp_sbref2t1=warp_sbref2t1,
        run_child=runner.run_child,
    )
    runner.add_steps(registration.steps)
    ants_forward_xfm = registration.products.ants_forward_xfm
    base_ref_in_t1 = registration.products.base_ref_in_t1
    base_static_warp = registration.products.base_static_warp
    pre_nonlinear_ref_in_t1 = registration.products.pre_nonlinear_ref_in_t1
    syn_moving = registration.products.syn_moving
    syn_refine_fnirt_warp = registration.products.syn_refine_fnirt_warp
    syn_work = registration.products.syn_work
    warp_regref2t1_refined = registration.products.warp_regref2t1_refined
    warp_sbref2t1_refined = registration.products.warp_sbref2t1_refined
    anatomical_refinement = plan_anatomical_refinement(
        ants_forward_xfm=ants_forward_xfm,
        base_ref_in_t1=base_ref_in_t1,
        base_static_warp=base_static_warp,
        env=env,
        layout=layout,
        opts=opts,
        policy=policy,
        run_stem=run_stem,
        syn_refine_fnirt_warp=syn_refine_fnirt_warp,
        syn_work=syn_work,
        t1_ref=t1_ref,
        warp_regref2t1_refined=warp_regref2t1_refined,
        warp_sbref2t1_refined=warp_sbref2t1_refined,
        warp_sbref2t1_refined_planned=warp_sbref2t1_refined_planned,
        run_child=runner.run_child,
    )
    runner.add_steps(anatomical_refinement.steps)
    warp_regref2t1_refined = anatomical_refinement.products.warp_regref2t1_refined
    warp_sbref2t1_refined = anatomical_refinement.products.warp_sbref2t1_refined

    assert (
        pre_nonlinear_ref_in_t1 is not None
        and warp_sbref2t1_refined is not None
        and warp_regref2t1_refined is not None
    )
    registration_qc = plan_registration_qc(
        base_ref_in_t1=base_ref_in_t1,
        env=env,
        fieldmap_hz_in_t1=fieldmap_hz_in_t1,
        layout=layout,
        opts=opts,
        policy=policy,
        pre_nonlinear_ref_in_t1=pre_nonlinear_ref_in_t1,
        syn_moving=syn_moving,
        t1_ref=t1_ref,
        topup_native=topup_native,
        warp_regref2t1_refined=warp_regref2t1_refined,
        warp_sbref2t1_refined=warp_sbref2t1_refined,
    )
    runner.add_steps(registration_qc.steps)

    space_outputs = plan_space_outputs(
        aroma_clean=aroma_clean,
        aroma_clean_mean=aroma_clean_mean,
        aroma_clean_mean_mni=aroma_clean_mean_mni,
        aroma_clean_mni=aroma_clean_mni,
        aroma_dir=aroma_dir,
        aroma_mni_dir=aroma_mni_dir,
        aroma_t1_dir=aroma_t1_dir,
        compute_t1=compute_t1,
        env=env,
        epi_for_proc=epi_for_proc,
        epi_mni=epi_mni,
        epi_t1=epi_t1,
        final_resampling_source=final_resampling_source,
        gradient_afni_warp=gradient_afni_warp,
        layout=layout,
        opts=opts,
        policy=policy,
        repetition_time=repetition_time,
        robust_ref=robust_ref,
        syn_moving=syn_moving,
        t1_ref=t1_ref,
        warp_regref2t1_refined=warp_regref2t1_refined,
        warp_sbref2t1_refined=warp_sbref2t1_refined,
        runner=runner,
        run_child=runner.run_child,
    )
    runner.add_steps(space_outputs.steps)
    final_masks = space_outputs.products.final_masks
    final_sources_4d = space_outputs.products.final_sources_4d
    final_sources_mean = space_outputs.products.final_sources_mean

    confounds_and_ica = plan_confounds_and_ica(
        aroma_t1_dir=aroma_t1_dir,
        final_masks=final_masks,
        final_sources_4d=final_sources_4d,
        final_sources_mean=final_sources_mean,
        jac_sbref=jac_sbref,
        layout=layout,
        opts=opts,
        policy=policy,
        reg_ref_tag=reg_ref_tag,
        repetition_time=repetition_time,
        synbold_rigid_qc=synbold_rigid_qc,
        synthetic_ref=synthetic_ref,
        topup_native=topup_native,
        warp_bold_to_reg_ref=warp_bold_to_reg_ref,
        warp_sbref=warp_sbref,
    )
    runner.add_steps(confounds_and_ica.steps)
    confounds_source_4d = confounds_and_ica.products.confounds_source_4d

    surface_outputs = plan_surface_outputs(
        anatomical_domain=anatomical_domain,
        confounds_source_4d=confounds_source_4d,
        env=env,
        final_sources_4d=final_sources_4d,
        fsnative_to_fsaverage_spheres=fsnative_to_fsaverage_spheres,
        layout=layout,
        opts=opts,
        policy=policy,
        preproc_fsaverage=preproc_fsaverage,
    )
    runner.add_steps(surface_outputs.steps)

    publication = plan_publication(
        anat_manifest=anat_manifest,
        anatomical_domain=anatomical_domain,
        aroma_t1_dir=aroma_t1_dir,
        configuration=configuration,
        fieldmap_transfer_details=fieldmap_transfer_details,
        layout=layout,
        marss_outputs=marss_outputs,
        opts=opts,
        pe_residual_details=pe_residual_details,
        policy=policy,
        preproc_fsaverage=preproc_fsaverage,
        publication_manifest=publication_manifest,
        registration_method=registration_method,
        robust_reference_metadata=robust_reference_metadata,
        selected_reference_metadata=selected_reference_metadata,
        source_inputs=source_inputs,
    )
    runner.add_steps(publication.steps)
    return runner


def run(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> None:
    """Construct the module graph and execute it through the shared runner.

    Freshness is evaluated after graph construction. Processing and validation
    errors propagate to the caller; partial private outputs can support resumption.
    """
    runner_started = time.perf_counter()
    runner = build_module(inputs, opts, execution_context=execution_context)
    runner.execute_graph(started_at=runner_started)


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
