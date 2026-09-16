"""Run anatomical preprocessing for one subject."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from nro.configuration.hardware import resolve_gradient_unwarping
from nro.configuration.runtime import SETTINGS
from nro.configuration.site import settings as site_settings
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
)
from nro.engine.gradient_unwarping import create_gradient_unwarping_step
from nro.engine.io import read_json, require_nonempty_file, write_json
from nro.engine.manifests import create_json_step
from nro.engine.neuroimaging import create_n4_bias_correction_step
from nro.engine.paths import (
    anat_subject_dir,
    anat_subject_work_dir,
    anatomical_manifest_path,
    module_derivatives_root,
    resolve_project_path,
    resolve_project_work_path,
)
from nro.engine.templates import find_fsaverage_template_surface
from nro.modules.anat.contract import anatomical_output_contract, validate_anatomical_manifest
from nro.modules.anat.inputs import (
    AnatImage,
    load_anat_image,
)
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import ContainerSpec, Runner, write_completion_breadcrumb
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import selected_configuration_fingerprint

from .constants import (
    _FREESURFER_GRAY_MATTER_SEGMENTATIONS,
    _FREESURFER_SUBCORTICAL_SEGMENTATIONS,
)
from .steps import (
    _aseg_label_ids,
    _brain_extract_anat_copy,
    _create_copy_or_average_step,
    _create_inverse_affine_step,
    _create_label_mask_step,
    _create_metric_conversion_step,
    _create_midthickness_step,
    _create_mni_qc_image_step,
    _create_mni_registration_step,
    _create_mri_conversion_step,
    _create_recon_all_step,
    _create_ribbon_mask_step,
    _create_surface_affine_step,
    _create_t1_to_fsnative_affine_step,
    _create_transform_finalization_step,
    _export_mgz,
    _hemi_label,
    _metric_names,
    _nested_paths,
    _plan_session_anatomicals,
    _register_t2_to_t1,
    _resample_mask_to_native,
    _strip_freesurfer_volgeom_metadata,
    _subject_preproc_path,
    _surface_names,
    _write_json_step,
)

LOG = logging.getLogger("anat")
# Standard aseg identifiers from FreeSurferColorLUT.txt:
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT


@dataclass(frozen=True)
class Inputs:
    """Subject identity and available T1w/T2w acquisitions for anatomical construction."""

    sub_id: str
    t1w: tuple[AnatImage, ...]
    t2w: tuple[AnatImage, ...]


@dataclass(frozen=True)
class Options:
    """Resolved anatomy resources, output paths, selection strategy, and execution controls."""

    project: str
    anat_id: str
    out_dir: Path
    work_dir: Path
    freesurfer_subjects_dir: Path
    fs_subject: str
    fsaverage_template: str
    selection_strategy: str
    gradient_unwarping: str
    gradient_unwarp_image: Path
    gradient_unwarp_runtime: str
    mni_template: Path
    container: Optional[ContainerSpec]
    synthstrip_image: Optional[Path]
    overwrite: bool


def build_module(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> Runner:
    """Construct the anatomical DAG, optionally routing writes to a selected owner.

    The caller authorizes the context. Raw inputs stay in the shared BIDS tree;
    public outputs, session copies, FreeSurfer files, and work stay with the owner.
    """
    if execution_context is not None:
        if execution_context.project != opts.project:
            raise ValueError("Anatomical project differs from its execution context")
        opts = replace(
            opts,
            out_dir=execution_context.output_path(opts.out_dir),
            work_dir=execution_context.output_path(opts.work_dir, private=True),
            freesurfer_subjects_dir=execution_context.output_path(opts.freesurfer_subjects_dir),
        )
        for image in (*inputs.t1w, *inputs.t2w):
            for path in (image.image, image.json):
                if path is not None and (
                    not path.absolute().is_relative_to(
                        execution_context.paths.source_project(opts.project)
                    )
                    or execution_context.input_path(path) != path
                ):
                    raise ValueError("Anatomical acquisitions must be shared source inputs")
    all_images = [*inputs.t1w, *inputs.t2w]
    if not all_images:
        raise SystemExit(f"No anatomical images provided for {inputs.sub_id}.")
    if opts.synthstrip_image is None:
        raise SystemExit("Missing SynthStrip image path.")
    require_nonempty_file(opts.synthstrip_image, "SynthStrip image")
    require_nonempty_file(opts.mni_template, "MNI template")

    gradient_resolutions = {
        image.image: resolve_gradient_unwarping(
            image.metadata,
            mode=opts.gradient_unwarping,
        )
        for image in all_images
    }
    if any(resolution.applied for resolution in gradient_resolutions.values()):
        require_nonempty_file(opts.gradient_unwarp_image, "gradient-unwarping image")

    public_inputs = [item.image for item in all_images]
    public_inputs.extend(source for item in all_images for source in item.metadata_sources)
    public_inputs = list(dict.fromkeys(public_inputs))
    env = neuroimaging_environment(subjects_dir=opts.freesurfer_subjects_dir)
    binds = collect_bind_directories(
        [
            *(item.image for item in all_images),
            *(item.json for item in all_images if item.json is not None),
            opts.out_dir,
            opts.work_dir,
            opts.freesurfer_subjects_dir,
            opts.mni_template,
            opts.gradient_unwarp_image
            if any(resolution.applied for resolution in gradient_resolutions.values())
            else None,
            *(resolution.coefficients for resolution in gradient_resolutions.values()),
            Path(env["FS_LICENSE"]) if Path(env["FS_LICENSE"]).is_file() else None,
        ]
    )
    runner = Runner(
        module_name="Anatomical Module",
        container=bind_container_to_execution(opts.container, execution_context),
        binds=binds,
        logger=LOG,
        execution_context=execution_context,
    )
    initialized = opts.work_dir / "initialized.complete"
    derivative_root = module_derivatives_root(
        "anat",
        opts.anat_id,
        project=opts.project,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        derivative_root = execution_context.output_path(derivative_root)

    def initialize_outputs() -> None:
        ensure_directory(derivative_root)
        ensure_directory(opts.out_dir)
        ensure_directory(opts.work_dir)
        ensure_directory(opts.freesurfer_subjects_dir)
        write_completion_breadcrumb(initialized, "Anatomical output initialized\n")

    runner.add_step(
        Step.python(
            name="Initialize Anatomical Outputs",
            outputs=(initialized,),
            action=initialize_outputs,
        )
    )

    configuration = {
        "selection_strategy": opts.selection_strategy,
        "gradient_unwarping": opts.gradient_unwarping,
        "fs_subject": opts.fs_subject,
        "mni_template": str(opts.mni_template),
        "synthstrip_image": str(opts.synthstrip_image),
    }
    configuration_snapshot = opts.work_dir / "configuration.json"

    def validate_configuration() -> tuple[bool, str]:
        try:
            current = read_json(configuration_snapshot)
        except (OSError, ValueError, TypeError):
            return False, "Anatomical configuration snapshot is missing or unreadable."
        # Only settings that govern the core anatomical steps belong to this
        # comparison. Other preprocessing settings must not invalidate them.
        effective = {key: current.get(key) for key in configuration}
        if effective != configuration:
            return False, "Anatomical configuration changed."
        return True, "Anatomical configuration is unchanged."

    runner.add_step(
        Step.python(
            name="Write Anatomical Configuration",
            outputs=(configuration_snapshot,),
            inputs=(initialized,),
            force=opts.overwrite,
            action=lambda: write_json(configuration_snapshot, configuration),
            validate=validate_configuration,
        )
    )

    dependency_check = opts.work_dir / "dependencies.complete"

    def check_dependencies() -> None:
        runner.require_cmds(
            [
                "mri_binarize",
                "mri_convert",
                "mri_vol2vol",
                "mri_robust_template",
                "N4BiasFieldCorrection",
                "recon-all",
                "mris_convert",
                "flirt",
                "fslmaths",
                "wb_command",
                "tkregister2",
                "convert_xfm",
                "antsRegistration",
                "antsApplyTransforms",
            ]
        )
        write_completion_breadcrumb(dependency_check, "Anatomical dependencies available\n")

    runner.add_step(
        Step.python(
            name="Check Anatomical Dependencies",
            outputs=(dependency_check,),
            action=check_dependencies,
            force=opts.overwrite,
        )
    )

    session_plans = _plan_session_anatomicals(
        images=all_images,
        project=opts.project,
        anat_id=opts.anat_id,
        sub_id=inputs.sub_id,
        execution_context=execution_context,
    )
    for plan in session_plans:
        runner.add_step(
            create_copy_file_step(
                src=plan.source.image,
                dst=plan.staged_raw,
                force=opts.overwrite,
                step_name="Stage Session Anatomical Image",
            )
        )
        gradient_resolution = gradient_resolutions[plan.source.image]
        n4_input = plan.staged_raw
        if gradient_resolution.applied:
            gradient_dir = plan.staged_raw.parent / "gradient_unwarping"
            n4_input = gradient_dir / plan.staged_raw.name
            gradient_warp = gradient_dir / (
                plan.staged_raw.name.removesuffix(".nii.gz").removesuffix(".nii") + "_warp.nii.gz"
            )
            gradient_metadata = gradient_dir / (
                plan.staged_raw.name.removesuffix(".nii.gz").removesuffix(".nii") + ".json"
            )
            runner.add_step(
                create_gradient_unwarping_step(
                    runner=runner,
                    source=plan.staged_raw,
                    corrected=n4_input,
                    warp=gradient_warp,
                    metadata=gradient_metadata,
                    resolution=gradient_resolution,
                    runtime=opts.gradient_unwarp_runtime,
                    image=opts.gradient_unwarp_image,
                    force=opts.overwrite,
                )
            )
        plan.metadata["GradientDistortionCorrection"] = gradient_resolution.scientific_record()
        runner.add_step(
            create_n4_bias_correction_step(
                env=env,
                in_img=n4_input,
                out_img=plan.staged_preprocessed,
                force=opts.overwrite,
                validate_gzip=True,
                step_name="N4 Bias Field Correction",
            )
        )
    for plan in session_plans:
        runner.add_step(
            _brain_extract_anat_copy(
                opts.synthstrip_image,
                env=env,
                source=plan.staged_preprocessed,
                dst=plan.output,
                mask=plan.mask,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            create_json_step(
                step_name="Write Session Anatomical Metadata",
                path=plan.metadata_output,
                payload=plan.metadata,
                inputs=(plan.staged_preprocessed, plan.output, plan.mask),
                force=opts.overwrite,
            )
        )
    copied_images = [
        AnatImage(
            image=plan.output,
            json=plan.metadata_output,
            modality=plan.source.modality,
            entities=dict(plan.source.entities),
            session_id=plan.source.session_id,
            time_kind=plan.source.time_kind,
            time_value=plan.source.time_value,
        )
        for plan in session_plans
    ]
    copied_session_files = [str(plan.output) for plan in session_plans]
    copied_t1 = [item for item in copied_images if item.modality == "T1w"]
    copied_t2 = [item for item in copied_images if item.modality == "T2w"]
    subj_t1_final: Optional[Path] = None
    subj_t2_final: Optional[Path] = None
    if copied_t1:
        subj_t1_final = _subject_preproc_path(
            images=copied_t1,
            modality="T1w",
            out_dir=opts.out_dir,
        )
    if copied_t2:
        subj_t2_final = _subject_preproc_path(
            images=copied_t2,
            modality="T2w",
            out_dir=opts.out_dir,
        )
    subj_t1 = subj_t1_final
    t1_meta: dict[str, object] = {
        "modality": "T1w",
        "sources": [],
        "strategy": opts.selection_strategy,
    }
    if subj_t1 is not None:
        t1_step, t1_meta = _create_copy_or_average_step(
            env=env,
            images=copied_t1,
            modality="T1w",
            strategy=opts.selection_strategy,
            out_img=subj_t1,
            work_dir=opts.work_dir,
            force=opts.overwrite,
        )
        runner.add_step(t1_step)
    # Select each modality independently. If both references exist, retain the
    # selected T2w privately until it has been resampled once onto the final
    # participant T1w grid.
    subj_t2_selected = subj_t2_final
    if subj_t1_final is not None and subj_t2_final is not None:
        subj_t2_selected = (
            opts.work_dir / "subject_reference" / f"{inputs.sub_id}_desc-selected_T2w.nii.gz"
        )
    t2_meta: dict[str, object] = {
        "modality": "T2w",
        "sources": [],
        "strategy": opts.selection_strategy,
    }
    if subj_t2_selected is not None:
        t2_step, t2_meta = _create_copy_or_average_step(
            env=env,
            images=copied_t2,
            modality="T2w",
            strategy=opts.selection_strategy,
            out_img=subj_t2_selected,
            work_dir=opts.work_dir,
            force=opts.overwrite,
        )
        runner.add_step(t2_step)
    subj_t2 = subj_t2_final
    t2w_to_t1w_xfms: dict[str, str] = {}
    t2w_to_t1w: Optional[Path] = None
    if subj_t1 is not None and subj_t2 is not None:
        assert subj_t2_selected is not None
        t2w_to_t1w = opts.out_dir / f"{inputs.sub_id}_from-T2w_to-T1w_mode-image_xfm.mat"
        runner.add_step(
            _register_t2_to_t1(
                env=env,
                t2_src=subj_t2_selected,
                t1_ref=subj_t1,
                out_t2=subj_t2,
                out_mat=t2w_to_t1w,
                force=opts.overwrite,
            )
        )
        t2w_to_t1w_xfms["t2w_to_t1w"] = str(t2w_to_t1w)
    if subj_t1 is not None:
        runner.add_step(
            create_json_step(
                step_name="Write Subject T1w Metadata",
                path=subj_t1.with_suffix("").with_suffix(".json"),
                payload={
                    "Sources": t1_meta["sources"],
                    "SelectionStrategy": opts.selection_strategy,
                    "BiasCorrection": "N4BiasFieldCorrection",
                },
                inputs=(subj_t1,),
                force=opts.overwrite,
            )
        )
    if subj_t2 is not None:
        runner.add_step(
            create_json_step(
                step_name="Write Subject T2w Metadata",
                path=subj_t2.with_suffix("").with_suffix(".json"),
                payload={
                    "Sources": t2_meta["sources"],
                    "SelectionStrategy": opts.selection_strategy,
                    "BiasCorrection": "N4BiasFieldCorrection",
                    "SpatialReference": "T1w" if subj_t1 is not None else None,
                    "TransformToT1w": str(t2w_to_t1w) if t2w_to_t1w is not None else None,
                },
                inputs=tuple(path for path in (subj_t2, t2w_to_t1w) if path is not None),
                force=opts.overwrite,
            )
        )
    myelin_map: Optional[Path] = None
    if subj_t1 is not None and subj_t2 is not None:
        myelin_map = opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-myelinMap_T1w.nii.gz"
        t2_nonzero = opts.work_dir / "myelin_map" / f"{inputs.sub_id}_desc-T2wNonzero_T2w.nii.gz"
        runner.add_step(
            Step.command_step(
                ["fslmaths", str(subj_t2), "-thr", "0.001", str(t2_nonzero)],
                name="Prepare Nonzero T2w for Myelin Map",
                outputs=(t2_nonzero,),
                inputs=(subj_t2,),
                force=opts.overwrite,
                env=env,
                prepare=lambda: t2_nonzero.parent.mkdir(parents=True, exist_ok=True),
            )
        )
        runner.add_step(
            Step.command_step(
                ["fslmaths", str(subj_t1), "-div", str(t2_nonzero), str(myelin_map)],
                name="Compute T1w/T2w Myelin Map",
                outputs=(myelin_map,),
                inputs=(subj_t1, subj_t2, t2_nonzero),
                force=opts.overwrite,
                env=env,
            )
        )
        runner.add_step(
            _write_json_step(
                myelin_map.with_suffix("").with_suffix(".json"),
                {
                    "Type": "myelin map",
                    "Space": "T1w",
                    "Sources": [str(subj_t1), str(subj_t2)],
                    "Description": "T1w/T2w ratio using preprocessed T1w and T2w images already aligned in T1w geometry.",
                },
            )
        )

    subject_anat = subj_t1 or subj_t2
    if subject_anat is None:
        raise SystemExit("No subject-level anatomical image available after selection.")

    subject_dir = opts.freesurfer_subjects_dir / opts.fs_subject
    runner.add_step(
        _create_recon_all_step(
            run_child=runner.run_child,
            env=env,
            t1w=subj_t1,
            t2w=subj_t2,
            subjects_dir=opts.freesurfer_subjects_dir,
            fs_subject=opts.fs_subject,
            force=opts.overwrite,
        )
    )
    aseg_mgz = subject_dir / "mri" / "aseg.mgz"
    subcortical_masks: dict[str, str] = {}
    for structure, segmentations in _FREESURFER_SUBCORTICAL_SEGMENTATIONS.items():
        labels = _aseg_label_ids(segmentations)
        output = opts.out_dir / f"{inputs.sub_id}_desc-{structure}_mask.nii.gz"
        temporary = (
            opts.work_dir
            / "subcortical_masks"
            / (f"{inputs.sub_id}_desc-{structure}_mask_fs.nii.gz")
        )
        runner.add_step(
            _create_label_mask_step(
                source=aseg_mgz,
                output=temporary,
                labels=labels,
                name=f"Extract {structure} FreeSurfer Labels",
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _resample_mask_to_native(
                env=env,
                src=temporary,
                ref_image=subject_anat,
                dst=output,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _write_json_step(
                output.with_suffix("").with_suffix(".json"),
                {
                    "Type": "ROI mask",
                    "Structure": structure,
                    "Sources": [str(aseg_mgz), str(subject_anat)],
                    "FreeSurferLabels": labels,
                    "FreeSurferSegmentations": list(segmentations),
                    "ReferenceImage": str(subject_anat),
                },
            )
        )
        subcortical_masks[structure] = str(output)

    brain_mask = opts.out_dir / f"{inputs.sub_id}_desc-brain_mask.nii.gz"
    gray_mask = opts.out_dir / f"{inputs.sub_id}_desc-grayMatter_mask.nii.gz"
    ribbon_mask = opts.out_dir / f"{inputs.sub_id}_desc-ribbon_mask.nii.gz"
    core_work = opts.work_dir / "core_masks"
    brain_temporary = core_work / f"{inputs.sub_id}_desc-brain_mask_fs.nii.gz"
    gray_temporary = core_work / f"{inputs.sub_id}_desc-grayMatter_mask_fs.nii.gz"
    ribbon_temporary = core_work / f"{inputs.sub_id}_desc-ribbon_mask_fs.nii.gz"
    brainmask_mgz = subject_dir / "mri" / "brainmask.mgz"
    ribbon_mgz = subject_dir / "mri" / "ribbon.mgz"
    runner.add_step(
        _export_mgz(
            env=env,
            src=brainmask_mgz,
            dst=brain_temporary,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _resample_mask_to_native(
            env=env,
            src=brain_temporary,
            ref_image=subject_anat,
            dst=brain_mask,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _write_json_step(
            brain_mask.with_suffix("").with_suffix(".json"),
            {
                "Type": "brain mask",
                "Sources": [str(brainmask_mgz), str(subject_anat)],
                "ReferenceImage": str(subject_anat),
            },
        )
    )
    gray_labels = _aseg_label_ids(_FREESURFER_GRAY_MATTER_SEGMENTATIONS)
    runner.add_step(
        _create_label_mask_step(
            source=aseg_mgz,
            output=gray_temporary,
            labels=gray_labels,
            name="Extract Gray Matter FreeSurfer Labels",
            env=env,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _resample_mask_to_native(
            env=env,
            src=gray_temporary,
            ref_image=subject_anat,
            dst=gray_mask,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _write_json_step(
            gray_mask.with_suffix("").with_suffix(".json"),
            {
                "Type": "gray matter mask",
                "Sources": [str(aseg_mgz), str(subject_anat)],
                "FreeSurferLabels": gray_labels,
                "FreeSurferSegmentations": list(_FREESURFER_GRAY_MATTER_SEGMENTATIONS),
                "ReferenceImage": str(subject_anat),
            },
        )
    )
    runner.add_step(
        _create_ribbon_mask_step(
            source=ribbon_mgz,
            output=ribbon_temporary,
            env=env,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _resample_mask_to_native(
            env=env,
            src=ribbon_temporary,
            ref_image=subject_anat,
            dst=ribbon_mask,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _write_json_step(
            ribbon_mask.with_suffix("").with_suffix(".json"),
            {
                "Type": "cortical ribbon mask",
                "Sources": [str(ribbon_mgz), str(subject_anat)],
                "ReferenceImage": str(subject_anat),
            },
        )
    )
    core_masks = {
        "brain": str(brain_mask),
        "gray_matter": str(gray_mask),
        "cortical_ribbon": str(ribbon_mask),
    }
    subject_t1 = subj_t1 or subject_anat
    fsnative_ref = opts.freesurfer_subjects_dir / opts.fs_subject / "mri" / "T1.mgz"
    t1_to_fsnative = opts.out_dir / f"{opts.fs_subject}_from-T1w_to-fsnative_mode-image_xfm.txt"
    fsnative_to_t1 = opts.out_dir / f"{opts.fs_subject}_from-fsnative_to-T1w_mode-image_xfm.txt"
    t1_to_fsnative_reg = opts.out_dir / f"{opts.fs_subject}_from-T1w_to-fsnative_mode-image_xfm.dat"
    runner.add_step(
        _create_t1_to_fsnative_affine_step(
            subject_t1=subject_t1,
            fsnative_reference=fsnative_ref,
            output=t1_to_fsnative,
            registration_file=t1_to_fsnative_reg,
            env=env,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _create_inverse_affine_step(
            source=t1_to_fsnative,
            output=fsnative_to_t1,
            env=env,
            force=opts.overwrite,
        )
    )
    fsnative_xfms = {
        "t1_to_fsnative": str(t1_to_fsnative),
        "fsnative_to_t1": str(fsnative_to_t1),
    }
    for path, from_space, to_space in (
        (t1_to_fsnative, "T1w", "fsnative"),
        (fsnative_to_t1, "fsnative", "T1w"),
    ):
        runner.add_step(
            _write_json_step(
                path.with_suffix(".json"),
                {
                    "Type": "affine",
                    "Format": "FSL",
                    "Sources": [str(subject_t1), str(fsnative_ref)],
                    "SpatialReference": to_space,
                    "From": from_space,
                    "To": to_space,
                },
            )
        )

    surface_work_dir = opts.work_dir / "surface_export"
    surface_dir = opts.freesurfer_subjects_dir / opts.fs_subject / "surf"
    fs_t1_ref_nii = surface_work_dir / f"{opts.fs_subject}_fs_t1_ref.nii.gz"
    runner.add_step(
        _create_mri_conversion_step(
            source=fsnative_ref,
            output=fs_t1_ref_nii,
            env=env,
            force=opts.overwrite,
            name="Convert FreeSurfer T1 Reference to NIfTI",
        )
    )
    exported_surfaces: dict[str, str] = {}
    for hemi in ("lh", "rh"):
        hemi_label = _hemi_label(hemi)
        hemi_paths: dict[str, Path] = {}
        white_source = surface_dir / f"{hemi}.white"
        for source_name, bids_suffix in _surface_names().items():
            source = surface_dir / f"{hemi}.{source_name}"
            output_name = (
                f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_{bids_suffix}.surf.gii"
            )
            output = opts.out_dir / output_name
            is_sphere = source_name in {"sphere", "sphere.reg"}
            if is_sphere:
                runner.add_step(
                    _create_mri_conversion_step(
                        source=source,
                        output=output,
                        env=env,
                        force=opts.overwrite,
                        name="Convert FreeSurfer Sphere",
                        executable="mris_convert",
                    )
                )
            else:
                raw = surface_work_dir / f"scanner_{output_name}"
                affined = surface_work_dir / f"affined_{output_name}"
                runner.add_step(
                    _create_mri_conversion_step(
                        source=source,
                        output=raw,
                        env=env,
                        force=opts.overwrite,
                        name="Surface Conversion",
                        to_scanner=True,
                        executable="mris_convert",
                    )
                )
                runner.add_step(
                    _create_surface_affine_step(
                        source=raw,
                        affine=fsnative_to_t1,
                        source_volume=fs_t1_ref_nii,
                        target_volume=subject_t1,
                        output=affined,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                runner.add_step(
                    _strip_freesurfer_volgeom_metadata(
                        surf_in=affined,
                        surf_out=output,
                        force=opts.overwrite,
                    )
                )
            runner.add_step(
                _write_json_step(
                    output.with_suffix(".json"),
                    {
                        "Hemisphere": hemi_label,
                        "Space": "fsnative",
                        "AnatomicalStructurePrimary": "Cortex",
                        "SurfaceType": bids_suffix,
                        "Sources": [str(source)],
                        "SpatialReference": "fsnative" if is_sphere else "T1w",
                    },
                )
            )
            exported_surfaces[f"{hemi}.{source_name}"] = str(output)
            hemi_paths[source_name] = output

        for metric_name, (bids_suffix, metric_description) in _metric_names().items():
            metric_source = surface_dir / f"{hemi}.{metric_name}"
            metric_output = (
                opts.out_dir
                / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_{bids_suffix}.shape.gii"
            )
            runner.add_step(
                _create_metric_conversion_step(
                    metric=metric_source,
                    surface=white_source,
                    output=metric_output,
                    description=metric_description,
                    env=env,
                    force=opts.overwrite,
                )
            )
            runner.add_step(
                _write_json_step(
                    metric_output.with_suffix(".json"),
                    {
                        "Hemisphere": hemi_label,
                        "Space": "fsnative",
                        "AnatomicalStructurePrimary": "Cortex",
                        "MetricType": metric_description,
                        "Sources": [str(metric_source), str(white_source)],
                    },
                )
            )
            exported_surfaces[f"{hemi}.{metric_name}"] = str(metric_output)

        white = hemi_paths["white"]
        pial = hemi_paths["pial"]
        midthickness = (
            opts.out_dir
            / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_midthickness.surf.gii"
        )
        runner.add_step(
            _create_midthickness_step(
                white=white,
                pial=pial,
                output=midthickness,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _write_json_step(
                midthickness.with_suffix(".json"),
                {
                    "Hemisphere": hemi_label,
                    "Space": "fsnative",
                    "AnatomicalStructurePrimary": "Cortex",
                    "SurfaceType": "midthickness",
                    "Sources": [str(white), str(pial)],
                },
            )
        )
        exported_surfaces[f"{hemi}.midthickness"] = str(midthickness)

    fsaverage_space = opts.fsaverage_template
    fsaverage_xfms: dict[str, str] = {}
    for hemi in ("lh", "rh"):
        hemi_label = _hemi_label(hemi)
        subject_registration = surface_dir / f"{hemi}.sphere.reg"
        fsaverage_sphere = find_fsaverage_template_surface(
            template=fsaverage_space,
            hemi=hemi_label,
            surface="sphere",
        )
        forward = (
            opts.out_dir
            / f"{opts.fs_subject}_from-fsnative_to-{fsaverage_space}_hemi-{hemi_label}_mode-surface_xfm.surf.gii"
        )
        inverse = (
            opts.out_dir
            / f"{opts.fs_subject}_from-{fsaverage_space}_to-fsnative_hemi-{hemi_label}_mode-surface_xfm.surf.gii"
        )
        runner.add_step(
            _create_mri_conversion_step(
                source=subject_registration,
                output=forward,
                name=f"Export Hemisphere {hemi_label} fsnative-to-{fsaverage_space} Sphere",
                env=env,
                force=opts.overwrite,
                executable="mris_convert",
            )
        )
        runner.add_step(
            create_copy_file_step(
                src=fsaverage_sphere,
                dst=inverse,
                force=opts.overwrite,
                step_name=f"Export Hemisphere {hemi_label} {fsaverage_space}-to-fsnative Sphere",
            )
        )
        for path, from_space, to_space, sources in (
            (
                forward,
                "fsnative",
                fsaverage_space,
                (subject_registration, fsaverage_sphere),
            ),
            (
                inverse,
                fsaverage_space,
                "fsnative",
                (fsaverage_sphere, subject_registration),
            ),
        ):
            runner.add_step(
                _write_json_step(
                    path.with_suffix(".json"),
                    {
                        "Type": "surface",
                        "Format": "FreeSurferSphereRegistration",
                        "Hemisphere": hemi_label,
                        "From": from_space,
                        "To": to_space,
                        "Sources": [str(source) for source in sources],
                    },
                )
            )
        fsaverage_xfms[f"hemi-{hemi_label}_fsnative_to_{fsaverage_space}"] = str(forward)
        fsaverage_xfms[f"hemi-{hemi_label}_{fsaverage_space}_to_fsnative"] = str(inverse)

    t1_fsaverage_xfms: dict[str, str] = {}
    for hemi_label in ("L", "R"):
        fsnative_to_fsaverage = Path(
            fsaverage_xfms[f"hemi-{hemi_label}_fsnative_to_{fsaverage_space}"]
        )
        fsaverage_to_fsnative = Path(
            fsaverage_xfms[f"hemi-{hemi_label}_{fsaverage_space}_to_fsnative"]
        )
        forward = (
            opts.out_dir
            / f"{opts.fs_subject}_from-T1w_to-{fsaverage_space}_hemi-{hemi_label}_mode-image+surface_xfm.json"
        )
        inverse = (
            opts.out_dir
            / f"{opts.fs_subject}_from-{fsaverage_space}_to-T1w_hemi-{hemi_label}_mode-surface+image_xfm.json"
        )
        runner.add_step(
            create_json_step(
                step_name=(
                    f"Write Hemisphere {hemi_label} T1w-to-{fsaverage_space} Transform Metadata"
                ),
                path=forward,
                payload={
                    "Type": "chain",
                    "Format": "surface",
                    "Hemisphere": hemi_label,
                    "From": "T1w",
                    "To": fsaverage_space,
                    "Steps": [str(fsnative_to_fsaverage)],
                },
                inputs=(fsnative_to_fsaverage,),
                force=opts.overwrite,
            )
        )
        runner.add_step(
            create_json_step(
                step_name=(
                    f"Write Hemisphere {hemi_label} {fsaverage_space}-to-T1w Transform Metadata"
                ),
                path=inverse,
                payload={
                    "Type": "chain",
                    "Format": "surface",
                    "Hemisphere": hemi_label,
                    "From": fsaverage_space,
                    "To": "T1w",
                    "Steps": [str(fsaverage_to_fsnative)],
                },
                inputs=(fsaverage_to_fsnative,),
                force=opts.overwrite,
            )
        )
        t1_fsaverage_xfms[f"hemi-{hemi_label}_t1_to_{fsaverage_space}"] = str(forward)
        t1_fsaverage_xfms[f"hemi-{hemi_label}_{fsaverage_space}_to_t1"] = str(inverse)

    mni_brain_template = Path(
        str(opts.mni_template).replace("_T1w.nii.gz", "_desc-brain_T1w.nii.gz")
    )
    mni_brain_mask = Path(str(opts.mni_template).replace("_T1w.nii.gz", "_desc-brain_mask.nii.gz"))
    mni_prefix = opts.work_dir / "ants_mni_" / f"{opts.fs_subject}_"
    t1_to_mni = (
        opts.out_dir / f"{opts.fs_subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )
    mni_to_t1 = (
        opts.out_dir / f"{opts.fs_subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )
    produced_t1_to_mni = mni_prefix.parent / f"{mni_prefix.name}Composite.h5"
    produced_mni_to_t1 = mni_prefix.parent / f"{mni_prefix.name}InverseComposite.h5"
    mni_registration_inputs = (
        subject_t1,
        Path(core_masks["brain"]),
        opts.mni_template,
        mni_brain_template,
        mni_brain_mask,
    )
    runner.add_step(
        _create_mni_registration_step(
            subject_t1=subject_t1,
            subject_brain_mask=Path(core_masks["brain"]),
            mni_brain_template=mni_brain_template,
            mni_brain_mask=mni_brain_mask,
            mni_template=opts.mni_template,
            prefix=mni_prefix,
            produced_forward=produced_t1_to_mni,
            produced_inverse=produced_mni_to_t1,
            env=env,
            force=opts.overwrite,
        )
    )
    runner.add_step(
        _create_transform_finalization_step(
            produced_forward=produced_t1_to_mni,
            produced_inverse=produced_mni_to_t1,
            forward=t1_to_mni,
            inverse=mni_to_t1,
            force=opts.overwrite,
        )
    )
    mni_xfms = {"t1_to_mni": str(t1_to_mni), "mni_to_t1": str(mni_to_t1)}
    for path, from_space, to_space in (
        (t1_to_mni, "T1w", "MNI152NLin2009cAsym"),
        (mni_to_t1, "MNI152NLin2009cAsym", "T1w"),
    ):
        runner.add_step(
            _write_json_step(
                path.with_suffix(".json"),
                {
                    "Type": "composite",
                    "Format": "ANTs",
                    "Sources": [str(value) for value in mni_registration_inputs],
                    "From": from_space,
                    "To": to_space,
                },
            )
        )

    mni_in_t1 = opts.out_dir / f"{opts.fs_subject}_space-T1w_desc-mniToT1wQC.nii.gz"
    t1_in_mni = opts.out_dir / f"{opts.fs_subject}_space-MNI152NLin2009cAsym_desc-t1ToMNIQC.nii.gz"
    mni_qc_images: dict[str, str] = {}
    for (
        key,
        output,
        input_image,
        reference,
        transform,
        from_space,
        to_space,
        description,
    ) in (
        (
            "mni_to_t1w",
            mni_in_t1,
            opts.mni_template,
            subject_t1,
            mni_to_t1,
            "MNI152NLin2009cAsym",
            "T1w",
            "Saved inverse composite transform applied in a single step to project the MNI template into subject T1w space for temporary QC.",
        ),
        (
            "t1w_to_mni",
            t1_in_mni,
            subject_t1,
            opts.mni_template,
            t1_to_mni,
            "T1w",
            "MNI152NLin2009cAsym",
            "Saved forward composite transform applied in a single step to project the preprocessed subject T1w image into MNI space for temporary QC.",
        ),
    ):
        runner.add_step(
            _create_mni_qc_image_step(
                input_image=input_image,
                reference=reference,
                transform=transform,
                output=output,
                from_space=from_space,
                to_space=to_space,
                env=env,
                force=opts.overwrite,
            )
        )
        runner.add_step(
            _write_json_step(
                output.with_suffix("").with_suffix(".json"),
                {
                    "Type": "temporary QC image",
                    "From": from_space,
                    "To": to_space,
                    "Transform": str(transform),
                    "Sources": [str(input_image), str(reference)],
                    "Description": description,
                },
            )
        )
        mni_qc_images[key] = str(output)
    manifest = {
        "subject": inputs.sub_id,
        "fs_subject": opts.fs_subject,
        "fsaverage_template": opts.fsaverage_template,
        "selection_strategy": opts.selection_strategy,
        "gradient_unwarping": {
            str(path): resolution.scientific_record()
            for path, resolution in gradient_resolutions.items()
        },
        "inputs": {
            "t1w": [str(item.image) for item in inputs.t1w],
            "t2w": [str(item.image) for item in inputs.t2w],
        },
        "copied_session_files": copied_session_files,
        "outputs": {
            "subject_t1w": (str(subj_t1) if subj_t1 is not None else None),
            "subject_t2w": (str(subj_t2) if subj_t2 is not None else None),
            "myelin_map": (str(myelin_map) if myelin_map is not None else None),
            "brain_image": str(subject_anat),
            "brain_mask": core_masks["brain"],
            "gray_matter_mask": core_masks["gray_matter"],
            "cortical_ribbon_mask": core_masks["cortical_ribbon"],
            "subcortical_masks": subcortical_masks,
            "surfaces": exported_surfaces,
            "mni_qc_images": mni_qc_images,
            "xfms": {
                **t2w_to_t1w_xfms,
                **fsnative_xfms,
                **fsaverage_xfms,
                **t1_fsaverage_xfms,
                **mni_xfms,
            },
        },
        "freesurfer_subjects_dir": str(opts.freesurfer_subjects_dir),
        "mni_template": str(opts.mni_template),
        "options": {
            "synthstrip_image": (
                str(opts.synthstrip_image) if opts.synthstrip_image is not None else None
            ),
            "configuration": {
                "selection_strategy": opts.selection_strategy,
                "gradient_unwarping": opts.gradient_unwarping,
                "fs_subject": opts.fs_subject,
                "fsaverage_template": opts.fsaverage_template,
                "mni_template": str(opts.mni_template),
                "synthstrip_image": (
                    str(opts.synthstrip_image) if opts.synthstrip_image is not None else None
                ),
            },
            "configuration_fingerprint": selected_configuration_fingerprint(),
        },
        "output_metadata_contract": anatomical_output_contract(),
        "complete": True,
    }
    manifest_path = anatomical_manifest_path(
        inputs.sub_id,
        project=opts.project,
        anat_id=opts.anat_id,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        manifest_path = execution_context.output_path(manifest_path)
    published_outputs = _nested_paths(manifest["outputs"]) + tuple(
        Path(path) for path in copied_session_files
    )

    def validate_publication() -> tuple[bool, str]:
        try:
            current = read_json(manifest_path)
            validate_anatomical_manifest(current)
        except (OSError, ValueError, TypeError):
            return False, "Anatomical publication manifest is missing or unreadable."
        if current != manifest:
            return False, "Anatomical publication manifest differs from the requested module."
        missing = [
            str(path)
            for path in published_outputs
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "Anatomical publication is missing outputs: " + ", ".join(missing)
        return True, "Anatomical publication is complete and current."

    def publish_manifest() -> None:
        validate_anatomical_manifest(manifest)
        write_json(manifest_path, manifest)

    runner.add_step(
        Step.python(
            name="Write Anatomical Publication Manifest",
            outputs=(manifest_path,),
            inputs=(*published_outputs, *public_inputs),
            force=opts.overwrite,
            action=publish_manifest,
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
    cfg = SETTINGS.anat
    p = argparse.ArgumentParser(prog="nro.modules.anat.module")
    p.add_argument("--project", default=SETTINGS.common.project)
    p.add_argument("--anat-id", default=SETTINGS.common.anat_id)
    p.add_argument("--sub-id", required=True)
    p.add_argument("--fs-subject", default=cfg.fs_subject)
    p.add_argument("--fsaverage-template", default=cfg.fsaverage_template)
    p.add_argument("--t1w", action="append", default=[], type=Path)
    p.add_argument("--t2w", action="append", default=[], type=Path)
    p.add_argument(
        "--selection-strategy", choices=["first", "robust_average"], default=cfg.selection_strategy
    )
    p.add_argument(
        "--gradient-unwarping",
        choices=["auto", "off"],
        default=cfg.gradient_unwarping,
    )
    p.add_argument("--mni-template", type=Path, default=cfg.mni_template)
    p.add_argument("--synthstrip-container", type=Path, default=cfg.synthstrip_container)
    p.add_argument("--freesurfer-subjects-dir", type=Path, default=cfg.freesurfer_subjects_dir)
    p.add_argument("--out-dir", type=Path, default=cfg.out_dir)
    p.add_argument("--work-dir", type=Path, default=cfg.work_dir)
    p.add_argument("--overwrite", action="store_true", default=cfg.overwrite)
    p.add_argument("--verbose", action="store_true", default=cfg.verbose)
    add_container_arguments(p, cfg.container)
    return p


def main(
    argv: Optional[Sequence[str]] = None, *, execution_context: ExecutionContext | None = None
) -> None:
    """Parse and execute one anatomical module work item."""
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project = str(args.project)
    t1w = [load_anat_image(resolve_project_path(path, project=project)) for path in args.t1w]
    t2w = [load_anat_image(resolve_project_path(path, project=project)) for path in args.t2w]
    out_dir = resolve_project_path(
        args.out_dir
        or anat_subject_dir(str(args.sub_id), project=project, anat_id=str(args.anat_id)),
        project=project,
    )
    work_dir = resolve_project_work_path(
        args.work_dir
        or (anat_subject_work_dir(str(args.sub_id), project=project, anat_id=str(args.anat_id))),
        project=project,
    )
    subjects_dir = resolve_project_path(
        args.freesurfer_subjects_dir
        or (
            module_derivatives_root("anat", str(args.anat_id), project=project)
            / "code"
            / "freesurfer"
        ),
        project=project,
    )
    mni_template = resolve_project_path(args.mni_template, project=project)
    if mni_template is None:
        raise SystemExit("Missing --mni-template path.")
    require_nonempty_file(mni_template, "MNI template")
    synthstrip_image = resolve_project_path(args.synthstrip_container, project=project)
    if synthstrip_image is None:
        raise SystemExit("Missing --synthstrip-container path.")
    site, _ = site_settings()
    container = build_container(
        ContainerSettings.from_args(args),
        work_directory=work_dir,
        execution_context=execution_context,
    )
    run(
        Inputs(sub_id=str(args.sub_id), t1w=tuple(t1w), t2w=tuple(t2w)),
        Options(
            project=project,
            anat_id=str(args.anat_id),
            out_dir=out_dir,
            work_dir=work_dir,
            freesurfer_subjects_dir=subjects_dir,
            fs_subject=str(args.fs_subject or args.sub_id),
            fsaverage_template=str(args.fsaverage_template),
            selection_strategy=str(args.selection_strategy),
            gradient_unwarping=str(args.gradient_unwarping),
            gradient_unwarp_image=Path(site["gradient_unwarp"]),
            gradient_unwarp_runtime=str(site["runtime"]),
            mni_template=mni_template,
            container=container,
            synthstrip_image=synthstrip_image,
            overwrite=bool(args.overwrite),
        ),
        execution_context=execution_context,
    )


if __name__ == "__main__":
    main()
