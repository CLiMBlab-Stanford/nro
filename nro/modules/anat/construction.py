"""Declare the anatomical graph's stages as immutable step plans.

Each planner returns the steps of one stage and the products that later stages
read. `build_module` adds the steps to its runner in the order the planners
return them.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nro.definitions.hardware import GradientUnwarpingResolution
from nro.engine.execution import (
    create_configuration_snapshot_step,
    create_copy_file_step,
    ensure_directory,
)
from nro.engine.gradient_unwarping import create_gradient_unwarping_step
from nro.engine.io import read_public_json, require_nonempty_file, write_public_json
from nro.engine.manifests import create_json_step
from nro.engine.neuroimaging import create_n4_bias_correction_step
from nro.engine.paths import anatomical_manifest_path
from nro.engine.source_metadata import semantic_metadata_values
from nro.engine.templates import find_fsaverage_template_surface
from nro.modules.anat.contract import anatomical_output_contract, validate_anatomical_manifest
from nro.modules.anat.inputs import AnatImage
from nro.modules.anat.layout import AnatLayout
from nro.orchestration.runner import write_completion_breadcrumb
from nro.orchestration.runner_graph import StagePlan, Step
from nro.orchestration.runtime import selected_configuration_fingerprint

from .constants import _FREESURFER_GRAY_MATTER_SEGMENTATIONS, _FREESURFER_SUBCORTICAL_SEGMENTATIONS
from .lesion_policy import (
    BOUNDARY_MARGIN_MM,
    MASKER_MODEL,
    MASKER_REVISION,
    PROBABILITY_THRESHOLD,
    TEST_TIME_AUGMENTATION,
    synthsr_scaffold_contract,
)
from .lesions import (
    create_cut_surfaces_step,
    create_lesion_excluded_mask_step,
    create_lesion_mask_step,
    create_lesion_qc_step,
    create_lesion_reconstruction_summary_step,
    create_synthetic_scaffold_metadata_step,
    create_synthsr_scaffold_plan,
)
from .policy import (
    bias_correction_contract,
    surface_reconstruction_contract,
    t1w_synthesis_contract,
)
from .reconstruction import (
    SurfaceReconstructionInputs,
    SurfaceReconstructionResources,
    create_surface_reconstruction_plan,
)
from .steps import (
    _aseg_label_ids,
    _brain_extract_anat_copy,
    _create_apply_brain_mask_step,
    _create_copy_or_average_step,
    _create_finalize_anatomy_step,
    _create_inverse_affine_step,
    _create_label_mask_step,
    _create_metric_conversion_step,
    _create_metric_structure_step,
    _create_midthickness_step,
    _create_mni_qc_image_step,
    _create_mni_registration_step,
    _create_mri_conversion_step,
    _create_pose_grid_step,
    _create_pose_inverse_step,
    _create_pose_qc_step,
    _create_pose_registration_step,
    _create_pose_resampling_step,
    _create_ribbon_mask_step,
    _create_surface_affine_step,
    _create_t1_to_fsnative_affine_step,
    _create_transform_finalization_step,
    _export_mgz,
    _hemi_label,
    _metric_names,
    _nested_paths,
    _register_t2_to_t1,
    _resample_mask_to_native,
    _strip_freesurfer_volgeom_metadata,
    _surface_names,
    _write_json_step,
)
from .t1w_synthesis import create_t1w_support_mask_step, create_t1w_synthesis_step

if TYPE_CHECKING:
    from nro.modules.anat.module import Options


def plan_initialization(
    *,
    opts: Options,
    layout: AnatLayout,
    synthetic_t1w: bool,
    derivative_root: Path,
    require_commands: Callable[[Sequence[str]], None],
) -> StagePlan[None]:
    """Declare output initialization, configuration capture, and dependency checks."""
    steps: list[Step] = []

    def initialize_outputs() -> None:
        ensure_directory(derivative_root)
        ensure_directory(opts.out_dir)
        ensure_directory(opts.work_dir)
        ensure_directory(opts.freesurfer_subjects_dir)
        write_completion_breadcrumb(layout.initialized, "Anatomical output initialized\n")

    steps.append(
        Step.python(
            name="Initialize Anatomical Outputs",
            outputs=(layout.initialized,),
            action=initialize_outputs,
        )
    )

    configuration = {
        "selection_strategy": opts.selection_strategy,
        "surface_reconstruction_engine": opts.surface_reconstruction_engine,
        "gradient_unwarping": opts.gradient_unwarping,
        "fs_subject": opts.fs_subject,
        "mni_template": str(opts.mni_template),
        "synthstrip_image": str(opts.synthstrip_image),
    }
    if synthetic_t1w:
        configuration["t1w_fallback"] = opts.t1w_fallback
    if opts.lesion:
        configuration["lesion"] = {
            "enabled": opts.lesion,
            "masker_model": MASKER_MODEL,
            "masker_revision": MASKER_REVISION,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "test_time_augmentation": TEST_TIME_AUGMENTATION,
            "scaffold_synthesis": synthsr_scaffold_contract(),
            "boundary_margin_mm": BOUNDARY_MARGIN_MM,
        }

    def effective_snapshot(recorded: dict) -> dict:
        # Snapshots created before surface backends were configurable used
        # FreeSurfer implicitly. Preserve that meaning without rewriting the
        # private record.
        recorded.setdefault("surface_reconstruction_engine", "freesurfer")
        # Only settings that govern the core anatomical steps belong to this
        # comparison. Other preprocessing settings must not invalidate them.
        return {key: recorded.get(key) for key in configuration}

    steps.append(
        create_configuration_snapshot_step(
            layout.configuration_snapshot,
            configuration,
            name="Write Anatomical Configuration",
            label="Anatomical configuration",
            inputs=(layout.initialized,),
            force=opts.overwrite,
            normalize=effective_snapshot,
        )
    )

    def check_dependencies() -> None:
        commands = [
            "mri_binarize",
            "mri_convert",
            "mri_vol2vol",
            "mri_robust_template",
            "N4BiasFieldCorrection",
            "mris_convert",
            "flirt",
            "fslmaths",
            "wb_command",
            "tkregister2",
            "convert_xfm",
            "antsRegistration",
            "antsApplyTransforms",
        ]
        require_commands(commands)
        write_completion_breadcrumb(layout.dependency_check, "Anatomical dependencies available\n")

    steps.append(
        Step.python(
            name="Check Anatomical Dependencies",
            outputs=(layout.dependency_check,),
            action=check_dependencies,
            force=opts.overwrite,
        )
    )

    return StagePlan(steps=tuple(steps), products=None)


@dataclass(frozen=True)
class SessionStagingProducts:
    """Per-acquisition inputs for subject-level selection."""

    gradient_corrected_sources: dict[Path, Path]
    """Map each source acquisition to its staged, gradient-corrected image."""


def plan_session_staging(
    *,
    opts: Options,
    session_plans: Sequence,
    gradient_resolutions: Mapping[Path, GradientUnwarpingResolution],
    env: dict[str, str],
    run_child: Callable,
) -> StagePlan[SessionStagingProducts]:
    """Declare staging, gradient unwarping, bias correction, and metadata per session.

    Each plan's `metadata` mapping is updated in place with the gradient and
    bias-correction records that its metadata step writes.
    """
    steps: list[Step] = []
    gradient_corrected_sources: dict[Path, Path] = {}
    for plan in session_plans:
        steps.append(
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
            steps.append(
                create_gradient_unwarping_step(
                    run_child=run_child,
                    source=plan.staged_raw,
                    corrected=n4_input,
                    warp=gradient_warp,
                    metadata=gradient_metadata,
                    resolution=gradient_resolution,
                    runtime=opts.container_runtime,
                    image=opts.gradient_unwarp_image,
                    force=opts.overwrite,
                )
            )
        gradient_corrected_sources[plan.source.image] = n4_input
        plan.metadata["GradientDistortionCorrection"] = gradient_resolution.scientific_record()
        preliminary_brain = plan.staged_preprocessed.with_name(
            plan.staged_preprocessed.name.replace("_desc-preproc_", "_desc-preN4Brain_")
        )
        bias_field = plan.staged_preprocessed.with_name(
            plan.staged_preprocessed.name.replace("_desc-preproc_", "_desc-biasField_")
        )
        steps.append(
            _brain_extract_anat_copy(
                opts.container_runtime,
                opts.synthstrip_image,
                env=env,
                source=n4_input,
                dst=preliminary_brain,
                mask=plan.mask,
                force=opts.overwrite,
            )
        )
        steps.append(
            create_n4_bias_correction_step(
                env=env,
                in_img=n4_input,
                out_img=plan.staged_preprocessed,
                mask=plan.mask,
                bias_field=bias_field,
                force=opts.overwrite,
                validate_gzip=True,
                step_name="N4 Bias Field Correction",
            )
        )
        plan.metadata["BiasCorrectionMask"] = str(plan.mask)
    for plan in session_plans:
        steps.append(
            _create_apply_brain_mask_step(
                source=plan.staged_preprocessed,
                mask=plan.mask,
                output=plan.output,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            create_json_step(
                step_name="Write Session Anatomical Metadata",
                path=plan.metadata_output,
                payload=plan.metadata,
                inputs=(plan.staged_preprocessed, plan.output, plan.mask),
                force=opts.overwrite,
                identity_payload={
                    **semantic_metadata_values(plan.metadata, module="anat"),
                    **{
                        key: plan.metadata[key]
                        for key in (
                            "Sources",
                            "BiasCorrection",
                            "GradientDistortionCorrection",
                            "BiasCorrectionMask",
                            "BrainExtraction",
                            "BrainMask",
                        )
                        if key in plan.metadata
                    },
                },
            )
        )
    return StagePlan(
        steps=tuple(steps),
        products=SessionStagingProducts(gradient_corrected_sources=gradient_corrected_sources),
    )


@dataclass(frozen=True)
class SubjectSelectionProducts:
    """Subject-level references and the source for pose normalization."""

    copied_session_files: list[str]
    lesion_raw_t1_selected: Path | None
    subj_t1_selected: Path | None
    subj_t2_selected: Path | None
    t1_meta: dict[str, object]
    t2_meta: dict[str, object]
    pose_source: Path
    pose_modality: str
    pose_mask: Path


def plan_subject_selection(
    *,
    opts: Options,
    inputs,
    layout: AnatLayout,
    session_plans: Sequence,
    gradient_corrected_sources: Mapping[Path, Path],
    synthetic_t1w: bool,
    env: dict[str, str],
    run_child: Callable,
) -> StagePlan[SubjectSelectionProducts]:
    """Declare subject-level T1w and T2w references and the preliminary pose mask.

    Session copies are combined by the selection strategy. Without an acquired
    T1w, a T1w reference is synthesized from the T2w reference. Raises
    SystemExit when no reference image is available.
    """
    steps: list[Step] = []
    synthetic_t1w_mask: Optional[Path] = None

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
    gradient_corrected_t1 = [
        AnatImage(
            image=gradient_corrected_sources[plan.source.image],
            json=plan.source.json,
            modality="T1w",
            entities=dict(plan.source.entities),
            session_id=plan.source.session_id,
            time_kind=plan.source.time_kind,
            time_value=plan.source.time_value,
        )
        for plan in session_plans
        if plan.source.modality == "T1w"
    ]
    subj_t1_selected = (
        layout.reference_work / f"{inputs.sub_id}_desc-selected_T1w.nii.gz" if copied_t1 else None
    )
    subj_t2_selected = (
        layout.reference_work / f"{inputs.sub_id}_desc-selected_T2w.nii.gz" if copied_t2 else None
    )
    t1_meta: dict[str, object] = {
        "modality": "T1w",
        "sources": [],
        "strategy": opts.selection_strategy,
    }
    lesion_bias_corrected_t1_selected: Optional[Path] = None
    lesion_raw_t1_selected: Optional[Path] = None
    lesion_selected_t1_mask: Optional[Path] = None
    if subj_t1_selected is not None:
        if opts.lesion:
            lesion_raw_t1_selected = (
                layout.reference_work / f"{inputs.sub_id}_desc-selectedRaw_T1w.nii.gz"
            )
            lesion_bias_corrected_t1_selected = (
                layout.reference_work / f"{inputs.sub_id}_desc-selectedBiasCorrected_T1w.nii.gz"
            )
            source_t1_step, t1_meta = _create_copy_or_average_step(
                env=env,
                images=gradient_corrected_t1,
                modality="T1w",
                strategy=opts.selection_strategy,
                out_img=lesion_raw_t1_selected,
                work_dir=opts.work_dir,
                force=opts.overwrite,
            )
            t1_meta["sources"] = [
                str(plan.source.image) for plan in session_plans if plan.source.modality == "T1w"
            ]
            steps.append(source_t1_step)
            lesion_selected_t1_mask = (
                layout.reference_work / f"{inputs.sub_id}_desc-selectedT1w_mask.nii.gz"
            )
            lesion_preliminary_brain = (
                layout.reference_work / f"{inputs.sub_id}_desc-selectedPreN4Brain_T1w.nii.gz"
            )
            lesion_bias_field = (
                layout.reference_work / f"{inputs.sub_id}_desc-selectedBiasField_T1w.nii.gz"
            )
            steps.append(
                _brain_extract_anat_copy(
                    opts.container_runtime,
                    opts.synthstrip_image,
                    env=env,
                    source=lesion_raw_t1_selected,
                    dst=lesion_preliminary_brain,
                    mask=lesion_selected_t1_mask,
                    force=opts.overwrite,
                )
            )
            steps.append(
                create_n4_bias_correction_step(
                    env=env,
                    in_img=lesion_raw_t1_selected,
                    out_img=lesion_bias_corrected_t1_selected,
                    mask=lesion_selected_t1_mask,
                    bias_field=lesion_bias_field,
                    force=opts.overwrite,
                    validate_gzip=True,
                    step_name="Bias Correct Selected Source T1w",
                )
            )
            steps.append(
                _create_apply_brain_mask_step(
                    source=lesion_bias_corrected_t1_selected,
                    mask=lesion_selected_t1_mask,
                    output=subj_t1_selected,
                    env=env,
                    force=opts.overwrite,
                )
            )
        else:
            t1_step, t1_meta = _create_copy_or_average_step(
                env=env,
                images=copied_t1,
                modality="T1w",
                strategy=opts.selection_strategy,
                out_img=subj_t1_selected,
                work_dir=opts.work_dir,
                force=opts.overwrite,
            )
            steps.append(t1_step)
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
        steps.append(t2_step)

    if synthetic_t1w:
        assert subj_t2_selected is not None
        assert opts.freesurfer_image is not None
        fs_license = Path(env["FS_LICENSE"])
        require_nonempty_file(fs_license, "FreeSurfer license")
        subj_t1_selected = layout.reference_work / f"{inputs.sub_id}_desc-synthetic_T1w.nii.gz"
        steps.append(
            create_t1w_synthesis_step(
                run_child=run_child,
                runtime=opts.container_runtime,
                image=opts.freesurfer_image,
                license_file=fs_license,
                t2w=subj_t2_selected,
                output=subj_t1_selected,
                threads=max(1, int(env.get("OMP_NUM_THREADS", "1"))),
                env=env,
                force=opts.overwrite,
            )
        )
        synthetic_t1w_mask = layout.reference_work / f"{inputs.sub_id}_desc-selectedT1w_mask.nii.gz"
        steps.append(
            create_t1w_support_mask_step(
                t2w=subj_t2_selected,
                synthetic_t1w=subj_t1_selected,
                output=synthetic_t1w_mask,
                force=opts.overwrite,
            )
        )
        t1_meta = {
            "modality": "T1w",
            "sources": list(t2_meta["sources"]),
            "strategy": "synthesize_from_t2w",
        }

    pose_source = lesion_bias_corrected_t1_selected or subj_t1_selected or subj_t2_selected
    if pose_source is None:
        raise SystemExit("No subject-level anatomical image available after selection.")
    pose_modality = "T1w" if subj_t1_selected is not None else "T2w"
    pose_mask = (
        lesion_selected_t1_mask
        or synthetic_t1w_mask
        or (layout.reference_work / f"{inputs.sub_id}_desc-selected{pose_modality}_mask.nii.gz")
    )
    if lesion_selected_t1_mask is None and synthetic_t1w_mask is None:
        steps.append(
            Step.command_step(
                ["fslmaths", str(pose_source), "-bin", str(pose_mask)],
                name="Construct Preliminary Anatomical Mask",
                outputs=(pose_mask,),
                inputs=(pose_source,),
                force=opts.overwrite,
                env=env,
            )
        )
    return StagePlan(
        steps=tuple(steps),
        products=SubjectSelectionProducts(
            copied_session_files=copied_session_files,
            lesion_raw_t1_selected=lesion_raw_t1_selected,
            subj_t1_selected=subj_t1_selected,
            subj_t2_selected=subj_t2_selected,
            t1_meta=t1_meta,
            t2_meta=t2_meta,
            pose_source=pose_source,
            pose_modality=pose_modality,
            pose_mask=pose_mask,
        ),
    )


@dataclass(frozen=True)
class PoseNormalizationProducts:
    """Values that later anatomical stages read from this stage."""

    pose_output: Path
    published_source_to_reference: Path
    reference_grid: Path
    reference_mask: Path
    reference_space: str
    reference_to_source: Path
    source_to_reference: Path
    subj_t1: Path | None
    subj_t2: Path | None


def plan_pose_normalization(
    *,
    env,
    inputs,
    layout,
    mni_brain_mask,
    mni_brain_template,
    opts,
    pose_mask,
    pose_modality,
    pose_source,
    subj_t1_selected,
    subj_t2_selected,
) -> StagePlan[PoseNormalizationProducts]:
    """Declare rigid pose normalization and the subject T1w and T2w references in that space."""
    steps: list[Step] = []
    reference_space = pose_modality
    pose_prefix = layout.pose_work / f"{inputs.sub_id}_"
    source_to_reference = layout.pose_work / f"{inputs.sub_id}_0GenericAffine.mat"
    reference_to_source = (
        opts.out_dir / f"{inputs.sub_id}_from-{reference_space}_to-orig_mode-image_xfm.mat"
    )
    published_source_to_reference = (
        opts.out_dir / f"{inputs.sub_id}_from-orig_to-{reference_space}_mode-image_xfm.mat"
    )
    reference_grid = layout.pose_work / f"{inputs.sub_id}_space-{reference_space}_reference.nii.gz"
    steps.append(
        _create_pose_registration_step(
            source=pose_source,
            source_mask=pose_mask,
            template=mni_brain_template,
            template_mask=mni_brain_mask,
            prefix=pose_prefix,
            transform=source_to_reference,
            env=env,
            force=opts.overwrite,
        )
    )
    steps.append(
        _create_pose_grid_step(
            source=pose_source,
            source_mask=pose_mask,
            template=opts.mni_template,
            transform=source_to_reference,
            output=reference_grid,
            force=opts.overwrite,
        )
    )
    steps.append(
        create_copy_file_step(
            src=source_to_reference,
            dst=published_source_to_reference,
            force=opts.overwrite,
            step_name="Publish Anatomical Pose Transform",
        )
    )
    steps.append(
        _create_pose_inverse_step(
            source=source_to_reference,
            output=reference_to_source,
            force=opts.overwrite,
        )
    )
    subj_t1 = (
        opts.out_dir / f"{inputs.sub_id}_desc-preproc_T1w.nii.gz"
        if subj_t1_selected is not None
        else None
    )
    subj_t2 = (
        opts.out_dir
        / (
            f"{inputs.sub_id}_space-T1w_desc-preproc_T2w.nii.gz"
            if subj_t1_selected is not None
            else f"{inputs.sub_id}_desc-preproc_T2w.nii.gz"
        )
        if subj_t2_selected is not None
        else None
    )
    pose_output = subj_t1 or subj_t2
    assert pose_output is not None
    pose_resampled = (
        layout.pose_work
        / f"{inputs.sub_id}_space-{reference_space}_desc-resampled_{pose_modality}.nii.gz"
    )
    reference_mask = (
        layout.pose_work / f"{inputs.sub_id}_space-{reference_space}_desc-brain_mask.nii.gz"
    )
    steps.append(
        _create_pose_resampling_step(
            source=pose_source,
            reference=reference_grid,
            transform=source_to_reference,
            output=pose_resampled,
            env=env,
            force=opts.overwrite,
        )
    )
    steps.append(
        _create_pose_resampling_step(
            source=pose_mask,
            reference=reference_grid,
            transform=source_to_reference,
            output=reference_mask,
            env=env,
            force=opts.overwrite,
            label=True,
        )
    )
    steps.append(
        _create_finalize_anatomy_step(
            source=pose_resampled,
            mask=reference_mask,
            output=pose_output,
            env=env,
            force=opts.overwrite,
        )
    )
    return StagePlan(
        steps=tuple(steps),
        products=PoseNormalizationProducts(
            pose_output=pose_output,
            published_source_to_reference=published_source_to_reference,
            reference_grid=reference_grid,
            reference_mask=reference_mask,
            reference_space=reference_space,
            reference_to_source=reference_to_source,
            source_to_reference=source_to_reference,
            subj_t1=subj_t1,
            subj_t2=subj_t2,
        ),
    )


@dataclass(frozen=True)
class ReferenceOutputProducts:
    """Values that later anatomical stages read from this stage."""

    lesion_synthsr_input: Optional[Path]
    myelin_map: Optional[Path]
    pose_qc: Path
    pose_xfms: dict[str, str]
    subject_anat: Path
    t2w_to_t1w_xfms: dict[str, str]


def plan_reference_outputs(
    *,
    env,
    inputs,
    layout,
    lesion_raw_t1_selected,
    opts,
    pose_mask,
    pose_output,
    pose_source,
    published_source_to_reference,
    reference_grid,
    reference_space,
    reference_to_source,
    source_to_reference,
    subj_t1,
    subj_t2,
    subj_t2_selected,
    synthetic_t1w,
    t1_meta,
    t2_meta,
) -> StagePlan[ReferenceOutputProducts]:
    """Declare pose quality control, T2w-to-T1w registration, and the myelin map."""
    steps: list[Step] = []
    lesion_synthsr_input: Optional[Path] = None
    if opts.lesion:
        assert lesion_raw_t1_selected is not None
        lesion_synthsr_input = (
            layout.reference_work / f"{inputs.sub_id}_desc-synthsrInput_T1w.nii.gz"
        )
        steps.append(
            _create_pose_resampling_step(
                source=lesion_raw_t1_selected,
                reference=reference_grid,
                transform=source_to_reference,
                output=lesion_synthsr_input,
                env=env,
                force=opts.overwrite,
            )
        )
    pose_qc = opts.out_dir / f"{inputs.sub_id}_space-{reference_space}_desc-poseQC_metrics.json"
    steps.append(
        _create_pose_qc_step(
            source=pose_source,
            source_mask=pose_mask,
            aligned=pose_output,
            template=opts.mni_template,
            transform=source_to_reference,
            grid=reference_grid,
            output=pose_qc,
            force=opts.overwrite,
        )
    )
    pose_xfms = {
        f"orig_to_{reference_space.lower()}": str(published_source_to_reference),
        f"{reference_space.lower()}_to_orig": str(reference_to_source),
    }
    t2w_to_t1w_xfms: dict[str, str] = {}
    t2w_to_t1w: Optional[Path] = None
    if subj_t1 is not None and subj_t2 is not None:
        assert subj_t2_selected is not None
        t2w_to_t1w = opts.out_dir / f"{inputs.sub_id}_from-T2w_to-T1w_mode-image_xfm.mat"
        t1w_to_t2w = opts.out_dir / f"{inputs.sub_id}_from-T1w_to-T2w_mode-image_xfm.mat"
        steps.append(
            _register_t2_to_t1(
                env=env,
                t2_src=subj_t2_selected,
                t1_ref=subj_t1,
                out_t2=subj_t2,
                out_mat=t2w_to_t1w,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_inverse_affine_step(
                source=t2w_to_t1w,
                output=t1w_to_t2w,
                env=env,
                force=opts.overwrite,
                name="Invert T2w-to-T1w Affine",
            )
        )
        t2w_to_t1w_xfms.update(
            t2w_to_t1w=str(t2w_to_t1w),
            t1w_to_t2w=str(t1w_to_t2w),
        )
    if subj_t1 is not None:
        steps.append(
            create_json_step(
                step_name="Write Subject T1w Metadata",
                path=subj_t1.with_suffix("").with_suffix(".json"),
                payload={
                    "Sources": t1_meta["sources"],
                    "SelectionStrategy": opts.selection_strategy,
                    "BiasCorrection": "N4BiasFieldCorrection",
                    "SpatialReference": "T1w",
                    "PoseTransform": str(published_source_to_reference),
                    "PoseQuality": str(pose_qc),
                    **(
                        {
                            "Synthetic": True,
                            "SourceContrast": "T2w",
                            "Synthesis": t1w_synthesis_contract(),
                        }
                        if synthetic_t1w
                        else {}
                    ),
                },
                inputs=(subj_t1, published_source_to_reference, pose_qc),
                force=opts.overwrite,
            )
        )
    if subj_t2 is not None:
        steps.append(
            create_json_step(
                step_name="Write Subject T2w Metadata",
                path=subj_t2.with_suffix("").with_suffix(".json"),
                payload={
                    "Sources": t2_meta["sources"],
                    "SelectionStrategy": opts.selection_strategy,
                    "BiasCorrection": "N4BiasFieldCorrection",
                    "SpatialReference": "T1w",
                    "TransformToT1w": str(t2w_to_t1w)
                    if t2w_to_t1w is not None
                    else str(published_source_to_reference),
                    "PoseQuality": str(pose_qc),
                },
                inputs=tuple(path for path in (subj_t2, t2w_to_t1w, pose_qc) if path is not None),
                force=opts.overwrite,
            )
        )
    myelin_map: Optional[Path] = None
    if subj_t1 is not None and subj_t2 is not None and not synthetic_t1w:
        myelin_map = opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-myelinMap_T1w.nii.gz"
        t2_nonzero = opts.work_dir / "myelin_map" / f"{inputs.sub_id}_desc-T2wNonzero_T2w.nii.gz"
        steps.append(
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
        steps.append(
            Step.command_step(
                ["fslmaths", str(subj_t1), "-div", str(t2_nonzero), str(myelin_map)],
                name="Compute T1w/T2w Myelin Map",
                outputs=(myelin_map,),
                inputs=(subj_t1, subj_t2, t2_nonzero),
                force=opts.overwrite,
                env=env,
            )
        )
        steps.append(
            _write_json_step(
                myelin_map.with_suffix("").with_suffix(".json"),
                {
                    "Type": "myelin map",
                    "Space": "T1w",
                    "Sources": [str(subj_t1), str(subj_t2)],
                    "Description": "T1w/T2w ratio using preprocessed images aligned on the participant T1w grid.",
                },
            )
        )

    subject_anat = pose_output
    return StagePlan(
        steps=tuple(steps),
        products=ReferenceOutputProducts(
            lesion_synthsr_input=lesion_synthsr_input,
            myelin_map=myelin_map,
            pose_qc=pose_qc,
            pose_xfms=pose_xfms,
            subject_anat=subject_anat,
            t2w_to_t1w_xfms=t2w_to_t1w_xfms,
        ),
    )


@dataclass(frozen=True)
class LesionInputProducts:
    """Values that later anatomical stages read from this stage."""

    lesion_mask: Optional[Path]
    lesion_metadata: Optional[Path]
    lesion_probability: Optional[Path]
    lesion_qc: Optional[Path]
    lesion_reconstruction_summary: Optional[Path]
    reconstruction_mask: Path
    reconstruction_t1: Path | None
    reconstruction_t2: Path | None
    synthetic_scaffold_t1: Optional[Path]


def plan_lesion_reconstruction_inputs(
    *,
    env,
    inputs,
    lesion_masker_command,
    lesion_masker_module,
    lesion_raw_t1_selected,
    lesion_synthsr_input,
    opts,
    reference_mask,
    subj_t1,
    subj_t2,
    synthstroke_data,
    run_child: Callable,
) -> StagePlan[LesionInputProducts]:
    """Declare lesion masking and the images that surface reconstruction receives.

    Without lesion processing, reconstruction uses the subject T1w, T2w, and
    brain mask unchanged."""
    steps: list[Step] = []
    lesion_mask: Optional[Path] = None
    lesion_probability: Optional[Path] = None
    lesion_metadata: Optional[Path] = None
    lesion_qc: Optional[Path] = None
    synthetic_scaffold_t1: Optional[Path] = None
    lesion_reconstruction_summary: Optional[Path] = None
    reconstruction_t1 = subj_t1
    reconstruction_t2 = subj_t2
    reconstruction_mask = reference_mask
    if opts.lesion:
        assert subj_t1 is not None
        assert lesion_synthsr_input is not None
        assert lesion_masker_command is not None
        assert opts.freesurfer_image is not None
        lesion_mask = opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-lesion_mask.nii.gz"
        lesion_probability = (
            opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-lesion_probability.nii.gz"
        )
        lesion_metadata = lesion_mask.with_suffix("").with_suffix(".json")
        steps.append(
            create_lesion_mask_step(
                run_child=run_child,
                command=lesion_masker_command,
                module=lesion_masker_module,
                source=subj_t1,
                probability=lesion_probability,
                mask=lesion_mask,
                metadata=lesion_metadata,
                model_directory=synthstroke_data,
                model=MASKER_MODEL,
                revision=MASKER_REVISION,
                threshold=PROBABILITY_THRESHOLD,
                test_time_augmentation=TEST_TIME_AUGMENTATION,
                use_gpu=False,
                force=opts.overwrite,
            )
        )
        lesion_qc = opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-lesionQC.png"
        steps.append(
            create_lesion_qc_step(
                source=subj_t1,
                mask=lesion_mask,
                output=lesion_qc,
                force=opts.overwrite,
            )
        )
        scaffold = create_synthsr_scaffold_plan(
            run_child=run_child,
            runtime=opts.container_runtime,
            image=opts.freesurfer_image,
            license_file=Path(env["FS_LICENSE"]),
            t1w=lesion_synthsr_input,
            output=opts.work_dir / "lesion_synthsr" / f"{inputs.sub_id}_desc-synthsr_T1w.nii.gz",
            threads=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))),
            env=env,
            force=opts.overwrite,
        )
        steps.append(scaffold.step)
        reconstruction_t1 = scaffold.image
        reconstruction_t2 = None
        reconstruction_mask = reference_mask
        synthetic_scaffold_t1 = opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-synthsr_T1w.nii.gz"
        steps.append(
            create_copy_file_step(
                src=scaffold.image,
                dst=synthetic_scaffold_t1,
                force=opts.overwrite,
                step_name="Publish SynthSR Anatomical Alternative",
                parameters={
                    "method": synthsr_scaffold_contract(),
                    "synthetic_tissue": True,
                },
            )
        )
        steps.append(
            create_synthetic_scaffold_metadata_step(
                image=synthetic_scaffold_t1,
                source_t1w=lesion_raw_t1_selected,
                synthesis_input=lesion_synthsr_input,
                output=synthetic_scaffold_t1.with_suffix("").with_suffix(".json"),
                force=opts.overwrite,
            )
        )
    return StagePlan(
        steps=tuple(steps),
        products=LesionInputProducts(
            lesion_mask=lesion_mask,
            lesion_metadata=lesion_metadata,
            lesion_probability=lesion_probability,
            lesion_qc=lesion_qc,
            lesion_reconstruction_summary=lesion_reconstruction_summary,
            reconstruction_mask=reconstruction_mask,
            reconstruction_t1=reconstruction_t1,
            reconstruction_t2=reconstruction_t2,
            synthetic_scaffold_t1=synthetic_scaffold_t1,
        ),
    )


@dataclass(frozen=True)
class SurfaceReconstructionProducts:
    """Values that later anatomical stages read from this stage."""

    core_masks: dict[str, str]
    fsnative_xfms: dict[str, str]
    lesion_reconstruction_summary: Path | None
    subcortical_masks: dict[str, str]
    subject_t1: Path


def plan_surface_reconstruction(
    *,
    derivative_root,
    env,
    inputs,
    layout,
    lesion_mask,
    lesion_reconstruction_summary,
    opts,
    reconstruction_mask,
    reconstruction_t1,
    reconstruction_t2,
    subj_t1,
    subject_anat,
    synthetic_scaffold_t1,
    run_child: Callable,
) -> StagePlan[SurfaceReconstructionProducts]:
    """Declare surface reconstruction and the volumes and transforms derived from it.

    This covers subcortical masks, the brain, gray-matter, and ribbon masks in
    T1w space, and the T1w-to-fsnative transforms."""
    steps: list[Step] = []
    reconstruction = create_surface_reconstruction_plan(
        engine=opts.surface_reconstruction_engine,
        inputs=SurfaceReconstructionInputs(
            t1w=reconstruction_t1,
            t2w=reconstruction_t2,
            brain_mask=reconstruction_mask,
        ),
        resources=SurfaceReconstructionResources(
            runtime=opts.container_runtime,
            freesurfer_image=opts.freesurfer_image,
            fastsurfer_image=opts.fastsurfer_image,
            license_file=Path(env["FS_LICENSE"]),
            subjects_dir=opts.freesurfer_subjects_dir,
            template_root=(derivative_root.parents[1] / ".nro/templates/freesurfer"),
            staging_subjects_dir=opts.work_dir / "fastsurfer-segmentation",
            subject=opts.fs_subject,
            cpu_threads=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))),
        ),
        run_child=run_child,
        env=env,
        force=opts.overwrite,
    )
    steps.extend(reconstruction.steps)
    if reconstruction.products.subject_dir != layout.subject_dir:
        raise RuntimeError("Surface reconstruction returned an unexpected subject directory")
    if opts.lesion:
        assert lesion_mask is not None
        lesion_reconstruction_summary = (
            opts.out_dir / f"{inputs.sub_id}_desc-lesionReconstruction_summary.yaml"
        )
        steps.append(
            create_lesion_reconstruction_summary_step(
                subject_dir=layout.subject_dir,
                lesion_mask=lesion_mask,
                scaffold_image=synthetic_scaffold_t1,
                scaffold_synthesis=synthsr_scaffold_contract(),
                surface_reconstruction=surface_reconstruction_contract(
                    opts.surface_reconstruction_engine
                ),
                output=lesion_reconstruction_summary,
                force=opts.overwrite,
            )
        )
    subcortical_masks: dict[str, str] = {}
    for structure, segmentations in _FREESURFER_SUBCORTICAL_SEGMENTATIONS.items():
        labels = _aseg_label_ids(segmentations)
        output = opts.out_dir / f"{inputs.sub_id}_desc-{structure}_mask.nii.gz"
        temporary = (
            opts.work_dir
            / "subcortical_masks"
            / (f"{inputs.sub_id}_desc-{structure}_mask_fs.nii.gz")
        )
        resampled = (
            opts.work_dir
            / "subcortical_masks"
            / f"{inputs.sub_id}_desc-{structure}_mask_scaffold.nii.gz"
            if opts.lesion
            else output
        )
        steps.append(
            _create_label_mask_step(
                source=layout.aseg_mgz,
                output=temporary,
                labels=labels,
                name=f"Extract {structure} FreeSurfer Labels",
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _resample_mask_to_native(
                env=env,
                src=temporary,
                ref_image=subject_anat,
                dst=resampled,
                force=opts.overwrite,
            )
        )
        if opts.lesion:
            assert lesion_mask is not None
            steps.append(
                create_lesion_excluded_mask_step(
                    scaffold_mask=resampled,
                    lesion_mask=lesion_mask,
                    output=output,
                    force=opts.overwrite,
                )
            )
        steps.append(
            _write_json_step(
                output.with_suffix("").with_suffix(".json"),
                {
                    "Type": "ROI mask",
                    "Structure": structure,
                    "Sources": [
                        str(layout.aseg_mgz),
                        str(subject_anat),
                        *([str(lesion_mask)] if lesion_mask is not None else []),
                    ],
                    "FreeSurferLabels": labels,
                    "FreeSurferSegmentations": list(segmentations),
                    "ReferenceImage": str(subject_anat),
                },
            )
        )
        subcortical_masks[structure] = str(output)

    brain_resampled = (
        layout.core_work / f"{inputs.sub_id}_desc-brain_mask_scaffold.nii.gz"
        if opts.lesion
        else layout.brain_mask
    )
    gray_resampled = (
        layout.core_work / f"{inputs.sub_id}_desc-grayMatter_mask_scaffold.nii.gz"
        if opts.lesion
        else layout.gray_mask
    )
    ribbon_resampled = (
        layout.core_work / f"{inputs.sub_id}_desc-ribbon_mask_scaffold.nii.gz"
        if opts.lesion
        else layout.ribbon_mask
    )
    steps.append(
        _export_mgz(
            env=env,
            src=layout.brainmask_mgz,
            dst=layout.brain_temporary,
            force=opts.overwrite,
        )
    )
    steps.append(
        _resample_mask_to_native(
            env=env,
            src=layout.brain_temporary,
            ref_image=subject_anat,
            dst=brain_resampled,
            force=opts.overwrite,
        )
    )
    if opts.lesion:
        assert lesion_mask is not None
        steps.append(
            create_lesion_excluded_mask_step(
                scaffold_mask=brain_resampled,
                lesion_mask=lesion_mask,
                output=layout.brain_mask,
                force=opts.overwrite,
            )
        )
    steps.append(
        _write_json_step(
            layout.brain_mask.with_suffix("").with_suffix(".json"),
            {
                "Type": "brain mask",
                "Sources": [
                    str(layout.brainmask_mgz),
                    str(subject_anat),
                    *([str(lesion_mask)] if lesion_mask is not None else []),
                ],
                "ReferenceImage": str(subject_anat),
            },
        )
    )
    gray_labels = _aseg_label_ids(_FREESURFER_GRAY_MATTER_SEGMENTATIONS)
    steps.append(
        _create_label_mask_step(
            source=layout.aseg_mgz,
            output=layout.gray_temporary,
            labels=gray_labels,
            name="Extract Gray Matter FreeSurfer Labels",
            env=env,
            force=opts.overwrite,
        )
    )
    steps.append(
        _resample_mask_to_native(
            env=env,
            src=layout.gray_temporary,
            ref_image=subject_anat,
            dst=gray_resampled,
            force=opts.overwrite,
        )
    )
    if opts.lesion:
        assert lesion_mask is not None
        steps.append(
            create_lesion_excluded_mask_step(
                scaffold_mask=gray_resampled,
                lesion_mask=lesion_mask,
                output=layout.gray_mask,
                force=opts.overwrite,
            )
        )
    steps.append(
        _write_json_step(
            layout.gray_mask.with_suffix("").with_suffix(".json"),
            {
                "Type": "gray matter mask",
                "Sources": [
                    str(layout.aseg_mgz),
                    str(subject_anat),
                    *([str(lesion_mask)] if lesion_mask is not None else []),
                ],
                "FreeSurferLabels": gray_labels,
                "FreeSurferSegmentations": list(_FREESURFER_GRAY_MATTER_SEGMENTATIONS),
                "ReferenceImage": str(subject_anat),
            },
        )
    )
    steps.append(
        _create_ribbon_mask_step(
            source=layout.ribbon_mgz,
            output=layout.ribbon_temporary,
            env=env,
            force=opts.overwrite,
        )
    )
    steps.append(
        _resample_mask_to_native(
            env=env,
            src=layout.ribbon_temporary,
            ref_image=subject_anat,
            dst=ribbon_resampled,
            force=opts.overwrite,
        )
    )
    if opts.lesion:
        assert lesion_mask is not None
        steps.append(
            create_lesion_excluded_mask_step(
                scaffold_mask=ribbon_resampled,
                lesion_mask=lesion_mask,
                output=layout.ribbon_mask,
                force=opts.overwrite,
            )
        )
    steps.append(
        _write_json_step(
            layout.ribbon_mask.with_suffix("").with_suffix(".json"),
            {
                "Type": "cortical ribbon mask",
                "Sources": [
                    str(layout.ribbon_mgz),
                    str(subject_anat),
                    *([str(lesion_mask)] if lesion_mask is not None else []),
                ],
                "ReferenceImage": str(subject_anat),
            },
        )
    )
    core_masks = {
        "brain": str(layout.brain_mask),
        "gray_matter": str(layout.gray_mask),
        "cortical_ribbon": str(layout.ribbon_mask),
    }
    subject_t1 = subj_t1 or subject_anat
    steps.append(
        _create_t1_to_fsnative_affine_step(
            subject_t1=subject_t1,
            fsnative_reference=layout.fsnative_ref,
            output=layout.t1_to_fsnative,
            registration_file=layout.t1_to_fsnative_reg,
            env=env,
            force=opts.overwrite,
        )
    )
    steps.append(
        _create_inverse_affine_step(
            source=layout.t1_to_fsnative,
            output=layout.fsnative_to_t1,
            env=env,
            force=opts.overwrite,
        )
    )
    fsnative_xfms = {
        "t1w_to_fsnative": str(layout.t1_to_fsnative),
        "fsnative_to_t1w": str(layout.fsnative_to_t1),
    }
    for path, from_space, to_space in (
        (layout.t1_to_fsnative, "T1w", "fsnative"),
        (layout.fsnative_to_t1, "fsnative", "T1w"),
    ):
        steps.append(
            _write_json_step(
                path.with_suffix(".json"),
                {
                    "Type": "affine",
                    "Format": "FSL",
                    "Sources": [str(subject_t1), str(layout.fsnative_ref)],
                    "SpatialReference": to_space,
                    "From": from_space,
                    "To": to_space,
                },
            )
        )

    steps.append(
        _create_mri_conversion_step(
            source=layout.fsnative_ref,
            output=layout.fs_t1_ref_nii,
            env=env,
            force=opts.overwrite,
            name="Convert FreeSurfer T1 Reference to NIfTI",
        )
    )
    return StagePlan(
        steps=tuple(steps),
        products=SurfaceReconstructionProducts(
            core_masks=core_masks,
            fsnative_xfms=fsnative_xfms,
            lesion_reconstruction_summary=lesion_reconstruction_summary,
            subcortical_masks=subcortical_masks,
            subject_t1=subject_t1,
        ),
    )


@dataclass(frozen=True)
class SurfaceExportProducts:
    """Values that later anatomical stages read from this stage."""

    exported_surfaces: dict[str, str]
    intact_surfaces: dict[str, str]
    lesion_scaffold_anatomy: dict[str, dict[str, Path]]
    lesion_scaffold_metrics: dict[str, dict[Path, Path]]
    lesion_scaffold_surfaces: dict[str, dict[Path, Path]]


def plan_surface_export(
    *,
    env,
    layout,
    opts,
    subject_t1,
) -> StagePlan[SurfaceExportProducts]:
    """Declare GIFTI surface export for both hemispheres in T1w space.

    For lesion anatomy, surfaces reconstructed on the synthetic scaffold are
    exported separately so that later stages can cut the lesion from them."""
    steps: list[Step] = []
    exported_surfaces: dict[str, str] = {}
    intact_surfaces: dict[str, str] = {}
    lesion_scaffold_surfaces: dict[str, dict[Path, Path]] = {"lh": {}, "rh": {}}
    lesion_scaffold_metrics: dict[str, dict[Path, Path]] = {"lh": {}, "rh": {}}
    lesion_scaffold_anatomy: dict[str, dict[str, Path]] = {"lh": {}, "rh": {}}
    for hemi in ("lh", "rh"):
        hemi_label = _hemi_label(hemi)
        hemi_paths: dict[str, Path] = {}
        white_source = layout.surface_dir / f"{hemi}.white"
        for source_name, bids_suffix in _surface_names().items():
            source = layout.surface_dir / f"{hemi}.{source_name}"
            output_name = (
                f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_{bids_suffix}.surf.gii"
            )
            output = opts.out_dir / output_name
            generated_output = (
                layout.surface_work_dir / "lesion_scaffold" / output_name if opts.lesion else output
            )
            is_sphere = source_name in {"sphere", "sphere.reg"}
            if is_sphere:
                steps.append(
                    _create_mri_conversion_step(
                        source=source,
                        output=generated_output,
                        env=env,
                        force=opts.overwrite,
                        name="Convert FreeSurfer Sphere",
                        executable="mris_convert",
                    )
                )
            else:
                raw = layout.surface_work_dir / f"scanner_{output_name}"
                affined = layout.surface_work_dir / f"affined_{output_name}"
                steps.append(
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
                steps.append(
                    _create_surface_affine_step(
                        source=raw,
                        affine=layout.fsnative_to_t1,
                        source_volume=layout.fs_t1_ref_nii,
                        target_volume=subject_t1,
                        output=affined,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                steps.append(
                    _strip_freesurfer_volgeom_metadata(
                        surf_in=affined,
                        surf_out=generated_output,
                        force=opts.overwrite,
                    )
                )
            steps.append(
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
            if opts.lesion and source_name in {"white", "pial", "inflated"}:
                intact_output = opts.out_dir / output_name.replace(
                    f"_{bids_suffix}.surf.gii",
                    f"_desc-inpainted_{bids_suffix}.surf.gii",
                )
                steps.append(
                    create_copy_file_step(
                        src=generated_output,
                        dst=intact_output,
                        force=opts.overwrite,
                        step_name=f"Publish Intact Hemisphere {hemi_label} {bids_suffix} Surface",
                    )
                )
                steps.append(
                    _write_json_step(
                        intact_output.with_suffix(".json"),
                        {
                            "Hemisphere": hemi_label,
                            "Space": "fsnative",
                            "AnatomicalStructurePrimary": "Cortex",
                            "SurfaceType": bids_suffix,
                            "Sources": [str(source)],
                            "SpatialReference": "T1w",
                            "SyntheticTissue": True,
                            "Description": "Intact scaffold reconstructed from the synthetic SynthSR T1w image.",
                        },
                    )
                )
                intact_surfaces[f"{hemi}.{source_name}"] = str(intact_output)
            exported_surfaces[f"{hemi}.{source_name}"] = str(output)
            hemi_paths[source_name] = generated_output
            if opts.lesion:
                lesion_scaffold_surfaces[hemi][generated_output] = output
                lesion_scaffold_anatomy[hemi][source_name] = generated_output

        for metric_name, (bids_suffix, metric_description) in _metric_names().items():
            metric_source = layout.surface_dir / f"{hemi}.{metric_name}"
            metric_output = (
                opts.out_dir
                / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_{bids_suffix}.shape.gii"
            )
            generated_metric_output = (
                layout.surface_work_dir / "lesion_scaffold" / metric_output.name
                if opts.lesion
                else metric_output
            )
            converted_metric_output = (
                layout.surface_work_dir / "metric_conversion" / metric_output.name
            )
            metric_structure = f"Cortex{'Left' if hemi_label == 'L' else 'Right'}"
            steps.append(
                _create_metric_conversion_step(
                    metric=metric_source,
                    surface=white_source,
                    output=converted_metric_output,
                    description=metric_description,
                    env=env,
                    force=opts.overwrite,
                )
            )
            steps.append(
                _create_metric_structure_step(
                    source=converted_metric_output,
                    output=generated_metric_output,
                    structure=metric_structure,
                    description=metric_description,
                    force=opts.overwrite,
                )
            )
            steps.append(
                _write_json_step(
                    metric_output.with_suffix(".json"),
                    {
                        "Hemisphere": hemi_label,
                        "Space": "fsnative",
                        "AnatomicalStructurePrimary": metric_structure,
                        "MetricType": metric_description,
                        "Sources": [str(metric_source), str(white_source)],
                    },
                )
            )
            exported_surfaces[f"{hemi}.{metric_name}"] = str(metric_output)
            if opts.lesion:
                lesion_scaffold_metrics[hemi][generated_metric_output] = metric_output

        white = hemi_paths["white"]
        pial = hemi_paths["pial"]
        midthickness = (
            opts.out_dir
            / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_midthickness.surf.gii"
        )
        generated_midthickness = (
            layout.surface_work_dir / "lesion_scaffold" / midthickness.name
            if opts.lesion
            else midthickness
        )
        steps.append(
            _create_midthickness_step(
                white=white,
                pial=pial,
                output=generated_midthickness,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
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
        if opts.lesion:
            intact_midthickness = opts.out_dir / midthickness.name.replace(
                "_midthickness.surf.gii",
                "_desc-inpainted_midthickness.surf.gii",
            )
            steps.append(
                create_copy_file_step(
                    src=generated_midthickness,
                    dst=intact_midthickness,
                    force=opts.overwrite,
                    step_name=f"Publish Intact Hemisphere {hemi_label} Midthickness Surface",
                )
            )
            steps.append(
                _write_json_step(
                    intact_midthickness.with_suffix(".json"),
                    {
                        "Hemisphere": hemi_label,
                        "Space": "fsnative",
                        "AnatomicalStructurePrimary": "Cortex",
                        "SurfaceType": "midthickness",
                        "Sources": [
                            intact_surfaces[f"{hemi}.white"],
                            intact_surfaces[f"{hemi}.pial"],
                        ],
                        "SpatialReference": "T1w",
                        "SyntheticTissue": True,
                        "Description": "Intact scaffold reconstructed from the synthetic SynthSR T1w image.",
                    },
                )
            )
            intact_surfaces[f"{hemi}.midthickness"] = str(intact_midthickness)
        exported_surfaces[f"{hemi}.midthickness"] = str(midthickness)
        if opts.lesion:
            lesion_scaffold_surfaces[hemi][generated_midthickness] = midthickness
            lesion_scaffold_anatomy[hemi]["midthickness"] = generated_midthickness
    return StagePlan(
        steps=tuple(steps),
        products=SurfaceExportProducts(
            exported_surfaces=exported_surfaces,
            intact_surfaces=intact_surfaces,
            lesion_scaffold_anatomy=lesion_scaffold_anatomy,
            lesion_scaffold_metrics=lesion_scaffold_metrics,
            lesion_scaffold_surfaces=lesion_scaffold_surfaces,
        ),
    )


@dataclass(frozen=True)
class FsaverageRegistrationProducts:
    """Values that later anatomical stages read from this stage."""

    fsaverage_space: str
    fsaverage_xfms: dict[str, str]


def plan_fsaverage_registration(
    *,
    env,
    layout,
    lesion_scaffold_surfaces,
    opts,
) -> StagePlan[FsaverageRegistrationProducts]:
    """Declare the spherical registrations between fsnative and the fsaverage template."""
    steps: list[Step] = []
    fsaverage_space = opts.fsaverage_template
    fsaverage_xfms: dict[str, str] = {}
    for hemi in ("lh", "rh"):
        hemi_label = _hemi_label(hemi)
        subject_registration = layout.surface_dir / f"{hemi}.sphere.reg"
        fsaverage_sphere = find_fsaverage_template_surface(
            template=fsaverage_space,
            hemi=hemi_label,
            surface="sphere",
        )
        forward = (
            opts.out_dir
            / f"{opts.fs_subject}_from-fsnative_to-{fsaverage_space}_hemi-{hemi_label}_mode-surface_xfm.surf.gii"
        )
        generated_forward = (
            layout.surface_work_dir / "lesion_scaffold" / forward.name if opts.lesion else forward
        )
        inverse = (
            opts.out_dir
            / f"{opts.fs_subject}_from-{fsaverage_space}_to-fsnative_hemi-{hemi_label}_mode-surface_xfm.surf.gii"
        )
        steps.append(
            _create_mri_conversion_step(
                source=subject_registration,
                output=generated_forward,
                name=f"Export Hemisphere {hemi_label} fsnative-to-{fsaverage_space} Sphere",
                env=env,
                force=opts.overwrite,
                executable="mris_convert",
            )
        )
        steps.append(
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
            steps.append(
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
        if opts.lesion:
            lesion_scaffold_surfaces[hemi][generated_forward] = forward
    return StagePlan(
        steps=tuple(steps),
        products=FsaverageRegistrationProducts(
            fsaverage_space=fsaverage_space,
            fsaverage_xfms=fsaverage_xfms,
        ),
    )


@dataclass(frozen=True)
class MniRegistrationProducts:
    """Values that later anatomical stages read from this stage."""

    mni_qc_images: dict[str, str]
    mni_xfms: dict[str, str]


def plan_mni_registration(
    *,
    core_masks,
    env,
    layout,
    mni_brain_mask,
    mni_brain_template,
    opts,
    subject_t1,
) -> StagePlan[MniRegistrationProducts]:
    """Declare the T1w-to-MNI registration, its published transforms, and QC images."""
    steps: list[Step] = []
    mni_registration_inputs = (
        subject_t1,
        Path(core_masks["brain"]),
        opts.mni_template,
        mni_brain_template,
        mni_brain_mask,
    )
    steps.append(
        _create_mni_registration_step(
            subject_t1=subject_t1,
            subject_brain_mask=Path(core_masks["brain"]),
            mni_brain_template=mni_brain_template,
            mni_brain_mask=mni_brain_mask,
            mni_template=opts.mni_template,
            prefix=layout.mni_prefix,
            produced_forward=layout.produced_t1_to_mni,
            produced_inverse=layout.produced_mni_to_t1,
            env=env,
            force=opts.overwrite,
        )
    )
    steps.append(
        _create_transform_finalization_step(
            produced_forward=layout.produced_t1_to_mni,
            produced_inverse=layout.produced_mni_to_t1,
            forward=layout.t1_to_mni,
            inverse=layout.mni_to_t1,
            force=opts.overwrite,
        )
    )
    mni_xfms = {"t1w_to_mni": str(layout.t1_to_mni), "mni_to_t1w": str(layout.mni_to_t1)}
    for path, from_space, to_space in (
        (layout.t1_to_mni, "T1w", "MNI152NLin2009cAsym"),
        (layout.mni_to_t1, "MNI152NLin2009cAsym", "T1w"),
    ):
        steps.append(
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
            layout.mni_in_t1,
            opts.mni_template,
            subject_t1,
            layout.mni_to_t1,
            "MNI152NLin2009cAsym",
            "T1w",
            "Saved inverse composite transform applied once to project the MNI template into participant T1w space for QC.",
        ),
        (
            "t1w_to_mni",
            layout.t1_in_mni,
            subject_t1,
            opts.mni_template,
            layout.t1_to_mni,
            "T1w",
            "MNI152NLin2009cAsym",
            "Saved forward composite transform applied once to project the participant T1w reference into MNI space for QC.",
        ),
    ):
        steps.append(
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
        steps.append(
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
    return StagePlan(
        steps=tuple(steps),
        products=MniRegistrationProducts(
            mni_qc_images=mni_qc_images,
            mni_xfms=mni_xfms,
        ),
    )


@dataclass(frozen=True)
class TemplateTransformProducts:
    """Values that later anatomical stages read from this stage."""

    t1_fsaverage_xfms: dict[str, str]


def plan_template_transforms(
    *,
    fsaverage_space,
    fsaverage_xfms,
    opts,
) -> StagePlan[TemplateTransformProducts]:
    """Declare T1w-to-fsaverage surface transforms for both hemispheres."""
    steps: list[Step] = []
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
        steps.append(
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
        steps.append(
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
        t1_fsaverage_xfms[f"hemi-{hemi_label}_t1w_to_{fsaverage_space}"] = str(forward)
        t1_fsaverage_xfms[f"hemi-{hemi_label}_{fsaverage_space}_to_t1w"] = str(inverse)
    return StagePlan(
        steps=tuple(steps),
        products=TemplateTransformProducts(
            t1_fsaverage_xfms=t1_fsaverage_xfms,
        ),
    )


@dataclass(frozen=True)
class LesionSurfaceProducts:
    """Values that later anatomical stages read from this stage."""

    lesion_surface_validity: dict[str, str]
    lesion_vertex_mappings: dict[str, str]


def plan_lesion_surfaces(
    *,
    env,
    layout,
    lesion_mask,
    lesion_scaffold_anatomy,
    lesion_scaffold_metrics,
    lesion_scaffold_surfaces,
    opts,
) -> StagePlan[LesionSurfaceProducts]:
    """Declare lesion cutting on scaffold surfaces and the vertex mappings it records."""
    steps: list[Step] = []
    lesion_vertex_mappings: dict[str, str] = {}
    lesion_surface_validity: dict[str, str] = {}
    if opts.lesion:
        assert lesion_mask is not None
        for hemi in ("lh", "rh"):
            hemi_label = _hemi_label(hemi)
            lesion_metric = (
                layout.surface_work_dir / "lesion_scaffold" / f"hemi-{hemi_label}_lesion.shape.gii"
            )
            steps.append(
                Step.command_step(
                    [
                        "wb_command",
                        "-volume-to-surface-mapping",
                        str(lesion_mask),
                        str(lesion_scaffold_anatomy[hemi]["midthickness"]),
                        str(lesion_metric),
                        "-ribbon-constrained",
                        str(lesion_scaffold_anatomy[hemi]["white"]),
                        str(lesion_scaffold_anatomy[hemi]["pial"]),
                    ],
                    name=f"Project Lesion to Hemisphere {hemi_label}",
                    inputs=(
                        lesion_mask,
                        lesion_scaffold_anatomy[hemi]["midthickness"],
                        lesion_scaffold_anatomy[hemi]["white"],
                        lesion_scaffold_anatomy[hemi]["pial"],
                    ),
                    outputs=(lesion_metric,),
                    env=env,
                    force=opts.overwrite,
                    parameters={"mapping": "ribbon_constrained"},
                )
            )
            projected_metric = lesion_metric
            if BOUNDARY_MARGIN_MM > 0:
                dilated = lesion_metric.with_name(
                    lesion_metric.name.replace("_lesion.", "_lesionDilated.")
                )
                steps.append(
                    Step.command_step(
                        [
                            "wb_command",
                            "-metric-dilate",
                            str(lesion_metric),
                            str(lesion_scaffold_anatomy[hemi]["midthickness"]),
                            str(BOUNDARY_MARGIN_MM),
                            str(dilated),
                        ],
                        name=f"Dilate Hemisphere {hemi_label} Lesion Boundary",
                        inputs=(lesion_metric, lesion_scaffold_anatomy[hemi]["midthickness"]),
                        outputs=(dilated,),
                        env=env,
                        force=opts.overwrite,
                        parameters={"distance_mm": BOUNDARY_MARGIN_MM},
                    )
                )
                projected_metric = dilated
            mapping = (
                opts.out_dir
                / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_desc-surfaceVertexMapping.tsv"
            )
            validity = (
                opts.out_dir
                / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_desc-surfaceValidity.json"
            )
            steps.append(
                create_cut_surfaces_step(
                    scaffold_surface=lesion_scaffold_anatomy[hemi]["midthickness"],
                    lesion_metric=projected_metric,
                    surfaces=lesion_scaffold_surfaces[hemi],
                    metrics=lesion_scaffold_metrics[hemi],
                    mapping=mapping,
                    summary=validity,
                    force=opts.overwrite,
                )
            )
            lesion_vertex_mappings[hemi] = str(mapping)
            lesion_surface_validity[hemi] = str(validity)
    return StagePlan(
        steps=tuple(steps),
        products=LesionSurfaceProducts(
            lesion_surface_validity=lesion_surface_validity,
            lesion_vertex_mappings=lesion_vertex_mappings,
        ),
    )


@dataclass(frozen=True)
class PublicationProducts:
    """Values that later anatomical stages read from this stage."""

    pass


def plan_publication(
    *,
    copied_session_files,
    core_masks,
    execution_context,
    exported_surfaces,
    fsaverage_xfms,
    fsnative_xfms,
    gradient_resolutions,
    inputs,
    intact_surfaces,
    layout,
    lesion_mask,
    lesion_metadata,
    lesion_probability,
    lesion_qc,
    lesion_reconstruction_summary,
    lesion_surface_validity,
    lesion_vertex_mappings,
    mni_qc_images,
    mni_xfms,
    msmall_outputs,
    myelin_map,
    opts,
    pose_qc,
    pose_xfms,
    public_inputs,
    subcortical_masks,
    subj_t1,
    subj_t2,
    subject_anat,
    synthetic_scaffold_t1,
    synthetic_t1w,
    t1_fsaverage_xfms,
    t2w_to_t1w_xfms,
) -> StagePlan[PublicationProducts]:
    """Declare the anatomical manifest and its atomic publication.

    The manifest lists every public output and the provenance of the inputs.
    Publication validates the manifest against the anatomical output contract."""
    steps: list[Step] = []
    manifest = {
        "subject": inputs.sub_id,
        "fs_subject": opts.fs_subject,
        "fsaverage_template": opts.fsaverage_template,
        "selection_strategy": opts.selection_strategy,
        "gradient_unwarping": {
            str(path): resolution.scientific_record()
            for path, resolution in gradient_resolutions.items()
        },
        "bias_correction": bias_correction_contract(),
        "surface_reconstruction": surface_reconstruction_contract(
            opts.surface_reconstruction_engine
        ),
        "inputs": {
            "t1w": [str(item.image) for item in inputs.t1w],
            "t2w": [str(item.image) for item in inputs.t2w],
        },
        "copied_session_files": copied_session_files,
        "outputs": {
            "t1w": (str(subj_t1) if subj_t1 is not None else None),
            "t2w": (str(subj_t2) if subj_t2 is not None else None),
            "myelin_map": (str(myelin_map) if myelin_map is not None else None),
            "brain_image": str(subject_anat),
            "brain_mask": core_masks["brain"],
            "gray_matter_mask": core_masks["gray_matter"],
            "cortical_ribbon_mask": core_masks["cortical_ribbon"],
            "subcortical_masks": subcortical_masks,
            "surfaces": exported_surfaces,
            "mni_qc_images": mni_qc_images,
            "pose_qc": str(pose_qc),
            "xfms": {
                **pose_xfms,
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
        "output_metadata_contract": anatomical_output_contract(
            lesion=opts.lesion,
            msmall=opts.msmall is not None,
            synthetic_t1w=synthetic_t1w,
        ),
        "complete": True,
    }
    if synthetic_t1w:
        manifest["t1w_synthesis"] = t1w_synthesis_contract()
        manifest_options = manifest["options"]
        assert isinstance(manifest_options, dict)
        manifest_configuration = manifest_options["configuration"]
        assert isinstance(manifest_configuration, dict)
        manifest_configuration["t1w_fallback"] = opts.t1w_fallback
    if opts.lesion:
        assert lesion_mask is not None
        assert lesion_probability is not None
        assert lesion_metadata is not None
        assert lesion_qc is not None
        assert synthetic_scaffold_t1 is not None
        assert lesion_reconstruction_summary is not None
        manifest["lesion"] = {
            "enabled": True,
            "masker": {
                "backend": "SynthStroke",
                "model": MASKER_MODEL,
                "revision": MASKER_REVISION,
                "probability_threshold": PROBABILITY_THRESHOLD,
                "test_time_augmentation": TEST_TIME_AUGMENTATION,
            },
            "reconstruction": {
                "strategy": "whole_brain_synthesis_surface_reconstruction_excision",
                "scaffold_synthesis": synthsr_scaffold_contract(),
                "surface_reconstruction": surface_reconstruction_contract(
                    opts.surface_reconstruction_engine
                ),
                "scaffold_directory": str(layout.subject_dir),
                "canonical_surface_policy": "lesion_excised",
            },
            "boundary_margin_mm": BOUNDARY_MARGIN_MM,
        }
        manifest_outputs = manifest["outputs"]
        assert isinstance(manifest_outputs, dict)
        manifest_outputs.update(
            inpainted_t1w=str(synthetic_scaffold_t1),
            inpainted_t1w_metadata=str(synthetic_scaffold_t1.with_suffix("").with_suffix(".json")),
            intact_surfaces=intact_surfaces,
            lesion_mask=str(lesion_mask),
            lesion_metadata=str(lesion_metadata),
            lesion_probability=str(lesion_probability),
            lesion_qc=str(lesion_qc),
            lesion_reconstruction_summary=str(lesion_reconstruction_summary),
            surface_vertex_mappings=lesion_vertex_mappings,
            surface_validity=lesion_surface_validity,
        )
    if opts.msmall is not None:
        assert msmall_outputs is not None
        manifest["msmall"] = {
            "enabled": True,
            "manifest": msmall_outputs["manifest"],
        }
        manifest_outputs = manifest["outputs"]
        assert isinstance(manifest_outputs, dict)
        manifest_outputs["msmall"] = msmall_outputs
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
            current = read_public_json(manifest_path)
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
        try:
            if read_public_json(manifest_path) == manifest:
                return
        except (OSError, ValueError, TypeError):
            pass
        write_public_json(manifest_path, manifest)

    steps.append(
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
    return StagePlan(
        steps=tuple(steps),
        products=PublicationProducts(),
    )
