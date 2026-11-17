"""High-level stage plans for the functional preprocessing DAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from nro.definitions.hardware import GradientUnwarpingResolution
from nro.definitions.runtime import SETTINGS
from nro.definitions.schema import scientific_values
from nro.engine.bids import bids_entity, bids_readout_time
from nro.engine.execution import (
    create_configuration_snapshot_step,
    ensure_directory,
    require_existing_path,
)
from nro.engine.gradient_unwarping import create_gradient_unwarping_step
from nro.engine.image_paths import nifti_stem, sidecar_json_path
from nro.engine.io import read_json, read_public_json, write_public_json
from nro.engine.manifests import create_json_step
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
from nro.engine.references import portable_public_payload, resolve_public_payload
from nro.engine.source_metadata import inherited_sidecar_metadata
from nro.modules.func.contract import (
    final_resampling_metadata,
    functional_output_contract,
    validate_functional_image_sidecar,
    validate_functional_manifest,
)
from nro.modules.func.marss import MarssOutputs, create_marss_motion_step, create_marss_step
from nro.modules.func.synbold_disco import create_synthetic_reference_step, ensure_image
from nro.orchestration.runner_graph import StagePlan, Step
from nro.orchestration.runner_support import write_completion_breadcrumb
from nro.orchestration.runtime import selected_configuration_fingerprint

from .config import Inputs, Options, functional_config_payload
from .constants import (
    _FIELDMAP_TRANSFER_POLICY_VERSION,
    _ICA_AROMA_MELODIC_MASK_DILATION_MM,
    _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
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
from .reference_steps import (
    _create_functional_reference_selection_step,
    _create_robust_bold_reference_step,
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
from .step_support import _with_suffix
from .surface_steps import (
    _create_wb_metric_mask_step,
    _create_wb_metric_resample_step,
    _create_wb_volume_to_surface_mapping_step,
)

LOG = logging.getLogger("func")


@dataclass(frozen=True)
class InitializationProducts:
    """Configuration provenance shared with publication steps."""

    configuration: dict[str, object]


@dataclass(frozen=True)
class PreparedInputs:
    """Sources and optional corrections passed to reference construction."""

    inputs: Inputs
    processing_bold: Path
    resampling_source: Path
    processing_volume_count: int
    gradient_relative_warp: Path | None
    gradient_directory: Path | None
    marss: MarssOutputs | None


@dataclass(frozen=True)
class ReferenceProducts:
    """Robust motion and selected registration references for later stages."""

    robust_reference: Path
    robust_metadata: Path
    selected_reference: Path
    epi_to_reference: Path
    selected_metadata: Path
    gradient_afni_warp: Path | None


def plan_initialization(
    *,
    opts: Options,
    derivative_root: Path,
    gradient_unwarping: GradientUnwarpingResolution,
    required_commands: Sequence[str],
    require_commands: Callable[[Sequence[str]], None],
) -> StagePlan[InitializationProducts]:
    """Declare output initialization, configuration capture, and dependency checks."""
    initialized = opts.work_dir / "initialized.complete"

    def initialize_outputs() -> None:
        ensure_directory(derivative_root)
        ensure_directory(opts.out_dir)
        ensure_directory(opts.work_dir)
        write_completion_breadcrumb(initialized, "Functional outputs initialized\n")

    configuration = {
        "configuration": scientific_values(
            "func",
            functional_config_payload(opts, gradient_unwarping=gradient_unwarping),
        ),
        "configuration_fingerprint": selected_configuration_fingerprint(),
    }
    configuration_snapshot = opts.work_dir / "configuration.json"

    def scientific_snapshot(recorded: dict) -> dict:
        if isinstance(recorded.get("configuration"), dict):
            recorded["configuration"] = scientific_values("func", recorded["configuration"])
        return recorded

    dependency_check = opts.work_dir / "dependencies.complete"

    def check_dependencies() -> None:
        require_commands(required_commands)
        write_completion_breadcrumb(dependency_check, "Functional dependencies available\n")

    return StagePlan(
        steps=(
            Step.python(
                name="Initialize Functional Outputs",
                outputs=(initialized,),
                action=initialize_outputs,
            ),
            create_configuration_snapshot_step(
                configuration_snapshot,
                configuration,
                name="Write Functional Configuration",
                label="Functional configuration",
                inputs=(initialized,),
                force=opts.overwrite,
                normalize=scientific_snapshot,
            ),
            Step.python(
                name="Check Functional Dependencies",
                outputs=(dependency_check,),
                force=opts.overwrite,
                action=check_dependencies,
            ),
        ),
        products=InitializationProducts(configuration=configuration),
    )


def plan_input_preparation(
    *,
    opts: Options,
    inputs: Inputs,
    epi_metadata: Mapping[str, Any],
    gradient_resolutions: Mapping[str, GradientUnwarpingResolution],
    environment: Mapping[str, str],
    run_stem: str,
    run_base: str,
    source_volume_count: int,
    run_child: Callable[..., str | None],
) -> StagePlan[PreparedInputs]:
    """Declare debug selection, MARSS, and gradient-unwarping preparation."""
    steps: list[Step] = []
    processing_volume_count = source_volume_count
    processing_bold = inputs.epi
    if int(opts.debug_first_nvols) > 0:
        count = int(opts.debug_first_nvols)
        processing_volume_count = min(count, source_volume_count)
        debug_dir = opts.work_dir / "debug"
        debug_bold = debug_dir / f"{run_stem}_first{count:04d}.nii.gz"
        steps.append(
            Step.command_step(
                ["fslroi", str(inputs.epi), str(debug_bold), "0", str(count)],
                name="Select Debug BOLD Volumes",
                outputs=(debug_bold,),
                inputs=(inputs.epi,),
                force=opts.overwrite,
                env=environment,
                prepare=lambda: ensure_directory(debug_dir),
            )
        )
        processing_bold = debug_bold

    marss_outputs = None
    marss_mode = str(opts.marss_mode).strip().lower()
    if marss_mode not in {"off", "diagnose", "auto"}:
        raise SystemExit(
            f"Unsupported MARSS mode {opts.marss_mode!r}; expected off, diagnose, or auto."
        )
    if marss_mode != "off":
        marss_dir = opts.work_dir / "marss"
        motion_step, marss_motion = create_marss_motion_step(
            run_child=run_child,
            source_bold=processing_bold,
            work_dir=marss_dir / "motion",
            env=environment,
            force=opts.overwrite,
        )
        steps.append(motion_step)
        marss_step, marss_outputs = create_marss_step(
            run_child=run_child,
            source_bold=processing_bold,
            metadata=epi_metadata,
            motion_parameters=marss_motion,
            work_dir=marss_dir,
            artifact_dir=opts.out_dir,
            run_stem=run_base,
            mode=marss_mode,
            min_multiband_factor=int(opts.marss_min_multiband_factor),
            chunk_volumes=max(1, int(opts.io_chunk_vols)),
            force=opts.overwrite,
        )
        steps.append(marss_step)
        processing_bold = marss_outputs.bold

    resampling_source = processing_bold
    gradient_relative_warp: Path | None = None
    gradient_directory: Path | None = None
    bold_resolution = gradient_resolutions["epi"]
    if bold_resolution.applied:
        gradient_directory = opts.work_dir / "gradient_unwarping" / "bold"
        corrected_bold = gradient_directory / f"{run_stem}_desc-gradientCorrected_bold.nii.gz"
        gradient_relative_warp = gradient_directory / f"{run_stem}_gradient_warp.nii.gz"
        steps.append(
            create_gradient_unwarping_step(
                run_child=run_child,
                source=resampling_source,
                corrected=corrected_bold,
                warp=gradient_relative_warp,
                metadata=gradient_directory / f"{run_stem}_gradient.json",
                resolution=bold_resolution,
                runtime=opts.gradient_unwarp_runtime,
                image=opts.gradient_unwarp_image,
                force=opts.overwrite,
            )
        )
        processing_bold = corrected_bold

    def corrected_auxiliary(kind: str, source: Path | None) -> tuple[Path | None, Step | None]:
        resolution = gradient_resolutions[kind]
        if source is None or not resolution.applied:
            return source, None
        directory = opts.work_dir / "gradient_unwarping" / kind
        corrected = directory / source.name
        stem = nifti_stem(source)
        return corrected, create_gradient_unwarping_step(
            run_child=run_child,
            source=source,
            corrected=corrected,
            warp=directory / f"{stem}_gradient_warp.nii.gz",
            metadata=directory / f"{stem}_gradient.json",
            resolution=resolution,
            runtime=opts.gradient_unwarp_runtime,
            image=opts.gradient_unwarp_image,
            force=opts.overwrite,
        )

    sbref, sbref_step = corrected_auxiliary("sbref", inputs.sbref)
    se1, se1_step = corrected_auxiliary("se1", inputs.se1)
    se2, se2_step = corrected_auxiliary("se2", inputs.se2)
    steps.extend(step for step in (sbref_step, se1_step, se2_step) if step is not None)
    prepared_inputs = replace(inputs, sbref=sbref, se1=se1, se2=se2)
    return StagePlan(
        steps=tuple(steps),
        products=PreparedInputs(
            inputs=prepared_inputs,
            processing_bold=processing_bold,
            resampling_source=resampling_source,
            processing_volume_count=processing_volume_count,
            gradient_relative_warp=gradient_relative_warp,
            gradient_directory=gradient_directory,
            marss=marss_outputs,
        ),
    )


def plan_reference_preparation(
    *,
    opts: Options,
    inputs: Inputs,
    processing_bold: Path,
    processing_volume_count: int,
    run_stem: str,
    motion_directory: Path,
    environment: dict[str, str],
    epi_metadata: dict[str, Any],
    sbref_metadata_sources: tuple[Path, ...],
    gradient_relative_warp: Path | None,
    gradient_directory: Path | None,
    run_child: Callable[..., str | None],
) -> StagePlan[ReferenceProducts]:
    """Declare robust-reference construction and optional SBRef selection."""
    steps: list[Step] = []
    robust = _create_robust_bold_reference_step(
        run_child=run_child,
        epi_in=processing_bold,
        volume_count=processing_volume_count,
        run_stem=run_stem,
        mc_dir=motion_directory,
        env=environment,
        force=opts.overwrite,
    )
    steps.append(robust.step)

    gradient_afni_warp: Path | None = None
    if gradient_relative_warp is not None and gradient_directory is not None:
        gradient_world_warp = gradient_directory / f"{run_stem}_gradient_world.nii.gz"
        gradient_afni_warp = gradient_directory / f"{run_stem}_gradient_afni_lps.nii.gz"
        steps.extend(
            (
                _create_world_warp_step(
                    run_child=run_child,
                    motion_ref_3d=robust.reference,
                    ref_3d=robust.reference,
                    fnirt_warp=gradient_relative_warp,
                    world_warp=gradient_world_warp,
                    env=environment,
                    force=opts.overwrite,
                ),
                _create_afni_warp_step(
                    world_warp=gradient_world_warp,
                    motion_ref_3d=robust.reference,
                    ref_3d=robust.reference,
                    afni_warp=gradient_afni_warp,
                    force=opts.overwrite,
                ),
            )
        )

    selected = _create_functional_reference_selection_step(
        run_child=run_child,
        robust_ref=robust.reference,
        epi_metadata=epi_metadata,
        sbref=inputs.sbref,
        sbref_json=inputs.sbref_json,
        sbref_metadata=inputs.sbref_metadata,
        sbref_metadata_sources=sbref_metadata_sources,
        sbref_metadata_inheritance=inputs.sbref_metadata_inheritance,
        work_dir=opts.work_dir / "reference" / "sbref_qc",
        env=environment,
        max_rotation_degrees=opts.sbref_max_rigid_rotation_degrees,
        max_displacement_mm=opts.sbref_max_rigid_displacement_mm,
        min_support_overlap=opts.sbref_min_support_overlap,
        min_correlation=opts.sbref_min_intensity_correlation,
        force=opts.overwrite,
    )
    steps.append(selected.step)
    return StagePlan(
        steps=tuple(steps),
        products=ReferenceProducts(
            robust_reference=robust.reference,
            robust_metadata=robust.metadata,
            selected_reference=selected.image,
            epi_to_reference=selected.epi_to_reference,
            selected_metadata=selected.metadata,
            gradient_afni_warp=gradient_afni_warp,
        ),
    )


@dataclass(frozen=True)
class FunctionalPublicationProducts:
    """Values that later functional stages read from this stage."""

    pass


def plan_publication(
    *,
    anat_manifest,
    anatomical_domain,
    aroma_t1_dir,
    configuration,
    fieldmap_transfer_details,
    layout,
    marss_outputs,
    opts,
    pe_residual_details,
    policy,
    preproc_fsaverage,
    publication_manifest,
    registration_method,
    robust_reference_metadata,
    selected_reference_metadata,
    source_inputs,
) -> StagePlan[FunctionalPublicationProducts]:
    """Declare the functional manifest and its atomic publication.

    The manifest lists every public image, its metadata, the confounds, and the
    provenance of the inputs."""
    steps: list[Step] = []
    clean_inputs: dict[str, object] = {
        "desc-preproc": {
            "volumes": [
                str(layout.preproc_t1),
                *([str(layout.preproc_mni)] if policy.want_mni else []),
            ],
            "surfaces": {
                **(
                    {
                        "fsnative": {
                            hemi: str(path) for hemi, path in layout.preproc_fsnative.items()
                        }
                    }
                    if policy.want_fsnative
                    else {}
                ),
                **(
                    {
                        policy.fsaverage_space: {
                            hemi: str(path) for hemi, path in preproc_fsaverage.items()
                        }
                    }
                    if policy.want_fsaverage
                    else {}
                ),
            },
        }
    }
    public_images = [
        layout.preproc_t1,
        *([layout.preproc_mni] if policy.want_mni else []),
        *(list(layout.preproc_fsnative.values()) if policy.want_fsnative else []),
        *(list(preproc_fsaverage.values()) if policy.want_fsaverage else []),
        layout.boldref_t1_out,
        layout.reg_prenonlinear_qc_out,
        layout.reg_base_qc_out,
        layout.reg_refine_qc_out,
        layout.anat_brain_mask_in_t1,
        *([layout.reg_mni_qc_out] if policy.want_mni else []),
        *([layout.melodic_ic_t1_out] if policy.ica_enabled else []),
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
    if not policy.use_syn_fallback:
        public_images.extend(
            [
                layout.fieldmap_hz_in_t1_out,
                layout.fmap_field_hz_out,
                layout.fmap_sdc_warp_out,
                layout.fmap_topup_coeff_out,
            ]
        )
        if policy.use_fieldmap_sdc:
            public_images.append(layout.fmap_bold_sdc_warp_out)
        if opts.use_jacobian:
            public_images.append(layout.fmap_jacobian_out)
        if policy.use_synbold_reference:
            public_images.extend((layout.fmap_synbold_ref_out, layout.fmap_synbold_rigid_out))
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
    public_files = tuple(public_images) + (
        layout.confounds_tsv,
        layout.confounds_json,
        *marss_public_files,
    )

    publication_identity = {
        "manifest_version": 2,
        "module": "func",
        "run_stem": layout.run_base,
        "complete": True,
        "inputs": {
            "epi": str(source_inputs.epi),
            "epi_metadata": [str(path) for path in policy.epi_metadata_sources],
            "sbref": str(source_inputs.sbref) if source_inputs.sbref is not None else None,
            "se1": str(source_inputs.se1) if source_inputs.se1 is not None else None,
            "se2": str(source_inputs.se2) if source_inputs.se2 is not None else None,
            "anatomical_manifest": str(anat_manifest),
        },
        "options": configuration,
        "public_outputs": {
            "clean_inputs": clean_inputs,
            "confounds_tsv": str(layout.confounds_tsv),
            "confounds_json": str(layout.confounds_json),
            "files": [str(path) for path in public_files],
        },
        "output_metadata_contract": functional_output_contract(),
    }
    if anatomical_domain.lesion_aware:
        publication_inputs = publication_identity["inputs"]
        assert isinstance(publication_inputs, dict)
        publication_inputs["anatomical_registration_image"] = str(policy.registration_t1)
        publication_inputs["anatomical_lesion_mask"] = str(anatomical_domain.lesion_mask)

    def publication_payload() -> dict[str, object]:
        selection = read_json(selected_reference_metadata)
        multiband_artifact = (
            read_json(marss_outputs.metadata) if marss_outputs is not None else None
        )
        motion_indices: list[int] = []
        classified = (
            aroma_t1_dir / "aroma/classified_motion_ICs.txt"
            if policy.classifier == "ica_aroma"
            else aroma_t1_dir / "cicada/classification/cicada_python/noise_components.txt"
        )
        if policy.ica_enabled and classified.is_file():
            try:
                motion_indices = [
                    int(value)
                    for value in classified.read_text(encoding="utf-8").strip().split(",")
                    if value.strip()
                ]
            except ValueError:
                motion_indices = []
        payload = {
            **publication_identity,
            "registration": {
                "method": registration_method,
                "requested_sdc_method": policy.requested_sdc_method,
                "sdc_method": policy.resolved_sdc_method,
                "sdc_fallback_reason": policy.sdc_fallback_reason,
                "reference_selection": selection,
                "static_warp": "BOLDToT1wComposite",
                "final_resampling": final_resampling_metadata(
                    gradient_unwarping=policy.gradient_resolution.applied
                ),
                "gradient_unwarping": {
                    key: resolution.scientific_record()
                    for key, resolution in policy.gradient_resolutions.items()
                },
                "fieldmap_transfer": fieldmap_transfer_details,
                "pe_residual_refinement": pe_residual_details,
            },
            "denoising": {
                "applied": policy.ica_enabled,
                "method": ("ICA-AROMA" if policy.classifier == "ica_aroma" else "CICADA")
                if policy.ica_enabled
                else None,
                "mode": policy.denoise_type if policy.ica_enabled else None,
                "removed_noise_ic_indices": motion_indices,
                **(
                    {"simultaneous_slice_artifact": multiband_artifact}
                    if multiband_artifact is not None
                    else {}
                ),
            },
        }
        return resolve_public_payload(
            publication_manifest,
            portable_public_payload(publication_manifest, payload),
        )

    def publish() -> None:
        payload = publication_payload()
        validate_functional_manifest(payload)
        for image_path, metadata_path in zip(public_images, metadata_outputs):
            space = bids_entity(image_path, "space", default=None)
            metadata = {
                **inherited_sidecar_metadata(policy.epi_input_meta),
                "Description": "nro functional preprocessing derivative.",
                "Sources": [str(source_inputs.epi), str(anat_manifest)],
                "SpatialReference": space,
                "Registration": payload["registration"],
                "Denoising": payload["denoising"],
                "Configuration": configuration["configuration"],
                "ConfigurationFingerprint": configuration["configuration_fingerprint"],
            }
            validate_functional_image_sidecar(metadata)
            write_public_json(metadata_path, metadata)
        write_public_json(publication_manifest, payload)

    def validate_publication() -> tuple[bool, str]:
        try:
            current = read_public_json(publication_manifest)
            validate_functional_manifest(current)
        except (OSError, ValueError, TypeError):
            return False, "Functional publication manifest is missing or unreadable."
        for key, expected in publication_identity.items():
            if current.get(key) != expected:
                return False, "Functional publication manifest differs from the requested module."
        registration = current.get("registration")
        expected_registration = publication_payload()["registration"]
        if registration != expected_registration:
            return False, "Functional registration publication differs from the requested module."
        denoising = current.get("denoising")
        if not isinstance(denoising, dict) or any(
            denoising.get(key) != expected
            for key, expected in {
                "applied": policy.ica_enabled,
                "method": ("ICA-AROMA" if policy.classifier == "ica_aroma" else "CICADA")
                if policy.ica_enabled
                else None,
                "mode": policy.denoise_type if policy.ica_enabled else None,
            }.items()
        ):
            return False, "Functional denoising publication differs from the requested module."
        try:
            for metadata_path in metadata_outputs:
                validate_functional_image_sidecar(read_public_json(metadata_path))
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

    steps.append(
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
    return StagePlan(
        steps=tuple(steps),
        products=FunctionalPublicationProducts(),
    )


@dataclass(frozen=True)
class SurfaceOutputProducts:
    """Values that later functional stages read from this stage."""

    pass


def plan_surface_outputs(
    *,
    anatomical_domain,
    confounds_source_4d,
    env,
    final_sources_4d,
    fsnative_to_fsaverage_spheres,
    layout,
    opts,
    policy,
    preproc_fsaverage,
) -> StagePlan[SurfaceOutputProducts]:
    """Declare fsnative and fsaverage surface time series and their metadata."""
    steps: list[Step] = []
    final_preproc_source = final_sources_4d.get("T1w", confounds_source_4d)
    final_t1_surface_source_4d = final_sources_4d.get("T1w", final_preproc_source)
    fsnative_metric_outputs = (
        layout.preproc_fsnative
        if policy.want_fsnative
        else {
            "L": layout.surf_dir
            / _with_suffix(
                f"{layout.run_base}_space-fsnative_hemi-L", "_desc-preproc_internal.func.gii"
            ),
            "R": layout.surf_dir
            / _with_suffix(
                f"{layout.run_base}_space-fsnative_hemi-R", "_desc-preproc_internal.func.gii"
            ),
        }
    )
    if policy.need_surface_outputs:
        for hemi in ("L", "R"):
            steps.append(
                _create_wb_volume_to_surface_mapping_step(
                    volume=final_t1_surface_source_4d,
                    midthickness=policy.fsnative_surfaces[f"{hemi}.midthickness"],
                    white=policy.fsnative_surfaces[f"{hemi}.white"],
                    pial=policy.fsnative_surfaces[f"{hemi}.pial"],
                    out_metric=fsnative_metric_outputs[hemi],
                    env=env,
                    force=opts.overwrite,
                )
            )
            if policy.want_fsaverage:
                if anatomical_domain.lesion_aware:
                    resampled = layout.surf_dir / _with_suffix(
                        f"{layout.run_base}_space-{policy.fsaverage_space}_hemi-{hemi}",
                        "_desc-preproc_unmasked.func.gii",
                    )
                    valid_roi = layout.surf_dir / _with_suffix(
                        f"{layout.run_base}_space-{policy.fsaverage_space}_hemi-{hemi}",
                        "_desc-validAnatomicalDomain_roi.shape.gii",
                    )
                    steps.append(
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
                    steps.append(
                        _create_wb_metric_mask_step(
                            metric=resampled,
                            roi=valid_roi,
                            output=preproc_fsaverage[hemi],
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                else:
                    steps.append(
                        _create_wb_metric_resample_step(
                            in_metric=fsnative_metric_outputs[hemi],
                            current_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.current"],
                            new_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.new"],
                            out_metric=preproc_fsaverage[hemi],
                            env=env,
                            force=opts.overwrite,
                        )
                    )
    return StagePlan(
        steps=tuple(steps),
        products=SurfaceOutputProducts(),
    )


@dataclass(frozen=True)
class ConfoundsAndIcaProducts:
    """Values that later functional stages read from this stage."""

    confounds_source_4d: Path


def plan_confounds_and_ica(
    *,
    aroma_t1_dir,
    final_masks,
    final_sources_4d,
    final_sources_mean,
    jac_sbref,
    layout,
    opts,
    policy,
    reg_ref_tag,
    repetition_time,
    synbold_rigid_qc,
    synthetic_ref,
    topup_native,
    warp_bold_to_reg_ref,
    warp_sbref,
) -> StagePlan[ConfoundsAndIcaProducts]:
    """Declare confound estimation, fieldmap QC, and ICA classification and denoising."""
    steps: list[Step] = []
    confounds_space = "T1w" if "T1w" in final_sources_4d else "MNI152NLin2009cAsym"
    confounds_source_4d = final_sources_4d[confounds_space]
    confounds_source_mean = final_sources_mean[confounds_space]
    confounds_mask = final_masks[confounds_space]

    steps.append(
        _create_confounds_step(
            epi_4d=confounds_source_4d,
            epi_mean_3d=confounds_source_mean,
            mc_dir=layout.mc_dir,
            subjects_dir=policy.subjects_dir,
            fs_subject=policy.fs_subject,
            brain_mask_in_epi=confounds_mask,
            repetition_time=repetition_time,
            out_tsv=layout.confounds_tsv,
            out_json=layout.confounds_json,
            force=opts.overwrite,
        )
    )

    if (not policy.use_syn_fallback) and topup_native is not None and warp_sbref is not None:
        steps.append(
            create_copy_nifti_step(
                src=layout.field_hz_regref,
                dst=layout.fmap_field_hz_out,
                force=opts.overwrite,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        steps.append(
            create_copy_nifti_step(
                src=topup_native.out_prefix.with_name("topup_results_fieldcoef.nii.gz"),
                dst=layout.fmap_topup_coeff_out,
                force=opts.overwrite,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        steps.append(
            create_copy_nifti_step(
                src=warp_sbref,
                dst=layout.fmap_sdc_warp_out,
                force=opts.overwrite,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        if warp_bold_to_reg_ref is not None:
            steps.append(
                create_copy_nifti_step(
                    src=warp_bold_to_reg_ref,
                    dst=layout.fmap_bold_sdc_warp_out,
                    force=opts.overwrite,
                    step_name="Finalize BOLD-Readout SDC Warp",
                )
            )
        jac_sbref = layout.sdc_dir / f"Jacobian_{reg_ref_tag}Space.nii.gz"
        if opts.use_jacobian:
            steps.append(
                create_copy_nifti_step(
                    src=jac_sbref,
                    dst=layout.fmap_jacobian_out,
                    force=opts.overwrite,
                    step_name="Finalize Fieldmap Derivative",
                )
            )
        if synthetic_ref is not None:
            steps.append(
                create_copy_nifti_step(
                    src=synthetic_ref,
                    dst=layout.fmap_synbold_ref_out,
                    force=opts.overwrite,
                    step_name="Finalize SynBOLD-DisCo Reference",
                )
            )
            assert synbold_rigid_qc is not None
            steps.append(
                create_copy_nifti_step(
                    src=synbold_rigid_qc,
                    dst=layout.fmap_synbold_rigid_out,
                    force=opts.overwrite,
                    step_name="Finalize SynBOLD Rigid-Registration QC",
                )
            )

    if policy.ica_enabled:
        steps.append(
            create_copy_nifti_step(
                src=aroma_t1_dir / "aroma" / "melodic.ica" / "melodic_IC.nii.gz",
                dst=layout.melodic_ic_t1_out,
                force=opts.overwrite,
                step_name="Finalize ICA-AROMA Derivative",
            )
        )
    return StagePlan(
        steps=tuple(steps),
        products=ConfoundsAndIcaProducts(
            confounds_source_4d=confounds_source_4d,
        ),
    )


@dataclass(frozen=True)
class SpaceOutputProducts:
    """Values that later functional stages read from this stage."""

    final_masks: dict[str, Path]
    final_sources_4d: dict[str, Path]
    final_sources_mean: dict[str, Path]


def plan_space_outputs(
    *,
    aroma_clean,
    aroma_clean_mean,
    aroma_clean_mean_mni,
    aroma_clean_mni,
    aroma_dir,
    aroma_mni_dir,
    aroma_t1_dir,
    compute_t1,
    env,
    epi_for_proc,
    epi_mni,
    epi_t1,
    final_resampling_source,
    gradient_afni_warp,
    layout,
    opts,
    policy,
    repetition_time,
    robust_ref,
    syn_moving,
    t1_ref,
    warp_regref2t1_refined,
    warp_sbref2t1_refined,
    runner,
    run_child: Callable,
) -> StagePlan[SpaceOutputProducts]:
    """Declare single-interpolation resampling of the BOLD run into each requested volume space.

    For each space, the run is resampled once with the complete warp, then
    MARSS-cleaned, masked, and optionally ICA-denoised."""
    steps: list[Step] = []
    final_sources_4d: dict[str, Path] = {}
    final_sources_mean: dict[str, Path] = {}
    final_masks: dict[str, Path] = {}

    space_sequence: list[str] = []
    if compute_t1:
        space_sequence.append("T1w")
    if policy.want_mni:
        space_sequence.append("MNI152NLin2009cAsym")
    log_space_sequence = []
    if policy.want_t1:
        log_space_sequence.append("T1w")
    elif compute_t1:
        log_space_sequence.append("T1w(required)")
    if policy.want_fsnative:
        log_space_sequence.append("fsnative")
    if policy.want_fsaverage:
        log_space_sequence.append(policy.fsaverage_space)
    if policy.want_mni:
        log_space_sequence.append("MNI152NLin2009cAsym")
    LOG.info("Output spaces execution order: %s", ", ".join(log_space_sequence))

    if policy.want_mni:
        steps.append(
            _create_convertwarp_merge_warps_step(
                ref=layout.mni_ref,
                warp1=warp_regref2t1_refined,
                warp2=layout.t1_to_mni_fnirt_warp,
                out_warp=layout.warp_regref2mni,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=layout.mni_ref,
                warp=layout.warp_regref2mni,
                out_img=layout.reg_mni_qc_out,
                env=env,
                force=opts.overwrite,
            )
        )

    steps.append(
        _create_afni_motion_affines_step(
            in_4d=epi_for_proc,
            motion_ref_3d=robust_ref,
            mc_mat_dir=layout.mc_mat_dir,
            out_affines=layout.afni_motion_affines,
            force=opts.overwrite,
        )
    )

    for space in space_sequence:
        if space == "T1w":
            raw_4d, mean_3d, mask_3d = epi_t1, layout.epi_mean_t1, layout.anat_brain_mask_in_t1
            ref_img, warp_img = t1_ref, warp_sbref2t1_refined
            aroma_out_4d, aroma_out_mean, aroma_work = aroma_clean, aroma_clean_mean, aroma_t1_dir
            final_dst = layout.preproc_t1
            resampling_work = opts.work_dir / "resampling_t1w"
        else:
            raw_4d, mean_3d, mask_3d = epi_mni, layout.epi_mean_mni, layout.anat_brain_mask_in_mni
            ref_img, warp_img = layout.mni_ref, layout.warp_epi2mni
            aroma_out_4d, aroma_out_mean, aroma_work = (
                aroma_clean_mni,
                aroma_clean_mean_mni,
                aroma_mni_dir,
            )
            final_dst = layout.preproc_mni
            resampling_work = opts.work_dir / "resampling_mni"

        space_name = (
            f"Output Space: {space}"
            if (space != "T1w" or policy.want_t1)
            else "Preparing Required T1w Source"
        )
        LOG.info("Constructing %s", space_name)
        world_warp = resampling_work / "warp_world.nii.gz"
        afni_warp = resampling_work / "warp_afni_lps.nii.gz"
        steps.append(
            _create_world_warp_step(
                run_child=run_child,
                motion_ref_3d=robust_ref,
                ref_3d=ref_img,
                fnirt_warp=warp_img,
                world_warp=world_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_afni_warp_step(
                world_warp=world_warp,
                motion_ref_3d=robust_ref,
                ref_3d=ref_img,
                afni_warp=afni_warp,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_afni_bold_resampling_step(
                run_child=run_child,
                in_4d=final_resampling_source,
                ref_3d=ref_img,
                afni_warp=afni_warp,
                motion_affines=layout.afni_motion_affines,
                gradient_warp=gradient_afni_warp,
                out_4d=raw_4d,
                repetition_time=repetition_time,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_temporal_mean_step(
                in_4d=raw_4d,
                out_3d=mean_3d,
                force=opts.overwrite,
                chunk_vols=opts.io_chunk_vols,
            )
        )
        steps.append(
            create_mask_resampling_step(
                src_mask=policy.anat_brain_mask,
                ref_img=mean_3d,
                out_mask=mask_3d,
                force=opts.overwrite,
            )
        )

        final_4d = raw_4d
        final_mean = mean_3d
        if policy.ica_enabled:
            support_brain = aroma_work / "epi_support_bet.nii.gz"
            support_mask = aroma_work / "epi_support_bet_mask.nii.gz"
            regression_mask = aroma_work / "regression_mask.nii.gz"
            policy_path = aroma_work / f"{policy.classifier}_policy.json"
            steps.append(
                _create_epi_support_step(
                    epi_mean=mean_3d,
                    support_brain=support_brain,
                    support_mask=support_mask,
                    work_dir=aroma_work,
                    env=env,
                    force=opts.overwrite,
                )
            )
            if space == "T1w":
                melodic_mask = aroma_work / "melodic_mask.nii.gz"
                melodic_input = aroma_work / "melodic_input_smooth6mm.nii.gz"
                steps.append(
                    _create_dilated_anatomical_mask_step(
                        anatomical_mask=mask_3d,
                        support_mask=support_mask,
                        output=melodic_mask,
                        dilation_mm=_ICA_AROMA_MELODIC_MASK_DILATION_MM,
                        role="MELODIC Estimation",
                        force=opts.overwrite,
                    )
                )
                steps.append(
                    _create_dilated_anatomical_mask_step(
                        anatomical_mask=mask_3d,
                        support_mask=support_mask,
                        output=regression_mask,
                        dilation_mm=_ICA_AROMA_REGRESSION_MASK_DILATION_MM,
                        role="ICA Regression",
                        force=opts.overwrite,
                    )
                )
                steps.append(
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
                steps.append(
                    _create_ants_composite_to_itk_warp_step(
                        composite_xfm=policy.t1_to_mni_xfm,
                        ref_img=policy.anat_mni_template,
                        out_warp=t1_to_mni_itk,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                steps.append(
                    _create_wb_convert_itk_warp_to_fnirt_step(
                        itk_warp=t1_to_mni_itk,
                        src_space_ref=policy.anat_t1,
                        out_warp=t1_to_mni_warp,
                        env=env,
                        force=opts.overwrite,
                    )
                )
                identity_transform = aroma_work / "epi_in_t1_identity.mat"
                steps.append(create_identity_transform_step(identity_transform))
                aroma_dir = aroma_work / "aroma"
                melodic_dir = aroma_dir / "melodic.ica"
                melodic_products = (
                    melodic_dir / "melodic_IC.nii.gz",
                    melodic_dir / "melodic_mix",
                    melodic_dir / "melodic_FTmix",
                    aroma_dir / "melodic.complete",
                )
                if policy.classifier == "ica_aroma":
                    aroma_outputs: list[Path] = [aroma_dir / "classification_overview.txt"]
                    aroma_outputs.append(
                        aroma_dir
                        / (
                            "denoised_func_data_aggr.nii.gz"
                            if policy.denoise_type == "aggr"
                            else "denoised_func_data_nonaggr.nii.gz"
                        )
                    )
                    steps.append(
                        _create_ica_aroma_workflow_step(
                            runner=runner,
                            epi=raw_4d,
                            melodic_input=melodic_input,
                            motion_parameters=layout.mc_dir / "motion.par",
                            melodic_mask=melodic_mask,
                            regression_mask=regression_mask,
                            aroma_dir=aroma_dir,
                            outputs=tuple((*aroma_outputs, *melodic_products)),
                            melodic_products=melodic_products,
                            input_is_mni=False,
                            identity_transform=identity_transform,
                            t1_to_mni_warp=t1_to_mni_warp,
                            mni_reference=layout.mni_ref,
                            configured_command=opts.ica_aroma_cmd,
                            repetition_time=repetition_time,
                            denoise_type=policy.denoise_type,
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                else:
                    if repetition_time is None or repetition_time <= 0:
                        raise SystemExit("CICADA requires a positive RepetitionTime.")
                    steps.append(
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
                            mni_reference=layout.mni_ref,
                            repetition_time=float(repetition_time),
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                classified_components = aroma_dir / "classified_motion_ICs.txt"
                classifier_policy = policy_path
                if policy.classifier == "cicada":
                    if opts.container is None:
                        raise SystemExit("CICADA requires the configured FSL container.")
                    cicada_root = aroma_work / "cicada"
                    cicada_metrics = cicada_root / "motion_metrics.tsv"
                    cicada_task = cicada_root / "classification"
                    cicada_adapter = cicada_root / "melodic_mni"
                    cicada_manifest = cicada_root / "classification.json"
                    steps.append(
                        _create_cicada_motion_metrics_step(
                            epi=epi_mni,
                            mask=layout.anat_brain_mask_in_mni,
                            motion_parameters=layout.mc_dir / "motion.par",
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
                    steps.append(
                        _create_cicada_classification_step(
                            runner=runner,
                            epi_mni=epi_mni,
                            mask_mni=layout.anat_brain_mask_in_mni,
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
                            if policy.denoise_type == "aggr"
                            else "denoised_func_data_nonaggr.nii.gz"
                        ),
                    )
                    steps.append(
                        _create_shared_aroma_regression_step(
                            runner=runner,
                            epi=raw_4d,
                            input_space="T1w",
                            regression_mask=regression_mask,
                            mixing_matrix=melodic_dir / "melodic_mix",
                            classified_components=classified_components,
                            shared_policy=classifier_policy,
                            aroma_dir=regression_dir,
                            outputs=denoised_outputs,
                            denoise_type=policy.denoise_type,
                            classifier_name="CICADA",
                            env=env,
                            force=opts.overwrite,
                        )
                    )
                cleaned_epi = (
                    aroma_work / "regression" if policy.classifier == "cicada" else aroma_dir
                ) / (
                    "denoised_func_data_aggr.nii.gz"
                    if policy.denoise_type == "aggr"
                    else "denoised_func_data_nonaggr.nii.gz"
                )
                steps.append(
                    create_copy_nifti_step(
                        src=cleaned_epi,
                        dst=aroma_out_4d,
                        force=opts.overwrite,
                        step_name=f"Install {policy.classifier} Denoised BOLD",
                    )
                )
                steps.append(
                    _create_temporal_mean_step(
                        in_4d=aroma_out_4d,
                        out_3d=aroma_out_mean,
                        force=opts.overwrite,
                        chunk_vols=opts.io_chunk_vols,
                    )
                )
                if policy.classifier == "ica_aroma":
                    steps.append(
                        create_json_step(
                            step_name="Write ICA-AROMA Estimation Policy",
                            path=policy_path,
                            payload=_ica_aroma_policy_payload(
                                input_is_mni=False,
                                denoise_type=policy.denoise_type,
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
                    if policy.classifier == "ica_aroma"
                    else aroma_t1_dir / "cicada/classification/cicada_python/noise_components.txt"
                )
                shared_policy = (
                    aroma_t1_dir / "ica_aroma_policy.json"
                    if policy.classifier == "ica_aroma"
                    else aroma_t1_dir / "cicada/classification.json"
                )
                steps.append(
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
                        if policy.denoise_type == "aggr"
                        else "denoised_func_data_nonaggr.nii.gz"
                    )
                ]
                steps.append(
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
                        denoise_type=policy.denoise_type,
                        classifier_name="ICA-AROMA"
                        if policy.classifier == "ica_aroma"
                        else "CICADA",
                        env=env,
                        force=opts.overwrite,
                    )
                )
                cleaned_epi = aroma_dir / (
                    "denoised_func_data_aggr.nii.gz"
                    if policy.denoise_type == "aggr"
                    else "denoised_func_data_nonaggr.nii.gz"
                )
                steps.append(
                    create_copy_nifti_step(
                        src=cleaned_epi,
                        dst=aroma_out_4d,
                        force=opts.overwrite,
                        step_name=f"Install {space} {policy.classifier} Denoised BOLD",
                    )
                )
                steps.append(
                    _create_temporal_mean_step(
                        in_4d=aroma_out_4d,
                        out_3d=aroma_out_mean,
                        force=opts.overwrite,
                        chunk_vols=opts.io_chunk_vols,
                    )
                )
                if policy.classifier == "ica_aroma":
                    steps.append(
                        create_json_step(
                            step_name=f"Write {space} ICA-AROMA Regression Policy",
                            path=policy_path,
                            payload=_ica_aroma_shared_regression_policy_payload(
                                input_space=space,
                                denoise_type=policy.denoise_type,
                                repetition_time=repetition_time,
                                shared_work_dir=aroma_t1_dir,
                            ),
                            inputs=(aroma_out_4d, aroma_out_mean),
                            force=opts.overwrite,
                        )
                    )
                final_4d = aroma_out_4d
                final_mean = aroma_out_mean

        steps.append(
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
    return StagePlan(
        steps=tuple(steps),
        products=SpaceOutputProducts(
            final_masks=final_masks,
            final_sources_4d=final_sources_4d,
            final_sources_mean=final_sources_mean,
        ),
    )


@dataclass(frozen=True)
class RegistrationQcProducts:
    """Values that later functional stages read from this stage."""

    pass


def plan_registration_qc(
    *,
    base_ref_in_t1,
    env,
    fieldmap_hz_in_t1,
    layout,
    opts,
    policy,
    pre_nonlinear_ref_in_t1,
    syn_moving,
    t1_ref,
    topup_native,
    warp_regref2t1_refined,
    warp_sbref2t1_refined,
) -> StagePlan[RegistrationQcProducts]:
    """Declare the final registration QC images and the T1w-to-MNI functional warps."""
    steps: list[Step] = []
    if (not policy.use_syn_fallback) and topup_native is not None:
        # The Hz field lives in registration-reference space, so its T1w
        # derivative must follow the final registration-reference-to-T1w warp.
        # In particular, defer this until after anatomical SyN refinement; the
        # base warp above is not the final spatial mapping in that pathway.
        fieldmap_hz_in_t1 = layout.fieldmap_hz_in_t1_out
        steps.append(
            _create_applywarp_step(
                in_img=layout.field_hz_regref,
                ref_img=t1_ref,
                warp=warp_regref2t1_refined,
                out_img=fieldmap_hz_in_t1,
                env=env,
                force=opts.overwrite,
            )
        )
    steps.append(
        create_copy_nifti_step(
            src=pre_nonlinear_ref_in_t1,
            dst=layout.reg_prenonlinear_qc_out,
            force=opts.overwrite,
            step_name="Finalize Registration QC",
        )
    )
    steps.append(
        create_copy_nifti_step(
            src=base_ref_in_t1,
            dst=layout.reg_base_qc_out,
            force=opts.overwrite,
            step_name="Finalize Registration QC",
        )
    )
    if policy.do_refinement:
        steps.append(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=warp_regref2t1_refined,
                out_img=layout.reg_refine_qc_out,
                env=env,
                force=opts.overwrite,
            )
        )
    else:
        steps.append(
            create_copy_nifti_step(
                src=layout.reg_base_qc_out,
                dst=layout.reg_refine_qc_out,
                force=opts.overwrite,
                step_name="Finalize Registration QC",
            )
        )
    if policy.want_mni:
        steps.append(
            _create_ants_composite_to_itk_warp_step(
                composite_xfm=policy.t1_to_mni_xfm,
                ref_img=layout.mni_ref,
                out_warp=layout.t1_to_mni_itk_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=layout.t1_to_mni_itk_warp,
                src_space_ref=policy.anat_t1,
                out_warp=layout.t1_to_mni_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_convertwarp_merge_warps_step(
                ref=layout.mni_ref,
                warp1=warp_sbref2t1_refined,
                warp2=layout.t1_to_mni_fnirt_warp,
                out_warp=layout.warp_epi2mni,
                env=env,
                force=opts.overwrite,
            )
        )
    return StagePlan(
        steps=tuple(steps),
        products=RegistrationQcProducts(),
    )


@dataclass(frozen=True)
class AnatomicalRefinementProducts:
    """Values that later functional stages read from this stage."""

    warp_regref2t1_refined: Path | None
    warp_sbref2t1_refined: Path | None


def plan_anatomical_refinement(
    *,
    ants_forward_xfm,
    base_ref_in_t1,
    base_static_warp,
    env,
    layout,
    opts,
    policy,
    run_stem,
    syn_refine_fnirt_warp,
    syn_work,
    t1_ref,
    warp_regref2t1_refined,
    warp_sbref2t1_refined,
    warp_sbref2t1_refined_planned,
    run_child: Callable,
) -> StagePlan[AnatomicalRefinementProducts]:
    """Declare the optional post-fieldmap ANTs SyN refinement against the T1w image.

    Without refinement, the base warps pass through unchanged."""
    steps: list[Step] = []
    if policy.do_refinement:
        syn_work = layout.reg_dir / "ants_syn_refine"
        steps.append(
            create_n4_bias_correction_step(
                in_img=base_ref_in_t1,
                out_img=layout.reg_ref_in_t1_refine_n4,
                env=env,
                force=opts.overwrite,
                mask=layout.syn_fixed_mask,
            )
        )
        refinement = _create_ants_registration_step(
            run_child=run_child,
            moving_img=layout.reg_ref_in_t1_refine_n4,
            fixed_img=t1_ref,
            work_dir=syn_work,
            out_prefix=f"{run_stem}_SyNRefine_",
            env=env,
            force=opts.overwrite,
            include_linear=False,
            write_composite=False,
            fixed_mask=layout.syn_fixed_mask,
            moving_mask=layout.syn_fixed_mask,
            syn_transform=opts.syn_refine_transform,
            syn_convergence=opts.syn_refine_convergence,
            syn_shrink_factors=opts.syn_refine_shrink_factors,
            syn_smoothing_sigmas=opts.syn_refine_smoothing_sigmas,
        )
        steps.append(refinement.step)
        refine_warp = refinement.forward_transform
        ants_forward_xfm = refine_warp if ants_forward_xfm is None else ants_forward_xfm
        syn_refine_fnirt_warp = layout.syn_refine_fnirt_planned
        steps.append(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=refine_warp,
                src_space_ref=layout.reg_ref_in_t1_refine_n4,
                out_warp=syn_refine_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        assert warp_regref2t1_refined is not None
        base_regref_warp = warp_regref2t1_refined
        warp_regref2t1_refined = layout.warp_regref2t1_refined_planned
        steps.append(
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
        steps.append(
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
    return StagePlan(
        steps=tuple(steps),
        products=AnatomicalRefinementProducts(
            warp_regref2t1_refined=warp_regref2t1_refined,
            warp_sbref2t1_refined=warp_sbref2t1_refined,
        ),
    )


@dataclass(frozen=True)
class RegistrationProducts:
    """Values that later functional stages read from this stage."""

    ants_forward_xfm: Path | None
    base_ref_in_t1: Optional[Path]
    base_static_warp: Optional[Path]
    pre_nonlinear_ref_in_t1: Path | None
    syn_moving: Path | None
    syn_refine_fnirt_warp: Optional[Path]
    syn_work: Optional[Path]
    warp_regref2t1_refined: Optional[Path]
    warp_sbref2t1_refined: Optional[Path]


def plan_registration(
    *,
    ants_forward_xfm,
    env,
    epi_mc_ref,
    epi_to_reference,
    layout,
    opts,
    policy,
    reg_ref_dc_ref,
    reg_ref_dist_ref,
    run_stem,
    se2sbref_mat,
    t1_ref,
    topup_native,
    topup_to_reg_ref_refined_mat,
    warp_bold_to_reg_ref,
    warp_sbref,
    warp_sbref2t1,
    run_child: Callable,
) -> StagePlan[RegistrationProducts]:
    """Declare the BOLD-to-T1w registration and the base static warp.

    The SyN fallback registers the distorted reference with nonlinear ANTs SyN.
    With a fieldmap, boundary-based registration aligns the corrected reference
    and the static warp combines it with the fieldmap correction."""
    steps: list[Step] = []
    syn_refine_fnirt_warp: Optional[Path] = None
    warp_regref2t1_refined: Optional[Path] = None
    warp_sbref2t1_refined: Optional[Path] = None
    base_static_warp: Optional[Path] = None
    base_ref_in_t1: Optional[Path] = None
    syn_work: Optional[Path] = None
    if policy.use_syn_fallback:
        assert reg_ref_dist_ref is not None and epi_mc_ref is not None
        syn_moving = reg_ref_dist_ref
        syn_work = layout.reg_dir / "ants_syn"
        fallback_rigid_step = _create_synbold_rigid_registration_step(
            run_child=run_child,
            distorted_reference=syn_moving,
            anatomical_t1=t1_ref,
            anatomical_mask=layout.syn_fixed_mask,
            work_dir=layout.reg_dir / "whole_brain_rigid",
            env=env,
            max_translation_mm=opts.synbold_max_rigid_translation_mm,
            max_rotation_degrees=opts.synbold_max_rigid_rotation_degrees,
            force=opts.overwrite,
            rigid_mat_out=layout.epi_mean_to_t1_fallback_mat,
            rigid_qc_out=layout.epi_mean_nodc_t1_affine,
            registration_label="fieldmapless EPI-to-T1",
        )
        steps.append(fallback_rigid_step.step)
        pre_nonlinear_ref_in_t1 = layout.epi_mean_nodc_t1_affine
        steps.append(
            create_n4_bias_correction_step(
                in_img=layout.epi_mean_nodc_t1_affine,
                out_img=layout.reg_ref_in_t1_base_n4,
                env=env,
                force=opts.overwrite,
                mask=layout.syn_fixed_mask,
            )
        )
        fallback_registration = _create_ants_synboldaff_step(
            moving_img=layout.reg_ref_in_t1_base_n4,
            fixed_img=t1_ref,
            work_dir=syn_work,
            out_prefix=f"{run_stem}_SyNBoldAff_",
            env=env,
            force=opts.overwrite,
            fixed_mask=layout.syn_fixed_mask,
            moving_mask=layout.syn_fixed_mask,
            syn_transform=opts.syn_base_transform,
            syn_convergence=opts.syn_base_convergence,
            syn_shrink_factors=opts.syn_base_shrink_factors,
            syn_smoothing_sigmas=opts.syn_base_smoothing_sigmas,
        )
        steps.append(fallback_registration.step)
        fallback_composite_xfm = fallback_registration.forward_transform
        ants_forward_xfm = fallback_composite_xfm
        fallback_base_itk_warp = layout.fallback_base_itk_planned
        steps.append(
            _create_ants_composite_to_itk_warp_step(
                composite_xfm=fallback_composite_xfm,
                ref_img=t1_ref,
                out_warp=fallback_base_itk_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        fallback_base_fnirt_warp = layout.fallback_base_fnirt_planned
        steps.append(
            _create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=fallback_base_itk_warp,
                src_space_ref=layout.reg_ref_in_t1_base_n4,
                out_warp=fallback_base_fnirt_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        syn_refine_fnirt_warp = fallback_base_fnirt_warp
        fallback_affine_warp = layout.fallback_affine_warp_planned
        steps.append(
            _create_convertwarp_premat_step(
                ref=t1_ref,
                premat=layout.epi_mean_to_t1_fallback_mat,
                out_warp=fallback_affine_warp,
                env=env,
                force=opts.overwrite,
            )
        )
        fallback_base_warp = layout.fallback_base_warp_planned
        steps.append(
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
        steps.append(
            _create_convertwarp_premat_and_warp_step(
                ref=t1_ref,
                premat=epi_to_reference,
                warp1=fallback_base_warp,
                out_warp=layout.epi_to_t1_warp_planned,
                env=env,
                force=opts.overwrite,
            )
        )
        base_static_warp = layout.epi_to_t1_warp_planned
        steps.append(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=fallback_base_warp,
                out_img=layout.reg_ref_in_t1_affine,
                env=env,
                force=opts.overwrite,
            )
        )
        base_ref_in_t1 = layout.reg_ref_in_t1_affine
        boldref_t1 = layout.boldref_t1_out
        steps.append(
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
            and layout.reg_ref_to_t1w_mat is not None
            and epi_mc_ref is not None
        )
        warp_sbref2t1 = layout.warp_regref2t1_planned
        assert warp_sbref is not None
        steps.append(
            create_flirt_transform_step(
                in_img=reg_ref_dist_ref,
                ref_img=t1_ref,
                mat=layout.reg_ref_to_t1w_mat,
                out_img=layout.reg_ref_in_t1_linear,
                env=env,
                force=opts.overwrite,
            )
        )
        pre_nonlinear_ref_in_t1 = layout.reg_ref_in_t1_linear
        steps.append(
            _create_convertwarp_postmat_step(
                ref=t1_ref,
                warp1=warp_sbref,
                postmat=layout.reg_ref_to_t1w_mat,
                out_warp=warp_sbref2t1,
                env=env,
                force=opts.overwrite,
            )
        )
        syn_moving = reg_ref_dist_ref
        steps.append(
            _create_applywarp_step(
                in_img=syn_moving,
                ref_img=t1_ref,
                warp=warp_sbref2t1,
                out_img=layout.reg_ref_in_t1_affine,
                env=env,
                force=opts.overwrite,
            )
        )
        if policy.use_fieldmap_sdc:
            assert warp_bold_to_reg_ref is not None
            steps.append(
                _create_convertwarp_postmat_step(
                    ref=t1_ref,
                    warp1=warp_bold_to_reg_ref,
                    postmat=layout.reg_ref_to_t1w_mat,
                    out_warp=layout.epi_to_t1_warp_planned,
                    env=env,
                    force=opts.overwrite,
                )
            )
            base_static_warp = layout.epi_to_t1_warp_planned
        else:
            steps.append(
                _create_convertwarp_premat_and_warp_step(
                    ref=t1_ref,
                    premat=epi_to_reference,
                    warp1=warp_sbref2t1,
                    out_warp=layout.epi_to_t1_warp_planned,
                    env=env,
                    force=opts.overwrite,
                )
            )
            base_static_warp = layout.epi_to_t1_warp_planned
        base_ref_in_t1 = layout.reg_ref_in_t1_affine
        warp_regref2t1_refined = warp_sbref2t1
        boldref_t1 = layout.boldref_t1_out
        steps.append(
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
        field_to_reg_ref_mat = (
            topup_to_reg_ref_refined_mat if policy.use_fieldmap_sdc else se2sbref_mat
        )
        assert field_to_reg_ref_mat is not None
        steps.append(
            create_flirt_transform_step(
                in_img=topup_native.field_hz,
                ref_img=reg_ref_dist_ref,
                mat=field_to_reg_ref_mat,
                out_img=layout.field_hz_regref,
                env=env,
                force=opts.overwrite,
            )
        )

    assert (
        base_static_warp is not None
        and base_ref_in_t1 is not None
        and warp_regref2t1_refined is not None
    )
    return StagePlan(
        steps=tuple(steps),
        products=RegistrationProducts(
            ants_forward_xfm=ants_forward_xfm,
            base_ref_in_t1=base_ref_in_t1,
            base_static_warp=base_static_warp,
            pre_nonlinear_ref_in_t1=pre_nonlinear_ref_in_t1,
            syn_moving=syn_moving,
            syn_refine_fnirt_warp=syn_refine_fnirt_warp,
            syn_work=syn_work,
            warp_regref2t1_refined=warp_regref2t1_refined,
            warp_sbref2t1_refined=warp_sbref2t1_refined,
        ),
    )


@dataclass(frozen=True)
class DistortionCorrectionProducts:
    """Values that later functional stages read from this stage."""

    ants_forward_xfm: Optional[Path]
    epi_mc_ref: Optional[Path]
    fieldmap_hz_in_t1: Optional[Path]
    fieldmap_transfer_details: Optional[dict[str, object]]
    fsl_mat: Optional[Path]
    jac_sbref: Optional[Path]
    pe_residual_details: Optional[dict[str, object]]
    reg_ref_dc_ref: Optional[Path]
    reg_ref_dist_ref: Optional[Path]
    se2sbref_mat: Optional[Path]
    synbold_rigid_qc: Optional[Path]
    synthetic_ref: Optional[Path]
    topup_native: TopupDfOutputs | None
    topup_to_reg_ref_refined_mat: Optional[Path]
    warp_bold_to_reg_ref: Optional[Path]
    warp_sbref: Optional[Path]
    warp_sbref2t1: Optional[Path]


def plan_distortion_correction(
    *,
    env,
    epi_to_reference,
    inputs,
    layout,
    opts,
    policy,
    reg_ref_tag,
    robust_ref,
    run_stem,
    selected_reference_image,
    source_fieldmap_geometry,
    source_spatial_shape,
    run_child: Callable,
) -> StagePlan[DistortionCorrectionProducts]:
    """Declare susceptibility distortion correction for the BOLD reference.

    With a reverse phase-encoding fieldmap pair, TOPUP estimates the field and
    it is transferred to the functional reference. SynBOLD-DisCo can supply a
    synthetic undistorted reference instead. With the SyN fallback, no field
    is estimated here and registration corrects distortion later."""
    steps: list[Step] = []
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
    jac_sbref: Optional[Path] = None
    synbold_rigid_qc: Optional[Path] = None
    ants_forward_xfm: Optional[Path] = None
    epi_mc_ref: Optional[Path] = robust_ref
    reg_ref_dist_ref: Optional[Path] = selected_reference_image
    reg_ref_dc_ref: Optional[Path] = None
    fsl_mat: Optional[Path] = None

    if policy.use_syn_fallback:
        reg_ref_dist_ref = selected_reference_image
    else:
        require_existing_path(
            policy.subjects_dir, "FreeSurfer SUBJECTS_DIR from anatomical manifest"
        )
        reg_ref_dist_ref = selected_reference_image

        topup_label = "fieldmap" if policy.use_fieldmap_sdc else "synbold_disco"
        topup_work_dir = opts.work_dir / f"topup_{topup_label}_native"
        reg_dat = layout.reg_dir / f"{run_stem}_{reg_ref_tag}2t1.dat"
        fsl_mat = layout.reg_dir / f"{run_stem}_{reg_ref_tag}2t1.mat"
        raw_epi_ped = policy.epi_input_meta.get("PhaseEncodingDirection")
        epi_ped = str(raw_epi_ped).strip() if raw_epi_ped is not None else ""
        if not epi_ped:
            raise SystemExit("Missing PhaseEncodingDirection in effective EPI metadata.")
        reg_ref_ped = epi_ped

        assert reg_ref_dist_ref is not None
        if policy.use_synbold_reference:
            try:
                readout = float(bids_readout_time(policy.epi_input_meta))
            except (KeyError, TypeError, ValueError):
                readout = None
            if readout is None and policy.use_synbold_topup:
                raise SystemExit(
                    "SynBOLD-DisCo requires TotalReadoutTime, or EffectiveEchoSpacing with "
                    "a usable phase-encoding matrix size in the effective BIDS metadata."
                )
            synbold_work = opts.work_dir / "synbold_disco"
            t1_brain = synbold_work / "T1_brain.nii.gz"
            t1_brain_cmd = [
                "fslmaths",
                str(policy.registration_t1),
                "-mas",
                str(policy.anat_brain_mask),
                str(t1_brain),
            ]
            steps.append(
                Step.command_step(
                    t1_brain_cmd,
                    outputs=(t1_brain,),
                    inputs=(policy.registration_t1, policy.anat_brain_mask),
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
                run_child=run_child,
                distorted_reference=reg_ref_dist_ref,
                anatomical_t1=policy.registration_t1,
                anatomical_mask=policy.anat_brain_mask,
                work_dir=synbold_work / "rigid_registration",
                env=env,
                max_translation_mm=opts.synbold_max_rigid_translation_mm,
                max_rotation_degrees=opts.synbold_max_rigid_rotation_degrees,
                force=opts.overwrite,
            )
            steps.append(synbold_rigid_step.step)
            synbold_rigid_mat = synbold_rigid_step.matrix
            synbold_rigid_qc = synbold_rigid_step.registered
            synthetic_step = create_synthetic_reference_step(
                run_child=run_child,
                distorted_reference=reg_ref_dist_ref,
                skull_stripped_t1=t1_brain,
                epi_to_t1_mat=synbold_rigid_mat,
                image=synbold_image,
                license_file=opts.synbold_disco_license,
                engine=opts.synbold_disco_engine,
                work_dir=synbold_work,
                force=opts.overwrite,
            )
            steps.append(synthetic_step)
            synthetic_ref = synthetic_step.outputs[0]
        if policy.use_synbold_topup:
            assert synthetic_ref is not None
            topup_native = _create_topup_dfout_step(
                run_child=run_child,
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
            steps.append(topup_native.step)
            se2sbref_mat = opts.work_dir / "pose" / f"se2{reg_ref_tag}_6dof.mat"
            steps.append(create_identity_transform_step(se2sbref_mat))
            sbref2se_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2se_6dof.mat"
            steps.append(create_identity_transform_step(sbref2se_mat))
            warp_sbref = layout.sdc_dir / f"WarpField_{reg_ref_tag}Space.nii.gz"
            steps.append(
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
                bold_readout = float(bids_readout_time(policy.epi_input_meta))
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
            geometry_by_prepared = {
                prepared: source_fieldmap_geometry[kind]
                for kind, prepared in (("se1", inputs.se1), ("se2", inputs.se2))
                if prepared is not None
            }
            shape_a, volumes_a = geometry_by_prepared[se_a]
            _shape_b, volumes_b = geometry_by_prepared[se_b]
            topup_native = _create_topup_dfout_step(
                run_child=run_child,
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
                spatial_shape=shape_a,
                volumes_a=volumes_a,
                volumes_b=volumes_b,
            )
            steps.append(topup_native.step)
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
            steps.append(
                create_nifti_volume_extraction_step(
                    img=matching_se,
                    index_zero_based=0,
                    out_3d=se_match_ref,
                    env=env,
                    force=opts.overwrite,
                    label="Extract Matching Distorted SE Reference",
                    source_volume_count=(
                        topup_native.a_nvols if matching_group == "A" else topup_native.b_nvols
                    ),
                )
            )
            se2sbref_mat = opts.work_dir / "pose" / f"se2{reg_ref_tag}_6dof.mat"
            sbref2se_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2se_6dof.mat"
            pose_work = se2sbref_mat.parent / f"{se2sbref_mat.stem}_qc"
            pose_mask = pose_work / "fixed_support_mask.nii.gz"
            steps.append(
                create_image_support_mask_step(
                    image=reg_ref_dist_ref,
                    out_mask=pose_mask,
                    erosion_voxels=1,
                    minimum_voxels=100,
                    force=opts.overwrite,
                )
            )
            steps.append(
                _create_flirt_registration_step(
                    run_child=run_child,
                    in_img=se_match_ref,
                    ref_img=reg_ref_dist_ref,
                    out_mat=se2sbref_mat,
                    work_dir=pose_work,
                    fixed_mask=pose_mask,
                    env=env,
                    force=opts.overwrite,
                )
            )
            steps.append(
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
            steps.append(
                _create_concat_mats_step(
                    first=matching_motion_mat,
                    second=sbref2se_mat,
                    out_mat=reg_ref_to_topup_mat,
                    env=env,
                    force=opts.overwrite,
                )
            )
            topup_to_reg_ref_mat = opts.work_dir / "pose" / f"topup2{reg_ref_tag}_6dof.mat"
            steps.append(
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
            steps.append(
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
            steps.append(
                _create_target_shift_step(
                    field_hz=topup_native.field_hz,
                    reference=corrected_se_median,
                    readout_time=float(registration_readout),
                    shift_vox=registration_shift_vox,
                    env=env,
                    force=opts.overwrite,
                )
            )
            steps.append(
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
            steps.append(
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
            steps.append(
                _create_local_rigid_refinement_step(
                    run_child=run_child,
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
            steps.append(
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
            steps.append(
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
            steps.append(
                _create_nifti_in_ants_frame_step(
                    source=postdc_registered,
                    out_image=pe_moving,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            steps.append(
                _create_nifti_in_ants_frame_step(
                    source=corrected_se_median,
                    out_image=pe_fixed,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            steps.append(
                _create_nifti_in_ants_frame_step(
                    source=pe_overlap,
                    out_image=pe_mask,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            pe_registration = _create_ants_registration_step(
                run_child=run_child,
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
            steps.append(pe_registration.step)
            pe_residual_itk = pe_work / "residual_itk_warp.nii.gz"
            steps.append(
                _create_restore_ants_warp_step(
                    aligned_warp=pe_registration.forward_transform,
                    original_reference=postdc_registered,
                    out_warp=pe_residual_itk,
                    frame=pe_frame,
                    force=opts.overwrite,
                )
            )
            pe_residual_fnirt = pe_work / "residual_fnirt_warp.nii.gz"
            steps.append(
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
            steps.append(
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
            steps.append(
                _create_convertwarp_merge_warps_step(
                    ref=corrected_se_median,
                    warp1=registration_rigid_warp_topup,
                    warp2=pe_residual_fnirt,
                    out_warp=registration_complete_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            warp_sbref = layout.sdc_dir / f"WarpField_{reg_ref_tag}Space.nii.gz"
            steps.append(
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
            steps.append(bold_ref_to_topup_step)
            bold_ref_to_topup_mat = bold_ref_to_topup_step.outputs[0]
            bold_shift_vox = topup_work_dir / "bold_target_shift_vox.nii.gz"
            bold_warp_topup = topup_work_dir / "WarpField_bold_targetReadout.nii.gz"
            steps.append(
                _create_target_shift_step(
                    field_hz=topup_native.field_hz,
                    reference=corrected_se_median,
                    readout_time=float(bold_readout),
                    shift_vox=bold_shift_vox,
                    env=env,
                    force=opts.overwrite,
                )
            )
            steps.append(
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
            steps.append(
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
            steps.append(
                _create_convertwarp_merge_warps_step(
                    ref=corrected_se_median,
                    warp1=bold_rigid_warp_topup,
                    warp2=pe_residual_fnirt,
                    out_warp=bold_complete_warp_topup,
                    env=env,
                    force=opts.overwrite,
                )
            )
            warp_bold_to_reg_ref = layout.sdc_dir / "WarpField_boldToRegRef.nii.gz"
            steps.append(
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
                "PostFieldmapRefinementTarget": policy.fieldmap_refinement_target,
                "PostFieldmapRefinementConstraint": (
                    "Unrestricted" if policy.do_refinement else None
                ),
                "FinalInterpolationCount": 1,
            }

        reg_ref_dc = layout.sdc_dir / f"{reg_ref_tag}_dc.nii.gz"
        steps.append(
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
            jac_sbref = layout.sdc_dir / f"Jacobian_{reg_ref_tag}Space.nii.gz"
            jacobian_temporary = jac_sbref.parent / (
                jac_sbref.name.replace(".nii.gz", "") + "_tmp8.nii.gz"
            )
            jacobian_junk = jac_sbref.parent / "junk_warp.nii.gz"
            steps.append(
                _create_warp_jacobian_step(
                    warp=warp_sbref,
                    ref=reg_ref_dist_ref,
                    temporary=jacobian_temporary,
                    junk=jacobian_junk,
                    env=env,
                    force=opts.overwrite,
                )
            )
            steps.append(
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
            reg_ref_dc_jac = layout.sdc_dir / f"{reg_ref_tag}_dc_jac.nii.gz"
            jac_cmd = ["fslmaths", str(reg_ref_dc), "-mul", str(jac_sbref), str(reg_ref_dc_jac)]
            steps.append(
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
        steps.append(
            create_n4_bias_correction_step(
                in_img=reg_ref_dc_ref,
                out_img=layout.reg_ref_dc_ref_n4,
                env=env,
                force=opts.overwrite,
            )
        )
        steps.append(
            _create_bbregister_step(
                sbref=layout.reg_ref_dc_ref_n4,
                fs_subject=policy.fs_subject,
                reg_dat=reg_dat,
                fsl_mat=fsl_mat,
                surf=opts.bbregister_surf,
                init=opts.bbregister_init,
                dof=opts.bbregister_dof,
                env=env,
                force=opts.overwrite,
            )
        )
    return StagePlan(
        steps=tuple(steps),
        products=DistortionCorrectionProducts(
            ants_forward_xfm=ants_forward_xfm,
            epi_mc_ref=epi_mc_ref,
            fieldmap_hz_in_t1=fieldmap_hz_in_t1,
            fieldmap_transfer_details=fieldmap_transfer_details,
            fsl_mat=fsl_mat,
            jac_sbref=jac_sbref,
            pe_residual_details=pe_residual_details,
            reg_ref_dc_ref=reg_ref_dc_ref,
            reg_ref_dist_ref=reg_ref_dist_ref,
            se2sbref_mat=se2sbref_mat,
            synbold_rigid_qc=synbold_rigid_qc,
            synthetic_ref=synthetic_ref,
            topup_native=topup_native,
            topup_to_reg_ref_refined_mat=topup_to_reg_ref_refined_mat,
            warp_bold_to_reg_ref=warp_bold_to_reg_ref,
            warp_sbref=warp_sbref,
            warp_sbref2t1=warp_sbref2t1,
        ),
    )


@dataclass(frozen=True)
class T1ReferenceProducts:
    """Values that later functional stages read from this stage."""

    t1_ref: Path


def plan_t1_reference(
    *,
    env,
    fsl_mat,
    inputs,
    layout,
    opts,
    policy,
    run_stem,
    t1_ref,
) -> StagePlan[T1ReferenceProducts]:
    """Declare the T1w output grid, the BBR reference, and the registration masks and targets."""
    steps: list[Step] = []
    if opts.output_grid == "t1_epi_vox":
        t1_ref = layout.reg_dir / f"{run_stem}_t1_grid_epi_vox.nii.gz"
        steps.append(
            _create_t1_epi_vox_target_step(
                t1_image=policy.registration_t1,
                source_epi=inputs.epi,
                out_target=t1_ref,
                env=env,
                force=opts.overwrite,
            )
        )
    else:
        t1_ref = layout.reg_dir / f"{run_stem}_t1_native.nii.gz"
        steps.append(
            _create_t1_native_target_step(
                t1_image=policy.registration_t1,
                out_target=t1_ref,
                env=env,
                force=opts.overwrite,
            )
        )
    if not policy.use_syn_fallback:
        fs_t1_image = policy.subjects_dir / policy.fs_subject / "mri" / "T1.mgz"
        if opts.output_grid == "t1_epi_vox":
            steps.append(
                _create_t1_epi_vox_target_step(
                    t1_image=fs_t1_image,
                    source_epi=inputs.epi,
                    out_target=layout.bbr_t1_ref,
                    env=env,
                    force=opts.overwrite,
                )
            )
        else:
            steps.append(
                _create_t1_native_target_step(
                    t1_image=fs_t1_image,
                    out_target=layout.bbr_t1_ref,
                    env=env,
                    force=opts.overwrite,
                )
            )
        steps.append(
            _create_tkregister2_regheader_fslmat_step(
                mov_img=layout.bbr_t1_ref,
                targ_img=t1_ref,
                out_mat=layout.fs_t1_to_t1w_mat,
                env=env,
                force=opts.overwrite,
            )
        )
        assert fsl_mat is not None
        steps.append(
            _create_concat_mats_step(
                first=layout.fs_t1_to_t1w_mat,
                second=fsl_mat,
                out_mat=layout.reg_ref_to_t1w_mat,
                env=env,
                force=opts.overwrite,
            )
        )
    steps.append(
        create_mask_resampling_step(
            src_mask=policy.anat_brain_mask,
            ref_img=t1_ref,
            out_mask=layout.syn_fixed_mask,
            force=opts.overwrite,
        )
    )
    steps.append(
        _create_mni_2mm_target_step(
            mni_template=policy.anat_mni_template,
            out_target=layout.mni_ref,
            env=env,
            force=opts.overwrite,
        )
    )
    return StagePlan(
        steps=tuple(steps),
        products=T1ReferenceProducts(
            t1_ref=t1_ref,
        ),
    )
