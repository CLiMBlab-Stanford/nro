#!/usr/bin/env python3
"""
Run a full preprocessing sequence for one BOLD run:

- Susceptibility distortion correction (SDC) from a blip-reversed spin-echo EPI pair using FSL topup
  with explicit warp outputs (topup --dfout/--jacout).
- Construct a robust BOLD reference with provisional motion correction, then estimate the final
  MCFLIRT transforms directly from the raw BOLD to that reference.
- Qualify an available SBRef against the robust BOLD reference; use the robust reference when the
  SBRef is absent or fails metadata, geometry, transform, overlap, or similarity checks.
- Derive a target-acquisition warp from TOPUP's Hz field using the functional readout time, keep
  TOPUP in a stable fieldmap frame, and refine the corrected functional-reference pose against
  the corrected spin-echo median before conjugating the complete SDC chain into reference space.
- Boundary-based registration (FreeSurfer bbregister) on the selected undistorted reference.
- Final ANTs SyN refinement is always estimated before the final BOLD-to-T1 resampling.
- Single-interpolation 4D resampling with the complete spatial warp and one
  MCFLIRT transform per volume using AFNI 3dNwarpApply.
- Optional ICA-AROMA denoising of the registered BOLD (enabled by default).
- Confounds TSV/JSON generation from the registered or ICA-AROMA-cleaned BOLD.

Notes:
- This script assumes your FSL build supports: `topup --dfout` and `topup --jacout`.
- Any selected SBRef must have phase-encoding and readout metadata compatible with the BOLD run.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

from nro.orchestration.runtime import selected_configuration_fingerprint
from nro.configuration.runtime import SETTINGS
from nro.engine.bids import (
    bids_entity,
    bids_readout_time,
)
from nro.engine.freesurfer import find_fsaverage_directory
from nro.engine.images import (
    copy_or_convert_nifti,
    nifti_is_valid,
    nifti_spatial_shape,
    nifti_stem,
    nifti_volume_count,
    nifti_zooms_xyz,
    sidecar_json_path,
    uncompressed_nifti_path,
)
from nro.engine.io import atomic_output_path, read_json, write_json
from nro.engine.manifests import (
    create_json_step,
    require_manifest_output,
    require_nested_manifest_output,
)
from nro.engine.execution import (
    collect_bind_directories,
    ensure_directory,
    new_step_counter,
    neuroimaging_environment,
    require_existing_path,
    resolve_runner_command,
    runner_path_exists,
    strip_ansi as _strip_ansi,
)
from nro.engine.neuroimaging import (
    create_flirt_transform_step,
    create_copy_nifti_step,
    create_image_support_mask_step,
    create_native_overlap_mask_step,
    create_nifti_volume_extraction_step,
    create_n4_bias_correction_step,
    create_mask_resampling_step,
    create_identity_transform_step,
)
from nro.engine.registration import rigid_transform_metrics, validate_rigid_transform
from nro.engine.paths import (
    is_bids_session_id,
    anatomical_manifest_path,
    functional_manifest_path,
    preprocessing_derivatives_root,
    preprocess_subject_func_dir,
    preprocess_subject_func_work_dir,
    preprocess_session_func_dir,
    preprocess_session_func_work_dir,
    resolve_cwd_path,
    resolve_project_path,
    resolve_project_work_path,
)
from nro.orchestration.runner import (
    ContainerSpec,
    Runner,
    shlex_quote,
    write_completion_breadcrumb,
)
from nro.orchestration.runner_graph import Step
from nro.func.confounds import get_confounds
from nro.func.contracts import (
    FINAL_RESAMPLING_INTERPOLATION,
    FINAL_WARP_INTERPOLATION,
    final_resampling_contract,
    final_resampling_metadata,
)
from nro.func.resampling import (
    validate_afni_motion_affines,
    validate_resampled_bold,
    write_afni_motion_affines,
    write_afni_warp,
)
from nro.func.resolver import (
    ResolvedFuncRun,
    resolve_func_run_request,
)
from nro.func.ica_aroma import denoising as run_ica_aroma_denoising
from nro.func.ica_aroma import make_dilated_anatomical_epi_mask
from nro.func.ica_aroma import run_ica_aroma_workflow
from nro.func.synbold_disco import ensure_image, create_synthetic_reference_step


LOG = logging.getLogger("preprocess")

DEFAULT_QUNEX_CONTAINER = Path(SETTINGS.common.qunex_container)
_FIELDMAP_TRANSFER_POLICY_VERSION = "separate-bold-readout-pe-residual-v4"
_ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM = 6.0
_ICA_AROMA_ESTIMATION_POLICY_VERSION = "smooth-estimation-shared-t1w-dual-liberal-masks-v4"
_ICA_AROMA_MELODIC_MASK_DILATION_MM = 5.0
_ICA_AROMA_REGRESSION_MASK_DILATION_MM = 10.0
_ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD = 0.1
_NONSTEADY_DETECTION_POLICY_VERSION = "bulk-load-volume-median-v1"
next_step = new_step_counter()


def _resolve_sdc_reference_policy(
    *,
    requested_sdc_method: str,
    fieldmap_pair_available: bool,
    fieldmap_syn_refine: bool,
) -> tuple[bool, Optional[str]]:
    """Resolve synthetic-reference use and the post-fieldmap refinement target."""
    use_synbold_reference = (
        requested_sdc_method == "synbold_disco" and not fieldmap_pair_available
    )
    fieldmap_refinement_target = (
        "T1wAnatomicalSyN"
        if fieldmap_pair_available and fieldmap_syn_refine
        else None
    )
    return use_synbold_reference, fieldmap_refinement_target


def _resolve_fieldmapless_sdc_method(
    requested_sdc_method: str,
    *,
    fieldmap_pair_available: bool,
    bold_metadata: dict[str, Any],
) -> tuple[str, Optional[str]]:
    """Resolve a metadata-compatible fieldmapless SDC method."""
    if requested_sdc_method != "synbold_disco" or fieldmap_pair_available:
        return requested_sdc_method, None
    missing: list[str] = []
    phase_encoding_direction = str(
        bold_metadata.get("PhaseEncodingDirection", "")
    ).strip()
    if phase_encoding_direction not in {"i", "i-", "j", "j-", "k", "k-"}:
        missing.append("PhaseEncodingDirection")
    try:
        readout_time = float(bids_readout_time(bold_metadata))
        if readout_time <= 0:
            raise ValueError("readout time must be positive")
    except (KeyError, TypeError, ValueError):
        missing.append(
            "TotalReadoutTime or EffectiveEchoSpacing with a phase-encoding matrix size"
        )
    if not missing:
        return requested_sdc_method, None
    return (
        "syn",
        "synbold_disco requested without a usable reverse-PE fieldmap pair, but "
        "the effective BIDS metadata lack "
        + " and ".join(missing)
        + "; using ordinary anatomical SyN",
    )


def _with_suffix(stem: str, suffix: str) -> str:
    if stem.endswith("_bold"):
        return stem[: -len("_bold")] + suffix
    return stem + suffix


def _space_name_for_log(path: Path) -> str:
    return bids_entity(path, "space", default="Unknown") or "Unknown"


def _ica_aroma_output_label(denoise_type: str) -> str:
    mode = str(denoise_type).strip().lower()
    if mode == "aggr":
        return "aromaAgg"
    return "aromaNonAgg"


def _create_mris_convert_step(
    *,
    source: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        ["mris_convert", str(source), str(output)],
        outputs=(output,),
        inputs=(source,),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(output.parent),
    )


@dataclass(frozen=True)
class _RigidRegistrationStep:
    step: Step
    matrix: Path
    registered: Path


def _create_synbold_rigid_registration_step(
    *,
    run_child: Callable[..., Optional[str]],
    distorted_reference: Path,
    anatomical_t1: Path,
    anatomical_mask: Path,
    work_dir: Path,
    env: dict[str, str],
    max_translation_mm: float,
    max_rotation_degrees: float,
    force: bool,
    rigid_mat_out: Optional[Path] = None,
    rigid_qc_out: Optional[Path] = None,
    registration_label: str = "SynBOLD",
) -> _RigidRegistrationStep:
    header_mat = work_dir / "header_init.mat"
    header_aligned = work_dir / "header_init_boldref.nii.gz"
    header_cmd = [
        "flirt", "-in", str(distorted_reference), "-ref", str(anatomical_t1),
        "-usesqform", "-applyxfm", "-omat", str(header_mat), "-out", str(header_aligned),
    ]
    rigid_mat = rigid_mat_out or (work_dir / "epi_reg_d.mat")
    rigid_qc = rigid_qc_out or (work_dir / "epi_reg_d.nii.gz")
    search = str(float(max_rotation_degrees))
    rigid_cmd = [
        "flirt", "-in", str(distorted_reference), "-ref", str(anatomical_t1),
        "-init", str(header_mat), "-dof", "6", "-cost", "corratio",
        "-searchrx", f"-{search}", search, "-searchry", f"-{search}", search,
        "-searchrz", f"-{search}", search, "-omat", str(rigid_mat),
        "-out", str(rigid_qc), "-interp", "trilinear",
    ]
    rigid_qc_json = work_dir / "rigid_registration_qc.json"

    def register() -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        rigid_mat.parent.mkdir(parents=True, exist_ok=True)
        rigid_qc.parent.mkdir(parents=True, exist_ok=True)
        run_child(header_cmd, env=env)
        run_child(rigid_cmd, env=env)
        selected_candidate = "searched"
        rejected_candidates: list[str] = []
        try:
            selected_metrics = validate_rigid_transform(
                matrix=rigid_mat,
                initial_matrix=header_mat,
                max_translation_mm=max_translation_mm,
                max_rotation_degrees=max_rotation_degrees,
                center_mask=anatomical_mask,
                label=f"{registration_label} rigid registration",
            )
        except SystemExit as searched_error:
            rejected_candidates.append(str(searched_error))
            LOG.warning(
                "%s Retrying from the header initialization without a global angle search.",
                searched_error,
            )
            run_child(
                [
                    "flirt", "-in", str(distorted_reference), "-ref", str(anatomical_t1),
                    "-init", str(header_mat), "-dof", "6", "-cost", "corratio",
                    "-nosearch", "-omat", str(rigid_mat), "-out", str(rigid_qc),
                    "-interp", "trilinear",
                ],
                env=env,
            )
            selected_candidate = "local_nosearch"
            try:
                selected_metrics = validate_rigid_transform(
                    matrix=rigid_mat,
                    initial_matrix=header_mat,
                    max_translation_mm=max_translation_mm,
                    max_rotation_degrees=max_rotation_degrees,
                    center_mask=anatomical_mask,
                    label=f"{registration_label} local rigid registration",
                )
            except SystemExit as local_error:
                rejected_candidates.append(str(local_error))
                LOG.warning(
                    "%s Retaining the header-initialized rigid transform after both optimized candidates failed QC.",
                    local_error,
                )
                shutil.copy2(header_mat, rigid_mat)
                shutil.copy2(header_aligned, rigid_qc)
                selected_candidate = "header_initialization"
                selected_metrics = validate_rigid_transform(
                    matrix=rigid_mat,
                    initial_matrix=header_mat,
                    max_translation_mm=max_translation_mm,
                    max_rotation_degrees=max_rotation_degrees,
                    center_mask=anatomical_mask,
                    label=f"{registration_label} header initialization",
                )
        write_json(rigid_qc_json, {
            "RegistrationLabel": registration_label,
            "CoverageAssumption": "whole_brain",
            "CostFunction": "corratio",
            "CostFunctionWeighting": "none",
            "SelectedCandidate": selected_candidate,
            "SelectedTransform": str(rigid_mat),
            "HeaderInitialization": str(header_mat),
            "MetricsRelativeToHeaderInitialization": selected_metrics,
            "Limits": {
                "MaximumRotationDegrees": float(max_rotation_degrees),
                "MaximumCenterDisplacementMillimeters": float(max_translation_mm),
            },
            "RejectedCandidates": rejected_candidates,
        })

    def validate() -> tuple[bool, str]:
        required = (header_mat, header_aligned, rigid_mat, rigid_qc, rigid_qc_json)
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            "Rigid registration is complete."
            if not missing
            else "Rigid registration is missing outputs: " + ", ".join(missing),
        )

    return _RigidRegistrationStep(
        step=Step.directory_step(
            name=f"Whole-Brain {registration_label} Rigid Registration",
            directory=work_dir,
            breadcrumb=work_dir / ".nro_complete",
            outputs=(rigid_mat, rigid_qc, rigid_qc_json),
            inputs=(distorted_reference, anatomical_t1, anatomical_mask),
            force=force,
            action=register,
            validate=validate,
            reset_directory=False,
        ),
        matrix=rigid_mat,
        registered=rigid_qc,
    )


@dataclass(frozen=True)
class _PEAlignedANTsFrame:
    """Rigid coordinate-frame change that aligns voxel PE with an ANTs component."""

    restriction: str
    voxel_axis: int
    physical_axis: int
    pe_direction_ras: tuple[float, float, float]
    rotation_ras: tuple[tuple[float, float, float], ...]
    world_transform_ras: tuple[tuple[float, float, float, float], ...]


def _ants_pe_aligned_frame(
    reference: Path,
    phase_encoding_direction: str,
) -> _PEAlignedANTsFrame:
    """Build a header-only physical frame in which voxel PE is axis aligned.

    ANTs restricts deformation components in ITK physical coordinates, whereas
    BIDS specifies phase encoding along a voxel axis.  A rigid change of world
    coordinates lets ANTs impose its component restriction even for an oblique
    acquisition.  The voxel array is not resampled.
    """
    return _ants_pe_aligned_frame_impl(reference, phase_encoding_direction)


def _ants_pe_aligned_frame_impl(
    reference: Path,
    phase_encoding_direction: str,
) -> _PEAlignedANTsFrame:
    import nibabel as nib  # type: ignore
    import numpy as np

    axis_name = phase_encoding_direction.rstrip("-")
    voxel_axis = {"i": 0, "j": 1, "k": 2}.get(axis_name)
    if voxel_axis is None:
        raise SystemExit(f"Unsupported PhaseEncodingDirection: {phase_encoding_direction!r}")
    image = nib.load(str(reference))
    affine = np.asarray(image.affine, dtype=np.float64)
    direction = np.asarray(affine[:3, voxel_axis], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= np.finfo(np.float64).eps:
        raise SystemExit(f"Invalid spatial affine for phase-encoding axis in {reference}")
    pe_direction = direction / direction_norm

    # Choose the closest signed physical axis.  The sign does not affect the
    # allowed displacement line, but choosing it minimizes the frame rotation.
    physical_axis = int(np.argmax(np.abs(pe_direction)))
    target = np.zeros(3, dtype=np.float64)
    target[physical_axis] = 1.0 if pe_direction[physical_axis] >= 0.0 else -1.0

    cross = np.cross(pe_direction, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(pe_direction, target), -1.0, 1.0))
    if sine <= 1.0e-12:
        rotation = np.eye(3, dtype=np.float64)
    else:
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        rotation = np.eye(3, dtype=np.float64) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))

    # Rotate about the EPI grid center to keep physical coordinates numerically
    # close to their originals.  Applying this transform to every ANTs input is
    # a coordinate-frame change, not an image transformation.
    center_voxel = (np.asarray(image.shape[:3], dtype=np.float64) - 1.0) / 2.0
    center_ras = affine[:3, :3] @ center_voxel + affine[:3, 3]
    world_transform = np.eye(4, dtype=np.float64)
    world_transform[:3, :3] = rotation
    world_transform[:3, 3] = center_ras - rotation @ center_ras

    aligned_direction = rotation @ pe_direction
    if not np.allclose(aligned_direction, target, atol=1.0e-10, rtol=0.0):
        raise SystemExit(f"Could not construct a PE-aligned ANTs coordinate frame for {reference}")
    values = [0, 0, 0]
    values[physical_axis] = 1
    return _PEAlignedANTsFrame(
        restriction="x".join(str(value) for value in values),
        voxel_axis=voxel_axis,
        physical_axis=physical_axis,
        pe_direction_ras=tuple(float(value) for value in pe_direction),
        rotation_ras=tuple(tuple(float(value) for value in row) for row in rotation),
        world_transform_ras=tuple(tuple(float(value) for value in row) for row in world_transform),
    )


def _create_nifti_in_ants_frame_step(
    *,
    source: Path,
    out_image: Path,
    frame: _PEAlignedANTsFrame,
    force: bool,
) -> Step:
    """Copy a scalar NIfTI into the aligned physical frame without resampling."""
    def write_aligned() -> None:
        import nibabel as nib  # type: ignore
        import numpy as np

        image = nib.load(str(source))
        world_transform = np.asarray(frame.world_transform_ras, dtype=np.float64)
        aligned_affine = world_transform @ np.asarray(image.affine, dtype=np.float64)
        data = np.asanyarray(image.dataobj)
        header = image.header.copy()
        aligned = nib.Nifti1Image(data, aligned_affine, header)
        qform_code = int(image.header["qform_code"]) or 1
        sform_code = int(image.header["sform_code"]) or 1
        aligned.set_qform(aligned_affine, code=qform_code)
        aligned.set_sform(aligned_affine, code=sform_code)
        ensure_directory(out_image.parent)
        nib.save(aligned, str(out_image))
    return Step.python(
        name="Write NIfTI in PE-Aligned ANTs Frame",
        outputs=(out_image,),
        inputs=(source,),
        force=force,
        action=write_aligned,
    )


def _create_restore_ants_warp_step(
    *,
    aligned_warp: Path,
    original_reference: Path,
    out_warp: Path,
    frame: _PEAlignedANTsFrame,
    force: bool,
) -> Step:
    """Restore an ANTs displacement field from the PE-aligned frame.

    ITK displacement components are stored in LPS physical coordinates.  The
    voxel lattice is unchanged by the header-only frame rotation, so restoring
    the field requires rotating its vector components and replacing its affine;
    no spatial interpolation is needed.
    """
    def restore() -> None:
        import nibabel as nib  # type: ignore
        import numpy as np

        warp_image = nib.load(str(aligned_warp))
        reference = nib.load(str(original_reference))
        if tuple(warp_image.shape[:3]) != tuple(reference.shape[:3]):
            raise SystemExit(
                "PE-aligned ANTs warp grid does not match its original reference: "
                f"{aligned_warp} versus {original_reference}"
            )
        vectors = np.asanyarray(warp_image.dataobj)
        if vectors.ndim == 5 and vectors.shape[3] == 1 and vectors.shape[4] == 3:
            component_view = vectors[..., 0, :]
        elif vectors.ndim == 4 and vectors.shape[3] == 3:
            component_view = vectors
        else:
            raise SystemExit(f"Expected a three-component ANTs displacement field: {aligned_warp}")

        ras_to_lps = np.diag([-1.0, -1.0, 1.0])
        rotation_ras = np.asarray(frame.rotation_ras, dtype=np.float64)
        rotation_lps = ras_to_lps @ rotation_ras @ ras_to_lps
        restored_components = np.einsum(
            "ij,...j->...i",
            rotation_lps.T,
            np.asarray(component_view, dtype=np.float64),
            optimize=True,
        )
        restored_data = np.asarray(vectors).copy()
        if restored_data.ndim == 5:
            restored_data[..., 0, :] = restored_components.astype(restored_data.dtype, copy=False)
        else:
            restored_data[...] = restored_components.astype(restored_data.dtype, copy=False)

        header = warp_image.header.copy()
        restored = nib.Nifti1Image(restored_data, reference.affine, header)
        qform_code = int(reference.header["qform_code"]) or 1
        sform_code = int(reference.header["sform_code"]) or 1
        restored.set_qform(reference.affine, code=qform_code)
        restored.set_sform(reference.affine, code=sform_code)
        ensure_directory(out_warp.parent)
        nib.save(restored, str(out_warp))
    return Step.python(
        name="Restore ANTs Warp from PE-Aligned Frame",
        outputs=(out_warp,),
        inputs=(aligned_warp, original_reference),
        force=force,
        action=restore,
    )


def _pe_to_topup_dir(phase_encoding_direction: str) -> Tuple[int, int, int]:
    ped = phase_encoding_direction.strip()
    mapping = {"i": (1, 0, 0), "i-": (-1, 0, 0), "j": (0, 1, 0), "j-": (0, -1, 0), "k": (0, 0, 1), "k-": (0, 0, -1)}
    if ped not in mapping:
        raise ValueError(f"Unsupported PhaseEncodingDirection: {phase_encoding_direction!r}")
    return mapping[ped]


def _pe_to_fsl_shift_direction(phase_encoding_direction: str) -> str:
    """Translate a BIDS PE direction into FSL's signed shift-axis syntax."""
    ped = str(phase_encoding_direction).strip()
    mapping = {
        "i": "x",
        "i-": "x-",
        "j": "y",
        "j-": "y-",
        "k": "z",
        "k-": "z-",
    }
    try:
        return mapping[ped]
    except KeyError as error:
        raise SystemExit(f"Unsupported PhaseEncodingDirection: {phase_encoding_direction!r}") from error


def _canonical_fieldmap_order(
    first: tuple[Path, dict[str, Any]],
    second: tuple[Path, dict[str, Any]],
) -> tuple[tuple[Path, dict[str, Any]], tuple[Path, dict[str, Any]]]:
    """Return a run-independent ordering for an opposite-PE fieldmap pair."""

    def key(item: tuple[Path, dict[str, Any]]) -> tuple[str, bool, str]:
        path, metadata = item
        ped = str(metadata.get("PhaseEncodingDirection", "")).strip()
        axis = ped.rstrip("-")
        return axis, ped.endswith("-"), str(path)

    return tuple(sorted((first, second), key=key))  # type: ignore[return-value]


def _write_topup_datain(
    *,
    out_txt: Path,
    ped_a: str,
    ped_b: str,
    readout_time: float,
    a_nvols: int,
    b_nvols: int,
    readout_time_b: Optional[float] = None,
) -> None:
    d1 = _pe_to_topup_dir(ped_a)
    d2 = _pe_to_topup_dir(ped_b)
    l1 = f"{d1[0]} {d1[1]} {d1[2]} {readout_time:.8f}"
    second_readout = readout_time if readout_time_b is None else float(readout_time_b)
    l2 = f"{d2[0]} {d2[1]} {d2[2]} {second_readout:.8f}"
    lines = ([l1] * max(1, int(a_nvols))) + ([l2] * max(1, int(b_nvols)))
    ensure_directory(out_txt.parent)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _motion_matrix_breadcrumb(matrix_dir: Path) -> Path:
    return matrix_dir.with_name(f"{matrix_dir.name}.complete")


def _detect_initial_nonsteady_volumes(
    image_path: Path,
    *,
    max_vols: int,
    rel_thresh: float,
    stable_run: int,
) -> dict[str, object]:
    """Detect initial non-steady volumes after decompressing the 4D image once."""
    import nibabel as nib  # type: ignore
    import numpy as np

    from nro.func.confounds import _nonsteady_spikes

    image = nib.load(str(image_path))
    total_volumes = int(image.shape[3]) if len(image.shape) > 3 else 1
    # Converting the complete proxy in one operation is important for .nii.gz:
    # indexing the proxy once per volume can restart gzip decompression hundreds
    # of times. float32 preserves the exact dtype used by the previous loop.
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    global_signal = np.zeros(total_volumes, dtype=np.float64)
    for volume_index in range(total_volumes):
        volume = data[..., volume_index]
        support = np.isfinite(volume) & (volume != 0)
        global_signal[volume_index] = float(np.median(volume[support])) if support.any() else 0.0
    nonsteady = set(
        _nonsteady_spikes(
            global_signal,
            max_vols=int(max_vols),
            rel_thresh=float(rel_thresh),
            stable_run=int(stable_run),
        )
    )
    dropped_initial_volumes = 0
    while dropped_initial_volumes in nonsteady:
        dropped_initial_volumes += 1
    if dropped_initial_volumes >= total_volumes:
        dropped_initial_volumes = 0
    return {
        "PolicyVersion": _NONSTEADY_DETECTION_POLICY_VERSION,
        "Input": str(image_path),
        "TotalBOLDVolumes": total_volumes,
        "InitialNonSteadyStateVolumesExcluded": dropped_initial_volumes,
        "DetectionParameters": {
            "MaxInitialVolumes": int(max_vols),
            "RelativeThreshold": float(rel_thresh),
            "StableRunLength": int(stable_run),
        },
        "PerVolumeNonzeroMedianSignal": global_signal.tolist(),
    }


@dataclass(frozen=True)
class _RobustBoldReferenceStep:
    step: Step
    reference: Path
    motion_corrected: Path
    metadata: Path


def _create_robust_bold_reference_step(
    *,
    run_child: Callable[..., Optional[str]],
    epi_in: Path,
    run_stem: str,
    mc_dir: Path,
    env: dict[str, str],
    force: bool,
) -> _RobustBoldReferenceStep:
    """Add fixed two-pass motion/reference construction as one directory step."""
    bootstrap_ref = mc_dir / f"{run_stem}_mc_bootstrap_ref.nii.gz"
    provisional_dir = mc_dir / "provisional"
    provisional_mc = provisional_dir / f"{run_stem}_mc_provisional.nii.gz"
    robust_ref = mc_dir / f"{run_stem}_desc-robust_boldref.nii.gz"
    robust_metadata = mc_dir / f"{run_stem}_desc-robust_boldref.json"
    nonsteady_metadata = mc_dir / f"{run_stem}_desc-nonsteadyDetection_boldref.json"
    final_mc = mc_dir / f"{run_stem}_mc.nii.gz"
    final_par = mc_dir / "motion.par"
    final_mats = mc_dir / f"{final_mc.name}.mat"
    final_matrices_complete = _motion_matrix_breadcrumb(final_mats)
    nvols = nifti_volume_count(epi_in)
    expected_matrices = tuple(final_mats / f"MAT_{index:04d}" for index in range(nvols))
    confound_cfg = SETTINGS.get_confounds
    detection_kwargs = {
        "max_vols": int(confound_cfg.nonsteady_max_vols),
        "rel_thresh": float(confound_cfg.nonsteady_rel_thresh),
        "stable_run": int(confound_cfg.nonsteady_stable_run),
    }

    def run_mcflirt(source: Path, reference: Path, output: Path, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        run_child(
            ["mcflirt", "-in", str(source), "-out", str(output), "-reffile", str(reference), "-mats", "-plots"],
            env=env,
            cwd=directory,
        )
        raw_par = directory / f"{output.name}.par"
        canonical_par = directory / "motion.par"
        if not raw_par.is_file() or raw_par.stat().st_size == 0:
            raise SystemExit(f"MCFLIRT did not produce its motion-parameter file: {raw_par}")
        shutil.move(str(raw_par), str(canonical_par))

    def construct() -> None:
        mc_dir.mkdir(parents=True, exist_ok=True)
        run_child(["fslroi", str(epi_in), str(bootstrap_ref), "0", "1"], env=env)
        run_mcflirt(epi_in, bootstrap_ref, provisional_mc, provisional_dir)
        detection = _detect_initial_nonsteady_volumes(provisional_mc, **detection_kwargs)
        write_json(nonsteady_metadata, detection)
        total_volumes = int(detection["TotalBOLDVolumes"])
        dropped = int(detection["InitialNonSteadyStateVolumesExcluded"])
        median_input = provisional_mc
        if dropped:
            median_input = provisional_dir / f"{run_stem}_mc_provisional_steady.nii.gz"
            run_child(
                ["fslroi", str(provisional_mc), str(median_input), str(dropped), str(total_volumes - dropped)],
                env=env,
            )
        run_child(["fslmaths", str(median_input), "-Tmedian", str(robust_ref)], env=env)
        details = {
            "Type": "RobustBOLDReference",
            "Construction": "Temporal median of a provisional MCFLIRT-aligned BOLD series",
            "Statistic": "median",
            "BootstrapReference": "first BOLD volume",
            "MotionCorrection": {
                "Method": "MCFLIRT",
                "Passes": 2,
                "FinalTransformsEstimatedFromRawBOLD": True,
                "FinalInterpolationUsesOnlyFinalPassTransforms": True,
            },
            "WorkingImage": str(robust_ref),
            "TotalBOLDVolumes": total_volumes,
            "InitialNonSteadyStateVolumesExcluded": dropped,
            "VolumesUsed": total_volumes - dropped,
        }
        write_json(robust_metadata, details)
        run_mcflirt(epi_in, robust_ref, final_mc, mc_dir)
        write_completion_breadcrumb(final_matrices_complete, f"matrices={nvols}\n")

    def validate() -> tuple[bool, str]:
        required = (robust_ref, robust_metadata, nonsteady_metadata, final_mc, final_par, *expected_matrices)
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            f"Robust reference and fixed {nvols}-matrix motion inventory are complete."
            if not missing
            else "Robust reference construction is missing outputs: " + ", ".join(missing),
        )

    return _RobustBoldReferenceStep(
        step=Step.directory_step(
            name="Robust BOLD Reference and Motion Correction",
            directory=mc_dir,
            breadcrumb=mc_dir / ".nro_complete",
            outputs=(robust_ref, robust_metadata, nonsteady_metadata, final_mc, final_par, final_matrices_complete),
            inputs=(epi_in,),
            force=force,
            action=construct,
            validate=validate,
            breadcrumb_text=f"matrices={nvols}\n",
        ),
        reference=robust_ref,
        motion_corrected=final_mc,
        metadata=robust_metadata,
    )


def _image_overlap_and_correlation(
    *,
    moving_registered: Path,
    moving_mask_registered: Path,
    fixed: Path,
    fixed_mask: Path,
) -> dict[str, float]:
    import nibabel as nib  # type: ignore
    import numpy as np

    moving_data = np.asarray(nib.load(str(moving_registered)).dataobj, dtype=np.float32)
    fixed_data = np.asarray(nib.load(str(fixed)).dataobj, dtype=np.float32)
    moving_support = np.asarray(nib.load(str(moving_mask_registered)).dataobj) > 0
    fixed_support = np.asarray(nib.load(str(fixed_mask)).dataobj) > 0
    overlap = moving_support & fixed_support & np.isfinite(moving_data) & np.isfinite(fixed_data)
    overlap_count = int(overlap.sum())
    denominator = max(1, min(int(moving_support.sum()), int(fixed_support.sum())))
    support_overlap = float(overlap_count / denominator)
    correlation = 0.0
    if overlap_count >= 2:
        x = moving_data[overlap].astype(np.float64)
        y = fixed_data[overlap].astype(np.float64)
        if float(x.std()) > 0.0 and float(y.std()) > 0.0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    return {
        "SupportOverlapFraction": support_overlap,
        "IntensityCorrelation": correlation,
        "OverlapVoxels": overlap_count,
    }


@dataclass(frozen=True)
class _SelectedFunctionalReferenceStep:
    step: Step
    image: Path
    epi_to_reference: Path
    reference_to_epi: Path
    metadata: Path


def _create_functional_reference_selection_step(
    *,
    run_child: Callable[..., Optional[str]],
    robust_ref: Path,
    epi_metadata: dict[str, Any],
    epi_metadata_sources: tuple[Path, ...],
    sbref: Optional[Path],
    sbref_json: Optional[Path],
    sbref_metadata: Optional[dict[str, Any]],
    sbref_metadata_sources: tuple[Path, ...],
    sbref_metadata_inheritance: Optional[dict[str, Any]],
    work_dir: Path,
    env: dict[str, str],
    max_rotation_degrees: float,
    max_displacement_mm: float,
    min_support_overlap: float,
    min_correlation: float,
    force: bool,
) -> _SelectedFunctionalReferenceStep:
    """Create the fixed registration-reference selection step."""
    selected_image = work_dir / "selected_reference.nii.gz"
    epi_to_selected = work_dir / "epi_to_selected.mat"
    selected_to_epi = work_dir / "selected_to_epi.mat"
    metadata_path = work_dir / "selection.json"
    candidate_3d = work_dir / "sbref_3d.nii.gz"
    header_mat = work_dir / "header_init.mat"
    header_image = work_dir / "header_init_boldref.nii.gz"
    selected_mat = work_dir / "robust_to_sbref.mat"
    registered = work_dir / "robust_in_sbref.nii.gz"
    robust_mask = work_dir / "robust_support_mask.nii.gz"
    sbref_mask = work_dir / "sbref_support_mask.nii.gz"
    registered_mask = work_dir / "robust_support_in_sbref.nii.gz"

    identity = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"

    def choose_robust(details: dict[str, object]) -> None:
        copy_or_convert_nifti(robust_ref, selected_image)
        epi_to_selected.write_text(identity, encoding="utf-8")
        selected_to_epi.write_text(identity, encoding="utf-8")
        details["Selected"] = False
        details["SelectedRegistrationReference"] = "RobustBOLDReference"
        write_json(metadata_path, details)

    def support_mask(source: Path, destination: Path) -> None:
        import nibabel as nib  # type: ignore
        import numpy as np
        from scipy.ndimage import binary_erosion

        image = nib.load(str(source))
        data = np.asarray(image.dataobj)
        if data.ndim > 3:
            data = data[..., 0]
        support = binary_erosion(
            np.isfinite(data) & (np.abs(data) > np.finfo(np.float32).eps),
            iterations=1,
            border_value=0,
        )
        nib.save(
            nib.Nifti1Image(support.astype(np.uint8), image.affine, image.header),
            str(destination),
        )

    def select() -> None:
        import nibabel as nib  # type: ignore
        import numpy as np

        work_dir.mkdir(parents=True, exist_ok=True)
        details: dict[str, object] = {
            "Available": bool(sbref is not None and sbref_json is not None),
            "Selected": False,
            "InputImage": str(sbref) if sbref is not None else None,
            "InputMetadata": [str(path) for path in sbref_metadata_sources],
            "Reasons": [],
            "Metrics": {},
            "Thresholds": {
                "MaximumRigidRotationDegrees": float(max_rotation_degrees),
                "MaximumRigidCenterDisplacementMillimeters": float(max_displacement_mm),
                "MinimumSupportOverlapFraction": float(min_support_overlap),
                "MinimumIntensityCorrelation": float(min_correlation),
                "MaximumReadoutTimeRelativeDifference": 0.05,
            },
        }
        if sbref_metadata_inheritance is not None:
            details["MetadataInheritance"] = dict(sbref_metadata_inheritance)
        reasons = details["Reasons"]
        assert isinstance(reasons, list)
        if sbref is None or sbref_json is None:
            reasons.append("No matched SBRef image and JSON sidecar were available.")
            choose_robust(details)
            return
        try:
            image = nib.load(str(sbref))
            data = np.asarray(image.dataobj)
            if data.ndim == 4:
                data = data[..., 0]
            finite_nonzero = int(np.count_nonzero(np.isfinite(data) & (data != 0)))
            if data.ndim != 3 or finite_nonzero < 100:
                reasons.append(
                    f"SBRef image integrity check failed ({finite_nonzero} finite nonzero 3D voxels)."
                )
            else:
                nib.save(nib.Nifti1Image(data, image.affine, image.header), str(candidate_3d))
        except Exception as error:
            reasons.append(f"SBRef image could not be read: {type(error).__name__}: {error}")

        raw_epi_ped = epi_metadata.get("PhaseEncodingDirection")
        epi_ped = str(raw_epi_ped).strip() if raw_epi_ped is not None else None
        try:
            epi_readout = float(bids_readout_time(epi_metadata))
        except (KeyError, TypeError, ValueError):
            epi_readout = None
        effective_sbref_metadata = (
            dict(sbref_metadata) if sbref_metadata is not None else read_json(sbref_json)
        )
        raw_sbref_ped = effective_sbref_metadata.get("PhaseEncodingDirection")
        sbref_ped = str(raw_sbref_ped).strip() if raw_sbref_ped is not None else None
        try:
            sbref_readout = float(bids_readout_time(effective_sbref_metadata))
        except (KeyError, TypeError, ValueError):
            sbref_readout = None
        metrics = details["Metrics"]
        assert isinstance(metrics, dict)
        metrics.update({
            "BOLDPhaseEncodingDirection": epi_ped,
            "SBRefPhaseEncodingDirection": sbref_ped,
            "BOLDTotalReadoutTime": epi_readout,
            "SBRefTotalReadoutTime": sbref_readout,
        })
        if epi_ped and not sbref_ped:
            reasons.append("SBRef PhaseEncodingDirection metadata are missing.")
        elif epi_ped and sbref_ped and epi_ped != sbref_ped:
            reasons.append(f"PhaseEncodingDirection differs (BOLD={epi_ped}, SBRef={sbref_ped}).")
        if epi_readout is not None and sbref_readout is None:
            reasons.append("SBRef total-readout-time metadata are missing.")
        elif epi_readout is not None and sbref_readout is not None:
            relative = abs(epi_readout - sbref_readout) / max(abs(epi_readout), 1.0e-8)
            metrics["ReadoutTimeRelativeDifference"] = relative
            if relative > 0.05:
                reasons.append(f"TotalReadoutTime differs by {100.0 * relative:.1f}% (limit 5.0%).")
        if reasons:
            choose_robust(details)
            return

        support_mask(robust_ref, robust_mask)
        support_mask(candidate_3d, sbref_mask)
        run_child([
            "flirt", "-in", str(robust_ref), "-ref", str(candidate_3d),
            "-usesqform", "-applyxfm", "-omat", str(header_mat), "-out", str(header_image),
        ], env=env)
        search = str(float(max_rotation_degrees))
        run_child([
            "flirt", "-in", str(robust_ref), "-ref", str(candidate_3d),
            "-init", str(header_mat), "-dof", "6", "-cost", "normcorr",
            "-searchrx", f"-{search}", search, "-searchry", f"-{search}", search,
            "-searchrz", f"-{search}", search, "-omat", str(selected_mat),
            "-out", str(registered), "-interp", "trilinear",
        ], env=env)
        rigid = rigid_transform_metrics(
            matrix=selected_mat, initial_matrix=header_mat, center_mask=sbref_mask
        )
        used_local = (
            rigid["RotationDegrees"] > max_rotation_degrees
            or rigid["CenterDisplacementMillimeters"] > max_displacement_mm
        )
        if used_local:
            run_child([
                "flirt", "-in", str(robust_ref), "-ref", str(candidate_3d),
                "-init", str(header_mat), "-dof", "6", "-cost", "normcorr",
                "-nosearch", "-omat", str(selected_mat), "-out", str(registered),
                "-interp", "trilinear",
            ], env=env)
            rigid = rigid_transform_metrics(
                matrix=selected_mat, initial_matrix=header_mat, center_mask=sbref_mask
            )
        run_child([
            "flirt", "-in", str(robust_mask), "-ref", str(candidate_3d),
            "-applyxfm", "-init", str(selected_mat), "-interp", "nearestneighbour",
            "-out", str(registered_mask),
        ], env=env)
        rigid.update(_image_overlap_and_correlation(
            moving_registered=registered,
            moving_mask_registered=registered_mask,
            fixed=candidate_3d,
            fixed_mask=sbref_mask,
        ))
        rigid["UsedLocalNoSearchFallback"] = used_local
        metrics.update(rigid)
        if rigid["RotationDegrees"] > max_rotation_degrees:
            reasons.append("SBRef rigid rotation exceeds its configured limit.")
        if rigid["CenterDisplacementMillimeters"] > max_displacement_mm:
            reasons.append("SBRef rigid displacement exceeds its configured limit.")
        if rigid["SupportOverlapFraction"] < min_support_overlap:
            reasons.append("SBRef registered support overlap is below its configured minimum.")
        if rigid["IntensityCorrelation"] < min_correlation:
            reasons.append("SBRef registered intensity correlation is below its configured minimum.")
        if reasons:
            choose_robust(details)
            return

        run_child(
            [
                "convert_xfm", "-inverse", "-omat", str(selected_to_epi),
                str(selected_mat),
            ],
            env=env,
        )
        run_child(
            [
                "flirt", "-in", str(candidate_3d), "-ref", str(robust_ref),
                "-applyxfm", "-init", str(selected_to_epi), "-interp", "sinc",
                "-out", str(selected_image),
            ],
            env=env,
        )
        epi_to_selected.write_text(identity, encoding="utf-8")
        selected_to_epi.write_text(identity, encoding="utf-8")
        details["Selected"] = True
        details["SelectedRegistrationReference"] = "SBRef"
        details["NormalizedToBOLDReferenceGrid"] = True
        details["OriginalBOLDToSBRefTransform"] = str(selected_mat)
        reasons.append("SBRef passed metadata, transform, overlap, and similarity checks.")
        write_json(metadata_path, details)

    def validate() -> tuple[bool, str]:
        required = (selected_image, epi_to_selected, selected_to_epi, metadata_path)
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            "Functional registration reference selection is complete."
            if not missing
            else "Functional reference selection is missing outputs: " + ", ".join(missing),
        )

    return _SelectedFunctionalReferenceStep(
        step=Step.python(
            name="Select Functional Registration Reference",
            outputs=(selected_image, epi_to_selected, selected_to_epi, metadata_path),
            inputs=(
                robust_ref,
                *epi_metadata_sources,
                sbref,
                *sbref_metadata_sources,
            ),
            force=force,
            action=select,
            validate=validate,
        ),
        image=selected_image,
        epi_to_reference=epi_to_selected,
        reference_to_epi=selected_to_epi,
        metadata=metadata_path,
    )


def _create_bbregister_step(
    *,
    sbref: Path,
    fs_subject: str,
    reg_dat: Path,
    fsl_mat: Path,
    surf: str,
    init: str,
    dof: int,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "bbregister",
        "--s",
        fs_subject,
        "--mov",
        str(sbref),
        "--bold",
        "--reg",
        str(reg_dat),
        "--fslmat",
        str(fsl_mat),
        "--surf",
        surf,
    ]
    init_flag = {"coreg": "--init-coreg", "fsl": "--init-fsl", "header": "--init-header", "rr": "--init-rr"}.get(init)
    dof_flag = {6: "--6", 9: "--9", 12: "--12"}.get(int(dof))
    if init_flag is None or dof_flag is None:
        raise SystemExit("Invalid bbregister init/dof")
    cmd.extend([init_flag, dof_flag])
    return Step.command_step(
        cmd,
        outputs=(reg_dat, fsl_mat),
        inputs=(sbref,),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(reg_dat.parent),
    )


def _create_flirt_registration_step(
    *, run_child: Callable[..., Optional[str]], in_img: Path, ref_img: Path,
    out_mat: Path, work_dir: Path, fixed_mask: Path,
    env: dict[str, str], force: bool,
) -> Step:
    header_mat = work_dir / "header_init.mat"
    header_image = work_dir / "header_init.nii.gz"
    header_cmd = [
        "flirt", "-in", str(in_img), "-ref", str(ref_img), "-usesqform", "-applyxfm",
        "-omat", str(header_mat), "-out", str(header_image),
    ]
    registered = work_dir / "registered_wholebrain.nii.gz"
    cmd = [
        "flirt",
        "-in",
        str(in_img),
        "-ref",
        str(ref_img),
        "-init",
        str(header_mat),
        "-dof",
        "6",
        "-cost",
        "normcorr",
        "-searchrx",
        "-20",
        "20",
        "-searchry",
        "-20",
        "20",
        "-searchrz",
        "-20",
        "20",
        "-omat",
        str(out_mat),
        "-out",
        str(registered),
    ]
    rigid_qc_json = work_dir / "rigid_registration_qc.json"

    def register() -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_mat.parent.mkdir(parents=True, exist_ok=True)
        run_child(header_cmd, env=env)
        run_child(cmd, env=env)
        selected_candidate = "searched"
        rejected_candidates: list[str] = []
        try:
            selected_metrics = validate_rigid_transform(
                matrix=out_mat, initial_matrix=header_mat,
                max_translation_mm=30.0, max_rotation_degrees=20.0,
                center_mask=fixed_mask, label="EPI pose registration",
            )
        except SystemExit as searched_error:
            rejected_candidates.append(str(searched_error))
            LOG.warning("%s Retrying locally without a global angle search.", searched_error)
            run_child([
                "flirt", "-in", str(in_img), "-ref", str(ref_img),
                "-init", str(header_mat), "-dof", "6", "-cost", "normcorr",
                "-nosearch", "-omat", str(out_mat), "-out", str(registered),
            ], env=env)
            selected_candidate = "local_nosearch"
            try:
                selected_metrics = validate_rigid_transform(
                    matrix=out_mat, initial_matrix=header_mat,
                    max_translation_mm=30.0, max_rotation_degrees=20.0,
                    center_mask=fixed_mask, label="local EPI pose registration",
                )
            except SystemExit as local_error:
                rejected_candidates.append(str(local_error))
                LOG.warning(
                    "%s Retaining the header-initialized EPI pose after both optimized candidates failed QC.",
                    local_error,
                )
                shutil.copy2(header_mat, out_mat)
                shutil.copy2(header_image, registered)
                selected_candidate = "header_initialization"
                selected_metrics = validate_rigid_transform(
                    matrix=out_mat, initial_matrix=header_mat,
                    max_translation_mm=30.0, max_rotation_degrees=20.0,
                    center_mask=fixed_mask, label="header-initialized EPI pose registration",
                )
        write_json(rigid_qc_json, {
            "RegistrationLabel": "EPI pose registration",
            "CoverageAssumption": "whole_brain",
            "CostFunction": "normcorr",
            "CostFunctionWeighting": "none",
            "SelectedCandidate": selected_candidate,
            "MetricsRelativeToHeaderInitialization": selected_metrics,
            "RejectedCandidates": rejected_candidates,
        })

    def validate() -> tuple[bool, str]:
        required = (header_mat, header_image, out_mat, registered, rigid_qc_json)
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            "EPI pose registration is complete."
            if not missing
            else "EPI pose registration is missing outputs: " + ", ".join(missing),
        )

    return Step.directory_step(
        name="Whole-Brain EPI Pose Registration",
        directory=work_dir,
        breadcrumb=work_dir / ".nro_complete",
        outputs=(out_mat, registered, rigid_qc_json),
        inputs=(in_img, ref_img, fixed_mask),
        force=force,
        action=register,
        validate=validate,
        reset_directory=False,
    )


def _create_invert_mat_step(*, mat: Path, out_mat: Path, env: dict[str, str], force: bool) -> Step:
    cmd = ["convert_xfm", "-inverse", "-omat", str(out_mat), str(mat)]
    return Step.command_step(
        cmd, outputs=(out_mat,), inputs=(mat,), force=force, env=env,
        prepare=lambda: ensure_directory(out_mat.parent),
    )


def _create_concat_mats_step(*, first: Path, second: Path, out_mat: Path, env: dict[str, str], force: bool) -> Step:
    # FSL concat applies the second matrix first, then the first matrix.
    cmd = ["convert_xfm", "-omat", str(out_mat), "-concat", str(first), str(second)]
    return Step.command_step(
        cmd, outputs=(out_mat,), inputs=(first, second), force=force, env=env,
        prepare=lambda: ensure_directory(out_mat.parent),
    )


def _create_bold_ref_to_topup_transform_step(
    *,
    reg_ref_to_topup_mat: Path,
    epi_ref_to_reg_ref_mat: Path,
    work_dir: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Compose BOLD-reference-to-selected-reference and reference-to-TOPUP transforms."""
    bold_ref_to_topup_mat = work_dir / "pose" / "epiRef2topup_6dof.mat"
    return _create_concat_mats_step(
        first=reg_ref_to_topup_mat,
        second=epi_ref_to_reg_ref_mat,
        out_mat=bold_ref_to_topup_mat,
        env=env,
        force=force,
    )


def _create_target_shift_step(
    *,
    field_hz: Path,
    reference: Path,
    readout_time: float,
    shift_vox: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Create a target-readout voxel-shift-map step."""
    readout = float(readout_time)
    if not (readout > 0.0):
        raise SystemExit(f"Target TotalReadoutTime must be positive (got {readout_time!r})")
    shift_cmd = ["fslmaths", str(field_hz), "-mul", f"{readout:.12g}", str(shift_vox)]
    return Step.command_step(
        shift_cmd,
        env=env,
        outputs=(shift_vox,),
        inputs=(field_hz, reference),
        force=force,
        name="Build Target-Readout Voxel Shift Map",
        prepare=lambda: ensure_directory(shift_vox.parent),
    )


def _create_target_readout_warp_step(
    *,
    field_hz: Path,
    reference: Path,
    phase_encoding_direction: str,
    shift_vox: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Create a displacement warp from a target-readout shift map."""
    shift_direction = _pe_to_fsl_shift_direction(phase_encoding_direction)
    warp_cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={reference}",
        f"--shiftmap={shift_vox}",
        f"--shiftdir={shift_direction}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        warp_cmd,
        env=env,
        outputs=(out_warp,),
        inputs=(field_hz, reference, shift_vox),
        force=force,
        name="Build Target-Readout TOPUP Warp",
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_local_rigid_refinement_step(
    *,
    run_child: Callable[..., Optional[str]],
    moving: Path,
    fixed: Path,
    out_mat: Path,
    out_registered: Path,
    qc_json: Path,
    env: dict[str, str],
    force: bool,
    max_translation_mm: float = 5.0,
    max_rotation_degrees: float = 5.0,
) -> Step:
    """Create a guarded local same-grid rigid-refinement step."""
    work_dir = out_mat.parent
    header_mat = work_dir / f"{out_mat.stem}_header.mat"
    header_image = work_dir / f"{out_mat.stem}_header.nii.gz"
    header_cmd = [
        "flirt",
        "-in",
        str(moving),
        "-ref",
        str(fixed),
        "-usesqform",
        "-applyxfm",
        "-omat",
        str(header_mat),
        "-out",
        str(header_image),
    ]
    cmd = [
        "flirt",
        "-in",
        str(moving),
        "-ref",
        str(fixed),
        "-init",
        str(header_mat),
        "-nosearch",
        "-dof",
        "6",
        "-cost",
        "normcorr",
        "-interp",
        "sinc",
        "-sincwidth",
        "7",
        "-sincwindow",
        "hanning",
        "-omat",
        str(out_mat),
        "-out",
        str(out_registered),
    ]
    def refine() -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        run_child(header_cmd, env=env)
        run_child(cmd, env=env)
        rejected_reason: Optional[str] = None
        try:
            metrics = validate_rigid_transform(
                matrix=out_mat,
                initial_matrix=header_mat,
                max_translation_mm=float(max_translation_mm),
                max_rotation_degrees=float(max_rotation_degrees),
                center_mask=None,
                label="post-SDC EPI-to-SE rigid refinement",
            )
            selected = "local_nosearch"
        except SystemExit as error:
            rejected_reason = str(error)
            LOG.warning("%s Retaining the header-initialized post-SDC pose.", rejected_reason)
            shutil.copy2(header_mat, out_mat)
            shutil.copy2(header_image, out_registered)
            metrics = validate_rigid_transform(
                matrix=out_mat,
                initial_matrix=header_mat,
                max_translation_mm=float(max_translation_mm),
                max_rotation_degrees=float(max_rotation_degrees),
                center_mask=None,
                label="header-initialized post-SDC EPI-to-SE pose",
            )
            selected = "header_initialization"
        write_json(qc_json, {
            "PolicyVersion": _FIELDMAP_TRANSFER_POLICY_VERSION,
            "RegistrationLabel": "Post-SDC EPI-to-corrected-SE pose refinement",
            "CostFunction": "normcorr",
            "AngleSearch": "none",
            "DegreesOfFreedom": 6,
            "SelectedCandidate": selected,
            "MetricsRelativeToHeaderInitialization": metrics,
            "Limits": {
                "MaximumTranslationMillimeters": float(max_translation_mm),
                "MaximumRotationDegrees": float(max_rotation_degrees),
            },
            "RejectedCandidate": rejected_reason,
        })

    return Step.python(
        name="Refine Post-SDC EPI-to-SE Pose",
        outputs=(out_mat, out_registered, qc_json),
        inputs=(moving, fixed),
        force=force,
        action=refine,
    )




def _create_tkregister2_regheader_fslmat_step(
    *,
    mov_img: Path,
    targ_img: Path,
    out_mat: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    reg_dat = out_mat.with_suffix(".dat")
    cmd = [
        "tkregister2",
        "--mov",
        str(mov_img),
        "--targ",
        str(targ_img),
        "--regheader",
        "--reg",
        str(reg_dat),
        "--noedit",
        "--fslregout",
        str(out_mat),
    ]
    return Step.command_step(
        cmd,
        outputs=(out_mat,),
        inputs=(mov_img, targ_img),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_mat.parent),
        finalize=lambda: reg_dat.unlink(missing_ok=True),
        validate=lambda: (
            nifti_is_valid(mov_img) and nifti_is_valid(targ_img),
            "tkregister2 source and target images are readable NIfTIs.",
        ),
    )


@dataclass(frozen=True)
class TopupDfOutputs:
    step: Step
    out_prefix: Path
    field_hz: Path
    iout: Path
    dfout: Path
    jacout: Path
    rbmout: Path
    a_nvols: int
    b_nvols: int


def _normalized_topup_matrix(prefix: Path, *, index_1based: int) -> Path:
    """Return one canonical matrix path from a normalized TOPUP directory."""
    index = int(index_1based)
    if index < 1:
        raise SystemExit(f"topup motion-matrix index must be >= 1 (got {index_1based})")
    return prefix.parent / f"{prefix.name}_{index:04d}.mat"


def _raw_topup_member(prefix: Path, *, index_1based: int, extension: str) -> Path:
    """Resolve a member inside TOPUP's opaque directory for normalization.

    FSL releases differ only in index padding.  These foreknown spellings are
    implementation details of the surrounding directory artifact and never
    become DAG outputs themselves.
    """
    index = int(index_1based)
    candidates = tuple(
        prefix.parent / f"{prefix.name}_{index:0{width}d}{extension}"
        for width in (2, 1, 3, 4)
    )
    for path in dict.fromkeys(candidates):
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise SystemExit(
        f"TOPUP did not produce indexed member {index} for {prefix}; checked: "
        + ", ".join(str(path) for path in dict.fromkeys(candidates))
    )


def _create_topup_dfout_step(
    *,
    run_child: Callable[..., Optional[str]],
    se_a: Path,
    se_b: Path,
    ped_a: str,
    ped_b: str,
    readout_time: float,
    topup_dir: Path,
    topup_config: str,
    env: dict[str, str],
    force: bool,
    readout_time_b: Optional[float] = None,
) -> TopupDfOutputs:
    if topup_config.strip().lower() == "auto":
        shape = nifti_spatial_shape(se_a)
        topup_config = "b02b0_2.cnf" if all(size % 2 == 0 for size in shape) else "b02b0_1.cnf"
        LOG.info("TOPUP configuration selected for image dimensions %s: %s", shape, topup_config)
    merged = topup_dir / "se_merged.nii.gz"
    datain = topup_dir / "acqparams.txt"
    out_prefix = topup_dir / "topup_results"
    iout = topup_dir / "se_unwarped.nii.gz"
    fout = topup_dir / "fieldmap_Hz.nii.gz"
    # Use extension-less prefixes to avoid confusing FSL's output naming.
    dfout_prefix = topup_dir / "WarpField"
    jacout_prefix = topup_dir / "Jacobian"
    rbmout_prefix = topup_dir / "MotionMatrix"
    spec_path = topup_dir / "topup_spec.json"
    result_manifest = topup_dir / "topup_outputs.json"
    topup_complete = topup_dir / "topup.complete"

    a_nvols = nifti_volume_count(se_a)
    b_nvols = nifti_volume_count(se_b)
    total_nvols = int(a_nvols) + int(b_nvols)
    second_readout = float(readout_time if readout_time_b is None else readout_time_b)
    spec: dict[str, object] = {
        "PolicyVersion": _FIELDMAP_TRANSFER_POLICY_VERSION,
        "InputA": str(se_a),
        "InputB": str(se_b),
        "PhaseEncodingDirectionA": str(ped_a),
        "PhaseEncodingDirectionB": str(ped_b),
        "TotalReadoutTimeA": float(readout_time),
        "TotalReadoutTimeB": second_readout,
        "VolumesA": int(a_nvols),
        "VolumesB": int(b_nvols),
        "Configuration": str(topup_config),
    }
    fieldcoef = topup_dir / "topup_results_fieldcoef.nii.gz"
    normalized_dir = topup_dir / "normalized"
    normalized_dfout = normalized_dir / "WarpField"
    normalized_jacout = normalized_dir / "Jacobian"
    normalized_rbmout = normalized_dir / "MotionMatrix"
    normalized_warps = tuple(
        normalized_dir / f"WarpField_{index:04d}.nii.gz"
        for index in range(1, total_nvols + 1)
    )
    normalized_jacobians = tuple(
        normalized_dir / f"Jacobian_{index:04d}.nii.gz"
        for index in range(1, total_nvols + 1)
    )
    normalized_matrices = tuple(
        normalized_dir / f"MotionMatrix_{index:04d}.mat"
        for index in range(1, total_nvols + 1)
    )
    topup_cmd = [
        "topup",
        f"--imain={merged}",
        f"--datain={datain}",
        f"--config={topup_config}",
        f"--out={out_prefix}",
        f"--iout={iout}",
        f"--fout={fout}",
        f"--dfout={dfout_prefix}",
        f"--jacout={jacout_prefix}",
        f"--rbmout={rbmout_prefix}",
    ]

    def execute_topup_directory() -> None:
        ensure_directory(topup_dir)
        write_json(spec_path, spec)
        run_child(
            ["fslmerge", "-t", str(merged), str(se_a), str(se_b)],
            env=env,
        )
        _write_topup_datain(
            out_txt=datain,
            ped_a=ped_a,
            ped_b=ped_b,
            readout_time=float(readout_time),
            readout_time_b=readout_time_b,
            a_nvols=int(a_nvols),
            b_nvols=int(b_nvols),
        )
        LOG.info("SDC: running topup (dfout/jacout) in %s", topup_dir)
        run_child(topup_cmd, env=env)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        for index, destination in enumerate(normalized_warps, start=1):
            shutil.copy2(
                _raw_topup_member(dfout_prefix, index_1based=index, extension=".nii.gz"),
                destination,
            )
        for index, destination in enumerate(normalized_jacobians, start=1):
            shutil.copy2(
                _raw_topup_member(jacout_prefix, index_1based=index, extension=".nii.gz"),
                destination,
            )
        for index, destination in enumerate(normalized_matrices, start=1):
            shutil.copy2(
                _raw_topup_member(rbmout_prefix, index_1based=index, extension=".mat"),
                destination,
            )
        write_json(
            result_manifest,
            {
                "spec": spec,
                "field_hz": str(fout),
                "iout": str(iout),
                "dfout": str(normalized_warps[0]),
                "jacout": str(normalized_jacobians[0]),
                "rbmout": str(normalized_rbmout),
                "warps": [str(path) for path in normalized_warps],
                "jacobians": [str(path) for path in normalized_jacobians],
                "motion_matrices": [str(path) for path in normalized_matrices],
            },
        )

    def validate_topup_directory() -> tuple[bool, str]:
        required = (
            spec_path,
            result_manifest,
            fieldcoef,
            iout,
            fout,
            *normalized_warps,
            *normalized_jacobians,
            *normalized_matrices,
        )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            return False, "TOPUP directory is missing required output(s): " + ", ".join(missing)
        try:
            recorded = read_json(result_manifest)
        except Exception:
            return False, f"TOPUP output inventory is unreadable: {result_manifest}"
        if recorded.get("spec") != spec:
            return False, "TOPUP output inventory does not match the current specification."
        return True, f"TOPUP directory contains the fixed {total_nvols}-volume output inventory."

    return TopupDfOutputs(
        step=Step.directory_step(
            name="TOPUP Distortion Estimation Directory",
            directory=topup_dir,
            breadcrumb=topup_complete,
            outputs=(result_manifest,),
            inputs=(se_a, se_b),
            force=force,
            action=execute_topup_directory,
            validate=validate_topup_directory,
            breadcrumb_text=f"volumes={total_nvols}\n",
        ),
        out_prefix=out_prefix,
        field_hz=fout,
        iout=iout,
        dfout=normalized_warps[0],
        jacout=normalized_jacobians[0],
        rbmout=normalized_rbmout,
        a_nvols=int(a_nvols),
        b_nvols=int(b_nvols),
    )


def _create_convertwarp_conjugate_affine_step(
    *,
    ref: Path,
    warp1: Path,
    premat: Path,
    postmat: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--premat={premat}",
        f"--warp1={warp1}",
        f"--postmat={postmat}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(ref, warp1, premat, postmat),
        force=force, env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_postmat_step(
    *,
    ref: Path,
    warp1: Path,
    postmat: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--warp1={warp1}",
        f"--postmat={postmat}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(ref, warp1, postmat), force=force,
        env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_premat_step(
    *,
    ref: Path,
    premat: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--premat={premat}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(ref, premat), force=force,
        env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_premat_and_warp_step(
    *,
    ref: Path,
    premat: Path,
    warp1: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--premat={premat}",
        f"--warp1={warp1}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(ref, premat, warp1), force=force,
        env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_merge_warps_step(
    *,
    ref: Path,
    warp1: Path,
    warp2: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--warp1={warp1}",
        f"--warp2={warp2}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(ref, warp1, warp2), force=force,
        env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_warp_jacobian_step(
    *, warp: Path, ref: Path, temporary: Path, junk: Path,
    env: dict[str, str], force: bool,
) -> Step:
    convert_cmd = [
        "convertwarp",
        "--rel",
        f"-w",
        str(warp),
        "-r",
        str(ref),
    ]
    convert_cmd.extend([f"--jacobian={temporary}", "-o", str(junk)])
    return Step.command_step(
        convert_cmd,
        name="Compute Full Warp Jacobian",
        outputs=(temporary, junk),
        inputs=(warp, ref),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(temporary.parent),
    )


def _create_average_jacobian_step(
    *, temporary: Path, junk: Path, warp: Path, ref: Path, out_jac: Path,
    env: dict[str, str], force: bool,
) -> Step:
    """Create the HCP-style mean over convertwarp's eight Jacobian volumes."""
    return Step.command_step(
        ["fslmaths", str(temporary), "-Tmean", str(out_jac)],
        name="Average Warp Jacobian Components",
        outputs=(out_jac,),
        inputs=(temporary, warp, ref),
        force=force,
        env=env,
        finalize=lambda: (temporary.unlink(missing_ok=True), junk.unlink(missing_ok=True)),
    )


def _create_applywarp_step(
    *,
    in_img: Path,
    ref_img: Path,
    warp: Path,
    out_img: Path,
    env: dict[str, str],
    force: bool,
    premat: Optional[Path] = None,
) -> Step:
    deps: list[Optional[Path]] = [in_img, ref_img, warp, premat]
    cmd = ["applywarp", "--rel", "--interp=spline", f"--in={in_img}", f"--ref={ref_img}", f"--warp={warp}", f"--out={out_img}"]
    if premat is not None:
        cmd.append(f"--premat={premat}")
    return Step.command_step(
        cmd, outputs=(out_img,), inputs=tuple(deps), force=force, env=env,
        prepare=lambda: ensure_directory(out_img.parent),
    )


def _create_temporal_mean_step(
    *,
    in_4d: Path,
    out_3d: Path,
    env: dict[str, str],
    force: bool,
    chunk_vols: int,
) -> Step:
    def calculate_mean() -> None:
        ensure_directory(out_3d.parent)
        import numpy as np  # type: ignore
        import nibabel as nib  # type: ignore

        img = nib.load(str(in_4d))
        if len(img.shape) != 4:
            raise RuntimeError(f"Temporal mean requires a 4D input, got shape {img.shape} for {in_4d}")
        nvols = int(img.shape[3])
        if nvols < 1:
            raise RuntimeError(f"Temporal mean requires at least one volume: {in_4d}")
        block_size = max(1, int(chunk_vols))
        accum = np.zeros(tuple(int(v) for v in img.shape[:3]), dtype=np.float32)
        # Materialize a compressed NIfTI exactly once. Proxy slicing a .nii.gz
        # for each block can restart decompression and multiply disk reads by
        # the number of temporal blocks. Retain the existing blockwise float32
        # summation order after the one-time load for numerical reproducibility.
        data = np.asarray(img.dataobj, dtype=np.float32)
        for start in range(0, nvols, block_size):
            stop = min(start + block_size, nvols)
            block = data[..., start:stop]
            accum += block.sum(axis=3, dtype=np.float32)
        accum /= float(nvols)
        header = img.header.copy()
        header.set_data_shape(accum.shape)
        header.set_data_dtype(np.float32)
        with atomic_output_path(out_3d) as staged:
            nib.save(nib.Nifti1Image(accum, img.affine, header), str(staged))
            if not nifti_is_valid(staged):
                raise RuntimeError(f"Temporal mean output is unreadable: {staged}")

    return Step.python(
        name="Temporal Mean Image",
        outputs=(out_3d,),
        inputs=(in_4d,),
        force=force,
        action=calculate_mean,
    )


def _create_afni_motion_affines_step(
    *,
    in_4d: Path,
    motion_ref_3d: Path,
    mc_mat_dir: Path,
    out_affines: Path,
    force: bool,
) -> Step:
    return Step.python(
        name="Consolidate Motion Transforms for 4D Resampling",
        outputs=(out_affines,),
        inputs=(in_4d, motion_ref_3d, _motion_matrix_breadcrumb(mc_mat_dir)),
        force=force,
        action=lambda: write_afni_motion_affines(
            source_path=in_4d,
            motion_reference_path=motion_ref_3d,
            matrix_dir=mc_mat_dir,
            output_path=out_affines,
        ),
        validate=lambda: validate_afni_motion_affines(
            source_path=in_4d,
            affine_path=out_affines,
        ),
    )


def _create_world_warp_step(
    *,
    run_child: Callable[..., Optional[str]],
    motion_ref_3d: Path,
    ref_3d: Path,
    fnirt_warp: Path,
    world_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    def convert_to_world() -> None:
        with atomic_output_path(world_warp) as staged:
            run_child(
                [
                    "wb_command",
                    "-convert-warpfield",
                    "-from-fnirt",
                    str(fnirt_warp),
                    str(motion_ref_3d),
                    "-to-world",
                    str(staged),
                ],
                env=env,
            )
            if not nifti_is_valid(staged):
                raise RuntimeError(f"World-coordinate warp is unreadable: {staged}")

    return Step.python(
        name=f"Convert Spatial Warp to World Coordinates ({_space_name_for_log(ref_3d)})",
        outputs=(world_warp,),
        inputs=(fnirt_warp, motion_ref_3d, ref_3d),
        force=force,
        action=convert_to_world,
        validate=lambda: (
            nifti_is_valid(world_warp),
            f"World-coordinate warp is unreadable: {world_warp}",
        ),
    )


def _create_afni_warp_step(
    *, world_warp: Path, motion_ref_3d: Path, ref_3d: Path,
    afni_warp: Path, force: bool,
) -> Step:
    return Step.python(
        name=f"Convert Spatial Warp for AFNI ({_space_name_for_log(ref_3d)})",
        outputs=(afni_warp,),
        inputs=(world_warp, motion_ref_3d, ref_3d),
        force=force,
        action=lambda: write_afni_warp(
            world_warp_path=world_warp,
            source_reference_path=motion_ref_3d,
            target_reference_path=ref_3d,
            output_path=afni_warp,
        ),
        validate=lambda: (
            nifti_is_valid(afni_warp),
            f"AFNI displacement field is unreadable: {afni_warp}",
        ),
    )


def _create_afni_bold_resampling_step(
    *,
    run_child: Callable[..., Optional[str]],
    in_4d: Path,
    motion_ref_3d: Path,
    ref_3d: Path,
    afni_warp: Path,
    motion_affines: Path,
    out_4d: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    step_name = f"Resample BOLD ({_space_name_for_log(out_4d)})"

    def resample() -> None:
        with atomic_output_path(out_4d) as staged:
            run_child(
                [
                    "3dNwarpApply",
                    "-source",
                    str(in_4d),
                    "-master",
                    str(ref_3d),
                    "-nwarp",
                    f"{afni_warp} {motion_affines}",
                    "-interp",
                    FINAL_WARP_INTERPOLATION,
                    "-ainterp",
                    FINAL_RESAMPLING_INTERPOLATION,
                    "-prefix",
                    str(staged),
                    "-quiet",
                ],
                env=env,
            )
            run_child(
                ["fslcpgeom", str(ref_3d), str(staged), "-d"],
                env=env,
            )
            valid, validation_reason = validate_resampled_bold(
                source_path=in_4d,
                reference_path=ref_3d,
                output_path=staged,
            )
            if not valid:
                raise RuntimeError(validation_reason)

    return Step.python(
        name=step_name,
        outputs=(out_4d,),
        inputs=(in_4d, ref_3d, afni_warp, motion_affines),
        force=force,
        action=resample,
        validate=lambda: validate_resampled_bold(
            source_path=in_4d,
            reference_path=ref_3d,
            output_path=out_4d,
        ),
    )


def _create_wb_volume_to_surface_mapping_step(
    *,
    volume: Path,
    midthickness: Path,
    white: Path,
    pial: Path,
    out_metric: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "wb_command",
        "-volume-to-surface-mapping",
        str(volume),
        str(midthickness),
        str(out_metric),
        "-ribbon-constrained",
        str(white),
        str(pial),
    ]
    return Step.command_step(
        cmd, outputs=(out_metric,), inputs=(volume, midthickness, white, pial),
        force=force, env=env, prepare=lambda: ensure_directory(out_metric.parent),
    )


def _create_wb_metric_resample_step(
    *,
    in_metric: Path,
    current_sphere: Path,
    new_sphere: Path,
    out_metric: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "wb_command",
        "-metric-resample",
        str(in_metric),
        str(current_sphere),
        str(new_sphere),
        "BARYCENTRIC",
        str(out_metric),
    ]
    return Step.command_step(
        cmd, outputs=(out_metric,), inputs=(in_metric, current_sphere, new_sphere),
        force=force, env=env, prepare=lambda: ensure_directory(out_metric.parent),
    )


def _create_confounds_step(
    *,
    epi_4d: Path,
    epi_mean_3d: Path,
    mc_dir: Path,
    subjects_dir: Path,
    fs_subject: str,
    brain_mask_in_epi: Path,
    out_tsv: Path,
    out_json: Path,
    force: bool,
) -> Step:
    par = mc_dir / "motion.par"
    cmd = [
        "get_confounds",
        "--epi",
        str(epi_4d),
        "--epi-mean",
        str(epi_mean_3d),
        "--mcflirt-par",
        str(par),
        "--brain-mask",
        str(brain_mask_in_epi),
        "--out-tsv",
        str(out_tsv),
        "--out-json",
        str(out_json),
    ]
    def calculate() -> None:
        ensure_directory(out_tsv.parent)
        LOG.info("Cmd: %s", Runner._format_cmd(cmd))
        get_confounds(
            epi=epi_4d,
            epi_mean=epi_mean_3d,
            mcflirt_par=par,
            subjects_dir=subjects_dir,
            fs_subject=fs_subject,
            brain_mask_in_epi=brain_mask_in_epi,
            out_tsv=out_tsv,
            out_json=out_json,
        )
    return Step.python(
        name="Confounds",
        outputs=(out_tsv, out_json),
        inputs=(epi_4d, epi_mean_3d, par, brain_mask_in_epi),
        force=force,
        action=calculate,
    )
def _resolve_ica_aroma_cmd(
    runner: Runner,
    env: dict[str, str],
    *,
    configured_cmd: Optional[Path],
) -> Optional[str]:
    if configured_cmd is not None:
        if runner_path_exists(runner, env, configured_cmd):
            return str(configured_cmd)
        raise SystemExit(f"Configured ICA-AROMA command does not exist: {configured_cmd}")
    return resolve_runner_command(runner, env, ["ICA_AROMA.py", "ica_aroma.py"])


def _resolve_container_command_for_wrapper(
    *,
    runner: Runner,
    env: dict[str, str],
    command: str,
) -> Optional[str]:
    resolved = resolve_runner_command(runner, env, [command])
    if resolved is None:
        return None
    resolved_path = Path(resolved)
    versioned_out = runner.run_out(
        [
            "bash",
            "-lc",
            (
                f"find /opt/fsl -maxdepth 3 -type f -path '*/bin/{command}' 2>/dev/null | "
                "sort -r || true"
            ),
        ],
        env=env,
        quiet=True,
    )
    versioned_candidates: list[Path] = []
    for raw_line in _strip_ansi(versioned_out).splitlines():
        line = raw_line.strip()
        if not line.startswith("/"):
            continue
        candidate = Path(line)
        versioned_candidates.append(candidate)
    for candidate in versioned_candidates:
        candidate_str = str(candidate)
        if f"/fsl-6." in candidate_str or re.search(r"/fsl-[0-9][^/]*/bin/", candidate_str):
            if runner_path_exists(runner, env, candidate):
                return candidate_str

    probe = runner.run_out(
        [
            "bash",
            "-lc",
            (
                f"if [ -f {shlex_quote(str(resolved_path))} ]; then "
                f"sed -n '1,5p' {shlex_quote(str(resolved_path))}; "
                "fi"
            ),
        ],
        env=env,
        quiet=True,
    )
    cleaned = _strip_ansi(probe)
    pattern = re.compile(rf"(/[^\"' \t]+/{re.escape(command)})")
    for line in cleaned.splitlines():
        for match in pattern.finditer(line):
            candidate = Path(match.group(1))
            if candidate != resolved_path and runner_path_exists(runner, env, candidate):
                return str(candidate)
    for candidate in versioned_candidates:
        if candidate != resolved_path and runner_path_exists(runner, env, candidate):
            return str(candidate)
    return str(resolved_path)


def _resolve_ica_aroma_fsl_commands(runner: Runner, env: dict[str, str]) -> dict[str, str]:
    commands = {
        "applywarp": None,
        "bet": None,
        "flirt": None,
        "fsl_regfilt": None,
        "fslinfo": None,
        "fslmaths": None,
        "fslmerge": None,
        "fslroi": None,
        "fslstats": None,
        "melodic": None,
    }
    resolved: dict[str, str] = {}
    for name in commands:
        path = _resolve_container_command_for_wrapper(runner=runner, env=env, command=name)
        if path is None:
            raise SystemExit(f"Missing required FSL command for ICA-AROMA: {name}")
        resolved[name] = path
    return resolved


def _ica_aroma_policy_payload(
    *,
    input_is_mni: bool,
    denoise_type: str,
    repetition_time: Optional[float],
    external_aroma: bool,
) -> dict[str, object]:
    return {
        "version": _ICA_AROMA_ESTIMATION_POLICY_VERSION,
        "input_space": "MNI152NLin2009cAsym" if input_is_mni else "T1w",
        "workflow": "external" if external_aroma else "internal",
        "denoise_type": str(denoise_type).strip().lower(),
        "repetition_time": (float(repetition_time) if repetition_time is not None else None),
        "melodic_estimation_smoothing_fwhm_mm": _ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM,
        "melodic_mask_dilation_mm": _ICA_AROMA_MELODIC_MASK_DILATION_MM,
        "regression_mask_dilation_mm": _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
        "epi_support_bet_fractional_intensity_threshold": _ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD,
        "shared_across_output_spaces": True,
    }


def _ica_aroma_shared_regression_policy_payload(
    *,
    input_space: str,
    denoise_type: str,
    repetition_time: Optional[float],
    shared_work_dir: Path,
) -> dict[str, object]:
    return {
        "version": _ICA_AROMA_ESTIMATION_POLICY_VERSION,
        "input_space": str(input_space),
        "workflow": "shared-t1w-component-regression",
        "denoise_type": str(denoise_type).strip().lower(),
        "repetition_time": (float(repetition_time) if repetition_time is not None else None),
        "estimation_space": "T1w",
        "shared_estimation_work_dir": str(shared_work_dir),
        "regression_mask_dilation_mm": _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
        "epi_support_bet_fractional_intensity_threshold": _ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD,
        "shared_across_output_spaces": True,
    }


def _create_epi_support_step(
    *, epi_mean: Path, support_brain: Path, support_mask: Path,
    work_dir: Path, env: dict[str, str], force: bool,
) -> Step:
    return Step.command_step(
        [
            "bet", str(epi_mean), str(support_brain), "-R", "-f",
            f"{_ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD:.8g}",
            "-g", "0", "-m",
        ],
        outputs=(support_brain, support_mask),
        inputs=(epi_mean,),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(work_dir),
    )


def _create_dilated_anatomical_mask_step(
    *, anatomical_mask: Path, support_mask: Path, output: Path,
    dilation_mm: float, role: str, force: bool,
) -> Step:
    command = [
        "dilate-anatomical-mask",
        f"--radius-mm={dilation_mm:.8g}",
        f"--support={support_mask}",
        f"--out={output}",
    ]

    def construct() -> None:
        LOG.info("Cmd: %s", Runner._format_cmd(command))
        make_dilated_anatomical_epi_mask(
            anatomical_mask=anatomical_mask,
            epi_support_mask=support_mask,
            dilation_mm=dilation_mm,
            out_mask=output,
        )

    return Step.python(
        name=f"Construct {role} Mask",
        outputs=(output,),
        inputs=(anatomical_mask, support_mask),
        force=force,
        action=construct,
    )


def _create_melodic_smoothing_step(
    *, epi: Path, output: Path, env: dict[str, str], force: bool,
) -> Step:
    sigma = _ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM / 2.3548200450309493
    return Step.command_step(
        ["fslmaths", str(epi), "-s", f"{sigma:.8g}", str(output)],
        name="Prepare Smoothed MELODIC Input",
        outputs=(output,),
        inputs=(epi,),
        force=force,
        env=env,
    )


def _create_ica_aroma_workflow_step(
    *,
    runner: Runner,
    epi: Path,
    melodic_input: Path,
    motion_parameters: Path,
    melodic_mask: Path,
    regression_mask: Path,
    aroma_dir: Path,
    outputs: tuple[Path, ...],
    melodic_products: tuple[Path, ...],
    input_is_mni: bool,
    identity_transform: Path,
    t1_to_mni_warp: Optional[Path],
    mni_reference: Path,
    configured_command: Optional[Path],
    repetition_time: Optional[float],
    denoise_type: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    def execute() -> None:
        aroma_command = (
            _resolve_ica_aroma_cmd(runner, env, configured_cmd=configured_command)
            if configured_command is not None
            else None
        )
        fsl_commands = _resolve_ica_aroma_fsl_commands(runner, env)
        if aroma_command is None:
            if repetition_time is None or repetition_time <= 0:
                raise SystemExit("ICA-AROMA requires a positive RepetitionTime in the EPI JSON.")
            run_ica_aroma_workflow(
                run_cmd=lambda cmd: runner.run_child(list(cmd), env=env),
                run_out=lambda cmd: runner.run_child(
                    list(cmd), env=env, capture_stdout=True
                ) or "",
                fsl_cmds=fsl_commands,
                in_file=epi,
                melodic_in_file=melodic_input,
                out_dir=aroma_dir,
                mc=motion_parameters,
                affmat=None if input_is_mni else identity_transform,
                warp=t1_to_mni_warp,
                mask=melodic_mask,
                regression_mask=regression_mask,
                tr=float(repetition_time),
                denoise_type=denoise_type,
                mni_ref=mni_reference,
                overwrite=True,
                melodic_dir=None,
            )
            return

        arguments = [
            "-in", str(melodic_input), "-out", str(aroma_dir),
            "-mc", str(motion_parameters), "-m", str(melodic_mask),
            "-den", "no",
        ]
        if not input_is_mni:
            arguments.extend(["-affmat", str(identity_transform), "-warp", str(t1_to_mni_warp)])
        if repetition_time is not None and repetition_time > 0:
            arguments.extend(["-tr", f"{float(repetition_time):.8g}"])
        runner.run_child([aroma_command, *arguments], env=env)
        for product in melodic_products[:-1]:
            if not product.is_file() or product.stat().st_size == 0:
                raise SystemExit(
                    f"ICA-AROMA completed without required MELODIC product: {product}"
                )
        write_completion_breadcrumb(
            melodic_products[-1], "MELODIC decomposition complete\n"
        )
        classified = aroma_dir / "classified_motion_ICs.txt"
        indices = [
            int(value) - 1
            for value in classified.read_text(encoding="utf-8").strip().split(",")
            if value.strip()
        ]
        run_ica_aroma_denoising(
            run_cmd=lambda cmd: runner.run_child(list(cmd), env=env),
            fsl_cmds=fsl_commands,
            in_file=epi,
            mask=regression_mask,
            out_dir=aroma_dir,
            melmix=aroma_dir / "melodic.ica" / "melodic_mix",
            denoise_type=denoise_type,
            denoise_indices=indices,
        )

    def validate() -> tuple[bool, str]:
        missing = [
            str(path) for path in outputs
            if not path.is_file() or path.stat().st_size == 0
        ]
        return (
            not missing,
            "ICA-AROMA directory contains all required products."
            if not missing
            else "ICA-AROMA directory is incomplete: " + ", ".join(missing),
        )

    inputs = [epi, melodic_input, motion_parameters, melodic_mask, regression_mask]
    if not input_is_mni:
        inputs.extend((identity_transform, t1_to_mni_warp))
    return Step.directory_step(
        name="Run ICA-AROMA Classification and Denoising",
        directory=aroma_dir,
        breadcrumb=aroma_dir / ".nro_complete",
        outputs=outputs,
        inputs=inputs,
        force=force,
        action=execute,
        validate=validate,
        breadcrumb_text="ICA-AROMA workflow complete\n",
    )


def _create_shared_aroma_regression_step(
    *,
    runner: Runner,
    epi: Path,
    input_space: str,
    regression_mask: Path,
    mixing_matrix: Path,
    classified_components: Path,
    shared_policy: Path,
    aroma_dir: Path,
    outputs: tuple[Path, ...],
    denoise_type: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    def regress() -> None:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore

        n_timepoints = int(nib.load(str(epi)).shape[3])
        mixing = np.loadtxt(mixing_matrix, ndmin=2)
        if int(mixing.shape[0]) != n_timepoints:
            raise SystemExit(
                "Shared T1w ICA-AROMA mixing matrix does not match registered "
                f"BOLD length: {mixing.shape[0]} != {n_timepoints}"
            )
        indices = [
            int(value)
            for value in classified_components.read_text(encoding="utf-8").strip().split(",")
            if value.strip()
        ]
        denoise_indices = np.asarray([value - 1 for value in indices], dtype=int)
        fsl_regfilt = _resolve_container_command_for_wrapper(
            runner=runner, env=env, command="fsl_regfilt"
        )
        if fsl_regfilt is None:
            raise SystemExit(
                "Missing required FSL command for shared ICA-AROMA regression: fsl_regfilt"
            )
        LOG.info(
            "Applying %d T1w-classified ICA-AROMA noise components to %s registered BOLD",
            len(indices), input_space,
        )
        aroma_dir.mkdir(parents=True, exist_ok=True)
        run_ica_aroma_denoising(
            run_cmd=lambda cmd: runner.run_child(list(cmd), env=env),
            fsl_cmds={"fsl_regfilt": fsl_regfilt},
            in_file=epi,
            mask=regression_mask,
            out_dir=aroma_dir,
            melmix=mixing_matrix,
            denoise_type=denoise_type,
            denoise_indices=denoise_indices,
        )

    return Step.python(
        name=f"Regress Shared ICA-AROMA Components in {input_space}",
        outputs=outputs,
        inputs=(
            epi, regression_mask, mixing_matrix,
            classified_components, shared_policy,
        ),
        force=force,
        action=regress,
    )






def _create_t1_epi_vox_target_step(
    *,
    t1_image: Path,
    source_epi: Path,
    out_target: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Create a T1w target with the source BOLD voxel dimensions.

    The source BOLD is a BIDS input available while the DAG is constructed.
    Motion correction preserves its voxel grid, so consulting a future
    motion-corrected artifact here would add no information and would make DAG
    construction incorrectly depend on runtime output.
    """
    vx, vy, vz = nifti_zooms_xyz(source_epi)
    cmd = ["mri_convert", "--voxsize", f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}", str(t1_image), str(out_target)]
    return Step.command_step(
        cmd,
        name="Prepare T1 Target at EPI Resolution",
        outputs=(out_target,),
        inputs=(t1_image, source_epi),
        force=force,
        env=env,
        prepare=lambda: (ensure_directory(out_target.parent), out_target.unlink(missing_ok=True)),
        validate=lambda: (
            nifti_is_valid(out_target),
            f"T1 EPI-resolution target is not a readable NIfTI: {out_target}",
        ),
    )


def _create_t1_native_target_step(
    *,
    t1_image: Path,
    out_target: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    same_suffix = "".join(t1_image.suffixes) == "".join(out_target.suffixes)
    if same_suffix:
        cmd = ["cp", str(t1_image), str(out_target)]
    else:
        cmd = ["mri_convert", str(t1_image), str(out_target)]
    validate = lambda: (
        nifti_is_valid(out_target),
        f"T1 native target is not a readable NIfTI: {out_target}",
    )
    if same_suffix:
        def copy_target() -> None:
            ensure_directory(out_target.parent)
            shutil.copyfile(t1_image, out_target)

        return Step.python(
            name="Prepare T1 Native Target",
            outputs=(out_target,),
            inputs=(t1_image,),
            force=force,
            action=copy_target,
            validate=validate,
        )
    return Step.command_step(
        cmd,
        name="Prepare T1 Native Target",
        outputs=(out_target,),
        inputs=(t1_image,),
        force=force,
        env=env,
        prepare=lambda: (ensure_directory(out_target.parent), out_target.unlink(missing_ok=True)),
        validate=validate,
    )


def _create_mni_2mm_target_step(
    *,
    mni_template: Path,
    out_target: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "flirt",
        "-in",
        str(mni_template),
        "-ref",
        str(mni_template),
        "-applyisoxfm",
        "2",
        "-interp",
        "trilinear",
        "-out",
        str(out_target),
    ]
    return Step.command_step(
        cmd, outputs=(out_target,), inputs=(mni_template,), force=force, env=env,
        prepare=lambda: ensure_directory(out_target.parent),
    )


@dataclass(frozen=True)
class _ANTsRegistrationStep:
    step: Step
    warped: Path
    forward_transform: Path
    inverse_transform: Path


def _create_ants_registration_step(
    *,
    run_child: Callable[..., Optional[str]],
    moving_img: Path,
    fixed_img: Path,
    work_dir: Path,
    out_prefix: str,
    env: dict[str, str],
    force: bool,
    include_linear: bool = True,
    write_composite: bool = True,
    fixed_mask: Optional[Path] = None,
    moving_mask: Optional[Path] = None,
    syn_transform: str = "SyN[0.08,3,0]",
    syn_convergence: str = "[140x110x80x50,1e-6,10]",
    syn_shrink_factors: str = "8x4x2x1",
    syn_smoothing_sigmas: str = "3x2x1x0vox",
    restrict_deformation: Optional[str] = None,
) -> _ANTsRegistrationStep:
    artifact_dir = work_dir / f"{out_prefix.rstrip('_')}_ants"
    prefix = artifact_dir / "raw_"
    warped = artifact_dir / "raw_Warped.nii.gz"
    breadcrumb = artifact_dir / ".nro_complete"
    if write_composite:
        forward_xfm = artifact_dir / "raw_Composite.h5"
        inverse_xfm = artifact_dir / "raw_InverseComposite.h5"
    else:
        forward_xfm = artifact_dir / "ForwardWarp.nii.gz"
        inverse_xfm = artifact_dir / "InverseWarp.nii.gz"

    cmd = [
        "antsRegistration",
        "--dimensionality",
        "3",
        "--float",
        "0",
        "--collapse-output-transforms",
        "1",
        "--write-composite-transform",
        "1" if write_composite else "0",
        "--output",
        f"[{prefix}]",
        "--interpolation",
        "LanczosWindowedSinc",
        "--use-histogram-matching",
        "0",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
    ]
    if fixed_mask is not None or moving_mask is not None:
        fixed_mask_arg = str(fixed_mask) if fixed_mask is not None else "NULL"
        moving_mask_arg = str(moving_mask) if moving_mask is not None else "NULL"
        cmd.extend(["--masks", f"[{fixed_mask_arg},{moving_mask_arg}]"])
    if include_linear:
        cmd.extend(
            [
                "--initial-moving-transform",
                f"[{fixed_img},{moving_img},1]",
                "--transform",
                "Rigid[0.1]",
                "--metric",
                f"MI[{fixed_img},{moving_img},1,32,Regular,0.25]",
                "--convergence",
                "[1000x500x250x0,1e-6,10]",
                "--shrink-factors",
                "8x4x2x1",
                "--smoothing-sigmas",
                "3x2x1x0vox",
                "--transform",
                "Affine[0.1]",
                "--metric",
                f"MI[{fixed_img},{moving_img},1,32,Regular,0.25]",
                "--convergence",
                "[1000x500x250x0,1e-6,10]",
                "--shrink-factors",
                "8x4x2x1",
                "--smoothing-sigmas",
                "3x2x1x0vox",
            ]
        )
    cmd.extend(
        [
            "--transform",
            syn_transform,
            "--metric",
            f"MI[{fixed_img},{moving_img},1,32,Regular,0.25]",
            "--convergence",
            syn_convergence,
            "--shrink-factors",
            syn_shrink_factors,
            "--smoothing-sigmas",
            syn_smoothing_sigmas,
        ]
    )
    if restrict_deformation is not None:
        cmd.extend(["--restrict-deformation", restrict_deformation])

    def execute_registration() -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run_child(cmd, env=env)
        if write_composite:
            return
        raw_forward = next(
            (path for path in (artifact_dir / "raw_1Warp.nii.gz", artifact_dir / "raw_0Warp.nii.gz") if path.is_file()),
            None,
        )
        raw_inverse = next(
            (path for path in (artifact_dir / "raw_1InverseWarp.nii.gz", artifact_dir / "raw_0InverseWarp.nii.gz") if path.is_file()),
            None,
        )
        if raw_forward is None or raw_inverse is None:
            raise SystemExit(f"ANTs registration did not produce forward and inverse warps under {artifact_dir}")
        shutil.copy2(raw_forward, forward_xfm)
        shutil.copy2(raw_inverse, inverse_xfm)

    def validate_registration() -> tuple[bool, str]:
        missing = [
            str(path)
            for path in (forward_xfm, inverse_xfm)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "ANTs SyN directory is missing required transform(s): " + ", ".join(missing)
        return True, "ANTs SyN directory contains its fixed transform outputs."

    return _ANTsRegistrationStep(
        step=Step.directory_step(
            name="ANTs SyN Registration",
            directory=artifact_dir,
            breadcrumb=breadcrumb,
            outputs=(forward_xfm, inverse_xfm),
            inputs=(moving_img, fixed_img, fixed_mask, moving_mask),
            force=force,
            action=execute_registration,
            validate=validate_registration,
            breadcrumb_text="ANTs SyN registration complete\n",
        ),
        warped=warped,
        forward_transform=forward_xfm,
        inverse_transform=inverse_xfm,
    )


def _create_ants_synboldaff_step(
    *,
    moving_img: Path,
    fixed_img: Path,
    work_dir: Path,
    out_prefix: str,
    env: dict[str, str],
    force: bool,
    fixed_mask: Optional[Path] = None,
    moving_mask: Optional[Path] = None,
    syn_transform: str = "SyN[0.2,3,0]",
    syn_convergence: str = "[40x20x0,1e-7,8]",
    syn_shrink_factors: str = "4x2x1",
    syn_smoothing_sigmas: str = "2x1x0vox",
) -> _ANTsRegistrationStep:
    prefix = work_dir / out_prefix
    warped = work_dir / f"{out_prefix}Warped.nii.gz"
    forward_xfm = work_dir / f"{out_prefix}Composite.h5"
    inverse_xfm = work_dir / f"{out_prefix}InverseComposite.h5"
    cmd = [
        "antsRegistration",
        "--dimensionality",
        "3",
        "--float",
        "0",
        "--collapse-output-transforms",
        "1",
        "--write-composite-transform",
        "1",
        "--output",
        f"[{prefix}]",
        "--interpolation",
        "LanczosWindowedSinc",
        "--use-histogram-matching",
        "0",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--initial-moving-transform",
        f"[{fixed_img},{moving_img},1]",
        "--transform",
        "Rigid[0.25]",
        "--metric",
        f"MI[{fixed_img},{moving_img},1,32,Regular,0.2]",
        "--convergence",
        "[1200x1200x100,1e-6,5]",
        "--shrink-factors",
        "4x2x1",
        "--smoothing-sigmas",
        "2x1x0vox",
        "--transform",
        "Affine[0.25]",
        "--metric",
        f"MI[{fixed_img},{moving_img},1,32,Regular,0.2]",
        "--convergence",
        "[200x20,1e-6,5]",
        "--shrink-factors",
        "2x1",
        "--smoothing-sigmas",
        "1x0vox",
        "--transform",
        syn_transform,
        "--metric",
        f"MI[{fixed_img},{moving_img},1,32]",
        "--convergence",
        syn_convergence,
        "--shrink-factors",
        syn_shrink_factors,
        "--smoothing-sigmas",
        syn_smoothing_sigmas,
    ]
    if fixed_mask is not None or moving_mask is not None:
        fixed_mask_arg = str(fixed_mask) if fixed_mask is not None else "NULL"
        moving_mask_arg = str(moving_mask) if moving_mask is not None else "NULL"
        cmd.extend(["--masks", f"[{fixed_mask_arg},{moving_mask_arg}]"])
    return _ANTsRegistrationStep(
        step=Step.command_step(
            cmd,
            name="ANTs SyNBoldAff Registration",
            outputs=(forward_xfm, inverse_xfm),
            inputs=(moving_img, fixed_img, fixed_mask, moving_mask),
            force=force,
            env=env,
            prepare=lambda: ensure_directory(work_dir),
        ),
        warped=warped,
        forward_transform=forward_xfm,
        inverse_transform=inverse_xfm,
    )


def _create_ants_composite_to_itk_warp_step(
    *,
    composite_xfm: Path,
    ref_img: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    # This does not resample an anatomical or functional image. antsApplyTransforms
    # is used here only to materialize the composite transform as a displacement field
    # so the downstream single-interpolation BOLD path can compose it with FSL warps.
    cmd = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-r",
        str(ref_img),
        "-t",
        str(composite_xfm),
        "-o",
        f"[{out_warp},1]",
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(composite_xfm, ref_img), force=force,
        env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_wb_convert_itk_warp_to_fnirt_step(
    *,
    itk_warp: Path,
    src_space_ref: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "wb_command",
        "-convert-warpfield",
        "-from-itk",
        str(itk_warp),
        "-to-fnirt",
        str(out_warp),
        str(src_space_ref),
    ]
    return Step.command_step(
        cmd, outputs=(out_warp,), inputs=(itk_warp, src_space_ref), force=force,
        env=env, prepare=lambda: ensure_directory(out_warp.parent),
    )


@dataclass(frozen=True)
class Inputs:
    sbref: Optional[Path]
    epi: Path
    se1: Optional[Path]
    se2: Optional[Path]
    se1_json: Optional[Path] = None
    se2_json: Optional[Path] = None
    epi_json: Optional[Path] = None
    sbref_json: Optional[Path] = None
    epi_metadata: Optional[dict[str, Any]] = None
    epi_metadata_sources: tuple[Path, ...] = ()
    sbref_metadata: Optional[dict[str, Any]] = None
    sbref_metadata_sources: tuple[Path, ...] = ()
    se1_metadata: Optional[dict[str, Any]] = None
    se1_metadata_sources: tuple[Path, ...] = ()
    se2_metadata: Optional[dict[str, Any]] = None
    se2_metadata_sources: tuple[Path, ...] = ()
    sbref_metadata_inheritance: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Options:
    out_dir: Path
    work_dir: Path
    project: str
    preprocessing_id: str
    sub_id: str
    ses_id: Optional[str]
    nthreads: int
    force: bool
    output_grid: str
    topup_config: str
    ica_aroma_cmd: Optional[Path]
    use_jacobian: bool
    fieldmap_syn_refine: bool
    syn_base_transform: str
    syn_base_convergence: str
    syn_base_shrink_factors: str
    syn_base_smoothing_sigmas: str
    syn_refine_transform: str
    syn_refine_convergence: str
    syn_refine_shrink_factors: str
    syn_refine_smoothing_sigmas: str
    clean_ica_aroma: bool
    ica_aroma_denoise_type: str
    bbregister_surf: str
    bbregister_init: str
    bbregister_dof: int
    container: Optional[ContainerSpec]
    debug_first_nvols: int
    output_spaces: tuple[str, ...]
    io_chunk_vols: int
    sdc_method: str
    synbold_disco_image: Path
    synbold_disco_license: Path
    synbold_disco_engine: str
    synbold_overlap_erosion_voxels: int
    synbold_min_overlap_voxels: int
    synbold_max_rigid_translation_mm: float
    synbold_max_rigid_rotation_degrees: float
    sbref_max_rigid_displacement_mm: float
    sbref_max_rigid_rotation_degrees: float
    sbref_min_support_overlap: float
    sbref_min_intensity_correlation: float


def _functional_config_payload(opts: Options) -> dict[str, object]:
    """Canonical output-affecting configuration recorded by every new run."""
    return {
        "output_grid": opts.output_grid,
        "topup_config": opts.topup_config,
        "ica_aroma_cmd": str(opts.ica_aroma_cmd) if opts.ica_aroma_cmd else None,
        "use_jacobian": bool(opts.use_jacobian),
        "fieldmap_syn_refine": bool(opts.fieldmap_syn_refine),
        "syn_base_transform": opts.syn_base_transform,
        "syn_base_convergence": opts.syn_base_convergence,
        "syn_base_shrink_factors": opts.syn_base_shrink_factors,
        "syn_base_smoothing_sigmas": opts.syn_base_smoothing_sigmas,
        "syn_refine_transform": opts.syn_refine_transform,
        "syn_refine_convergence": opts.syn_refine_convergence,
        "syn_refine_shrink_factors": opts.syn_refine_shrink_factors,
        "syn_refine_smoothing_sigmas": opts.syn_refine_smoothing_sigmas,
        "clean_ica_aroma": bool(opts.clean_ica_aroma),
        "ica_aroma_denoise_type": opts.ica_aroma_denoise_type,
        "bbregister_surf": opts.bbregister_surf,
        "bbregister_init": opts.bbregister_init,
        "bbregister_dof": int(opts.bbregister_dof),
        "debug_first_nvols": int(opts.debug_first_nvols),
        "output_spaces": list(opts.output_spaces),
        "io_chunk_vols": int(opts.io_chunk_vols),
        "final_resampling": final_resampling_contract(),
        "sdc_method": opts.sdc_method,
        "synbold_disco_image": str(opts.synbold_disco_image),
        "synbold_disco_engine": opts.synbold_disco_engine,
        "synbold_overlap_erosion_voxels": int(opts.synbold_overlap_erosion_voxels),
        "synbold_min_overlap_voxels": int(opts.synbold_min_overlap_voxels),
        "synbold_max_rigid_translation_mm": float(opts.synbold_max_rigid_translation_mm),
        "synbold_max_rigid_rotation_degrees": float(opts.synbold_max_rigid_rotation_degrees),
        "sbref_max_rigid_displacement_mm": float(opts.sbref_max_rigid_displacement_mm),
        "sbref_max_rigid_rotation_degrees": float(opts.sbref_max_rigid_rotation_degrees),
        "sbref_min_support_overlap": float(opts.sbref_min_support_overlap),
        "sbref_min_intensity_correlation": float(opts.sbref_min_intensity_correlation),
    }


def _normalize_output_spaces(values: Sequence[str]) -> tuple[str, ...]:
    mapping = {
        "t1w": "T1w",
        "t1": "T1w",
        "fsnative": "fsnative",
        "mni": "MNI152NLin2009cAsym",
        "mni152nlin2009casym": "MNI152NLin2009cAsym",
        "fsaverage": "fsaverage",
    }
    out: list[str] = []
    for raw in values:
        key = str(raw).strip()
        if not key:
            continue
        canon = mapping.get(key.lower())
        if canon is None:
            raise SystemExit(
                "Unknown output space "
                f"{raw!r}. Expected one of: T1w, fsnative, MNI152NLin2009cAsym, fsaverage"
            )
        if canon not in out:
            out.append(canon)
    if not out:
        raise SystemExit("At least one output space must be requested.")
    return tuple(out)


def build_module(inputs: Inputs, opts: Options) -> Runner:
    """Resolve BIDS inputs and construct the complete functional DAG."""
    require_existing_path(inputs.epi, "epi")
    require_existing_path(inputs.epi_json, "epi-json")
    assert inputs.epi_json is not None
    epi_metadata_sources = inputs.epi_metadata_sources or (inputs.epi_json,)
    for source in epi_metadata_sources:
        require_existing_path(source, "EPI metadata source")
    epi_input_meta = (
        dict(inputs.epi_metadata)
        if inputs.epi_metadata is not None
        else read_json(inputs.epi_json)
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
    anat_manifest = anatomical_manifest_path(opts.sub_id, project=opts.project, preprocessing_id=opts.preprocessing_id)
    if not anat_manifest.exists():
        raise SystemExit(
            "Anatomical preprocessing must be completed before functional preprocessing.\n"
            f"Missing anatomical manifest: {anat_manifest}"
        )
    anat_info = read_json(anat_manifest)
    if not bool(anat_info.get("complete")):
        raise SystemExit(
            "Anatomical preprocessing must be completed before functional preprocessing.\n"
            f"Anatomical manifest is present but not marked complete: {anat_manifest}"
        )
    subjects_dir_raw = str(anat_info.get("freesurfer_subjects_dir", "")).strip()
    if not subjects_dir_raw:
        raise SystemExit(f"Anatomical manifest is missing freesurfer_subjects_dir: {anat_manifest}")
    subjects_dir = Path(subjects_dir_raw)
    fs_subject = str(anat_info.get("fs_subject") or opts.sub_id).strip()
    if not fs_subject:
        raise SystemExit(f"Anatomical manifest is missing fs_subject: {anat_manifest}")
    anat_t1 = require_manifest_output(
        anat_info,
        "subject_t1w",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    anat_brain_mask = require_manifest_output(
        anat_info,
        "brain_mask",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    anat_mni_template = Path(str(anat_info.get("mni_template", "")).strip())
    require_existing_path(anat_mni_template, "MNI template from anatomical manifest")
    t1_to_mni_xfm = require_nested_manifest_output(
        anat_info,
        "xfms",
        "t1_to_mni",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    mni_to_t1_xfm = require_nested_manifest_output(
        anat_info,
        "xfms",
        "mni_to_t1",
        manifest_path=anat_manifest,
        manifest_name="Anatomical",
    )
    fsnative_surfaces = {
        f"{hemi}.{surface}": require_nested_manifest_output(
            anat_info,
            "surfaces",
            f"{source_hemi}.{surface}",
            manifest_path=anat_manifest,
            manifest_name="Anatomical",
        )
        for hemi, source_hemi in (("L", "lh"), ("R", "rh"))
        for surface in ("white", "pial", "midthickness")
    }
    env = neuroimaging_environment(opts.nthreads, subjects_dir=subjects_dir)
    raw_repetition_time = epi_input_meta.get("RepetitionTime")
    try:
        repetition_time = (
            float(raw_repetition_time) if raw_repetition_time is not None else None
        )
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
            anat_brain_mask,
            opts.ica_aroma_cmd,
            (subjects_dir if subjects_dir.exists() else None),
            opts.out_dir,
            opts.work_dir,
        ]
    )
    runner = Runner(
        module_name="Functional Preprocessing Module",
        container=opts.container,
        binds=binds,
        logger=LOG,
        next_step=next_step,
    )
    initialized = opts.work_dir / "initialized.complete"

    def initialize_outputs() -> None:
        ensure_directory(
            preprocessing_derivatives_root(
                project=opts.project, preprocessing_id=opts.preprocessing_id
            )
        )
        ensure_directory(opts.out_dir)
        ensure_directory(opts.work_dir)
        write_completion_breadcrumb(initialized, "Functional outputs initialized\n")

    runner.add_step(
        Step.python(
            name="Initialize Functional Outputs",
            outputs=(initialized,),
            action=initialize_outputs,
        )
    )
    configuration = {
        "configuration": _functional_config_payload(opts),
        "configuration_fingerprint": selected_configuration_fingerprint(),
    }
    configuration_snapshot = opts.work_dir / "configuration.json"

    def validate_configuration() -> tuple[bool, str]:
        try:
            current = read_json(configuration_snapshot)
        except (OSError, ValueError, TypeError):
            return False, "Functional configuration snapshot is missing or unreadable."
        if current != configuration:
            return False, "Functional configuration changed."
        return True, "Functional configuration is unchanged."

    runner.add_step(
        Step.python(
            name="Write Functional Configuration",
            outputs=(configuration_snapshot,),
            inputs=(initialized,),
            force=opts.force,
            action=lambda: write_json(configuration_snapshot, configuration),
            validate=validate_configuration,
        )
    )
    runner.set_definition_inputs((configuration_snapshot,))
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
        raise SystemExit(
            f"Unknown SDC method {opts.sdc_method!r}; expected syn or synbold_disco."
        )
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
    if opts.clean_ica_aroma:
        base_cmds.extend(("bet", "fsl_regfilt", "fslinfo", "fslstats", "melodic"))
    dependency_check = opts.work_dir / "dependencies.complete"

    def check_dependencies() -> None:
        runner.require_cmds(
            base_cmds
            if use_syn_fallback
            else base_cmds + ["topup", "bbregister", "tkregister2"]
        )
        write_completion_breadcrumb(
            dependency_check, "Functional dependencies available\n"
        )

    runner.add_step(Step.python(
        name="Check Functional Dependencies",
        outputs=(dependency_check,),
        force=opts.force,
        action=check_dependencies,
    ))
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
    run_prefix = f"{run_base}_space-T1w"
    func_dir = opts.out_dir
    fmap_dir = func_dir.parent / "fmap"
    sdc_dir = opts.work_dir / "sdc"
    reg_dir = opts.work_dir / "reg"
    mc_dir = opts.work_dir / "mc"
    surf_dir = opts.work_dir / "surf"
    qc_dir = opts.work_dir / "qc"
    # Debug truncation must precede reference construction so the provisional
    # and final motion passes operate on exactly the series being processed.
    epi_for_proc = inputs.epi
    debug_epi: Optional[Path] = None
    if int(opts.debug_first_nvols) > 0:
        n = int(opts.debug_first_nvols)
        debug_dir = opts.work_dir / "debug"
        debug_epi = debug_dir / f"{run_stem}_first{n:04d}.nii.gz"
        debug_cmd = ["fslroi", str(inputs.epi), str(debug_epi), "0", str(n)]
        runner.add_step(Step.command_step(
            debug_cmd,
            name="Select Debug BOLD Volumes",
            outputs=(debug_epi,),
            inputs=(inputs.epi,),
            force=opts.force,
            env=env,
            prepare=lambda: ensure_directory(debug_dir),
        ))
        epi_for_proc = debug_epi

    robust_reference_step = _create_robust_bold_reference_step(
        run_child=runner.run_child,
        epi_in=epi_for_proc,
        run_stem=run_stem,
        mc_dir=mc_dir,
        env=env,
        force=opts.force,
    )
    runner.add_step(robust_reference_step.step)
    robust_ref = robust_reference_step.reference
    epi_mc = robust_reference_step.motion_corrected
    robust_reference_metadata = robust_reference_step.metadata
    selected_reference = _create_functional_reference_selection_step(
        run_child=runner.run_child,
        robust_ref=robust_ref,
        epi_metadata=epi_input_meta,
        epi_metadata_sources=epi_metadata_sources,
        sbref=inputs.sbref,
        sbref_json=inputs.sbref_json,
        sbref_metadata=inputs.sbref_metadata,
        sbref_metadata_sources=sbref_metadata_sources,
        sbref_metadata_inheritance=inputs.sbref_metadata_inheritance,
        work_dir=opts.work_dir / "reference" / "sbref_qc",
        env=env,
        max_rotation_degrees=opts.sbref_max_rigid_rotation_degrees,
        max_displacement_mm=opts.sbref_max_rigid_displacement_mm,
        min_support_overlap=opts.sbref_min_support_overlap,
        min_correlation=opts.sbref_min_intensity_correlation,
        force=opts.force,
    )
    runner.add_step(selected_reference.step)
    reference_selection = {
        "PolicyVersion": "robust-bold-reference-v3-normalized-grid",
        "CanonicalBOLDReferenceMetadata": str(robust_reference_metadata),
        "SelectionMetadata": str(selected_reference.metadata),
    }
    reg_ref_tag = "regRef"
    reg_ref_space = "FunctionalReference"
    registration_reference_label = "SelectedFunctionalReference"
    requested_spaces = set(opts.output_spaces)
    want_t1 = "T1w" in requested_spaces
    want_mni = "MNI152NLin2009cAsym" in requested_spaces
    want_fsnative = "fsnative" in requested_spaces
    want_fsaverage = "fsaverage" in requested_spaces
    compute_t1 = True
    need_surface_outputs = want_fsnative or want_fsaverage
    LOG.info("Output spaces requested: %s", ", ".join(opts.output_spaces))
    fsnative_to_fsaverage_spheres: dict[str, Path] = {}
    if want_fsaverage:
        fsaverage_dir = find_fsaverage_directory(
            runner,
            environment=env,
            subjects_directory=subjects_dir,
        )
        sphere_work = surf_dir / "fsaverage_spheres"
        for hemi, hemi_label in (("lh", "L"), ("rh", "R")):
            subject_sphere = require_nested_manifest_output(
                anat_info,
                "surfaces",
                f"{hemi}.sphere.reg",
                manifest_path=anat_manifest,
                manifest_name="Anatomical",
            )
            fsaverage_sphere = fsaverage_dir / "surf" / f"{hemi}.sphere"
            subject_output = sphere_work / (
                f"subject_hemi-{hemi_label}_from-fsnative_to-fsaverage_sphere.surf.gii"
            )
            fsaverage_output = sphere_work / f"fsaverage_hemi-{hemi_label}_sphere.surf.gii"
            runner.add_step(_create_mris_convert_step(
                source=subject_sphere, output=subject_output,
                env=env, force=opts.force,
            ))
            runner.add_step(_create_mris_convert_step(
                source=fsaverage_sphere, output=fsaverage_output,
                env=env, force=opts.force,
            ))
            fsnative_to_fsaverage_spheres[f"{hemi_label}.current"] = subject_output
            fsnative_to_fsaverage_spheres[f"{hemi_label}.new"] = fsaverage_output

    preproc_t1_name = _with_suffix(run_prefix, "_desc-preproc_bold.nii.gz")
    preproc_t1 = func_dir / preproc_t1_name
    preproc_t1_json = sidecar_json_path(preproc_t1)
    preproc_t1_noaroma = func_dir / _with_suffix(run_prefix, "_desc-preprocNoAROMA_bold.nii.gz")
    preproc_t1_noaroma_json = sidecar_json_path(preproc_t1_noaroma)
    preproc_mni = func_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_bold.nii.gz")
    preproc_mni_json = sidecar_json_path(preproc_mni)
    preproc_mni_noaroma = func_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preprocNoAROMA_bold.nii.gz")
    preproc_mni_noaroma_json = sidecar_json_path(preproc_mni_noaroma)
    preproc_fsnative = {
        "L": func_dir / _with_suffix(f"{run_base}_space-fsnative_hemi-L", "_desc-preproc_bold.func.gii"),
        "R": func_dir / _with_suffix(f"{run_base}_space-fsnative_hemi-R", "_desc-preproc_bold.func.gii"),
    }
    preproc_fsnative_noaroma = {
        "L": func_dir / _with_suffix(f"{run_base}_space-fsnative_hemi-L", "_desc-preprocNoAROMA_bold.func.gii"),
        "R": func_dir / _with_suffix(f"{run_base}_space-fsnative_hemi-R", "_desc-preprocNoAROMA_bold.func.gii"),
    }
    preproc_fsaverage = {
        "L": func_dir / _with_suffix(f"{run_base}_space-fsaverage_hemi-L", "_desc-preproc_bold.func.gii"),
        "R": func_dir / _with_suffix(f"{run_base}_space-fsaverage_hemi-R", "_desc-preproc_bold.func.gii"),
    }
    preproc_fsaverage_noaroma = {
        "L": func_dir / _with_suffix(f"{run_base}_space-fsaverage_hemi-L", "_desc-preprocNoAROMA_bold.func.gii"),
        "R": func_dir / _with_suffix(f"{run_base}_space-fsaverage_hemi-R", "_desc-preprocNoAROMA_bold.func.gii"),
    }
    preproc_fsnative_json = {hemi: sidecar_json_path(path) for hemi, path in preproc_fsnative.items()}
    preproc_fsnative_noaroma_json = {hemi: sidecar_json_path(path) for hemi, path in preproc_fsnative_noaroma.items()}
    preproc_fsaverage_json = {hemi: sidecar_json_path(path) for hemi, path in preproc_fsaverage.items()}
    preproc_fsaverage_noaroma_json = {hemi: sidecar_json_path(path) for hemi, path in preproc_fsaverage_noaroma.items()}
    epi_t1 = uncompressed_nifti_path(reg_dir / _with_suffix(run_prefix, "_desc-preproc_bold.nii.gz"))
    epi_mni = uncompressed_nifti_path(reg_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_bold.nii.gz"))
    epi_mean_t1 = qc_dir / _with_suffix(run_prefix, "_desc-preproc_mean.nii.gz")
    epi_mean_mni = qc_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_mean.nii.gz")
    epi_mean_nodc_t1 = qc_dir / _with_suffix(run_prefix, "_desc-preprocNoDC_mean.nii.gz")
    boldref_t1_out = func_dir / _with_suffix(run_prefix, "_boldref.nii.gz")
    boldref_t1_json = sidecar_json_path(boldref_t1_out)
    reg_prenonlinear_qc_out = func_dir / _with_suffix(run_prefix, "_desc-preNonlinearReg_boldref.nii.gz")
    reg_prenonlinear_qc_json = sidecar_json_path(reg_prenonlinear_qc_out)
    reg_base_qc_out = func_dir / _with_suffix(run_prefix, "_desc-baseReg_boldref.nii.gz")
    reg_base_qc_json = sidecar_json_path(reg_base_qc_out)
    reg_refine_qc_out = func_dir / _with_suffix(run_prefix, "_desc-refineReg_boldref.nii.gz")
    reg_refine_qc_json = sidecar_json_path(reg_refine_qc_out)
    reg_mni_qc_out = func_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_boldref.nii.gz")
    reg_mni_qc_json = sidecar_json_path(reg_mni_qc_out)
    melodic_ic_t1_out = func_dir / _with_suffix(run_prefix, "_desc-melodicIC_bold.nii.gz")
    melodic_ic_t1_json = sidecar_json_path(melodic_ic_t1_out)
    fieldmap_hz_in_t1_out = fmap_dir / _with_suffix(run_prefix, "_desc-fieldmapHz_fieldmap.nii.gz")
    fieldmap_hz_in_t1_json = sidecar_json_path(fieldmap_hz_in_t1_out)
    fmap_field_hz_out = fmap_dir / _with_suffix(f"{run_base}_space-{reg_ref_space}", "_desc-fieldmapHz_fieldmap.nii.gz")
    fmap_field_hz_json = sidecar_json_path(fmap_field_hz_out)
    fmap_sdc_warp_out = fmap_dir / _with_suffix(f"{run_base}_space-{reg_ref_space}", "_desc-sdcwarp_fieldmap.nii.gz")
    fmap_sdc_warp_json = sidecar_json_path(fmap_sdc_warp_out)
    fmap_bold_sdc_warp_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-boldSDCwarp_fieldmap.nii.gz"
    )
    fmap_bold_sdc_warp_json = sidecar_json_path(fmap_bold_sdc_warp_out)
    fmap_jacobian_out = fmap_dir / _with_suffix(f"{run_base}_space-{reg_ref_space}", "_desc-jacobian_fieldmap.nii.gz")
    fmap_jacobian_json = sidecar_json_path(fmap_jacobian_out)
    fmap_topup_coeff_out = fmap_dir / _with_suffix(run_base, "_desc-topupcoeff_fieldmap.nii.gz")
    fmap_topup_coeff_json = sidecar_json_path(fmap_topup_coeff_out)
    fmap_synbold_ref_out = fmap_dir / _with_suffix(
        f"{run_base}_space-{reg_ref_space}", "_desc-synbold_boldref.nii.gz"
    )
    fmap_synbold_ref_json = sidecar_json_path(fmap_synbold_ref_out)
    fmap_synbold_rigid_out = fmap_dir / _with_suffix(
        f"{run_base}_space-T1w", "_desc-synboldRigid_boldref.nii.gz"
    )
    fmap_synbold_rigid_json = sidecar_json_path(fmap_synbold_rigid_out)
    anat_brain_mask_in_t1 = func_dir / _with_suffix(run_prefix, "_desc-brain_mask.nii.gz")
    anat_brain_mask_in_t1_json = sidecar_json_path(anat_brain_mask_in_t1)
    anat_brain_mask_in_mni = qc_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", "_desc-brain_mask.nii.gz")
    aroma_dir = opts.work_dir / "ica_aroma"
    aroma_label = _ica_aroma_output_label(opts.ica_aroma_denoise_type)
    aroma_t1_dir = aroma_dir / "space-T1w"
    aroma_mni_dir = aroma_dir / "space-MNI152NLin2009cAsym"
    aroma_clean = uncompressed_nifti_path(aroma_t1_dir / _with_suffix(run_prefix, f"_desc-{aroma_label}_bold.nii.gz"))
    aroma_clean_mean = aroma_t1_dir / _with_suffix(run_prefix, f"_desc-{aroma_label}_mean.nii.gz")
    aroma_clean_mni = uncompressed_nifti_path(aroma_mni_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", f"_desc-{aroma_label}_bold.nii.gz"))
    aroma_clean_mean_mni = aroma_mni_dir / _with_suffix(f"{run_base}_space-MNI152NLin2009cAsym", f"_desc-{aroma_label}_mean.nii.gz")
    confounds_tsv = func_dir / _with_suffix(run_stem, "_desc-confounds_timeseries.tsv")
    confounds_json = func_dir / _with_suffix(run_stem, "_desc-confounds_timeseries.json")
    publication_manifest = functional_manifest_path(
        opts.sub_id,
        run_base,
        project=opts.project,
        preprocessing_id=opts.preprocessing_id,
        ses_id=opts.ses_id,
    )
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
    epi_ref_in_t1_affine = reg_dir / f"{run_stem}_epiRef_in_t1_affine.nii.gz"
    fs_t1_to_t1w_mat = reg_dir / f"{run_stem}_fsT1ToT1w.mat"
    reg_ref_to_t1w_mat = reg_dir / f"{run_stem}_{reg_ref_tag}2t1w.mat"
    epi_mean_nodc_t1_affine = reg_dir / f"{run_stem}_epi_mean_nodc_t1_affine.nii.gz"
    reg_ref_in_t1_linear = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_linear.nii.gz"
    reg_ref_dc_ref_n4 = sdc_dir / f"{reg_ref_tag}_dc_ref_n4.nii.gz"
    reg_ref_in_t1_base_n4 = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_base_n4.nii.gz"
    reg_ref_in_t1_refine_n4 = reg_dir / f"{run_stem}_{reg_ref_tag}_in_t1_refine_n4.nii.gz"
    epi_ref_in_t1_base = reg_dir / f"{run_stem}_epiRef_in_t1_base.nii.gz"
    epi_to_t1_warp_planned = reg_dir / f"{run_stem}_EPIToT1w_warp.nii.gz"
    fallback_affine_warp_planned = reg_dir / f"{run_stem}_SyNAffine_warp.nii.gz"
    fallback_base_itk_planned = reg_dir / f"{run_stem}_SyNBoldAff_itk_warp.nii.gz"
    fallback_base_fnirt_planned = reg_dir / f"{run_stem}_SyNBoldAff_fnirt_warp.nii.gz"
    fallback_base_warp_planned = reg_dir / f"{run_stem}_SyNBoldAffBase_warp.nii.gz"
    warp_regref2t1_planned = reg_dir / f"{run_stem}_{reg_ref_tag}ToT1wBase_warp.nii.gz"
    syn_refine_itk_planned = reg_dir / f"{run_stem}_SyNBoldAffRefine_itk_warp.nii.gz"
    syn_refine_fnirt_planned = reg_dir / f"{run_stem}_SyNBoldAffRefine_fnirt_warp.nii.gz"
    warp_sbref2t1_refined_planned = epi_to_t1_warp_planned
    warp_regref2t1_refined_planned = reg_dir / f"{run_stem}_{reg_ref_tag}ToT1wRefined_warp.nii.gz"
    field_hz_regref = sdc_dir / f"{run_stem}_fieldmap_Hz_{reg_ref_tag}Space.nii.gz"
    syn_fallback_work = reg_dir / "ants_syn"
    syn_refine_work = reg_dir / "ants_syn_refine"
    fallback_syn_outputs = [
        syn_fallback_work / f"{run_stem}_SyNBoldAff_Composite.h5",
        syn_fallback_work / f"{run_stem}_SyNBoldAff_InverseComposite.h5",
    ]
    refine_syn_outputs = [
        syn_refine_work / f"{run_stem}_SyNRefine_0Warp.nii.gz",
        syn_refine_work / f"{run_stem}_SyNRefine_0InverseWarp.nii.gz",
    ]

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
    epi_ref_to_reg_ref_mat: Optional[Path] = selected_reference.epi_to_reference
    reg_ref_to_epi_ref_mat: Optional[Path] = selected_reference.reference_to_epi
    fieldmap_hz_in_t1: Optional[Path] = None
    synthetic_ref: Optional[Path] = None
    synbold_rigid_qc: Optional[Path] = None
    ants_forward_xfm: Optional[Path] = None
    syn_refine_warp: Optional[Path] = None
    syn_refine_pe_frame: Optional[_PEAlignedANTsFrame] = None
    epi_mc_ref: Optional[Path] = robust_ref
    reg_ref_dist_ref: Optional[Path] = selected_reference.image
    reg_ref_dc_ref: Optional[Path] = None
    pre_nonlinear_ref_in_t1: Optional[Path] = None
    sbref_dist_ref: Optional[Path] = selected_reference.image
    sbref_dc_ref: Optional[Path] = None
    fsl_mat: Optional[Path] = None
    boldref_t1: Path
    mc_mat_dir = mc_dir / f"{run_stem}_mc.nii.gz.mat"

    if use_syn_fallback:
        reg_ref_dist_ref = selected_reference.image
    else:
        require_existing_path(subjects_dir, "FreeSurfer SUBJECTS_DIR from anatomical manifest")
        reg_ref_dist_ref = selected_reference.image

        topup_label = "fieldmap" if use_fieldmap_sdc else "synbold_disco"
        topup_work_dir = opts.work_dir / f"topup_{topup_label}_native"
        topup_fieldcoef = topup_work_dir / "topup_results_fieldcoef.nii.gz"
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
            t1_brain_cmd = ["fslmaths", str(anat_t1), "-mas", str(anat_brain_mask), str(t1_brain)]
            runner.add_step(Step.command_step(
                t1_brain_cmd,
                outputs=(t1_brain,),
                inputs=(anat_t1, anat_brain_mask),
                force=opts.force,
                env=env,
                name="Prepare SynBOLD-DisCo T1",
                prepare=lambda: ensure_directory(t1_brain.parent),
            ))
            synbold_image = ensure_image(
                image=opts.synbold_disco_image,
                engine=opts.synbold_disco_engine,
            )
            synbold_rigid_step = _create_synbold_rigid_registration_step(
                run_child=runner.run_child,
                distorted_reference=reg_ref_dist_ref,
                anatomical_t1=anat_t1,
                anatomical_mask=anat_brain_mask,
                work_dir=synbold_work / "rigid_registration",
                env=env,
                max_translation_mm=opts.synbold_max_rigid_translation_mm,
                max_rotation_degrees=opts.synbold_max_rigid_rotation_degrees,
                force=opts.force,
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
                force=opts.force,
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
                force=opts.force,
            )
            runner.add_step(topup_native.step)
            se2sbref_mat = opts.work_dir / "pose" / f"se2{reg_ref_tag}_6dof.mat"
            runner.add_step(create_identity_transform_step(se2sbref_mat))
            sbref2se_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2se_6dof.mat"
            runner.add_step(create_identity_transform_step(sbref2se_mat))
            warp_sbref = sdc_dir / f"WarpField_{reg_ref_tag}Space.nii.gz"
            runner.add_step(create_copy_nifti_step(
                src=topup_native.dfout,
                dst=warp_sbref,
                force=opts.force,
                step_name="Place SynBOLD-DisCo TOPUP Warp",
            ))
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
                se1_readout = float(bids_readout_time(se1_meta))
                se2_readout = float(bids_readout_time(se2_meta))
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
                force=opts.force,
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
            runner.add_step(create_nifti_volume_extraction_step(
                img=matching_se,
                index_zero_based=0,
                out_3d=se_match_ref,
                env=env,
                force=opts.force,
                label="Extract Matching Distorted SE Reference",
            ))
            se2sbref_mat = opts.work_dir / "pose" / f"se2{reg_ref_tag}_6dof.mat"
            sbref2se_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2se_6dof.mat"
            pose_work = se2sbref_mat.parent / f"{se2sbref_mat.stem}_qc"
            pose_mask = pose_work / "fixed_support_mask.nii.gz"
            runner.add_step(create_image_support_mask_step(
                image=reg_ref_dist_ref,
                out_mask=pose_mask,
                erosion_voxels=1,
                minimum_voxels=100,
                force=opts.force,
            ))
            runner.add_step(_create_flirt_registration_step(
                run_child=runner.run_child,
                in_img=se_match_ref,
                ref_img=reg_ref_dist_ref,
                out_mat=se2sbref_mat,
                work_dir=pose_work,
                fixed_mask=pose_mask,
                env=env,
                force=opts.force,
            ))
            runner.add_step(
                _create_invert_mat_step(
                    mat=se2sbref_mat,
                    out_mat=sbref2se_mat,
                    env=env,
                    force=opts.force,
                )
            )

            matching_motion_mat = _normalized_topup_matrix(
                topup_native.rbmout,
                index_1based=matching_index_1based,
            )
            reg_ref_to_topup_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2topup_6dof.mat"
            runner.add_step(_create_concat_mats_step(
                first=matching_motion_mat,
                second=sbref2se_mat,
                out_mat=reg_ref_to_topup_mat,
                env=env,
                force=opts.force,
            ))
            topup_to_reg_ref_mat = opts.work_dir / "pose" / f"topup2{reg_ref_tag}_6dof.mat"
            runner.add_step(_create_invert_mat_step(
                mat=reg_ref_to_topup_mat,
                out_mat=topup_to_reg_ref_mat,
                env=env,
                force=opts.force,
            ))

            corrected_se_median = topup_work_dir / "se_unwarped_median.nii.gz"
            corrected_se_cmd = [
                "fslmaths",
                str(topup_native.iout),
                "-Tmedian",
                str(corrected_se_median),
            ]
            runner.add_step(Step.command_step(
                corrected_se_cmd,
                env=env,
                outputs=(corrected_se_median,),
                inputs=(topup_native.iout,),
                force=opts.force,
                name="Build Corrected SE Reference",
            ))

            registration_shift_vox = (
                topup_work_dir / f"{reg_ref_tag}_registrationReadout_shift_vox.nii.gz"
            )
            registration_warp_topup = (
                topup_work_dir / f"WarpField_{reg_ref_tag}_registrationReadout.nii.gz"
            )
            runner.add_step(_create_target_shift_step(
                field_hz=topup_native.field_hz,
                reference=corrected_se_median,
                readout_time=float(registration_readout),
                shift_vox=registration_shift_vox,
                env=env,
                force=opts.force,
            ))
            runner.add_step(_create_target_readout_warp_step(
                field_hz=topup_native.field_hz,
                reference=corrected_se_median,
                phase_encoding_direction=reg_ref_ped,
                shift_vox=registration_shift_vox,
                out_warp=registration_warp_topup,
                env=env,
                force=opts.force,
            ))

            initial_dc_topup = topup_work_dir / f"{reg_ref_tag}_dc_initial_topupSpace.nii.gz"
            runner.add_step(_create_applywarp_step(
                in_img=reg_ref_dist_ref,
                ref_img=corrected_se_median,
                warp=registration_warp_topup,
                premat=reg_ref_to_topup_mat,
                out_img=initial_dc_topup,
                env=env,
                force=opts.force,
            ))
            postdc_refine_mat = opts.work_dir / "pose" / f"{reg_ref_tag}2topup_postdc_6dof.mat"
            postdc_registered = topup_work_dir / f"{reg_ref_tag}_dc_postRigid_topupSpace.nii.gz"
            postdc_refine_qc_path = (
                opts.work_dir / "pose" / f"{reg_ref_tag}2topup_postdc_6dof_qc.json"
            )
            runner.add_step(_create_local_rigid_refinement_step(
                run_child=runner.run_child,
                moving=initial_dc_topup,
                fixed=corrected_se_median,
                out_mat=postdc_refine_mat,
                out_registered=postdc_registered,
                qc_json=postdc_refine_qc_path,
                env=env,
                force=opts.force,
            ))
            postdc_refine_qc = postdc_refine_qc_path
            topup_to_reg_ref_refined_mat = (
                opts.work_dir / "pose" / f"topup2{reg_ref_tag}_postdcRefined_6dof.mat"
            )
            runner.add_step(_create_concat_mats_step(
                first=topup_to_reg_ref_mat,
                second=postdc_refine_mat,
                out_mat=topup_to_reg_ref_refined_mat,
                env=env,
                force=opts.force,
            ))

            pe_work = topup_work_dir / "pe_residual_to_corrected_se"
            pe_overlap = pe_work / "overlap_mask.nii.gz"
            runner.add_step(create_native_overlap_mask_step(
                images=[postdc_registered, corrected_se_median],
                out_mask=pe_overlap,
                erosion_voxels=opts.synbold_overlap_erosion_voxels,
                minimum_voxels=opts.synbold_min_overlap_voxels,
                force=opts.force,
            ))
            pe_frame = _ants_pe_aligned_frame(inputs.epi, reg_ref_ped)
            pe_aligned_dir = pe_work / "pe_aligned"
            pe_moving = pe_aligned_dir / "moving_reference.nii.gz"
            pe_fixed = pe_aligned_dir / "fixed_corrected_se.nii.gz"
            pe_mask = pe_aligned_dir / "overlap_mask.nii.gz"
            runner.add_step(_create_nifti_in_ants_frame_step(
                source=postdc_registered, out_image=pe_moving,
                frame=pe_frame, force=opts.force,
            ))
            runner.add_step(_create_nifti_in_ants_frame_step(
                source=corrected_se_median, out_image=pe_fixed,
                frame=pe_frame, force=opts.force,
            ))
            runner.add_step(_create_nifti_in_ants_frame_step(
                source=pe_overlap, out_image=pe_mask,
                frame=pe_frame, force=opts.force,
            ))
            pe_registration = _create_ants_registration_step(
                run_child=runner.run_child,
                moving_img=pe_moving,
                fixed_img=pe_fixed,
                work_dir=pe_aligned_dir / "ants",
                out_prefix="PEResidual_",
                env=env,
                force=opts.force,
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
            runner.add_step(_create_restore_ants_warp_step(
                aligned_warp=pe_registration.forward_transform,
                original_reference=postdc_registered,
                out_warp=pe_residual_itk,
                frame=pe_frame,
                force=opts.force,
            ))
            pe_residual_fnirt = pe_work / "residual_fnirt_warp.nii.gz"
            runner.add_step(_create_wb_convert_itk_warp_to_fnirt_step(
                itk_warp=pe_residual_itk,
                src_space_ref=postdc_registered,
                out_warp=pe_residual_fnirt,
                env=env,
                force=opts.force,
            ))
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
            runner.add_step(_create_convertwarp_postmat_step(
                ref=corrected_se_median,
                warp1=registration_warp_topup,
                postmat=postdc_refine_mat,
                out_warp=registration_rigid_warp_topup,
                env=env,
                force=opts.force,
            ))
            registration_complete_warp_topup = (
                topup_work_dir / f"WarpField_{reg_ref_tag}_registrationReadout_complete.nii.gz"
            )
            runner.add_step(_create_convertwarp_merge_warps_step(
                ref=corrected_se_median,
                warp1=registration_rigid_warp_topup,
                warp2=pe_residual_fnirt,
                out_warp=registration_complete_warp_topup,
                env=env,
                force=opts.force,
            ))
            warp_sbref = sdc_dir / f"WarpField_{reg_ref_tag}Space.nii.gz"
            runner.add_step(_create_convertwarp_conjugate_affine_step(
                ref=reg_ref_dist_ref,
                warp1=registration_complete_warp_topup,
                premat=reg_ref_to_topup_mat,
                postmat=topup_to_reg_ref_mat,
                out_warp=warp_sbref,
                env=env,
                force=opts.force,
            ))

            bold_ref_to_topup_step = _create_bold_ref_to_topup_transform_step(
                reg_ref_to_topup_mat=reg_ref_to_topup_mat,
                epi_ref_to_reg_ref_mat=selected_reference.epi_to_reference,
                work_dir=opts.work_dir,
                env=env,
                force=opts.force,
            )
            runner.add_step(bold_ref_to_topup_step)
            bold_ref_to_topup_mat = bold_ref_to_topup_step.outputs[0]
            bold_shift_vox = topup_work_dir / "bold_target_shift_vox.nii.gz"
            bold_warp_topup = topup_work_dir / "WarpField_bold_targetReadout.nii.gz"
            runner.add_step(_create_target_shift_step(
                field_hz=topup_native.field_hz,
                reference=corrected_se_median,
                readout_time=float(bold_readout),
                shift_vox=bold_shift_vox,
                env=env,
                force=opts.force,
            ))
            runner.add_step(_create_target_readout_warp_step(
                field_hz=topup_native.field_hz,
                reference=corrected_se_median,
                phase_encoding_direction=epi_ped,
                shift_vox=bold_shift_vox,
                out_warp=bold_warp_topup,
                env=env,
                force=opts.force,
            ))
            bold_rigid_warp_topup = topup_work_dir / "WarpField_bold_postRigid.nii.gz"
            runner.add_step(_create_convertwarp_postmat_step(
                ref=corrected_se_median,
                warp1=bold_warp_topup,
                postmat=postdc_refine_mat,
                out_warp=bold_rigid_warp_topup,
                env=env,
                force=opts.force,
            ))
            bold_complete_warp_topup = topup_work_dir / "WarpField_bold_complete.nii.gz"
            runner.add_step(_create_convertwarp_merge_warps_step(
                ref=corrected_se_median,
                warp1=bold_rigid_warp_topup,
                warp2=pe_residual_fnirt,
                out_warp=bold_complete_warp_topup,
                env=env,
                force=opts.force,
            ))
            warp_bold_to_reg_ref = sdc_dir / "WarpField_boldToRegRef.nii.gz"
            runner.add_step(_create_convertwarp_conjugate_affine_step(
                ref=reg_ref_dist_ref,
                warp1=bold_complete_warp_topup,
                premat=bold_ref_to_topup_mat,
                postmat=topup_to_reg_ref_mat,
                out_warp=warp_bold_to_reg_ref,
                env=env,
                force=opts.force,
            ))
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
                "PostFieldmapRefinementConstraint": (
                    "Unrestricted" if do_refinement else None
                ),
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
                force=opts.force,
            )
        )
        if opts.use_jacobian:
            jac_sbref = sdc_dir / f"Jacobian_{reg_ref_tag}Space.nii.gz"
            jacobian_temporary = jac_sbref.parent / (
                jac_sbref.name.replace(".nii.gz", "") + "_tmp8.nii.gz"
            )
            jacobian_junk = jac_sbref.parent / "junk_warp.nii.gz"
            runner.add_step(_create_warp_jacobian_step(
                warp=warp_sbref,
                ref=reg_ref_dist_ref,
                temporary=jacobian_temporary,
                junk=jacobian_junk,
                env=env,
                force=opts.force,
            ))
            runner.add_step(_create_average_jacobian_step(
                temporary=jacobian_temporary,
                junk=jacobian_junk,
                warp=warp_sbref,
                ref=reg_ref_dist_ref,
                out_jac=jac_sbref,
                env=env,
                force=opts.force,
            ))
            reg_ref_dc_jac = sdc_dir / f"{reg_ref_tag}_dc_jac.nii.gz"
            jac_cmd = ["fslmaths", str(reg_ref_dc), "-mul", str(jac_sbref), str(reg_ref_dc_jac)]
            runner.add_step(Step.command_step(
                jac_cmd,
                outputs=(reg_ref_dc_jac,),
                inputs=(reg_ref_dc, jac_sbref),
                force=opts.force,
                env=env,
            ))
            reg_ref_dc = reg_ref_dc_jac

        reg_ref_dc_ref = reg_ref_dc
        sbref_dc_ref = reg_ref_dc_ref
        runner.add_step(create_n4_bias_correction_step(
            in_img=reg_ref_dc_ref,
            out_img=reg_ref_dc_ref_n4,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_bbregister_step(
            sbref=reg_ref_dc_ref_n4,
            fs_subject=fs_subject,
            reg_dat=reg_dat,
            fsl_mat=fsl_mat,
            surf=opts.bbregister_surf,
            init=opts.bbregister_init,
            dof=opts.bbregister_dof,
            env=env,
            force=opts.force,
        ))
    # Output grid
    if opts.output_grid == "t1_epi_vox":
        t1_ref = reg_dir / f"{run_stem}_t1_grid_epi_vox.nii.gz"
        runner.add_step(_create_t1_epi_vox_target_step(
            t1_image=anat_t1,
            source_epi=inputs.epi,
            out_target=t1_ref,
            env=env,
            force=opts.force,
        ))
    else:
        t1_ref = reg_dir / f"{run_stem}_t1_native.nii.gz"
        runner.add_step(_create_t1_native_target_step(
            t1_image=anat_t1,
            out_target=t1_ref,
            env=env,
            force=opts.force,
        ))
    if not use_syn_fallback:
        fs_t1_image = subjects_dir / fs_subject / "mri" / "T1.mgz"
        if opts.output_grid == "t1_epi_vox":
            runner.add_step(_create_t1_epi_vox_target_step(
                t1_image=fs_t1_image,
                source_epi=inputs.epi,
                out_target=bbr_t1_ref,
                env=env,
                force=opts.force,
            ))
        else:
            runner.add_step(_create_t1_native_target_step(
                t1_image=fs_t1_image,
                out_target=bbr_t1_ref,
                env=env,
                force=opts.force,
            ))
        runner.add_step(_create_tkregister2_regheader_fslmat_step(
            mov_img=bbr_t1_ref,
            targ_img=t1_ref,
            out_mat=fs_t1_to_t1w_mat,
            env=env,
            force=opts.force,
        ))
        assert fsl_mat is not None
        runner.add_step(_create_concat_mats_step(
            first=fs_t1_to_t1w_mat,
            second=fsl_mat,
            out_mat=reg_ref_to_t1w_mat,
            env=env,
            force=opts.force,
        ))
    syn_fixed_mask = reg_dir / f"{run_stem}_t1_brain_mask.nii.gz"
    runner.add_step(create_mask_resampling_step(
        src_mask=anat_brain_mask,
        ref_img=t1_ref,
        out_mask=syn_fixed_mask,
        force=opts.force,

    ))
    runner.add_step(_create_mni_2mm_target_step(
        mni_template=anat_mni_template,
        out_target=mni_ref,
        env=env,
        force=opts.force,
    ))
    syn_refine_fnirt_warp: Optional[Path] = None
    warp_regref2t1_refined: Optional[Path] = None
    warp_sbref2t1_refined: Optional[Path] = None
    base_static_warp: Optional[Path] = None
    base_ref_in_t1: Optional[Path] = None
    reg_ref_space_for_qc = reg_ref_space
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
            force=opts.force,
            rigid_mat_out=epi_mean_to_t1_fallback_mat,
            rigid_qc_out=epi_mean_nodc_t1_affine,
            registration_label="fieldmapless EPI-to-T1",
        )
        runner.add_step(fallback_rigid_step.step)
        pre_nonlinear_ref_in_t1 = epi_mean_nodc_t1_affine
        runner.add_step(create_n4_bias_correction_step(
            in_img=epi_mean_nodc_t1_affine,
            out_img=reg_ref_in_t1_base_n4,
            env=env,
            force=opts.force,
            mask=syn_fixed_mask,
        ))
        fallback_registration = _create_ants_synboldaff_step(
            moving_img=reg_ref_in_t1_base_n4,
            fixed_img=t1_ref,
            work_dir=syn_work,
            out_prefix=f"{run_stem}_SyNBoldAff_",
            env=env,
            force=opts.force,
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
        runner.add_step(_create_ants_composite_to_itk_warp_step(
            composite_xfm=fallback_composite_xfm,
            ref_img=t1_ref,
            out_warp=fallback_base_itk_warp,
            env=env,
            force=opts.force,
        ))
        fallback_base_fnirt_warp = fallback_base_fnirt_planned
        runner.add_step(_create_wb_convert_itk_warp_to_fnirt_step(
            itk_warp=fallback_base_itk_warp,
            src_space_ref=reg_ref_in_t1_base_n4,
            out_warp=fallback_base_fnirt_warp,
            env=env,
            force=opts.force,
        ))
        syn_refine_fnirt_warp = fallback_base_fnirt_warp
        fallback_affine_warp = fallback_affine_warp_planned
        runner.add_step(_create_convertwarp_premat_step(
            ref=t1_ref,
            premat=epi_mean_to_t1_fallback_mat,
            out_warp=fallback_affine_warp,
            env=env,
            force=opts.force,
        ))
        fallback_base_warp = fallback_base_warp_planned
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=t1_ref,
            warp1=fallback_affine_warp,
            warp2=fallback_base_fnirt_warp,
            out_warp=fallback_base_warp,
            env=env,
            force=opts.force,
        ))
        warp_regref2t1_refined = fallback_base_warp
        _create_convertwarp_premat_and_warp_step(
            runner=runner,
            ref=t1_ref,
            premat=selected_reference.epi_to_reference,
            warp1=fallback_base_warp,
            out_warp=epi_to_t1_warp_planned,
            env=env,
            force=opts.force,
        )
        base_static_warp = epi_to_t1_warp_planned
        runner.add_step(_create_applywarp_step(
            in_img=syn_moving,
            ref_img=t1_ref,
            warp=fallback_base_warp,
            out_img=reg_ref_in_t1_affine,
            env=env,
            force=opts.force,
        ))
        base_ref_in_t1 = reg_ref_in_t1_affine
        boldref_t1 = boldref_t1_out
        runner.add_step(_create_applywarp_step(
            in_img=syn_moving,
            ref_img=t1_ref,
            warp=fallback_base_warp,
            out_img=boldref_t1,
            env=env,
            force=opts.force,
        ))
    else:
        assert reg_ref_dc_ref is not None and reg_ref_dist_ref is not None and reg_ref_to_t1w_mat is not None and epi_mc_ref is not None
        warp_sbref2t1 = warp_regref2t1_planned
        assert warp_sbref is not None
        runner.add_step(create_flirt_transform_step(
            in_img=reg_ref_dist_ref,
            ref_img=t1_ref,
            mat=reg_ref_to_t1w_mat,
            out_img=reg_ref_in_t1_linear,
            env=env,
            force=opts.force,
        ))
        pre_nonlinear_ref_in_t1 = reg_ref_in_t1_linear
        runner.add_step(_create_convertwarp_postmat_step(
            ref=t1_ref,
            warp1=warp_sbref,
            postmat=reg_ref_to_t1w_mat,
            out_warp=warp_sbref2t1,
            env=env,
            force=opts.force,
        ))
        syn_moving = reg_ref_dist_ref
        runner.add_step(_create_applywarp_step(
            in_img=syn_moving,
            ref_img=t1_ref,
            warp=warp_sbref2t1,
            out_img=reg_ref_in_t1_affine,
            env=env,
            force=opts.force,
        ))
        if use_fieldmap_sdc:
            assert warp_bold_to_reg_ref is not None
            runner.add_step(_create_convertwarp_postmat_step(
                ref=t1_ref,
                warp1=warp_bold_to_reg_ref,
                postmat=reg_ref_to_t1w_mat,
                out_warp=epi_to_t1_warp_planned,
                env=env,
                force=opts.force,
            ))
            base_static_warp = epi_to_t1_warp_planned
        else:
            _create_convertwarp_premat_and_warp_step(
                runner=runner,
                ref=t1_ref,
                premat=selected_reference.epi_to_reference,
                warp1=warp_sbref2t1,
                out_warp=epi_to_t1_warp_planned,
                env=env,
                force=opts.force,
            )
            base_static_warp = epi_to_t1_warp_planned
        base_ref_in_t1 = reg_ref_in_t1_affine
        warp_regref2t1_refined = warp_sbref2t1
        boldref_t1 = boldref_t1_out
        runner.add_step(_create_applywarp_step(
            in_img=syn_moving,
            ref_img=t1_ref,
            warp=warp_sbref2t1,
            out_img=boldref_t1,
            env=env,
            force=opts.force,
        ))
        assert topup_native is not None and se2sbref_mat is not None
        field_to_reg_ref_mat = (
            topup_to_reg_ref_refined_mat
            if use_fieldmap_sdc
            else se2sbref_mat
        )
        assert field_to_reg_ref_mat is not None
        runner.add_step(create_flirt_transform_step(
            in_img=topup_native.field_hz,
            ref_img=reg_ref_dist_ref,
            mat=field_to_reg_ref_mat,
            out_img=field_hz_regref,
            env=env,
            force=opts.force,
        ))

    assert base_static_warp is not None and base_ref_in_t1 is not None and warp_regref2t1_refined is not None
    if do_refinement and use_synbold_reference:
        assert synthetic_ref is not None
        assert reg_ref_dc_ref is not None
        assert reg_ref_dist_ref is not None
        assert warp_sbref is not None
        assert reg_ref_to_t1w_mat is not None
        native_refine_work = reg_dir / "synbold_native_refine"
        native_overlap_mask = native_refine_work / f"overlap_erode-{int(opts.synbold_overlap_erosion_voxels)}_mask.nii.gz"
        runner.add_step(create_native_overlap_mask_step(
            images=[reg_ref_dc_ref, synthetic_ref],
            out_mask=native_overlap_mask,
            erosion_voxels=opts.synbold_overlap_erosion_voxels,
            minimum_voxels=opts.synbold_min_overlap_voxels,
            force=opts.force,
        ))
        pe_frame = _ants_pe_aligned_frame(inputs.epi, reg_ref_ped)
        syn_refine_pe_frame = pe_frame
        aligned_work = native_refine_work / f"pe-{reg_ref_ped.rstrip('-')}_aligned"
        aligned_moving = aligned_work / "moving_epi_reference.nii.gz"
        runner.add_step(_create_nifti_in_ants_frame_step(
            source=reg_ref_dc_ref,
            out_image=aligned_moving,
            frame=pe_frame,
            force=opts.force,
        ))
        aligned_fixed = aligned_work / "fixed_synthetic_reference.nii.gz"
        runner.add_step(_create_nifti_in_ants_frame_step(
            source=synthetic_ref,
            out_image=aligned_fixed,
            frame=pe_frame,
            force=opts.force,
        ))
        aligned_mask = aligned_work / "overlap_mask.nii.gz"
        runner.add_step(_create_nifti_in_ants_frame_step(
            source=native_overlap_mask,
            out_image=aligned_mask,
            frame=pe_frame,
            force=opts.force,
        ))
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
            force=opts.force,
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
        runner.add_step(_create_restore_ants_warp_step(
            aligned_warp=aligned_refine_warp,
            original_reference=reg_ref_dc_ref,
            out_warp=refine_warp,
            frame=pe_frame,
            force=opts.force,
        ))
        syn_refine_warp = refine_warp
        ants_forward_xfm = refine_warp
        syn_refine_fnirt_warp = native_refine_work / "residual_fnirt_warp.nii.gz"
        runner.add_step(_create_wb_convert_itk_warp_to_fnirt_step(
            itk_warp=refine_warp,
            src_space_ref=reg_ref_dc_ref,
            out_warp=syn_refine_fnirt_warp,
            env=env,
            force=opts.force,
        ))
        combined_native_warp = native_refine_work / "fieldmap_plus_residual_warp.nii.gz"
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=reg_ref_dist_ref,
            warp1=warp_sbref,
            warp2=syn_refine_fnirt_warp,
            out_warp=combined_native_warp,
            env=env,
            force=opts.force,
        ))
        warp_regref2t1_refined = warp_regref2t1_refined_planned
        runner.add_step(_create_convertwarp_postmat_step(
            ref=t1_ref,
            warp1=combined_native_warp,
            postmat=reg_ref_to_t1w_mat,
            out_warp=warp_regref2t1_refined,
            env=env,
            force=opts.force,
        ))
        warp_sbref2t1_refined = warp_sbref2t1_refined_planned
        assert use_fieldmap_sdc and warp_bold_to_reg_ref is not None
        bold_combined_native_warp = native_refine_work / "bold_fieldmap_plus_residual_warp.nii.gz"
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=reg_ref_dist_ref,
            warp1=warp_bold_to_reg_ref,
            warp2=syn_refine_fnirt_warp,
            out_warp=bold_combined_native_warp,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_convertwarp_postmat_step(
            ref=t1_ref,
            warp1=bold_combined_native_warp,
            postmat=reg_ref_to_t1w_mat,
            out_warp=warp_sbref2t1_refined,
            env=env,
            force=opts.force,
        ))
    elif do_refinement:
        syn_work = reg_dir / "ants_syn_refine"
        runner.add_step(create_n4_bias_correction_step(
            in_img=base_ref_in_t1,
            out_img=reg_ref_in_t1_refine_n4,
            env=env,
            force=opts.force,
            mask=syn_fixed_mask,
        ))
        refinement = _create_ants_registration_step(
            run_child=runner.run_child,
            moving_img=reg_ref_in_t1_refine_n4,
            fixed_img=t1_ref,
            work_dir=syn_work,
            out_prefix=f"{run_stem}_SyNRefine_",
            env=env,
            force=opts.force,
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
        syn_refine_warp = refine_warp
        ants_forward_xfm = refine_warp if ants_forward_xfm is None else ants_forward_xfm
        syn_refine_fnirt_warp = syn_refine_fnirt_planned
        runner.add_step(_create_wb_convert_itk_warp_to_fnirt_step(
            itk_warp=refine_warp,
            src_space_ref=reg_ref_in_t1_refine_n4,
            out_warp=syn_refine_fnirt_warp,
            env=env,
            force=opts.force,
        ))
        assert warp_regref2t1_refined is not None
        base_regref_warp = warp_regref2t1_refined
        warp_regref2t1_refined = warp_regref2t1_refined_planned
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=t1_ref,
            warp1=base_regref_warp,
            warp2=syn_refine_fnirt_warp,
            out_warp=warp_regref2t1_refined,
            env=env,
            force=opts.force,
        ))
        warp_sbref2t1_refined = warp_sbref2t1_refined_planned
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=t1_ref,
            warp1=base_static_warp,
            warp2=syn_refine_fnirt_warp,
            out_warp=warp_sbref2t1_refined,
            env=env,
            force=opts.force,
        ))
    else:
        # Without refinement, the selected base warp is the final warp.
        warp_sbref2t1_refined = base_static_warp

    assert pre_nonlinear_ref_in_t1 is not None and warp_sbref2t1_refined is not None and warp_regref2t1_refined is not None
    if (not use_syn_fallback) and topup_native is not None:
        # The Hz field lives in registration-reference space, so its T1w
        # derivative must follow the final registration-reference-to-T1w warp.
        # In particular, defer this until after anatomical SyN refinement; the
        # base warp above is not the final spatial mapping in that pathway.
        fieldmap_hz_in_t1 = fieldmap_hz_in_t1_out
        runner.add_step(_create_applywarp_step(
            in_img=field_hz_regref,
            ref_img=t1_ref,
            warp=warp_regref2t1_refined,
            out_img=fieldmap_hz_in_t1,
            env=env,
            force=opts.force,
        ))
    runner.add_step(create_copy_nifti_step(
        src=pre_nonlinear_ref_in_t1,
        dst=reg_prenonlinear_qc_out,
        force=opts.force,
        step_name="Finalize Registration QC",
    ))
    runner.add_step(create_copy_nifti_step(
        src=base_ref_in_t1,
        dst=reg_base_qc_out,
        force=opts.force,
        step_name="Finalize Registration QC",
    ))
    if do_refinement:
        runner.add_step(_create_applywarp_step(
            in_img=syn_moving,
            ref_img=t1_ref,
            warp=warp_regref2t1_refined,
            out_img=reg_refine_qc_out,
            env=env,
            force=opts.force,
        ))
    else:
        runner.add_step(create_copy_nifti_step(
            src=reg_base_qc_out,
            dst=reg_refine_qc_out,
            force=opts.force,
            step_name="Finalize Registration QC",
        ))
    if want_mni:
        runner.add_step(_create_ants_composite_to_itk_warp_step(
            composite_xfm=t1_to_mni_xfm,
            ref_img=mni_ref,
            out_warp=t1_to_mni_itk_warp,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_wb_convert_itk_warp_to_fnirt_step(
            itk_warp=t1_to_mni_itk_warp,
            src_space_ref=anat_t1,
            out_warp=t1_to_mni_fnirt_warp,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=mni_ref,
            warp1=warp_sbref2t1_refined,
            warp2=t1_to_mni_fnirt_warp,
            out_warp=warp_epi2mni,
            env=env,
            force=opts.force,
            ))

    final_sources_4d: dict[str, Path] = {}
    final_sources_mean: dict[str, Path] = {}
    final_masks: dict[str, Path] = {}
    t1_final_4d: Optional[Path] = None
    t1_final_mean: Optional[Path] = None
    afni_motion_affines = mc_dir / f"{run_stem}_mc_pull.aff12.1D"

    space_sequence: list[str] = []
    if compute_t1:
        space_sequence.append("T1w")
    if want_mni:
        space_sequence.append("MNI152NLin2009cAsym")
    log_space_sequence = []
    if want_t1:
        log_space_sequence.append("T1w")
    elif compute_t1:
        log_space_sequence.append("T1w(required)")
    if want_fsnative:
        log_space_sequence.append("fsnative")
    if want_fsaverage:
        log_space_sequence.append("fsaverage")
    if want_mni:
        log_space_sequence.append("MNI152NLin2009cAsym")
    LOG.info("Output spaces execution order: %s", ", ".join(log_space_sequence))

    if want_mni:
        runner.add_step(_create_convertwarp_merge_warps_step(
            ref=mni_ref,
            warp1=warp_regref2t1_refined,
            warp2=t1_to_mni_fnirt_warp,
            out_warp=warp_regref2mni,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_applywarp_step(
            in_img=syn_moving,
            ref_img=mni_ref,
            warp=warp_regref2mni,
            out_img=reg_mni_qc_out,
            env=env,
            force=opts.force,
        ))

    runner.add_step(_create_afni_motion_affines_step(
        in_4d=epi_for_proc,
        motion_ref_3d=robust_ref,
        mc_mat_dir=mc_mat_dir,
        out_affines=afni_motion_affines,
        force=opts.force,
    ))

    for space in space_sequence:
        if space == "T1w":
            raw_4d, mean_3d, mask_3d = epi_t1, epi_mean_t1, anat_brain_mask_in_t1
            ref_img, warp_img = t1_ref, warp_sbref2t1_refined
            aroma_out_4d, aroma_out_mean, aroma_work = aroma_clean, aroma_clean_mean, aroma_t1_dir
            registered_derivative_dst = preproc_t1_noaroma if opts.clean_ica_aroma else None
            final_dst = preproc_t1
            resampling_work = opts.work_dir / "resampling_t1w"
        else:
            raw_4d, mean_3d, mask_3d = epi_mni, epi_mean_mni, anat_brain_mask_in_mni
            ref_img, warp_img = mni_ref, warp_epi2mni
            aroma_out_4d, aroma_out_mean, aroma_work = aroma_clean_mni, aroma_clean_mean_mni, aroma_mni_dir
            registered_derivative_dst = preproc_mni_noaroma if opts.clean_ica_aroma else None
            final_dst = preproc_mni
            resampling_work = opts.work_dir / "resampling_mni"

        space_name = f"Output Space: {space}" if (space != "T1w" or want_t1) else "Preparing Required T1w Source"
        LOG.info("Constructing %s", space_name)
        world_warp = resampling_work / "warp_world.nii.gz"
        afni_warp = resampling_work / "warp_afni_lps.nii.gz"
        runner.add_step(_create_world_warp_step(
            run_child=runner.run_child,
            motion_ref_3d=robust_ref,
            ref_3d=ref_img,
            fnirt_warp=warp_img,
            world_warp=world_warp,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_afni_warp_step(
            world_warp=world_warp,
            motion_ref_3d=robust_ref,
            ref_3d=ref_img,
            afni_warp=afni_warp,
            force=opts.force,
        ))
        runner.add_step(_create_afni_bold_resampling_step(
            run_child=runner.run_child,
            in_4d=epi_for_proc,
            motion_ref_3d=robust_ref,
            ref_3d=ref_img,
            afni_warp=afni_warp,
            motion_affines=afni_motion_affines,
            out_4d=raw_4d,
            env=env,
            force=opts.force,
        ))
        runner.add_step(_create_temporal_mean_step(
            in_4d=raw_4d,
            out_3d=mean_3d,
            env=env,
            force=opts.force,
            chunk_vols=opts.io_chunk_vols,
        ))
        runner.add_step(create_mask_resampling_step(
            src_mask=anat_brain_mask,
            ref_img=mean_3d,
            out_mask=mask_3d,
            force=opts.force,

        ))

        if registered_derivative_dst is not None:
            runner.add_step(
                create_copy_nifti_step(
                    src=raw_4d,
                    dst=registered_derivative_dst,
                    force=opts.force,
                )
            )

        final_4d = raw_4d
        final_mean = mean_3d
        if opts.clean_ica_aroma:
            denoise_type = str(opts.ica_aroma_denoise_type).strip().lower()
            if denoise_type not in {"nonaggr", "aggr", "both"}:
                raise SystemExit(f"Unsupported ICA-AROMA denoise type: {denoise_type!r}")
            support_brain = aroma_work / "epi_support_bet.nii.gz"
            support_mask = aroma_work / "epi_support_bet_mask.nii.gz"
            regression_mask = aroma_work / "regression_mask.nii.gz"
            policy_path = aroma_work / "ica_aroma_policy.json"
            runner.add_step(_create_epi_support_step(
                epi_mean=mean_3d,
                support_brain=support_brain,
                support_mask=support_mask,
                work_dir=aroma_work,
                env=env,
                force=opts.force,
            ))
            if space == "T1w":
                melodic_mask = aroma_work / "melodic_mask.nii.gz"
                melodic_input = aroma_work / "melodic_input_smooth6mm.nii.gz"
                runner.add_step(_create_dilated_anatomical_mask_step(
                    anatomical_mask=mask_3d,
                    support_mask=support_mask,
                    output=melodic_mask,
                    dilation_mm=_ICA_AROMA_MELODIC_MASK_DILATION_MM,
                    role="MELODIC Estimation",
                    force=opts.force,
                ))
                runner.add_step(_create_dilated_anatomical_mask_step(
                    anatomical_mask=mask_3d,
                    support_mask=support_mask,
                    output=regression_mask,
                    dilation_mm=_ICA_AROMA_REGRESSION_MASK_DILATION_MM,
                    role="ICA Regression",
                    force=opts.force,
                ))
                runner.add_step(_create_melodic_smoothing_step(
                    epi=raw_4d,
                    output=melodic_input,
                    env=env,
                    force=opts.force,
                ))
                aroma_anat = aroma_work / "anat"
                t1_to_mni_itk = aroma_anat / "t1_to_mni_itk_warp.nii.gz"
                t1_to_mni_warp = aroma_anat / "t1_to_mni_warp.nii.gz"
                runner.add_step(_create_ants_composite_to_itk_warp_step(
                    composite_xfm=t1_to_mni_xfm,
                    ref_img=anat_mni_template,
                    out_warp=t1_to_mni_itk,
                    env=env,
                    force=opts.force,
                ))
                runner.add_step(_create_wb_convert_itk_warp_to_fnirt_step(
                    itk_warp=t1_to_mni_itk,
                    src_space_ref=anat_t1,
                    out_warp=t1_to_mni_warp,
                    env=env,
                    force=opts.force,
                ))
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
                aroma_outputs: list[Path] = [aroma_dir / "classification_overview.txt"]
                if denoise_type in {"nonaggr", "both"}:
                    aroma_outputs.append(aroma_dir / "denoised_func_data_nonaggr.nii.gz")
                if denoise_type in {"aggr", "both"}:
                    aroma_outputs.append(aroma_dir / "denoised_func_data_aggr.nii.gz")
                runner.add_step(_create_ica_aroma_workflow_step(
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
                    mni_reference=anat_mni_template,
                    configured_command=opts.ica_aroma_cmd,
                    repetition_time=repetition_time,
                    denoise_type=denoise_type,
                    env=env,
                    force=opts.force,
                ))
                cleaned_epi = aroma_dir / (
                    "denoised_func_data_aggr.nii.gz"
                    if denoise_type == "aggr"
                    else "denoised_func_data_nonaggr.nii.gz"
                )
                runner.add_step(create_copy_nifti_step(
                    src=cleaned_epi,
                    dst=aroma_out_4d,
                    force=opts.force,
                    step_name="Install ICA-AROMA Denoised BOLD",
                ))
                runner.add_step(_create_temporal_mean_step(
                    in_4d=aroma_out_4d,
                    out_3d=aroma_out_mean,
                    env=env,
                    force=opts.force,
                    chunk_vols=opts.io_chunk_vols,
                ))
                runner.add_step(create_json_step(
                    step_name="Write ICA-AROMA Estimation Policy",
                    path=policy_path,
                    payload=_ica_aroma_policy_payload(
                        input_is_mni=False,
                        denoise_type=denoise_type,
                        repetition_time=repetition_time,
                        external_aroma=opts.ica_aroma_cmd is not None,
                    ),
                    inputs=(aroma_out_4d, aroma_out_mean),
                    force=opts.force,
                ))
                final_4d = aroma_out_4d
                final_mean = aroma_out_mean
            else:
                shared_aroma_dir = aroma_t1_dir / "aroma"
                shared_mixing = shared_aroma_dir / "melodic.ica" / "melodic_mix"
                shared_classified = shared_aroma_dir / "classified_motion_ICs.txt"
                shared_policy = aroma_t1_dir / "ica_aroma_policy.json"
                runner.add_step(_create_dilated_anatomical_mask_step(
                    anatomical_mask=mask_3d,
                    support_mask=support_mask,
                    output=regression_mask,
                    dilation_mm=_ICA_AROMA_REGRESSION_MASK_DILATION_MM,
                    role=f"{space} ICA Regression",
                    force=opts.force,
                ))
                aroma_dir = aroma_work / "aroma"
                denoised_outputs: list[Path] = []
                if denoise_type in {"nonaggr", "both"}:
                    denoised_outputs.append(aroma_dir / "denoised_func_data_nonaggr.nii.gz")
                if denoise_type in {"aggr", "both"}:
                    denoised_outputs.append(aroma_dir / "denoised_func_data_aggr.nii.gz")
                runner.add_step(_create_shared_aroma_regression_step(
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
                    env=env,
                    force=opts.force,
                ))
                cleaned_epi = aroma_dir / (
                    "denoised_func_data_aggr.nii.gz"
                    if denoise_type == "aggr"
                    else "denoised_func_data_nonaggr.nii.gz"
                )
                runner.add_step(create_copy_nifti_step(
                    src=cleaned_epi,
                    dst=aroma_out_4d,
                    force=opts.force,
                    step_name=f"Install {space} ICA-AROMA Denoised BOLD",
                ))
                runner.add_step(_create_temporal_mean_step(
                    in_4d=aroma_out_4d,
                    out_3d=aroma_out_mean,
                    env=env,
                    force=opts.force,
                    chunk_vols=opts.io_chunk_vols,
                ))
                runner.add_step(create_json_step(
                    step_name=f"Write {space} ICA-AROMA Regression Policy",
                    path=policy_path,
                    payload=_ica_aroma_shared_regression_policy_payload(
                        input_space=space,
                        denoise_type=denoise_type,
                        repetition_time=repetition_time,
                        shared_work_dir=aroma_t1_dir,
                    ),
                    inputs=(aroma_out_4d, aroma_out_mean),
                    force=opts.force,
                ))
                final_4d = aroma_out_4d
                final_mean = aroma_out_mean

        runner.add_step(
            create_copy_nifti_step(
                src=final_4d,
                dst=final_dst,
                force=opts.force,
            )
        )
        final_output_4d = final_dst

        final_sources_4d[space] = final_output_4d
        final_sources_mean[space] = final_mean
        final_masks[space] = mask_3d

        if space == "T1w":
            t1_final_4d = final_output_4d
            t1_final_mean = final_mean

    confounds_space = "T1w" if "T1w" in final_sources_4d else "MNI152NLin2009cAsym"
    confounds_source_4d = final_sources_4d[confounds_space]
    confounds_source_mean = final_sources_mean[confounds_space]
    confounds_mask = final_masks[confounds_space]

    runner.add_step(_create_confounds_step(
        epi_4d=confounds_source_4d,
        epi_mean_3d=confounds_source_mean,
        mc_dir=mc_dir,
        subjects_dir=subjects_dir,
        fs_subject=fs_subject,
        brain_mask_in_epi=confounds_mask,
        out_tsv=confounds_tsv,
        out_json=confounds_json,
        force=opts.force,
    ))

    if (not use_syn_fallback) and topup_native is not None and warp_sbref is not None:
        runner.add_step(
            create_copy_nifti_step(
                src=field_hz_regref,
                dst=fmap_field_hz_out,
                force=opts.force,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        runner.add_step(
            create_copy_nifti_step(
                src=topup_native.out_prefix.with_name(
                    "topup_results_fieldcoef.nii.gz"
                ),
                dst=fmap_topup_coeff_out,
                force=opts.force,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        runner.add_step(
            create_copy_nifti_step(
                src=warp_sbref,
                dst=fmap_sdc_warp_out,
                force=opts.force,
                step_name="Finalize Fieldmap Derivative",
            )
        )
        if warp_bold_to_reg_ref is not None:
            runner.add_step(create_copy_nifti_step(
                src=warp_bold_to_reg_ref,
                dst=fmap_bold_sdc_warp_out,
                force=opts.force,
                step_name="Finalize BOLD-Readout SDC Warp",
            ))
        jac_sbref = sdc_dir / f"Jacobian_{reg_ref_tag}Space.nii.gz"
        if opts.use_jacobian:
            runner.add_step(
                create_copy_nifti_step(
                    src=jac_sbref,
                    dst=fmap_jacobian_out,
                    force=opts.force,
                    step_name="Finalize Fieldmap Derivative",
                )
            )
        if synthetic_ref is not None:
            runner.add_step(create_copy_nifti_step(
                src=synthetic_ref,
                dst=fmap_synbold_ref_out,
                force=opts.force,
                step_name="Finalize SynBOLD-DisCo Reference",
            ))
            assert synbold_rigid_qc is not None
            runner.add_step(create_copy_nifti_step(
                src=synbold_rigid_qc,
                dst=fmap_synbold_rigid_out,
                force=opts.force,
                step_name="Finalize SynBOLD Rigid-Registration QC",
            ))

    if opts.clean_ica_aroma:
        runner.add_step(create_copy_nifti_step(
            src=aroma_t1_dir / "aroma" / "melodic.ica" / "melodic_IC.nii.gz",
            dst=melodic_ic_t1_out,
            force=opts.force,
            step_name="Finalize ICA-AROMA Derivative",
        ))

    final_preproc_source = final_sources_4d.get("T1w", confounds_source_4d)
    final_preproc_mean = final_sources_mean.get("T1w", confounds_source_mean)
    final_mni_source_4d = final_sources_4d.get("MNI152NLin2009cAsym", epi_mni)
    final_t1_surface_source_4d = final_sources_4d.get("T1w", final_preproc_source)
    noaroma_t1_surface_source_4d = preproc_t1_noaroma if opts.clean_ica_aroma else final_t1_surface_source_4d
    fsnative_metric_outputs = preproc_fsnative if want_fsnative else {
        "L": surf_dir / _with_suffix(f"{run_base}_space-fsnative_hemi-L", "_desc-preproc_internal.func.gii"),
        "R": surf_dir / _with_suffix(f"{run_base}_space-fsnative_hemi-R", "_desc-preproc_internal.func.gii"),
    }
    if need_surface_outputs:
        for hemi in ("L", "R"):
            if opts.clean_ica_aroma:
                runner.add_step(_create_wb_volume_to_surface_mapping_step(
                    volume=noaroma_t1_surface_source_4d,
                    midthickness=fsnative_surfaces[f"{hemi}.midthickness"],
                    white=fsnative_surfaces[f"{hemi}.white"],
                    pial=fsnative_surfaces[f"{hemi}.pial"],
                    out_metric=preproc_fsnative_noaroma[hemi],
                    env=env,
                    force=opts.force,
                ))
            runner.add_step(_create_wb_volume_to_surface_mapping_step(
                volume=final_t1_surface_source_4d,
                midthickness=fsnative_surfaces[f"{hemi}.midthickness"],
                white=fsnative_surfaces[f"{hemi}.white"],
                pial=fsnative_surfaces[f"{hemi}.pial"],
                out_metric=fsnative_metric_outputs[hemi],
                env=env,
                force=opts.force,
            ))
            if want_fsaverage:
                if opts.clean_ica_aroma:
                    runner.add_step(_create_wb_metric_resample_step(
                        in_metric=preproc_fsnative_noaroma[hemi],
                        current_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.current"],
                        new_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.new"],
                        out_metric=preproc_fsaverage_noaroma[hemi],
                        env=env,
                        force=opts.force,
                    ))
                runner.add_step(_create_wb_metric_resample_step(
                    in_metric=fsnative_metric_outputs[hemi],
                    current_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.current"],
                    new_sphere=fsnative_to_fsaverage_spheres[f"{hemi}.new"],
                    out_metric=preproc_fsaverage[hemi],
                    env=env,
                    force=opts.force,
                ))

    clean_inputs: dict[str, object] = {
        "desc-preproc": {
            "volumes": [str(preproc_t1), *([str(preproc_mni)] if want_mni else [])],
            "surfaces": {
                **(
                    {"fsnative": {hemi: str(path) for hemi, path in preproc_fsnative.items()}}
                    if want_fsnative else {}
                ),
                **(
                    {"fsaverage": {hemi: str(path) for hemi, path in preproc_fsaverage.items()}}
                    if want_fsaverage else {}
                ),
            },
        }
    }
    if opts.clean_ica_aroma:
        clean_inputs["desc-preprocNoAROMA"] = {
            "volumes": [
                str(preproc_t1_noaroma),
                *([str(preproc_mni_noaroma)] if want_mni else []),
            ],
            "surfaces": {
                **(
                    {"fsnative": {hemi: str(path) for hemi, path in preproc_fsnative_noaroma.items()}}
                    if want_fsnative else {}
                ),
                **(
                    {"fsaverage": {hemi: str(path) for hemi, path in preproc_fsaverage_noaroma.items()}}
                    if want_fsaverage else {}
                ),
            },
        }

    public_images = [
        preproc_t1,
        *([preproc_t1_noaroma] if opts.clean_ica_aroma else []),
        *([preproc_mni] if want_mni else []),
        *([preproc_mni_noaroma] if want_mni and opts.clean_ica_aroma else []),
        *(list(preproc_fsnative.values()) if want_fsnative else []),
        *(list(preproc_fsnative_noaroma.values()) if want_fsnative and opts.clean_ica_aroma else []),
        *(list(preproc_fsaverage.values()) if want_fsaverage else []),
        *(list(preproc_fsaverage_noaroma.values()) if want_fsaverage and opts.clean_ica_aroma else []),
        boldref_t1_out,
        reg_prenonlinear_qc_out,
        reg_base_qc_out,
        reg_refine_qc_out,
        anat_brain_mask_in_t1,
        *([reg_mni_qc_out] if want_mni else []),
        *([melodic_ic_t1_out] if opts.clean_ica_aroma else []),
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
    public_files = tuple(public_images) + (confounds_tsv, confounds_json)

    publication_identity = {
        "manifest_version": 2,
        "module": "func",
        "run_stem": run_base,
        "complete": True,
        "inputs": {
            "epi": str(inputs.epi),
            "epi_metadata": [str(path) for path in epi_metadata_sources],
            "sbref": str(inputs.sbref) if inputs.sbref is not None else None,
            "se1": str(inputs.se1) if inputs.se1 is not None else None,
            "se2": str(inputs.se2) if inputs.se2 is not None else None,
            "anatomical_manifest": str(anat_manifest),
        },
        "options": configuration,
        "public_outputs": {
            "clean_inputs": clean_inputs,
            "confounds_tsv": str(confounds_tsv),
            "confounds_json": str(confounds_json),
            "files": [str(path) for path in public_files],
        },
    }

    def publication_payload() -> dict[str, object]:
        selection = read_json(selected_reference.metadata)
        motion_indices: list[int] = []
        classified = aroma_t1_dir / "aroma" / "classified_motion_ICs.txt"
        if opts.clean_ica_aroma and classified.is_file():
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
                "final_resampling": final_resampling_metadata(),
                "fieldmap_transfer": fieldmap_transfer_details,
                "pe_residual_refinement": pe_residual_details,
            },
            "denoising": {
                "applied": bool(opts.clean_ica_aroma),
                "method": "ICA-AROMA" if opts.clean_ica_aroma else None,
                "mode": opts.ica_aroma_denoise_type if opts.clean_ica_aroma else None,
                "removed_noise_ic_indices": motion_indices,
            },
        }

    def publish() -> None:
        payload = publication_payload()
        for image_path, metadata_path in zip(public_images, metadata_outputs):
            space = bids_entity(image_path, "space", default=None)
            write_json(
                metadata_path,
                {
                    **epi_input_meta,
                    "Description": "nro functional preprocessing derivative.",
                    "Sources": [str(inputs.epi), str(anat_manifest)],
                    "SpatialReference": space,
                    "Registration": payload["registration"],
                    "Denoising": payload["denoising"],
                    "Configuration": configuration["configuration"],
                    "ConfigurationFingerprint": configuration["configuration_fingerprint"],
                },
            )
        write_json(publication_manifest, payload)

    def validate_publication() -> tuple[bool, str]:
        try:
            current = read_json(publication_manifest)
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
                "final_resampling": final_resampling_metadata(),
                "fieldmap_transfer": fieldmap_transfer_details,
                "pe_residual_refinement": pe_residual_details,
            }.items()
        ):
            return False, "Functional registration publication differs from the requested module."
        denoising = current.get("denoising")
        if not isinstance(denoising, dict) or any(
            denoising.get(key) != expected
            for key, expected in {
                "applied": bool(opts.clean_ica_aroma),
                "method": "ICA-AROMA" if opts.clean_ica_aroma else None,
                "mode": opts.ica_aroma_denoise_type if opts.clean_ica_aroma else None,
            }.items()
        ):
            return False, "Functional denoising publication differs from the requested module."
        missing = [
            str(path)
            for path in (*public_files, *metadata_outputs)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "Functional publication is missing outputs: " + ", ".join(missing)
        return True, "Functional publication is complete and current."

    runner.add_step(Step.python(
        name="Publish Functional Derivatives",
        outputs=(*metadata_outputs, publication_manifest),
        inputs=(*public_files, selected_reference.metadata, robust_reference_metadata),
        force=opts.force,
        action=publish,
        validate=validate_publication,
        completion_boundary=True,
    ))
    return runner


def run(inputs: Inputs, opts: Options) -> None:
    runner_started = time.perf_counter()
    runner = build_module(inputs, opts)
    with runner.run_context(started_at=runner_started):
        runner.execute()


def _build_argparser() -> argparse.ArgumentParser:
    cfg = SETTINGS.preprocess
    p = argparse.ArgumentParser(prog="nro.func.module", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument("--sbref", type=Path)
    p.add_argument("--epi", type=Path)
    p.add_argument("--se1", type=Path)
    p.add_argument("--se2", type=Path)
    p.add_argument("--se1-json", type=Path)
    p.add_argument("--se2-json", type=Path)
    p.add_argument("--epi-json", type=Path)
    p.add_argument("--sbref-json", type=Path)
    p.add_argument("--run-stem", type=str, default=None, help="Minimal-mode BIDS run identifier, e.g. sub-c001_ses-ex123_task-langloc_run-01")
    p.add_argument(
        "--sdc-from-sbref-pair",
        action="store_true",
        default=bool(cfg.sdc_from_sbref_pair),
        help="In minimal mode, resolve the topup pair from opposite-PE SBRefs instead of fmap/ SE fieldmaps.",
    )

    p.add_argument("--bbregister-surf", choices=["white", "pial"], default=cfg.bbregister_surf)
    p.add_argument("--bbregister-init", choices=["coreg", "fsl", "header", "rr"], default=cfg.bbregister_init)
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
    p.add_argument("--synbold-overlap-erosion-voxels", type=int, default=int(cfg.synbold_overlap_erosion_voxels))
    p.add_argument("--synbold-min-overlap-voxels", type=int, default=int(cfg.synbold_min_overlap_voxels))
    p.add_argument("--synbold-max-rigid-translation-mm", type=float, default=float(cfg.synbold_max_rigid_translation_mm))
    p.add_argument("--synbold-max-rigid-rotation-degrees", type=float, default=float(cfg.synbold_max_rigid_rotation_degrees))
    p.add_argument("--sbref-max-rigid-displacement-mm", type=float, default=float(cfg.sbref_max_rigid_displacement_mm))
    p.add_argument("--sbref-max-rigid-rotation-degrees", type=float, default=float(cfg.sbref_max_rigid_rotation_degrees))
    p.add_argument("--sbref-min-support-overlap", type=float, default=float(cfg.sbref_min_support_overlap))
    p.add_argument("--sbref-min-intensity-correlation", type=float, default=float(cfg.sbref_min_intensity_correlation))
    p.add_argument("--topup-config", type=str, default=cfg.topup_config)
    p.add_argument("--ica-aroma-cmd", type=Path, default=cfg.ica_aroma_cmd)
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
    p.add_argument("--syn-base-smoothing-sigmas", type=str, default=str(cfg.syn_base_smoothing_sigmas))
    p.add_argument("--syn-refine-transform", type=str, default=str(cfg.syn_refine_transform))
    p.add_argument("--syn-refine-convergence", type=str, default=str(cfg.syn_refine_convergence))
    p.add_argument("--syn-refine-shrink-factors", type=str, default=str(cfg.syn_refine_shrink_factors))
    p.add_argument("--syn-refine-smoothing-sigmas", type=str, default=str(cfg.syn_refine_smoothing_sigmas))
    p.add_argument("--no-ica-aroma", action="store_true", default=not bool(cfg.clean_ica_aroma), help="Skip ICA-AROMA cleaning.")
    p.add_argument("--ica-aroma-denoise-type", choices=["nonaggr", "aggr", "both"], default=cfg.ica_aroma_denoise_type)

    p.add_argument("--project", default=SETTINGS.common.project, help="BIDS project name under the configured top-level data directory.")
    p.add_argument(
        "--preprocessing-id",
        default=SETTINGS.common.preprocessing_id,
        help="Preprocessing collection name under derivatives/preprocessing/.",
    )
    p.add_argument("--sub-id", required=True, help="Subject identifier, e.g. sub-c001")
    p.add_argument("--ses-id", default=None, help="Optional session identifier, e.g. ses-ex31524")
    p.add_argument("--work-dir", type=Path, default=cfg.work_dir)

    p.add_argument("--nthreads", type=int, default=max(int(cfg.nthreads_min), (os.cpu_count() or 1) // int(cfg.nthreads_divisor)))
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
    p.add_argument("--force", action="store_true", default=cfg.force)
    p.add_argument(
        "--output-spaces",
        nargs="+",
        default=list(cfg.output_spaces),
        help="Subset of output spaces to generate. Choices: T1w fsnative MNI152NLin2009cAsym fsaverage",
    )
    p.add_argument("--verbose", action="store_true", default=cfg.verbose)

    # Container
    p.add_argument("--container", type=Path, default=DEFAULT_QUNEX_CONTAINER)
    p.add_argument("--no-container", action="store_true", default=cfg.no_container)
    p.add_argument("--container-engine", type=str, default=cfg.container_engine)
    p.add_argument("--container-no-cleanenv", action="store_true", default=not bool(cfg.container_cleanenv))
    p.add_argument("--container-bind", action="append", default=list(cfg.container_bind))
    p.add_argument("--container-home", type=Path, default=cfg.container_home)
    p.add_argument("--container-inner-setup", type=str, default=cfg.container_inner_setup)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and run the functional module."""
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    project = str(args.project)
    explicit_mode = args.epi is not None or args.epi_json is not None
    minimal_mode = bool(str(args.run_stem or "").strip())
    if explicit_mode and minimal_mode:
        raise SystemExit("Use either explicit input paths (--epi/--epi-json/...) or minimal mode (--run-stem), not both.")
    if (not explicit_mode) and (not minimal_mode):
        raise SystemExit("Either explicit input paths (--epi and --epi-json) or minimal mode (--run-stem) is required.")

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
        args.sbref_json = (
            resolved.sbref.metadata_path if resolved.sbref is not None else None
        )
        args.se1 = resolved.pair.se1.img if resolved.pair is not None else None
        args.se2 = resolved.pair.se2.img if resolved.pair is not None else None
        args.se1_json = (
            resolved.pair.se1.metadata_path if resolved.pair is not None else None
        )
        args.se2_json = (
            resolved.pair.se2.metadata_path if resolved.pair is not None else None
        )
        if resolved.selection_warning:
            LOG.info("Resolved %s -> %s (%s)", resolved.requested_run_stem, resolved.resolved_run_stem, resolved.selection_warning)
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
        out_dir = preprocess_subject_func_dir(str(args.sub_id), project=project, preprocessing_id=str(args.preprocessing_id))
        work_session_dir = preprocess_subject_func_work_dir(str(args.sub_id), project=project, preprocessing_id=str(args.preprocessing_id))
    else:
        out_dir = preprocess_session_func_dir(str(args.sub_id), ses_id, project=project, preprocessing_id=str(args.preprocessing_id))
        work_session_dir = preprocess_session_func_work_dir(str(args.sub_id), ses_id, project=project, preprocessing_id=str(args.preprocessing_id))
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

    container: Optional[ContainerSpec]
    if args.no_container:
        container = None
    else:
        container_home = args.container_home or (work_dir / "_qunex_home")
        container = ContainerSpec(
            image=Path(args.container),
            engine=str(args.container_engine),
            cleanenv=not bool(args.container_no_cleanenv),
            extra_binds=tuple(args.container_bind or []),
            home_dir=Path(container_home),
            inner_setup=str(args.container_inner_setup or ""),
        )

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
        preprocessing_id=str(args.preprocessing_id),
        sub_id=str(args.sub_id),
        ses_id=ses_id,
        nthreads=int(args.nthreads),
        force=bool(args.force),
        output_grid=str(args.output_grid),
        topup_config=str(args.topup_config),
        ica_aroma_cmd=(Path(args.ica_aroma_cmd) if args.ica_aroma_cmd is not None else None),
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
        clean_ica_aroma=not bool(args.no_ica_aroma),
        ica_aroma_denoise_type=str(args.ica_aroma_denoise_type),
        bbregister_surf=str(args.bbregister_surf),
        bbregister_init=str(args.bbregister_init),
        bbregister_dof=int(args.bbregister_dof),
        container=container,
        debug_first_nvols=int(args.debug_first_nvols),
        output_spaces=_normalize_output_spaces(args.output_spaces),
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

    run(inputs, opts)


if __name__ == "__main__":
    main()
