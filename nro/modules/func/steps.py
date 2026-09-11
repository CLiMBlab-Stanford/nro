"""Build functional preprocessing steps and their validation helpers."""

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from nro.configuration.runtime import SETTINGS
from nro.engine.bids import (
    bids_entity,
    bids_readout_time,
)
from nro.engine.execution import (
    ensure_directory,
    new_step_counter,
    resolve_runner_command,
    runner_path_exists,
)
from nro.engine.execution import (
    strip_ansi as _strip_ansi,
)
from nro.engine.images import (
    copy_or_convert_nifti,
    nifti_is_valid,
    nifti_zooms_xyz,
)
from nro.engine.io import atomic_output_path, read_json, write_json
from nro.engine.registration import rigid_transform_metrics, validate_rigid_transform
from nro.modules.func.confounds import get_confounds
from nro.modules.func.contracts import (
    FINAL_RESAMPLING_INTERPOLATION,
    FINAL_WARP_INTERPOLATION,
)
from nro.modules.func.ica_aroma import denoising as run_ica_aroma_denoising
from nro.modules.func.ica_aroma import make_dilated_anatomical_epi_mask, run_ica_aroma_workflow
from nro.modules.func.resampling import (
    validate_afni_motion_affines,
    validate_resampled_bold,
    write_afni_motion_affines,
    write_afni_warp,
)
from nro.orchestration.runner import (
    Runner,
    shlex_quote,
    write_completion_breadcrumb,
)
from nro.orchestration.runner_graph import Step

from .constants import (
    _FIELDMAP_TRANSFER_POLICY_VERSION,
    _ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD,
    _ICA_AROMA_ESTIMATION_POLICY_VERSION,
    _ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM,
    _ICA_AROMA_MELODIC_MASK_DILATION_MM,
    _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
    _NONSTEADY_DETECTION_POLICY_VERSION,
)

next_step = new_step_counter()
LOG = logging.getLogger("preprocess")


def _resolve_sdc_reference_policy(
    *,
    requested_sdc_method: str,
    fieldmap_pair_available: bool,
    fieldmap_syn_refine: bool,
) -> tuple[bool, Optional[str]]:
    """Resolve synthetic-reference use and the post-fieldmap refinement target."""
    use_synbold_reference = requested_sdc_method == "synbold_disco" and not fieldmap_pair_available
    fieldmap_refinement_target = (
        "T1wAnatomicalSyN" if fieldmap_pair_available and fieldmap_syn_refine else None
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
    phase_encoding_direction = str(bold_metadata.get("PhaseEncodingDirection", "")).strip()
    if phase_encoding_direction not in {"i", "i-", "j", "j-", "k", "k-"}:
        missing.append("PhaseEncodingDirection")
    try:
        readout_time = float(bids_readout_time(bold_metadata))
        if readout_time <= 0:
            raise ValueError("readout time must be positive")
    except (KeyError, TypeError, ValueError):
        missing.append("TotalReadoutTime or EffectiveEchoSpacing with a phase-encoding matrix size")
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
        "flirt",
        "-in",
        str(distorted_reference),
        "-ref",
        str(anatomical_t1),
        "-usesqform",
        "-applyxfm",
        "-omat",
        str(header_mat),
        "-out",
        str(header_aligned),
    ]
    rigid_mat = rigid_mat_out or (work_dir / "epi_reg_d.mat")
    rigid_qc = rigid_qc_out or (work_dir / "epi_reg_d.nii.gz")
    search = str(float(max_rotation_degrees))
    rigid_cmd = [
        "flirt",
        "-in",
        str(distorted_reference),
        "-ref",
        str(anatomical_t1),
        "-init",
        str(header_mat),
        "-dof",
        "6",
        "-cost",
        "corratio",
        "-searchrx",
        f"-{search}",
        search,
        "-searchry",
        f"-{search}",
        search,
        "-searchrz",
        f"-{search}",
        search,
        "-omat",
        str(rigid_mat),
        "-out",
        str(rigid_qc),
        "-interp",
        "trilinear",
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
                    "flirt",
                    "-in",
                    str(distorted_reference),
                    "-ref",
                    str(anatomical_t1),
                    "-init",
                    str(header_mat),
                    "-dof",
                    "6",
                    "-cost",
                    "corratio",
                    "-nosearch",
                    "-omat",
                    str(rigid_mat),
                    "-out",
                    str(rigid_qc),
                    "-interp",
                    "trilinear",
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
        write_json(
            rigid_qc_json,
            {
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
            },
        )

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
        rotation = (
            np.eye(3, dtype=np.float64) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))
        )

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
    mapping = {
        "i": (1, 0, 0),
        "i-": (-1, 0, 0),
        "j": (0, 1, 0),
        "j-": (0, -1, 0),
        "k": (0, 0, 1),
        "k-": (0, 0, -1),
    }
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
        raise SystemExit(
            f"Unsupported PhaseEncodingDirection: {phase_encoding_direction!r}"
        ) from error


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

    from nro.modules.func.confounds import _nonsteady_spikes

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
    volume_count: int,
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
    nvols = int(volume_count)
    if nvols < 1:
        raise ValueError("Robust BOLD reference construction requires at least one volume")
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
            [
                "mcflirt",
                "-in",
                str(source),
                "-out",
                str(output),
                "-reffile",
                str(reference),
                "-mats",
                "-plots",
            ],
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
                [
                    "fslroi",
                    str(provisional_mc),
                    str(median_input),
                    str(dropped),
                    str(total_volumes - dropped),
                ],
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
        required = (
            robust_ref,
            robust_metadata,
            nonsteady_metadata,
            final_mc,
            final_par,
            *expected_matrices,
        )
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
            outputs=(
                robust_ref,
                robust_metadata,
                nonsteady_metadata,
                final_mc,
                final_par,
                final_matrices_complete,
            ),
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
        metrics.update(
            {
                "BOLDPhaseEncodingDirection": epi_ped,
                "SBRefPhaseEncodingDirection": sbref_ped,
                "BOLDTotalReadoutTime": epi_readout,
                "SBRefTotalReadoutTime": sbref_readout,
            }
        )
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
        run_child(
            [
                "flirt",
                "-in",
                str(robust_ref),
                "-ref",
                str(candidate_3d),
                "-usesqform",
                "-applyxfm",
                "-omat",
                str(header_mat),
                "-out",
                str(header_image),
            ],
            env=env,
        )
        search = str(float(max_rotation_degrees))
        run_child(
            [
                "flirt",
                "-in",
                str(robust_ref),
                "-ref",
                str(candidate_3d),
                "-init",
                str(header_mat),
                "-dof",
                "6",
                "-cost",
                "normcorr",
                "-searchrx",
                f"-{search}",
                search,
                "-searchry",
                f"-{search}",
                search,
                "-searchrz",
                f"-{search}",
                search,
                "-omat",
                str(selected_mat),
                "-out",
                str(registered),
                "-interp",
                "trilinear",
            ],
            env=env,
        )
        rigid = rigid_transform_metrics(
            matrix=selected_mat, initial_matrix=header_mat, center_mask=sbref_mask
        )
        used_local = (
            rigid["RotationDegrees"] > max_rotation_degrees
            or rigid["CenterDisplacementMillimeters"] > max_displacement_mm
        )
        if used_local:
            run_child(
                [
                    "flirt",
                    "-in",
                    str(robust_ref),
                    "-ref",
                    str(candidate_3d),
                    "-init",
                    str(header_mat),
                    "-dof",
                    "6",
                    "-cost",
                    "normcorr",
                    "-nosearch",
                    "-omat",
                    str(selected_mat),
                    "-out",
                    str(registered),
                    "-interp",
                    "trilinear",
                ],
                env=env,
            )
            rigid = rigid_transform_metrics(
                matrix=selected_mat, initial_matrix=header_mat, center_mask=sbref_mask
            )
        run_child(
            [
                "flirt",
                "-in",
                str(robust_mask),
                "-ref",
                str(candidate_3d),
                "-applyxfm",
                "-init",
                str(selected_mat),
                "-interp",
                "nearestneighbour",
                "-out",
                str(registered_mask),
            ],
            env=env,
        )
        rigid.update(
            _image_overlap_and_correlation(
                moving_registered=registered,
                moving_mask_registered=registered_mask,
                fixed=candidate_3d,
                fixed_mask=sbref_mask,
            )
        )
        rigid["UsedLocalNoSearchFallback"] = used_local
        metrics.update(rigid)
        if rigid["RotationDegrees"] > max_rotation_degrees:
            reasons.append("SBRef rigid rotation exceeds its configured limit.")
        if rigid["CenterDisplacementMillimeters"] > max_displacement_mm:
            reasons.append("SBRef rigid displacement exceeds its configured limit.")
        if rigid["SupportOverlapFraction"] < min_support_overlap:
            reasons.append("SBRef registered support overlap is below its configured minimum.")
        if rigid["IntensityCorrelation"] < min_correlation:
            reasons.append(
                "SBRef registered intensity correlation is below its configured minimum."
            )
        if reasons:
            choose_robust(details)
            return

        run_child(
            [
                "convert_xfm",
                "-inverse",
                "-omat",
                str(selected_to_epi),
                str(selected_mat),
            ],
            env=env,
        )
        run_child(
            [
                "flirt",
                "-in",
                str(candidate_3d),
                "-ref",
                str(robust_ref),
                "-applyxfm",
                "-init",
                str(selected_to_epi),
                "-interp",
                "sinc",
                "-out",
                str(selected_image),
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
            parameters={
                "maximum_rotation_degrees": max_rotation_degrees,
                "maximum_displacement_millimeters": max_displacement_mm,
                "minimum_support_overlap": min_support_overlap,
                "minimum_intensity_correlation": min_correlation,
            },
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
    init_flag = {
        "coreg": "--init-coreg",
        "fsl": "--init-fsl",
        "header": "--init-header",
        "rr": "--init-rr",
    }.get(init)
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
    *,
    run_child: Callable[..., Optional[str]],
    in_img: Path,
    ref_img: Path,
    out_mat: Path,
    work_dir: Path,
    fixed_mask: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    header_mat = work_dir / "header_init.mat"
    header_image = work_dir / "header_init.nii.gz"
    header_cmd = [
        "flirt",
        "-in",
        str(in_img),
        "-ref",
        str(ref_img),
        "-usesqform",
        "-applyxfm",
        "-omat",
        str(header_mat),
        "-out",
        str(header_image),
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
                matrix=out_mat,
                initial_matrix=header_mat,
                max_translation_mm=30.0,
                max_rotation_degrees=20.0,
                center_mask=fixed_mask,
                label="EPI pose registration",
            )
        except SystemExit as searched_error:
            rejected_candidates.append(str(searched_error))
            LOG.warning("%s Retrying locally without a global angle search.", searched_error)
            run_child(
                [
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
                    "-nosearch",
                    "-omat",
                    str(out_mat),
                    "-out",
                    str(registered),
                ],
                env=env,
            )
            selected_candidate = "local_nosearch"
            try:
                selected_metrics = validate_rigid_transform(
                    matrix=out_mat,
                    initial_matrix=header_mat,
                    max_translation_mm=30.0,
                    max_rotation_degrees=20.0,
                    center_mask=fixed_mask,
                    label="local EPI pose registration",
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
                    matrix=out_mat,
                    initial_matrix=header_mat,
                    max_translation_mm=30.0,
                    max_rotation_degrees=20.0,
                    center_mask=fixed_mask,
                    label="header-initialized EPI pose registration",
                )
        write_json(
            rigid_qc_json,
            {
                "RegistrationLabel": "EPI pose registration",
                "CoverageAssumption": "whole_brain",
                "CostFunction": "normcorr",
                "CostFunctionWeighting": "none",
                "SelectedCandidate": selected_candidate,
                "MetricsRelativeToHeaderInitialization": selected_metrics,
                "RejectedCandidates": rejected_candidates,
            },
        )

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
        cmd,
        outputs=(out_mat,),
        inputs=(mat,),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_mat.parent),
    )


def _create_concat_mats_step(
    *, first: Path, second: Path, out_mat: Path, env: dict[str, str], force: bool
) -> Step:
    # FSL concat applies the second matrix first, then the first matrix.
    cmd = ["convert_xfm", "-omat", str(out_mat), "-concat", str(first), str(second)]
    return Step.command_step(
        cmd,
        outputs=(out_mat,),
        inputs=(first, second),
        force=force,
        env=env,
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
        write_json(
            qc_json,
            {
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
            },
        )

    return Step.python(
        name="Refine Post-SDC EPI-to-SE Pose",
        outputs=(out_mat, out_registered, qc_json),
        inputs=(moving, fixed),
        force=force,
        action=refine,
        parameters={
            "maximum_translation_millimeters": max_translation_mm,
            "maximum_rotation_degrees": max_rotation_degrees,
        },
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
    """Declared TOPUP outputs and displacement-field paths for composing SDC transforms."""

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
        prefix.parent / f"{prefix.name}_{index:0{width}d}{extension}" for width in (2, 1, 3, 4)
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
    spatial_shape: tuple[int, int, int],
    volumes_a: int,
    volumes_b: int,
    readout_time_b: Optional[float] = None,
) -> TopupDfOutputs:
    if topup_config.strip().lower() == "auto":
        topup_config = (
            "b02b0_2.cnf" if all(size % 2 == 0 for size in spatial_shape) else "b02b0_1.cnf"
        )
        LOG.info(
            "TOPUP configuration selected for image dimensions %s: %s",
            spatial_shape,
            topup_config,
        )
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

    a_nvols = int(volumes_a)
    b_nvols = int(volumes_b)
    if a_nvols < 1 or b_nvols < 1:
        raise ValueError("TOPUP inputs must each contain at least one volume")
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
    normalized_rbmout = normalized_dir / "MotionMatrix"
    normalized_warps = tuple(
        normalized_dir / f"WarpField_{index:04d}.nii.gz" for index in range(1, total_nvols + 1)
    )
    normalized_jacobians = tuple(
        normalized_dir / f"Jacobian_{index:04d}.nii.gz" for index in range(1, total_nvols + 1)
    )
    normalized_matrices = tuple(
        normalized_dir / f"MotionMatrix_{index:04d}.mat" for index in range(1, total_nvols + 1)
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
        cmd,
        outputs=(out_warp,),
        inputs=(ref, warp1, premat, postmat),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
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
        cmd,
        outputs=(out_warp,),
        inputs=(ref, warp1, postmat),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
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
        cmd,
        outputs=(out_warp,),
        inputs=(ref, premat),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
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
        cmd,
        outputs=(out_warp,),
        inputs=(ref, premat, warp1),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
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
        cmd,
        outputs=(out_warp,),
        inputs=(ref, warp1, warp2),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_warp_jacobian_step(
    *,
    warp: Path,
    ref: Path,
    temporary: Path,
    junk: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    convert_cmd = [
        "convertwarp",
        "--rel",
        "-w",
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
    *,
    temporary: Path,
    junk: Path,
    warp: Path,
    ref: Path,
    out_jac: Path,
    env: dict[str, str],
    force: bool,
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
    cmd = [
        "applywarp",
        "--rel",
        "--interp=spline",
        f"--in={in_img}",
        f"--ref={ref_img}",
        f"--warp={warp}",
        f"--out={out_img}",
    ]
    if premat is not None:
        cmd.append(f"--premat={premat}")
    return Step.command_step(
        cmd,
        outputs=(out_img,),
        inputs=tuple(deps),
        force=force,
        env=env,
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
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore

        img = nib.load(str(in_4d))
        if len(img.shape) != 4:
            raise RuntimeError(
                f"Temporal mean requires a 4D input, got shape {img.shape} for {in_4d}"
            )
        nvols = int(img.shape[3])
        if nvols < 1:
            raise RuntimeError(f"Temporal mean requires at least one volume: {in_4d}")
        block_size = max(1, int(chunk_vols))
        accum = np.zeros(tuple(int(v) for v in img.shape[:3]), dtype=np.float64)
        # Materialize a compressed NIfTI exactly once. Proxy slicing a .nii.gz
        # for each block can restart decompression and multiply disk reads by
        # the number of temporal blocks. Accumulate in float64 so changing the
        # I/O chunk size does not amplify float32 cancellation error.
        data = np.asarray(img.dataobj, dtype=np.float32)
        for start in range(0, nvols, block_size):
            stop = min(start + block_size, nvols)
            block = data[..., start:stop]
            accum += block.sum(axis=3, dtype=np.float64)
        accum /= float(nvols)
        header = img.header.copy()
        header.set_data_shape(accum.shape)
        header.set_data_dtype(np.float32)
        with atomic_output_path(out_3d) as staged:
            nib.save(nib.Nifti1Image(accum.astype(np.float32), img.affine, header), str(staged))
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
    *,
    world_warp: Path,
    motion_ref_3d: Path,
    ref_3d: Path,
    afni_warp: Path,
    force: bool,
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
        cmd,
        outputs=(out_metric,),
        inputs=(volume, midthickness, white, pial),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_metric.parent),
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
        cmd,
        outputs=(out_metric,),
        inputs=(in_metric, current_sphere, new_sphere),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_metric.parent),
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
        if "/fsl-6." in candidate_str or re.search(r"/fsl-[0-9][^/]*/bin/", candidate_str):
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
    *,
    epi_mean: Path,
    support_brain: Path,
    support_mask: Path,
    work_dir: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        [
            "bet",
            str(epi_mean),
            str(support_brain),
            "-R",
            "-f",
            f"{_ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD:.8g}",
            "-g",
            "0",
            "-m",
        ],
        outputs=(support_brain, support_mask),
        inputs=(epi_mean,),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(work_dir),
    )


def _create_dilated_anatomical_mask_step(
    *,
    anatomical_mask: Path,
    support_mask: Path,
    output: Path,
    dilation_mm: float,
    role: str,
    force: bool,
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
    *,
    epi: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
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
                run_out=lambda cmd: runner.run_child(list(cmd), env=env, capture_stdout=True) or "",
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
            "-in",
            str(melodic_input),
            "-out",
            str(aroma_dir),
            "-mc",
            str(motion_parameters),
            "-m",
            str(melodic_mask),
            "-den",
            "no",
        ]
        if not input_is_mni:
            arguments.extend(["-affmat", str(identity_transform), "-warp", str(t1_to_mni_warp)])
        if repetition_time is not None and repetition_time > 0:
            arguments.extend(["-tr", f"{float(repetition_time):.8g}"])
        runner.run_child([aroma_command, *arguments], env=env)
        for product in melodic_products[:-1]:
            if not product.is_file() or product.stat().st_size == 0:
                raise SystemExit(f"ICA-AROMA completed without required MELODIC product: {product}")
        write_completion_breadcrumb(melodic_products[-1], "MELODIC decomposition complete\n")
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
        missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
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
            len(indices),
            input_space,
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
            epi,
            regression_mask,
            mixing_matrix,
            classified_components,
            shared_policy,
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
    cmd = [
        "mri_convert",
        "--voxsize",
        f"{vx:.6f}",
        f"{vy:.6f}",
        f"{vz:.6f}",
        str(t1_image),
        str(out_target),
    ]
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

    def validate() -> tuple[bool, str]:
        return (
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
        cmd,
        outputs=(out_target,),
        inputs=(mni_template,),
        force=force,
        env=env,
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
            (
                path
                for path in (artifact_dir / "raw_1Warp.nii.gz", artifact_dir / "raw_0Warp.nii.gz")
                if path.is_file()
            ),
            None,
        )
        raw_inverse = next(
            (
                path
                for path in (
                    artifact_dir / "raw_1InverseWarp.nii.gz",
                    artifact_dir / "raw_0InverseWarp.nii.gz",
                )
                if path.is_file()
            ),
            None,
        )
        if raw_forward is None or raw_inverse is None:
            raise SystemExit(
                f"ANTs registration did not produce forward and inverse warps under {artifact_dir}"
            )
        shutil.copy2(raw_forward, forward_xfm)
        shutil.copy2(raw_inverse, inverse_xfm)

    def validate_registration() -> tuple[bool, str]:
        missing = [
            str(path)
            for path in (forward_xfm, inverse_xfm)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "ANTs SyN directory is missing required transform(s): " + ", ".join(
                missing
            )
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
        cmd,
        outputs=(out_warp,),
        inputs=(composite_xfm, ref_img),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
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
        cmd,
        outputs=(out_warp,),
        inputs=(itk_warp, src_space_ref),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )
