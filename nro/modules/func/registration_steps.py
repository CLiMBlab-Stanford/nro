"""Registration steps for functional preprocessing."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from nro.engine.execution import (
    ensure_directory,
)
from nro.engine.images import (
    nifti_is_valid,
    nifti_zooms_xyz,
)
from nro.engine.io import write_json
from nro.engine.registration import validate_rigid_transform
from nro.orchestration.runner_graph import Step

from .constants import (
    _FIELDMAP_TRANSFER_POLICY_VERSION,
)
from .sdc_steps import _pe_to_fsl_shift_direction
from .step_support import LOG


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
