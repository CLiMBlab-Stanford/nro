"""Resampling steps for functional preprocessing."""

from pathlib import Path
from typing import Callable, Optional

from nro.engine.execution import (
    ensure_directory,
)
from nro.engine.images import (
    nifti_is_valid,
)
from nro.engine.io import atomic_output_path
from nro.modules.func.contract import (
    FINAL_RESAMPLING_INTERPOLATION,
    FINAL_WARP_INTERPOLATION,
)
from nro.modules.func.resampling import (
    validate_afni_motion_affines,
    validate_resampled_bold,
    write_afni_motion_affines,
    write_afni_warp,
)
from nro.orchestration.runner_graph import Step

from .sdc_steps import _motion_matrix_breadcrumb
from .step_support import _space_name_for_log


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
    gradient_warp: Path | None = None,
    out_4d: Path,
    repetition_time: float,
    env: dict[str, str],
    force: bool,
) -> Step:
    step_name = f"Resample BOLD ({_space_name_for_log(out_4d)})"

    def resample() -> None:
        transforms = _afni_bold_warp_chain(
            spatial_warp=afni_warp,
            motion_affines=motion_affines,
            gradient_warp=gradient_warp,
        )
        with atomic_output_path(out_4d) as staged:
            run_child(
                [
                    "3dNwarpApply",
                    "-source",
                    str(in_4d),
                    "-master",
                    str(ref_3d),
                    "-nwarp",
                    transforms,
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
            run_child(
                ["3drefit", "-TR", f"{float(repetition_time):.8g}", str(staged)],
                env=env,
            )
            valid, validation_reason = validate_resampled_bold(
                source_path=in_4d,
                reference_path=ref_3d,
                output_path=staged,
                repetition_time=repetition_time,
            )
            if not valid:
                raise RuntimeError(validation_reason)

    return Step.python(
        name=step_name,
        outputs=(out_4d,),
        inputs=(in_4d, ref_3d, afni_warp, motion_affines, gradient_warp),
        force=force,
        action=resample,
        parameters={"repetition_time": float(repetition_time)},
        validate=lambda: validate_resampled_bold(
            source_path=in_4d,
            reference_path=ref_3d,
            output_path=out_4d,
            repetition_time=repetition_time,
        ),
    )


def _afni_bold_warp_chain(
    *,
    spatial_warp: Path,
    motion_affines: Path,
    gradient_warp: Path | None,
) -> str:
    """Order pull transforms from target space through corrected BOLD to raw BOLD."""
    transforms = [spatial_warp, motion_affines]
    if gradient_warp is not None:
        transforms.append(gradient_warp)
    return " ".join(str(path) for path in transforms)
