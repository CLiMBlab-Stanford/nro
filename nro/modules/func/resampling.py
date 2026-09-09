"""Coordinate conversions for AFNI's single-pass 4D BOLD resampling."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.io import atomic_output_path, atomic_write_text

LPS_FROM_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def voxel_to_fsl(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Return FSL's scaled-voxel affine, including its neurological-image flip."""
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    result = np.diag([*zooms, 1.0])
    if np.linalg.det(image.affine) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = (int(image.shape[0]) - 1) * zooms[0]
        result = flip @ result
    return result


def flirt_to_world(
    flirt: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    """Convert a FLIRT source-to-reference matrix into a RAS+ world affine."""
    source_world_to_fsl = voxel_to_fsl(source) @ np.linalg.inv(source.affine)
    reference_fsl_to_world = reference.affine @ np.linalg.inv(voxel_to_fsl(reference))
    return reference_fsl_to_world @ flirt @ source_world_to_fsl


def plumb_affine(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Return the cardinal affine AFNI uses internally for an oblique grid."""
    orientation = nib.orientations.io_orientation(image.affine)
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    result = np.eye(4)
    result[:3, :3] = 0.0
    for voxel_axis, (world_axis, direction) in enumerate(orientation):
        result[int(world_axis), voxel_axis] = float(direction) * zooms[voxel_axis]
    result[:3, 3] = image.affine[:3, 3]
    return result


def flirt_to_afni_pull(
    flirt: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    """Convert a FLIRT push affine to AFNI's plumb-grid LPS pull form."""
    source_to_reference_ras = flirt_to_world(flirt, source, reference)
    reference_to_source_ras = np.linalg.inv(source_to_reference_ras)
    plumb_pull_ras = (
        plumb_affine(source)
        @ np.linalg.inv(source.affine)
        @ reference_to_source_ras
        @ reference.affine
        @ np.linalg.inv(plumb_affine(reference))
    )
    return LPS_FROM_RAS @ plumb_pull_ras @ LPS_FROM_RAS


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", matrix[:3, :3], points) + matrix[:3, 3]


def world_warp_to_afni(
    world_warp: nib.spatialimages.SpatialImage,
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    slab_depth: int = 8,
) -> nib.Nifti1Image:
    """Express a RAS+ pull displacement in AFNI's plumb-grid LPS coordinates.

    AFNI treats oblique NIfTI grids as cardinal grids while applying nonlinear
    warps.  Transforming both the reference positions and displaced source
    positions into that cardinal representation preserves the intended real
    coordinates without first deobliquing or resampling either image.
    """
    shape = tuple(int(value) for value in reference.shape[:3])
    if tuple(int(value) for value in world_warp.shape[:3]) != shape:
        raise ValueError(
            f"World warp and reference spatial shapes differ: {world_warp.shape[:3]} != {shape}"
        )
    world_data = np.asarray(world_warp.dataobj, dtype=np.float32)
    if world_data.shape != (*shape, 3):
        raise ValueError(
            f"Expected an XYZ world displacement field with shape {(*shape, 3)}, "
            f"got {world_data.shape}"
        )

    source_inverse = np.linalg.inv(source.affine)
    source_plumb = plumb_affine(source)
    reference_plumb = plumb_affine(reference)
    displacement_lps = np.empty((*shape, 3), dtype=np.float32)
    depth = max(1, int(slab_depth))
    for start in range(0, shape[2], depth):
        stop = min(start + depth, shape[2])
        grid = np.moveaxis(
            np.indices((shape[0], shape[1], stop - start), dtype=np.float32),
            0,
            -1,
        )
        grid[..., 2] += start
        reference_real = _apply_affine(reference.affine, grid)
        source_real = reference_real + world_data[:, :, start:stop, :]
        source_voxel = _apply_affine(source_inverse, source_real)
        reference_cardinal = _apply_affine(reference_plumb, grid)
        source_cardinal = _apply_affine(source_plumb, source_voxel)
        displacement_lps[:, :, start:stop, :] = np.einsum(
            "ij,...j->...i",
            LPS_FROM_RAS[:3, :3],
            source_cardinal - reference_cardinal,
        ).astype(np.float32)

    header = reference.header.copy()
    header.set_data_shape((*shape, 3))
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(displacement_lps, reference_plumb, header)


def write_afni_motion_affines(
    *,
    source_path: Path,
    motion_reference_path: Path,
    matrix_dir: Path,
    output_path: Path,
) -> None:
    """Consolidate one MCFLIRT matrix per BOLD volume into an AFNI 1D file."""
    source = nib.load(str(source_path))
    reference = nib.load(str(motion_reference_path))
    if len(source.shape) != 4 or int(source.shape[3]) < 1:
        raise ValueError(f"Expected a nonempty 4D BOLD image: {source_path}")
    rows: list[np.ndarray] = []
    for index in range(int(source.shape[3])):
        matrix_path = matrix_dir / f"MAT_{index:04d}"
        if not matrix_path.is_file() or matrix_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing MCFLIRT matrix: {matrix_path}")
        matrix = np.asarray(np.loadtxt(matrix_path), dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 MCFLIRT matrix in {matrix_path}")
        rows.append(flirt_to_afni_pull(matrix, source, reference)[:3, :].reshape(-1))
    text = "\n".join(" ".join(f"{value:.12g}" for value in row) for row in rows)
    atomic_write_text(output_path, text + "\n")


def validate_afni_motion_affines(
    *,
    source_path: Path,
    affine_path: Path,
) -> tuple[bool, str]:
    """Check that an AFNI affine file contains one twelve-value row per volume."""
    try:
        source = nib.load(str(source_path))
        rows = np.atleast_2d(np.loadtxt(affine_path, dtype=np.float64))
    except (OSError, ValueError) as error:
        return False, f"AFNI motion-affine file is unreadable: {error}"
    expected_rows = int(source.shape[3]) if len(source.shape) == 4 else 0
    if rows.shape != (expected_rows, 12):
        return False, (f"AFNI motion-affine shape is {rows.shape}; expected {(expected_rows, 12)}")
    if not np.all(np.isfinite(rows)):
        return False, "AFNI motion-affine file contains non-finite values."
    return True, f"AFNI motion-affine file contains {expected_rows} valid rows."


def write_afni_warp(
    *,
    world_warp_path: Path,
    source_reference_path: Path,
    target_reference_path: Path,
    output_path: Path,
) -> None:
    """Convert one world-coordinate warp into AFNI's displacement convention."""
    output = world_warp_to_afni(
        nib.load(str(world_warp_path)),
        nib.load(str(source_reference_path)),
        nib.load(str(target_reference_path)),
    )
    with atomic_output_path(output_path) as staged:
        nib.save(output, str(staged))
        checked = nib.load(str(staged))
        if checked.shape != output.shape:
            raise RuntimeError(f"AFNI warp was written with the wrong shape: {staged}")


def validate_resampled_bold(
    *,
    source_path: Path,
    reference_path: Path,
    output_path: Path,
) -> tuple[bool, str]:
    """Check the volume count and exact requested geometry of a resampled BOLD."""
    try:
        source = nib.load(str(source_path))
        reference = nib.load(str(reference_path))
        output = nib.load(str(output_path))
    except (OSError, ValueError) as error:
        return False, f"Resampled BOLD is unreadable: {error}"
    if len(source.shape) != 4:
        return False, f"Resampling source is not 4D: {source_path}"
    expected_shape = (*reference.shape[:3], int(source.shape[3]))
    if output.shape != expected_shape:
        return False, f"Resampled BOLD shape is {output.shape}; expected {expected_shape}"
    if not np.array_equal(output.affine, reference.affine):
        return False, "Resampled BOLD affine does not exactly match its reference."
    return True, "Resampled BOLD has the requested geometry and volume count."
