"""Rigid anatomical pose transforms and deterministic ACPC output grids."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

_LPS_RAS = np.diag((-1.0, -1.0, 1.0, 1.0))


def _moving_to_fixed_ras(transform: Path) -> np.ndarray:
    """Return the forward point map represented by an ANTs affine.

    ANTs stores the fixed-to-moving map used during image resampling. Mapping
    points from the moving source image into fixed space therefore requires its
    inverse.
    """
    fixed_to_moving_lps, _center, _document = read_itk_affine(transform)
    fixed_to_moving_ras = _LPS_RAS @ fixed_to_moving_lps @ _LPS_RAS
    return np.linalg.inv(fixed_to_moving_ras)


def read_itk_affine(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read an ITK affine stored in the MATLAB file emitted by ANTs."""
    from scipy.io import loadmat  # type: ignore

    document = {
        key: np.asarray(value)
        for key, value in loadmat(str(path)).items()
        if not key.startswith("__")
    }
    parameters = next((value for value in document.values() if value.size == 12), None)
    fixed = next((value for value in document.values() if value.size == 3), None)
    if parameters is None or fixed is None:
        raise ValueError(f"ANTs affine has an unsupported structure: {path}")
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    center = np.asarray(fixed, dtype=np.float64).reshape(-1)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = values[:9].reshape(3, 3)
    affine[:3, 3] = values[9:12] + center - affine[:3, :3] @ center
    if not np.isfinite(affine).all():
        raise ValueError(f"ANTs affine contains non-finite values: {path}")
    return affine, center, document


def invert_itk_affine(source: Path, destination: Path) -> None:
    """Write the exact inverse of an ANTs rigid affine."""
    from scipy.io import savemat  # type: ignore

    affine, _center, document = read_itk_affine(source)
    inverse = np.linalg.inv(affine)
    matrix_key = next(key for key, value in document.items() if value.size == 12)
    fixed_key = next(key for key, value in document.items() if value.size == 3)
    center = np.zeros(3, dtype=np.float64)
    parameters = np.concatenate((inverse[:3, :3].reshape(-1), inverse[:3, 3]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        str(destination),
        {
            matrix_key: parameters.reshape(document[matrix_key].shape),
            fixed_key: center.reshape(document[fixed_key].shape),
        },
        format="4",
    )


def _masked_corners(image, mask) -> np.ndarray:
    """Return voxel-edge corners around the nonzero mask extent."""
    if image.shape[:3] != mask.shape[:3] or not np.allclose(image.affine, mask.affine, atol=1e-4):
        raise ValueError("The anatomical image and pose mask must share a grid")
    values = np.asarray(mask.dataobj)
    if values.ndim != 3:
        raise ValueError("ACPC grid construction requires a three-dimensional pose mask")
    occupied = np.isfinite(values) & (values > 0)
    if not np.any(occupied):
        raise ValueError("The anatomical pose mask is empty")
    bounds = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        indices = np.flatnonzero(np.any(occupied, axis=other_axes))
        bounds.append((float(indices[0]) - 0.5, float(indices[-1]) + 0.5))
    return np.array(list(itertools.product(*bounds)), dtype=np.float64)


def create_acpc_grid(
    source: Path,
    source_mask: Path,
    template: Path,
    transform: Path,
    destination: Path,
    *,
    margin_mm: float = 5.0,
) -> None:
    """Create a template-oriented grid around the transformed anatomical mask."""
    import nibabel as nib  # type: ignore

    image = nib.load(str(source))
    mask = nib.load(str(source_mask))
    reference = nib.load(str(template))
    if len(image.shape) < 3 or len(reference.shape) < 3:
        raise ValueError("ACPC grid construction requires three-dimensional images")
    moving_to_fixed_ras = _moving_to_fixed_ras(transform)
    corners = _masked_corners(image, mask)
    source_world = nib.affines.apply_affine(image.affine, corners)
    transformed = nib.affines.apply_affine(moving_to_fixed_ras, source_world)
    directions = np.asarray(reference.affine[:3, :3], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=0)
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError("ACPC template axes are not orthogonal")
    coordinates = np.linalg.solve(directions, transformed.T).T
    lower = coordinates.min(axis=0) - float(margin_mm)
    upper = coordinates.max(axis=0) + float(margin_mm)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if not np.isfinite(zooms).all() or np.any(zooms <= 0):
        raise ValueError("Source anatomical voxel sizes are invalid")
    shape = np.ceil((upper - lower) / zooms).astype(int) + 1
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = directions * zooms
    affine[:3, 3] = directions @ lower
    grid = nib.Nifti1Image(np.zeros(tuple(shape), dtype=np.uint8), affine)
    grid.set_qform(affine, code=1)
    grid.set_sform(affine, code=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(grid, str(destination))


def acpc_quality(
    source: Path,
    source_mask: Path,
    aligned: Path,
    template: Path,
    transform: Path,
    grid: Path,
) -> dict[str, object]:
    """Validate a rigid pose transform and summarize alignment geometry."""
    import nibabel as nib  # type: ignore
    from nibabel.processing import resample_from_to  # type: ignore

    rigid_lps, _center, _document = read_itk_affine(transform)
    rotation = rigid_lps[:3, :3]
    determinant = float(np.linalg.det(rotation))
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if abs(determinant - 1.0) > 1e-3 or orthogonality_error > 1e-3:
        raise ValueError("ACPC transform contains scale, shear, or reflection")
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1, 1))))
    translation = float(np.linalg.norm(rigid_lps[:3, 3]))

    source_image = nib.load(str(source))
    grid_image = nib.load(str(grid))
    moving_to_fixed_ras = _moving_to_fixed_ras(transform)
    corners = _masked_corners(source_image, nib.load(str(source_mask)))
    transformed = nib.affines.apply_affine(
        moving_to_fixed_ras, nib.affines.apply_affine(source_image.affine, corners)
    )
    voxels = nib.affines.apply_affine(np.linalg.inv(grid_image.affine), transformed)
    covered = bool(
        np.all(voxels >= -1e-3)
        and np.all(voxels <= np.asarray(grid_image.shape[:3], dtype=float) - 1 + 1e-3)
    )
    if not covered:
        raise ValueError("ACPC grid does not cover the transformed anatomical mask")

    aligned_image = nib.load(str(aligned))
    reference = resample_from_to(nib.load(str(template)), aligned_image, order=1)
    observed = np.asarray(aligned_image.dataobj, dtype=np.float32)
    expected = np.asarray(reference.dataobj, dtype=np.float32)
    mask = np.isfinite(observed) & np.isfinite(expected) & (observed > 0) & (expected > 0)
    correlation = None
    if int(mask.sum()) >= 100 and np.std(observed[mask]) > 0 and np.std(expected[mask]) > 0:
        correlation = float(np.corrcoef(observed[mask], expected[mask])[0, 1])
    return {
        "Transform": str(transform),
        "Source": str(source),
        "AlignedImage": str(aligned),
        "Template": str(template),
        "Grid": str(grid),
        "Determinant": determinant,
        "OrthogonalityError": orthogonality_error,
        "RotationDegrees": angle,
        "TranslationMillimeters": translation,
        "GridCoversTransformedMask": covered,
        "IntensityCorrelation": correlation,
    }
