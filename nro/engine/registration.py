"""Registration transform measurements and quality limits."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np


LOG = logging.getLogger(__name__)


def fsl_scaled_mm_center(mask: Path) -> np.ndarray:
    """Return a mask centroid in the scaled-mm coordinates used by FLIRT."""
    import nibabel as nib

    image = nib.load(str(mask))
    indices = np.argwhere(np.asarray(image.dataobj) > 0)
    if not len(indices):
        raise ValueError(f"Cannot determine registration center from an empty mask: {mask}")
    center = indices.mean(axis=0).astype(np.float64)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    center *= zooms
    if float(np.linalg.det(np.asarray(image.affine)[:3, :3])) > 0.0:
        center[0] = (image.shape[0] - 1) * zooms[0] - center[0]
    return center


def rigid_transform_metrics(
    *,
    matrix: Path,
    initial_matrix: Path,
    center_mask: Path | None = None,
) -> dict[str, float]:
    """Measure rotation and center displacement between two FSL affines."""
    final = np.loadtxt(matrix, dtype=np.float64)
    initial = np.loadtxt(initial_matrix, dtype=np.float64)
    if final.shape != (4, 4) or initial.shape != (4, 4):
        raise ValueError(f"Invalid FSL rigid-transform matrix: {matrix}")
    delta = final @ np.linalg.inv(initial)
    left, _, right = np.linalg.svd(delta[:3, :3])
    rotation = left @ right
    rotation_degrees = math.degrees(
        math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    )
    center = (
        np.zeros(3, dtype=np.float64)
        if center_mask is None
        else fsl_scaled_mm_center(center_mask)
    )
    displacement = rotation @ center + delta[:3, 3] - center
    return {
        "RotationDegrees": float(rotation_degrees),
        "CenterDisplacementMillimeters": float(np.linalg.norm(displacement)),
    }


def validate_rigid_transform(
    *,
    matrix: Path,
    initial_matrix: Path,
    max_translation_mm: float,
    max_rotation_degrees: float,
    center_mask: Path | None = None,
    label: str = "Rigid registration",
) -> dict[str, float]:
    """Enforce rotation and center-displacement limits for a rigid affine."""
    metrics = rigid_transform_metrics(
        matrix=matrix,
        initial_matrix=initial_matrix,
        center_mask=center_mask,
    )
    rotation_degrees = metrics["RotationDegrees"]
    translation_mm = metrics["CenterDisplacementMillimeters"]
    if rotation_degrees > float(max_rotation_degrees):
        raise SystemExit(
            f"{label} exceeds rotation limit: {rotation_degrees:.2f} > "
            f"{max_rotation_degrees:.2f} degrees."
        )
    if translation_mm > float(max_translation_mm):
        raise SystemExit(
            f"{label} exceeds center-displacement limit: {translation_mm:.2f} > "
            f"{max_translation_mm:.2f} mm."
        )
    LOG.info(
        "%s QC relative to header initialization: %.2f degrees, %.2f mm",
        label,
        rotation_degrees,
        translation_mm,
    )
    return metrics
