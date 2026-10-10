"""Construct the subject-space sampling grid for an MSMAll inverse warp."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np


def create_inverse_reference(
    source: Path,
    output: Path,
    *,
    resolution_mm: float = 2.0,
    margin_mm: float = 32.0,
) -> None:
    """Write an isotropic grid that encloses the source image's full voxel extent."""
    image = nib.load(str(source))
    if len(image.shape) < 3:
        raise ValueError("MSMAll inverse-warp reference requires a three-dimensional image")
    shape = np.asarray(image.shape[:3], dtype=np.int64)
    if np.any(shape < 1):
        raise ValueError("MSMAll inverse-warp reference source has an empty spatial dimension")
    if not np.isfinite(resolution_mm) or resolution_mm <= 0:
        raise ValueError("MSMAll inverse-warp reference resolution must be positive")
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("MSMAll inverse-warp reference margin cannot be negative")

    linear = np.asarray(image.affine[:3, :3], dtype=np.float64)
    zooms = np.linalg.norm(linear, axis=0)
    if not np.isfinite(zooms).all() or np.any(zooms <= 0):
        raise ValueError("MSMAll AC-PC image has invalid voxel geometry")
    directions = linear / zooms
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError("MSMAll AC-PC image axes are not orthogonal")

    bounds = [(-0.5, float(size) - 0.5) for size in shape]
    corners = np.asarray(list(itertools.product(*bounds)), dtype=np.float64)
    world = nib.affines.apply_affine(image.affine, corners)
    coordinates = np.linalg.solve(directions, world.T).T
    lower = coordinates.min(axis=0) - margin_mm
    upper = coordinates.max(axis=0) + margin_mm
    output_shape = np.ceil((upper - lower) / resolution_mm).astype(np.int64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = directions * resolution_mm
    affine[:3, 3] = directions @ (lower + resolution_mm / 2.0)
    reference = nib.Nifti1Image(np.zeros(tuple(output_shape), dtype=np.uint8), affine)
    reference.set_qform(affine, code=1)
    reference.set_sform(affine, code=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(reference, str(output))


def main() -> None:
    """Build one inverse-warp reference from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-mm", type=float, default=2.0)
    parser.add_argument("--margin-mm", type=float, default=32.0)
    arguments = parser.parse_args()
    create_inverse_reference(
        arguments.source,
        arguments.output,
        resolution_mm=arguments.resolution_mm,
        margin_mm=arguments.margin_mm,
    )


if __name__ == "__main__":
    main()
