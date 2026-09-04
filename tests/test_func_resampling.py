from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.func.resampling import (
    flirt_to_afni_pull,
    plumb_affine,
    validate_afni_motion_affines,
    validate_resampled_bold,
    voxel_to_fsl,
    world_warp_to_afni,
    write_afni_motion_affines,
)


def _oblique_affine() -> np.ndarray:
    angle = math.radians(21.0)
    return np.array(
        [
            [-2.0, 0.0, 0.0, 30.0],
            [0.0, 2.0 * math.cos(angle), -2.0 * math.sin(angle), -28.0],
            [0.0, 2.0 * math.sin(angle), 2.0 * math.cos(angle), -26.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _world_to_flirt(
    world_source_to_reference: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    source_fsl_to_world = source.affine @ np.linalg.inv(voxel_to_fsl(source))
    reference_world_to_fsl = voxel_to_fsl(reference) @ np.linalg.inv(reference.affine)
    return reference_world_to_fsl @ world_source_to_reference @ source_fsl_to_world


def test_motion_affines_preserve_oblique_world_transform(tmp_path: Path) -> None:
    affine = _oblique_affine()
    source = nib.Nifti1Image(np.zeros((7, 8, 9, 2), dtype=np.float32), affine)
    reference = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.float32), affine)
    source_path = tmp_path / "source.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    matrix_dir = tmp_path / "matrices"
    output_path = tmp_path / "motion_pull.aff12.1D"
    matrix_dir.mkdir()
    nib.save(source, source_path)
    nib.save(reference, reference_path)

    identity = np.eye(4)
    translation = np.eye(4)
    translation[:3, 3] = (1.5, -2.0, 0.75)
    for index, world in enumerate((identity, translation)):
        np.savetxt(
            matrix_dir / f"MAT_{index:04d}",
            _world_to_flirt(world, source, reference),
        )

    write_afni_motion_affines(
        source_path=source_path,
        motion_reference_path=reference_path,
        matrix_dir=matrix_dir,
        output_path=output_path,
    )

    rows = np.atleast_2d(np.loadtxt(output_path))
    expected = np.stack(
        [
            flirt_to_afni_pull(
                _world_to_flirt(world, source, reference),
                source,
                reference,
            )[:3, :].reshape(-1)
            for world in (identity, translation)
        ]
    )
    np.testing.assert_allclose(rows, expected, atol=1e-10)
    np.testing.assert_allclose(rows[0], np.eye(4)[:3, :].reshape(-1), atol=1e-10)
    assert validate_afni_motion_affines(
        source_path=source_path,
        affine_path=output_path,
    )[0]


def test_zero_world_warp_remains_zero_for_an_oblique_grid() -> None:
    affine = _oblique_affine()
    shape = (7, 8, 9)
    reference = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    world_warp = nib.Nifti1Image(np.zeros((*shape, 3), dtype=np.float32), affine)

    converted = world_warp_to_afni(
        world_warp,
        reference,
        reference,
        slab_depth=2,
    )

    np.testing.assert_allclose(np.asarray(converted.dataobj), 0.0, atol=1e-5)
    np.testing.assert_allclose(converted.affine, plumb_affine(reference))


def test_resampled_bold_validation_requires_reference_geometry(tmp_path: Path) -> None:
    source = nib.Nifti1Image(np.zeros((4, 5, 6, 3), dtype=np.float32), np.eye(4))
    reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.float32), reference_affine)
    output = nib.Nifti1Image(
        np.zeros((7, 8, 9, 3), dtype=np.float32),
        reference_affine,
    )
    source_path = tmp_path / "source.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    output_path = tmp_path / "output.nii.gz"
    nib.save(source, source_path)
    nib.save(reference, reference_path)
    nib.save(output, output_path)

    assert validate_resampled_bold(
        source_path=source_path,
        reference_path=reference_path,
        output_path=output_path,
    )[0]

    shifted = output.affine.copy()
    shifted[0, 3] = 1.0
    nib.save(nib.Nifti1Image(np.asarray(output.dataobj), shifted), output_path)
    valid, reason = validate_resampled_bold(
        source_path=source_path,
        reference_path=reference_path,
        output_path=output_path,
    )
    assert not valid
    assert "affine" in reason
