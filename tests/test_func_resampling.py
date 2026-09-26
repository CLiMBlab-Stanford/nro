from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nro.modules.func.resampling import (
    flirt_to_afni_pull,
    plumb_affine,
    validate_afni_motion_affines,
    validate_resampled_bold,
    voxel_to_fsl,
    world_warp_to_afni,
    write_afni_motion_affines,
)
from nro.modules.func.resampling_steps import (
    _afni_bold_warp_chain,
    _create_afni_bold_resampling_step,
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
    source.header.set_zooms((1.0, 1.0, 1.0, 1.2))
    reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.float32), reference_affine)
    output = nib.Nifti1Image(
        np.zeros((7, 8, 9, 3), dtype=np.float32),
        reference_affine,
    )
    output.header.set_zooms((2.0, 2.0, 2.0, 1.2))
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


def test_resampled_bold_validation_requires_repetition_time(tmp_path: Path) -> None:
    source = nib.Nifti1Image(np.zeros((4, 5, 6, 3), dtype=np.float32), np.eye(4))
    source.header.set_zooms((1.0, 1.0, 1.0, 1.2))
    reference = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.float32), np.eye(4))
    output = nib.Nifti1Image(np.zeros((7, 8, 9, 3), dtype=np.float32), np.eye(4))
    output.header.set_zooms((1.0, 1.0, 1.0, 0.0))
    source_path = tmp_path / "source.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    output_path = tmp_path / "output.nii.gz"
    nib.save(source, source_path)
    nib.save(reference, reference_path)
    nib.save(output, output_path)

    valid, reason = validate_resampled_bold(
        source_path=source_path,
        reference_path=reference_path,
        output_path=output_path,
        repetition_time=1.2,
    )

    assert not valid
    assert "repetition time" in reason


def test_afni_resampling_restores_repetition_time(tmp_path: Path) -> None:
    source_path = tmp_path / "source.nii.gz"
    reference_path = tmp_path / "reference.nii.gz"
    output_path = tmp_path / "output.nii"
    source = nib.Nifti1Image(np.zeros((4, 5, 6, 3), dtype=np.float32), np.eye(4))
    source.header.set_zooms((1.0, 1.0, 1.0, 1.2))
    reference = nib.Nifti1Image(np.zeros((7, 8, 9), dtype=np.float32), np.eye(4))
    nib.save(source, source_path)
    nib.save(reference, reference_path)
    commands: list[list[str]] = []

    def run_child(command, *, env):
        del env
        command = list(command)
        commands.append(command)
        if command[0] == "3dNwarpApply":
            staged = Path(command[command.index("-prefix") + 1])
            output = nib.Nifti1Image(
                np.zeros((*reference.shape, source.shape[3]), dtype=np.float32),
                reference.affine,
            )
            output.header.set_zooms((1.0, 1.0, 1.0, 0.0))
            nib.save(output, staged)
        elif command[0] == "3drefit":
            staged = Path(command[-1])
            output = nib.load(staged)
            header = output.header.copy()
            header.set_zooms((*header.get_zooms()[:3], float(command[2])))
            data = np.asarray(output.dataobj).copy()
            nib.save(nib.Nifti1Image(data, output.affine, header), staged)

    step = _create_afni_bold_resampling_step(
        run_child=run_child,
        in_4d=source_path,
        motion_ref_3d=reference_path,
        ref_3d=reference_path,
        afni_warp=tmp_path / "warp.nii.gz",
        motion_affines=tmp_path / "motion.aff12.1D",
        out_4d=output_path,
        repetition_time=1.2,
        env={},
        force=False,
    )

    step.action()

    assert [command[0] for command in commands] == ["3dNwarpApply", "fslcpgeom", "3drefit"]
    assert nib.load(output_path).header.get_zooms()[3] == pytest.approx(1.2)


def test_gradient_warp_follows_spatial_and_motion_pull_transforms() -> None:
    assert (
        _afni_bold_warp_chain(
            spatial_warp=Path("spatial.nii.gz"),
            motion_affines=Path("motion.aff12.1D"),
            gradient_warp=Path("gradient.nii.gz"),
        )
        == "spatial.nii.gz motion.aff12.1D gradient.nii.gz"
    )
