"""Rigid-pose geometry tests that do not require ANTs execution."""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat

from nro.engine.pose import create_acpc_grid, invert_itk_affine, read_itk_affine


def _write_itk(path: Path, matrix: np.ndarray, translation: np.ndarray) -> None:
    parameters = np.concatenate((matrix.reshape(-1), translation)).reshape(12, 1)
    savemat(
        path,
        {
            "AffineTransform_double_3_3": parameters,
            "fixed": np.zeros((3, 1), dtype=np.float64),
        },
        format="4",
    )


def test_itk_rigid_inverse_composes_to_identity(tmp_path: Path) -> None:
    angle = np.deg2rad(23.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source = tmp_path / "forward.mat"
    inverse = tmp_path / "inverse.mat"
    _write_itk(source, rotation, np.array([4.0, -7.0, 2.0]))

    invert_itk_affine(source, inverse)

    forward_affine, _, _ = read_itk_affine(source)
    inverse_affine, _, _ = read_itk_affine(inverse)
    np.testing.assert_allclose(inverse_affine @ forward_affine, np.eye(4), atol=1e-10)
    assert loadmat(inverse)["AffineTransform_double_3_3"].size == 12


def test_acpc_grid_keeps_source_resolution_and_covers_rotated_field(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    source_mask = tmp_path / "source_mask.nii.gz"
    template = tmp_path / "template.nii.gz"
    transform = tmp_path / "rigid.mat"
    destination = tmp_path / "grid.nii.gz"
    source_affine = np.diag([0.8, 1.2, 1.5, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((20, 16, 12), dtype=np.uint8), source_affine), source)
    mask = np.zeros((20, 16, 12), dtype=np.uint8)
    mask[3:17, 2:14, 1:11] = 1
    nib.save(nib.Nifti1Image(mask, source_affine), source_mask)
    template_affine = np.array(
        [[-1.0, 0.0, 0.0, 10.0], [0.0, 0.0, 1.0, -20.0], [0.0, -1.0, 0.0, 30.0], [0, 0, 0, 1]]
    )
    nib.save(nib.Nifti1Image(np.zeros((30, 30, 30), dtype=np.uint8), template_affine), template)
    _write_itk(transform, np.eye(3), np.array([3.0, 4.0, -2.0]))

    create_acpc_grid(source, source_mask, template, transform, destination, margin_mm=5.0)

    grid = nib.load(destination)
    np.testing.assert_allclose(grid.header.get_zooms()[:3], (0.8, 1.2, 1.5), atol=1e-6)
    directions = grid.affine[:3, :3] / np.linalg.norm(grid.affine[:3, :3], axis=0)
    template_directions = template_affine[:3, :3]
    np.testing.assert_allclose(directions, template_directions, atol=1e-6)


def test_acpc_grid_excludes_empty_rotated_field_of_view(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    source_mask = tmp_path / "source_mask.nii.gz"
    template = tmp_path / "template.nii.gz"
    transform = tmp_path / "rigid.mat"
    destination = tmp_path / "grid.nii.gz"
    affine = np.diag([0.7, 0.7, 0.7, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((256, 320, 320), dtype=np.uint8), affine), source)
    mask = np.zeros((256, 320, 320), dtype=np.uint8)
    mask[30:225, 25:281, 53:255] = 1
    nib.save(nib.Nifti1Image(mask, affine), source_mask)
    nib.save(nib.Nifti1Image(np.zeros((200, 200, 200), dtype=np.uint8), np.eye(4)), template)
    angle = np.deg2rad(8.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    _write_itk(transform, rotation, np.zeros(3))

    create_acpc_grid(source, source_mask, template, transform, destination, margin_mm=5.0)

    grid = nib.load(destination)
    physical_extent = np.asarray(grid.shape) * np.asarray(grid.header.get_zooms()[:3])
    assert np.all(physical_extent < 256.0)


def test_acpc_grid_inverts_ants_fixed_to_moving_point_map(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    source_mask = tmp_path / "source_mask.nii.gz"
    template = tmp_path / "template.nii.gz"
    transform = tmp_path / "rigid.mat"
    destination = tmp_path / "grid.nii.gz"
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), affine), source)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 1
    nib.save(nib.Nifti1Image(mask, affine), source_mask)
    nib.save(nib.Nifti1Image(np.zeros((30, 30, 30), dtype=np.uint8), affine), template)
    # In LPS, this affine maps fixed points 10 mm left into the moving image.
    # The moving anatomy consequently lies 10 mm right in fixed RAS space.
    _write_itk(transform, np.eye(3), np.array([-10.0, 0.0, 0.0]))

    create_acpc_grid(source, source_mask, template, transform, destination, margin_mm=0.0)

    grid = nib.load(destination)
    np.testing.assert_allclose(grid.affine[:3, 3], (-8.5, 1.5, 1.5), atol=1e-6)
    assert grid.shape == (7, 7, 7)
