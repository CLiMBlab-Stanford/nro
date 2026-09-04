from pathlib import Path
from itertools import count
import logging

import nibabel as nib
import numpy as np
import pytest

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.func.module import (
    _ants_pe_aligned_frame,
    _create_nifti_in_ants_frame_step,
    _create_restore_ants_warp_step,
)
from nro.orchestration.runner import Runner


def _runner() -> Runner:
    return Runner(
        module_name="PE-aligned frame test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.func.pe-frame"),
        next_step=count(1).__next__,
    )


def _oblique_reference(path: Path, *, angle_degrees: float = 15.0) -> Path:
    angle = np.deg2rad(angle_degrees)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation_x @ np.diag([2.0, 2.0, 2.0])
    affine[:3, 3] = [91.0, -87.0, -73.0]
    data = np.arange(5 * 6 * 7, dtype=np.float32).reshape((5, 6, 7))
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_pe_aligned_frame_rotates_oblique_voxel_axis_without_resampling(tmp_path: Path) -> None:
    reference = _oblique_reference(tmp_path / "reference.nii.gz")
    frame = _ants_pe_aligned_frame(reference, "j-")

    assert frame.restriction == "0x1x0"
    assert frame.voxel_axis == 1
    source = nib.load(reference)
    rotation = np.asarray(frame.rotation_ras)
    original_direction = source.affine[:3, 1].copy()
    original_direction /= np.linalg.norm(original_direction)
    np.testing.assert_allclose(rotation @ original_direction, [0.0, 1.0, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    aligned_path = tmp_path / "aligned.nii.gz"
    runner = _runner()
    runner.add_step(
        _create_nifti_in_ants_frame_step(
            source=reference,
            out_image=aligned_path,
            frame=frame,
            force=False,
        )
    )
    with runner.run_context():
        runner.execute()
    aligned = nib.load(aligned_path)
    np.testing.assert_array_equal(np.asanyarray(aligned.dataobj), np.asanyarray(source.dataobj))
    np.testing.assert_allclose(
        aligned.affine,
        np.asarray(frame.world_transform_ras) @ source.affine,
        atol=1.0e-5,
    )
    aligned_direction = aligned.affine[:3, 1] / np.linalg.norm(aligned.affine[:3, 1])
    np.testing.assert_allclose(aligned_direction, [0.0, 1.0, 0.0], atol=1.0e-6)


def test_restore_ants_warp_rotates_lps_vectors_back_to_oblique_pe(tmp_path: Path) -> None:
    reference_path = _oblique_reference(tmp_path / "reference.nii.gz")
    frame = _ants_pe_aligned_frame(reference_path, "j")
    aligned_reference_path = tmp_path / "aligned_reference.nii.gz"
    runner = _runner()
    runner.add_step(
        _create_nifti_in_ants_frame_step(
            source=reference_path,
            out_image=aligned_reference_path,
            frame=frame,
            force=False,
        )
    )
    with runner.run_context():
        runner.execute()
    aligned_reference = nib.load(aligned_reference_path)

    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    target_ras = np.zeros(3)
    target_ras[frame.physical_axis] = np.sign(frame.pe_direction_ras[frame.physical_axis])
    target_lps = ras_to_lps @ target_ras
    amplitudes = np.linspace(-3.0, 3.0, num=np.prod(aligned_reference.shape), dtype=np.float32).reshape(
        aligned_reference.shape
    )
    aligned_vectors = np.zeros((*aligned_reference.shape, 1, 3), dtype=np.float32)
    aligned_vectors[..., 0, :] = amplitudes[..., None] * target_lps
    warp_header = nib.Nifti1Header()
    warp_header.set_data_dtype(np.float32)
    warp_header.set_intent("vector")
    aligned_warp_path = tmp_path / "aligned_warp.nii.gz"
    nib.save(nib.Nifti1Image(aligned_vectors, aligned_reference.affine, warp_header), aligned_warp_path)

    restored_path = tmp_path / "restored_warp.nii.gz"
    runner = _runner()
    runner.add_step(
        _create_restore_ants_warp_step(
            aligned_warp=aligned_warp_path,
            original_reference=reference_path,
            out_warp=restored_path,
            frame=frame,
            force=False,
        )
    )
    with runner.run_context():
        runner.execute()
    restored = nib.load(restored_path)
    reference = nib.load(reference_path)
    np.testing.assert_allclose(restored.affine, reference.affine, atol=1.0e-5)
    assert restored.header.get_intent()[0] == "vector"

    restored_vectors_lps = np.asanyarray(restored.dataobj)[..., 0, :]
    expected_direction_lps = ras_to_lps @ np.asarray(frame.pe_direction_ras)
    expected = amplitudes[..., None] * expected_direction_lps
    np.testing.assert_allclose(restored_vectors_lps, expected, atol=1.0e-6)

    # Every nonzero restored vector lies on the original physical PE line.
    restored_vectors_ras = np.einsum("ij,...j->...i", ras_to_lps, restored_vectors_lps)
    cross = np.cross(restored_vectors_ras, np.asarray(frame.pe_direction_ras))
    np.testing.assert_allclose(cross, 0.0, atol=1.0e-6)


def test_pe_aligned_frame_rejects_unknown_bids_axis(tmp_path: Path) -> None:
    reference = _oblique_reference(tmp_path / "reference.nii.gz")
    with pytest.raises(SystemExit, match="Unsupported PhaseEncodingDirection"):
        _ants_pe_aligned_frame(reference, "x")
