import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from nro.modules.func.marss import (
    _apply_correction,
    _compact_artifact,
    _slice_correlation_summary,
    create_marss_step,
    derive_slice_grouping,
    slice_correlation_diagnostics,
)
from nro.modules.func.marss_worker import _publish_nifti


def metadata(*, slices=12, factor=3):
    times = [float(index) for index in range(slices // factor)] * factor
    return {
        "MultibandAccelerationFactor": factor,
        "SliceTiming": times,
    }


def save_bold(path: Path, data: np.ndarray, *, repetition_time=1.2) -> None:
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))
    image.header.set_zooms((2.0, 2.0, 2.0, repetition_time))
    nib.save(image, path)


def test_slice_groups_come_from_timing_and_validate_official_layout():
    grouping = derive_slice_grouping(metadata(), (5, 6, 12))
    assert grouping.groups == (
        (0, 4, 8),
        (1, 5, 9),
        (2, 6, 10),
        (3, 7, 11),
    )

    irregular = metadata()
    irregular["SliceTiming"] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    with pytest.raises(ValueError, match="cannot represent"):
        derive_slice_grouping(irregular, (5, 6, 12))
    with pytest.raises(ValueError, match="axis k"):
        derive_slice_grouping(
            {**metadata(), "SliceEncodingDirection": "j-"},
            (5, 12, 6),
        )


def test_slice_groups_support_diagnosing_multiband_two():
    grouping = derive_slice_grouping(metadata(slices=6, factor=2), (5, 6, 6))

    assert grouping.multiband_factor == 2
    assert grouping.groups == ((0, 3), (1, 4), (2, 5))


def test_slice_diagnostic_weights_each_target_slice_equally():
    grouping = derive_slice_grouping(metadata(slices=6), (4, 5, 6))
    fisher = np.zeros((6, 6), dtype=np.float64)
    for group in grouping.groups:
        for left in group:
            for right in group:
                if left != right:
                    fisher[left, right] = 1.0
    fisher[1, 2] = fisher[2, 1] = 4.0

    within, adjacent, _ = _slice_correlation_summary(fisher, grouping)

    assert within == pytest.approx(1.0)
    assert adjacent == pytest.approx(5.0 / 9.0)


def test_correction_uses_multiband_factor_and_not_diagnostic_score():
    assert not _apply_correction("auto", 5, 6)
    assert _apply_correction("auto", 6, 6)
    assert _apply_correction("auto", 8, 6)
    assert not _apply_correction("diagnose", 8, 6)
    assert _apply_correction("auto", 4, 4)


def test_slice_diagnostic_detects_shared_simultaneous_signal(tmp_path):
    rng = np.random.default_rng(9)
    volumes = 160
    data = rng.normal(scale=0.2, size=(4, 5, 6, volumes))
    for group in ((0, 2, 4), (1, 3, 5)):
        shared = rng.normal(size=volumes)
        data[:, :, group, :] += shared[None, None, None, :]
    bold = tmp_path / "bold.nii.gz"
    motion = tmp_path / "motion.par"
    save_bold(bold, data)
    np.savetxt(motion, np.zeros((volumes, 6)))

    result = slice_correlation_diagnostics(
        bold,
        motion,
        derive_slice_grouping(metadata(slices=6), (4, 5, 6)),
        chunk_volumes=23,
    )
    assert result["simultaneous_minus_adjacent_delta_r"] > 0.8


def test_compact_artifact_reconstructs_official_rank_one_output(tmp_path):
    rng = np.random.default_rng(4)
    shape = (3, 4, 6, 20)
    raw = rng.normal(size=shape).astype(np.float32)
    artifact = np.empty(shape, dtype=np.float32)
    for slice_index in range(shape[2]):
        loading = rng.normal(size=shape[:2])
        timecourse = rng.normal(size=shape[3])
        artifact[:, :, slice_index, :] = loading[:, :, None] * timecourse[None, None, :]
    corrected = raw - artifact
    source = tmp_path / "source.nii.gz"
    corrected_path = tmp_path / "corrected.nii.gz"
    artifact_path = tmp_path / "artifact.nii.gz"
    output = tmp_path / "output.nii.gz"
    loadings = tmp_path / "loadings.nii.gz"
    timecourses = tmp_path / "timecourses.tsv"
    mean_absolute = tmp_path / "meanabs.nii.gz"
    save_bold(source, raw, repetition_time=0.8)
    save_bold(corrected_path, corrected, repetition_time=1.0)
    save_bold(artifact_path, artifact, repetition_time=1.0)

    error, correction_error, mean_artifact_variance = _compact_artifact(
        source,
        corrected_path,
        artifact_path,
        output,
        loadings,
        timecourses,
        mean_absolute,
    )
    saved_loadings = np.asarray(nib.load(loadings).dataobj)
    saved_timecourses = np.loadtxt(timecourses, delimiter="\t", skiprows=1)
    reconstructed = saved_loadings[:, :, :, None] * saved_timecourses.T[None, None, :, :]
    assert error < 1e-5
    assert correction_error < 1e-5
    assert mean_artifact_variance > 0
    np.testing.assert_allclose(reconstructed, artifact, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(nib.load(output).dataobj), corrected, rtol=1e-6)
    assert nib.load(output).header.get_zooms()[3] == pytest.approx(0.8)


def test_diagnose_mode_aliases_input_and_keeps_fixed_outputs(tmp_path):
    rng = np.random.default_rng(2)
    bold = tmp_path / "source.nii.gz"
    motion = tmp_path / "motion.par"
    sidecar = tmp_path / "source.json"
    save_bold(bold, rng.normal(size=(3, 3, 6, 30)))
    np.savetxt(motion, np.zeros((30, 6)))
    sidecar.write_text(json.dumps(metadata(slices=6)))
    runner = SimpleNamespace(run_direct=lambda *args, **kwargs: pytest.fail("MARSS was invoked"))

    step, outputs = create_marss_step(
        runner=runner,
        source_bold=bold,
        metadata=metadata(slices=6),
        metadata_sources=(sidecar,),
        motion_parameters=motion,
        work_dir=tmp_path / "work",
        artifact_dir=tmp_path / "public",
        run_stem="sub-01_task-rest",
        mode="diagnose",
        min_multiband_factor=6,
        chunk_volumes=8,
        force=False,
    )
    assert step.action is not None
    step.action()

    assert outputs.bold.is_symlink()
    assert outputs.bold.resolve() == bold.resolve()
    result = json.loads(outputs.metadata.read_text())
    assert result["Applied"] is False
    assert result["CompactArtifact"]["Available"] is False
    assert all(path.exists() for path in step.outputs)
    assert step.validate is not None and step.validate()[0]


def test_auto_mode_diagnoses_but_does_not_correct_multiband_two(tmp_path):
    rng = np.random.default_rng(7)
    volumes = 40
    data = rng.normal(scale=0.05, size=(3, 3, 6, volumes))
    for group in ((0, 3), (1, 4), (2, 5)):
        shared = rng.normal(size=volumes)
        data[:, :, group, :] += shared[None, None, None, :]
    bold = tmp_path / "source.nii.gz"
    motion = tmp_path / "motion.par"
    save_bold(bold, data)
    np.savetxt(motion, np.zeros((volumes, 6)))
    runner = SimpleNamespace(run_direct=lambda *args, **kwargs: pytest.fail("MARSS was invoked"))

    step, outputs = create_marss_step(
        runner=runner,
        source_bold=bold,
        metadata=metadata(slices=6, factor=2),
        metadata_sources=(),
        motion_parameters=motion,
        work_dir=tmp_path / "work",
        artifact_dir=tmp_path / "public",
        run_stem="sub-01_task-rest",
        mode="auto",
        min_multiband_factor=6,
        chunk_volumes=8,
        force=False,
    )
    assert step.action is not None
    step.action()

    result = json.loads(outputs.metadata.read_text())
    assert result["DiagnosticAvailable"] is True
    assert result["Applied"] is False
    assert result["Decision"] == "multiband_factor_below_recommended_minimum"


def test_auto_mode_passes_through_when_diagnostic_metadata_are_unavailable(tmp_path):
    rng = np.random.default_rng(8)
    bold = tmp_path / "source.nii.gz"
    motion = tmp_path / "motion.par"
    save_bold(bold, rng.normal(size=(3, 3, 5, 20)))
    np.savetxt(motion, np.zeros((20, 6)))
    runner = SimpleNamespace(run_direct=lambda *args, **kwargs: pytest.fail("MARSS was invoked"))

    step, outputs = create_marss_step(
        runner=runner,
        source_bold=bold,
        metadata={},
        metadata_sources=(),
        motion_parameters=motion,
        work_dir=tmp_path / "work",
        artifact_dir=tmp_path / "public",
        run_stem="sub-01_task-rest",
        mode="auto",
        min_multiband_factor=6,
        chunk_volumes=8,
        force=False,
    )
    assert step.action is not None
    step.action()

    result = json.loads(outputs.metadata.read_text())
    assert result["DiagnosticAvailable"] is False
    assert result["Applied"] is False
    assert result["Decision"] == "diagnostic_unavailable"
    assert all(path.exists() for path in step.outputs)


def test_marss_worker_preserves_nifti_compression_declared_by_destination(tmp_path):
    source = tmp_path / "source.nii"
    destination = tmp_path / "destination.nii.gz"
    save_bold(source, np.ones((2, 2, 3, 4)))

    _publish_nifti(source, destination)

    assert not source.exists()
    assert nib.load(destination).shape == (2, 2, 3, 4)
