from __future__ import annotations

from pathlib import Path

import pytest

from nro.modules.anat.fastsurfer import create_fastsurfer_plan


def _plan(tmp_path: Path):
    return create_fastsurfer_plan(
        runtime="singularity",
        image=tmp_path / "fastsurfer.sif",
        t1w=tmp_path / "input" / "t1.nii.gz",
        subjects_directory=tmp_path / "subjects",
        subject="sub-t20_fastsurfer",
        license_file=tmp_path / "license.txt",
        segmentation_threads=2,
        surface_threads=8,
    )


def test_fastsurfer_plan_separates_gpu_segmentation_from_cpu_surfaces(tmp_path) -> None:
    plan = _plan(tmp_path)

    assert plan.segmentation.resource == "gpu"
    assert "--nv" in plan.segmentation.command
    assert "--seg_only" in plan.segmentation.command
    assert plan.segmentation.command[plan.segmentation.command.index("--device") + 1] == "cuda"
    assert plan.surfaces.resource == "cpu"
    assert "--nv" not in plan.surfaces.command
    assert "--surf_only" in plan.surfaces.command
    assert plan.surfaces.command[plan.surfaces.command.index("--device") + 1] == "cpu"


def test_fastsurfer_plan_uses_one_subject_directory_and_fixed_outputs(tmp_path) -> None:
    plan = _plan(tmp_path)

    assert plan.subject_directory == tmp_path / "subjects" / "sub-t20_fastsurfer"
    assert all(path.is_relative_to(plan.subject_directory) for path in plan.segmentation.outputs)
    assert all(path.is_relative_to(plan.subject_directory) for path in plan.surfaces.outputs)
    assert plan.subject_directory / "mri" / "aparc.DKTatlas+aseg.deep.mgz" in (
        plan.segmentation.outputs
    )
    assert plan.subject_directory / "surf" / "lh.white" in plan.surfaces.outputs


def test_fastsurfer_validation_rejects_partial_and_accepts_complete_outputs(tmp_path) -> None:
    plan = _plan(tmp_path)

    valid, reason = plan.validate()
    assert not valid
    assert "segmentation" in reason.lower()

    for path in (*plan.segmentation.outputs, *plan.surfaces.outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete", encoding="utf-8")

    assert plan.validate() == (True, "FastSurfer segmentation and surface outputs are complete.")


@pytest.mark.parametrize("subject", ["", ".", "..", "../escape", "nested/subject"])
def test_fastsurfer_plan_rejects_unsafe_subject_ids(tmp_path, subject) -> None:
    with pytest.raises(ValueError, match="path components"):
        create_fastsurfer_plan(
            runtime="singularity",
            image=tmp_path / "fastsurfer.sif",
            t1w=tmp_path / "input" / "t1.nii.gz",
            subjects_directory=tmp_path / "subjects",
            subject=subject,
            license_file=tmp_path / "license.txt",
            segmentation_threads=2,
            surface_threads=8,
        )


@pytest.mark.parametrize("segmentation_threads,surface_threads", [(0, 8), (2, 0), (-1, 8)])
def test_fastsurfer_plan_requires_positive_thread_counts(
    tmp_path, segmentation_threads, surface_threads
) -> None:
    with pytest.raises(ValueError, match="positive"):
        create_fastsurfer_plan(
            runtime="singularity",
            image=tmp_path / "fastsurfer.sif",
            t1w=tmp_path / "input" / "t1.nii.gz",
            subjects_directory=tmp_path / "subjects",
            subject="sub-t20",
            license_file=tmp_path / "license.txt",
            segmentation_threads=segmentation_threads,
            surface_threads=surface_threads,
        )
