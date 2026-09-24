from __future__ import annotations

import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nro.modules.anat.fastsurfer import (
    FASTSURFER_VOXEL_SIZE_MM,
    _reconcile_mask,
    create_fastsurfer_plan,
    create_fastsurfer_steps,
)
from nro.modules.anat.policy import surface_reconstruction_contract


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
    assert plan.segmentation.command[plan.segmentation.command.index("--vox_size") + 1] == str(
        FASTSURFER_VOXEL_SIZE_MM
    )
    assert "--no_cc" not in plan.segmentation.command
    assert "--no_cc" not in plan.surfaces.command
    assert plan.subject_directory / "mri" / "aseg.auto.mgz" in plan.segmentation.outputs


def test_fastsurfer_contract_excludes_execution_resource_routing() -> None:
    contract = surface_reconstruction_contract("fastsurfer")

    assert contract["backend"] == "FastSurfer"
    assert not any("device" in key for key in contract)


def test_fastsurfer_mask_includes_disconnected_segmented_tissue(tmp_path) -> None:
    subject = tmp_path / "subjects/sub-test"
    mri = subject / "mri"
    mri.mkdir(parents=True)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    segmentation = np.zeros_like(mask)
    segmentation[1, 1, 1] = 2
    segmentation[2, 2, 2] = 41
    nib.save(nib.MGHImage(mask, np.eye(4)), str(mri / "mask.mgz"))
    nib.save(
        nib.MGHImage(segmentation, np.eye(4)),
        str(mri / "aparc.DKTatlas+aseg.deep.mgz"),
    )

    _reconcile_mask(subject)

    reconciled = np.asarray(nib.load(str(mri / "mask.mgz")).dataobj)
    metadata = json.loads(
        (subject / "stats/nro-mask-reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciled[2, 2, 2] == 1
    assert metadata["AddedVoxelCount"] == 1
    assert metadata["FinalMaskVoxelCount"] == 2


def test_fastsurfer_runner_steps_stage_gpu_outputs_before_cpu_publication(tmp_path) -> None:
    steps = create_fastsurfer_steps(
        run_child=lambda *args, **kwargs: None,
        runtime="singularity",
        image=tmp_path / "fastsurfer.sif",
        t1w=tmp_path / "input" / "t1.nii.gz",
        staging_subjects_dir=tmp_path / "work" / "segmentation",
        subjects_dir=tmp_path / "subjects",
        subject="sub-t20",
        license_file=tmp_path / "license.txt",
        segmentation_threads=2,
        surface_threads=8,
        force=False,
    )

    segmentation, surfaces = steps
    assert segmentation.resource_class == "gpu"
    assert surfaces.resource_class is None
    assert segmentation.directory is None
    assert surfaces.directory == tmp_path / "subjects/sub-t20"
    assert any(path.suffix == ".tar" for path in surfaces.inputs)


def test_fastsurfer_runner_rejects_incomplete_segmentation_archive(tmp_path) -> None:
    segmentation, _ = create_fastsurfer_steps(
        run_child=lambda *args, **kwargs: None,
        runtime="singularity",
        image=tmp_path / "fastsurfer.sif",
        t1w=tmp_path / "input/t1.nii.gz",
        staging_subjects_dir=tmp_path / "work/segmentation",
        subjects_dir=tmp_path / "subjects",
        subject="sub-t20",
        license_file=tmp_path / "license.txt",
        segmentation_threads=2,
        surface_threads=8,
        force=False,
    )
    archive = segmentation.outputs[0]
    source = tmp_path / "partial/sub-t20/scripts/deep-seg.log"
    source.parent.mkdir(parents=True)
    source.write_text("complete", encoding="utf-8")
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source, arcname="sub-t20/scripts/deep-seg.log")

    assert segmentation.validate is not None
    valid, reason = segmentation.validate()

    assert not valid
    assert "incomplete" in reason


def test_fastsurfer_runner_rejects_archive_with_unexpected_root(tmp_path) -> None:
    segmentation, _ = create_fastsurfer_steps(
        run_child=lambda *args, **kwargs: None,
        runtime="singularity",
        image=tmp_path / "fastsurfer.sif",
        t1w=tmp_path / "input/t1.nii.gz",
        staging_subjects_dir=tmp_path / "work/segmentation",
        subjects_dir=tmp_path / "subjects",
        subject="sub-t20",
        license_file=tmp_path / "license.txt",
        segmentation_threads=2,
        surface_threads=8,
        force=False,
    )
    archive = segmentation.outputs[0]
    source = tmp_path / "foreign.txt"
    source.write_text("complete", encoding="utf-8")
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source, arcname="other-subject/foreign.txt")

    assert segmentation.validate is not None
    valid, reason = segmentation.validate()

    assert not valid
    assert "unexpected root" in reason


def test_fastsurfer_runner_steps_execute_across_resource_boundary(tmp_path, monkeypatch) -> None:
    calls = []
    targets = []

    def run_child(command, **kwargs):
        calls.append((tuple(command), kwargs))
        for path in targets[len(calls) - 1]:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name in {"mask.mgz", "aparc.DKTatlas+aseg.deep.mgz"}:
                nib.save(
                    nib.MGHImage(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)),
                    str(path),
                )
            else:
                path.write_text("complete", encoding="utf-8")

    steps = create_fastsurfer_steps(
        run_child=run_child,
        runtime="singularity",
        image=tmp_path / "fastsurfer.sif",
        t1w=tmp_path / "input" / "t1.nii.gz",
        staging_subjects_dir=tmp_path / "work" / "segmentation",
        subjects_dir=tmp_path / "subjects",
        subject="sub-t20",
        license_file=tmp_path / "license.txt",
        segmentation_threads=2,
        surface_threads=8,
        force=False,
    )
    segmentation, surfaces = steps
    segmentation_subject = tmp_path / "work/segmentation/sub-t20"
    plan = _plan(tmp_path)
    targets.extend(
        [
            tuple(
                segmentation_subject / path.relative_to(plan.subject_directory)
                for path in plan.segmentation.outputs
            ),
            tuple(path for path in surfaces.outputs if path != surfaces.breadcrumb),
        ]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert segmentation.action is not None
    segmentation.action()
    assert segmentation.validate is not None and segmentation.validate()[0]
    assert surfaces.action is not None
    surfaces.action()
    assert surfaces.validate is not None and surfaces.validate()[0]

    assert "--nv" in calls[0][0]
    assert ("--env", "CUDA_VISIBLE_DEVICES=0") == calls[0][0][
        calls[0][0].index("--env") : calls[0][0].index("--env") + 2
    ]
    assert "--nv" not in calls[1][0]
    assert not segmentation_subject.exists()


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
