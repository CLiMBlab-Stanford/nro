"""Verify the sealed surface-reconstruction stage and backend adapters."""

from pathlib import Path

from nro.modules.anat.fastsurfer import create_fastsurfer_steps
from nro.modules.anat.reconstruction import (
    SurfaceReconstructionInputs,
    SurfaceReconstructionResources,
    create_surface_reconstruction_plan,
)
from nro.modules.anat.steps import _create_recon_all_step


def _resources(tmp_path: Path) -> SurfaceReconstructionResources:
    return SurfaceReconstructionResources(
        runtime="singularity",
        freesurfer_image=tmp_path / "freesurfer.sif",
        fastsurfer_image=tmp_path / "fastsurfer.sif",
        license_file=tmp_path / "license.txt",
        subjects_dir=tmp_path / "subjects",
        staging_subjects_dir=tmp_path / "staging",
        subject="sub-1",
        cpu_threads=2,
    )


def test_freesurfer_adapter_preserves_existing_step_declaration(tmp_path: Path) -> None:
    """Introducing the stage interface must not dirty ordinary FreeSurfer work."""
    inputs = SurfaceReconstructionInputs(
        t1w=tmp_path / "T1w.nii.gz",
        t2w=tmp_path / "T2w.nii.gz",
        brain_mask=tmp_path / "brain_mask.nii.gz",
    )
    resources = _resources(tmp_path)

    def run_child(*_args, **_kwargs):
        return None

    env = {"FS_LICENSE": str(resources.license_file)}

    plan = create_surface_reconstruction_plan(
        engine="freesurfer",
        inputs=inputs,
        resources=resources,
        run_child=run_child,
        env=env,
        force=False,
    )
    previous = _create_recon_all_step(
        run_child=run_child,
        env=env,
        t1w=inputs.t1w,
        t2w=inputs.t2w,
        brain_mask=inputs.brain_mask,
        subjects_dir=resources.subjects_dir,
        fs_subject=resources.subject,
        runtime=resources.runtime,
        image=resources.freesurfer_image,
        license_file=resources.license_file,
        force=False,
    )

    assert plan.steps == (previous,)
    assert plan.products.subject_dir == resources.subjects_dir / resources.subject


def test_fastsurfer_adapter_preserves_steps_and_ignores_unused_roles(tmp_path: Path) -> None:
    """FastSurfer freshness must not depend on FreeSurfer-only auxiliary inputs."""
    inputs = SurfaceReconstructionInputs(
        t1w=tmp_path / "T1w.nii.gz",
        t2w=tmp_path / "T2w.nii.gz",
        brain_mask=tmp_path / "brain_mask.nii.gz",
    )
    resources = _resources(tmp_path)

    def run_child(*_args, **_kwargs):
        return None

    plan = create_surface_reconstruction_plan(
        engine="fastsurfer",
        inputs=inputs,
        resources=resources,
        run_child=run_child,
        env={"FS_LICENSE": str(resources.license_file)},
        force=False,
    )
    previous = create_fastsurfer_steps(
        run_child=run_child,
        runtime=resources.runtime,
        image=resources.fastsurfer_image,
        t1w=inputs.t1w,
        staging_subjects_dir=resources.staging_subjects_dir,
        subjects_dir=resources.subjects_dir,
        subject=resources.subject,
        license_file=resources.license_file,
        segmentation_threads=resources.cpu_threads,
        surface_threads=resources.cpu_threads,
        force=False,
    )

    assert plan.steps == previous
    assert all(inputs.t2w not in step.inputs for step in plan.steps)
    assert all(inputs.brain_mask not in step.inputs for step in plan.steps)
