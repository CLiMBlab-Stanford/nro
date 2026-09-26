"""Verify the sealed surface-reconstruction stage and backend adapters."""

from pathlib import Path

from nro.modules.anat.fastsurfer import create_fastsurfer_steps
from nro.modules.anat.reconstruction import (
    SurfaceReconstructionInputs,
    SurfaceReconstructionResources,
    create_surface_reconstruction_plan,
)
from nro.modules.anat.steps import (
    _create_metric_conversion_step,
    _create_recon_all_step,
    _freesurfer_progress,
    _FreeSurferProgressMonitor,
)


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


def test_freesurfer_progress_classifies_phase_and_hemisphere() -> None:
    assert _freesurfer_progress("#@# Fix Topology lh Thu Sep 25 12:34:56 PDT 2026") == (
        3,
        "Fix Topology lh, left hemisphere",
    )
    assert _freesurfer_progress("#@# Surf Reg rh") == (
        4,
        "Surf Reg rh, right hemisphere",
    )
    assert _freesurfer_progress("#@# AParc-to-ASeg") == (6, "AParc-to-ASeg")
    assert _freesurfer_progress("#@# ASeg Stats") == (7, "ASeg Stats")


def test_freesurfer_progress_monitor_reads_native_status_log(tmp_path: Path, caplog) -> None:
    status = tmp_path / "scripts" / "recon-all-status.log"
    status.parent.mkdir(parents=True)
    status.write_text("#@# Tessellate rh\n")

    monitor = _FreeSurferProgressMonitor(tmp_path)
    with caplog.at_level("INFO", logger="anat"):
        monitor._consume()

    assert "FreeSurfer 3/7: Tessellate rh, right hemisphere" in caplog.messages


def test_metric_conversion_normalizes_freesurfer_prefixed_output(tmp_path: Path) -> None:
    output = (
        tmp_path
        / "surface_export"
        / "metric_conversion"
        / "sub-test_space-fsnative_hemi-L_thickness.shape.gii"
    )
    step = _create_metric_conversion_step(
        metric=tmp_path / "lh.thickness",
        surface=tmp_path / "lh.white",
        output=output,
        description="cortical thickness",
        env={},
        force=False,
    )

    assert step.prepare is not None
    assert step.finalize is not None
    step.prepare()
    converted = Path(step.command[-1])
    assert converted.name == "lh.sub-test_space-fsnative_hemi-L_thickness.shape.gii"
    converted.write_text("metric")
    step.finalize()
    assert output.read_text() == "metric"
    assert not converted.exists()
