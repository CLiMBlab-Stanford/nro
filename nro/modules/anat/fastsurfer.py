"""FastSurfer reconstruction backend with separate GPU and CPU stages."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np

from nro.engine.io import atomic_output_path, atomic_write_text
from nro.modules.anat.policy import (
    FASTSURFER_OCI_DIGEST,
    FASTSURFER_SOURCE_REVISION,
    FASTSURFER_VERSION,
    FASTSURFER_VOXEL_SIZE_MM,
)
from nro.orchestration.runner_graph import Step

_SEGMENTATION_OUTPUTS = (
    "scripts/deep-seg.log",
    "mri/orig.mgz",
    "mri/orig_nu.mgz",
    "mri/aparc.DKTatlas+aseg.deep.mgz",
    "mri/mask.mgz",
)

_SURFACE_OUTPUTS = (
    "scripts/recon-surf.done",
    "mri/T1.mgz",
    "mri/aseg.mgz",
    "mri/brainmask.mgz",
    "mri/ribbon.mgz",
    "stats/aseg.stats",
    "stats/lh.aparc.stats",
    "stats/rh.aparc.stats",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
    "surf/lh.inflated",
    "surf/rh.inflated",
    "surf/lh.smoothwm",
    "surf/rh.smoothwm",
    "surf/lh.sphere",
    "surf/rh.sphere",
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "surf/lh.thickness",
    "surf/rh.thickness",
    "surf/lh.sulc",
    "surf/rh.sulc",
)


def _validate_files(paths: tuple[Path, ...], label: str) -> tuple[bool, str]:
    missing = [str(path) for path in paths if not path.is_file() or not path.stat().st_size]
    if missing:
        return False, f"{label} outputs are incomplete: " + ", ".join(missing)
    return True, f"{label} outputs are complete."


def _reconcile_mask(subject_directory: Path) -> None:
    """Include every segmented voxel in FastSurfer's surface-processing mask."""
    mri_directory = subject_directory / "mri"
    mask_path = mri_directory / "mask.mgz"
    segmentation_path = mri_directory / "aparc.DKTatlas+aseg.deep.mgz"
    mask_image = nib.load(str(mask_path))
    segmentation_image = nib.load(str(segmentation_path))
    if mask_image.shape != segmentation_image.shape or not np.allclose(
        mask_image.affine, segmentation_image.affine, atol=1e-4
    ):
        raise ValueError("FastSurfer mask and segmentation grids differ")
    mask = np.asarray(mask_image.dataobj)
    segmentation = np.asarray(segmentation_image.dataobj)
    added = (segmentation != 0) & (mask == 0)
    reconciled = np.asarray(mask).copy()
    reconciled[segmentation != 0] = 1
    temporary = mask_path.with_name(f".partial-{mask_path.name}")
    temporary.unlink(missing_ok=True)
    nib.save(nib.MGHImage(reconciled, mask_image.affine, mask_image.header), str(temporary))
    os.replace(temporary, mask_path)
    atomic_write_text(
        subject_directory / "stats" / "nro-mask-reconciliation.json",
        json.dumps(
            {
                "Method": "segmentation_union",
                "Mask": str(mask_path),
                "Segmentation": str(segmentation_path),
                "AddedVoxelCount": int(added.sum()),
                "FinalMaskVoxelCount": int(np.count_nonzero(reconciled)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


@dataclass(frozen=True)
class FastSurferStage:
    """Describe one independently executable stage of a FastSurfer run."""

    name: str
    resource: Literal["cpu", "gpu"]
    command: tuple[str, ...]
    outputs: tuple[Path, ...]

    def validate(self) -> tuple[bool, str]:
        """Check that every required stage output is a nonempty regular file."""
        return _validate_files(self.outputs, self.name)


@dataclass(frozen=True)
class FastSurferPlan:
    """Hold the two stages and pinned identity of one FastSurfer reconstruction."""

    subject_directory: Path
    segmentation: FastSurferStage
    surfaces: FastSurferStage
    version: str = FASTSURFER_VERSION
    source_revision: str = FASTSURFER_SOURCE_REVISION
    image_digest: str = FASTSURFER_OCI_DIGEST

    def validate(self) -> tuple[bool, str]:
        """Validate the complete subject directory in execution order."""
        for stage in (self.segmentation, self.surfaces):
            valid, reason = stage.validate()
            if not valid:
                return valid, reason
        return True, "FastSurfer segmentation and surface outputs are complete."


def _subject_id(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("FastSurfer subject IDs must be nonempty path components")
    return value


def _validate_segmentation_archive(archive: Path, *, subject: str) -> tuple[bool, str]:
    """Check archive containment and the files required by the surface stage."""
    if not archive.is_file():
        return False, f"FastSurfer segmentation archive is missing: {archive}"
    root = f"{subject}/"
    try:
        with tarfile.open(archive, "r") as bundle:
            members = bundle.getmembers()
    except (OSError, tarfile.TarError):
        return False, f"FastSurfer segmentation archive is invalid: {archive}"
    if any(member.name != subject and not member.name.startswith(root) for member in members):
        return False, f"FastSurfer segmentation archive has an unexpected root: {archive}"
    by_name = {member.name: member for member in members}
    missing = [
        relative
        for relative in _SEGMENTATION_OUTPUTS
        if not (
            (member := by_name.get(f"{subject}/{relative}")) and member.isfile() and member.size
        )
    ]
    if missing:
        return False, "FastSurfer segmentation archive is incomplete: " + ", ".join(missing)
    return True, "FastSurferVINN segmentation outputs are complete."


def _container_prefix(
    *,
    runtime: str,
    image: Path,
    t1w: Path,
    subjects_directory: Path,
    license_file: Path,
    gpu: bool,
    cuda_visible_devices: str | None = None,
) -> tuple[str, ...]:
    if cuda_visible_devices is not None and not gpu:
        raise ValueError("CUDA device visibility applies only to GPU FastSurfer stages")
    command = [runtime, "exec"]
    if gpu:
        command.append("--nv")
    command.append("--cleanenv")
    if cuda_visible_devices is not None:
        command.extend(("--env", f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}"))
    command.extend(
        (
            "--bind",
            f"{t1w.parent.resolve()}:/input:ro",
            "--bind",
            f"{subjects_directory.resolve()}:/subjects",
            "--bind",
            f"{license_file.resolve()}:/license.txt:ro",
            str(image.resolve()),
            "/fastsurfer/run_fastsurfer.sh",
        )
    )
    return tuple(command)


def create_fastsurfer_plan(
    *,
    runtime: str,
    image: Path,
    t1w: Path,
    subjects_directory: Path,
    subject: str,
    license_file: Path,
    segmentation_threads: int,
    surface_threads: int,
    cuda_visible_devices: str | None = None,
) -> FastSurferPlan:
    """Build the tested FastSurfer 2.5.4 segmentation and surface invocations.

    The commands share one private FreeSurfer-style subject directory.  Run the
    segmentation command first.  The surface command consumes its files through
    ``--surf_only`` and does not request a GPU.
    """
    subject = _subject_id(subject)
    if segmentation_threads < 1 or surface_threads < 1:
        raise ValueError("FastSurfer thread counts must be positive")
    subject_directory = subjects_directory / subject
    common = (
        "--t1",
        f"/input/{t1w.name}",
        "--sid",
        subject,
        "--sd",
        "/subjects",
        "--fs_license",
        "/license.txt",
        "--no_cereb",
        "--no_hypothal",
        "--no_cc",
        "--vox_size",
        str(FASTSURFER_VOXEL_SIZE_MM),
    )
    segmentation = FastSurferStage(
        name="FastSurferVINN Segmentation",
        resource="gpu",
        command=(
            *_container_prefix(
                runtime=runtime,
                image=image,
                t1w=t1w,
                subjects_directory=subjects_directory,
                license_file=license_file,
                gpu=True,
                cuda_visible_devices=cuda_visible_devices,
            ),
            *common,
            "--seg_only",
            "--device",
            "cuda",
            "--viewagg_device",
            "cuda",
            "--threads_seg",
            str(segmentation_threads),
        ),
        outputs=tuple(subject_directory / relative for relative in _SEGMENTATION_OUTPUTS),
    )
    surfaces = FastSurferStage(
        name="FastSurfer Surface Reconstruction",
        resource="cpu",
        command=(
            *_container_prefix(
                runtime=runtime,
                image=image,
                t1w=t1w,
                subjects_directory=subjects_directory,
                license_file=license_file,
                gpu=False,
            ),
            *common,
            "--surf_only",
            "--device",
            "cpu",
            "--threads_surf",
            str(surface_threads),
            "--fsaparc",
        ),
        outputs=tuple(subject_directory / relative for relative in _SURFACE_OUTPUTS),
    )
    return FastSurferPlan(
        subject_directory=subject_directory,
        segmentation=segmentation,
        surfaces=surfaces,
    )


def create_fastsurfer_steps(
    *,
    run_child: Callable[..., Any],
    runtime: str,
    image: Path,
    t1w: Path,
    staging_subjects_dir: Path,
    subjects_dir: Path,
    subject: str,
    license_file: Path,
    segmentation_threads: int,
    surface_threads: int,
    force: bool,
) -> tuple[Step, Step]:
    """Create GPU segmentation and CPU surface-reconstruction steps.

    FastSurfer's segmentation is first written to a private staging tree and
    published as one complete archive. The CPU stage extracts that archive into
    a freshly managed final subject directory, so either stage can restart
    without contaminating the other.
    """
    subject = _subject_id(subject)
    if segmentation_threads < 1 or surface_threads < 1:
        raise ValueError("FastSurfer thread counts must be positive")
    staging_subject = staging_subjects_dir / subject
    final_subject = subjects_dir / subject
    segmentation_outputs = tuple(staging_subject / value for value in _SEGMENTATION_OUTPUTS)
    surface_outputs = (
        *(final_subject / value for value in _SURFACE_OUTPUTS),
        final_subject / "stats" / "nro-mask-reconciliation.json",
    )
    segmentation_archive = staging_subjects_dir / f"{subject}.tar"
    surface_breadcrumb = final_subject / ".nro_fastsurfer_complete"

    def segmentation_action() -> None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible_devices:
            raise RuntimeError("FastSurfer segmentation requires a Slurm-assigned GPU")
        if staging_subject.exists():
            shutil.rmtree(staging_subject)
        staging_subjects_dir.mkdir(parents=True, exist_ok=True)
        plan = create_fastsurfer_plan(
            runtime=runtime,
            image=image,
            t1w=t1w,
            subjects_directory=staging_subjects_dir,
            subject=subject,
            license_file=license_file,
            segmentation_threads=segmentation_threads,
            surface_threads=surface_threads,
            cuda_visible_devices=visible_devices,
        )
        run_child(plan.segmentation.command, direct=True, stream_output=True)
        valid, reason = _validate_files(segmentation_outputs, "FastSurferVINN segmentation")
        if not valid:
            raise RuntimeError(reason)
        with atomic_output_path(segmentation_archive) as staged_archive:
            with tarfile.open(staged_archive, "w") as bundle:
                bundle.dereference = True
                bundle.add(staging_subject, arcname=subject)
        shutil.rmtree(staging_subject)

    def segmentation_valid() -> tuple[bool, str]:
        return _validate_segmentation_archive(segmentation_archive, subject=subject)

    def surface_action() -> None:
        if final_subject.exists():
            raise RuntimeError("Runner did not clear the stale FastSurfer subject directory")
        subjects_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(segmentation_archive, "r") as bundle:
            bundle.extractall(subjects_dir, filter="data")
        if not final_subject.is_dir():
            raise RuntimeError("FastSurfer segmentation archive lacks its subject directory")
        _reconcile_mask(final_subject)
        plan = create_fastsurfer_plan(
            runtime=runtime,
            image=image,
            t1w=t1w,
            subjects_directory=subjects_dir,
            subject=subject,
            license_file=license_file,
            segmentation_threads=segmentation_threads,
            surface_threads=surface_threads,
        )
        run_child(plan.surfaces.command, direct=True, stream_output=True)

    scientific = {
        "backend": "FastSurfer",
        "version": FASTSURFER_VERSION,
        "source_revision": FASTSURFER_SOURCE_REVISION,
        "oci_digest": FASTSURFER_OCI_DIGEST,
        "voxel_size_mm": FASTSURFER_VOXEL_SIZE_MM,
        "mask_reconciliation": "segmentation_union",
        "options": ["fsaparc", "no_cereb", "no_hypothal", "no_cc"],
    }
    return (
        Step.python(
            id="fastsurfer-segmentation",
            name="FastSurferVINN Segmentation",
            inputs=(t1w, image, license_file),
            outputs=(segmentation_archive,),
            action=segmentation_action,
            validate=segmentation_valid,
            force=force,
            resource_class="gpu",
            parameters={**scientific, "stage": "segmentation"},
        ),
        Step.directory_step(
            id="fastsurfer-surfaces",
            name="FastSurfer Surface Reconstruction",
            directory=final_subject,
            breadcrumb=surface_breadcrumb,
            inputs=(segmentation_archive, image, license_file),
            outputs=surface_outputs,
            action=surface_action,
            validate=lambda: _validate_files(surface_outputs, "FastSurfer surface reconstruction"),
            force=force,
            parameters={**scientific, "stage": "surfaces"},
        ),
    )
