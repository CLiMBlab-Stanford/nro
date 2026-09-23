"""Dormant FastSurfer reconstruction backend.

This module defines an executable FastSurfer plan without connecting it to the
anatomical runner.  The split plan lets a future backend selector schedule the
short segmentation stage on a GPU and the longer surface stage on a CPU worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nro.modules.anat.lesion_policy import (
    FASTSURFER_OCI_DIGEST,
    FASTSURFER_SOURCE_REVISION,
    FASTSURFER_VERSION,
)

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
    "surf/lh.sphere",
    "surf/rh.sphere",
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "surf/lh.thickness",
    "surf/rh.thickness",
    "surf/lh.sulc",
    "surf/rh.sulc",
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
        missing = tuple(
            path for path in self.outputs if not path.is_file() or not path.stat().st_size
        )
        if missing:
            return False, f"{self.name} outputs are incomplete: " + ", ".join(map(str, missing))
        return True, f"{self.name} outputs are complete."


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


def _container_prefix(
    *,
    runtime: str,
    image: Path,
    t1w: Path,
    subjects_directory: Path,
    license_file: Path,
    gpu: bool,
) -> tuple[str, ...]:
    command = [runtime, "exec"]
    if gpu:
        command.append("--nv")
    command.extend(
        (
            "--cleanenv",
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
