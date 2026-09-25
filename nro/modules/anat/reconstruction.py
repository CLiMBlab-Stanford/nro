"""Backend-neutral construction of the anatomical surface-reconstruction stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from nro.orchestration.runner_graph import StagePlan

from .fastsurfer import create_fastsurfer_steps
from .steps import _create_recon_all_step


@dataclass(frozen=True)
class SurfaceReconstructionInputs:
    """Scientific inputs available to a surface-reconstruction backend.

    Upstream anatomy determines whether ``t1w`` is observed or lesion-inpainted.
    Backend adapters declare only the paths they actually consume on their DAG
    steps, so an unused auxiliary input cannot affect step freshness.
    """

    t1w: Path | None
    t2w: Path | None
    brain_mask: Path

    def __post_init__(self) -> None:
        """Require at least one anatomical contrast at the stage boundary."""
        if self.t1w is None and self.t2w is None:
            raise ValueError("Surface reconstruction requires a T1w or T2w input")


@dataclass(frozen=True)
class SurfaceReconstructionResources:
    """Execution resources shared by surface-reconstruction backends."""

    runtime: str
    freesurfer_image: Path | None
    fastsurfer_image: Path | None
    license_file: Path
    subjects_dir: Path
    staging_subjects_dir: Path
    subject: str
    cpu_threads: int


@dataclass(frozen=True)
class SurfaceReconstructionProducts:
    """Backend-independent products exposed to downstream anatomy stages."""

    subject_dir: Path


def create_surface_reconstruction_plan(
    *,
    engine: str,
    inputs: SurfaceReconstructionInputs,
    resources: SurfaceReconstructionResources,
    run_child: Callable[..., Any],
    env: Mapping[str, str],
    force: bool,
) -> StagePlan[SurfaceReconstructionProducts]:
    """Construct the selected backend behind one stable stage interface.

    FreeSurfer consumes the optional T2w image and external whole-brain mask.
    FastSurfer consumes only T1w and manages its own segmentation mask. Both
    adapters produce the same private FreeSurfer-style subject directory.
    """
    subject_dir = resources.subjects_dir / resources.subject
    products = SurfaceReconstructionProducts(subject_dir=subject_dir)
    if engine == "freesurfer":
        if resources.freesurfer_image is None:
            raise ValueError("FreeSurfer reconstruction requires its container image")
        step = _create_recon_all_step(
            run_child=run_child,
            env=dict(env),
            t1w=inputs.t1w,
            t2w=inputs.t2w,
            brain_mask=inputs.brain_mask,
            subjects_dir=resources.subjects_dir,
            fs_subject=resources.subject,
            runtime=resources.runtime,
            image=resources.freesurfer_image,
            license_file=resources.license_file,
            force=force,
        )
        return StagePlan(steps=(step,), products=products)
    if engine == "fastsurfer":
        if inputs.t1w is None:
            raise ValueError("FastSurfer reconstruction requires a T1w input")
        if resources.fastsurfer_image is None:
            raise ValueError("FastSurfer reconstruction requires its container image")
        steps = create_fastsurfer_steps(
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
            force=force,
        )
        return StagePlan(steps=steps, products=products)
    raise ValueError(f"Unsupported surface-reconstruction engine: {engine}")
