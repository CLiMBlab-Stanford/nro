"""High-level stage plans for the functional preprocessing DAG."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from nro.configuration.hardware import GradientUnwarpingResolution
from nro.configuration.schema import scientific_values
from nro.engine.execution import ensure_directory
from nro.engine.gradient_unwarping import create_gradient_unwarping_step
from nro.engine.image_paths import nifti_stem
from nro.engine.io import read_json, write_json
from nro.modules.func.marss import MarssOutputs, create_marss_motion_step, create_marss_step
from nro.orchestration.runner_graph import StagePlan, Step
from nro.orchestration.runner_support import write_completion_breadcrumb
from nro.orchestration.runtime import selected_configuration_fingerprint

from .config import Inputs, Options, functional_config_payload
from .reference_steps import (
    _create_functional_reference_selection_step,
    _create_robust_bold_reference_step,
)
from .resampling_steps import _create_afni_warp_step, _create_world_warp_step


@dataclass(frozen=True)
class InitializationProducts:
    """Configuration provenance shared with publication steps."""

    configuration: dict[str, object]


@dataclass(frozen=True)
class PreparedInputs:
    """Sources and optional corrections passed to reference construction."""

    inputs: Inputs
    processing_bold: Path
    resampling_source: Path
    processing_volume_count: int
    gradient_relative_warp: Path | None
    gradient_directory: Path | None
    marss: MarssOutputs | None


@dataclass(frozen=True)
class ReferenceProducts:
    """Robust motion and selected registration references for later stages."""

    robust_reference: Path
    robust_metadata: Path
    selected_reference: Path
    epi_to_reference: Path
    selected_metadata: Path
    gradient_afni_warp: Path | None


def plan_initialization(
    *,
    opts: Options,
    derivative_root: Path,
    gradient_unwarping: GradientUnwarpingResolution,
    required_commands: Sequence[str],
    require_commands: Callable[[Sequence[str]], None],
) -> StagePlan[InitializationProducts]:
    """Declare output initialization, configuration capture, and dependency checks."""
    initialized = opts.work_dir / "initialized.complete"

    def initialize_outputs() -> None:
        ensure_directory(derivative_root)
        ensure_directory(opts.out_dir)
        ensure_directory(opts.work_dir)
        write_completion_breadcrumb(initialized, "Functional outputs initialized\n")

    configuration = {
        "configuration": scientific_values(
            "func",
            functional_config_payload(opts, gradient_unwarping=gradient_unwarping),
        ),
        "configuration_fingerprint": selected_configuration_fingerprint(),
    }
    configuration_snapshot = opts.work_dir / "configuration.json"

    def validate_configuration() -> tuple[bool, str]:
        try:
            current = read_json(configuration_snapshot)
        except (OSError, ValueError, TypeError):
            return False, "Functional configuration snapshot is missing or unreadable."
        if isinstance(current.get("configuration"), dict):
            current["configuration"] = scientific_values("func", current["configuration"])
        if current != configuration:
            return False, "Functional configuration changed."
        return True, "Functional configuration is unchanged."

    dependency_check = opts.work_dir / "dependencies.complete"

    def check_dependencies() -> None:
        require_commands(required_commands)
        write_completion_breadcrumb(dependency_check, "Functional dependencies available\n")

    return StagePlan(
        steps=(
            Step.python(
                name="Initialize Functional Outputs",
                outputs=(initialized,),
                action=initialize_outputs,
            ),
            Step.python(
                name="Write Functional Configuration",
                outputs=(configuration_snapshot,),
                inputs=(initialized,),
                force=opts.overwrite,
                action=lambda: write_json(configuration_snapshot, configuration),
                validate=validate_configuration,
            ),
            Step.python(
                name="Check Functional Dependencies",
                outputs=(dependency_check,),
                force=opts.overwrite,
                action=check_dependencies,
            ),
        ),
        products=InitializationProducts(configuration=configuration),
    )


def plan_input_preparation(
    *,
    opts: Options,
    inputs: Inputs,
    epi_metadata: Mapping[str, Any],
    epi_metadata_sources: Sequence[Path],
    gradient_resolutions: Mapping[str, GradientUnwarpingResolution],
    environment: Mapping[str, str],
    run_stem: str,
    run_base: str,
    source_volume_count: int,
    run_child: Callable[..., str | None],
) -> StagePlan[PreparedInputs]:
    """Declare debug selection, MARSS, and gradient-unwarping preparation."""
    steps: list[Step] = []
    processing_volume_count = source_volume_count
    processing_bold = inputs.epi
    if int(opts.debug_first_nvols) > 0:
        count = int(opts.debug_first_nvols)
        processing_volume_count = min(count, source_volume_count)
        debug_dir = opts.work_dir / "debug"
        debug_bold = debug_dir / f"{run_stem}_first{count:04d}.nii.gz"
        steps.append(
            Step.command_step(
                ["fslroi", str(inputs.epi), str(debug_bold), "0", str(count)],
                name="Select Debug BOLD Volumes",
                outputs=(debug_bold,),
                inputs=(inputs.epi,),
                force=opts.overwrite,
                env=environment,
                prepare=lambda: ensure_directory(debug_dir),
            )
        )
        processing_bold = debug_bold

    marss_outputs = None
    marss_mode = str(opts.marss_mode).strip().lower()
    if marss_mode not in {"off", "diagnose", "auto"}:
        raise SystemExit(
            f"Unsupported MARSS mode {opts.marss_mode!r}; expected off, diagnose, or auto."
        )
    if marss_mode != "off":
        marss_dir = opts.work_dir / "marss"
        motion_step, marss_motion = create_marss_motion_step(
            run_child=run_child,
            source_bold=processing_bold,
            work_dir=marss_dir / "motion",
            env=environment,
            force=opts.overwrite,
        )
        steps.append(motion_step)
        marss_step, marss_outputs = create_marss_step(
            run_child=run_child,
            source_bold=processing_bold,
            metadata=epi_metadata,
            metadata_sources=epi_metadata_sources,
            motion_parameters=marss_motion,
            work_dir=marss_dir,
            artifact_dir=opts.out_dir,
            run_stem=run_base,
            mode=marss_mode,
            min_multiband_factor=int(opts.marss_min_multiband_factor),
            chunk_volumes=max(1, int(opts.io_chunk_vols)),
            force=opts.overwrite,
        )
        steps.append(marss_step)
        processing_bold = marss_outputs.bold

    resampling_source = processing_bold
    gradient_relative_warp: Path | None = None
    gradient_directory: Path | None = None
    bold_resolution = gradient_resolutions["epi"]
    if bold_resolution.applied:
        gradient_directory = opts.work_dir / "gradient_unwarping" / "bold"
        corrected_bold = gradient_directory / f"{run_stem}_desc-gradientCorrected_bold.nii.gz"
        gradient_relative_warp = gradient_directory / f"{run_stem}_gradient_warp.nii.gz"
        steps.append(
            create_gradient_unwarping_step(
                run_child=run_child,
                source=resampling_source,
                corrected=corrected_bold,
                warp=gradient_relative_warp,
                metadata=gradient_directory / f"{run_stem}_gradient.json",
                resolution=bold_resolution,
                runtime=opts.gradient_unwarp_runtime,
                image=opts.gradient_unwarp_image,
                force=opts.overwrite,
            )
        )
        processing_bold = corrected_bold

    def corrected_auxiliary(kind: str, source: Path | None) -> tuple[Path | None, Step | None]:
        resolution = gradient_resolutions[kind]
        if source is None or not resolution.applied:
            return source, None
        directory = opts.work_dir / "gradient_unwarping" / kind
        corrected = directory / source.name
        stem = nifti_stem(source)
        return corrected, create_gradient_unwarping_step(
            run_child=run_child,
            source=source,
            corrected=corrected,
            warp=directory / f"{stem}_gradient_warp.nii.gz",
            metadata=directory / f"{stem}_gradient.json",
            resolution=resolution,
            runtime=opts.gradient_unwarp_runtime,
            image=opts.gradient_unwarp_image,
            force=opts.overwrite,
        )

    sbref, sbref_step = corrected_auxiliary("sbref", inputs.sbref)
    se1, se1_step = corrected_auxiliary("se1", inputs.se1)
    se2, se2_step = corrected_auxiliary("se2", inputs.se2)
    steps.extend(step for step in (sbref_step, se1_step, se2_step) if step is not None)
    prepared_inputs = replace(inputs, sbref=sbref, se1=se1, se2=se2)
    return StagePlan(
        steps=tuple(steps),
        products=PreparedInputs(
            inputs=prepared_inputs,
            processing_bold=processing_bold,
            resampling_source=resampling_source,
            processing_volume_count=processing_volume_count,
            gradient_relative_warp=gradient_relative_warp,
            gradient_directory=gradient_directory,
            marss=marss_outputs,
        ),
    )


def plan_reference_preparation(
    *,
    opts: Options,
    inputs: Inputs,
    processing_bold: Path,
    processing_volume_count: int,
    run_stem: str,
    motion_directory: Path,
    environment: dict[str, str],
    epi_metadata: dict[str, Any],
    epi_metadata_sources: tuple[Path, ...],
    sbref_metadata_sources: tuple[Path, ...],
    gradient_relative_warp: Path | None,
    gradient_directory: Path | None,
    run_child: Callable[..., str | None],
) -> StagePlan[ReferenceProducts]:
    """Declare robust-reference construction and optional SBRef selection."""
    steps: list[Step] = []
    robust = _create_robust_bold_reference_step(
        run_child=run_child,
        epi_in=processing_bold,
        volume_count=processing_volume_count,
        run_stem=run_stem,
        mc_dir=motion_directory,
        env=environment,
        force=opts.overwrite,
    )
    steps.append(robust.step)

    gradient_afni_warp: Path | None = None
    if gradient_relative_warp is not None and gradient_directory is not None:
        gradient_world_warp = gradient_directory / f"{run_stem}_gradient_world.nii.gz"
        gradient_afni_warp = gradient_directory / f"{run_stem}_gradient_afni_lps.nii.gz"
        steps.extend(
            (
                _create_world_warp_step(
                    run_child=run_child,
                    motion_ref_3d=robust.reference,
                    ref_3d=robust.reference,
                    fnirt_warp=gradient_relative_warp,
                    world_warp=gradient_world_warp,
                    env=environment,
                    force=opts.overwrite,
                ),
                _create_afni_warp_step(
                    world_warp=gradient_world_warp,
                    motion_ref_3d=robust.reference,
                    ref_3d=robust.reference,
                    afni_warp=gradient_afni_warp,
                    force=opts.overwrite,
                ),
            )
        )

    selected = _create_functional_reference_selection_step(
        run_child=run_child,
        robust_ref=robust.reference,
        epi_metadata=epi_metadata,
        epi_metadata_sources=epi_metadata_sources,
        sbref=inputs.sbref,
        sbref_json=inputs.sbref_json,
        sbref_metadata=inputs.sbref_metadata,
        sbref_metadata_sources=sbref_metadata_sources,
        sbref_metadata_inheritance=inputs.sbref_metadata_inheritance,
        work_dir=opts.work_dir / "reference" / "sbref_qc",
        env=environment,
        max_rotation_degrees=opts.sbref_max_rigid_rotation_degrees,
        max_displacement_mm=opts.sbref_max_rigid_displacement_mm,
        min_support_overlap=opts.sbref_min_support_overlap,
        min_correlation=opts.sbref_min_intensity_correlation,
        force=opts.overwrite,
    )
    steps.append(selected.step)
    return StagePlan(
        steps=tuple(steps),
        products=ReferenceProducts(
            robust_reference=robust.reference,
            robust_metadata=robust.metadata,
            selected_reference=selected.image,
            epi_to_reference=selected.epi_to_reference,
            selected_metadata=selected.metadata,
            gradient_afni_warp=gradient_afni_warp,
        ),
    )
