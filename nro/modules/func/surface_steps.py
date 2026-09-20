"""Surface steps for functional preprocessing."""

from pathlib import Path

from nro.engine.execution import (
    ensure_directory,
)
from nro.orchestration.runner_graph import Step


def _create_wb_volume_to_surface_mapping_step(
    *,
    volume: Path,
    midthickness: Path,
    white: Path,
    pial: Path,
    out_metric: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "wb_command",
        "-volume-to-surface-mapping",
        str(volume),
        str(midthickness),
        str(out_metric),
        "-ribbon-constrained",
        str(white),
        str(pial),
    ]
    return Step.command_step(
        cmd,
        outputs=(out_metric,),
        inputs=(volume, midthickness, white, pial),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_metric.parent),
    )


def _create_wb_metric_resample_step(
    *,
    in_metric: Path,
    current_sphere: Path,
    new_sphere: Path,
    out_metric: Path,
    valid_roi_out: Path | None = None,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "wb_command",
        "-metric-resample",
        str(in_metric),
        str(current_sphere),
        str(new_sphere),
        "BARYCENTRIC",
        str(out_metric),
    ]
    if valid_roi_out is not None:
        cmd.extend(("-valid-roi-out", str(valid_roi_out)))
    return Step.command_step(
        cmd,
        outputs=(out_metric, *((valid_roi_out,) if valid_roi_out is not None else ())),
        inputs=(in_metric, current_sphere, new_sphere),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_metric.parent),
    )


def _create_wb_metric_mask_step(
    *,
    metric: Path,
    roi: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Mask a resampled metric to vertices supported by the source anatomy."""
    return Step.command_step(
        ["wb_command", "-metric-mask", str(metric), str(roi), str(output)],
        name="Mask Resampled Surface to Valid Anatomical Domain",
        outputs=(output,),
        inputs=(metric, roi),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(output.parent),
    )
