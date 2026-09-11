"""Typed configuration for the dynamic-connectivity module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from nro.configuration.schema import validate_parameters
from nro.configuration.store import main_configuration_factory
from nro.engine.targets import DEFAULT_SMOOTHING_MM

dynconn_default = partial(main_configuration_factory, "dynconn")


@dataclass(frozen=True)
class InputsConfig:
    """Cleaned runs and target identity selected for one output."""

    functional: tuple[tuple[Path, ...], ...]
    temporal_masks: tuple[Path, ...]
    domain: str
    space: str
    smoothing_mm: int = DEFAULT_SMOOTHING_MM


@dataclass(frozen=True)
class InclusionConfig:
    """Sidecar-based thresholds for admitting cleaned runs."""

    minimum_retained_frames: int = field(
        default_factory=dynconn_default("inclusion", "minimum_retained_frames")
    )
    minimum_retained_fraction: float = field(
        default_factory=dynconn_default("inclusion", "minimum_retained_fraction")
    )
    minimum_residual_design_dof: int = field(
        default_factory=dynconn_default("inclusion", "minimum_residual_design_dof")
    )
    minimum_participation_effective_rank: float = field(
        default_factory=dynconn_default("inclusion", "minimum_participation_effective_rank")
    )
    maximum_dominant_temporal_variance_fraction: float = field(
        default_factory=dynconn_default("inclusion", "maximum_dominant_temporal_variance_fraction")
    )
    minimum_usable_runs: int = field(
        default_factory=dynconn_default("inclusion", "minimum_usable_runs")
    )
    minimum_aggregate_retained_frames: int = field(
        default_factory=dynconn_default("inclusion", "minimum_aggregate_retained_frames")
    )


@dataclass(frozen=True)
class OutputConfig:
    """Public and private destinations for one module instance."""

    directory: Path
    work_directory: Path
    prefix: str
    overwrite: bool = field(default_factory=dynconn_default("overwrite"))


@dataclass(frozen=True)
class LowRankConfig:
    """Scientific controls for the compressed correlation representation."""

    dimensions: int = field(default_factory=dynconn_default("low_rank_options", "dimensions"))
    oversampling: int = field(default_factory=dynconn_default("low_rank_options", "oversampling"))
    power_iterations: int = field(
        default_factory=dynconn_default("low_rank_options", "power_iterations")
    )


@dataclass(frozen=True)
class ModuleConfig:
    """Complete dynamic-connectivity module configuration."""

    inputs: InputsConfig
    output: OutputConfig
    inclusion: InclusionConfig = field(default_factory=InclusionConfig)
    low_rank: bool = field(default_factory=dynconn_default("low_rank"))
    low_rank_options: LowRankConfig = field(default_factory=LowRankConfig)
    weighting: str = field(default_factory=dynconn_default("weighting"))


def validate_config(cfg: ModuleConfig) -> None:
    """Reject incomplete targets and invalid admission settings."""

    if cfg.inputs.domain not in {"surface", "volume"}:
        raise ValueError("inputs.domain must be 'surface' or 'volume'")
    if not cfg.inputs.functional:
        raise ValueError("At least one cleaned functional run is required")
    if len(cfg.inputs.functional) != len(cfg.inputs.temporal_masks):
        raise ValueError("Each cleaned run must have one temporal-mask table")
    validate_parameters(
        "dynconn",
        {
            "inclusion": asdict(cfg.inclusion),
            "weighting": cfg.weighting,
            "low_rank": cfg.low_rank,
            "low_rank_options": asdict(cfg.low_rank_options),
        },
    )
