"""Validate and represent microparcellation configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from nro.configuration.schema import validate_parameters
from nro.configuration.store import main_configuration_factory
from nro.engine.targets import DEFAULT_SMOOTHING_MM

microparcellation_default = partial(
    main_configuration_factory,
    "microparcellation",
)


@dataclass(frozen=True)
class InputsConfig:
    """Cleaned runs, spatial geometry, masks, and target identity for microparcellation."""

    functional: tuple[tuple[Path, ...], ...]
    temporal_masks: tuple[Path, ...] = ()
    domain: str = "surface"
    space: str | None = None
    smoothing_mm: int = DEFAULT_SMOOTHING_MM
    surface: tuple[Path, ...] = ()
    mask: tuple[Path, ...] | Path | None = None
    mask_threshold: float = field(default_factory=microparcellation_default("mask_threshold"))
    volume_connectivity: int = field(
        default_factory=microparcellation_default("volume_connectivity")
    )


@dataclass(frozen=True)
class CoarseningConfig:
    """Target parcel count and local-variation spectral coarsening controls."""

    target_vertices: int = field(
        default_factory=microparcellation_default("coarsening", "target_vertices")
    )
    iterations: int = field(default_factory=microparcellation_default("coarsening", "iterations"))
    exponential_temperature: float = field(
        default_factory=microparcellation_default("coarsening", "exponential_temperature")
    )
    eigenvectors: int = field(
        default_factory=microparcellation_default("coarsening", "eigenvectors")
    )
    max_levels: int = field(default_factory=microparcellation_default("coarsening", "max_levels"))
    eigensolver_tolerance: float = field(
        default_factory=microparcellation_default("coarsening", "eigensolver_tolerance")
    )


@dataclass(frozen=True)
class ConnectivityConfig:
    """Sidecar-based run admission thresholds and streaming connectivity controls."""

    minimum_retained_frames: int = field(
        default_factory=microparcellation_default("connectivity", "minimum_retained_frames")
    )
    minimum_retained_fraction: float = field(
        default_factory=microparcellation_default("connectivity", "minimum_retained_fraction")
    )
    minimum_residual_design_dof: int = field(
        default_factory=microparcellation_default("connectivity", "minimum_residual_design_dof")
    )
    minimum_participation_effective_rank: float = field(
        default_factory=microparcellation_default(
            "connectivity", "minimum_participation_effective_rank"
        )
    )
    maximum_dominant_temporal_variance_fraction: float = field(
        default_factory=microparcellation_default(
            "connectivity", "maximum_dominant_temporal_variance_fraction"
        )
    )
    minimum_usable_runs: int = field(
        default_factory=microparcellation_default("connectivity", "minimum_usable_runs")
    )
    minimum_aggregate_retained_frames: int = field(
        default_factory=microparcellation_default(
            "connectivity", "minimum_aggregate_retained_frames"
        )
    )
    temporal_block_size: int = field(
        default_factory=microparcellation_default("connectivity", "temporal_block_size")
    )
    reliability_weighting: bool = field(
        default_factory=microparcellation_default("connectivity", "reliability_weighting")
    )
    reliability_vertex_block_size: int = field(
        default_factory=microparcellation_default("connectivity", "reliability_vertex_block_size")
    )
    global_signal_regression: bool = field(
        default_factory=microparcellation_default("connectivity", "global_signal_regression")
    )


@dataclass(frozen=True)
class QualityConfig:
    """Spatial-null generation and connectome diagnostic approximation settings."""

    split_half_block_frames: int = field(
        default_factory=microparcellation_default("quality", "split_half_block_frames")
    )
    null_parcellations: int = field(
        default_factory=microparcellation_default("quality", "null_parcellations")
    )
    random_seed: int = field(default_factory=microparcellation_default("quality", "random_seed"))
    region_growing_attempts: int = field(
        default_factory=microparcellation_default("quality", "region_growing_attempts")
    )
    connectome_power_iterations: int = field(
        default_factory=microparcellation_default("quality", "connectome_power_iterations")
    )


@dataclass(frozen=True)
class OutputConfig:
    """Public/private output directories, prefix, and overwrite policy."""

    directory: Path
    work_directory: Path
    prefix: str
    overwrite: bool = field(default_factory=microparcellation_default("overwrite"))


@dataclass(frozen=True)
class ModuleConfig:
    """Complete input, coarsening, connectivity, quality, and output configuration."""

    inputs: InputsConfig
    output: OutputConfig
    coarsening: CoarseningConfig = field(default_factory=CoarseningConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


def validate_config(cfg: ModuleConfig) -> None:
    """Reject unsupported domains, missing input resources, and inconsistent algorithm settings."""
    if cfg.inputs.domain not in {"surface", "volume"}:
        raise ValueError("inputs.domain must be 'surface' or 'volume'")
    if not cfg.inputs.functional:
        raise ValueError("At least one functional run is required")
    if len(cfg.inputs.temporal_masks) != len(cfg.inputs.functional):
        raise ValueError("inputs.temporal_masks must contain one mask for each functional run")
    if cfg.inputs.smoothing_mm < 0:
        raise ValueError("inputs.smoothing_mm must be nonnegative")
    if cfg.inputs.domain == "surface":
        if not cfg.inputs.surface:
            raise ValueError("Surface microparcellation requires surface geometry")
        if cfg.inputs.mask is not None and isinstance(cfg.inputs.mask, Path):
            raise ValueError("Surface masks must be an ordered tuple of metric files")
    else:
        if cfg.inputs.surface:
            raise ValueError("Volumetric microparcellation does not accept surface geometry")
        if not isinstance(cfg.inputs.mask, Path):
            raise ValueError("Volumetric microparcellation requires a gray-matter volume mask")
    validate_parameters(
        "microparcellation",
        {
            "coarsening": asdict(cfg.coarsening),
            "connectivity": asdict(cfg.connectivity),
            "quality": asdict(cfg.quality),
            "mask_threshold": cfg.inputs.mask_threshold,
            "volume_connectivity": cfg.inputs.volume_connectivity,
        },
    )
