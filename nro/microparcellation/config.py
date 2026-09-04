from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from nro.configuration.store import main_configuration_factory
from nro.engine.targets import DEFAULT_SMOOTHING_MM


microparcellation_default = partial(
    main_configuration_factory,
    "microparcellation",
)


@dataclass(frozen=True)
class InputsConfig:
    functional: tuple[tuple[Path, ...], ...]
    domain: str = "surface"
    space: str | None = None
    smoothing_mm: int = DEFAULT_SMOOTHING_MM
    surface: tuple[Path, ...] = ()
    mask: tuple[Path, ...] | Path | None = None
    mask_threshold: float = field(
        default_factory=microparcellation_default("mask_threshold")
    )
    volume_connectivity: int = field(
        default_factory=microparcellation_default("volume_connectivity")
    )


@dataclass(frozen=True)
class CoarseningConfig:
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
        default_factory=microparcellation_default(
            "coarsening", "eigensolver_tolerance"
        )
    )


@dataclass(frozen=True)
class ConnectivityConfig:
    minimum_trs: int = field(
        default_factory=microparcellation_default("connectivity", "minimum_trs")
    )
    temporal_block_size: int = field(
        default_factory=microparcellation_default(
            "connectivity", "temporal_block_size"
        )
    )
    reliability_weighting: bool = field(
        default_factory=microparcellation_default("connectivity", "reliability_weighting")
    )
    reliability_vertex_block_size: int = field(
        default_factory=microparcellation_default("connectivity", "reliability_vertex_block_size")
    )
    global_signal_regression: bool = field(
        default_factory=microparcellation_default(
            "connectivity", "global_signal_regression"
        )
    )


@dataclass(frozen=True)
class QualityConfig:
    null_parcellations: int = field(
        default_factory=microparcellation_default("quality", "null_parcellations")
    )
    random_seed: int = field(default_factory=microparcellation_default("quality", "random_seed"))
    region_growing_attempts: int = field(
        default_factory=microparcellation_default("quality", "region_growing_attempts")
    )


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    work_directory: Path
    prefix: str
    overwrite: bool = field(
        default_factory=microparcellation_default("overwrite")
    )


@dataclass(frozen=True)
class ModuleConfig:
    inputs: InputsConfig
    output: OutputConfig
    wb_command: str
    coarsening: CoarseningConfig = field(default_factory=CoarseningConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


def validate_config(cfg: ModuleConfig) -> None:
    if cfg.inputs.domain not in {"surface", "volume"}:
        raise ValueError("inputs.domain must be 'surface' or 'volume'")
    if not cfg.inputs.functional:
        raise ValueError("At least one functional run is required")
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
        if not 0 <= cfg.inputs.mask_threshold < 1:
            raise ValueError("inputs.mask_threshold must be in [0, 1)")
        if cfg.inputs.volume_connectivity not in {6, 18, 26}:
            raise ValueError("inputs.volume_connectivity must be 6, 18, or 26")
    if cfg.coarsening.target_vertices < 2:
        raise ValueError("coarsening.target_vertices must be >= 2")
    if cfg.coarsening.iterations < 1:
        raise ValueError("coarsening.iterations must be positive")
    if cfg.coarsening.eigenvectors < 2 or cfg.coarsening.max_levels < 1:
        raise ValueError("coarsening.eigenvectors must be >= 2 and max_levels must be positive")
    if cfg.coarsening.exponential_temperature <= 0:
        raise ValueError("coarsening.exponential_temperature must be positive")
    if cfg.connectivity.temporal_block_size < 1 or cfg.connectivity.reliability_vertex_block_size < 1:
        raise ValueError("Connectivity block sizes must be positive")
    if cfg.connectivity.reliability_weighting and cfg.connectivity.minimum_trs < 8:
        raise ValueError("connectivity.minimum_trs must be at least 8 for quarter-split reliability")
    if cfg.connectivity.minimum_trs < 2:
        raise ValueError("connectivity.minimum_trs must be at least 2")
    if cfg.quality.null_parcellations < 1:
        raise ValueError("quality.null_parcellations must be positive")
    if cfg.quality.random_seed < 0:
        raise ValueError("quality.random_seed must be nonnegative")
    if cfg.quality.region_growing_attempts < 1:
        raise ValueError("quality.region_growing_attempts must be positive")
