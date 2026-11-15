"""Typed configuration for the functional-connectivity gradient module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from nro.definitions.schema import validate_parameters
from nro.definitions.store import main_configuration_factory

from .contract import BACKENDS

fcgradients_default = partial(main_configuration_factory, "fcgradients")


@dataclass(frozen=True)
class InputsConfig:
    """Dynamic-connectivity series and the geometry needed to take gradients.

    Surface targets need one midthickness surface per hemisphere, in the order of
    the series' cortical structures. Volume targets need a gray-matter mask; the
    mask is resampled to the series grid when the grids differ.
    """

    source_manifest: Path
    timeseries: Path
    representation: str
    domain: str
    space: str
    smoothing_mm: int
    surfaces: tuple[Path, ...] = ()
    mask: Path | None = None
    mask_threshold: float = field(default_factory=fcgradients_default("mask_threshold"))


@dataclass(frozen=True)
class WorkbenchConfig:
    """Options passed to ``wb_command -cifti-correlation-gradient``."""

    fisher_z: bool = field(default_factory=fcgradients_default("workbench", "fisher_z"))
    double_correlation: bool = field(
        default_factory=fcgradients_default("workbench", "double_correlation")
    )
    fisher_z_first: bool = field(default_factory=fcgradients_default("workbench", "fisher_z_first"))
    presmooth_fwhm_mm: float = field(
        default_factory=fcgradients_default("workbench", "presmooth_fwhm_mm")
    )
    surface_exclude_mm: float = field(
        default_factory=fcgradients_default("workbench", "surface_exclude_mm")
    )
    volume_exclude_mm: float = field(
        default_factory=fcgradients_default("workbench", "volume_exclude_mm")
    )
    memory_limit_gb: float | None = field(
        default_factory=fcgradients_default("workbench", "memory_limit_gb")
    )


@dataclass(frozen=True)
class OutputConfig:
    """Public and private destinations for one module work item."""

    directory: Path
    work_directory: Path
    prefix: str
    overwrite: bool = field(default_factory=fcgradients_default("overwrite"))


@dataclass(frozen=True)
class ModuleConfig:
    """Complete functional-connectivity gradient configuration."""

    inputs: InputsConfig
    output: OutputConfig
    backend: str = field(default_factory=fcgradients_default("backend"))
    workbench: WorkbenchConfig = field(default_factory=WorkbenchConfig)
    wb_command: str = "wb_command"


def scientific_settings(cfg: ModuleConfig) -> dict[str, object]:
    """Return the backend settings that define the gradient map."""

    settings: dict[str, object] = {"backend": cfg.backend}
    if cfg.backend == "workbench_correlation_gradient":
        workbench = asdict(cfg.workbench)
        workbench.pop("memory_limit_gb")
        if not cfg.workbench.double_correlation:
            workbench.pop("fisher_z_first")
        settings["workbench"] = workbench
    if cfg.inputs.domain == "volume":
        settings["mask_threshold"] = cfg.inputs.mask_threshold
    return settings


def validate_config(cfg: ModuleConfig) -> None:
    """Reject incomplete geometry and invalid backend settings."""

    if cfg.inputs.domain not in {"surface", "volume"}:
        raise ValueError("inputs.domain must be 'surface' or 'volume'")
    if cfg.backend not in BACKENDS:
        raise ValueError(f"Unsupported functional-connectivity gradient backend: {cfg.backend}")
    if cfg.inputs.domain == "surface" and len(cfg.inputs.surfaces) != 2:
        raise ValueError("Surface gradients require left and right midthickness surfaces")
    if cfg.inputs.domain == "volume" and cfg.inputs.mask is None:
        raise ValueError("Volume gradients require a gray-matter mask")
    validate_parameters(
        "fcgradients",
        {
            "backend": cfg.backend,
            "mask_threshold": cfg.inputs.mask_threshold,
            "workbench": asdict(cfg.workbench),
        },
    )
