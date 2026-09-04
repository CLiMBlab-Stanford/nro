from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from nro.configuration.store import main_configuration_factory
from nro.engine.paths import optional_path
from nro.engine.targets import DEFAULT_SMOOTHING_MM


networks_default = partial(main_configuration_factory, "networks")


@dataclass(frozen=True)
class InputsConfig:
    microparcels: Path
    connectivity: Path
    domain: str = "surface"
    space: str = "fsnative"
    smoothing_mm: int = DEFAULT_SMOOTHING_MM
    source_surfaces: tuple[Path, ...] = ()
    scene_surfaces: tuple[Path, ...] = ()
    label_volume: Path | None = None
    anatomical_manifest: Path | None = None
    anatomical_reference: Path | None = None
    mni_to_t1_transform: Path | None = None


@dataclass(frozen=True)
class ConnectivityConfig:
    transform: str = field(default_factory=networks_default("connectivity", "transform"))
    minimum_weight: float = field(
        default_factory=networks_default("connectivity", "minimum_weight")
    )
    percentile_cutoff: float | None = field(
        default_factory=networks_default("connectivity", "percentile_cutoff")
    )
    write_matrix: bool = field(default_factory=networks_default("connectivity", "write_matrix"))


@dataclass(frozen=True)
class OslomConfig:
    executable: Path | None = field(
        default_factory=networks_default(
            "oslom", "executable", converter=optional_path
        )
    )
    initialization: str = field(default_factory=networks_default("oslom", "initialization"))
    initial_partition: Path | None = field(
        default_factory=networks_default(
            "oslom", "initial_partition", converter=optional_path
        )
    )
    leiden_resolution: float = field(default_factory=networks_default("oslom", "leiden_resolution"))
    leiden_iterations: int = field(default_factory=networks_default("oslom", "leiden_iterations"))
    leiden_seed: int | None = field(default_factory=networks_default("oslom", "leiden_seed"))
    weighted: bool = field(default_factory=networks_default("oslom", "weighted"))
    directed: bool = field(default_factory=networks_default("oslom", "directed"))
    significance: float = field(default_factory=networks_default("oslom", "significance"))
    repetitions: int = field(default_factory=networks_default("oslom", "repetitions"))
    internal_runs: int = field(default_factory=networks_default("oslom", "internal_runs"))
    hierarchical_runs: int = field(default_factory=networks_default("oslom", "hierarchical_runs"))
    extra_args: tuple[str, ...] = field(
        default_factory=networks_default("oslom", "extra_args", converter=tuple)
    )
    timeout_seconds: int | None = field(default_factory=networks_default("oslom", "timeout_seconds"))


@dataclass(frozen=True)
class ConsensusConfig:
    assignment_threshold: float = field(
        default_factory=networks_default("consensus", "assignment_threshold")
    )
    homeless_threshold: float = field(
        default_factory=networks_default("consensus", "homeless_threshold")
    )
    minimum_match_jaccard: float = field(
        default_factory=networks_default("consensus", "minimum_match_jaccard")
    )


@dataclass(frozen=True)
class LabelingConfig:
    enabled: bool = field(default_factory=networks_default("labeling", "enabled"))
    candidates_per_reference: int = field(
        default_factory=networks_default("labeling", "candidates_per_reference")
    )


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    work_directory: Path
    prefix: str
    overwrite: bool = field(
        default_factory=networks_default("overwrite")
    )


@dataclass(frozen=True)
class ModuleConfig:
    inputs: InputsConfig
    output: OutputConfig
    oslom: OslomConfig
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)


def validate_config(cfg: ModuleConfig) -> None:
    required_inputs = [cfg.inputs.microparcels, cfg.inputs.connectivity]
    required_inputs.extend(cfg.inputs.source_surfaces)
    required_inputs.extend(cfg.inputs.scene_surfaces)
    if cfg.inputs.label_volume is not None:
        required_inputs.append(cfg.inputs.label_volume)
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError("Missing networks input(s): " + ", ".join(missing_inputs))
    if cfg.inputs.domain not in {"surface", "volume"}:
        raise ValueError("Networks input domain must be surface or volume")
    if cfg.inputs.smoothing_mm < 0:
        raise ValueError("Networks input smoothing_mm must be nonnegative")
    if cfg.inputs.domain == "surface" and len(cfg.inputs.source_surfaces) not in {1, 2}:
        raise ValueError("Networks requires one surface or an ordered left/right pair")
    if cfg.inputs.domain == "surface" and len(cfg.inputs.scene_surfaces) != 8:
        raise ValueError("Surface networks require eight packaged scene surfaces")
    if cfg.inputs.domain == "volume" and cfg.inputs.source_surfaces:
        raise ValueError("Volumetric networks do not accept source surfaces")
    if cfg.inputs.domain == "volume" and cfg.inputs.scene_surfaces:
        raise ValueError("Volumetric networks do not accept scene surfaces")
    if cfg.connectivity.transform not in {"clip_positive", "absolute", "square"}:
        raise ValueError(f"Unsupported connectivity transform: {cfg.connectivity.transform}")
    if cfg.connectivity.percentile_cutoff is not None and not 0 <= cfg.connectivity.percentile_cutoff <= 100:
        raise ValueError("connectivity.percentile_cutoff must lie in [0, 100]")
    if cfg.oslom.repetitions < 1 or cfg.oslom.internal_runs < 1:
        raise ValueError("OSLOM run counts must be positive")
    if cfg.oslom.initialization not in {"none", "leiden", "file"}:
        raise ValueError("oslom.initialization must be none, leiden, or file")
    if cfg.oslom.initialization == "file" and cfg.oslom.initial_partition is None:
        raise ValueError("oslom.initial_partition is required when oslom.initialization is file")
    if cfg.oslom.leiden_resolution <= 0 or cfg.oslom.leiden_iterations < 1:
        raise ValueError("Leiden resolution and iteration count must be positive")
    if not 0 < cfg.oslom.significance < 1:
        raise ValueError("oslom.significance must lie in (0, 1)")
    if cfg.oslom.directed:
        raise ValueError("Directed graphs are not supported")
    for name, value in (
        ("assignment_threshold", cfg.consensus.assignment_threshold),
        ("homeless_threshold", cfg.consensus.homeless_threshold),
        ("minimum_match_jaccard", cfg.consensus.minimum_match_jaccard),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"consensus.{name} must lie in [0, 1]")
    if cfg.labeling.candidates_per_reference < 1:
        raise ValueError("labeling.candidates_per_reference must be positive")
    if cfg.labeling.enabled and cfg.inputs.space in {"T1w", "fsnative"}:
        if (
            cfg.inputs.anatomical_manifest is None
            or cfg.inputs.anatomical_reference is None
            or cfg.inputs.mni_to_t1_transform is None
        ):
            raise ValueError(
                f"Heuristic labeling in space-{cfg.inputs.space} requires an anatomical "
                "reference and MNI-to-T1w transform"
            )
        missing_anatomical = [
            str(path)
            for path in (
                cfg.inputs.anatomical_manifest,
                cfg.inputs.anatomical_reference,
                cfg.inputs.mni_to_t1_transform,
            )
            if path is not None and not path.is_file()
        ]
        if missing_anatomical:
            raise FileNotFoundError(
                "Missing anatomical labeling input(s): " + ", ".join(missing_anatomical)
            )
