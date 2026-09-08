from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from nro.configuration.store import main_configuration_factory
from nro.configuration.schema import validate_parameters
from nro.engine.paths import optional_path
from nro.engine.targets import DEFAULT_SMOOTHING_MM


networks_default = partial(main_configuration_factory, "networks")


@dataclass(frozen=True)
class InputsConfig:
    """CIFTI microparcellation inputs and anatomy needed for reference-map projection."""
    microparcellation_manifest: Path
    microparcels: Path
    connectivity: Path
    domain: str = "surface"
    space: str = "fsnative"
    smoothing_mm: int = DEFAULT_SMOOTHING_MM
    source_surfaces: tuple[Path, ...] = ()
    anatomical_manifest: Path | None = None
    anatomical_reference: Path | None = None
    mni_to_t1_transform: Path | None = None


@dataclass(frozen=True)
class ConnectivityConfig:
    """Weight transformation and sparsification controls for candidate-network estimation."""
    transform: str = field(default_factory=networks_default("connectivity", "transform"))
    minimum_weight: float = field(
        default_factory=networks_default("connectivity", "minimum_weight")
    )
    percentile_cutoff: float | None = field(
        default_factory=networks_default("connectivity", "percentile_cutoff")
    )


@dataclass(frozen=True)
class IcaConfig:
    """ICA component count, randomized reduction, sign-tail normalization, and solver controls."""
    n_networks: int = field(default_factory=networks_default("ica", "n_networks"))
    random_seed: int | None = field(
        default_factory=networks_default("ica", "random_seed")
    )
    max_iterations: int = field(
        default_factory=networks_default("ica", "max_iterations")
    )
    tolerance: float = field(default_factory=networks_default("ica", "tolerance"))
    upper_quantile: float = field(
        default_factory=networks_default("ica", "upper_quantile")
    )
    svd_oversamples: int = field(
        default_factory=networks_default("ica", "svd_oversamples")
    )
    svd_power_iterations: int = field(
        default_factory=networks_default("ica", "svd_power_iterations")
    )


@dataclass(frozen=True)
class ClusteringConfig:
    """Repeated mini-batch k-means controls for binarized connectivity profiles."""
    n_networks: int = field(
        default_factory=networks_default("clustering", "n_networks")
    )
    repetitions: int = field(
        default_factory=networks_default("clustering", "repetitions")
    )
    random_seed: int | None = field(
        default_factory=networks_default("clustering", "random_seed")
    )
    n_init: int = field(default_factory=networks_default("clustering", "n_init"))
    max_iterations: int = field(
        default_factory=networks_default("clustering", "max_iterations")
    )
    batch_size: int = field(
        default_factory=networks_default("clustering", "batch_size")
    )
    max_no_improvement: int | None = field(
        default_factory=networks_default("clustering", "max_no_improvement")
    )
    reassignment_ratio: float = field(
        default_factory=networks_default("clustering", "reassignment_ratio")
    )


@dataclass(frozen=True)
class OslomConfig:
    """OSLOM executable, graph interpretation, initialization, repetitions, and timeout."""
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
    """Assignment, homelessness, and replicate-overlap matching thresholds."""
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
    """Whether to rank networks against references and how many candidates to retain."""
    enabled: bool = field(default_factory=networks_default("labeling", "enabled"))
    candidates_per_reference: int = field(
        default_factory=networks_default("labeling", "candidates_per_reference")
    )


@dataclass(frozen=True)
class OutputConfig:
    """Network publication/work directories, filename prefix, and overwrite policy."""
    directory: Path
    work_directory: Path
    prefix: str
    overwrite: bool = field(
        default_factory=networks_default("overwrite")
    )


@dataclass(frozen=True)
class ModuleConfig:
    """Complete network inputs, strategy, algorithm settings, labeling, and output policy."""
    inputs: InputsConfig
    output: OutputConfig
    parcellation_strategy: str = field(
        default_factory=networks_default("parcellation_strategy")
    )
    ica: IcaConfig = field(default_factory=IcaConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    oslom: OslomConfig = field(default_factory=OslomConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)


def validate_config(cfg: ModuleConfig) -> None:
    """Reject unsupported domains, missing input resources, and inconsistent algorithm settings."""
    required_inputs = [
        cfg.inputs.microparcellation_manifest,
        cfg.inputs.microparcels,
        cfg.inputs.connectivity,
    ]
    required_inputs.extend(cfg.inputs.source_surfaces)
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError("Missing networks input(s): " + ", ".join(missing_inputs))
    if cfg.inputs.domain not in {"surface", "volume"}:
        raise ValueError("Networks input domain must be surface or volume")
    if cfg.inputs.smoothing_mm < 0:
        raise ValueError("Networks input smoothing_mm must be nonnegative")
    if cfg.inputs.domain == "surface" and len(cfg.inputs.source_surfaces) not in {1, 2}:
        raise ValueError("Networks requires one surface or an ordered left/right pair")
    if cfg.inputs.domain == "volume" and cfg.inputs.source_surfaces:
        raise ValueError("Volumetric networks do not accept source surfaces")
    validate_parameters("networks", {
        "parcellation_strategy": cfg.parcellation_strategy,
        **{name: asdict(getattr(cfg, name)) for name in
           ("connectivity", "ica", "clustering", "oslom", "consensus", "labeling")},
    })
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
