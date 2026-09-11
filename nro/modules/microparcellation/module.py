"""Construct the complete microparcellation runner graph."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from functools import partial
from itertools import count
from pathlib import Path

import numpy as np
import yaml

from nro.configuration.schema import scientific_values
from nro.engine.bids import parse_bids_entities
from nro.engine.cifti import load_dlabel
from nro.engine.cleaned_timeseries import (
    CleanedRunInclusionPolicy,
    CleanedRunMetadata,
    cleaned_run_exclusions,
    load_cleaned_run_metadata,
    load_retained_frame_mask,
)
from nro.engine.connectivity import connectivity_run_weights
from nro.engine.images import (
    gifti_vertex_count,
    load_surface_timeseries,
    sidecar_json_path,
)
from nro.engine.io import (
    atomic_save_npy,
    atomic_save_npz,
    atomic_write_text,
    json_path_default,
    manifest_value,
    temporary_sibling,
)
from nro.engine.surface_geometry import load_surface_mask, load_surfaces, mesh_edges
from nro.orchestration.runner import Runner, write_completion_breadcrumb
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import selected_configuration_fingerprint

from .cifti import (
    surface_parcel_axis,
    volume_parcel_axis,
    write_dlabel,
    write_pconn,
    write_volume_dlabel,
)
from .coarsen import loukas_variation_edges
from .config import ModuleConfig, validate_config
from .contract import (
    microparcellation_output_contract,
    validate_microparcellation_manifest,
    validate_microparcellation_quality,
)
from .paths import output_paths
from .quality import spatial_null_partitions
from .statistics import (
    local_edge_correlations,
    make_parcel_mean_loader,
    parcel_correlations,
)
from .volume import (
    load_volume_functional,
    load_volume_labels,
    load_volume_space,
    write_volume_labels,
)

LOG = logging.getLogger(__name__)


def _flatten_functional(functional: tuple[tuple[Path, ...], ...]) -> tuple[Path, ...]:
    return tuple(path for run in functional for path in run)


def _exponential_loukas_weights(correlations: np.ndarray, temperature: float) -> np.ndarray:
    """Map correlations monotonically to strictly positive Loukas weights."""
    correlations = np.asarray(correlations, dtype=np.float32)
    if not np.all(np.isfinite(correlations)):
        raise ValueError("Spatial-edge correlations contain non-finite values")
    scaled = (correlations - correlations.max()) / np.float32(temperature)
    scaled = np.maximum(scaled, np.log(np.finfo(np.float32).tiny))
    return np.exp(scaled).astype(np.float32)


def _coarsening_targets(n_nodes: int, target: int, iterations: int) -> tuple[int, ...]:
    """Return a target-relative sequence of halving passes.

    For ``m`` requested iterations, targets are ``2**(m-1) * K`` through
    ``K``.  Targets at or above the current native granularity are omitted
    because they would not perform any coarsening.
    """
    if iterations < 1:
        raise ValueError("Coarsening iterations must be positive")
    target = min(target, n_nodes)
    if target == n_nodes:
        return ()
    candidates = (target * (1 << exponent) for exponent in range(iterations - 1, -1, -1))
    steps: list[int] = []
    current = n_nodes
    for step_target in candidates:
        if step_target >= current:
            continue
        steps.append(step_target)
        current = step_target
    return tuple(steps)


def _region_edges(base_edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Collapse original spatial edges into a unique parcel-adjacency graph."""
    pairs = labels[base_edges]
    keep = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] != pairs[:, 1])
    pairs = np.sort(pairs[keep], axis=1)
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(pairs, axis=0).astype(np.int64, copy=False)


def build_module(
    cfg: ModuleConfig,
    runner: Runner,
    *,
    completion_boundary: bool = True,
) -> dict[str, Path | tuple[Path, ...]]:
    """Build the complete microparcellation DAG without executing it."""
    validate_config(cfg)
    out = cfg.output.directory
    work = cfg.output.work_directory
    force = bool(cfg.output.overwrite)

    paths = output_paths(out, cfg.output.prefix)
    manifest_path = paths["manifest"]
    dlabel_path = paths["microparcels"]
    pconn_path = paths["connectivity"]
    quality_path = paths["quality"]
    label_volume_path = paths["microparcels_volume"]
    outputs: dict[str, Path | tuple[Path, ...]] = {
        "microparcels": dlabel_path,
        "connectivity": pconn_path,
        "quality": quality_path,
    }
    if cfg.inputs.domain == "surface":
        label_outputs = (dlabel_path,)
    else:
        outputs["microparcels_volume"] = label_volume_path
        label_outputs = (label_volume_path, dlabel_path)

    fixed_public_outputs = tuple(
        path
        for value in outputs.values()
        for path in ((value,) if isinstance(value, Path) else value)
    )
    functional_inputs = _flatten_functional(cfg.inputs.functional)
    functional_sidecars = tuple(
        dict.fromkeys(sidecar_json_path(path) for path in functional_inputs)
    )
    temporal_masks = tuple(cfg.inputs.temporal_masks)
    if cfg.inputs.mask is None:
        mask_inputs: tuple[Path, ...] = ()
    elif isinstance(cfg.inputs.mask, Path):
        mask_inputs = (cfg.inputs.mask,)
    else:
        mask_inputs = tuple(cfg.inputs.mask)
    source_inputs = tuple(
        dict.fromkeys(
            functional_inputs
            + functional_sidecars
            + temporal_masks
            + tuple(cfg.inputs.surface)
            + mask_inputs
        )
    )

    inclusion_policy = CleanedRunInclusionPolicy(
        minimum_retained_frames=cfg.connectivity.minimum_retained_frames,
        minimum_retained_fraction=cfg.connectivity.minimum_retained_fraction,
        minimum_residual_design_dof=cfg.connectivity.minimum_residual_design_dof,
        minimum_participation_effective_rank=(
            cfg.connectivity.minimum_participation_effective_rank
        ),
        maximum_dominant_temporal_variance_fraction=(
            cfg.connectivity.maximum_dominant_temporal_variance_fraction
        ),
    )
    cleaning_metadata_cache: dict[tuple[Path, ...], CleanedRunMetadata] = {}
    temporal_mask_by_run = dict(zip(cfg.inputs.functional, temporal_masks))

    def run_metadata(run: tuple[Path, ...]) -> CleanedRunMetadata:
        if run not in cleaning_metadata_cache:
            metadata = load_cleaned_run_metadata(run)
            expected_mask = temporal_mask_by_run[run]
            if metadata.temporal_mask_file.resolve() != expected_mask.resolve():
                raise ValueError(
                    "Cleaned sidecar names an unexpected temporal mask: "
                    f"{metadata.temporal_mask_file} != {expected_mask}"
                )
            cleaning_metadata_cache[run] = metadata
        return cleaning_metadata_cache[run]

    def cleaning_eligibility() -> tuple[np.ndarray, dict[int, tuple[dict[str, object], ...]]]:
        included: list[int] = []
        excluded: dict[int, tuple[dict[str, object], ...]] = {}
        for index, run in enumerate(cfg.inputs.functional):
            reasons = cleaned_run_exclusions(run_metadata(run), inclusion_policy)
            if not reasons:
                included.append(index)
            else:
                excluded[index] = reasons
        retained_total = sum(
            run_metadata(cfg.inputs.functional[index]).retained_frames for index in included
        )
        if len(included) < cfg.connectivity.minimum_usable_runs:
            raise ValueError(
                "Too few cleaned runs satisfy the sidecar inclusion policy: "
                f"{len(included)} < {cfg.connectivity.minimum_usable_runs}"
            )
        if retained_total < cfg.connectivity.minimum_aggregate_retained_frames:
            raise ValueError(
                "Eligible cleaned runs have too few aggregate retained frames: "
                f"{retained_total} < "
                f"{cfg.connectivity.minimum_aggregate_retained_frames}"
            )
        return np.asarray(included, dtype=np.int64), excluded

    def weights_for_indices(indices: np.ndarray):
        return connectivity_run_weights(
            [run_metadata(cfg.inputs.functional[int(index)]) for index in indices],
            weighting=cfg.connectivity.weighting,
            global_signal_regression=cfg.connectivity.global_signal_regression,
        )

    def retained_run_loader(run: tuple[Path, ...], *, base_loader=None) -> np.ndarray:
        loader = load_surface_timeseries if base_loader is None else base_loader
        data = loader(run)
        metadata = run_metadata(run)
        if len(data) != metadata.total_frames:
            raise ValueError(
                "Functional frame count disagrees with clean metadata: "
                f"{len(data)} != {metadata.total_frames} ({run})"
            )
        return data[load_retained_frame_mask(metadata)]

    # Source geometry is BIDS input state, so it may determine topology during
    # construction. No derivative or step result is consulted here.
    spatial_cache: dict[str, object] = {}
    if cfg.inputs.domain == "surface":
        vertex_counts = tuple(gifti_vertex_count(path) for path in cfg.inputs.surface)
        n_vertices = sum(vertex_counts)
        surface_mask = cfg.inputs.mask if isinstance(cfg.inputs.mask, tuple) else None
        mask = load_surface_mask(surface_mask, vertex_counts)
        volume_space = None
    else:
        assert isinstance(cfg.inputs.mask, Path)
        volume_space = load_volume_space(
            cfg.inputs.functional[0][0],
            cfg.inputs.mask,
            threshold=cfg.inputs.mask_threshold,
            connectivity=cfg.inputs.volume_connectivity,
        )
        n_vertices = len(volume_space.voxel_indices)
        vertex_counts = ()
        mask = np.ones(n_vertices, dtype=bool)
        spatial_cache["volume_space"] = volume_space

    def spatial_data():
        if "edges" in spatial_cache:
            return (
                spatial_cache["edges"],
                spatial_cache.get("volume_space"),
                spatial_cache.get("load_run"),
            )
        if cfg.inputs.domain == "surface":
            _coords, faces, loaded_counts = load_surfaces(cfg.inputs.surface)
            if loaded_counts != vertex_counts:
                raise ValueError("Surface geometry changed during module execution.")
            loaded_edges = mesh_edges(faces)
            loaded_edges = loaded_edges[loaded_edges[:, 0] != loaded_edges[:, 1]]
            loaded_edges = loaded_edges[mask[loaded_edges].all(axis=1)]
            loaded_volume_space = None
            loaded_run = None
        else:
            loaded_volume_space = spatial_cache["volume_space"]
            loaded_edges = loaded_volume_space.edges
            loaded_run = partial(load_volume_functional, space=loaded_volume_space)
        spatial_cache["edges"] = loaded_edges
        spatial_cache["load_run"] = loaded_run
        return loaded_edges, loaded_volume_space, loaded_run

    initialization_breadcrumb = work / "initialized.complete"

    def initialize() -> None:
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        write_completion_breadcrumb(
            initialization_breadcrumb,
            "Microparcellation output initialized\n",
        )

    runner.add_step(
        Step.python(
            name="Initialize Microparcellation Outputs",
            outputs=(initialization_breadcrumb,),
            action=initialize,
        )
    )

    config_snapshot = work / "module-config.json"
    config_payload = {
        "inputs": asdict(cfg.inputs),
        "coarsening": asdict(cfg.coarsening),
        "connectivity": asdict(cfg.connectivity),
        "quality": asdict(cfg.quality),
        "output_prefix": cfg.output.prefix,
    }
    config_payload = scientific_values("microparcellation", config_payload)
    config_text = (
        json.dumps(config_payload, default=json_path_default, indent=2, sort_keys=True) + "\n"
    )

    def validate_config_snapshot() -> tuple[bool, str]:
        try:
            recorded = json.loads(config_snapshot.read_text(encoding="utf-8"))
            matches = scientific_values("microparcellation", recorded) == json.loads(config_text)
        except (OSError, ValueError, TypeError):
            matches = False
        return matches, (
            "Recorded microparcellation configuration matches the requested workflow."
            if matches
            else "Recorded microparcellation configuration differs from the requested workflow."
        )

    runner.add_step(
        Step.python(
            name="Resolve Microparcellation Configuration",
            outputs=(config_snapshot,),
            inputs=(initialization_breadcrumb,),
            force=force,
            action=lambda: atomic_write_text(config_snapshot, config_text),
            validate=validate_config_snapshot,
        )
    )

    label_format_inputs: tuple[Path, ...] = ()
    if cfg.inputs.domain == "volume":
        volume_cifti_format = work / "volume_cifti_format.txt"
        format_text = "plumb-volume-cifti-v1\n"

        def validate_volume_format() -> tuple[bool, str]:
            try:
                matches = volume_cifti_format.read_text(encoding="utf-8") == format_text
            except OSError:
                matches = False
            return matches, (
                "Volumetric CIFTI format is current."
                if matches
                else "Volumetric CIFTI format changed."
            )

        runner.add_step(
            Step.python(
                name="Resolve Volumetric CIFTI Format",
                outputs=(volume_cifti_format,),
                inputs=(config_snapshot,),
                force=force,
                action=lambda: atomic_write_text(volume_cifti_format, format_text),
                validate=validate_volume_format,
            )
        )
        label_format_inputs = (volume_cifti_format,)

    target = min(cfg.coarsening.target_vertices, int(mask.sum()))
    step_targets = _coarsening_targets(int(mask.sum()), target, cfg.coarsening.iterations)
    previous_labels_checkpoint: Path | None = None
    correlation_checkpoints: list[Path] = []
    label_checkpoints: list[Path] = []
    run_to_index = {run: index for index, run in enumerate(cfg.inputs.functional)}

    def input_labels(previous: Path | None) -> np.ndarray:
        if previous is not None:
            return np.asarray(np.load(previous, allow_pickle=False), dtype=np.int64)
        labels = np.full(n_vertices, -1, dtype=np.int64)
        labels[mask] = np.arange(int(mask.sum()), dtype=np.int64)
        return labels

    for pass_index, step_target in enumerate(step_targets, start=1):
        correlation_checkpoint = work / f"pass-{pass_index:02d}_edge_correlations.npz"
        labels_checkpoint = work / f"pass-{pass_index:02d}_labels.npy"
        correlation_checkpoints.append(correlation_checkpoint)
        label_checkpoints.append(labels_checkpoint)
        correlation_inputs = (
            functional_inputs
            + functional_sidecars
            + temporal_masks
            + tuple(cfg.inputs.surface)
            + mask_inputs
            + (config_snapshot,)
            + ((previous_labels_checkpoint,) if previous_labels_checkpoint is not None else ())
        )
        progress_label = f"Streaming coarsening pass {pass_index}/{len(step_targets)}"

        def calculate_correlations(
            *,
            previous: Path | None = previous_labels_checkpoint,
            checkpoint: Path = correlation_checkpoint,
            target_regions: int = step_target,
            progress: str = progress_label,
        ) -> None:
            edges, _volume_space, load_run = spatial_data()
            labels = input_labels(previous)
            current_count = int(labels[mask].max()) + 1
            current_edges = _region_edges(edges, labels)
            retained_loader = partial(retained_run_loader, base_loader=load_run)
            step_loader, node_masses = make_parcel_mean_loader(
                labels, mask, load_run=retained_loader
            )
            if correlation_checkpoints and checkpoint != correlation_checkpoints[0]:
                with np.load(correlation_checkpoints[0], allow_pickle=False) as first:
                    included = np.asarray(first["included_indices"], dtype=np.int64)
                step_files = tuple(cfg.inputs.functional[int(index)] for index in included)
            else:
                eligible, exclusions = cleaning_eligibility()
                step_files = tuple(cfg.inputs.functional[int(index)] for index in eligible)
                included = eligible
            run_weights = weights_for_indices(included)
            LOG.info(
                "%s: correlations for %d edges among %d regions (target %d)",
                progress,
                len(current_edges),
                current_count,
                target_regions,
            )
            result = local_edge_correlations(
                step_files,
                current_edges,
                current_count,
                cfg.connectivity.temporal_block_size,
                mask=np.ones(current_count, dtype=bool),
                global_signal_regression=cfg.connectivity.global_signal_regression,
                run_weights=np.asarray(
                    [record.normalized_weight for record in run_weights], dtype=np.float64
                ),
                load_run=step_loader,
                node_weights=node_masses,
                progress_label=progress,
            )
            global_included = np.asarray(
                [run_to_index[run] for run in result.included_runs],
                dtype=np.int64,
            )
            if checkpoint == correlation_checkpoints[0]:
                excluded_indices = np.asarray(sorted(exclusions), dtype=np.int64)
                exclusion_records = np.asarray(
                    [json.dumps(exclusions[index], sort_keys=True) for index in excluded_indices],
                    dtype=np.str_,
                )
            else:
                excluded_indices = np.empty(0, dtype=np.int64)
                exclusion_records = np.empty(0, dtype=np.str_)
            atomic_save_npz(
                checkpoint,
                compressed=True,
                correlations=result.correlations.astype(np.float32, copy=False),
                included_indices=global_included,
                excluded_indices=excluded_indices,
                exclusion_records=exclusion_records,
            )

        runner.add_step(
            Step.python(
                name=(f"Microparcellation Pass {pass_index}: Stream Edge Correlations"),
                outputs=(correlation_checkpoint,),
                inputs=correlation_inputs,
                force=force,
                action=calculate_correlations,
            )
        )

        def coarsen_step(
            *,
            previous: Path | None = previous_labels_checkpoint,
            correlations_path: Path = correlation_checkpoint,
            checkpoint: Path = labels_checkpoint,
            target_regions: int = step_target,
        ) -> None:
            edges, _volume_space, _load_run = spatial_data()
            labels = input_labels(previous)
            current_count = int(labels[mask].max()) + 1
            current_edges = _region_edges(edges, labels)
            with np.load(correlations_path, allow_pickle=False) as values:
                correlations = np.asarray(values["correlations"], dtype=np.float32)
            if correlations.shape != (len(current_edges),):
                raise RuntimeError(f"Invalid edge-correlation checkpoint: {correlations_path}")
            weights = _exponential_loukas_weights(
                correlations, cfg.coarsening.exponential_temperature
            )
            step_labels = loukas_variation_edges(
                current_count,
                current_edges,
                weights,
                np.ones(current_count, dtype=bool),
                target_regions,
                k=cfg.coarsening.eigenvectors,
                max_levels=cfg.coarsening.max_levels,
                eigensolver_tolerance=cfg.coarsening.eigensolver_tolerance,
            )
            updated = labels.copy()
            updated[mask] = step_labels[labels[mask]]
            atomic_save_npy(checkpoint, updated)

        runner.add_step(
            Step.python(
                name=f"Microparcellation Pass {pass_index}: Loukas Coarsening",
                outputs=(labels_checkpoint,),
                inputs=(
                    correlation_checkpoint,
                    config_snapshot,
                    *(
                        (previous_labels_checkpoint,)
                        if previous_labels_checkpoint is not None
                        else ()
                    ),
                ),
                force=force,
                action=coarsen_step,
            )
        )
        previous_labels_checkpoint = labels_checkpoint

    def write_labels() -> None:
        _edges, active_volume_space, _load_run = spatial_data()
        labels = input_labels(previous_labels_checkpoint)
        expected_regions = step_targets[-1] if step_targets else int(mask.sum())
        if labels.shape != (n_vertices,) or int(labels[mask].max()) + 1 != expected_regions:
            raise RuntimeError("Final coarsening checkpoint has invalid labels.")
        temporary_dlabel = temporary_sibling(dlabel_path)
        if cfg.inputs.domain == "surface":
            write_dlabel(temporary_dlabel, labels, vertex_counts, cfg.inputs.surface)
            temporary_dlabel.replace(dlabel_path)
        else:
            assert active_volume_space is not None
            temporary_volume = temporary_sibling(label_volume_path)
            write_volume_labels(temporary_volume, labels, active_volume_space)
            write_volume_dlabel(
                temporary_dlabel,
                labels,
                active_volume_space.mask,
                active_volume_space.affine,
            )
            temporary_volume.replace(label_volume_path)
            temporary_dlabel.replace(dlabel_path)

    runner.add_step(
        Step.python(
            name="Write Microparcel Labels",
            outputs=label_outputs,
            inputs=(
                source_inputs
                + label_format_inputs
                + (config_snapshot,)
                + ((previous_labels_checkpoint,) if previous_labels_checkpoint is not None else ())
            ),
            force=force,
            action=write_labels,
        )
    )

    def load_final_labels() -> np.ndarray:
        if cfg.inputs.domain == "surface":
            labels, _ = load_dlabel(dlabel_path)
            return labels
        return load_volume_labels(label_volume_path)

    def read_run_eligibility() -> tuple[
        np.ndarray,
        dict[int, tuple[dict[str, object], ...]],
    ]:
        if not correlation_checkpoints:
            included, exclusions = cleaning_eligibility()
            return included, exclusions
        with np.load(correlation_checkpoints[0], allow_pickle=False) as checkpoint:
            included = np.asarray(checkpoint["included_indices"], dtype=np.int64)
            excluded_indices = np.asarray(checkpoint["excluded_indices"], dtype=np.int64)
            exclusion_records = np.asarray(checkpoint["exclusion_records"], dtype=np.str_)
        return included, {
            int(index): tuple(json.loads(str(records)))
            for index, records in zip(excluded_indices, exclusion_records)
        }

    def write_connectivity() -> None:
        edges, active_volume_space, load_run = spatial_data()
        labels = load_final_labels()
        included_indices, _ = read_run_eligibility()
        active_stage_files = tuple(cfg.inputs.functional[int(index)] for index in included_indices)
        run_weights = weights_for_indices(included_indices)
        nulls = spatial_null_partitions(
            labels,
            mask,
            edges,
            count=cfg.quality.null_parcellations,
            seed=cfg.quality.random_seed,
            candidate_attempts=cfg.quality.region_growing_attempts,
        )
        retained_loader = partial(retained_run_loader, base_loader=load_run)
        result = parcel_correlations(
            active_stage_files,
            labels,
            mask,
            cfg.connectivity.temporal_block_size,
            global_signal_regression=cfg.connectivity.global_signal_regression,
            run_weights=np.asarray(
                [record.normalized_weight for record in run_weights], dtype=np.float64
            ),
            effective_dof=np.asarray(
                [record.effective_dof for record in run_weights], dtype=np.int64
            ),
            connectome_power_iterations=cfg.quality.connectome_power_iterations,
            split_half_block_frames=cfg.quality.split_half_block_frames,
            load_run=retained_loader,
            null_partitions=tuple(null.labels for null in nulls),
        )
        adjacency = result.correlations
        np.fill_diagonal(adjacency, 0.0)
        if cfg.inputs.domain == "surface":
            parcel_axis = surface_parcel_axis(labels, vertex_counts, cfg.inputs.surface)
        else:
            assert active_volume_space is not None
            parcel_axis = volume_parcel_axis(
                labels, active_volume_space.mask, active_volume_space.affine
            )
        temporary = temporary_sibling(pconn_path)
        write_pconn(temporary, adjacency, parcel_axis)
        temporary.replace(pconn_path)
        null_scores = np.asarray(result.null_variance_preserved, dtype=np.float64)
        null_records = [
            {
                "variance_preserved": score,
                "residual_sum_squares": residual,
                "mean_absolute_size_error": null.mean_absolute_size_error,
                "maximum_absolute_size_error": null.maximum_absolute_size_error,
                "exactly_matched_fraction": null.exactly_matched_fraction,
            }
            for score, residual, null in zip(
                result.null_variance_preserved,
                result.null_residual_sum_squares,
                nulls,
            )
        ]
        run_contributions = []
        for paths_for_run, record in zip(active_stage_files, result.run_contributions):
            run_contributions.append(
                {
                    **record,
                    "files": [str(path) for path in paths_for_run],
                    "entities": parse_bids_entities(paths_for_run[0].name),
                }
            )
        quality = {
            "metric": (
                "fraction of standardized temporal variance preserved by parcel-mean reconstruction"
            ),
            "source_nodes": int(mask.sum()),
            "microparcels": int(labels[mask].max()) + 1,
            "included_runs": len(active_stage_files),
            "runwise_standardization": True,
            "global_signal_regression": cfg.connectivity.global_signal_regression,
            "weighting": cfg.connectivity.weighting,
            "variance_preserved": result.variance_preserved,
            "variance_lost": 1.0 - result.variance_preserved,
            "residual_sum_squares": result.residual_sum_squares,
            "total_sum_squares": result.total_sum_squares,
            "parcel_support": {
                "minimum_supporting_runs": int(result.parcel_supporting_runs.min()),
                "mean_supporting_runs": float(result.parcel_supporting_runs.mean()),
                "mean_effective_runs": float(result.parcel_effective_runs.mean()),
                "minimum_effective_runs": float(result.parcel_effective_runs.min()),
            },
            "run_contributions": run_contributions,
            "split_half": result.split_half,
            "connectome": result.connectome,
            "null_baseline": {
                "method": ("quota-matched random region growing on the source spatial graph"),
                "seed": cfg.quality.random_seed,
                "parcellations": null_records,
                "variance_preserved_mean": float(null_scores.mean()),
                "variance_preserved_standard_deviation": float(null_scores.std()),
                "variance_preserved_minimum": float(null_scores.min()),
                "variance_preserved_maximum": float(null_scores.max()),
                "fitted_minus_null_mean": float(result.variance_preserved - null_scores.mean()),
            },
        }
        validate_microparcellation_quality(quality)
        atomic_write_text(quality_path, json.dumps(quality, indent=2) + "\n")

    runner.add_step(
        Step.python(
            name="Stream Final Microparcel Connectivity",
            outputs=(pconn_path, quality_path),
            inputs=functional_inputs + label_outputs + tuple(correlation_checkpoints),
            force=force,
            action=write_connectivity,
        )
    )

    manifest_inputs = fixed_public_outputs + source_inputs

    def coarsening_records() -> list[dict[str, int]]:
        edges, _volume_space, _load_run = spatial_data()
        records: list[dict[str, int]] = []
        previous = input_labels(None)
        previous_count = int(mask.sum())
        for index, step_target in enumerate(step_targets, start=1):
            records.append(
                {
                    "pass": index,
                    "input_regions": previous_count,
                    "spatial_edges": len(_region_edges(edges, previous)),
                    "target_regions": step_target,
                }
            )
            previous = np.asarray(
                np.load(label_checkpoints[index - 1], allow_pickle=False),
                dtype=np.int64,
            )
            previous_count = step_target
        return records

    def write_manifest() -> None:
        current_outputs = {*fixed_public_outputs, manifest_path}
        for path in out.iterdir():
            if (
                path.name.startswith(f"{cfg.output.prefix}_")
                and path not in current_outputs
                and (path.is_file() or path.is_symlink())
            ):
                path.unlink()
        labels = load_final_labels()
        included_indices, excluded_runs = read_run_eligibility()
        manifest = {
            "domain": cfg.inputs.domain,
            "space": cfg.inputs.space,
            "smoothing_fwhm_mm": cfg.inputs.smoothing_mm,
            "n_spatial_nodes": n_vertices,
            "n_surface_vertices": (n_vertices if cfg.inputs.domain == "surface" else None),
            "hemisphere_vertex_counts": list(vertex_counts),
            "n_active_nodes": int(mask.sum()),
            "n_active_vertices": (int(mask.sum()) if cfg.inputs.domain == "surface" else None),
            "n_gray_matter_voxels": (n_vertices if cfg.inputs.domain == "volume" else None),
            "n_microparcels": int(labels[mask].max()) + 1,
            "coarsening_steps": coarsening_records(),
            "source_surfaces": [str(path) for path in cfg.inputs.surface],
            "source_volume_mask": (str(cfg.inputs.mask) if cfg.inputs.domain == "volume" else None),
            "volume_mask_resampled": (
                volume_space.mask_resampled if volume_space is not None else None
            ),
            "volume_connectivity": (
                cfg.inputs.volume_connectivity if cfg.inputs.domain == "volume" else None
            ),
            "functional_runs": {
                "inclusion_policy": asdict(inclusion_policy),
                "minimum_usable_runs": cfg.connectivity.minimum_usable_runs,
                "minimum_aggregate_retained_frames": (
                    cfg.connectivity.minimum_aggregate_retained_frames
                ),
                "aggregate_retained_frames": sum(
                    run_metadata(cfg.inputs.functional[int(index)]).retained_frames
                    for index in included_indices
                ),
                "included": [
                    [str(path) for path in cfg.inputs.functional[int(index)]]
                    for index in included_indices
                ],
                "skipped": [
                    {
                        "files": [str(path) for path in cfg.inputs.functional[index]],
                        "reasons": list(reasons),
                    }
                    for index, reasons in sorted(excluded_runs.items())
                ],
            },
            "outputs": {name: manifest_value(path) for name, path in outputs.items()},
            "connectivity_encoding": {
                "format": "CIFTI-2 pconn",
                "dtype": "int8",
                "scale": 127.0,
                "range": [-1.0, 1.0],
                "diagonal": 0.0,
            },
            "quality": json.loads(quality_path.read_text(encoding="utf-8")),
            "config": asdict(cfg),
            "configuration_fingerprint": selected_configuration_fingerprint(),
            "output_metadata_contract": microparcellation_output_contract(),
        }
        validate_microparcellation_manifest(manifest)
        atomic_write_text(
            manifest_path,
            yaml.safe_dump(
                json.loads(json.dumps(manifest, default=json_path_default)),
                sort_keys=False,
            ),
        )

    def validate_manifest() -> tuple[bool, str]:
        try:
            published = yaml.safe_load(manifest_path.read_text()) or {}
            validate_microparcellation_manifest(published)
            quality_metadata = json.loads(quality_path.read_text(encoding="utf-8"))
            validate_microparcellation_quality(quality_metadata)
        except (OSError, TypeError, yaml.YAMLError):
            return False, f"Microparcellation manifest is unreadable: {manifest_path}"
        except (ValueError, json.JSONDecodeError):
            return False, "Microparcellation metadata violates its artifact contract."
        expected_inventory = {name: manifest_value(value) for name, value in outputs.items()}
        if published.get("outputs") != expected_inventory:
            return False, "Published microparcellation inventory is stale."
        current = json.loads(json.dumps(asdict(cfg), default=json_path_default))
        if scientific_values("microparcellation", published.get("config", {})) != scientific_values(
            "microparcellation", current
        ):
            return False, "Published microparcellation configuration is stale."
        fingerprint = selected_configuration_fingerprint()
        if fingerprint is not None and published.get("configuration_fingerprint") != fingerprint:
            return False, "Published microparcellation configuration fingerprint is stale."
        missing = [
            str(path)
            for path in fixed_public_outputs
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "Published microparcellation output is missing: " + ", ".join(missing)
        return True, "Microparcellation publication is complete and current."

    runner.add_step(
        Step.python(
            name="Write Microparcellation Manifest",
            outputs=(manifest_path,),
            inputs=manifest_inputs,
            force=force,
            action=write_manifest,
            validate=validate_manifest,
            completion_boundary=completion_boundary,
        )
    )
    outputs["manifest"] = manifest_path
    return outputs


def run(
    cfg: ModuleConfig,
    *,
    runner: Runner | None = None,
) -> dict[str, Path | tuple[Path, ...]]:
    """Construct the module graph, execute it through the shared runner, and publish outputs.

    Freshness is evaluated after graph construction. Processing and validation
    errors propagate to the caller; partial private outputs can support resumption.
    """
    runner_started = time.perf_counter()
    active_runner = runner or Runner(
        module_name="Microparcellation Module",
        container=None,
        binds=(),
        logger=LOG,
        next_step=count(1).__next__,
    )
    outputs = build_module(cfg, active_runner)
    with active_runner.run_context(started_at=runner_started):
        active_runner.execute()
    return outputs
