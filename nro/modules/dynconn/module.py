"""Construct and execute dynamic-connectivity packaging steps."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from nro.configuration.schema import scientific_values
from nro.engine.cleaned_timeseries import (
    CleanedRunInclusionPolicy,
    cleaned_run_exclusions,
    load_cleaned_run_metadata,
    load_retained_frame_mask,
)
from nro.engine.connectivity import connectivity_run_weights
from nro.engine.images import (
    load_surface_timeseries,
    sidecar_json_path,
    surface_timeseries_shape,
)
from nro.engine.io import atomic_output_path, atomic_write_text, json_path_default
from nro.engine.publication import write_json_atomic
from nro.orchestration.runner import Runner, write_completion_breadcrumb
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import selected_configuration_fingerprint

from .config import ModuleConfig, validate_config
from .contract import dynconn_output_contract, validate_dynconn_manifest
from .low_rank import fit_low_rank_correlation, pseudo_timeseries_block
from .paths import output_paths

TEMPORAL_BLOCK_FRAMES = 32
SPATIAL_BLOCK_LOCATIONS = 65_536


def _flatten(groups: tuple[tuple[Path, ...], ...]) -> tuple[Path, ...]:
    return tuple(path for group in groups for path in group)


def _repetition_time(sidecar: Path) -> float:
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    value = document.get("RepetitionTime")
    if value is None or float(value) <= 0:
        raise ValueError(f"Cleaned sidecar lacks a positive RepetitionTime: {sidecar}")
    return float(value)


def _retained_blocks(
    cfg: ModuleConfig,
    included: list[dict[str, object]],
    *,
    spatial_shape: tuple[int, ...],
    reference_affine: np.ndarray | None,
):
    for record in included:
        run = cfg.inputs.functional[int(record["run_index"])]
        metadata = load_cleaned_run_metadata(run)
        retained = np.flatnonzero(load_retained_frame_mask(metadata))
        if cfg.inputs.domain == "surface":
            _, vertex_counts = surface_timeseries_shape(run)
            if vertex_counts != spatial_shape:
                raise ValueError(f"Cleaned surface geometry differs across runs: {run}")
            data = load_surface_timeseries(run)
            if data.shape != (metadata.total_frames, sum(vertex_counts)):
                raise ValueError(f"Cleaned surface dimensions disagree with metadata: {run}")
        else:
            import nibabel as nib

            image = nib.load(str(run[0]))
            if tuple(image.shape[:3]) != spatial_shape or not np.allclose(
                image.affine, reference_affine
            ):
                raise ValueError(f"Cleaned volume grids differ across runs: {run[0]}")
            volume = np.asarray(image.dataobj, dtype=np.float32)
            if volume.shape[-1] != metadata.total_frames:
                raise ValueError(f"Cleaned volume dimensions disagree with metadata: {run[0]}")
            data = np.moveaxis(volume, -1, 0).reshape(metadata.total_frames, -1)
        data = np.asarray(data[retained], dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"Cleaned data contain non-finite values: {run}")
        mean = np.mean(data, axis=0, dtype=np.float64).astype(np.float32)
        data -= mean
        sum_squares = np.einsum("ti,ti->i", data, data, dtype=np.float64)
        valid = sum_squares > 0
        data[:, valid] /= np.sqrt(sum_squares[valid]).astype(np.float32)
        data[:, ~valid] = 0.0
        data *= np.sqrt(np.float32(record["normalized_weight"]))
        for start in range(0, len(data), TEMPORAL_BLOCK_FRAMES):
            yield data[start : start + TEMPORAL_BLOCK_FRAMES]


def build_module(
    cfg: ModuleConfig,
    runner: Runner,
    *,
    completion_boundary: bool = True,
) -> dict[str, Path | tuple[Path, ...]]:
    """Build the complete dynamic-connectivity DAG without executing it."""

    validate_config(cfg)
    out, work = cfg.output.directory, cfg.output.work_directory
    force = bool(cfg.output.overwrite)
    paths = output_paths(out, cfg.output.prefix, cfg.inputs.domain)
    functional_inputs = _flatten(cfg.inputs.functional)
    functional_sidecars = tuple(
        dict.fromkeys(sidecar_json_path(path) for path in functional_inputs)
    )
    source_inputs = (
        *functional_inputs,
        *functional_sidecars,
        *cfg.inputs.temporal_masks,
    )
    initialized = work / "initialized.complete"

    def initialize() -> None:
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        write_completion_breadcrumb(initialized, "Dynamic-connectivity output initialized\n")

    runner.add_step(
        Step.python(
            name="Initialize Dynamic-Connectivity Outputs",
            outputs=(initialized,),
            action=initialize,
        )
    )

    config_snapshot = work / "module-config.json"
    config_payload = scientific_values(
        "dynconn",
        {
            "inputs": asdict(cfg.inputs),
            "inclusion": asdict(cfg.inclusion),
            "weighting": cfg.weighting,
            "low_rank": cfg.low_rank,
            "low_rank_options": asdict(cfg.low_rank_options),
            "output_prefix": cfg.output.prefix,
        },
    )
    config_text = (
        json.dumps(config_payload, default=json_path_default, indent=2, sort_keys=True) + "\n"
    )

    def validate_config_snapshot() -> tuple[bool, str]:
        matches = (
            config_snapshot.is_file() and config_snapshot.read_text(encoding="utf-8") == config_text
        )
        return matches, "Configuration snapshot matches."

    runner.add_step(
        Step.python(
            name="Resolve Dynamic-Connectivity Configuration",
            inputs=(initialized,),
            outputs=(config_snapshot,),
            force=force,
            action=lambda: atomic_write_text(config_snapshot, config_text),
            validate=validate_config_snapshot,
        )
    )

    policy = CleanedRunInclusionPolicy(
        minimum_retained_frames=cfg.inclusion.minimum_retained_frames,
        minimum_retained_fraction=cfg.inclusion.minimum_retained_fraction,
        minimum_residual_design_dof=cfg.inclusion.minimum_residual_design_dof,
        minimum_participation_effective_rank=(cfg.inclusion.minimum_participation_effective_rank),
        maximum_dominant_temporal_variance_fraction=(
            cfg.inclusion.maximum_dominant_temporal_variance_fraction
        ),
    )
    eligibility_path = work / "run-eligibility.json"

    def assess_runs() -> None:
        included, skipped = [], []
        included_metadata = []
        repetition_times = set()
        for index, (run, expected_mask) in enumerate(
            zip(cfg.inputs.functional, cfg.inputs.temporal_masks)
        ):
            metadata = load_cleaned_run_metadata(run)
            if metadata.temporal_mask_file.resolve() != expected_mask.resolve():
                raise ValueError(
                    "Cleaned sidecar names an unexpected temporal mask: "
                    f"{metadata.temporal_mask_file} != {expected_mask}"
                )
            reasons = cleaned_run_exclusions(metadata, policy)
            record = {
                "run_index": index,
                "files": [str(path) for path in run],
                "sidecars": [str(path) for path in metadata.sidecars],
                "temporal_mask": str(metadata.temporal_mask_file),
                "total_frames": metadata.total_frames,
                "retained_frames": metadata.retained_frames,
                "algebraic_temporal_rank": metadata.algebraic_temporal_rank,
            }
            if reasons:
                skipped.append({**record, "reasons": list(reasons)})
            else:
                repetition_times.add(_repetition_time(metadata.sidecars[0]))
                included.append(record)
                included_metadata.append(metadata)
        retained = sum(int(record["retained_frames"]) for record in included)
        if len(included) < cfg.inclusion.minimum_usable_runs:
            raise ValueError(
                f"Only {len(included)} cleaned runs are usable; at least "
                f"{cfg.inclusion.minimum_usable_runs} are required"
            )
        if retained < cfg.inclusion.minimum_aggregate_retained_frames:
            raise ValueError(
                f"Usable runs contain {retained} retained frames; at least "
                f"{cfg.inclusion.minimum_aggregate_retained_frames} are required"
            )
        if len(repetition_times) != 1:
            raise ValueError("Included runs must have one common RepetitionTime for concatenation")
        weights = connectivity_run_weights(included_metadata, weighting=cfg.weighting)
        for record, weight in zip(included, weights):
            record["effective_dof"] = weight.effective_dof
            record["normalized_weight"] = weight.normalized_weight
        start = 0
        for record in included:
            stop = start + int(record["retained_frames"])
            record["start_frame"] = start
            record["stop_frame"] = stop
            start = stop
        write_json_atomic(
            eligibility_path,
            {
                "included": included,
                "skipped": skipped,
                "concatenated_frames": retained,
                "repetition_time_seconds": repetition_times.pop(),
            },
        )

    runner.add_step(
        Step.python(
            name="Assess Cleaned Runs",
            inputs=(*source_inputs, config_snapshot),
            outputs=(eligibility_path,),
            force=force,
            action=assess_runs,
        )
    )

    low_rank_summary = work / "low-rank-summary.json"

    def concatenate() -> None:
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        included = eligibility["included"]
        n_frames = int(eligibility["concatenated_frames"])
        selected = [cfg.inputs.functional[int(record["run_index"])] for record in included]
        if cfg.low_rank:
            import nibabel as nib

            if cfg.inputs.domain == "surface":
                _, vertex_counts = surface_timeseries_shape(selected[0])
                n_locations = sum(vertex_counts)
                spatial_layout = vertex_counts
                published_spatial_shape = (n_locations,)
                first = None
                reference_affine = None
            else:
                first = nib.load(str(selected[0][0]))
                spatial_layout = tuple(int(value) for value in first.shape[:3])
                published_spatial_shape = spatial_layout
                n_locations = int(np.prod(spatial_layout))
                vertex_counts = ()
                reference_affine = first.affine

            def blocks():
                return _retained_blocks(
                    cfg,
                    included,
                    spatial_shape=spatial_layout,
                    reference_affine=reference_affine,
                )

            fit = fit_low_rank_correlation(
                blocks,
                n_locations=n_locations,
                dimensions=cfg.low_rank_options.dimensions,
                oversampling=cfg.low_rank_options.oversampling,
                power_iterations=cfg.low_rank_options.power_iterations,
            )
            if fit.input_frames != n_frames:
                raise RuntimeError(
                    f"Low-rank input yielded {fit.input_frames} frames; expected {n_frames}"
                )
            matrix_path = work / "low-rank-timeseries.float32.dat"
            matrix = np.memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    (fit.synthetic_frames, n_locations)
                    if cfg.inputs.domain == "surface"
                    else (*published_spatial_shape, fit.synthetic_frames)
                ),
            )
            flat_volume = (
                matrix.reshape(n_locations, fit.synthetic_frames)
                if cfg.inputs.domain == "volume"
                else None
            )
            for start in range(0, n_locations, SPATIAL_BLOCK_LOCATIONS):
                stop = min(start + SPATIAL_BLOCK_LOCATIONS, n_locations)
                samples = pseudo_timeseries_block(fit, start, stop)
                if flat_volume is None:
                    matrix[:, start:stop] = samples
                else:
                    flat_volume[start:stop] = samples.T
            matrix.flush()
            with atomic_output_path(paths["timeseries"]) as staged:
                if cfg.inputs.domain == "surface":
                    axis = None
                    structures = (
                        "CIFTI_STRUCTURE_CORTEX_LEFT",
                        "CIFTI_STRUCTURE_CORTEX_RIGHT",
                    )
                    for name, count in zip(structures, vertex_counts):
                        part = nib.cifti2.BrainModelAxis.from_surface(
                            np.arange(count), count, name=name
                        )
                        axis = part if axis is None else axis + part
                    series = nib.cifti2.SeriesAxis(
                        0.0,
                        1.0,
                        fit.synthetic_frames,
                        unit="SECOND",
                    )
                    header = nib.cifti2.Cifti2Header.from_axes((series, axis))
                    nib.save(
                        nib.Cifti2Image(matrix, header=header, dtype=np.float32),
                        str(staged),
                    )
                else:
                    assert first is not None
                    header = first.header.copy()
                    header.set_data_dtype(np.float32)
                    header.set_zooms((*header.get_zooms()[:3], 1.0))
                    header.set_xyzt_units(t="sec")
                    nib.save(nib.Nifti1Image(matrix, first.affine, header=header), str(staged))
            del matrix
            matrix_path.unlink(missing_ok=True)
            write_json_atomic(
                low_rank_summary,
                {
                    "method": "randomized spectral approximation of weighted run correlations",
                    "requested_dimensions": fit.requested_dimensions,
                    "dimensions": fit.dimensions,
                    "oversampling": cfg.low_rank_options.oversampling,
                    "power_iterations": cfg.low_rank_options.power_iterations,
                    "random_seed": fit.random_seed,
                    "input_frames": fit.input_frames,
                    "synthetic_frames": fit.synthetic_frames,
                    "spatial_locations": n_locations,
                    "valid_locations": fit.valid_locations,
                    "zero_variance_locations": n_locations - fit.valid_locations,
                    "realized_rank": fit.realized_rank,
                    "retained_variance_fraction": fit.retained_variance_fraction,
                    "eigenvalues": [float(value) for value in fit.eigenvalues],
                    "standardization": "centered and unit sum of squares per spatial location",
                    "synthetic_basis": "orthonormal zero-mean Helmert contrasts",
                },
            )
            return
        if cfg.inputs.domain == "surface":
            _, vertex_counts = surface_timeseries_shape(selected[0])
            matrix_path = work / "concatenated.float32.dat"
            matrix = np.memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=(n_frames, sum(vertex_counts)),
            )
            cursor = 0
            for block in _retained_blocks(
                cfg,
                included,
                spatial_shape=vertex_counts,
                reference_affine=None,
            ):
                matrix[cursor : cursor + len(block)] = block
                cursor += len(block)
            if cursor != n_frames:
                raise RuntimeError(f"Weighted input yielded {cursor} frames; expected {n_frames}")
            matrix.flush()
            import nibabel as nib

            axis = None
            structures = (
                "CIFTI_STRUCTURE_CORTEX_LEFT",
                "CIFTI_STRUCTURE_CORTEX_RIGHT",
            )
            for name, count in zip(structures, vertex_counts):
                part = nib.cifti2.BrainModelAxis.from_surface(np.arange(count), count, name=name)
                axis = part if axis is None else axis + part
            series = nib.cifti2.SeriesAxis(
                0.0,
                float(eligibility["repetition_time_seconds"]),
                n_frames,
                unit="SECOND",
            )
            header = nib.cifti2.Cifti2Header.from_axes((series, axis))
            with atomic_output_path(paths["timeseries"]) as staged:
                nib.save(
                    nib.Cifti2Image(matrix, header=header, dtype=np.float32),
                    str(staged),
                )
            del matrix
            matrix_path.unlink(missing_ok=True)
        else:
            import nibabel as nib

            first = nib.load(str(selected[0][0]))
            shape = tuple(int(value) for value in first.shape[:3])
            matrix_path = work / "concatenated-volume.float32.dat"
            matrix = np.memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=(*shape, n_frames),
            )
            flat = matrix.reshape(-1, n_frames)
            cursor = 0
            for block in _retained_blocks(
                cfg,
                included,
                spatial_shape=shape,
                reference_affine=first.affine,
            ):
                flat[:, cursor : cursor + len(block)] = block.T
                cursor += len(block)
            if cursor != n_frames:
                raise RuntimeError(f"Weighted input yielded {cursor} frames; expected {n_frames}")
            matrix.flush()
            header = first.header.copy()
            header.set_data_dtype(np.float32)
            header.set_zooms(
                (
                    *header.get_zooms()[:3],
                    float(eligibility["repetition_time_seconds"]),
                )
            )
            header.set_xyzt_units(t="sec")
            with atomic_output_path(paths["timeseries"]) as staged:
                nib.save(nib.Nifti1Image(matrix, first.affine, header=header), str(staged))
            del matrix
            matrix_path.unlink(missing_ok=True)

    runner.add_step(
        Step.python(
            name=(
                "Fit Low-Rank Dynamic-Connectivity Representation"
                if cfg.low_rank
                else "Concatenate Retained Time Series"
            ),
            inputs=(*source_inputs, eligibility_path),
            outputs=(
                (paths["timeseries"], low_rank_summary) if cfg.low_rank else (paths["timeseries"],)
            ),
            force=force,
            action=concatenate,
        )
    )

    def write_manifest() -> None:
        current_outputs = set(paths.values())
        for path in out.iterdir():
            if (
                path.name.startswith(f"{cfg.output.prefix}_")
                and path not in current_outputs
                and (path.is_file() or path.is_symlink())
            ):
                path.unlink()
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        compression = (
            json.loads(low_rank_summary.read_text(encoding="utf-8")) if cfg.low_rank else None
        )
        import nibabel as nib

        image = nib.load(str(paths["timeseries"]))
        spatial_shape = list(image.shape[1:] if cfg.inputs.domain == "surface" else image.shape[:3])
        published_frames = int(image.shape[0] if cfg.inputs.domain == "surface" else image.shape[3])
        payload = {
            "domain": cfg.inputs.domain,
            "space": cfg.inputs.space,
            "smoothing_fwhm_mm": cfg.inputs.smoothing_mm,
            "representation": "low_rank" if cfg.low_rank else "full",
            "weighting": cfg.weighting,
            "repetition_time_seconds": eligibility["repetition_time_seconds"],
            "series_axis_interpretation": (
                "synthetic low-rank coordinates"
                if cfg.low_rank
                else "weighted standardized retained acquisition frames"
            ),
            "spatial_shape": spatial_shape,
            "concatenated_frames": eligibility["concatenated_frames"],
            "published_frames": published_frames,
            "low_rank": compression,
            "functional_runs": {
                "inclusion_policy": asdict(cfg.inclusion),
                "minimum_usable_runs": cfg.inclusion.minimum_usable_runs,
                "minimum_aggregate_retained_frames": cfg.inclusion.minimum_aggregate_retained_frames,
                "included": eligibility["included"],
                "skipped": eligibility["skipped"],
            },
            "outputs": {
                "timeseries": str(paths["timeseries"]),
            },
            "config": scientific_values(
                "dynconn",
                {
                    "inclusion": asdict(cfg.inclusion),
                    "weighting": cfg.weighting,
                    "low_rank": cfg.low_rank,
                    "low_rank_options": asdict(cfg.low_rank_options),
                },
            ),
            "configuration_fingerprint": selected_configuration_fingerprint(),
            "output_metadata_contract": dynconn_output_contract(),
        }
        atomic_write_text(paths["manifest"], yaml.safe_dump(payload, sort_keys=False))

    def validate_manifest() -> tuple[bool, str]:
        try:
            payload = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
            validate_dynconn_manifest(payload)
            complete = all(
                path.is_file() and path.stat().st_size for path in (paths["timeseries"],)
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            complete = False
        return complete, (
            "Dynamic-connectivity publication is complete."
            if complete
            else "Dynamic-connectivity publication is incomplete."
        )

    runner.add_step(
        Step.python(
            name="Write Dynamic-Connectivity Manifest",
            inputs=(
                paths["timeseries"],
                eligibility_path,
                config_snapshot,
                *((low_rank_summary,) if cfg.low_rank else ()),
            ),
            outputs=(paths["manifest"],),
            force=force,
            action=write_manifest,
            validate=validate_manifest,
            completion_boundary=completion_boundary,
        )
    )
    return paths
