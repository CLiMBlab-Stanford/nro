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
from .paths import output_paths
from .scene import surface_output_paths, write_surface_scene, write_volume_scene


def _flatten(groups: tuple[tuple[Path, ...], ...]) -> tuple[Path, ...]:
    return tuple(path for group in groups for path in group)


def _repetition_time(sidecar: Path) -> float:
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    value = document.get("RepetitionTime")
    if value is None or float(value) <= 0:
        raise ValueError(f"Cleaned sidecar lacks a positive RepetitionTime: {sidecar}")
    return float(value)


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
    packaged_surfaces = (
        surface_output_paths(out, cfg.output.prefix) if cfg.inputs.domain == "surface" else ()
    )
    functional_inputs = _flatten(cfg.inputs.functional)
    functional_sidecars = tuple(
        dict.fromkeys(sidecar_json_path(path) for path in functional_inputs)
    )
    source_inputs = (
        *functional_inputs,
        *functional_sidecars,
        *cfg.inputs.temporal_masks,
        *cfg.inputs.surface,
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
            }
            if reasons:
                skipped.append({**record, "reasons": list(reasons)})
            else:
                repetition_times.add(_repetition_time(metadata.sidecars[0]))
                included.append(record)
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

    def concatenate() -> None:
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        included = eligibility["included"]
        n_frames = int(eligibility["concatenated_frames"])
        selected = [cfg.inputs.functional[int(record["run_index"])] for record in included]
        if cfg.inputs.domain == "surface":
            _, vertex_counts = surface_timeseries_shape(selected[0])
            matrix_path = work / "concatenated.float32.dat"
            matrix = np.memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=(n_frames, sum(vertex_counts)),
            )
            for record, run in zip(included, selected):
                metadata = load_cleaned_run_metadata(run)
                data = load_surface_timeseries(run)
                if data.shape != (metadata.total_frames, sum(vertex_counts)):
                    raise ValueError(f"Cleaned surface dimensions disagree with metadata: {run}")
                if not np.all(np.isfinite(data)):
                    raise ValueError(f"Cleaned surface data contain non-finite values: {run}")
                matrix[int(record["start_frame"]) : int(record["stop_frame"])] = data[
                    load_retained_frame_mask(metadata)
                ]
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
            for record, run in zip(included, selected):
                metadata = load_cleaned_run_metadata(run)
                image = nib.load(str(run[0]))
                if tuple(image.shape[:3]) != shape or not np.allclose(image.affine, first.affine):
                    raise ValueError(f"Cleaned volume grids differ across runs: {run[0]}")
                data = np.asarray(image.dataobj, dtype=np.float32)
                if data.shape != (*shape, metadata.total_frames):
                    raise ValueError(f"Cleaned volume dimensions disagree with metadata: {run[0]}")
                if not np.all(np.isfinite(data)):
                    raise ValueError(f"Cleaned volume data contain non-finite values: {run[0]}")
                matrix[..., int(record["start_frame"]) : int(record["stop_frame"])] = data[
                    ..., load_retained_frame_mask(metadata)
                ]
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
            name="Concatenate Retained Time Series",
            inputs=(*source_inputs, eligibility_path),
            outputs=(paths["timeseries"],),
            force=force,
            action=concatenate,
        )
    )

    def package_scene() -> None:
        if cfg.inputs.domain == "surface":
            write_surface_scene(out, cfg.output.prefix, paths["timeseries"], cfg.inputs.surface)
        else:
            write_volume_scene(out, cfg.output.prefix, paths["timeseries"])

    runner.add_step(
        Step.python(
            name="Package Workbench Dynamic-Connectivity Scene",
            inputs=(paths["timeseries"], *cfg.inputs.surface),
            outputs=(paths["scene"], *packaged_surfaces),
            force=force,
            action=package_scene,
        )
    )

    def write_manifest() -> None:
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        import nibabel as nib

        image = nib.load(str(paths["timeseries"]))
        spatial_shape = list(image.shape[1:] if cfg.inputs.domain == "surface" else image.shape[:3])
        payload = {
            "domain": cfg.inputs.domain,
            "space": cfg.inputs.space,
            "smoothing_fwhm_mm": cfg.inputs.smoothing_mm,
            "repetition_time_seconds": eligibility["repetition_time_seconds"],
            "spatial_shape": spatial_shape,
            "source_surfaces": [str(path) for path in cfg.inputs.surface],
            "concatenated_frames": eligibility["concatenated_frames"],
            "functional_runs": {
                "inclusion_policy": asdict(cfg.inclusion),
                "minimum_usable_runs": cfg.inclusion.minimum_usable_runs,
                "minimum_aggregate_retained_frames": cfg.inclusion.minimum_aggregate_retained_frames,
                "included": eligibility["included"],
                "skipped": eligibility["skipped"],
            },
            "outputs": {
                "timeseries": str(paths["timeseries"]),
                "scene": str(paths["scene"]),
                "surfaces": [str(path) for path in packaged_surfaces],
            },
            "config": scientific_values("dynconn", {"inclusion": asdict(cfg.inclusion)}),
            "configuration_fingerprint": selected_configuration_fingerprint(),
            "output_metadata_contract": dynconn_output_contract(),
        }
        atomic_write_text(paths["manifest"], yaml.safe_dump(payload, sort_keys=False))

    def validate_manifest() -> tuple[bool, str]:
        try:
            payload = yaml.safe_load(paths["manifest"].read_text(encoding="utf-8"))
            validate_dynconn_manifest(payload)
            complete = all(
                path.is_file() and path.stat().st_size
                for path in (paths["timeseries"], paths["scene"], *packaged_surfaces)
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
                paths["scene"],
                paths["timeseries"],
                *packaged_surfaces,
                eligibility_path,
                config_snapshot,
            ),
            outputs=(paths["manifest"],),
            force=force,
            action=write_manifest,
            validate=validate_manifest,
            completion_boundary=completion_boundary,
        )
    )
    return {**paths, "surfaces": packaged_surfaces}
