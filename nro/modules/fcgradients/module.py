"""Construct functional-connectivity gradient steps for one target."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from nro.engine.io import atomic_output_path, atomic_write_text, json_path_default
from nro.engine.references import portable_public_payload, resolve_public_payload
from nro.orchestration.runner import Runner, write_completion_breadcrumb
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import selected_configuration_fingerprint

from .config import ModuleConfig, WorkbenchConfig, scientific_settings, validate_config
from .contract import (
    fcgradients_output_contract,
    fcgradients_output_paths,
    validate_fcgradients_manifest,
)

VOLUME_STRUCTURE = "OTHER"
"""The single CIFTI structure that holds every gray-matter voxel.

``-cifti-correlation-gradient`` correlates only within a structure, so one
structure lets every voxel see every gray-matter seed.
"""


def correlation_gradient_command(
    wb_command: str,
    dense: Path,
    output: Path,
    *,
    options: WorkbenchConfig,
    surfaces: tuple[Path, ...] = (),
) -> list[str]:
    """Return the ``wb_command -cifti-correlation-gradient`` invocation.

    ``surfaces`` holds the left and right midthickness surfaces for a surface
    series and is empty for a volume-only series. Zero-valued smoothing and
    exclusion distances, and a null memory limit, omit the option.
    """

    command = [wb_command, "-cifti-correlation-gradient", str(dense), str(output)]
    if surfaces:
        left, right = surfaces
        command += ["-left-surface", str(left), "-right-surface", str(right)]
    if options.presmooth_fwhm_mm > 0:
        kernel = f"{options.presmooth_fwhm_mm:g}"
        command += (
            ["-surface-presmooth", kernel] if surfaces else ["-volume-presmooth", kernel]
        ) + ["-presmooth-fwhm"]
    if options.fisher_z:
        command.append("-fisher-z")
    if surfaces and options.surface_exclude_mm > 0:
        command += ["-surface-exclude", f"{options.surface_exclude_mm:g}"]
    if not surfaces and options.volume_exclude_mm > 0:
        command += ["-volume-exclude", f"{options.volume_exclude_mm:g}"]
    if options.memory_limit_gb is not None:
        command += ["-mem-limit", f"{options.memory_limit_gb:g}"]
    if options.double_correlation:
        command.append("-double-correlation")
        if options.fisher_z_first:
            command.append("-fisher-z-first")
    return command


MEMORY_FRACTION = 0.75
"""Share of the allocation given to Workbench when no limit is configured."""


def allocated_memory_gb(environment: Mapping[str, str]) -> float | None:
    """Return the memory allocated to this process, if the environment states it.

    Workers export ``NRO_WORKER_MEMORY_GB``. A direct Slurm job states
    ``SLURM_MEM_PER_NODE`` in megabytes.
    """

    value = environment.get("NRO_WORKER_MEMORY_GB")
    if value:
        return float(value)
    value = environment.get("SLURM_MEM_PER_NODE")
    if value and value.isdigit():
        return int(value) / 1024
    return None


def effective_options(options: WorkbenchConfig, environment: Mapping[str, str]) -> WorkbenchConfig:
    """Fill a null memory limit from the allocation, leaving explicit limits unchanged."""

    if options.memory_limit_gb is not None:
        return options
    allocated = allocated_memory_gb(environment)
    if allocated is None:
        return options
    return replace(options, memory_limit_gb=round(allocated * MEMORY_FRACTION, 1))


def write_mask_volume(mask: Path, reference: Path, output: Path, *, threshold: float) -> int:
    """Write a binary gray-matter mask on the reference grid and return its voxel count.

    A mask on another grid is resampled: linearly for a probability map (a name
    containing ``probseg``) and by nearest neighbor otherwise. Voxels above
    ``threshold`` are retained.
    """

    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to

    target = nib.load(str(reference))
    shape = tuple(int(value) for value in target.shape[:3])
    image = nib.load(str(mask))
    if tuple(image.shape[:3]) != shape or not np.allclose(
        image.affine, target.affine, rtol=0.0, atol=1e-5
    ):
        image = resample_from_to(
            image, (shape, target.affine), order=1 if "probseg" in mask.name else 0
        )
    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.ndim == 4 and values.shape[3] == 1:
        values = values[..., 0]
    if values.shape != shape:
        raise ValueError(f"Gray-matter mask is not three-dimensional: {mask}")
    selected = np.isfinite(values) & (values > np.float32(threshold))
    count = int(selected.sum())
    if count < 2:
        raise ValueError(f"Gray-matter mask retains fewer than two voxels: {mask}")
    result = nib.Nifti1Image(selected.astype(np.int32), target.affine)
    result.header.set_data_dtype(np.int32)
    nib.save(result, str(output))
    return count


def build_module(
    cfg: ModuleConfig,
    runner: Runner,
    *,
    completion_boundary: bool = True,
) -> dict[str, Path]:
    """Build the gradient DAG without executing it."""

    validate_config(cfg)
    out, work = cfg.output.directory, cfg.output.work_directory
    force = bool(cfg.output.overwrite)
    domain = cfg.inputs.domain
    paths = fcgradients_output_paths(out, cfg.output.prefix, domain)
    settings = scientific_settings(cfg)
    initialized = work / "initialized.complete"

    def initialize() -> None:
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        write_completion_breadcrumb(initialized, "Gradient output initialized\n")

    runner.add_step(
        Step.python(
            name="Initialize Gradient Outputs",
            outputs=(initialized,),
            action=initialize,
        )
    )

    version_path = work / "workbench-version.txt"

    def record_version() -> None:
        text = runner.run_child([cfg.wb_command, "-version"], capture_stdout=True) or ""
        atomic_write_text(version_path, text)

    runner.add_step(
        Step.python(
            name="Record Workbench Version",
            inputs=(initialized,),
            outputs=(version_path,),
            force=force,
            action=record_version,
        )
    )

    geometry_inputs: tuple[Path, ...]
    if domain == "surface":
        dense = cfg.inputs.timeseries
        geometry_inputs = cfg.inputs.surfaces
    else:
        assert cfg.inputs.mask is not None
        dense = work / "gray-matter.dtseries.nii"
        mask_volume = work / "gray-matter-mask.nii.gz"
        label_volume = work / "gray-matter-label.nii.gz"
        label_list = work / "gray-matter-label.txt"
        geometry_inputs = (cfg.inputs.mask,)

        def build_dense_series() -> None:
            write_mask_volume(
                cfg.inputs.mask,
                cfg.inputs.timeseries,
                mask_volume,
                threshold=cfg.inputs.mask_threshold,
            )
            atomic_write_text(label_list, f"{VOLUME_STRUCTURE}\n1 255 255 255 255\n")
            runner.run_child(
                [
                    cfg.wb_command,
                    "-volume-label-import",
                    str(mask_volume),
                    str(label_list),
                    str(label_volume),
                    "-discard-others",
                ]
            )
            with atomic_output_path(dense) as staged:
                runner.run_child(
                    [
                        cfg.wb_command,
                        "-cifti-create-dense-timeseries",
                        str(staged),
                        "-volume",
                        str(cfg.inputs.timeseries),
                        str(label_volume),
                    ]
                )

        runner.add_step(
            Step.python(
                name="Restrict Volume Series to Gray Matter",
                inputs=(initialized, cfg.inputs.timeseries, cfg.inputs.mask),
                outputs=(dense, mask_volume, label_volume, label_list),
                force=force,
                action=build_dense_series,
                parameters={"mask_threshold": cfg.inputs.mask_threshold},
            )
        )

    def compute_gradient() -> None:
        with atomic_output_path(paths["gradient"]) as staged:
            runner.run_child(
                correlation_gradient_command(
                    cfg.wb_command,
                    dense,
                    staged,
                    options=effective_options(cfg.workbench, os.environ),
                    surfaces=cfg.inputs.surfaces if domain == "surface" else (),
                ),
                stream_output=True,
            )

    runner.add_step(
        Step.python(
            name="Compute Correlation Gradient",
            inputs=(initialized, dense, *geometry_inputs),
            outputs=(paths["gradient"],),
            force=force,
            action=compute_gradient,
            parameters=settings,
        )
    )

    if domain == "volume":

        def separate_volume() -> None:
            with atomic_output_path(paths["gradient_volume"]) as staged:
                runner.run_child(
                    [
                        cfg.wb_command,
                        "-cifti-separate",
                        str(paths["gradient"]),
                        "COLUMN",
                        "-volume-all",
                        str(staged),
                    ]
                )

        runner.add_step(
            Step.python(
                name="Write Gradient Volume",
                inputs=(paths["gradient"],),
                outputs=(paths["gradient_volume"],),
                force=force,
                action=separate_volume,
            )
        )

    published = tuple(paths[name] for name in ("gradient", "gradient_volume") if name in paths)

    def write_manifest() -> None:
        import nibabel as nib

        current = set(paths.values())
        for path in out.iterdir():
            if (
                path.name.startswith(f"{cfg.output.prefix}_")
                and path not in current
                and (path.is_file() or path.is_symlink())
            ):
                path.unlink()
        axis = nib.load(str(paths["gradient"])).header.get_axis(1)
        structures = [
            {"name": str(name), "locations": len(part)} for name, _, part in axis.iter_structures()
        ]
        source = resolve_public_payload(
            cfg.inputs.source_manifest,
            yaml.safe_load(cfg.inputs.source_manifest.read_text(encoding="utf-8")) or {},
        )
        version = version_path.read_text(encoding="utf-8").strip().splitlines()
        payload = {
            "domain": domain,
            "space": cfg.inputs.space,
            "smoothing_fwhm_mm": cfg.inputs.smoothing_mm,
            "backend": cfg.backend,
            "backend_details": {
                "command": "wb_command -cifti-correlation-gradient",
                "workbench_version": next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in version
                        if line.startswith("Version")
                    ),
                    None,
                ),
                "options": asdict(cfg.workbench),
                "statistic": "mean gradient magnitude of per-seed correlation maps",
                "correlation_scope": "within each CIFTI structure",
            },
            "source": {
                "module": "dynconn",
                "manifest": str(cfg.inputs.source_manifest),
                "timeseries": str(cfg.inputs.timeseries),
                "representation": cfg.inputs.representation,
                "low_rank": source.get("low_rank"),
            },
            "geometry": (
                {"surfaces": [str(path) for path in cfg.inputs.surfaces]}
                if domain == "surface"
                else {
                    "mask": str(cfg.inputs.mask),
                    "mask_threshold": cfg.inputs.mask_threshold,
                    "structure": VOLUME_STRUCTURE,
                }
            ),
            "structures": structures,
            "locations": int(len(axis)),
            "outputs": {
                "gradient": str(paths["gradient"]),
                **(
                    {"gradient_volume": str(paths["gradient_volume"])}
                    if "gradient_volume" in paths
                    else {}
                ),
            },
            "config": json.loads(json.dumps(settings, default=json_path_default)),
            "configuration_fingerprint": selected_configuration_fingerprint(),
            "output_metadata_contract": fcgradients_output_contract(),
        }
        atomic_write_text(
            paths["manifest"],
            yaml.safe_dump(portable_public_payload(paths["manifest"], payload), sort_keys=False),
        )

    def validate_manifest() -> tuple[bool, str]:
        try:
            payload = resolve_public_payload(
                paths["manifest"],
                yaml.safe_load(paths["manifest"].read_text(encoding="utf-8")),
            )
            validate_fcgradients_manifest(payload)
            complete = all(path.is_file() and path.stat().st_size for path in published)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            complete = False
        return complete, (
            "Gradient publication is complete."
            if complete
            else "Gradient publication is incomplete."
        )

    runner.add_step(
        Step.python(
            name="Write Gradient Manifest",
            inputs=(*published, version_path, cfg.inputs.source_manifest),
            outputs=(paths["manifest"],),
            force=force,
            action=write_manifest,
            validate=validate_manifest,
            completion_boundary=completion_boundary,
        )
    )
    return {name: path for name, path in paths.items() if name != "index"}
