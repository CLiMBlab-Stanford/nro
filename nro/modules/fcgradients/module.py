"""Construct functional-connectivity gradient steps for one target."""

from __future__ import annotations

import json
import os
import tempfile
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


HEMISPHERES = (("L", "CORTEX_LEFT"), ("R", "CORTEX_RIGHT"))
MESH_RESAMPLING = "ADAP_BARY_AREA"
"""Workbench method for series and gradient resampling between meshes.

Area-weighted averaging lets each 32k vertex average the target vertices it
covers. Plain barycentric interpolation would sample the target series at points
and keep part of its voxel-scale structure.
"""


def write_label_roi(label: Path, output: Path) -> None:
    """Write a 0/1 metric ROI holding the nonzero locations of a GIFTI label."""

    import nibabel as nib
    import numpy as np

    image = nib.load(str(label))
    if not image.darrays:
        raise ValueError(f"GIFTI label has no data arrays: {label}")
    selected = np.asarray(image.darrays[0].data).reshape(-1) > 0
    roi = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(selected.astype(np.float32))])
    with atomic_output_path(output) as staged:
        nib.save(roi, str(staged))


def series_timestep(path: Path) -> float:
    """Return the sampling interval of a dense time series."""

    import nibabel as nib

    return float(nib.load(str(path)).header.get_axis(0).step)


def surface_geometry_record(cfg: ModuleConfig) -> dict[str, object]:
    """Describe the surfaces, and any intermediate mesh, behind a surface gradient."""

    record: dict[str, object] = {
        "surfaces": [str(path) for path in cfg.inputs.surfaces],
        "surface_anatomy": "participant" if cfg.inputs.space == "fsnative" else "template",
        "surface_mesh": cfg.surface_mesh,
    }
    if cfg.surface_mesh == "fsLR32k":
        record["intermediate_mesh"] = {
            "name": "fsLR",
            "density": "32k",
            "series_resampling": MESH_RESAMPLING,
            "gradient_resampling": MESH_RESAMPLING,
            "midthickness_resampling": "BARYCENTRIC",
            "target_spheres": [str(path) for path in cfg.inputs.spheres],
            "mesh_spheres": [str(path) for path in cfg.inputs.mesh_spheres],
            "medial_wall_labels": [str(path) for path in cfg.inputs.mesh_labels],
            "target_mask": "vertices with nonzero series variance",
        }
    return record


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
    gradient_surfaces: tuple[Path, ...] = ()
    gradient_output = paths["gradient"]
    on_mesh = domain == "surface" and cfg.surface_mesh == "fsLR32k"
    if domain == "surface" and not on_mesh:
        dense = cfg.inputs.timeseries
        geometry_inputs = cfg.inputs.surfaces
        gradient_surfaces = cfg.inputs.surfaces
    elif on_mesh:
        mesh_dir = work / "fsLR32k"
        mesh_surfaces = tuple(
            mesh_dir / f"hemi-{hemi}_den-32k_midthickness.surf.gii" for hemi, _ in HEMISPHERES
        )
        target_rois = tuple(
            mesh_dir / f"hemi-{hemi}_desc-targetValid_mask.shape.gii" for hemi, _ in HEMISPHERES
        )
        mesh_rois = tuple(
            mesh_dir / f"hemi-{hemi}_den-32k_desc-valid_mask.shape.gii" for hemi, _ in HEMISPHERES
        )
        dense = mesh_dir / "den-32k_desc-dynamicConnectivity_bold.dtseries.nii"
        gradient_output = mesh_dir / "den-32k_desc-fcGradient_stat.dscalar.nii"
        geometry_inputs = mesh_surfaces
        gradient_surfaces = mesh_surfaces
        mesh_parameters = {"surface_mesh": cfg.surface_mesh, "resampling": MESH_RESAMPLING}

        def resample_midthickness() -> None:
            for surface, sphere, mesh_sphere, output in zip(
                cfg.inputs.surfaces,
                cfg.inputs.spheres,
                cfg.inputs.mesh_spheres,
                mesh_surfaces,
            ):
                with atomic_output_path(output) as staged:
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-surface-resample",
                            str(surface),
                            str(sphere),
                            str(mesh_sphere),
                            "BARYCENTRIC",
                            str(staged),
                        ]
                    )

        runner.add_step(
            Step.python(
                name="Resample Midthickness to 32k fsLR",
                inputs=(
                    initialized,
                    *cfg.inputs.surfaces,
                    *cfg.inputs.spheres,
                    *cfg.inputs.mesh_spheres,
                ),
                outputs=mesh_surfaces,
                force=force,
                action=resample_midthickness,
            )
        )

        def resample_series() -> None:
            with tempfile.TemporaryDirectory(dir=mesh_dir, prefix=".series-") as scratch:
                scratch_dir = Path(scratch)
                target = {hemi: scratch_dir / f"{hemi}.func.gii" for hemi, _ in HEMISPHERES}
                runner.run_child(
                    [
                        cfg.wb_command,
                        "-cifti-separate",
                        str(cfg.inputs.timeseries),
                        "COLUMN",
                        *(
                            item
                            for hemi, structure in HEMISPHERES
                            for item in ("-metric", structure, str(target[hemi]))
                        ),
                    ]
                )
                resampled = {}
                for index, (hemi, _) in enumerate(HEMISPHERES):
                    deviation = scratch_dir / f"{hemi}.stdev.func.gii"
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-metric-reduce",
                            str(target[hemi]),
                            "STDEV",
                            str(deviation),
                        ]
                    )
                    # Zero-variance vertices, mostly the medial wall, carry no signal.
                    with atomic_output_path(target_rois[index]) as staged:
                        runner.run_child(
                            [
                                cfg.wb_command,
                                "-metric-math",
                                "x > 0",
                                str(staged),
                                "-var",
                                "x",
                                str(deviation),
                            ]
                        )
                    resampled[hemi] = scratch_dir / f"{hemi}.32k.func.gii"
                    covered = scratch_dir / f"{hemi}.32k.covered.shape.gii"
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-metric-resample",
                            str(target[hemi]),
                            str(cfg.inputs.spheres[index]),
                            str(cfg.inputs.mesh_spheres[index]),
                            MESH_RESAMPLING,
                            str(resampled[hemi]),
                            "-area-surfs",
                            str(cfg.inputs.surfaces[index]),
                            str(mesh_surfaces[index]),
                            "-current-roi",
                            str(target_rois[index]),
                            "-valid-roi-out",
                            str(covered),
                        ]
                    )
                    target[hemi].unlink()
                    cortex = scratch_dir / f"{hemi}.cortex.shape.gii"
                    write_label_roi(cfg.inputs.mesh_labels[index], cortex)
                    with atomic_output_path(mesh_rois[index]) as staged:
                        runner.run_child(
                            [
                                cfg.wb_command,
                                "-metric-math",
                                "cortex * covered",
                                str(staged),
                                "-var",
                                "cortex",
                                str(cortex),
                                "-var",
                                "covered",
                                str(covered),
                            ]
                        )
                with atomic_output_path(dense) as staged:
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-cifti-create-dense-timeseries",
                            str(staged),
                            "-left-metric",
                            str(resampled["L"]),
                            "-roi-left",
                            str(mesh_rois[0]),
                            "-right-metric",
                            str(resampled["R"]),
                            "-roi-right",
                            str(mesh_rois[1]),
                            "-timestep",
                            f"{series_timestep(cfg.inputs.timeseries):g}",
                        ]
                    )

        runner.add_step(
            Step.python(
                name="Resample Series to 32k fsLR",
                inputs=(
                    initialized,
                    cfg.inputs.timeseries,
                    *cfg.inputs.surfaces,
                    *cfg.inputs.spheres,
                    *cfg.inputs.mesh_spheres,
                    *cfg.inputs.mesh_labels,
                    *mesh_surfaces,
                ),
                outputs=(dense, *target_rois, *mesh_rois),
                force=force,
                action=resample_series,
                parameters=mesh_parameters,
            )
        )
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
        with atomic_output_path(gradient_output) as staged:
            runner.run_child(
                correlation_gradient_command(
                    cfg.wb_command,
                    dense,
                    staged,
                    options=effective_options(cfg.workbench, os.environ),
                    surfaces=gradient_surfaces,
                ),
                stream_output=True,
            )

    runner.add_step(
        Step.python(
            name="Compute Correlation Gradient",
            inputs=(initialized, dense, *geometry_inputs),
            outputs=(gradient_output,),
            force=force,
            action=compute_gradient,
            parameters=settings,
        )
    )

    if on_mesh:

        def resample_gradient() -> None:
            with tempfile.TemporaryDirectory(dir=mesh_dir, prefix=".gradient-") as scratch:
                scratch_dir = Path(scratch)
                mesh_maps = {hemi: scratch_dir / f"{hemi}.32k.func.gii" for hemi, _ in HEMISPHERES}
                runner.run_child(
                    [
                        cfg.wb_command,
                        "-cifti-separate",
                        str(gradient_output),
                        "COLUMN",
                        *(
                            item
                            for hemi, structure in HEMISPHERES
                            for item in ("-metric", structure, str(mesh_maps[hemi]))
                        ),
                    ]
                )
                target_maps = {}
                for index, (hemi, _) in enumerate(HEMISPHERES):
                    resampled = scratch_dir / f"{hemi}.target.func.gii"
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-metric-resample",
                            str(mesh_maps[hemi]),
                            str(cfg.inputs.mesh_spheres[index]),
                            str(cfg.inputs.spheres[index]),
                            MESH_RESAMPLING,
                            str(resampled),
                            "-area-surfs",
                            str(mesh_surfaces[index]),
                            str(cfg.inputs.surfaces[index]),
                            "-current-roi",
                            str(mesh_rois[index]),
                        ]
                    )
                    target_maps[hemi] = scratch_dir / f"{hemi}.masked.func.gii"
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-metric-mask",
                            str(resampled),
                            str(target_rois[index]),
                            str(target_maps[hemi]),
                        ]
                    )
                # The published map keeps the dynconn series' brain-model layout.
                with atomic_output_path(paths["gradient"]) as staged:
                    runner.run_child(
                        [
                            cfg.wb_command,
                            "-cifti-create-dense-from-template",
                            str(cfg.inputs.timeseries),
                            str(staged),
                            *(
                                item
                                for hemi, structure in HEMISPHERES
                                for item in ("-metric", structure, str(target_maps[hemi]))
                            ),
                        ]
                    )

        runner.add_step(
            Step.python(
                name="Resample Gradient to Target Mesh",
                inputs=(
                    gradient_output,
                    cfg.inputs.timeseries,
                    *cfg.inputs.surfaces,
                    *cfg.inputs.spheres,
                    *cfg.inputs.mesh_spheres,
                    *mesh_surfaces,
                    *target_rois,
                    *mesh_rois,
                ),
                outputs=(paths["gradient"],),
                force=force,
                action=resample_gradient,
                parameters=mesh_parameters,
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
                surface_geometry_record(cfg)
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
