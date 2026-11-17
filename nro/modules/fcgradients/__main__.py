"""Command-line entry point for one functional-connectivity gradient work item."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import yaml

from nro.definitions.markup import load_source_markup
from nro.definitions.runtime import load_runtime_configuration
from nro.engine.bids import discover_raw_runs, matches_filter
from nro.engine.cli import module_target_parser, start_module_target, stderr
from nro.engine.gray_matter import infer_gray_matter_mask
from nro.engine.io import read_public_json, write_public_json
from nro.engine.paths import (
    anatomical_manifest_path,
    module_derivatives_root,
    module_output_roots,
)
from nro.engine.references import resolve_public_payload
from nro.engine.surface_geometry import dense_surface_geometry
from nro.engine.targets import (
    is_surface_space,
    supported_output_spaces,
    target_output_names,
)
from nro.engine.templates import find_fslr_32k_resampling_geometry
from nro.engine.workbench import resolve_workbench_command
from nro.modules.dynconn.contract import dynconn_output_paths
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import (
    load_runtime_workflow_snapshot,
    select_runtime_config,
    selected_configuration_fingerprint,
)
from nro.site.paths import bids_path

from .config import InputsConfig, ModuleConfig, OutputConfig, WorkbenchConfig
from .contract import fcgradients_output_paths
from .module import build_module


def _source_series(manifest_path: Path, domain: str) -> tuple[Path, str]:
    """Return the dynconn series and representation named by its manifest."""

    manifest = resolve_public_payload(
        manifest_path, yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    )
    if manifest.get("domain") != domain:
        raise ValueError(f"Dynamic-connectivity manifest is not a {domain} target: {manifest_path}")
    value = (manifest.get("outputs") or {}).get("timeseries")
    representation = str(manifest.get("representation") or "")
    if not value or representation not in {"full", "low_rank"}:
        raise ValueError(
            f"Dynamic-connectivity manifest lacks a time series or representation: {manifest_path}"
        )
    path = Path(str(value)).expanduser()
    timeseries = path if path.is_absolute() else manifest_path.parent / path
    if not timeseries.is_file():
        raise FileNotFoundError(f"Missing dynamic-connectivity time series: {timeseries}")
    return timeseries, representation


def make_target_config(
    project: str,
    participant: str,
    fcgradients_id: str,
    config: dict,
    *,
    space: str,
    smoothing_mm: int,
    source_manifest: Path,
    anatomical_manifest: Path,
    overwrite: bool | None = None,
    execution_context: ExecutionContext | None = None,
) -> ModuleConfig:
    """Resolve one requested target into an executable module configuration."""

    domain = "surface" if is_surface_space(space) else "volume"
    timeseries, representation = _source_series(source_manifest, domain)
    surfaces: tuple[Path, ...] = ()
    spheres: tuple[Path, ...] = ()
    mesh_spheres: tuple[Path, ...] = ()
    mesh_labels: tuple[Path, ...] = ()
    mask = None
    surface_mesh = str(config["surface_mesh"])
    if domain == "surface":
        surfaces = dense_surface_geometry(
            anatomical_manifest.parent, participant, "midthickness", space, timeseries
        )
        if surface_mesh == "fsLR32k":
            # FreeSurfer's sphere.reg places native vertices on the fsaverage
            # sphere; template targets use their own spheres, which are already there.
            spheres = dense_surface_geometry(
                anatomical_manifest.parent,
                participant,
                "sphere.reg" if space == "fsnative" else "sphere",
                space,
                timeseries,
            )
            geometry = tuple(find_fslr_32k_resampling_geometry(hemi=hemi) for hemi in ("L", "R"))
            mesh_spheres = tuple(sphere for sphere, _ in geometry)
            mesh_labels = tuple(label for _, label in geometry)
    else:
        mask = infer_gray_matter_mask(
            ((timeseries,),),
            config["mask"],
            project=project,
            participant=participant,
            anat_directory=config["anat_directory"],
            space=space,
            execution_context=execution_context,
        )
    sub_id = f"sub-{participant}"
    output_base, work_base = module_output_roots(
        "fcgradients", fcgradients_id, project=project, execution_context=execution_context
    )
    target, prefix = target_output_names(sub_id, space, smoothing_mm)
    return ModuleConfig(
        inputs=InputsConfig(
            source_manifest=source_manifest,
            timeseries=timeseries,
            representation=representation,
            domain=domain,
            space=space,
            smoothing_mm=smoothing_mm,
            surfaces=surfaces,
            spheres=spheres,
            mesh_spheres=mesh_spheres,
            mesh_labels=mesh_labels,
            mask=mask,
            mask_threshold=float(config["mask_threshold"]),
        ),
        output=OutputConfig(
            directory=output_base / sub_id,
            work_directory=work_base / target / sub_id,
            prefix=prefix,
            overwrite=config["overwrite"] if overwrite is None else overwrite,
        ),
        backend=str(config["backend"]),
        surface_mesh=surface_mesh,
        workbench=WorkbenchConfig(**config["workbench"]),
        wb_command=resolve_workbench_command(config["wb_command"]),
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the standalone module parser used by orchestration workers."""
    return module_target_parser("Estimate a functional-connectivity gradient map for one target.")


def main(
    argv: list[str] | None = None,
    *,
    execution_context: ExecutionContext | None = None,
) -> None:
    """Locate the dynconn target and execute one gradient work item."""

    args = build_parser().parse_args(argv)
    participant = start_module_target(args, execution_context)
    participant_id = f"sub-{participant}"
    bids_root = Path(bids_path() if execution_context is None else execution_context.paths.bids)
    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        configuration_class="fcgradients",
        execution_context=execution_context,
    )
    fcgradients_id, config = load_runtime_configuration(runtime_config, "fcgradients")
    snapshot = load_runtime_workflow_snapshot(runtime_config)
    configurations = snapshot["configurations"]
    load_source_markup(config["markup"], args.project, bids_root / args.project / participant_id)
    spaces = supported_output_spaces(str(configurations["anat"]["resolved"]["fsaverage_template"]))
    if args.space not in spaces:
        raise SystemExit(
            f"space-{args.space} is not published by func; choose from {', '.join(spaces)}"
        )
    domain = "surface" if is_surface_space(args.space) else "volume"
    anat_manifest = anatomical_manifest_path(
        participant_id,
        project=args.project,
        anat_id=str(config["anat_directory"]),
        bids_root=bids_root,
    )
    _, source_prefix = target_output_names(participant_id, args.space, args.smoothing)
    source_paths = dynconn_output_paths(
        module_derivatives_root(
            "dynconn",
            str(config["dynconn_directory"]),
            project=args.project,
            bids_root=bids_root,
        )
        / participant_id,
        source_prefix,
        domain,
    )
    source_manifest = source_paths["manifest"]
    source_index = source_paths["index"]
    if execution_context is not None:
        anat_manifest = execution_context.input_path(anat_manifest)
        source_manifest = execution_context.input_path(source_manifest)
        source_index = execution_context.input_path(source_index)
    if not source_index.is_file():
        raise FileNotFoundError(f"Missing dynconn publication index: {source_index}")
    index = read_public_json(source_index)
    if (
        index.get("space") != args.space
        or index.get("smoothing_fwhm_mm") != args.smoothing
        or str(Path(str(index.get("target_manifest", ""))).resolve())
        != str(source_manifest.resolve())
    ):
        raise ValueError(
            "dynconn publication does not match the requested "
            f"space-{args.space} smoothing-{args.smoothing}mm target: {source_index}"
        )
    cfg = make_target_config(
        args.project,
        participant,
        fcgradients_id,
        config,
        space=args.space,
        smoothing_mm=args.smoothing,
        source_manifest=source_manifest,
        anatomical_manifest=anat_manifest,
        overwrite=True if args.overwrite else None,
        execution_context=execution_context,
    )
    dynconn_config = configurations["dynconn"].get("resolved") or {}
    source_runs = tuple(
        run
        for run in discover_raw_runs(bids_root / args.project / participant_id)
        if matches_filter(run.entities, dynconn_config.get("input_filter"))
    )
    stderr(f"Planned {domain} space-{args.space} smoothing-{args.smoothing}mm gradient target\n")
    runner = Runner(
        module_name="Subject Functional-Connectivity Gradient Module",
        container=None,
        binds=(),
        logger=logging.getLogger("fcgradients"),
        execution_context=execution_context,
    )
    result = build_module(cfg, runner, completion_boundary=False)
    output_index = fcgradients_output_paths(cfg.output.directory, cfg.output.prefix, domain)[
        "index"
    ]
    published_outputs = tuple(result.values())
    payload = {
        "manifest_version": 1,
        "module": "fcgradients",
        "participant": participant_id,
        "domain": domain,
        "space": args.space,
        "smoothing_fwhm_mm": args.smoothing,
        "source_runs": [run.stem for run in source_runs],
        "target_manifest": str(result["manifest"]),
        "public_outputs": [str(path) for path in published_outputs],
        "configuration_fingerprint": selected_configuration_fingerprint(),
        "complete": True,
    }

    def validate_index() -> tuple[bool, str]:
        try:
            matches = read_public_json(output_index) == payload
        except (OSError, TypeError, ValueError):
            matches = False
        return matches, (
            "Gradient publication index is current."
            if matches
            else "Gradient publication index is missing or outdated."
        )

    runner.add_step(
        Step.python(
            name="Write Gradient Publication Index",
            inputs=published_outputs,
            outputs=(output_index,),
            force=bool(args.overwrite),
            action=lambda: write_public_json(output_index, payload),
            validate=validate_index,
            completion_boundary=True,
        )
    )
    started = time.perf_counter()
    stderr(f"Running {cfg.backend} on {cfg.inputs.timeseries.name}\n")
    runner.execute_graph(started_at=started)
    for name, path in result.items():
        print(f"{domain}/space-{args.space}/smoothing-{args.smoothing}mm/{name}: {path}")


if __name__ == "__main__":
    main()
