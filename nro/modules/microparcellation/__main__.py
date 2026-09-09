"""Select inputs and run one microparcellation module instance."""

import argparse
import json
import logging
import time
from itertools import count
from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.configuration.runtime import load_runtime_configuration
from nro.engine.bids import (
    discover_raw_runs,
    matches_filter,
)
from nro.engine.clean_targets import CleanTarget, expected_clean_target
from nro.engine.cli import stderr
from nro.engine.io import flatten_paths
from nro.engine.paths import preprocessing_subject_anat_dir
from nro.engine.publication import write_json_atomic
from nro.engine.surface_geometry import surface_geometry
from nro.engine.targets import (
    DEFAULT_SMOOTHING_MM,
    DEFAULT_SPACE,
    target_output_names,
)
from nro.engine.templates import find_mni_gray_matter_mask
from nro.modules.microparcellation.config import (
    CoarseningConfig,
    ConnectivityConfig,
    InputsConfig,
    ModuleConfig,
    OutputConfig,
    QualityConfig,
)
from nro.modules.microparcellation.module import build_module
from nro.modules.microparcellation.paths import output_paths
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import (
    load_runtime_workflow_snapshot,
    select_runtime_config,
    selected_configuration_fingerprint,
)


def infer_gray_matter_mask(
    functionals,
    configured_mask,
    *,
    project,
    participant=None,
    preprocessing_directory=None,
    space,
    execution_context: ExecutionContext | None = None,
):
    """Resolve a configured or space-specific gray-matter mask."""
    if configured_mask:
        path = Path(configured_mask).expanduser()
        if not path.is_absolute():
            path = (
                Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
                / project
                / path
            )
        if execution_context is not None:
            path = execution_context.input_path(path)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured gray-matter mask does not exist: {path}")
        return path

    if space.startswith("MNI"):
        return find_mni_gray_matter_mask(space=space, functional=functionals[0][0])
    if participant is None or preprocessing_directory is None:
        raise ValueError(
            "participant and preprocessing_directory are required for a native-space mask"
        )
    anat_dir = preprocessing_subject_anat_dir(
        f"sub-{participant}",
        project=project,
        preprocessing_id=preprocessing_directory,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    manifest_path = Path(anat_dir) / f"sub-{participant}_desc-preprocessAnat_manifest.json"
    if execution_context is not None:
        manifest_path = execution_context.input_path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing anatomical publication manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = (manifest.get("outputs") or {}).get("gray_matter_mask")
    if not value:
        raise FileNotFoundError(
            f"Anatomical manifest has no outputs.gray_matter_mask: {manifest_path}"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing published anatomical gray-matter mask: {path}")
    return path.resolve()


def make_target_config(
    project,
    participant,
    microparcellation_id,
    config,
    *,
    overwrite=None,
    clean_target: CleanTarget | None = None,
    execution_context: ExecutionContext | None = None,
):
    """Construct a module config for one requested space and smoothing level."""
    sub_id = f"sub-{participant}"
    if execution_context is not None and execution_context.project != project:
        raise ValueError("Microparcellation project differs from its execution context")
    if clean_target is None:
        raise ValueError("clean_target must be constructed from source BIDS and requested entities")
    target = clean_target
    default_base = (
        Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
        / project
        / "derivatives"
        / "microparcellation"
        / microparcellation_id
    )
    output_base = Path(config.get("output_dir") or default_base)
    work_base = (
        Path(WORK_PATH if execution_context is None else execution_context.paths.work)
        / project
        / "derivatives"
        / "microparcellation"
        / microparcellation_id
    )
    if execution_context is not None:
        output_base = execution_context.output_path(output_base)
        work_base = execution_context.output_path(work_base, private=True)
    if target.domain == "surface":
        anat_path = preprocessing_subject_anat_dir(
            sub_id,
            project=project,
            preprocessing_id=config["preprocessing_directory"],
            bids_root=None if execution_context is None else execution_context.paths.bids,
        )
        if execution_context is not None and target.space == "fsnative":
            anat_path = execution_context.input_path(
                anat_path / f"{sub_id}_desc-preprocessAnat_manifest.json"
            ).parent
        surfaces = surface_geometry(
            anat_path,
            participant,
            config["surface"],
            target.space,
            target.functional[0],
        )
        mask = None
    else:
        surfaces = ()
        mask = infer_gray_matter_mask(
            target.functional,
            config.get("mask"),
            project=project,
            participant=participant,
            preprocessing_directory=config["preprocessing_directory"],
            space=target.space,
            execution_context=execution_context,
        )
    target_name, target_prefix = target_output_names(
        config.get("prefix") or sub_id, target.space, target.smoothing_mm
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            functional=target.functional,
            temporal_masks=target.temporal_masks,
            domain=target.domain,
            space=target.space,
            smoothing_mm=target.smoothing_mm,
            surface=surfaces,
            mask=mask,
            mask_threshold=float(config.get("mask_threshold", 0.5)),
            volume_connectivity=int(config.get("volume_connectivity", 6)),
        ),
        output=OutputConfig(
            directory=output_base / target_name / sub_id,
            work_directory=work_base / target_name / sub_id,
            prefix=target_prefix,
            overwrite=config["overwrite"] if overwrite is None else overwrite,
        ),
        coarsening=CoarseningConfig(**config["coarsening"]),
        connectivity=ConnectivityConfig(**config["connectivity"]),
        quality=QualityConfig(**config["quality"]),
    )
    return target, cfg


def build_parser() -> argparse.ArgumentParser:
    """Build the direct microparcellation-module parser."""
    parser = argparse.ArgumentParser(
        "Compute surface or gray-matter volume microparcels and their dense connectivity"
    )
    parser.add_argument("-p", "--participant", required=True, help="BIDS participant ID")
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("-s", "--space", default=DEFAULT_SPACE)
    parser.add_argument("-S", "--smoothing", type=int, default=DEFAULT_SMOOTHING_MM, metavar="MM")
    parser.add_argument("-w", "--workflow", default="main")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None, *, execution_context: ExecutionContext | None = None):
    """Resolve selected inputs and execute microparcellation in its owner's tree."""
    args = build_parser().parse_args(argv)
    if execution_context is not None and execution_context.project != args.project:
        raise ValueError("Microparcellation project differs from its execution context")
    if args.smoothing < 0:
        raise SystemExit("--smoothing must be a nonnegative integer FWHM in mm")
    smoothing_mm = args.smoothing
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    participant = args.participant.replace("sub-", "")
    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        derivative_class="microparcellation",
        execution_context=execution_context,
    )
    microparcellation_id, config = load_runtime_configuration(runtime_config, "microparcellation")
    snapshot = load_runtime_workflow_snapshot(runtime_config)
    participant_id = f"sub-{participant}"
    source_subject = (
        Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
        / args.project
        / participant_id
    )
    source_runs = discover_raw_runs(source_subject)
    selected_runs = tuple(
        run for run in source_runs if matches_filter(run.entities, config.get("input_filter"))
    )
    preprocessing_config = snapshot["configurations"]["preprocessing"]["resolved"]
    output_spaces = tuple(str(value) for value in preprocessing_config["func"]["output_spaces"])
    if args.space not in output_spaces:
        raise SystemExit(
            f"space-{args.space} is not published by preprocessing; "
            f"choose from {', '.join(output_spaces)}"
        )
    clean_target = expected_clean_target(
        selected_runs,
        space=args.space,
        smoothing_mm=smoothing_mm,
        project=args.project,
        clean_id=str(snapshot["configurations"]["clean"]["directory"]),
        execution_context=execution_context,
    )
    target, cfg = make_target_config(
        args.project,
        participant,
        microparcellation_id,
        config,
        overwrite=True if args.overwrite else None,
        clean_target=clean_target,
        execution_context=execution_context,
    )
    stderr(
        f"Planned {target.domain} space-{target.space} "
        f"smoothing-{target.smoothing_mm}mm "
        "microparcellation target\n"
    )
    runner = Runner(
        module_name="Subject Microparcellation Module",
        container=None,
        binds=(),
        logger=logging.getLogger("microparcellation"),
        next_step=count(1).__next__,
        execution_context=execution_context,
    )
    result = build_module(cfg, runner, completion_boundary=False)
    manifest = Path(result["manifest"])
    publication_index = output_paths(cfg.output.directory, cfg.output.prefix)["index"]
    payload = {
        "manifest_version": 1,
        "module": "microparcellation",
        "participant": participant_id,
        "domain": target.domain,
        "space": target.space,
        "smoothing_fwhm_mm": target.smoothing_mm,
        "source_runs": [source.stem for source in selected_runs],
        "target_manifest": str(manifest),
        "public_outputs": [str(path) for value in result.values() for path in flatten_paths(value)],
        "configuration_fingerprint": selected_configuration_fingerprint(),
        "complete": True,
    }

    def validate_index() -> tuple[bool, str]:
        try:
            current = json.loads(publication_index.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False, "Microparcellation publication index is missing or unreadable."
        if current != payload:
            return False, "Microparcellation publication index differs from the requested module."
        return True, "Microparcellation publication index is complete and current."

    runner.add_step(
        Step.python(
            name="Write Microparcellation Publication Index",
            outputs=(publication_index,),
            inputs=(manifest,),
            force=bool(args.overwrite),
            action=lambda: write_json_atomic(publication_index, payload),
            validate=validate_index,
            completion_boundary=True,
        )
    )
    started = time.perf_counter()
    stderr(
        f"Running {target.domain} space-{target.space} "
        f"smoothing-{target.smoothing_mm}mm "
        f"microparcellation using {len(target.functional)} functional run(s)\n"
    )
    with runner.run_context(started_at=started):
        runner.execute()
    for name, path in result.items():
        print(
            f"{target.domain}/space-{target.space}/smoothing-{target.smoothing_mm}mm/{name}: {path}"
        )


if __name__ == "__main__":
    main()
