"""Command-line entry point for one dynamic-connectivity module instance."""

from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import count
from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.configuration.runtime import load_runtime_configuration
from nro.engine.bids import discover_raw_runs, matches_filter
from nro.engine.clean_targets import CleanTarget, expected_clean_target
from nro.engine.cli import stderr
from nro.engine.io import flatten_paths
from nro.engine.publication import write_json_atomic
from nro.engine.targets import (
    DEFAULT_SMOOTHING_MM,
    DEFAULT_SPACE,
    target_output_names,
)
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step
from nro.orchestration.runtime import (
    load_runtime_workflow_snapshot,
    select_runtime_config,
    selected_configuration_fingerprint,
)

from .config import InclusionConfig, InputsConfig, LowRankConfig, ModuleConfig, OutputConfig
from .module import build_module
from .paths import output_paths


def make_target_config(
    project: str,
    participant: str,
    dynconn_id: str,
    config: dict,
    *,
    clean_target: CleanTarget,
    overwrite: bool | None = None,
    execution_context: ExecutionContext | None = None,
) -> ModuleConfig:
    """Resolve one selected target into an executable module configuration."""

    sub_id = f"sub-{participant}"
    bids_root = Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
    output_base = Path(
        config.get("output_dir") or bids_root / project / "derivatives" / "dynconn" / dynconn_id
    )
    work_base = (
        Path(WORK_PATH if execution_context is None else execution_context.paths.work)
        / project
        / "derivatives"
        / "dynconn"
        / dynconn_id
    )
    if execution_context is not None:
        output_base = execution_context.output_path(output_base)
        work_base = execution_context.output_path(work_base, private=True)
    target, prefix = target_output_names(
        config.get("prefix") or sub_id,
        clean_target.space,
        clean_target.smoothing_mm,
    )
    return ModuleConfig(
        inputs=InputsConfig(
            functional=clean_target.functional,
            temporal_masks=clean_target.temporal_masks,
            domain=clean_target.domain,
            space=clean_target.space,
            smoothing_mm=clean_target.smoothing_mm,
        ),
        output=OutputConfig(
            directory=output_base / sub_id,
            work_directory=work_base / target / sub_id,
            prefix=prefix,
            overwrite=config["overwrite"] if overwrite is None else overwrite,
        ),
        inclusion=InclusionConfig(**config["inclusion"]),
        low_rank=config["low_rank"],
        low_rank_options=LowRankConfig(**config["low_rank_options"]),
        weighting=config["weighting"],
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the standalone module parser used by orchestration workers."""

    parser = argparse.ArgumentParser(
        description="Package cleaned time series for Workbench dynamic connectivity."
    )
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("-s", "--space", default=DEFAULT_SPACE)
    parser.add_argument("-S", "--smoothing", type=int, default=DEFAULT_SMOOTHING_MM, metavar="MM")
    parser.add_argument("-w", "--workflow", default="main")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    execution_context: ExecutionContext | None = None,
) -> None:
    """Select cleaned runs and execute one dynamic-connectivity instance."""

    args = build_parser().parse_args(argv)
    if args.smoothing < 0:
        raise SystemExit("--smoothing must be a nonnegative integer FWHM in mm")
    participant = args.participant.removeprefix("sub-")
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        derivative_class="dynconn",
        execution_context=execution_context,
    )
    dynconn_id, config = load_runtime_configuration(runtime_config, "dynconn")
    snapshot = load_runtime_workflow_snapshot(runtime_config)
    source_subject = (
        Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
        / args.project
        / f"sub-{participant}"
    )
    runs = tuple(
        run
        for run in discover_raw_runs(source_subject)
        if matches_filter(run.entities, config.get("input_filter"))
    )
    spaces = tuple(
        str(value)
        for value in snapshot["configurations"]["preprocessing"]["resolved"]["func"][
            "output_spaces"
        ]
    )
    if args.space not in spaces:
        raise SystemExit(
            f"space-{args.space} is not published by preprocessing; choose from {', '.join(spaces)}"
        )
    target = expected_clean_target(
        runs,
        space=args.space,
        smoothing_mm=args.smoothing,
        project=args.project,
        clean_id=str(snapshot["configurations"]["clean"]["directory"]),
        execution_context=execution_context,
    )
    cfg = make_target_config(
        args.project,
        participant,
        dynconn_id,
        config,
        clean_target=target,
        overwrite=True if args.overwrite else None,
        execution_context=execution_context,
    )
    stderr(
        f"Planned {target.domain} space-{target.space} "
        f"smoothing-{target.smoothing_mm}mm dynamic-connectivity target\n"
    )
    runner = Runner(
        module_name="Subject Dynamic-Connectivity Module",
        container=None,
        binds=(),
        logger=logging.getLogger("dynconn"),
        next_step=count(1).__next__,
        execution_context=execution_context,
    )
    result = build_module(cfg, runner, completion_boundary=False)
    index = output_paths(cfg.output.directory, cfg.output.prefix, target.domain)["index"]
    manifest = Path(result["manifest"])
    payload = {
        "manifest_version": 1,
        "module": "dynconn",
        "participant": f"sub-{participant}",
        "domain": target.domain,
        "space": target.space,
        "smoothing_fwhm_mm": target.smoothing_mm,
        "source_runs": [run.stem for run in runs],
        "target_manifest": str(manifest),
        "public_outputs": [str(path) for value in result.values() for path in flatten_paths(value)],
        "configuration_fingerprint": selected_configuration_fingerprint(),
        "complete": True,
    }

    def validate_index() -> tuple[bool, str]:
        try:
            matches = json.loads(index.read_text(encoding="utf-8")) == payload
        except (OSError, TypeError, ValueError):
            matches = False
        return matches, (
            "Dynamic-connectivity publication index is current."
            if matches
            else "Dynamic-connectivity publication index is missing or outdated."
        )

    runner.add_step(
        Step.python(
            name="Write Dynamic-Connectivity Publication Index",
            inputs=(manifest,),
            outputs=(index,),
            force=bool(args.overwrite),
            action=lambda: write_json_atomic(index, payload),
            validate=validate_index,
            completion_boundary=True,
        )
    )
    started = time.perf_counter()
    stderr(
        f"Running {target.domain} space-{target.space} "
        f"smoothing-{target.smoothing_mm}mm dynamic connectivity using "
        f"{len(target.functional)} run(s)\n"
    )
    with runner.run_context(started_at=started):
        runner.execute()
    for name, path in result.items():
        print(
            f"{target.domain}/space-{target.space}/smoothing-{target.smoothing_mm}mm/{name}: {path}"
        )


if __name__ == "__main__":
    main()
