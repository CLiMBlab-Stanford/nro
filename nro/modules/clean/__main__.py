"""Clean one preprocessed functional run for connectivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.configuration.runtime import configure_clean, load_runtime_configuration
from nro.engine.bids import (
    discover_raw_runs,
    parse_selectors,
    resolve_run,
    strip_bids_prefix,
)
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE
from nro.orchestration.runtime import load_runtime_workflow_snapshot, select_runtime_config


def build_parser() -> argparse.ArgumentParser:
    """Build the direct cleaning-module parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("-s", "--space", default=DEFAULT_SPACE)
    parser.add_argument("-S", "--smoothing", type=int, default=DEFAULT_SMOOTHING_MM, metavar="MM")
    parser.add_argument(
        "-r",
        "--run",
        nargs="+",
        metavar="ENTITY=VALUE",
        help="BIDS entity selectors sufficient to identify one preprocessed BOLD run",
    )
    parser.add_argument("-w", "--workflow", default="main")
    return parser


def main(argv: list[str] | None = None, *, execution_context=None) -> None:
    """Resolve one cleaning instance and execute its runner graph."""
    args = build_parser().parse_args(argv)
    if execution_context is not None and execution_context.project != args.project:
        raise ValueError("Execution context project differs from the requested project")
    if args.smoothing < 0:
        raise SystemExit("--smoothing must be a nonnegative integer FWHM in mm")
    participant = strip_bids_prefix(args.participant, "sub")
    sub_id = f"sub-{participant}"

    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        derivative_class="clean",
        execution_context=execution_context,
    )
    workflow_snapshot = load_runtime_workflow_snapshot(runtime_config)
    preprocessing_func = workflow_snapshot["configurations"]["preprocessing"]["resolved"]["func"]
    if args.space not in preprocessing_func["output_spaces"]:
        available = ", ".join(str(value) for value in preprocessing_func["output_spaces"])
        raise SystemExit(
            f"space-{args.space} is not published by preprocessing; choose from {available}"
        )
    clean_id, cfg = load_runtime_configuration(runtime_config, "clean")
    configure_clean(args.project, clean_id, cfg)
    source_subject = (
        (execution_context.paths.bids if execution_context else Path(BIDS_PATH))
        / args.project
        / sub_id
    )
    try:
        selectors = parse_selectors(args.run)
        selected = resolve_run(
            discover_raw_runs(source_subject),
            selectors,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    ses_id = f"ses-{selected.session}" if selected.session else None
    run_stem = selected.stem

    from nro.modules.clean.module import main as run_clean

    module_argv = [
        "--project",
        args.project,
        "--preprocessing-id",
        cfg["preprocessing_directory"],
        "--clean-id",
        clean_id,
        "--sub-id",
        sub_id,
        "--run-stem",
        run_stem,
        "--space",
        str(args.space),
        "--smoothing",
        str(args.smoothing),
    ]
    if preprocessing_func["clean_ica_aroma"]:
        module_argv.append("--functional-ica-aroma")
    if ses_id:
        module_argv.extend(("--ses-id", ses_id))
    return_code = run_clean(
        module_argv, **({"execution_context": execution_context} if execution_context else {})
    )
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
