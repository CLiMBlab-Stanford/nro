"""Run one functional BIDS run through the preprocessing workflow."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

from nro.configuration.paths import BIDS_PATH
from nro.configuration.runtime import configure_preprocessing, load_runtime_configuration
from nro.engine.bids import (
    discover_raw_runs,
    parse_selectors,
    resolve_run,
    strip_bids_prefix,
)
from nro.orchestration.runtime import select_runtime_config


def build_parser() -> argparse.ArgumentParser:
    """Build the direct functional-module parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument(
        "-r",
        "--run",
        nargs="+",
        metavar="ENTITY=VALUE",
        help="BIDS entity selectors sufficient to identify one BOLD run (for example task=rest dir=LR)",
    )
    parser.add_argument("-w", "--workflow", default="main")
    return parser


def _run_with_error_logging(run_func: Callable[[Sequence[str]], object], argv: list[str]) -> object:
    """Ensure fatal module exits are visibly marked in scheduler logs."""
    try:
        return run_func(argv)
    except SystemExit as error:
        if error.code in (None, 0):
            raise
        message = str(error.code).strip() if error.code is not None else "unknown fatal error"
        logging.getLogger("preprocess").error("FATAL: %s", message)
        raise SystemExit(error.code if isinstance(error.code, int) else 1) from error
    except Exception:
        logging.getLogger("preprocess").exception("FATAL: unhandled preprocessing exception")
        raise


def main(argv: list[str] | None = None, *, execution_context=None) -> None:
    """Resolve one functional instance and execute its runner graph."""
    args = build_parser().parse_args(argv)
    if execution_context is not None and execution_context.project != args.project:
        raise ValueError("Execution context project differs from the requested project")
    participant = strip_bids_prefix(args.participant, "sub")
    sub_id = f"sub-{participant}"
    try:
        selectors = parse_selectors(args.run)
        selected = resolve_run(
            discover_raw_runs(
                (execution_context.paths.bids if execution_context else Path(BIDS_PATH))
                / args.project
                / sub_id
            ),
            selectors,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    ses_id = f"ses-{selected.session}" if selected.session else None
    run_stem = selected.stem

    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        derivative_class="preprocessing",
        execution_context=execution_context,
    )
    preprocessing_id, cfg = load_runtime_configuration(runtime_config, "preprocessing")
    configure_preprocessing(args.project, preprocessing_id, cfg)

    from nro.modules.func.module import main as run_func

    module_argv = [
        "--run-stem",
        run_stem,
        "--project",
        args.project,
        "--preprocessing-id",
        preprocessing_id,
        "--sub-id",
        sub_id,
        "--nthreads",
        str(cfg["func"]["nthreads"]),
    ]
    if ses_id:
        module_argv.extend(("--ses-id", ses_id))
    implementation = (
        partial(run_func, execution_context=execution_context) if execution_context else run_func
    )
    _run_with_error_logging(implementation, module_argv)


if __name__ == "__main__":
    main()
