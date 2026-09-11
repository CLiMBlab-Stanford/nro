"""Fit one registered task/model for a participant and spatial target."""

import argparse
import json
import logging
from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.configuration.runtime import load_runtime_configuration
from nro.engine.bids import discover_raw_runs
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runtime import select_runtime_config

from .module import run_module
from .planning import selected_runs
from .task_models import load_task_model, model_path, validate_task_model


def main(
    argv: list[str] | None = None, *, execution_context: ExecutionContext | None = None
) -> None:
    """Resolve a workflow snapshot and execute one complete firstlevels instance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("--model", required=True, metavar="TASK/VARIANT")
    parser.add_argument("--model-definition", help=argparse.SUPPRESS)
    parser.add_argument("-s", "--space", default=DEFAULT_SPACE)
    parser.add_argument("-S", "--smoothing", type=int, default=DEFAULT_SMOOTHING_MM)
    parser.add_argument("-w", "--workflow", default="main")
    args = parser.parse_args(argv)
    if execution_context is not None and execution_context.project != args.project:
        raise ValueError("Firstlevels project differs from its execution context")
    model_path(args.model)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    participant = args.participant.removeprefix("sub-")
    runtime = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        derivative_class="firstlevels",
        execution_context=execution_context,
    )
    config_id, config = load_runtime_configuration(runtime, "firstlevels")
    model = (
        validate_task_model(json.loads(args.model_definition))
        if args.model_definition
        else load_task_model(args.model)
    )
    project_root = (
        Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
        / args.project
    )
    runs = selected_runs(
        discover_raw_runs(project_root / f"sub-{participant}"), args.model, participant
    )
    run_module(
        runs=runs,
        participant=participant,
        project_root=project_root,
        preprocessing_id=config["preprocessing_directory"],
        config_id=config_id,
        model_id=args.model,
        model=model,
        config=config,
        space=args.space,
        smoothing=args.smoothing,
        work_root=Path(WORK_PATH if execution_context is None else execution_context.paths.work)
        / args.project
        / "derivatives"
        / "firstlevels"
        / config_id,
        execution_context=execution_context,
    )


if __name__ == "__main__":
    main()
