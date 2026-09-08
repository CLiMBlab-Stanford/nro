"""Fit one registered task/model for a participant and spatial target."""

import argparse
import logging
import json
from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.configuration.runtime import load_runtime_configuration
from nro.engine.bids import discover_raw_runs
from nro.engine.targets import DEFAULT_SPACE, DEFAULT_SMOOTHING_MM
from nro.orchestration.runtime import select_runtime_config
from .task_models import load_task_model, validate_task_model, model_path
from .planning import selected_runs
from .module import run_module


def main(argv: list[str] | None = None) -> None:
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
    model_path(args.model)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    participant = args.participant.removeprefix("sub-")
    runtime = select_runtime_config(project=args.project, workflow_id=args.workflow, derivative_class="firstlevels")
    config_id, config = load_runtime_configuration(runtime, "firstlevels")
    model = validate_task_model(json.loads(args.model_definition)) if args.model_definition else load_task_model(args.model)
    project_root = Path(BIDS_PATH) / args.project
    runs = selected_runs(discover_raw_runs(project_root / f"sub-{participant}"), args.model, participant)
    run_module(runs=runs, participant=participant, project_root=project_root,
               preprocessing_id=config["preprocessing_directory"], config_id=config_id,
               model_id=args.model, model=model, config=config, space=args.space, smoothing=args.smoothing,
               work_root=Path(WORK_PATH) / args.project / "derivatives" / "firstlevels" / config_id,
               definition_inputs=(runtime,))


if __name__ == "__main__":
    main()
