"""List, inspect, validate, register or compile YAML task models."""

import argparse
import json
from pathlib import Path

import yaml

from nro.configuration.store import ConfigStore
from nro.modules.firstlevels.compiler import compile_model
from nro.modules.firstlevels.task_models import (
    load_task_model,
    register_model,
    select_models,
    validate_task_model,
)


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    """Manage task YAML without touching the registry or submitting work."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("show").add_argument("model")
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("model")
    compile_command.add_argument("--config", default="main", help="Firstlevels configuration ID")
    register = commands.add_parser("register")
    register.add_argument("model")
    register.add_argument("file", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("task")
    validate.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for identifier, model in select_models(model_sets=()).items():
                print(f"{identifier}\t{','.join(model['model_set']) or '-'}")
        elif args.command == "show":
            print(yaml.safe_dump(load_task_model(args.model), sort_keys=False), end="")
        elif args.command == "compile":
            config = ConfigStore().load_configuration("firstlevels", args.config).values
            print(
                json.dumps(compile_model(load_task_model(args.model), args.model, config), indent=2)
            )
        elif args.command == "register":
            print(register_model(args.model, args.file))
        else:
            from nro.modules.firstlevels.task_models import model_path

            model_path(f"{args.task}/validation")
            validate_task_model(yaml.safe_load(args.file.read_text()))
            print("Model is supported")
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
