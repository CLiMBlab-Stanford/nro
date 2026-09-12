"""Shared authoring interface for scientific configs, workflows, and task models."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from nro.configuration.parsing import parse_mapping
from nro.configuration.site import bids_root as _configured_bids_root
from nro.configuration.store import DERIVATIVE_CLASSES, ConfigStore, validate_config_id
from nro.engine.definition_editor import (
    delete_definition,
    read_definition,
    review_definition,
    save_definition,
)
from nro.modules.firstlevels.authoring import discover_event_files, model_draft
from nro.modules.firstlevels.task_models import model_path, validate_task_model


@dataclass(frozen=True)
class DefinitionTarget:
    """A validated store identifier and its publication path."""

    kind: str
    identifier: str
    path: Path
    derivative_class: str | None = None


def definition_target(store: ConfigStore, kind: str, identifier: str) -> DefinitionTarget:
    """Resolve an authoring ID without requiring an existing file."""
    if kind == "model":
        identifier = identifier if "/" in identifier else f"{identifier}/main"
        return DefinitionTarget(kind, identifier, model_path(identifier, store.root / "models"))
    if kind == "workflow":
        identifier = validate_config_id(identifier, kind="workflow")
        return DefinitionTarget(
            kind, identifier, store.root / "workflows" / f"{identifier}_workflow.yml"
        )
    if kind != "config" or len(identifier.split("/")) != 2:
        raise ValueError("Config IDs must be CLASS/ID, for example clean/alternative")
    derivative_class, config_id = identifier.split("/")
    if derivative_class not in DERIVATIVE_CLASSES:
        raise ValueError(f"Choose a configuration class from {', '.join(DERIVATIVE_CLASSES)}")
    config_id = validate_config_id(config_id, kind="configuration")
    return DefinitionTarget(
        kind,
        f"{derivative_class}/{config_id}",
        store.configs / derivative_class / f"{config_id}_{derivative_class}.yml",
        derivative_class,
    )


def validate_definition(store: ConfigStore, target: DefinitionTarget, text: str) -> None:
    """Validate staged YAML without changing the store or registry.

    Use the same compiler as ordinary loading for configs and workflows.
    Data-dependent estimability and scientific suitability require
    separate review and are not established by this validation.
    """
    try:
        value = parse_mapping(text, source=str(target.path))
        if target.kind == "model":
            validate_task_model(value)
        elif target.kind == "workflow":
            store.resolve(target.identifier, document=value)
        else:
            config_id = target.identifier.split("/")[1]
            store.load_configuration(target.derivative_class, config_id, document=value)
    except (yaml.YAMLError, OSError, TypeError, KeyError) as error:
        raise ValueError(str(error)) from error


def _choose_column(columns: tuple[str, ...]) -> str:
    print("Possible condition columns: " + ", ".join(columns))
    return input("Condition column: ").strip()


def _draft(store: ConfigStore, target: DefinitionTarget, args: argparse.Namespace) -> str:
    if args.source:
        source = definition_target(store, target.kind, args.source)
        if source.derivative_class != target.derivative_class:
            raise ValueError("--from must use the same configuration class")
        text = source.path.read_text(encoding="utf-8")
        validate_definition(store, source, text)
        if target.kind == "model":
            value = parse_mapping(text, source=str(source.path))
            value["model_set"] = []
            return yaml.safe_dump(value, sort_keys=False)
        if target.kind == "workflow":
            return yaml.safe_dump(store.resolve(source.identifier).selections, sort_keys=False)
        if target.kind != "config" or source.identifier.split("/")[1] != "main":
            return text
    if target.kind == "config":
        defaults = store.configuration_path(target.derivative_class, "main").read_text()
        return (
            "# Add overrides below. Omitted settings follow this class's main config.\n"
            "\n# Current main defaults (reference only):\n"
            + "".join(f"# {line}\n" for line in defaults.splitlines())
        )
    if target.kind == "workflow":
        return yaml.safe_dump({name: "main" for name in DERIVATIVE_CLASSES}, sort_keys=False)
    paths = args.events or discover_event_files(
        target.identifier.split("/")[0],
        _configured_bids_root(),
        projects=args.project or (),
        participants=args.participant or (),
    )
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    return model_draft(
        paths, conditions=args.conditions, choose=_choose_column if interactive else None
    )


def build_parser(action: str, *, prog: str) -> argparse.ArgumentParser:
    """Build definition-management parsers with object-specific options."""
    parser = argparse.ArgumentParser(
        prog=prog, description=f"{action.capitalize()} a model, config, or workflow definition."
    )
    commands = parser.add_subparsers(dest="kind", required=True)
    for kind, metavar in (("model", "TASK[/VARIANT]"), ("config", "CLASS/ID"), ("workflow", "ID")):
        command = commands.add_parser(kind)
        command.add_argument("identifier", metavar=metavar)
        if action != "delete":
            command.add_argument(
                "--file", type=Path, help="Use a local YAML file instead of an editor"
            )
        command.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help=(
                "Delete without confirmation"
                if action == "delete"
                else "Save without confirmation; noninteractive use also requires --file"
            ),
        )
        if action == "create":
            command.add_argument(
                "--from", dest="source", help="Copy an existing definition of the same kind"
            )
            command.add_argument(
                "--output", type=Path, help="Write a local draft without registration or an editor"
            )
            if kind == "model":
                command.add_argument("-P", "--project", nargs="+", action="extend")
                command.add_argument("-p", "--participant", nargs="+", action="extend")
                command.add_argument(
                    "--events", nargs="+", type=Path, help="Infer a model from these event files"
                )
                command.add_argument(
                    "--conditions", help="Event column to use as categorical conditions"
                )
    return parser


def _delete(store: ConfigStore, target: DefinitionTarget, expected: bytes, *, yes: bool) -> None:
    print(f"Delete {target.kind} {target.identifier}: {target.path}")
    if target.kind == "config":
        config_id = target.identifier.split("/")[1]
        if config_id == "main":
            print("The class will inherit its packaged main configuration after deletion.")
        else:
            references = []
            for path in sorted((store.root / "workflows").glob("*_workflow.yml")):
                try:
                    value = parse_mapping(path.read_text(encoding="utf-8"), source=str(path))
                except (ValueError, yaml.YAMLError) as error:
                    raise ValueError(
                        f"Cannot check workflow references in {path}: {error}"
                    ) from error
                if value.get(target.derivative_class, "main") == config_id:
                    references.append(path.name.removesuffix("_workflow.yml"))
            if references:
                print(
                    "Warning: these workflows will not resolve until the config is restored "
                    "or their selections change: " + ", ".join(references)
                )
    if target.kind == "workflow" and target.identifier == "main":
        print("Warning: default requests will need this workflow to be recreated.")
    print(
        "Only this definition will be removed. Derivatives, logs, registry records, and workers are unchanged."
    )
    print("Deleting a shared definition may affect later requests and artifact assessments.")
    if not yes:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise ValueError("Deletion requires interactive confirmation or --yes")
        if input("Delete this definition? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Cancelled; the stored definition is unchanged.")
            return
    backup = delete_definition(target.path, expected=expected)
    print(
        f"Deleted {target.path}\nRecovery copy: {backup} (temporary; copy elsewhere to retain it)"
    )


def main(action: str, argv: list[str] | None = None, *, prog: str) -> None:
    """Manage a definition without changing scientific work or registry state."""
    parser = build_parser(action, prog=prog)
    args = parser.parse_args(argv)
    try:
        if getattr(args, "file", None):
            args.file = args.file.expanduser()
        store = ConfigStore()
        target = definition_target(store, args.kind, args.identifier)
        if not target.path.resolve().is_relative_to(store.root):
            raise ValueError("Definition path escapes the central store through a symbolic link")
        expected = read_definition(target.path)
        if action in {"edit", "delete"} and expected is None:
            raise ValueError(f"Definition does not exist: {target.path}; use nro create")
        if action == "delete":
            _delete(store, target, expected, yes=args.yes)
            return
        if action == "create":
            discovery = any(
                getattr(args, name, None)
                for name in ("events", "conditions", "project", "participant")
            )
            if args.source and args.file:
                raise ValueError("Use either --from or --file")
            if (args.source or args.file) and discovery:
                raise ValueError("Event discovery options cannot be combined with --from or --file")
            if getattr(args, "events", None) and any(
                getattr(args, name, None) for name in ("project", "participant")
            ):
                raise ValueError("Use --events or BIDS discovery selectors, not both")
            if args.output and (args.file or args.yes):
                raise ValueError("--output cannot be combined with --file or --yes")
            if expected is not None:
                if args.source or args.file or args.output or discovery:
                    raise ValueError(
                        "Definition already exists; creation-only options do not apply. Use nro edit or a new ID"
                    )
                if not (sys.stdin.isatty() and sys.stdout.isatty()):
                    raise ValueError(
                        "Definition already exists; use nro edit in noninteractive mode"
                    )
                print(
                    f"{args.kind.capitalize()} already exists; opening for editing: {target.path}"
                )
            if getattr(args, "events", None):
                args.events = [path.expanduser() for path in args.events]
        if expected is not None:
            initial = expected.decode("utf-8")
        elif args.file:
            initial = ""
        else:
            initial = _draft(store, target, args)

        def validate(text: str) -> None:
            validate_definition(store, target, text)

        if action == "create" and args.output:
            validate(initial)
            output = args.output.expanduser().absolute()
            if output.resolve().is_relative_to(store.root):
                raise ValueError(
                    "--output must be outside the central store; omit it to register a definition"
                )
            save_definition(output, initial, expected=None)
            print(f"Draft written to {output}; not registered.")
            return
        review_definition(
            target.path,
            initial,
            expected=expected,
            validate=validate,
            source=args.file,
            yes=args.yes,
        )
    except (KeyboardInterrupt, EOFError):
        print("Cancelled; no definition was changed.", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, yaml.YAMLError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{prog}: {error}\n")
