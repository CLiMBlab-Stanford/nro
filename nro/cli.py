"""Installed command dispatcher for nro's user-facing executables."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from collections.abc import Callable

import nro.bin

COMMAND_HELP = {
    "release": "Inspect installed releases or perform legacy release maintenance.",
    "cutover": "Convert the private-control layout during a maintenance window.",
    "branch": "Register branches and locate their shared scientific registries.",
    "bidsify": "Download, review, convert, and approve BIDS sessions.",
    "create": "Create a model, config, or workflow definition.",
    "edit": "Edit an existing model, config, or workflow definition.",
    "delete": "Delete a definition without removing derivatives.",
    "definitions": "Create or validate an external definitions store.",
    "models": "Inspect, validate, register, and compile task models.",
    "setup": "Set up dependencies or connect to a shared installation.",
    "paths": "View and edit site paths.",
    "doctor": "Check dependencies and site access.",
    "log": "Browse worker or derivative-instance logs.",
    "publish": "Publish a completed request as a standalone derivative dataset.",
    "promote": "Accept equivalent development artifacts after an approved merge.",
    "purge": "Remove nro-controlled derivatives and logs.",
    "qc": "Run an ad hoc quality control.",
    "run": "Plan work and supply the shared worker pool.",
    "set": "Update live planner settings.",
    "status": "Report derivative status.",
    "stop": "Cancel derivative demand or stop workers.",
    "wb_view": "Open Workbench scenes stored with completed derivatives.",
}


def available_commands() -> tuple[str, ...]:
    """Return exactly the executable module names present in ``nro.bin``."""
    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules(nro.bin.__path__)
            if not module.name.startswith("_")
        )
    )


def build_parser(*, prog: str = "nro") -> argparse.ArgumentParser:
    """Construct the command dispatcher from public nro.bin executables."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run and inspect nro derivative workflows.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command in available_commands():
        commands.add_parser(
            command,
            add_help=False,
            help=COMMAND_HELP.get(command, f"Run the {command} command."),
        )
    return parser


def _command_main(command: str) -> Callable[..., None]:
    module = importlib.import_module(f"nro.bin.{command}")
    implementation = getattr(module, "main", None)
    if not callable(implementation):
        raise RuntimeError(f"nro.bin.{command} does not define a callable main()")
    return implementation


def main(argv: list[str] | None = None, *, prog: str = "nro") -> None:
    """Dispatch argv to a user command; unknown commands exit with a parser error."""
    values = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(prog=prog)
    if not values or values[0] in {"-h", "--help"}:
        parser.parse_args(values)
        return

    command = values.pop(0)
    if command not in available_commands():
        parser.error(f"unknown command {command!r}; choose from " + ", ".join(available_commands()))
    from nro.configuration.site import installation_record

    if installation_record().get("mode") == "branch" and command not in {
        "branch",
        "doctor",
        "paths",
        "setup",
        "definitions",
        "create",
        "edit",
        "delete",
        "models",
        "run",
        "status",
        "stop",
        "log",
        "set",
        "purge",
        "bidsify",
        "promote",
        "wb_view",
        "qc",
        "publish",
    }:
        if "-h" not in values and "--help" not in values:
            parser.error(
                f"{command} requires the central/main installation; a development installation does not grant maintenance authority"
            )
    _command_main(command)(values, prog=f"{prog} {command}")


if __name__ == "__main__":
    main()
