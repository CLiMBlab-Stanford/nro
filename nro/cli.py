"""Installed command dispatcher for nro's user-facing executables."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from collections.abc import Callable

import nro.bin


COMMAND_HELP = {
    "log": "Browse worker or derivative-instance logs.",
    "publish": "Publish a completed request as a standalone derivative dataset.",
    "purge": "Remove nro-controlled derivatives and logs.",
    "qc": "Run an ad hoc quality control.",
    "run": "Plan work and supply the shared worker pool.",
    "set": "Update live planner settings.",
    "status": "Report derivative status.",
    "stop": "Cancel derivative demand or stop workers.",
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
    values = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(prog=prog)
    if not values or values[0] in {"-h", "--help"}:
        parser.parse_args(values)
        return

    command = values.pop(0)
    if command not in available_commands():
        parser.error(
            f"unknown command {command!r}; choose from "
            + ", ".join(available_commands())
        )
    _command_main(command)(values, prog=f"{prog} {command}")


if __name__ == "__main__":
    main()
