"""Show task-oriented guidance and current command references."""

from __future__ import annotations

import argparse
import textwrap

TASK_GUIDES = {
    "getting-started": """
        GETTING STARTED

        Check the selected installation and site:

          nro --version
          nro doctor
          nro paths show

        Inspect registered work before requesting anything:

          nro status -P PROJECT -p PARTICIPANT

        Omitted selectors often mean all matches. Specify a project and participant
        until you are familiar with the site's data.
    """,
    "processing": """
        REQUEST AND MONITOR PROCESSING

        Request an endpoint; nro includes its upstream dependencies:

          nro run -P PROJECT -p PARTICIPANT -m MODULE

        Inspect work and read its logs:

          nro status -P PROJECT -p PARTICIPANT
          nro log -P PROJECT -p PARTICIPANT -m MODULE

        Resume existing demand after a stop or error:

          nro run --resume -P PROJECT -p PARTICIPANT

        New requests default to workflow `main` when `--workflow` is omitted.
        Resume treats an omitted workflow as all matching resumable workflows.

        Stop selected demand:

          nro stop -P PROJECT -p PARTICIPANT -m MODULE

        Use `nro help --command run` for the current selectors and resource options.
    """,
    "definitions": """
        MANAGE DEFINITIONS

        Edit or create a typed definition, list definitions, or remove one:

          nro def edit config MODULE ID
          nro def edit workflow ID
          nro def ls config MODULE
          nro def rm markup ID

        Initialize or check a complete definitions store:

          nro def init [PATH]
          nro def migrate [PATH]
          nro def validate [PATH]

        Use `nro def edit file RELATIVE` for definition types without a typed
        editor. Interactive edits publish when the editor writes the draft.
    """,
    "inspection": """
        INSPECT WORK

        Read the cached registry state quickly:

          nro status -P PROJECT -p PARTICIPANT

        Revalidate artifacts and update the registry first:

          nro status --update -P PROJECT -p PARTICIPANT

        Find source BIDS images without consulting the registry:

          nro find -P PROJECT -p PARTICIPANT -r task='Rest.*'

        Open matching work-item logs or worker logs:

          nro log -P PROJECT -p PARTICIPANT -m MODULE
          nro log --worker

        Add `--json` to status when another program will consume the report.
    """,
    "viewing": """
        VIEW AND RENDER DERIVATIVES

        Build one scene from all matching derivatives and open it:

          nro scene -P PROJECT -p PARTICIPANT -m MODULE --open

        Copy scene inputs into a portable directory:

          nro scene -P PROJECT -p PARTICIPANT -m MODULE --publish

        Render matching maps without an interactive viewer:

          nro render -P PROJECT -p PARTICIPANT -m MODULE
    """,
    "bidsification": """
        PREPARE BIDS DATA

        List or continue sessions from a configured Flywheel source:

          nro bidsify -P PROJECT

        Continue one known request:

          nro bidsify --request REQUEST_ID

        Bidsification stages data before publication. Reinvoke the command when
        status reports that a request needs input or approval.
    """,
    "development": """
        WORK ON A DEVELOPMENT BRANCH

        Inspect the current branch registration:

          nro branch show

        Select a private definitions store or restore shared definitions:

          nro branch definitions --definitions PATH
          nro branch definitions --shared

        Run checks for the affected development sphere:

          nro dev test --sphere SPHERE

        Use `nro help --command branch` for registration and lifecycle options.
    """,
    "maintenance": """
        MAINTAIN CONTROLLED DATA

        Preview deletion interactively:

          nro purge -P PROJECT -p PARTICIPANT -m MODULE

        Remove unclaimed public and WORK derivative files while preserving
        registered work:

          nro gc -P PROJECT -p PARTICIPANT -m MODULE

        Rebuild registry knowledge from existing controlled artifacts:

          nro run --repair

        Shared installation changes belong in the tagged main checkout and use
        `./install --maintain`. Purge and repair can affect broad scopes when
        selectors are omitted.
    """,
}


def build_parser(*, prog: str = "nro help") -> argparse.ArgumentParser:
    """Build the task-guide parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Show task guidance or help from an installed command's current parser.",
    )
    parser.add_argument("topic", nargs="?", help="Task topic or installed command")
    parser.add_argument(
        "--command",
        metavar="COMMAND",
        help="Show the current parser help for an installed command",
    )
    return parser


def _show_command(command: str, parser: argparse.ArgumentParser) -> None:
    from nro.cli import _command_main, available_commands

    if command not in available_commands():
        parser.error(f"unknown command {command!r}")
    try:
        _command_main(command)(["--help"], prog=f"nro {command}")
    except SystemExit as error:
        if error.code not in {None, 0}:
            raise


def _show_index() -> None:
    from nro.cli import COMMAND_HELP, available_commands

    print("Task guides:")
    for topic in TASK_GUIDES:
        print(f"  {topic}")
    print("\nInstalled commands:")
    for command in available_commands():
        print(f"  {command:<12} {COMMAND_HELP[command]}")
    print("\nUse `nro help TOPIC` for a guide or `nro help --command COMMAND` for syntax.")


def main(argv=None, *, prog: str = "nro help") -> None:
    """Print an index, a task guide, or live help for one command."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if args.topic and args.command:
        parser.error("choose a task topic or --command, not both")
    if args.command:
        _show_command(args.command, parser)
        return
    if args.topic in TASK_GUIDES:
        print(textwrap.dedent(TASK_GUIDES[args.topic]).strip())
        return
    if args.topic:
        _show_command(args.topic, parser)
        return
    _show_index()


if __name__ == "__main__":
    main()
