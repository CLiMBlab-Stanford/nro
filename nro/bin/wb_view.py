"""Run the site's Connectome Workbench viewer with unmodified arguments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DESCRIPTION = (
    "Run the wb_view executable beside the configured wb_command and pass every "
    "argument to it unchanged. The viewer runs on the current host and needs an X11 "
    "display. Run `nro wb_view -help` for Workbench's own syntax."
)


def build_parser(*, prog: str = "nro wb_view") -> argparse.ArgumentParser:
    """Construct the parser used only to describe this pass-through command."""
    parser = argparse.ArgumentParser(prog=prog, description=DESCRIPTION)
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help="Workbench files or options passed to wb_view",
    )
    return parser


def viewer_path() -> Path:
    """Return the wb_view path beside the site's configured wb_command."""
    from nro.site.paths import WB_COMMAND_PATH

    return WB_COMMAND_PATH.expanduser().with_name("wb_view")


def main(argv: list[str] | None = None, *, prog: str = "nro wb_view") -> None:
    """Replace this process with wb_view, or print help for a leading -h/--help.

    Raises SystemExit when the configured viewer is missing or not executable.
    """
    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] in (["-h"], ["--help"]):
        build_parser(prog=prog).print_help()
        return
    viewer = viewer_path()
    if not viewer.is_file() or not os.access(viewer, os.X_OK):
        raise SystemExit(
            f"Connectome Workbench viewer is not available: {viewer}. "
            "Install it by rerunning ./install without --without-viewer, or configure "
            "the site workbench path."
        )
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(viewer, [str(viewer), *values])


if __name__ == "__main__":
    main()
