"""View and edit protected site settings."""

import argparse
from pathlib import Path

from nro.configuration.site import settings, site_definition_path, site_file
from nro.engine.site_setup import edit_settings


def main(argv=None, *, prog="nro paths"):
    """Display or atomically update site settings, with confirmation for interactive edits.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("action", choices=("show", "set"), nargs="?")
    parser.add_argument("assignments", nargs="*")
    parser.add_argument("--maintain", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "show":
            values, sources = settings()
            print(f"Definitions locator: {site_file()}")
            print(f"Protected site definition: {site_definition_path(Path(values['definitions']))}")
            for key, value in sorted(values.items()):
                print(f"{key:16} {value!s:60} [{sources[key]}]")
        else:
            if args.action == "set" and not args.assignments:
                parser.error("set requires key=value pairs")
            edit_settings(args.assignments or None, maintain=args.maintain)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, "\nPath editing cancelled. Settings were not saved.\n")


if __name__ == "__main__":
    main()
