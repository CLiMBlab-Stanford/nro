"""Update live lab-wide planner settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.registry import Registry


def build_parser(*, prog: str = "nro.bin.set") -> argparse.ArgumentParser:
    """Construct the set parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("assignments", nargs="+", metavar="NAME=VALUE")
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument("--json", action="store_true")
    return parser


def _parse_assignments(values: list[str]) -> dict[str, int]:
    settings: dict[str, int] = {}
    for assignment in values:
        name, separator, raw_value = assignment.partition("=")
        if not separator or not name or not raw_value:
            raise SystemExit(f"Invalid setting {assignment!r}; expected NAME=VALUE")
        if name != "concurrency":
            print(
                f"WARNING: ignoring unsupported registry setting: {name}",
                file=sys.stderr,
            )
            continue
        try:
            value = int(raw_value)
        except ValueError as error:
            raise SystemExit("concurrency must be an integer") from error
        if value < 1:
            raise SystemExit("concurrency must be at least one")
        settings[name] = value
    return settings


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.set") -> None:
    """Update recognized active-request settings; warn for unsupported keys.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    settings = _parse_assignments(args.assignments)
    if not settings:
        result = {"settings": {}, "updated_requests": 0}
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("No supported registry settings were provided.")
        return
    bids_root = Path(args.bids_root).expanduser().resolve()
    registry = Registry.for_project("", bids_root=bids_root)
    if not registry.existing_database_path().is_file():
        raise SystemExit("No central nro registry found")
    updated_requests = registry.set_active_concurrency(settings["concurrency"])
    if updated_requests == 0:
        raise SystemExit("No active requests have a concurrency setting to update")
    result = {
        "settings": settings,
        "updated_requests": updated_requests,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(
        f"Set concurrency={settings['concurrency']} on "
        f"{updated_requests} active request(s)."
    )


if __name__ == "__main__":
    main()
