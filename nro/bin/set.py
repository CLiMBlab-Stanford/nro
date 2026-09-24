"""Update live lab-wide scheduler settings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nro.orchestration.pool_settings import SETTABLE_SETTINGS
from nro.orchestration.registry import Registry


def build_parser(*, prog: str = "nro.bin.set") -> argparse.ArgumentParser:
    """Construct the set parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "assignments",
        nargs="*",
        metavar="NAME=VALUE",
        help="setting assignments, or ls to list accepted names",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _parse_assignments(values: list[str]) -> dict[str, int]:
    settings: dict[str, int] = {}
    for assignment in values:
        name, separator, raw_value = assignment.partition("=")
        if not separator or not name or not raw_value:
            raise SystemExit(f"Invalid setting {assignment!r}; expected NAME=VALUE")
        if name not in SETTABLE_SETTINGS:
            print(
                f"WARNING: ignoring unsupported registry setting: {name}",
                file=sys.stderr,
            )
            continue
        try:
            value = int(raw_value)
        except ValueError as error:
            raise SystemExit(f"{name} must be an integer") from error
        if value < 1:
            raise SystemExit(f"{name} must be at least one")
        settings[name] = value
    return settings


def _list_settings(*, as_json: bool) -> None:
    settings = [
        {"name": name, **asdict(specification)} for name, specification in SETTABLE_SETTINGS.items()
    ]
    if as_json:
        print(json.dumps({"settings": settings}, indent=2))
        return
    widths = {
        field: max(len(field.upper()), *(len(setting[field]) for setting in settings))
        for field in ("name", "values", "scope")
    }
    print(
        f"{'NAME':<{widths['name']}}  {'VALUES':<{widths['values']}}  "
        f"{'SCOPE':<{widths['scope']}}  DESCRIPTION"
    )
    for setting in settings:
        print(
            f"{setting['name']:<{widths['name']}}  "
            f"{setting['values']:<{widths['values']}}  "
            f"{setting['scope']:<{widths['scope']}}  {setting['description']}"
        )


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.set") -> None:
    """Update recognized scheduler settings; warn for unsupported keys.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if "ls" in args.assignments:
        if args.assignments != ["ls"]:
            parser.error("ls cannot be combined with setting assignments")
        _list_settings(as_json=args.json)
        return
    if not args.assignments:
        parser.error("provide ls or at least one NAME=VALUE assignment")
    settings = _parse_assignments(args.assignments)
    if not settings:
        result = {"settings": {}, "updated_requests": 0}
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("No supported registry settings were provided.")
        return
    from nro.configuration import site
    from nro.orchestration.scheduler_implementation import implementation_path

    values = site.settings()[0]
    bids_root = site.bids_root()
    if (
        site.installation_record().get("mode") == "branch"
        or implementation_path(Path(values["registry"])).is_file()
    ):
        from nro.orchestration.scheduler_client import pool_operation

        updated_requests = 0
        for name, value in settings.items():
            updated_requests += pool_operation(
                Path(values["registry"]),
                bids_root,
                checkout=site.CHECKOUT,
                operation=name,
                concurrency=value,
            )["updated_requests"]
    else:
        registry = Registry.for_project("", bids_root=bids_root)
        if not registry.existing_database_path().is_file():
            raise SystemExit("No central nro registry found")
        updated_requests = sum(
            registry.set_active_concurrency(value)
            if name == "concurrency"
            else registry.set_gpu_concurrency(value)
            for name, value in settings.items()
        )
    if updated_requests == 0:
        raise SystemExit("No active requests have a concurrency setting to update")
    result = {
        "settings": settings,
        "updated_requests": updated_requests,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    rendered = ", ".join(f"{name}={value}" for name, value in settings.items())
    print(f"Set {rendered}; updated {updated_requests} registry setting(s).")


if __name__ == "__main__":
    main()
