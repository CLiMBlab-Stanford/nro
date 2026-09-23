"""Update live lab-wide scheduler settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nro.orchestration.registry import Registry


def build_parser(*, prog: str = "nro.bin.set") -> argparse.ArgumentParser:
    """Construct the set parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("assignments", nargs="+", metavar="NAME=VALUE")
    parser.add_argument("--json", action="store_true")
    return parser


def _parse_assignments(values: list[str]) -> dict[str, int]:
    settings: dict[str, int] = {}
    for assignment in values:
        name, separator, raw_value = assignment.partition("=")
        if not separator or not name or not raw_value:
            raise SystemExit(f"Invalid setting {assignment!r}; expected NAME=VALUE")
        if name not in {"concurrency", "gpu_concurrency"}:
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


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.set") -> None:
    """Update recognized scheduler settings; warn for unsupported keys.

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
