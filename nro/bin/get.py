"""Read live lab-wide scheduler settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nro.orchestration.pool_settings import SETTABLE_SETTINGS
from nro.orchestration.registry import Registry


def build_parser(*, prog: str = "nro.bin.get") -> argparse.ArgumentParser:
    """Construct the get parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "keys",
        nargs="*",
        metavar="NAME",
        help="setting names; omit them to return every setting",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _selected_keys(parser: argparse.ArgumentParser, keys: list[str]) -> tuple[str, ...]:
    unknown = sorted(set(keys).difference(SETTABLE_SETTINGS))
    if unknown:
        parser.error("unknown setting(s): " + ", ".join(unknown))
    return tuple(dict.fromkeys(keys)) if keys else tuple(SETTABLE_SETTINGS)


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.get") -> None:
    """Print current scheduler settings selected by name."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    keys = _selected_keys(parser, args.keys)

    from nro.configuration import site
    from nro.orchestration.scheduler_implementation import implementation_path

    values = site.settings()[0]
    bids_root = site.bids_root()
    if (
        site.installation_record().get("mode") == "branch"
        or implementation_path(Path(values["registry"])).is_file()
    ):
        from nro.orchestration.scheduler_client import pool_operation

        result = pool_operation(
            Path(values["registry"]),
            bids_root,
            checkout=site.CHECKOUT,
            operation="settings",
        )
        current = result["settings"]
    else:
        registry = Registry.for_project("", bids_root=bids_root)
        if not registry.existing_database_path().is_file():
            raise SystemExit("No central nro registry found")
        current = registry.pool_settings()

    selected = {key: current[key] for key in keys}
    if args.json:
        print(json.dumps({"settings": selected}, indent=2))
        return
    for key, value in selected.items():
        print(f"{key}={'unset' if value is None else value}")


if __name__ == "__main__":
    main()
