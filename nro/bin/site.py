"""Inspect and update durable site-wide configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nro.site.policy import SITE_SETTINGS, canonical_key, format_cli_value, parse_cli_value


def build_parser(*, prog: str = "nro site") -> argparse.ArgumentParser:
    """Construct the site configuration parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("ls", help="List settings and their application lifecycle.")
    listing.add_argument("prefix", nargs="?")
    listing.add_argument("--json", action="store_true")
    getting = commands.add_parser("get", help="Print effective setting values.")
    getting.add_argument("keys", nargs="*")
    getting.add_argument("--json", action="store_true")
    setting = commands.add_parser("set", help="Atomically update durable setting values.")
    setting.add_argument("assignments", nargs="+", metavar="KEY=VALUE")
    setting.add_argument("--maintain", action="store_true")
    setting.add_argument("--json", action="store_true")
    editing = commands.add_parser("edit", help="Run the interactive site editor.")
    editing.add_argument("--maintain", action="store_true")
    commands.add_parser("validate", help="Validate the protected site configuration.")
    return parser


def _selected(keys: list[str], prefix: str | None = None) -> tuple[str, ...]:
    if keys:
        return tuple(dict.fromkeys(canonical_key(key) for key in keys))
    if prefix is None:
        return tuple(SITE_SETTINGS)
    prefix = prefix.rstrip(".")
    selected = tuple(key for key in SITE_SETTINGS if key == prefix or key.startswith(prefix + "."))
    if not selected:
        raise ValueError(f"No site settings match {prefix!r}")
    return selected


def _propagate(changes: dict[str, object], values: dict) -> list[str]:
    """Relay marked settings to a live scheduler without starting one."""
    operations = {
        SITE_SETTINGS[key].scheduler_operation: value
        for key, value in changes.items()
        if SITE_SETTINGS[key].scheduler_operation is not None
    }
    if not operations:
        return []
    from nro.orchestration.scheduler_bus import create_message, read_active

    control = Path(values["registry"])
    active = read_active(control)
    if active is None:
        return []
    from nro.orchestration.scheduler_rpc import request
    from nro.site import configuration as site

    applied = []
    for operation, value in operations.items():
        response = request(
            active,
            create_message(
                {
                    "operation": str(operation),
                    "checkout": str(site.CHECKOUT),
                    "concurrency": int(value),
                }
            ),
            timeout=10.0,
            durable=False,
        )
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        if not isinstance(response.get("result"), dict):
            raise RuntimeError("Scheduler returned an invalid setting response")
        applied.append(str(operation))
    return applied


def _set(args) -> None:
    from nro.site import configuration as site
    from nro.site.setup import edit_settings

    changes: dict[str, object] = {}
    for assignment in args.assignments:
        raw_key, separator, raw_value = assignment.partition("=")
        if not separator or not raw_key:
            raise ValueError(f"Expected KEY=VALUE: {assignment!r}")
        key = canonical_key(raw_key)
        try:
            changes[key] = parse_cli_value(SITE_SETTINGS[key], raw_value)
        except ValueError as error:
            raise ValueError(f"{key}: {error}") from error
    maintenance = any(SITE_SETTINGS[key].maintenance for key in changes)
    assignments = []
    for key, value in changes.items():
        storage_key = SITE_SETTINGS[key].storage_key
        encoded = json.dumps(value) if isinstance(value, list) else str(value)
        assignments.append(f"{storage_key}={encoded}")
    edit_settings(
        assignments,
        maintain=args.maintain,
        live=not maintenance,
        quiet=True,
    )
    values = site.settings()[0]
    propagated: list[str] = []
    warning = None
    try:
        propagated = _propagate(changes, values)
    except (OSError, RuntimeError, ValueError) as error:
        warning = str(error)
        print(
            "WARNING: settings were saved, but the live scheduler could not be updated: " + warning,
            file=sys.stderr,
        )
    result = {
        "settings": {key: changes[key] for key in sorted(changes)},
        "propagated": propagated,
        "warning": warning,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("Saved site configuration.")
    effects = sorted({SITE_SETTINGS[key].effect for key in changes})
    print("Application: " + ", ".join(effects) + ".")
    if any(SITE_SETTINGS[key].scheduler_operation for key in changes) and not propagated:
        print("No live scheduler was updated; the durable values apply when coordination resumes.")


def main(argv: list[str] | None = None, *, prog: str = "nro site") -> None:
    """Run one site configuration operation."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        from nro.site import configuration as site

        if args.action == "ls":
            values, sources = site.settings()
            keys = _selected([], args.prefix)
            rows = [
                {
                    "key": key,
                    "value": values[SITE_SETTINGS[key].storage_key],
                    "effect": SITE_SETTINGS[key].effect,
                    "source": sources[SITE_SETTINGS[key].storage_key],
                    "description": SITE_SETTINGS[key].description,
                }
                for key in keys
            ]
            if args.json:
                print(json.dumps({"settings": rows}, indent=2, sort_keys=True))
                return
            width = max(len("KEY"), *(len(row["key"]) for row in rows))
            print(f"{'KEY':<{width}}  VALUE  EFFECT")
            for row in rows:
                print(f"{row['key']:<{width}}  {format_cli_value(row['value'])}  {row['effect']}")
            return
        if args.action == "get":
            values, _sources = site.settings()
            selected = _selected(args.keys)
            result = {key: values[SITE_SETTINGS[key].storage_key] for key in selected}
            if args.json:
                print(json.dumps({"settings": result}, indent=2, sort_keys=True))
            else:
                for key, value in result.items():
                    print(f"{key}={format_cli_value(value)}")
            return
        if args.action == "set":
            _set(args)
            return
        if args.action == "edit":
            from nro.site.setup import edit_settings

            edit_settings(maintain=args.maintain)
            return
        definitions = site.definitions_root()
        from nro.definitions.migrations import validate_store_integrity

        validate_store_integrity(definitions)
        site.read_site_definition(definitions)
        print(f"Validated {site.site_definition_path(definitions)}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(130, "\nSite configuration cancelled.\n")


if __name__ == "__main__":
    main()
