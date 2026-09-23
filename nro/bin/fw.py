"""Manage private Flywheel API keys for configured site servers."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from nro.bidsify.credentials import has_key, remove_key, store_key
from nro.configuration.site import read_site_definition, settings


def _site() -> tuple[Path, dict]:
    root = Path(settings()[0]["definitions"]).expanduser().resolve()
    _settings, bidsify = read_site_definition(root)
    return root, bidsify


def _profile(bidsify: dict, server: str) -> dict:
    profile = bidsify["servers"].get(server)
    if profile is None:
        available = ", ".join(sorted(bidsify["servers"])) or "none"
        raise ValueError(f"Unknown Flywheel server {server!r}; configured servers: {available}")
    return profile


def build_parser(*, prog: str = "nro fw") -> argparse.ArgumentParser:
    """Build private Flywheel credential-management commands."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    add_key = actions.add_parser("addkey", help="Add or replace an API key")
    add_key.add_argument("server", help="Configured Flywheel server ID")
    add_key.add_argument(
        "key",
        nargs="?",
        help="API key; omit this argument to enter it without terminal echo",
    )

    remove_key_parser = actions.add_parser("removekey", help="Remove an API key")
    remove_key_parser.add_argument("server", help="Configured Flywheel server ID")

    actions.add_parser("list", help="Show whether this user has keys for configured servers")
    return parser


def main(argv=None, *, prog: str = "nro fw") -> None:
    """Manage keys without printing or adding them to tracked definitions."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        root, bidsify = _site()
        if args.action == "list":
            for server, profile in sorted(bidsify["servers"].items()):
                state = "configured" if has_key(root, server, host=profile["host"]) else "missing"
                print(f"{server}\t{state}")
            return
        profile = _profile(bidsify, args.server)
        if args.action == "addkey":
            value = args.key
            if value is None:
                value = getpass.getpass(f"Flywheel API key for {args.server}: ")
            store_key(root, args.server, value, host=profile["host"])
            print(f"Stored Flywheel key for {args.server}.")
        else:
            removed = remove_key(root, args.server)
            print(
                f"Removed Flywheel key for {args.server}."
                if removed
                else f"No Flywheel key was stored for {args.server}."
            )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(130, "\nFlywheel credential operation cancelled.\n")


if __name__ == "__main__":
    main()
