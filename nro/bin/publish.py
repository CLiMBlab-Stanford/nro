"""Freeze a completed nro request into a standalone derivative dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.publish import publish
from nro.orchestration.registry import Registry


def build_parser(*, prog: str = "nro.bin.publish") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("request")
    parser.add_argument("destination")
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.publish") -> None:
    args = build_parser(prog=prog).parse_args(argv)
    registry = Registry.for_project(args.project, bids_root=args.bids_root)
    destination = publish(
        registry,
        request_id=args.request,
        destination=Path(args.destination),
        validate=not args.no_validate,
    )
    print(f"Published immutable derivative snapshot: {destination}")


if __name__ == "__main__":
    main()
