"""Migrate durable nro metadata representations without changing science."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.cli import package_version
from nro.configuration.site import settings


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Preview or apply explicit nro metadata migrations.",
    )
    commands = parser.add_subparsers(dest="migration", required=True)
    provenance = commands.add_parser(
        "provenance",
        help="replace host paths in nro public metadata with portable references",
    )
    provenance.add_argument("-P", "--project", nargs="+", dest="projects")
    provenance.add_argument(
        "--execute",
        action="store_true",
        help="apply the previewed conversion; the default is read-only",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro migrate") -> None:
    """Run a selected representation migration after an explicit preview."""
    args = _parser(prog).parse_args(argv)
    values = settings()[0]
    bids = Path(values["bids"])
    projects = args.projects or sorted(path.name for path in bids.iterdir() if path.is_dir())
    from nro.configuration.site import CHECKOUT
    from nro.orchestration.scheduler_client import maintenance

    report = maintenance(
        Path(values["registry"]),
        bids,
        checkout=CHECKOUT,
        operation="provenance_migration",
        projects=projects,
        execute=args.execute,
        version=package_version(),
    )
    action = "Migrated" if args.execute else "Would migrate"
    print(f"Scanned {report['scanned']} metadata file(s).")
    print(f"{action} {len(report['changed'])} file(s).")
    for path in report["changed"]:
        print(path)
    if report["errors"]:
        detail = "\n".join(f"- {error}" for error in report["errors"])
        raise SystemExit(f"Migration blocked by {len(report['errors'])} error(s):\n{detail}")
