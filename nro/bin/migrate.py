"""Migrate durable nro metadata representations without changing science."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.cli import package_version
from nro.configuration.site import settings
from nro.engine.cli import page_text


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Preview or apply explicit nro metadata migrations.",
    )
    parser.add_argument("-P", "--project", nargs="+", dest="projects", metavar="PROJECT")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview without applying changes")
    mode.add_argument(
        "-f", "--force", action="store_true", help="Apply changes without confirmation"
    )
    mode.add_argument("--execute", dest="force", action="store_true", help=argparse.SUPPRESS)
    return parser


def _render(report: dict) -> str:
    """Format a project migration preview for interactive review."""
    lines = ["Planned project migration", ""]
    lines.append(f"Metadata files scanned: {report['scanned']}")
    lines.append(f"Metadata files to rewrite: {len(report['changed'])}")
    lines.append(f"Raw BIDS links to materialize: {len(report.get('source_links', ()))}")
    lines.append(f"FreeSurfer template links to repair: {len(report.get('templates', ()))}")
    lines.append(f"Work-item contracts to migrate: {report['contracts']}")
    if report.get("recovery"):
        lines.append(f"Interrupted migrations to recover: {len(report['recovery'])}")
    if report["changed"]:
        lines.extend(("", "Files:"))
        lines.extend(f"  {path}" for path in report["changed"])
    if report.get("templates"):
        lines.extend(("", "Template links:"))
        lines.extend(f"  {path}" for path in report["templates"])
    if report.get("source_links"):
        lines.extend(("", "Raw BIDS links:"))
        lines.extend(f"  {path}" for path in report["source_links"])
    if report["errors"]:
        lines.extend(("", f"Blocking errors ({len(report['errors'])}):"))
        lines.extend(f"  {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def _confirm() -> bool:
    try:
        response = input("Proceed with project migration? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None, *, prog: str = "nro migrate") -> None:
    """Preview and optionally apply a selected representation migration."""
    args = _parser(prog).parse_args(argv)
    values = settings()[0]
    bids = Path(values["bids"])
    projects = args.projects or sorted(path.name for path in bids.iterdir() if path.is_dir())
    from nro.configuration.site import CHECKOUT
    from nro.orchestration.scheduler_client import maintenance

    preview = maintenance(
        Path(values["registry"]),
        bids,
        checkout=CHECKOUT,
        operation="dataset_migration",
        projects=projects,
        execute=False,
        version=package_version(),
    )
    page_text(_render(preview))
    if preview["errors"]:
        raise SystemExit(f"Migration blocked by {len(preview['errors'])} error(s)")
    pending = (
        len(preview["changed"])
        + len(preview.get("source_links", ()))
        + len(preview.get("templates", ()))
        + int(preview["contracts"])
        + len(preview.get("recovery", ()))
    )
    if args.dry_run or not pending:
        return
    if not args.force and not _confirm():
        print("Project migration cancelled.")
        return
    report = maintenance(
        Path(values["registry"]),
        bids,
        checkout=CHECKOUT,
        operation="dataset_migration",
        projects=projects,
        execute=True,
        version=package_version(),
        preparation=preview["preparation"],
    )
    if report["errors"]:
        detail = "\n".join(f"- {error}" for error in report["errors"])
        raise SystemExit(f"Migration blocked by {len(report['errors'])} error(s):\n{detail}")
    print(
        f"Migrated {len(report['changed'])} metadata file(s), "
        f"{len(report.get('source_links', ()))} raw BIDS link(s), "
        f"{len(report.get('templates', ()))} template link(s), and "
        f"{report['contracts']} work-item contract(s)."
    )
