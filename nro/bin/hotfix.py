"""Preview and apply release-scoped repairs for identified historical defects."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.engine.cli import confirm, page_text
from nro.orchestration.hotfixes import available
from nro.site.configuration import settings


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="List, preview, or apply release-scoped repairs.",
    )
    parser.add_argument("hotfix", nargs="?", help="Hotfix identifier; omit to list available fixes")
    parser.add_argument("-P", "--project", nargs="+", dest="projects", metavar="PROJECT")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview without applying changes")
    mode.add_argument("-f", "--force", action="store_true", help="Apply without confirmation")
    return parser


def _render(report: dict) -> str:
    lines = [report["summary"], "", f"Hotfix: {report['identifier']}"]
    lines.append("Projects: " + ", ".join(report["projects"]))
    lines.append(f"Records to repair: {report['records']}")
    if report["paths"]:
        lines.extend(("", "Files:"))
        lines.extend(f"  {path}" for path in report["paths"])
    return "\n".join(lines) + "\n"


def _purge_hotfix(args, implementation, *, values: dict) -> None:
    """Purge the work a hotfix identifies, with its downstream consumers.

    The purge previews and confirms like `nro purge`, refuses work with active
    attempts, and withdraws demand for the purged work.
    """
    from nro.bin import purge
    from nro.engine.cli import core_selection
    from nro.site.configuration import CHECKOUT

    argv = ["-P", *args.projects] if args.projects else []
    if args.dry_run:
        argv.append("--dry-run")
    if args.force:
        argv.append("--force")
    purge_args = purge.build_parser().parse_args(argv)
    try:
        selection = core_selection(purge_args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    purge._branch_purge(
        purge_args,
        selection,
        values=values,
        checkout=CHECKOUT,
        defect_roots=implementation.purge_roots,
        title=f"{implementation.SUMMARY}\n\nHotfix: {implementation.HOTFIX_ID}",
    )


def main(argv: list[str] | None = None, *, prog: str = "nro hotfix") -> None:
    """List available hotfixes or apply one through the central scheduler."""
    args = _parser(prog).parse_args(argv)
    fixes = available()
    if args.hotfix is None:
        if not fixes:
            print("No release-scoped hotfixes are available.")
            return
        for identifier, implementation in sorted(fixes.items()):
            print(f"{identifier}\t{implementation.SUMMARY}")
        return
    values = settings()[0]
    implementation = fixes.get(args.hotfix)
    if implementation is not None and hasattr(implementation, "purge_roots"):
        _purge_hotfix(args, implementation, values=values)
        return
    bids = Path(values["bids"])
    projects = tuple(args.projects or sorted(path.name for path in bids.iterdir() if path.is_dir()))
    from nro.orchestration.scheduler_client import maintenance
    from nro.site.configuration import CHECKOUT

    fields = dict(identifier=args.hotfix, projects=projects)
    preview = maintenance(
        Path(values["registry"]),
        bids,
        checkout=CHECKOUT,
        operation="hotfix",
        execute=False,
        **fields,
    )
    rendered = _render(preview)
    if args.force:
        print(rendered, end="")
    else:
        page_text(rendered)
    if args.dry_run or not preview["records"]:
        return
    if not args.force and not confirm("Apply this hotfix?"):
        print("Hotfix cancelled.")
        return
    result = maintenance(
        Path(values["registry"]),
        bids,
        checkout=CHECKOUT,
        operation="hotfix",
        execute=True,
        **fields,
    )
    print(f"Repaired {result['records']} record(s) across {len(projects)} project(s).")
