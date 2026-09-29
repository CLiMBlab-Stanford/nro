"""Rename a BIDS project and its nro-managed state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser(*, prog: str = "nro project") -> argparse.ArgumentParser:
    """Construct project maintenance arguments without accessing site state."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    rename = subcommands.add_parser(
        "rename",
        help="Preview or execute an atomic project rename.",
        description=(
            "Rename a BIDS project across managed BIDS, WORK, development, definitions, "
            "BIDSification, ownership, and registry state. The default is a preview."
        ),
    )
    rename.add_argument("old", help="Current project identifier")
    rename.add_argument("new", help="Replacement project identifier")
    rename.add_argument(
        "--execute",
        action="store_true",
        help="Apply the previewed rename; the project must have no active demand or work.",
    )
    rename.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser


def _render(result: dict) -> str:
    lines = [f"Project rename: {result['old']} -> {result['new']}", ""]
    lines.append(f"Registered work items: {result['work_items']}")
    lines.append(f"Ownership receipts: {result['ownership_receipts']}")
    lines.append(f"Workbench scenes: {result['scene_files']}")
    lines.append(f"Structured metadata files: {result['metadata_files']}")
    lines.append(f"Raw BIDS symbolic links to materialize: {result['source_symlinks']}")
    lines.append(f"Absolute symbolic links: {result['absolute_symlinks']}")
    lines.append(f"BIDSification records: {result['ingestion_records']}")
    lines.append(f"Definition files: {len(result['definition_files'])}")
    lines.extend(("", "Managed directory moves:"))
    lines.extend(f"  {item['source']} -> {item['destination']}" for item in result["moves"])
    if result["blockers"]:
        lines.extend(("", "Blocked:"))
        lines.extend(f"  {reason}" for reason in result["blockers"])
    elif not result.get("executed"):
        lines.extend(("", "Preview only. Re-run with --execute to apply this rename."))
    else:
        lines.extend(("", f"Rename complete. Recovery journal: {result['journal']}"))
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, prog: str = "nro project") -> None:
    """Preview or execute project-level maintenance through the central scheduler."""
    args = build_parser(prog=prog).parse_args(argv)
    from nro.configuration.site import CHECKOUT, settings
    from nro.orchestration.scheduler_client import maintenance

    values = settings()[0]
    result = maintenance(
        Path(values["registry"]),
        Path(values["bids"]),
        checkout=CHECKOUT,
        operation="project_rename",
        old=args.old,
        new=args.new,
        execute=args.execute,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_render(result))


if __name__ == "__main__":
    main()
