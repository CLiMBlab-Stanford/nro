"""Recover unchanged derivatives after an erroneous upstream rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nro.engine.cli import add_core_selection_arguments, core_selection, page_text
from nro.orchestration.catalog import MODULES
from nro.orchestration.scheduler_client import maintenance
from nro.site.configuration import CHECKOUT, settings


def build_parser(*, prog: str = "nro freshen") -> argparse.ArgumentParser:
    """Construct the freshen command parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--dry-run", action="store_true", help="Preview without changing artifacts")
    parser.add_argument("-f", "--force", action="store_true", help="Apply without confirmation")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser


def _selection(args) -> dict:
    selected = core_selection(args)
    return {
        "participants": list(selected.participants),
        "projects": list(selected.projects),
        "modules": list(selected.modules),
        "workflows": list(selected.workflows),
        "lineages": list(selected.lineages),
        "entities": {
            key: list(value) if value is not None else None
            for key, value in selected.work_item_entities.items()
        },
    }


def _render(report: dict) -> str:
    lines = [f"Freshen preview for branch {report['branch']}", ""]
    for item in report["items"]:
        entities = " ".join(f"{key}={value}" for key, value in sorted(item["entities"].items()))
        label = f"{item['project']} sub-{item['participant']} {item['module']}/{item['lineage']}"
        if entities:
            label += f" ({entities})"
        lines.append(f"{item['action'].upper():8} {label}")
        lines.append(f"         {item['reason']}")
    if not report["items"]:
        lines.append("No work items match the selectors.")
    lines.extend(
        (
            "",
            f"Will freshen: {report['freshen']}",
            f"Already fresh: {report['retained']}",
            f"Rejected: {report['rejected']}",
        )
    )
    return "\n".join(lines) + "\n"


def _confirm() -> bool:
    try:
        return input("Proceed with freshening? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None, *, prog: str = "nro freshen") -> None:
    """Preview and re-attest matching unchanged derivatives."""
    args = build_parser(prog=prog).parse_args(argv)
    if args.json and not (args.dry_run or args.force):
        raise SystemExit("--json requires --dry-run or --force")
    values = settings()[0]
    control = Path(values["registry"])
    bids = Path(values["bids"])
    selection = _selection(args)
    report = maintenance(
        control,
        bids,
        checkout=CHECKOUT,
        operation="freshen",
        execute=False,
        selection=selection,
    )
    if args.dry_run or not report["freshen"]:
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            page_text(_render(report))
        return
    if not args.json:
        if args.force:
            print(_render(report), end="")
        else:
            page_text(_render(report))
            if not _confirm():
                print("Freshen cancelled.")
                return
    result = maintenance(
        control,
        bids,
        checkout=CHECKOUT,
        operation="freshen",
        execute=True,
        selection=selection,
        approved=report,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Freshened {result['executed']} work item(s); rejected {result['rejected']}.")


if __name__ == "__main__":
    main()
