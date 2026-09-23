"""Remove unclaimed files from selected nro derivative namespaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nro.engine.cli import add_core_selection_arguments, core_selection, page_text
from nro.orchestration.catalog import MODULES


def build_parser(*, prog: str = "nro.bin.gc") -> argparse.ArgumentParser:
    """Construct the garbage-collection parser without executing it."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument(
        "-f", "--force", action="store_true", help="Proceed without interactive confirmation"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _selection_dict(selection) -> dict:
    return {
        "participants": list(selection.participants),
        "projects": list(selection.projects),
        "modules": list(selection.modules),
        "workflows": list(selection.workflows),
        "lineages": list(selection.lineages),
        "selectors": {
            key: None if values is None else list(values)
            for key, values in selection.work_item_entities.items()
        },
    }


def _render(public: list[str], work: list[str]) -> str:
    lines = ["Planned garbage collection", ""]
    for title, paths in (("Public derivative files", public), ("Private WORK files", work)):
        lines.append(f"Unclaimed {title.lower()} ({len(paths)}):")
        lines.extend(f"  {path}" for path in paths)
        if not paths:
            lines.append("  (none)")
        lines.append("")
    lines.extend(
        (
            "",
            "Registered artifacts and files protected by public ownership records will be preserved.",
        )
    )
    return "\n".join(lines) + "\n"


def _confirm() -> bool:
    try:
        response = input("Proceed with garbage collection? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.gc") -> None:
    """Preview and optionally remove unclaimed files in selected public namespaces."""
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.json and not args.force and not args.dry_run:
        raise SystemExit("--json requires --force when garbage collection is not a dry run")

    from nro.configuration.site import CHECKOUT, settings
    from nro.orchestration.scheduler_client import maintenance

    values = settings()[0]
    control = Path(values["registry"])
    bids_root = Path(values["bids"]).resolve()
    encoded = _selection_dict(selection)
    preview = maintenance(
        control,
        bids_root,
        checkout=CHECKOUT,
        operation="gc",
        selection=encoded,
        dry_run=True,
        approved=None,
    )
    if not args.json:
        page_text(_render(preview["public_paths"], preview["work_paths"]))
    if not args.dry_run and preview["paths"] and not args.force and not _confirm():
        print("Garbage collection cancelled.")
        return
    result = preview
    if not args.dry_run and preview["paths"]:
        result = maintenance(
            control,
            bids_root,
            checkout=CHECKOUT,
            operation="gc",
            selection=encoded,
            dry_run=False,
            approved=preview["paths"],
        )
    result["dry_run"] = args.dry_run
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "Would remove" if args.dry_run else "Removed"
        count = len(result["paths"]) if args.dry_run else result["removed"]
        print(f"{verb} {count} unclaimed derivative file(s).")


if __name__ == "__main__":
    main()
