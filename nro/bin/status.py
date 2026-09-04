"""Report registry-backed nro derivative status."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

from nro.engine.bids import matches_selectors
from nro.engine.cli import add_core_selection_arguments, core_selection, page_text
from nro.configuration.paths import BIDS_PATH
from nro.orchestration.registry import Registry
from nro.orchestration.catalog import MODULES
from nro.orchestration.selection import selected_projects


_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_RED = "\x1b[91m"
_GREEN = "\x1b[92m"
_YELLOW = "\x1b[93m"
_BLUE = "\x1b[94m"
_MAGENTA = "\x1b[95m"
_CYAN = "\x1b[96m"
_GRAY = "\x1b[90m"

_STATUS_COLORS = {
    "Success": _GREEN,
    "Running": _CYAN,
    "Queued": _BLUE,
    "Blocked": _YELLOW,
    "Error": _RED + _BOLD,
    "Unsubmitted": _GRAY,
}


def _paint(text: str, style: str, *, color: bool) -> str:
    return f"{style}{text}{_RESET}" if color else text


def _terminal_colors_enabled() -> bool:
    """Use color only for interactive output and honor the NO_COLOR convention."""
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _instance_identifier(item: dict) -> str:
    entities = item.get("entities", "")
    if isinstance(entities, dict):
        entities = " ".join(f"{key}={value}" for key, value in sorted(entities.items()))
    return (
        f"{item['project']} sub-{item['participant']} {item['module']}"
        + (f" ({entities})" if entities else "")
    )


def _render_report(
    output: list[dict], *, errors: list[dict] | None = None,
    blocked_instances: list[dict] | None = None,
    color: bool = False,
) -> str:
    lines = [
        _paint(
            f"{'PROJECT':14} {'PARTICIPANT':14} {'MODULE':20} "
            f"{'STATUS':12} {'MEM':8} ENTITIES",
            _BOLD + _CYAN,
            color=color,
        )
    ]
    for row in output:
        entities = " ".join(
            f"{key}={value}" for key, value in sorted(row["entities"].items())
        ) or "-"
        status = str(row["status"])
        status_column = _paint(
            f"{status:12}",
            _STATUS_COLORS.get(status, _MAGENTA),
            color=color,
        )
        lines.append(
            f"{row['project'][:14]:14} {row['participant'][:14]:14} "
            f"{row['module'][:20]:20} {status_column} "
            f"{str(row['memory_gb']) + 'G':8} "
            f"{_paint(entities, _DIM, color=color)}"
        )
    if errors:
        lines.extend((
            "",
            _paint("=" * 50, _DIM + _RED, color=color),
            _paint("Errors", _BOLD + _RED, color=color),
        ))
        for error in errors:
            lines.append(f"- {_instance_identifier(error)}")
            if error.get("step"):
                lines.append(
                    "  " + _paint("Failed step:", _YELLOW, color=color)
                    + f" {error['step']}"
                )
            if error.get("message"):
                lines.append(
                    "  " + _paint("Error:", _RED, color=color)
                    + f" {error['message']}"
                )
            if error.get("log"):
                lines.append(
                    "  " + _paint("Log:", _CYAN, color=color)
                    + f" {_paint(str(error['log']), _DIM, color=color)}"
                )
            if error.get("blocked_instances"):
                lines.append("  " + _paint("Blocks:", _YELLOW, color=color))
                lines.extend(f"  - {item}" for item in error["blocked_instances"])
    if blocked_instances:
        lines.extend((
            "",
            _paint("=" * 50, _DIM + _YELLOW, color=color),
            _paint("Blocked instances", _BOLD + _YELLOW, color=color),
        ))
        for instance in blocked_instances:
            lines.append(f"- {_instance_identifier(instance)}")
            lines.append("  " + _paint("Blocked by:", _YELLOW, color=color))
            lines.extend(f"  - {item}" for item in instance["upstream_errors"])
    return "\n".join(lines) + "\n"


def _failure_detail(row: dict, *, project: str) -> dict:
    """Extract the most specific root-failure information available."""
    log = Path(str(row.get("log_path") or ""))
    step = None
    message = str(row.get("error_message") or "")
    ledger = log.parent / "current-steps.json" if log else None
    if ledger is not None:
        try:
            records = json.loads(ledger.read_text(encoding="utf-8"))
            failures = [value for value in records.values() if isinstance(value, dict) and value.get("status") == "error"]
            if failures:
                latest = sorted(failures, key=lambda value: str(value.get("timestamp", "")))[-1]
                step = str(latest.get("name") or "") or None
                message = str(latest.get("error") or message)
        except (OSError, json.JSONDecodeError):
            pass
    if not step and log.is_file():
        try:
            for line in deque(log.open(encoding="utf-8", errors="replace"), maxlen=500):
                if "Failed Step:" in line:
                    step = line.split("Failed Step:", 1)[1].strip()
                if "Error:" in line:
                    message = line.split("Error:", 1)[1].strip()
        except OSError:
            pass
    entities = " ".join(f"{key}={value}" for key, value in sorted(json.loads(row["entities_json"]).items()))
    return {"instance_id": int(row["id"]), "project": project, "participant": row["participant"], "module": row["module"], "entities": entities, "step": step, "message": message, "log": str(log) if log else None}


def _matches_request(
    row: dict, *, participants: set[str], modules: set[str], workflows: set[str],
    selectors: dict[str, tuple[str, ...] | None],
) -> bool:
    if participants and row["participant"] not in participants:
        return False
    if modules and row["module"] not in modules:
        return False
    if workflows and not workflows.intersection(
        (row.get("workflow_ids") or "").split(",")
    ):
        return False
    return not selectors or matches_selectors(json.loads(row["entities_json"]), selectors)


def build_parser(*, prog: str = "nro.bin.status") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--no-pager", action="store_true",
        help="Print the human-readable report directly instead of opening less",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.status") -> None:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bids_root = Path(args.bids_root).expanduser().resolve()
    participants = set(selection.participants)
    selectors = selection.instance_entities
    modules = set(selection.modules)
    workflows = set(selection.workflows)
    output: list[dict] = []
    critical_errors: dict[tuple[str, int], dict] = {}
    registry = Registry.for_project("", bids_root=bids_root)
    if not registry.existing_database_path().is_file():
        page_text(
            _render_report(output, color=_terminal_colors_enabled()),
            use_pager=not args.no_pager,
            header_lines=1,
        )
        return
    projects = selected_projects(bids_root, selection.projects)
    selected_project_set = set(projects)
    # Status is a strictly observational projection of the scheduler's current
    # lab-wide registry state. Freshness assessment, cancellation, and request
    # reconciliation belong to ``run`` and workers.
    rows = registry.instance_status_snapshot(read_only=True)
    by_id = {int(row["id"]): row for row in rows}
    for row in rows:
        project = str(row["project"])
        if project not in selected_project_set or not _matches_request(
            row,
            participants=participants,
            modules=modules,
            workflows=workflows,
            selectors=selectors,
        ):
            continue
        entities = json.loads(row["entities_json"])
        root_ids = tuple(int(value) for value in row["root_failure_ids"])
        for root_id in root_ids:
            critical_errors[(project, root_id)] = _failure_detail(
                by_id[root_id], project=project
            )
        output.append(
            {
                "project": project,
                "participant": row["participant"],
                "module": row["module"],
                "entities": entities,
                "workflows": (row.get("workflow_ids") or "").split(",")
                if row.get("workflow_ids")
                else [],
                "status": row["status"],
                "reason": row.get("error_message") or row.get("artifact_reason"),
                "root_ids": root_ids,
                "instance_id": row["id"],
                "generation": row["current_generation"],
                "log": row.get("log_path"),
                "memory_gb": row.get("memory_gb"),
                "max_memory_gb": row.get("max_memory_gb"),
                "oom_count": row.get("oom_count", 0),
            }
        )
    error_details = list(critical_errors.values())
    error_labels = {detail["instance_id"]: _instance_identifier(detail) for detail in error_details}
    for detail in error_details:
        detail["blocked_instances"] = [
            _instance_identifier(row)
            for row in output
            if row["status"] == "Blocked" and detail["instance_id"] in row.get("root_ids", ())
        ]
    blocked_details = [
        {
            **row,
            "upstream_errors": [error_labels[root_id] for root_id in row.get("root_ids", ())],
        }
        for row in output
        if row["status"] == "Blocked"
    ]
    if args.json:
        print(json.dumps(
            {"instances": output, "errors": error_details, "blocked_instances": blocked_details},
            indent=2,
            sort_keys=True,
        ))
        return
    page_text(
        _render_report(
            output,
            errors=error_details,
            blocked_instances=blocked_details,
            color=_terminal_colors_enabled(),
        ),
        use_pager=not args.no_pager,
        header_lines=1,
    )


if __name__ == "__main__":
    main()
