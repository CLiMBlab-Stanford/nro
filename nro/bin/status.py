"""Report registry-backed nro derivative status."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

from nro.engine.cli import (
    add_core_selection_arguments,
    core_selection,
    matches_module_lineage,
    page_text,
)
from nro.engine.cli import matches_work_item_selectors as matches_selectors
from nro.modules import MODULE_NAMES

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


def selected_projects(bids_root: Path, projects: list[str]) -> list[str]:
    """Resolve source projects only on paths that require discovery."""
    from nro.orchestration.selection import selected_projects as resolve_projects

    return resolve_projects(bids_root, projects)


_STATUS_COLORS = {
    "Success": _GREEN,
    "Running": _CYAN,
    "Stopping": _YELLOW,
    "Stopped": _GRAY,
    "Queued": _BLUE,
    "Waiting": _BLUE,
    "Blocked": _YELLOW,
    "Timeout": _YELLOW + _BOLD,
    "Error": _RED + _BOLD,
    "Corrupt": _RED + _BOLD,
    "Missing": _MAGENTA,
    "Stale": _YELLOW,
    "Unavailable": _GRAY,
}


def _paint(text: str, style: str, *, color: bool) -> str:
    return f"{style}{text}{_RESET}" if color else text


def _terminal_colors_enabled() -> bool:
    """Use color only for interactive output and honor the NO_COLOR convention."""
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _format_elapsed(seconds: float | int | None) -> str:
    """Format cumulative execution seconds in a fixed-width clock form."""
    if seconds is None:
        return "-"
    total = max(0, int(float(seconds)))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{days}-{clock}" if days else clock


def _work_item_identifier(item: dict) -> str:
    entities = item.get("entities", "")
    if isinstance(entities, dict):
        entities = " ".join(f"{key}={value}" for key, value in sorted(entities.items()))
    lineage = item.get("lineage") or item.get("directory_label")
    label = f"{item['module']}/{lineage}" if lineage else item["module"]
    return f"{item['project']} sub-{item['participant']} {label}" + (
        f" ({entities})" if entities else ""
    )


def _render_report(
    output: list[dict],
    *,
    errors: list[dict] | None = None,
    blocked_work_items: list[dict] | None = None,
    show_lineages: bool = False,
    color: bool = False,
) -> str:
    lineage_width = max([20, *(len(str(row.get("lineage") or "-")) for row in output)])
    lines = [
        _paint(
            f"{'PROJECT':14} {'PARTICIPANT':14} {'MODULE':20} "
            f"{'LINEAGE':{lineage_width}} "
            f"{'STATUS':12} {'ELAPSED':12} {'MEM':8} ENTITIES",
            _BOLD + _CYAN,
            color=color,
        )
    ]
    for row in output:
        entities = (
            " ".join(f"{key}={value}" for key, value in sorted(row["entities"].items())) or "-"
        )
        status = str(row["status"])
        status_column = _paint(
            f"{status:12}",
            _STATUS_COLORS.get(status, _MAGENTA),
            color=color,
        )
        lineage = str(row.get("lineage") or "-")
        elapsed = _format_elapsed(row.get("elapsed_seconds"))
        lines.append(
            f"{row['project'][:14]:14} {row['participant'][:14]:14} "
            f"{row['module'][:20]:20} {lineage:{lineage_width}} {status_column} "
            f"{elapsed:12} "
            f"{str(row['memory_gb']) + 'G':8} "
            f"{_paint(entities, _DIM, color=color)}"
        )
    variants: dict[tuple[str, str], dict] = {}
    for row in output:
        lineage = str(row.get("lineage") or "-")
        key = row["module"], lineage
        detail = variants.setdefault(
            key,
            {"route": row.get("route", []), "workflows": set()},
        )
        detail["workflows"].update(row.get("workflows", ()))
    if variants and show_lineages:
        lines.extend(("", _paint("Module lineages", _BOLD + _CYAN, color=color)))
        for (module, lineage), detail in variants.items():
            lines.append(f"- {module}/{lineage}")
            route = " -> ".join(f"{item['module']}={item['config']}" for item in detail["route"])
            lines.append(f"  Route: {route or '-'}")
            workflows = ", ".join(sorted(detail["workflows"])) or "-"
            lines.append(f"  Workflows: {workflows}")
    if errors:
        lines.extend(
            (
                "",
                _paint("=" * 50, _DIM + _RED, color=color),
                _paint("Errors", _BOLD + _RED, color=color),
            )
        )
        for error in errors:
            lines.append(f"- {_work_item_identifier(error)}")
            if error.get("step"):
                lines.append(
                    "  " + _paint("Failed step:", _YELLOW, color=color) + f" {error['step']}"
                )
            if error.get("message"):
                lines.append("  " + _paint("Error:", _RED, color=color) + f" {error['message']}")
            if error.get("log"):
                lines.append(
                    "  "
                    + _paint("Log:", _CYAN, color=color)
                    + f" {_paint(str(error['log']), _DIM, color=color)}"
                )
            if error.get("blocked_work_items"):
                lines.append("  " + _paint("Blocks:", _YELLOW, color=color))
                lines.extend(f"  - {item}" for item in error["blocked_work_items"])
    if blocked_work_items:
        lines.extend(
            (
                "",
                _paint("=" * 50, _DIM + _YELLOW, color=color),
                _paint("Blocked work items", _BOLD + _YELLOW, color=color),
            )
        )
        for work_item in blocked_work_items:
            lines.append(f"- {_work_item_identifier(work_item)}")
            lines.append("  " + _paint("Blocked by:", _YELLOW, color=color))
            lines.extend(f"  - {item}" for item in work_item["upstream_errors"])
    return "\n".join(lines) + "\n"


def _display_entities(row: dict) -> dict:
    """Return user-facing entities with the contrasts namespace named accurately."""
    entities = json.loads(row["entities_json"])
    if row["module"] == "contrasts" and "task" in entities:
        entities["analysis"] = entities.pop("task")
    return entities


def _ordered_status_rows(rows: list[dict], dependencies: list[tuple[int, int]]) -> list[dict]:
    """Order status rows by project, participant, and module dependency depth."""
    by_id = {int(row["id"]): row for row in rows}
    modules_by_group: dict[tuple[str, str], set[str]] = {}
    parents_by_group: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in rows:
        group = str(row["project"]), str(row["participant"])
        module = str(row["module"])
        modules_by_group.setdefault(group, set()).add(module)
        parents_by_group.setdefault(group, {}).setdefault(module, set())
    for child_id, parent_id in dependencies:
        child = by_id.get(int(child_id))
        parent = by_id.get(int(parent_id))
        if child is None or parent is None:
            continue
        child_group = str(child["project"]), str(child["participant"])
        parent_group = str(parent["project"]), str(parent["participant"])
        child_module = str(child["module"])
        parent_module = str(parent["module"])
        if child_group == parent_group and child_module != parent_module:
            parents_by_group[child_group][child_module].add(parent_module)

    ranks: dict[tuple[str, str], dict[str, tuple[int, str]]] = {}
    for group, modules in modules_by_group.items():
        remaining = set(modules)
        depths: dict[str, int] = {}
        while remaining:
            ready = sorted(
                module for module in remaining if not (parents_by_group[group][module] & remaining)
            )
            if not ready:
                cycle_depth = max(depths.values(), default=-1) + 1
                for module in sorted(remaining):
                    depths[module] = cycle_depth
                break
            for module in ready:
                parent_depths = [
                    depths[parent] for parent in parents_by_group[group][module] if parent in depths
                ]
                depths[module] = max(parent_depths, default=-1) + 1
                remaining.remove(module)
        ranks[group] = {module: (depths[module], module) for module in modules}

    def key(row: dict) -> tuple:
        group = str(row["project"]), str(row["participant"])
        entities = tuple(
            sorted((str(name), str(value)) for name, value in _display_entities(row).items())
        )
        lineage = str(row.get("directory_label") or row.get("config_id") or "-")
        stable = str(row.get("work_item_key") or row["id"])
        return (
            *group,
            ranks[group][str(row["module"])],
            lineage,
            entities,
            stable,
        )

    return sorted(rows, key=key)


def _failure_detail(row: dict, *, project: str) -> dict:
    """Extract the most specific root-failure information available."""
    log = Path(str(row.get("log_path") or ""))
    step = None
    message = str(
        row.get("artifact_reason") or ""
        if row.get("status") == "Corrupt"
        else row.get("error_message") or ""
    )
    ledger = log.parent / "current-steps.json" if log else None
    if ledger is not None:
        try:
            records = json.loads(ledger.read_text(encoding="utf-8"))
            failures = [
                value
                for value in records.values()
                if isinstance(value, dict) and value.get("status") == "error"
            ]
            if failures:
                latest = sorted(failures, key=lambda value: str(value.get("timestamp", "")))[-1]
                step = str(latest.get("name") or "") or None
                message = str(latest.get("error") or message)
        except (OSError, json.JSONDecodeError):
            pass
    if not step and not message and log.is_file():
        try:
            for line in deque(log.open(encoding="utf-8", errors="replace"), maxlen=500):
                if "Failed Step:" in line:
                    step = line.split("Failed Step:", 1)[1].strip()
                if "Error:" in line:
                    message = line.split("Error:", 1)[1].strip()
        except OSError:
            pass
    message = _concise_error(message)
    entities = " ".join(f"{key}={value}" for key, value in sorted(_display_entities(row).items()))
    return {
        "work_item_id": int(row["id"]),
        "project": project,
        "participant": row["participant"],
        "module": row["module"],
        "lineage": row.get("directory_label") or row.get("config_id"),
        "entities": entities,
        "step": step,
        "message": message,
        "log": str(log) if log else None,
    }


def _concise_error(message: str, *, limit: int = 400) -> str:
    """Reduce captured command output to one useful status-line summary."""
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return ""
    marked = []
    for line in lines:
        position = line.upper().find("ERROR:")
        if position >= 0:
            marked.append(line[position + len("ERROR:") :].strip())
    summary = (marked or lines[-1:])[0]
    summary = " ".join(summary.split())
    if len(summary) > limit:
        summary = summary[: limit - 1].rstrip() + "…"
    return summary


def _matches_request(
    row: dict,
    *,
    participants: set[str],
    modules: set[str],
    workflows: set[str],
    lineages: set[str],
    selectors: dict[str, tuple[str, ...] | None],
) -> bool:
    if participants and row["participant"] not in participants:
        return False
    if modules and row["module"] not in modules:
        return False
    if workflows and not workflows.intersection((row.get("workflow_ids") or "").split(",")):
        return False
    if not matches_module_lineage(
        str(row["module"]), str(row.get("directory_label") or ""), lineages
    ):
        return False
    return not selectors or matches_selectors(json.loads(row["entities_json"]), selectors)


def build_parser(*, prog: str = "nro.bin.status") -> argparse.ArgumentParser:
    """Construct the status parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULE_NAMES)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Thoroughly reassess artifacts, update the registry, and report its new state",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--show-lineages",
        action="store_true",
        help="Append configured routes and their workflow associations",
    )
    parser.add_argument(
        "--no-pager",
        action="store_true",
        help="Print the human-readable report directly instead of opening less",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.status") -> None:
    """Report saved or reassessed work-item status for the selected scope.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    from nro.site import configuration as site

    bids_root = site.bids_root()
    participants = set(selection.participants)
    selectors = selection.work_item_entities
    modules = set(selection.modules)
    workflows = set(selection.workflows)
    lineages = set(selection.lineages)
    output: list[dict] = []
    critical_errors: dict[tuple[str, int], dict] = {}
    from nro.bidsify.status import render as render_ingestion
    from nro.bidsify.status import selected_records

    values = site.settings()[0]
    branch_execution = site.installation_record().get("mode") in {"personal", "shared", "branch"}
    if not branch_execution:
        from nro.orchestration.scheduler_implementation import implementation_path

        branch_execution = implementation_path(Path(values["registry"])).is_file()
    projects = list(dict.fromkeys(selection.projects))
    if not branch_execution:
        projects = selected_projects(bids_root, projects)
    selected_project_set = set(projects)
    visible_ids = None
    dependencies: list[tuple[int, int]] = []
    if branch_execution:
        if args.update:
            from nro.orchestration.branch_status import refresh
            from nro.orchestration.scheduler_client import status

            current = status(
                Path(values["registry"]), bids_root, checkout=site.CHECKOUT, mode="cached"
            )
            visible = set(current["visible_ids"])
            try:
                refresh([row for row in current["rows"] if row["id"] in visible], selection)
            except (ValueError, RuntimeError, OSError) as error:
                raise SystemExit(str(error)) from error
            result = status(
                Path(values["registry"]), bids_root, checkout=site.CHECKOUT, mode="verify"
            )
        else:
            from nro.orchestration.scheduler_bus import read_branch_snapshot

            result = read_branch_snapshot(Path(values["registry"]), site.CHECKOUT)
        rows, visible_ids = result["rows"], set(result["visible_ids"])
        dependencies = [tuple(pair) for pair in result.get("dependencies", [])]
        if not projects:
            selected_project_set = {str(row["project"]) for row in rows if row["id"] in visible_ids}
        if args.update:
            from nro.orchestration.branch_status import record_observations

            try:
                record_observations(rows, visible_ids)
            except (ValueError, RuntimeError, OSError) as error:
                raise SystemExit(str(error)) from error
            from nro.orchestration.scheduler_client import maintenance

            try:
                maintenance(
                    Path(values["registry"]),
                    bids_root,
                    checkout=site.CHECKOUT,
                    operation="cache",
                    dry_run=False,
                    approved=None,
                )
            except (ValueError, RuntimeError, OSError) as error:
                print(f"WARNING: execution cache cleanup deferred: {error}", file=sys.stderr)
        from nro.bidsify.status import filter_records

        ingestion = filter_records(result.get("ingestion", []), selection)
    else:
        from nro.orchestration.registry import Registry

        registry = Registry.for_project("", bids_root=bids_root)
        ingestion = selected_records(registry, selection)
        rows = []
    if not branch_execution and registry.existing_database_path().is_file() and args.update:
        from nro.orchestration.assessment import AssessmentConflict
        from nro.orchestration.manifests import assess_registry

        try:
            assess_registry(registry, projects=projects)
        except AssessmentConflict as error:
            raise SystemExit(
                "Registry kept changing during the update; retry nro status --update."
            ) from error
        rows = registry.work_item_status_snapshot(read_only=True)
    elif not branch_execution and registry.existing_database_path().is_file():
        rows = registry.work_item_status_snapshot(read_only=True)
    if not branch_execution and registry.existing_database_path().is_file():
        dependencies = registry.work_item_dependencies(read_only=True)
    visible_rows = [row for row in rows if visible_ids is None or int(row["id"]) in visible_ids]
    ordered_rows = _ordered_status_rows(visible_rows, dependencies)
    by_id = {int(row["id"]): row for row in rows}
    order_by_id = {int(row["id"]): position for position, row in enumerate(ordered_rows)}
    for row in ordered_rows:
        project = str(row["project"])
        if project not in selected_project_set or not _matches_request(
            row,
            participants=participants,
            modules=modules,
            workflows=workflows,
            lineages=lineages,
            selectors=selectors,
        ):
            continue
        entities = _display_entities(row)
        root_ids = tuple(int(value) for value in row["root_failure_ids"])
        if row["status"] in {"Corrupt", "Timeout", "Error"}:
            key = (project, int(row["id"]))
            if key not in critical_errors:
                critical_errors[key] = _failure_detail(row, project=project)
        for root_id in root_ids:
            key = (project, root_id)
            if key not in critical_errors:
                critical_errors[key] = _failure_detail(by_id[root_id], project=project)
        output.append(
            {
                "project": project,
                "participant": row["participant"],
                "module": row["module"],
                "lineage": row.get("directory_label") or row.get("config_id") or "-",
                "route": json.loads(row.get("configuration_route_json") or "[]"),
                "entities": entities,
                "workflows": (row.get("workflow_ids") or "").split(",")
                if row.get("workflow_ids")
                else [],
                "status": row["status"],
                "elapsed_seconds": row.get("elapsed_seconds"),
                "reason": (
                    "No current workflow selects this module lineage"
                    if row["status"] == "Unavailable"
                    else row.get("error_message") or row.get("artifact_reason")
                ),
                "recomputable": bool(row.get("recomputable")),
                "root_ids": root_ids,
                "work_item_id": row["id"],
                "generation": row["current_generation"],
                "log": row.get("log_path"),
                "memory_gb": row.get("memory_gb"),
                "max_memory_gb": row.get("max_memory_gb"),
                "oom_count": row.get("oom_count", 0),
            }
        )
    error_details = sorted(
        critical_errors.values(),
        key=lambda detail: order_by_id.get(int(detail["work_item_id"]), len(order_by_id)),
    )
    error_labels = {
        detail["work_item_id"]: _work_item_identifier(detail) for detail in error_details
    }
    for detail in error_details:
        detail["blocked_work_items"] = [
            _work_item_identifier(row)
            for row in output
            if row["status"] == "Blocked" and detail["work_item_id"] in row.get("root_ids", ())
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
        print(
            json.dumps(
                {
                    "work_items": output,
                    "errors": error_details,
                    "blocked_work_items": blocked_details,
                    "bidsification": ingestion,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    page_text(
        _render_report(
            output,
            errors=error_details,
            blocked_work_items=blocked_details,
            show_lineages=args.show_lineages,
            color=_terminal_colors_enabled(),
        )
        + render_ingestion(ingestion, color=_terminal_colors_enabled()),
        use_pager=not args.no_pager,
        header_lines=1,
    )


if __name__ == "__main__":
    main()
