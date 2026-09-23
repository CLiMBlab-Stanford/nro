"""Remove nro-controlled derivatives and logs within a selected scope."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nro.configuration.paths import WORK_PATH
from nro.engine.cli import (
    add_core_selection_arguments,
    core_selection,
    matches_module_lineage,
    page_text,
)
from nro.engine.cli import matches_work_item_selectors as matches_selectors
from nro.orchestration.artifact_ownership import (
    work_item_paths as _work_item_paths,
)
from nro.orchestration.catalog import MODULES, normalize_module
from nro.orchestration.ownership import (
    remove_empty_ownership_root,
)
from nro.orchestration.purge_paths import (
    _is_removal_within,
    _purge_attempt_logs,
    _purge_inactive_worker_logs,
    _remove_path,
)
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.selection import selected_projects


@dataclass(frozen=True)
class PurgeResult:
    """Counters and removed-path records accumulated during a purge."""

    projects: int = 0
    work_items: int = 0
    derivative_paths: int = 0
    work_paths: int = 0
    attempt_logs: int = 0
    worker_logs: int = 0

    def add(self, **values: int) -> "PurgeResult":
        """Accumulate another purge result into this mutable report."""
        current = self.__dict__.copy()
        for key, value in values.items():
            current[key] += value
        return PurgeResult(**current)


def _modules(values: Iterable[str] | None) -> set[str]:
    modules: set[str] = set()
    for value in values or ():
        modules.add(normalize_module(value))
    return modules


def _matching_work_items(
    registry: Registry,
    *,
    participants: Iterable[str],
    modules: set[str],
    workflows: set[str],
    lineages: set[str],
    selectors: dict[str, tuple[str, ...] | None],
) -> list[dict]:
    participant_set = {value.removeprefix("sub-") for value in participants}
    result = []
    for row in registry.work_item_rows():
        if str(row["project"]) != registry.paths.project:
            continue
        if participant_set and str(row["participant"]) not in participant_set:
            continue
        if modules and str(row["module"]) not in modules:
            continue
        if workflows and not workflows.intersection(str(row.get("workflow_ids") or "").split(",")):
            continue
        if not matches_module_lineage(str(row["module"]), str(row["directory_label"]), lineages):
            continue
        entities = json.loads(row["entities_json"])
        if selectors and not matches_selectors(entities, selectors):
            continue
        result.append(row)
    return result


def _active_work_items(registry: Registry, work_item_ids: set[int]) -> list[dict]:
    if not work_item_ids:
        return []
    placeholders = ",".join("?" for _ in work_item_ids)
    with registry.connection() as db:
        rows = db.execute(
            f"""SELECT DISTINCT i.project, i.module, i.participant, i.id
                FROM attempts a JOIN work_items i ON i.id=a.work_item_id
                WHERE a.work_item_id IN ({placeholders})
                  AND a.state IN ('queued', 'running', 'cancel_requested')
                ORDER BY i.project, i.module, i.participant, i.id""",
            tuple(sorted(work_item_ids)),
        ).fetchall()
    return [dict(row) for row in rows]


def _active_error(active: list[dict]) -> str:
    rendered = ", ".join(
        f"{work_item['project']}:{work_item['module']}:sub-{work_item['participant']}"
        for work_item in active
    )
    return f"Refusing to purge active derivative attempt(s): {rendered}"


def _purge_work_items(
    planned: list[tuple[Registry, list[dict]]],
    *,
    work_root: Path,
    dry_run: bool,
) -> PurgeResult:
    ids = {int(work_item["id"]) for _registry, work_items in planned for work_item in work_items}
    if dry_run or not ids:
        return _purge_reserved_work_items(planned, work_root=work_root, dry_run=dry_run)
    registry = planned[0][0]
    active = _active_work_items(registry, ids)
    if active:
        raise SystemExit(_active_error(active))
    try:
        with registry.artifact_mutation(ids):
            return _purge_reserved_work_items(planned, work_root=work_root, dry_run=False)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


def _purge_reserved_work_items(
    planned: list[tuple[Registry, list[dict]]],
    *,
    work_root: Path,
    dry_run: bool,
) -> PurgeResult:
    selected = [
        (registry, work_item) for registry, work_items in planned for work_item in work_items
    ]
    if not selected:
        return PurgeResult()
    work_item_ids = {int(work_item["id"]) for _registry, work_item in selected}
    derivative_count = 0
    work_count = 0
    central_registry = planned[0][0]
    inventories: dict[Path, tuple[Path, ...]] = {}
    # artifact_mutation() fences the selected outputs before this function is
    # called. Keep database transactions short while filesystem removal runs.
    if not dry_run:
        with central_registry.connection() as db:
            placeholders = ",".join("?" for _ in work_item_ids)
            active = db.execute(
                f"""SELECT DISTINCT i.project, i.module, i.participant, i.id
                    FROM attempts a JOIN work_items i ON i.id=a.work_item_id
                    WHERE a.work_item_id IN ({placeholders})
                      AND a.state IN ('queued', 'running', 'cancel_requested')
                    ORDER BY i.project, i.module, i.participant, i.id""",
                tuple(sorted(work_item_ids)),
            ).fetchall()
        if active:
            raise SystemExit(_active_error([dict(row) for row in active]))

    for registry, work_item in selected:
        derivatives, work = _work_item_paths(
            work_item,
            registry=registry,
            work_root=work_root,
            inventories=inventories,
        )
        derivative_roots = (
            registry.paths.project_root / "derivatives" / "nro",
            registry.paths.control,
        )
        for path in derivatives:
            prune_root = next(root for root in derivative_roots if _is_removal_within(path, root))
            derivative_count += int(_remove_path(path, dry_run=dry_run, prune_root=prune_root))
        work_root_boundary = work_root / registry.paths.project / "derivatives" / "nro"
        for path in work:
            work_count += int(_remove_path(path, dry_run=dry_run, prune_root=work_root_boundary))

    if not dry_run:
        placeholders = ",".join("?" for _ in work_item_ids)
        with central_registry.connection(write=True) as db:
            db.execute(
                f"""UPDATE work_items SET artifact_state='missing', artifact_reason='Purged by user',
                    updated_at=? WHERE id IN ({placeholders})""",
                (utcnow(), *tuple(sorted(work_item_ids))),
            )

    if not dry_run:
        lineage_roots = {
            (
                registry.paths.project_root,
                str(work_item["configuration_class"]),
                str(work_item["directory_label"]),
            )
            for registry, work_item in selected
        }
        for project_root, configuration_class, directory_label in lineage_roots:
            remove_empty_ownership_root(project_root, configuration_class, directory_label)

    return PurgeResult(
        work_items=len(work_item_ids),
        derivative_paths=derivative_count,
        work_paths=work_count,
    )


def _planned_paths(
    planned: list[tuple[Registry, list[dict]]],
    *,
    work_root: Path,
) -> tuple[list[Path], list[Path]]:
    public: set[Path] = set()
    private: set[Path] = set()
    inventories: dict[Path, tuple[Path, ...]] = {}
    for registry, work_items in planned:
        for work_item in work_items:
            derivatives, work = _work_item_paths(
                work_item,
                registry=registry,
                work_root=work_root,
                inventories=inventories,
            )
            for path in derivatives:
                if not path.exists() and not path.is_symlink():
                    continue
                target = private if _is_removal_within(path, registry.paths.control) else public
                target.add(path.absolute())
            private.update(path.absolute() for path in work if path.exists() or path.is_symlink())
    return sorted(public), sorted(private)


def _render_plan(public: list[Path], private: list[Path], *, logs_only: bool) -> str:
    lines = ["Planned purge", ""]
    if logs_only:
        lines.append("No derivative paths will be removed; only eligible logs are selected.")
    else:
        for title, paths in (
            ("Public derivative paths", public),
            ("Private WORK/control paths", private),
        ):
            lines.append(f"{title} ({len(paths)}):")
            lines.extend(f"  {path}" for path in paths)
            if not paths:
                lines.append("  (none)")
            lines.append("")
        lines.append(
            "Matching attempt logs and worker logs whose workers are no longer "
            "running will also be removed."
        )
    return "\n".join(lines).rstrip() + "\n"


def _confirm() -> bool:
    try:
        response = input("Proceed with purge? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() in {"y", "yes"}


def build_parser(*, prog: str = "nro.bin.purge") -> argparse.ArgumentParser:
    """Construct the purge parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--work-root", default=WORK_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-l",
        "--logs",
        action="store_true",
        help="Remove only matching attempt logs and logs from inactive workers",
    )
    mode.add_argument(
        "--cache",
        action="store_true",
        help="Remove unused execution snapshots lab-wide, ignoring selectors",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Proceed without interactive confirmation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _purge_cache(args) -> None:
    from nro.configuration.site import CHECKOUT, installation_record, settings
    from nro.orchestration.execution_cache import CacheCollection, collect_cache
    from nro.orchestration.scheduler_client import maintenance
    from nro.orchestration.scheduler_implementation import implementation_path

    values = settings()[0]
    bids_root = Path(values["bids"]).resolve()
    remote = (
        installation_record().get("mode") == "branch"
        or implementation_path(Path(values["registry"])).is_file()
    )
    registry = None if remote else Registry.for_project("", bids_root=bids_root)

    def collect(*, dry_run=False, approved=None):
        if not remote:
            return collect_cache(registry, dry_run=dry_run, approved=approved)
        result = maintenance(
            Path(values["registry"]),
            bids_root,
            checkout=CHECKOUT,
            operation="cache",
            dry_run=dry_run,
            approved=None if approved is None else list(map(str, approved)),
        )
        return CacheCollection(
            tuple(map(Path, result["paths"])),
            tuple(map(Path, result["retained"])),
            result["reason"],
        )

    if args.json and not args.force and not args.dry_run:
        raise SystemExit("--json requires --force when purge is not a dry run")
    preview = collect(dry_run=True)
    if not args.json:
        report = [
            "Execution cache purge (all projects; artifact selectors are ignored).",
            *(str(path) for path in preview.paths),
        ]
        if preview.reason:
            report.append(f"Cache retained: {preview.reason}.")
        elif not preview.paths:
            report.append("No unused cache entries to remove.")
        page_text("\n".join(report) + "\n")
    if not args.dry_run and preview.paths and not args.force and not _confirm():
        print("Purge cancelled.")
        return
    result = preview if args.dry_run else collect(approved=preview.paths)
    payload = dict(
        mode="cache",
        dry_run=args.dry_run,
        paths=[str(path) for path in result.paths],
        retained=[str(path) for path in result.retained],
        reason=result.reason,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        verb = "Would purge" if args.dry_run else "Purged"
        print(
            f"{verb} {len(result.paths)} execution cache entries; retained {len(result.retained)}."
        )
        if result.reason:
            print(f"Cleanup deferred: {result.reason}.")


def _branch_purge(args, selection, *, values: dict, checkout: Path) -> None:
    from types import SimpleNamespace

    from nro.orchestration.execution_context import ExecutionContext
    from nro.orchestration.scheduler_client import maintenance

    bids_root = Path(values["bids"]).resolve()
    work_root = Path(args.work_root).resolve()
    if work_root != Path(values["work"]).resolve():
        raise SystemExit(
            "Branch purge uses the shared WORK root and maps output paths automatically"
        )
    control = Path(values["registry"])
    snapshot = maintenance(control, bids_root, checkout=checkout, operation="purge_snapshot")
    plan, public, private, projects = [], set(), set(), set()
    for row in snapshot["rows"]:
        if (
            (selection.projects and row["project"] not in selection.projects)
            or (selection.participants and row["participant"] not in selection.participants)
            or (selection.modules and row["module"] not in selection.modules)
            or (
                selection.workflows
                and not set(selection.workflows).intersection(
                    str(row.get("workflow_ids") or "").split(",")
                )
            )
            or not matches_module_lineage(
                str(row["module"]), str(row["directory_label"]), selection.lineages
            )
            or not matches_selectors(json.loads(row["entities_json"]), selection.work_item_entities)
        ):
            continue
        if not args.logs and row["attempt_state"] in {"queued", "running", "cancel_requested"}:
            raise SystemExit(_active_error([row]))
        context = ExecutionContext.from_dict(row["execution_context"])
        facade = SimpleNamespace(
            paths=SimpleNamespace(
                project=row["project"],
                control=control,
                project_root=context.paths.output_project(row["project"]),
            )
        )
        derivatives, work = (
            ([], [])
            if args.logs
            else _work_item_paths(
                row, registry=facade, work_root=context.paths.private_project(row["project"]).parent
            )
        )
        derivatives = [path for path in derivatives if path.exists() or path.is_symlink()]
        work = [path for path in work if path.exists() or path.is_symlink()]
        public.update(path for path in derivatives if not path.is_relative_to(control))
        private.update(path for path in derivatives if path.is_relative_to(control))
        private.update(work)
        projects.add(row["project"])
        plan.append(
            dict(
                id=row["id"],
                token=row["purge_token"],
                public=list(map(str, derivatives)),
                private=list(map(str, work)),
            )
        )
    if args.json and not args.force and not args.dry_run:
        raise SystemExit("--json requires --force when purge is not a dry run")
    if not args.json:
        page_text(_render_plan(sorted(public), sorted(private), logs_only=args.logs))
    if not args.force and not args.dry_run and not _confirm():
        print("Purge cancelled.")
        return
    result = maintenance(
        control,
        bids_root,
        checkout=checkout,
        operation="purge",
        plan=plan,
        logs_only=args.logs,
        dry_run=args.dry_run,
    )
    result.update(
        projects=len(projects),
        mode="logs" if args.logs else "all",
        dry_run=args.dry_run,
        planned_public_paths=list(map(str, sorted(public))),
        planned_private_paths=list(map(str, sorted(private))),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "Would purge" if args.dry_run else "Purged"
        print(
            f"{verb} {result['work_items']} work item(s), {result['derivative_paths']} derivative/control paths, "
            f"{result['work_paths']} WORK paths, and {result['attempt_logs'] + result['worker_logs']} logs."
        )


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.purge") -> None:
    """Preview and optionally delete selected owned artifacts and logs.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    if args.cache:
        _purge_cache(args)
        return
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    from nro.configuration.site import CHECKOUT, installation_record, settings
    from nro.orchestration.scheduler_implementation import implementation_path

    values = settings()[0]
    if (
        installation_record().get("mode") == "branch"
        or implementation_path(Path(values["registry"])).is_file()
    ):
        _branch_purge(args, selection, values=values, checkout=CHECKOUT)
        return
    bids_root = Path(values["bids"]).resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    selectors = selection.work_item_entities
    modules = _modules(selection.modules)
    projects = selected_projects(bids_root, selection.projects)
    if not projects:
        raise SystemExit("No nro projects found in the central registry")

    planned: list[tuple[Registry, list[dict]]] = []
    for project in projects:
        registry = Registry.for_project(project, bids_root=bids_root)
        work_items = _matching_work_items(
            registry,
            participants=selection.participants,
            modules=modules,
            workflows=set(selection.workflows),
            lineages=set(selection.lineages),
            selectors=selectors,
        )
        planned.append((registry, work_items))

    selected_work_item_ids = {
        int(work_item["id"]) for _registry, work_items in planned for work_item in work_items
    }
    central_registry = planned[0][0]
    if not args.logs:
        active = _active_work_items(central_registry, selected_work_item_ids)
        if active:
            raise SystemExit(_active_error(active))

    public_paths, private_paths = (
        ([], []) if args.logs else _planned_paths(planned, work_root=work_root)
    )
    report = _render_plan(public_paths, private_paths, logs_only=args.logs)
    if args.json:
        if not args.force and not args.dry_run:
            raise SystemExit("--json requires --force when purge is not a dry run")
    else:
        page_text(report)
    if not args.force and not args.dry_run and not _confirm():
        print("Purge cancelled.")
        return

    total = PurgeResult(projects=len(projects))
    if not args.logs:
        result = _purge_work_items(planned, work_root=work_root, dry_run=args.dry_run)
        total = total.add(
            work_items=result.work_items,
            derivative_paths=result.derivative_paths,
            work_paths=result.work_paths,
        )
    total = total.add(
        attempt_logs=_purge_attempt_logs(
            central_registry,
            work_item_ids=selected_work_item_ids,
            dry_run=args.dry_run,
        ),
        worker_logs=_purge_inactive_worker_logs(central_registry, dry_run=args.dry_run),
    )

    payload = total.__dict__ | {
        "mode": "logs" if args.logs else "all",
        "dry_run": args.dry_run,
        "planned_public_paths": [str(path) for path in public_paths],
        "planned_private_paths": [str(path) for path in private_paths],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        verb = "Would purge" if args.dry_run else "Purged"
        if not args.logs:
            print(
                f"{verb} {total.work_items} work item(s): "
                f"{total.derivative_paths} derivative/control "
                f"path(s), {total.work_paths} WORK path(s), {total.attempt_logs} attempt log(s), "
                f"and {total.worker_logs} inactive worker log(s)."
            )
        else:
            print(
                f"{verb} {total.attempt_logs} inactive attempt log(s) and "
                f"{total.worker_logs} inactive Slurm worker log(s); active logs were preserved."
            )


if __name__ == "__main__":
    main()
