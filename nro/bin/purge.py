"""Remove nro-controlled derivatives and logs within a selected scope."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nro.engine.cli import matches_instance_selectors as matches_selectors
from nro.engine.cli import add_core_selection_arguments, core_selection, page_text
from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.selection import selected_projects
from nro.orchestration.catalog import MODULES, module_descriptor, normalize_module
from nro.orchestration.ownership import (
    instance_record_path,
    remove_empty_ownership_root,
)
from nro.engine.bids import parse_bids_entities


@dataclass(frozen=True)
class PurgeResult:
    """Counters and removed-path records accumulated during a purge."""
    projects: int = 0
    instances: int = 0
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


def _matching_instances(
    registry: Registry,
    *,
    participants: Iterable[str],
    modules: set[str],
    workflows: set[str],
    selectors: dict[str, tuple[str, ...] | None],
) -> list[dict]:
    participant_set = {value.removeprefix("sub-") for value in participants}
    result = []
    for row in registry.instance_rows():
        if str(row["project"]) != registry.paths.project:
            continue
        if participant_set and str(row["participant"]) not in participant_set:
            continue
        if modules and str(row["module"]) not in modules:
            continue
        if workflows and not workflows.intersection(
            str(row.get("workflow_ids") or "").split(",")
        ):
            continue
        entities = json.loads(row["entities_json"])
        if selectors and not matches_selectors(entities, selectors):
            continue
        result.append(row)
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if dry_run:
        return True
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)
    return True


RUN_IDENTITY_ENTITIES = {
    "ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part", "chunk"
}


def _prefix_owned_paths(
    root: Path,
    prefix: str,
    *,
    entities: dict[str, str],
    inventories: dict[Path, tuple[Path, ...]] | None = None,
) -> list[Path]:
    """Return files owned by one exact instance prefix.

    A simple subject prefix is not sufficient: for example, ``sub-01`` is also
    a prefix of every run and of the subject's anatomical products. Compare
    source-run identity when present, require the complete target prefix, and
    exclude anatomy directories as a structural guardrail.
    """
    if not root.is_dir() or not prefix:
        return []
    expected = {
        key: str(value) for key, value in entities.items() if key in RUN_IDENTITY_ENTITIES
    }
    inventory_key = root.resolve()
    if inventories is None:
        paths = tuple(sorted(root.rglob("*")))
    else:
        if inventory_key not in inventories:
            inventories[inventory_key] = tuple(sorted(root.rglob("*")))
        paths = inventories[inventory_key]
    selected = []
    for path in paths:
        if not (path.is_file() or path.is_symlink()):
            continue
        relative_parts = path.relative_to(root).parts[:-1]
        if "anat" in relative_parts or "freesurfer" in relative_parts:
            continue
        target_breadcrumb = path.name == f".{prefix}_complete"
        if (
            not target_breadcrumb
            and path.name != prefix
            and not path.name.startswith(prefix + "_")
        ):
            continue
        if target_breadcrumb:
            selected.append(path)
            continue
        candidate = {
            key: str(value)
            for key, value in parse_bids_entities(path.name).items()
            if key in RUN_IDENTITY_ENTITIES
        }
        if candidate != expected:
            continue
        selected.append(path)
    return selected


def _instance_paths(
    instance: dict,
    *,
    registry: Registry,
    work_root: Path,
    inventories: dict[Path, tuple[Path, ...]] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Return instance-owned derivative/control paths and external WORK paths."""
    module = str(instance["module"])
    participant = str(instance["participant"]).removeprefix("sub-")
    sub_id = f"sub-{participant}"
    output_root = Path(instance["output_root"])
    output_prefix = str(instance.get("output_prefix") or "")
    derivatives_root = registry.paths.project_root / "derivatives"
    project_work_derivatives = work_root / registry.paths.project / "derivatives"

    derivative_paths: list[Path] = []
    work_paths: list[Path] = []
    entities = json.loads(instance["entities_json"])
    if module == "anat":
        derivative_paths.append(output_root)
    elif module in {"microparcellation", "networks"}:
        derivative_paths.append(output_root)
    elif module == "firstlevels":
        # A task root spans participants, variants and levels. Only this exact
        # participant/model/target prefix is owned by the selected instance.
        derivative_paths.extend(path for path in output_root.glob(f"node-*/{sub_id}/{output_prefix}_*") if path.is_file())
    else:
        derivative_paths.extend(
            _prefix_owned_paths(
                output_root,
                output_prefix,
                entities=entities,
                inventories=inventories,
            )
        )

    if module == "anat":
        preprocessing_root = output_root.parent.parent
        derivative_paths.append(preprocessing_root / "code" / "freesurfer" / sub_id)

    try:
        relative_output = output_root.relative_to(derivatives_root)
    except ValueError:
        relative_output = None
    if relative_output is not None:
        if module == "anat":
            work_paths.append(project_work_derivatives / relative_output)
        elif module == "func":
            base = project_work_derivatives / relative_output
            if entities.get("ses"):
                base /= f"ses-{entities['ses']}"
            work_paths.append(base / "func" / f"{output_prefix}_bold")
        elif module == "clean":
            base = project_work_derivatives / relative_output
            if entities.get("ses"):
                base /= f"ses-{entities['ses']}"
            target = (
                f"space-{entities['space']}_smoothing-{entities['smoothing']}mm"
            )
            filename_target = target
            run_prefix = output_prefix.removesuffix(f"_{filename_target}")
            work_paths.append(base / run_prefix / target)
        elif module in {"microparcellation", "networks"}:
            work_paths.append(project_work_derivatives / relative_output)
        elif module == "firstlevels":
            work_paths.append(project_work_derivatives / "firstlevels" / str(instance["directory_label"]) / output_prefix)

    derivative_paths.append(Path(instance["manifest_path"]))
    descriptor = module_descriptor(module)
    derivative_paths.append(
        instance_record_path(
            registry.paths.project_root,
            descriptor.configuration_class,
            str(instance["directory_label"]),
            module,
            str(instance["instance_key"]),
        )
    )
    allowed_derivative_roots = (derivatives_root, registry.paths.control)
    derivative_root_resolved = {
        root.resolve(strict=False) for root in allowed_derivative_roots
    }
    derivative_paths = [
        path
        for path in dict.fromkeys(derivative_paths)
        if any(_is_within(path, root) for root in allowed_derivative_roots)
        and path.resolve(strict=False) not in derivative_root_resolved
    ]
    work_paths = [
        path
        for path in dict.fromkeys(work_paths)
        if _is_within(path, project_work_derivatives)
        and path.resolve(strict=False) != project_work_derivatives.resolve(strict=False)
    ]
    return derivative_paths, work_paths


def _active_instances(registry: Registry, instance_ids: set[int]) -> list[dict]:
    if not instance_ids:
        return []
    placeholders = ",".join("?" for _ in instance_ids)
    with registry.connection() as db:
        rows = db.execute(
            f"""SELECT DISTINCT i.project, i.module, i.participant, i.id
                FROM attempts a JOIN instances i ON i.id=a.instance_id
                WHERE a.instance_id IN ({placeholders})
                  AND a.state IN ('queued', 'running', 'cancel_requested')
                ORDER BY i.project, i.module, i.participant, i.id""",
            tuple(sorted(instance_ids)),
        ).fetchall()
    return [dict(row) for row in rows]


def _active_error(active: list[dict]) -> str:
    rendered = ", ".join(
        f"{instance['project']}:{instance['module']}:sub-{instance['participant']}"
        for instance in active
    )
    return f"Refusing to purge active derivative attempt(s): {rendered}"


def _purge_instances(
    planned: list[tuple[Registry, list[dict]]],
    *,
    work_root: Path,
    dry_run: bool,
) -> PurgeResult:
    selected = [
        (registry, instance)
        for registry, instances in planned
        for instance in instances
    ]
    if not selected:
        return PurgeResult()
    instance_ids = {int(instance["id"]) for _registry, instance in selected}
    derivative_count = 0
    work_count = 0
    central_registry = planned[0][0]
    inventories: dict[Path, tuple[Path, ...]] = {}
    # Hold the one lab-wide registry lock across the full multi-project purge.
    # This prevents a worker from claiming any selected instance after the
    # active-attempt check but before its artifact state is updated.
    with central_registry.connection(write=True) as db:
        placeholders = ",".join("?" for _ in instance_ids)
        active = db.execute(
            f"""SELECT DISTINCT i.project, i.module, i.participant, i.id
                FROM attempts a JOIN instances i ON i.id=a.instance_id
                WHERE a.instance_id IN ({placeholders})
                  AND a.state IN ('queued', 'running', 'cancel_requested')
                ORDER BY i.project, i.module, i.participant, i.id""",
            tuple(sorted(instance_ids)),
        ).fetchall()
        if active:
            raise SystemExit(_active_error([dict(row) for row in active]))

        for registry, instance in selected:
            derivatives, work = _instance_paths(
                instance,
                registry=registry,
                work_root=work_root,
                inventories=inventories,
            )
            for path in derivatives:
                derivative_count += int(_remove_path(path, dry_run=dry_run))
            for path in work:
                work_count += int(_remove_path(path, dry_run=dry_run))

        if not dry_run:
            db.execute(
                f"""UPDATE instances SET artifact_state='missing', artifact_reason='Purged by user',
                    updated_at=? WHERE id IN ({placeholders})""",
                (utcnow(), *tuple(sorted(instance_ids))),
            )

    if not dry_run:
        lineage_roots = {
            (
                registry.paths.project_root,
                module_descriptor(str(instance["module"])).configuration_class,
                str(instance["directory_label"]),
            )
            for registry, instance in selected
        }
        for project_root, derivative_class, directory_label in lineage_roots:
            remove_empty_ownership_root(
                project_root, derivative_class, directory_label
            )

    return PurgeResult(
        instances=len(instance_ids),
        derivative_paths=derivative_count,
        work_paths=work_count,
    )


def _purge_attempt_logs(
    registry: Registry,
    *,
    instance_ids: set[int],
    dry_run: bool,
) -> int:
    """Remove terminal attempt logs belonging to the selected instances."""
    if not instance_ids:
        return 0
    attempt_count = 0
    with registry.connection() as db:
        terminal_attempts = db.execute(
            """SELECT id, instance_id, log_path FROM attempts
               WHERE state NOT IN ('queued', 'running', 'cancel_requested')"""
        ).fetchall()
        active_instance_logs = {
            str(row["log_path"])
            for row in db.execute(
                """SELECT DISTINCT log_path FROM attempts
                   WHERE state IN ('queued', 'running', 'cancel_requested')
                     AND log_path IS NOT NULL"""
            )
        }

    deleted_attempt_ids = []
    for attempt in terminal_attempts:
        if int(attempt["instance_id"]) not in instance_ids:
            continue
        raw_path = str(attempt["log_path"] or "").strip()
        # Sequential attempts deliberately share one current instance log. Never
        # let bare log cleanup remove it while a newer attempt is active.
        if not raw_path or raw_path in active_instance_logs:
            continue
        path = Path(raw_path)
        if _is_within(path, registry.paths.events) and _remove_path(path, dry_run=dry_run):
            attempt_count += 1
            deleted_attempt_ids.append(int(attempt["id"]))
    if deleted_attempt_ids and not dry_run:
        with registry.connection(write=True) as db:
            placeholders = ",".join("?" for _ in deleted_attempt_ids)
            db.execute(
                f"""UPDATE attempts SET log_path=NULL
                    WHERE id IN ({placeholders})
                      AND state NOT IN ('queued', 'running', 'cancel_requested')""",
                deleted_attempt_ids,
            )
    return attempt_count


def _purge_inactive_worker_logs(registry: Registry, *, dry_run: bool) -> int:
    """Remove worker logs only when their Slurm jobs are known to be inactive."""
    worker_count = 0
    with registry.connection() as db:
        active_job_ids = {
            str(row["slurm_job_id"])
            for row in db.execute(
                """SELECT slurm_job_id FROM scheduler_submissions
                   WHERE state IN ('prepared', 'submitted', 'running')
                     AND slurm_job_id IS NOT NULL"""
            )
        }
        active_job_ids.update(
            str(row["slurm_job_id"])
            for row in db.execute(
                """SELECT slurm_job_id FROM workers
                   WHERE state IN ('idle', 'running', 'draining')
                     AND slurm_job_id IS NOT NULL"""
            )
        )
        terminal_job_ids = {
            str(row["slurm_job_id"])
            for row in db.execute(
                """SELECT slurm_job_id FROM scheduler_submissions
                   WHERE state NOT IN ('prepared', 'submitted', 'running')
                     AND slurm_job_id IS NOT NULL"""
            )
        }

    for path in registry.paths.workers.glob("slurm-*.log"):
        job_id = path.stem.removeprefix("slurm-")
        if job_id in active_job_ids:
            continue
        if job_id not in terminal_job_ids and _slurm_job_may_be_active(job_id):
            continue
        worker_count += int(_remove_path(path, dry_run=dry_run))

    return worker_count


def _planned_paths(
    planned: list[tuple[Registry, list[dict]]],
    *,
    work_root: Path,
) -> tuple[list[Path], list[Path]]:
    public: set[Path] = set()
    private: set[Path] = set()
    inventories: dict[Path, tuple[Path, ...]] = {}
    for registry, instances in planned:
        for instance in instances:
            derivatives, work = _instance_paths(
                instance,
                registry=registry,
                work_root=work_root,
                inventories=inventories,
            )
            for path in derivatives:
                if not path.exists() and not path.is_symlink():
                    continue
                target = private if _is_within(path, registry.paths.control) else public
                target.add(path.absolute())
            private.update(
                path.absolute()
                for path in work
                if path.exists() or path.is_symlink()
            )
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


def _slurm_job_may_be_active(job_id: str) -> bool:
    """Protect an untracked worker log unless Slurm confirms it is absent."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def build_parser(*, prog: str = "nro.bin.purge") -> argparse.ArgumentParser:
    """Construct the purge parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument("--work-root", default=WORK_PATH)
    parser.add_argument(
        "-l", "--logs", action="store_true",
        help="Remove only matching attempt logs and logs from inactive workers",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Proceed without interactive confirmation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.purge") -> None:
    """Preview and optionally delete selected owned artifacts and logs.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bids_root = Path(args.bids_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    selectors = selection.instance_entities
    modules = _modules(selection.modules)
    projects = selected_projects(bids_root, selection.projects)
    if not projects:
        raise SystemExit("No nro projects found in the central registry")

    planned: list[tuple[Registry, list[dict]]] = []
    for project in projects:
        registry = Registry.for_project(project, bids_root=bids_root)
        instances = _matching_instances(
            registry,
            participants=selection.participants,
            modules=modules,
            workflows=set(selection.workflows),
            selectors=selectors,
        )
        planned.append((registry, instances))

    selected_instance_ids = {
        int(instance["id"])
        for _registry, instances in planned
        for instance in instances
    }
    central_registry = planned[0][0]
    if not args.logs:
        active = _active_instances(central_registry, selected_instance_ids)
        if active:
            raise SystemExit(_active_error(active))

    public_paths, private_paths = (
        ([], [])
        if args.logs
        else _planned_paths(planned, work_root=work_root)
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
        result = _purge_instances(
            planned, work_root=work_root, dry_run=args.dry_run
        )
        total = total.add(
            instances=result.instances,
            derivative_paths=result.derivative_paths,
            work_paths=result.work_paths,
        )
    total = total.add(
        attempt_logs=_purge_attempt_logs(
            central_registry,
            instance_ids=selected_instance_ids,
            dry_run=args.dry_run,
        ),
        worker_logs=_purge_inactive_worker_logs(
            central_registry, dry_run=args.dry_run
        ),
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
                f"{verb} {total.instances} instance(s): "
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
