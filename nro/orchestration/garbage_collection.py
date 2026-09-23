"""Find and remove unclaimed files from nro public derivative namespaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from nro.engine.bids import parse_bids_entities
from nro.engine.cli import matches_module_lineage, matches_work_item_selectors
from nro.orchestration.artifact_ownership import (
    public_ownership_index,
    rows_with_public_receipts,
    work_ownership_index,
)
from nro.orchestration.branch_purge import snapshot
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.purge_paths import _remove_path
from nro.orchestration.selection import discover_bids_inventory


def _selected_rows(rows: list[dict], selection: dict) -> list[dict]:
    """Apply the common selector vocabulary to scheduler snapshot rows."""
    projects = set(selection.get("projects", ()))
    participants = {str(value).removeprefix("sub-") for value in selection.get("participants", ())}
    modules = set(selection.get("modules", ()))
    workflows = set(selection.get("workflows", ()))
    lineages = tuple(selection.get("lineages", ()))
    selectors = selection.get("selectors", {})
    return [
        row
        for row in rows
        if (not projects or row["project"] in projects)
        and (not participants or row["participant"] in participants)
        and (not modules or row["module"] in modules)
        and (not workflows or workflows.intersection(str(row.get("workflow_ids") or "").split(",")))
        and matches_module_lineage(row["module"], row["directory_label"], lineages)
        and matches_work_item_selectors(json.loads(row["entities_json"]), selectors)
    ]


def _path_entities(path: Path) -> dict[str, str]:
    entities: dict[str, str] = {}
    for part in path.parts:
        entities.update(parse_bids_entities(part))
    smoothing = entities.get("smoothing")
    if smoothing and smoothing.endswith("mm"):
        entities["smoothing"] = smoothing[:-2]
    return entities


def _path_in_selected_scope(
    path: Path,
    *,
    derivatives_root: Path,
    selection: dict,
    selected_labels: dict[str, set[str]],
) -> bool:
    relative = path.relative_to(derivatives_root)
    if not relative.parts or ".nro" in relative.parts:
        return False
    module = relative.parts[0]
    modules = set(selection.get("modules", ()))
    if modules and module not in modules:
        return False
    if selection.get("workflows") or selection.get("lineages"):
        if len(relative.parts) < 2 or relative.parts[1] not in selected_labels.get(module, set()):
            return False
    entities = _path_entities(relative)
    participants = {str(value).removeprefix("sub-") for value in selection.get("participants", ())}
    if participants:
        structural = {
            part.removeprefix("sub-") for part in relative.parts if part.startswith("sub-")
        }
        if entities.get("sub") not in participants and not structural.intersection(participants):
            return False
    return matches_work_item_selectors(entities, selection.get("selectors", {}))


def _unclaimed_paths(
    derivatives_root: Path,
    ownership,
    *,
    selected: list[dict],
    selection: dict,
) -> tuple[Path, ...]:
    if not derivatives_root.is_dir():
        return ()
    selected_labels: dict[str, set[str]] = {}
    for row in selected:
        selected_labels.setdefault(str(row["module"]), set()).add(str(row["directory_label"]))
    garbage = []
    for path in sorted(derivatives_root.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        if not _path_in_selected_scope(
            path,
            derivatives_root=derivatives_root,
            selection=selection,
            selected_labels=selected_labels,
        ):
            continue
        if not ownership.owns(path):
            garbage.append(path.absolute())
    return tuple(garbage)


def garbage_paths(
    project_root: Path,
    rows: list[dict],
    selected: list[dict],
    *,
    selection: dict,
    control: Path,
    work_root: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return unclaimed files in selected public and private derivative scopes."""
    facade = SimpleNamespace(
        paths=SimpleNamespace(
            project=str(project_root.name),
            project_root=project_root,
            control=control,
        )
    )
    rows = rows_with_public_receipts(project_root, rows)
    selected = _selected_rows(rows, selection)
    public_root = project_root / "derivatives" / "nro"
    private_root = work_root / project_root.name / "derivatives" / "nro"
    public = _unclaimed_paths(
        public_root,
        public_ownership_index(rows, registry=facade, work_root=work_root),
        selected=selected,
        selection=selection,
    )
    private = _unclaimed_paths(
        private_root,
        work_ownership_index(rows, registry=facade, work_root=work_root),
        selected=selected,
        selection=selection,
    )
    return public, private


def collect(
    registry,
    *,
    checkout: Path,
    site_values: dict,
    selection: dict,
    dry_run: bool,
    approved: list[str] | None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Collect a confirmed set of unclaimed public derivative files."""
    view = snapshot(registry, checkout=checkout, site_values=site_values)
    rows = view["rows"]
    selected = _selected_rows(rows, selection)
    active = [
        row
        for row in selected
        if row.get("attempt_state") in {"queued", "running", "cancel_requested"}
    ]
    if active:
        rendered = ", ".join(
            f"{row['project']}:{row['module']}:sub-{row['participant']}" for row in active
        )
        raise ValueError(
            f"Refusing to collect garbage during active derivative attempt(s): {rendered}"
        )

    topology = BranchStore(registry.paths.control).read().topology
    branch = topology.require_checkout(checkout)
    paths = BranchPaths(
        branch, *(Path(site_values[key]) for key in ("bids", "work", "development"))
    )
    projects = list(selection.get("projects", ())) or sorted(
        discover_bids_inventory(Path(site_values["bids"]))
    )
    public_candidates: set[Path] = set()
    private_candidates: set[Path] = set()
    if progress is not None:
        progress("Scanning derivative namespaces", 0, len(projects))
    for position, project in enumerate(projects, start=1):
        project_rows = [row for row in rows if row["project"] == project]
        project_selected = [row for row in selected if row["project"] == project]
        public, private = garbage_paths(
            paths.output_project(project),
            project_rows,
            project_selected,
            selection=selection,
            control=registry.paths.control,
            work_root=paths.private_project(project).parent,
        )
        public_candidates.update(public)
        private_candidates.update(private)
        if progress is not None:
            progress("Scanning derivative namespaces", position, len(projects))
    ordered_public = tuple(sorted(public_candidates))
    ordered_private = tuple(sorted(private_candidates))
    ordered = ordered_public + ordered_private
    if approved is not None and set(map(Path, approved)) != set(ordered):
        raise ValueError("Garbage-collection targets changed; generate and confirm a new report")
    removed = 0
    if not dry_run:
        if progress is not None:
            progress("Removing unclaimed files", 0, len(ordered))
        for position, path in enumerate(ordered, start=1):
            boundaries = [
                root
                for project in projects
                for root in (
                    paths.output_project(project) / "derivatives" / "nro",
                    paths.private_project(project) / "derivatives" / "nro",
                )
            ]
            boundary = next(root for root in boundaries if path.is_relative_to(root))
            removed += int(
                _remove_path(
                    path,
                    dry_run=False,
                    prune_root=boundary,
                )
            )
            if progress is not None and (position % 25 == 0 or position == len(ordered)):
                progress("Removing unclaimed files", position, len(ordered))
    return {
        "paths": [str(path) for path in ordered],
        "public_paths": [str(path) for path in ordered_public],
        "work_paths": [str(path) for path in ordered_private],
        "removed": removed,
        "projects": len(projects),
        "dry_run": dry_run,
    }
