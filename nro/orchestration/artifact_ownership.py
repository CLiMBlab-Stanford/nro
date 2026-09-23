"""Resolve filesystem ownership boundaries for purge and garbage collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nro.engine.bids import parse_bids_entities
from nro.orchestration.ownership import work_item_record_path
from nro.orchestration.purge_paths import _entry_location, _is_entry_within

RUN_IDENTITY_ENTITIES = {
    "ses",
    "task",
    "acq",
    "ce",
    "rec",
    "dir",
    "run",
    "echo",
    "part",
    "chunk",
}


def prefix_owned_paths(
    root: Path,
    prefix: str,
    *,
    entities: dict[str, str],
    inventories: dict[Path, tuple[Path, ...]] | None = None,
) -> list[Path]:
    """Return files claimed by one exact work-item prefix."""
    if not root.is_dir() or not prefix:
        return []
    expected = {key: str(value) for key, value in entities.items() if key in RUN_IDENTITY_ENTITIES}
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
        if not target_breadcrumb and path.name != prefix and not path.name.startswith(prefix + "_"):
            continue
        if target_breadcrumb:
            selected.append(path)
            continue
        candidate = {
            key: str(value)
            for key, value in parse_bids_entities(path.name).items()
            if key in RUN_IDENTITY_ENTITIES
        }
        if candidate == expected:
            selected.append(path)
    return selected


def work_item_paths(
    work_item: dict,
    *,
    registry,
    work_root: Path,
    inventories: dict[Path, tuple[Path, ...]] | None = None,
    include_public: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Return public/control paths and private WORK paths claimed by a work item."""
    module = str(work_item["module"])
    participant = str(work_item["participant"]).removeprefix("sub-")
    sub_id = f"sub-{participant}"
    output_root = Path(work_item["output_root"])
    owned_output_root = output_root.parent if module == "anat" else output_root
    output_prefix = str(work_item.get("output_prefix") or "")
    derivatives_root = registry.paths.project_root / "derivatives" / "nro"
    project_work_derivatives = work_root / registry.paths.project / "derivatives" / "nro"

    derivative_paths: list[Path] = []
    work_paths: list[Path] = []
    entities = json.loads(work_item["entities_json"])
    if include_public:
        if module == "anat":
            derivative_paths.append(owned_output_root)
        elif module in {"dynconn", "microparcellation", "networks"}:
            derivative_paths.extend(
                prefix_owned_paths(
                    output_root,
                    output_prefix,
                    entities=entities,
                    inventories=inventories,
                )
            )
        elif module == "firstlevels":
            derivative_paths.extend(
                path
                for path in output_root.rglob(f"{output_prefix}_*")
                if path.is_file() or path.is_symlink()
            )
        else:
            derivative_paths.extend(
                prefix_owned_paths(
                    output_root,
                    output_prefix,
                    entities=entities,
                    inventories=inventories,
                )
            )

    if include_public and module == "anat":
        derivative_paths.append(owned_output_root.parent / "code" / "freesurfer" / sub_id)

    try:
        relative_output = owned_output_root.relative_to(derivatives_root)
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
            target = f"space-{entities['space']}_smoothing-{entities['smoothing']}mm"
            run_prefix = output_prefix.removesuffix(f"_{target}")
            work_paths.append(base / run_prefix / target)
        elif module in {"dynconn", "microparcellation", "networks"}:
            target = f"space-{entities['space']}_smoothing-{entities['smoothing']}mm"
            work_paths.append(
                project_work_derivatives
                / module
                / str(work_item["directory_label"])
                / target
                / sub_id
            )
        elif module == "firstlevels":
            work_paths.append(
                project_work_derivatives
                / "firstlevels"
                / str(work_item["directory_label"])
                / output_prefix
            )

    if include_public:
        derivative_paths.append(
            work_item_record_path(
                registry.paths.project_root,
                str(work_item["configuration_class"]),
                str(work_item["directory_label"]),
                module,
                str(work_item["work_item_key"]),
            )
        )
    allowed_derivative_roots = (derivatives_root, registry.paths.control)
    derivative_root_resolved = {root.resolve(strict=False) for root in allowed_derivative_roots}
    derivative_paths = [
        path
        for path in dict.fromkeys(derivative_paths)
        if any(_is_entry_within(path, root) for root in allowed_derivative_roots)
        and path.resolve(strict=False) not in derivative_root_resolved
    ]
    work_paths = [
        path
        for path in dict.fromkeys(work_paths)
        if _is_entry_within(path, project_work_derivatives)
        and path.resolve(strict=False) != project_work_derivatives.resolve(strict=False)
    ]
    return derivative_paths, work_paths


@dataclass(frozen=True)
class OwnershipIndex:
    """Conservative public-file ownership claims for one project."""

    files: frozenset[Path]
    trees: tuple[Path, ...]

    def owns(self, path: Path) -> bool:
        """Return whether a registered claim or public receipt protects a path."""
        location = _entry_location(path)
        if ".nro" in path.parts:
            return True
        if location in self.files:
            return True
        return any(_is_entry_within(path, tree) for tree in self.trees)


def public_ownership_index(rows: Iterable[dict], *, registry, work_root: Path) -> OwnershipIndex:
    """Index public claims without inventorying files inside owned directory trees."""
    files: set[Path] = set()
    trees: set[Path] = set()
    inventories: dict[Path, tuple[Path, ...]] = {}
    for row in rows:
        public, _private = work_item_paths(
            row,
            registry=registry,
            work_root=work_root,
            inventories=inventories,
        )
        for path in public:
            if _is_entry_within(path, registry.paths.control):
                continue
            resolved = path.resolve(strict=False)
            if str(row["module"]) == "anat" and path.suffix != ".json":
                trees.add(resolved)
            elif path.is_dir() and not path.is_symlink():
                trees.add(resolved)
            else:
                files.add(_entry_location(path))
    return OwnershipIndex(frozenset(files), tuple(sorted(trees)))


def work_ownership_index(rows: Iterable[dict], *, registry, work_root: Path) -> OwnershipIndex:
    """Index private work-item directories as opaque owned trees."""
    trees: set[Path] = set()
    inventories: dict[Path, tuple[Path, ...]] = {}
    for row in rows:
        _public, private = work_item_paths(
            row,
            registry=registry,
            work_root=work_root,
            inventories=inventories,
            include_public=False,
        )
        trees.update(path.resolve(strict=False) for path in private)
    return OwnershipIndex(frozenset(), tuple(sorted(trees)))


def rows_with_public_receipts(project_root: Path, rows: Iterable[dict]) -> list[dict]:
    """Add durable public ownership claims that are absent from the registry view."""
    from nro.orchestration.ownership import read_ownership_records

    result = list(rows)
    known = {str(row["work_item_key"]) for row in result}
    _lineages, receipts, errors = read_ownership_records(project_root.parent, (project_root.name,))
    if errors:
        rendered = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Cannot collect garbage with invalid ownership records:\n{rendered}")
    for receipt, _path in receipts:
        key = str(receipt["work_item_key"])
        if key in known:
            continue
        output = receipt["artifact_contract"]["output"]
        result.append(
            {
                "work_item_key": key,
                "module": str(receipt["module"]),
                "project": str(receipt["project"]),
                "configuration_class": str(receipt["module"]),
                "directory_label": str(receipt["directory_label"]),
                "participant": str(receipt["participant"]),
                "entities_json": json.dumps(receipt["entities"], sort_keys=True),
                "output_root": str(output["root"]),
                "output_prefix": output.get("prefix"),
                "workflow_ids": "",
            }
        )
        known.add(key)
    return result
