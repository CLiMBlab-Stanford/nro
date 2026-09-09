"""Register existing nro artifacts without creating demand."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from nro.configuration.store import DERIVATIVE_CLASSES, ConfigStore
from nro.engine.bids import ENTITY_ORDER, parse_bids_entities
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.manifests import assess_registry
from nro.orchestration.ownership import (
    materialize_instance_specs,
    read_ownership_records,
    write_instance_ownership,
)
from nro.orchestration.planner import Planner
from nro.orchestration.planning_context import ParticipantUnavailableError
from nro.orchestration.registry import Registry

_TARGET_RE = re.compile(r"(?:^|_)space-([^_]+)_smoothing-([0-9]+)mm(?:_|\.|$)")


@dataclass(frozen=True)
class ArtifactDiscovery:
    """Summary of one registry bootstrap scan."""

    workflows: int
    artifacts: int
    instances: int
    unavailable: tuple[str, ...]


def _workflow_ids(store: ConfigStore) -> tuple[str, ...]:
    suffix = "_workflow.yml"
    identifiers = sorted(
        path.name[: -len(suffix)] for path in (store.root / "workflows").glob(f"*{suffix}")
    )
    return tuple(sorted(identifiers, key=lambda value: (value != "main", value)))


@dataclass(frozen=True, order=True)
class _RecoveryTarget:
    derivative_class: str
    directory: str
    module: str
    participant: str
    entities: tuple[tuple[str, str], ...]


def _matches_prefix(path: Path, prefix: str) -> bool:
    return bool(prefix) and (
        path.name == prefix
        or path.name.startswith(f"{prefix}_")
        or path.name == f".{prefix}_complete"
    )


def _recovery_target(path: Path, root: Path, derivative_class: str) -> _RecoveryTarget | None:
    if path.name.startswith(".") and not path.name.endswith("_complete"):
        return None
    parts = path.relative_to(root).parts
    entities = parse_bids_entities(path.name.lstrip("."))
    participant = entities.get("sub")
    if not participant or f"sub-{participant}" not in parts[:-1]:
        return None
    selected: dict[str, str] = {}
    module = derivative_class
    if derivative_class == "preprocessing":
        if len(parts) > 2 and parts[1] == "anat":
            module = "anat"
        elif (len(parts) > 2 and parts[1] == "func") or (
            len(parts) > 3 and parts[1].startswith("ses-") and parts[2] == "func"
        ):
            module = "func"
        else:
            return None
    if module in {"func", "clean"}:
        selected = {key: entities[key] for key in ENTITY_ORDER if key in entities}
        if "task" not in selected:
            return None
        if "ses" not in selected:
            session = next((part[4:] for part in parts[:-1] if part.startswith("ses-")), None)
            if session:
                selected["ses"] = session
    if module not in {"anat", "func"}:
        pair = _TARGET_RE.search(path.name) if module == "clean" else _TARGET_RE.search(parts[0])
        if pair is None:
            return None
        selected.update(space=pair.group(1), smoothing=str(int(pair.group(2))))
    if module == "firstlevels":
        if "model" not in entities or "task" not in entities:
            return None
        selected.update(model=entities["model"], task=entities["task"])
    return _RecoveryTarget(
        derivative_class, root.name, module, participant, tuple(sorted(selected.items()))
    )


def _unrecorded_targets(
    project_root: Path,
    owned: Sequence[InstanceSpec],
    directories: Mapping[str, set[str]],
) -> dict[_RecoveryTarget, list[Path]]:
    """Index existing public files not covered by recovered ownership records."""
    prefixes: dict[Path, set[str]] = {}
    for spec in owned:
        prefixes.setdefault(spec.output_root, set()).add(str(spec.output_prefix or ""))
    complete_roots = {
        spec.output_root
        for spec in owned
        if spec.module in {"anat", "dynconn", "microparcellation", "networks"}
    }
    targets: dict[_RecoveryTarget, list[Path]] = {}
    for derivative_class in DERIVATIVE_CLASSES:
        for directory in sorted(directories[derivative_class]):
            root = project_root / "derivatives" / derivative_class / directory
            if not root.is_dir():
                continue
            if derivative_class in {"preprocessing", "clean"}:
                subjects = root.glob("sub-*")
            elif derivative_class == "firstlevels":
                subjects = root.glob("space-*_smoothing-*mm/*/node-*/sub-*")
            else:
                subjects = root.glob("space-*_smoothing-*mm/sub-*")
            for subject in subjects:
                for parent, children, filenames in os.walk(subject):
                    parent = Path(parent)
                    if parent in complete_roots:
                        children[:] = []
                        continue
                    children[:] = [
                        name for name in children if name != ".nro" and name != "derivatives"
                    ]
                    for filename in filenames:
                        path = parent / filename
                        if any(
                            _matches_prefix(path, prefix)
                            for ancestor in path.parents
                            for prefix in prefixes.get(ancestor, ())
                        ):
                            continue
                        if not path.is_file():
                            continue
                        target = _recovery_target(path, root, derivative_class)
                        if target is not None:
                            targets.setdefault(target, []).append(path)
    return targets


def _dependency_closure(existing: set[str], specs: Mapping[str, InstanceSpec]) -> set[str]:
    selected = set(existing)
    pending = list(existing)
    while pending:
        key = pending.pop()
        for dependency in specs[key].dependencies:
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return selected


def register_existing_artifacts(
    registry: Registry,
    *,
    bids_root: Path,
    inventory: Mapping[str, Sequence[str]],
    store: ConfigStore | None = None,
    memory_gb: int = 32,
    max_memory_gb: int = 256,
) -> ArtifactDiscovery:
    """Scan controlled roots and register only artifacts represented on disk."""
    store = store or ConfigStore()
    lineage_records, instance_records, ownership_errors = read_ownership_records(
        bids_root, inventory
    )
    known_lineages = {str(record["lineage_fingerprint"]) for record in lineage_records}
    incomplete_lineages = {
        str(record["lineage_fingerprint"])
        for record in lineage_records
        if any(
            str(upstream["lineage_fingerprint"]) not in known_lineages
            for upstream in record["upstream"]
        )
    }
    while True:
        downstream = {
            str(record["lineage_fingerprint"])
            for record in lineage_records
            if str(record["lineage_fingerprint"]) not in incomplete_lineages
            and any(
                str(upstream["lineage_fingerprint"]) in incomplete_lineages
                for upstream in record["upstream"]
            )
        }
        if not downstream:
            break
        incomplete_lineages.update(downstream)
    if incomplete_lineages:
        ownership_errors.extend(
            f"Stored lineage {value} lacks a complete upstream lineage chain"
            for value in sorted(incomplete_lineages)
        )
        lineage_records = [
            record
            for record in lineage_records
            if str(record["lineage_fingerprint"]) not in incomplete_lineages
        ]
        instance_records = [
            item
            for item in instance_records
            if str(item[0]["lineage_fingerprint"]) not in incomplete_lineages
        ]
    lineage_ids = registry.register_owned_lineages(lineage_records) if lineage_records else {}
    owned_specs, materialization_errors = materialize_instance_specs(
        registry, instance_records, lineage_ids
    )
    workflows = {workflow_id: store.resolve(workflow_id) for workflow_id in _workflow_ids(store)}
    registered = {
        workflow_id: registry.register_workflow(workflow)
        for workflow_id, workflow in workflows.items()
    }
    all_specs: dict[str, InstanceSpec] = {spec.key: spec for spec in owned_specs}
    existing_keys: set[str] = set(all_specs)
    fallback_existing: set[str] = set()
    unavailable: list[str] = [*ownership_errors, *materialization_errors]
    directories = {
        name: {item.directories[name] for item in registered.values()}
        for name in DERIVATIVE_CLASSES
    }

    for project in sorted(inventory):
        project_registry = Registry.for_project(project, bids_root=bids_root)
        planner = Planner(project_registry, bids_root=bids_root)
        targets = _unrecorded_targets(
            bids_root / project,
            [spec for spec in owned_specs if spec.project == project],
            directories,
        )
        for target, files in sorted(targets.items()):
            lineages_seen: set[str] = set()
            for workflow_id, workflow in workflows.items():
                registration = registered[workflow_id]
                if registration.directories[target.derivative_class] != target.directory:
                    continue
                lineage = registration.lineage_fingerprints[target.derivative_class]
                if lineage in lineages_seen:
                    continue
                lineages_seen.add(lineage)
                entities = dict(target.entities)
                selectors = (
                    {key: (value,) for key, value in entities.items() if key in ENTITY_ORDER}
                    if target.module in {"func", "clean"}
                    else None
                )
                try:
                    planned = planner.plan_subject(
                        project=project,
                        participant=target.participant,
                        module=target.module,
                        workflow=workflow,
                        registered=registration,
                        selectors=selectors,
                        spaces=(entities["space"],) if "space" in entities else None,
                        smoothing_levels=(int(entities["smoothing"]),)
                        if "smoothing" in entities
                        else None,
                        models=(f"{entities['task']}/{entities['model']}",)
                        if target.module == "firstlevels"
                        else (),
                        model_sets=(),
                        memory_gb=memory_gb,
                        max_memory_gb=max_memory_gb,
                    )
                except (ParticipantUnavailableError, FileNotFoundError, ValueError) as error:
                    unavailable.append(
                        f"{project}/sub-{target.participant} ({workflow_id}, {target.module}): {error}"
                    )
                    continue
                for spec in planned:
                    all_specs.setdefault(spec.key, spec)
                    if spec.module == target.module and any(
                        path.is_relative_to(spec.output_root)
                        and _matches_prefix(path, str(spec.output_prefix or ""))
                        for path in files
                    ):
                        existing_keys.add(spec.key)
                        fallback_existing.add(spec.key)

    incomplete = {
        key
        for key, spec in all_specs.items()
        if any(dependency not in all_specs for dependency in spec.dependencies)
    }
    while True:
        downstream = {
            key
            for key, spec in all_specs.items()
            if key not in incomplete
            and any(dependency in incomplete for dependency in spec.dependencies)
        }
        if not downstream:
            break
        incomplete.update(downstream)
    for key in sorted(incomplete):
        unavailable.append(f"Stored ownership record {key} lacks a complete dependency closure")
        all_specs.pop(key, None)
        existing_keys.discard(key)
        fallback_existing.discard(key)

    selected_keys = _dependency_closure(existing_keys, all_specs)
    by_project: dict[str, list[InstanceSpec]] = {}
    for key in selected_keys:
        spec = all_specs[key]
        by_project.setdefault(spec.project, []).append(spec)
    registered_ids: dict[str, int] = {}
    for project, specs in by_project.items():
        registered_ids.update(
            Registry.for_project(project, bids_root=bids_root).register_instances(specs)
        )
    if selected_keys:
        assess_registry(registry, projects=tuple(by_project))
    for key in sorted(fallback_existing.intersection(registered_ids)):
        project_registry = Registry.for_project(all_specs[key].project, bids_root=bids_root)
        write_instance_ownership(project_registry, registered_ids[key])
    return ArtifactDiscovery(
        workflows=len(workflows),
        artifacts=len(existing_keys),
        instances=len(selected_keys),
        unavailable=tuple(unavailable),
    )
