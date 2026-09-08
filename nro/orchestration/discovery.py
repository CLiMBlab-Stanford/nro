"""Register existing nro artifacts without creating demand."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from nro.configuration.store import ConfigStore, ResolvedWorkflow
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.manifests import assess_registry
from nro.orchestration.ownership import (
    materialize_instance_specs,
    read_ownership_records,
    write_instance_ownership,
)
from nro.orchestration.planner import Planner
from nro.orchestration.planning_context import ParticipantUnavailableError
from nro.orchestration.registry import RegisteredWorkflow, Registry


_TARGET_RE = re.compile(
    r"(?:^|_)space-([^_]+)_smoothing-([0-9]+)mm(?:_|\.)"
)


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
        path.name[: -len(suffix)]
        for path in (store.root / "workflows").glob(f"*{suffix}")
    )
    return tuple(sorted(identifiers, key=lambda value: (value != "main", value)))


def _target_pairs(
    project_root: Path,
    participant: str,
    registered: RegisteredWorkflow,
    workflow: ResolvedWorkflow,
) -> tuple[tuple[str, int], ...]:
    subject = f"sub-{participant.removeprefix('sub-')}"
    pairs: set[tuple[str, int]] = set()
    for derivative_class in ("clean", "microparcellation", "networks", "firstlevels"):
        configuration_root = (
            project_root
            / "derivatives"
            / derivative_class
            / registered.directories[derivative_class]
        )
        if derivative_class == "clean":
            roots = (configuration_root / subject,)
        elif derivative_class == "firstlevels":
            roots = tuple(configuration_root.glob(f"space-*_smoothing-*mm/*/node-*/{subject}"))
        else:
            roots = tuple(configuration_root.glob(f"space-*_smoothing-*mm/{subject}"))
        for root in roots:
            if not root.is_dir():
                continue
            paths = root.rglob("*")
            for path in paths:
                if not path.is_file():
                    continue
                match = _TARGET_RE.search(path.name)
                if match:
                    pairs.add((match.group(1), int(match.group(2))))
    published_spaces = {
        str(value)
        for value in workflow.configuration("preprocessing").values["func"][
            "output_spaces"
        ]
    }
    return tuple(
        sorted(
            (space, smoothing)
            for space, smoothing in pairs
            if space in published_spaces
        )
    )


def _owned_files(
    spec: InstanceSpec,
    inventories: dict[Path, tuple[Path, ...]],
) -> tuple[Path, ...]:
    root = spec.output_root.resolve()
    if root not in inventories:
        inventories[root] = (
            tuple(path for path in root.rglob("*") if path.is_file())
            if root.is_dir()
            else ()
        )
    prefix = str(spec.output_prefix or "")
    if not prefix:
        return ()
    return tuple(
        path
        for path in inventories[root]
        if path.name == prefix
        or path.name.startswith(f"{prefix}_")
        or path.name == f".{prefix}_complete"
    )


def _dependency_closure(
    existing: set[str], specs: Mapping[str, InstanceSpec]
) -> set[str]:
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
    known_lineages = {
        str(record["lineage_fingerprint"]) for record in lineage_records
    }
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
    lineage_ids = (
        registry.register_owned_lineages(lineage_records) if lineage_records else {}
    )
    owned_specs, materialization_errors = materialize_instance_specs(
        registry, instance_records, lineage_ids
    )
    workflows = {
        workflow_id: store.resolve(workflow_id)
        for workflow_id in _workflow_ids(store)
    }
    registered = {
        workflow_id: registry.register_workflow(workflow)
        for workflow_id, workflow in workflows.items()
    }
    all_specs: dict[str, InstanceSpec] = {spec.key: spec for spec in owned_specs}
    existing_keys: set[str] = set(all_specs)
    fallback_existing: set[str] = set()
    inventories: dict[Path, tuple[Path, ...]] = {}
    unavailable: list[str] = [*ownership_errors, *materialization_errors]

    for project, participants in sorted(inventory.items()):
        project_registry = Registry.for_project(project, bids_root=bids_root)
        planner = Planner(project_registry, bids_root=bids_root)
        for workflow_id, workflow in workflows.items():
            registration = registered[workflow_id]
            for participant in participants:
                pairs = _target_pairs(
                    bids_root / project,
                    participant,
                    registration,
                    workflow,
                )
                spaces = tuple(dict.fromkeys(space for space, _ in pairs)) or (
                    DEFAULT_SPACE,
                )
                smoothing = tuple(
                    dict.fromkeys(value for _, value in pairs)
                ) or (DEFAULT_SMOOTHING_MM,)
                planned: list[InstanceSpec] = []
                try:
                    planned.extend(
                        planner.plan_subject(
                            project=project,
                            participant=participant,
                            module="anat",
                            workflow=workflow,
                            registered=registration,
                            memory_gb=memory_gb,
                            max_memory_gb=max_memory_gb,
                        )
                    )
                    planned.extend(
                        planner.plan_subject(
                            project=project,
                            participant=participant,
                            module="networks",
                            workflow=workflow,
                            registered=registration,
                            spaces=spaces,
                            smoothing_levels=smoothing,
                            memory_gb=memory_gb,
                            max_memory_gb=max_memory_gb,
                        )
                    )
                except (ParticipantUnavailableError, FileNotFoundError, ValueError) as error:
                    unavailable.append(
                        f"{project}/sub-{participant} ({workflow_id}): {error}"
                    )
                firstlevels_root = bids_root / project / "derivatives" / "firstlevels" / registration.directories["firstlevels"]
                if firstlevels_root.is_dir():
                    try:
                        planned.extend(planner.plan_subject(
                            project=project, participant=participant, module="firstlevels",
                            workflow=workflow, registered=registration, spaces=spaces,
                            model_sets=(),
                            smoothing_levels=smoothing, memory_gb=memory_gb,
                            max_memory_gb=max_memory_gb,
                        ))
                    except (ParticipantUnavailableError, FileNotFoundError, ValueError) as error:
                        unavailable.append(f"{project}/sub-{participant} ({workflow_id}, firstlevels): {error}")
                for spec in planned:
                    all_specs[spec.key] = spec
                    if _owned_files(spec, inventories):
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
        unavailable.append(
            f"Stored ownership record {key} lacks a complete dependency closure"
        )
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
        project_registry = Registry.for_project(
            all_specs[key].project, bids_root=bids_root
        )
        write_instance_ownership(project_registry, registered_ids[key])
    return ArtifactDiscovery(
        workflows=len(workflows),
        artifacts=len(existing_keys),
        instances=len(selected_keys),
        unavailable=tuple(unavailable),
    )
