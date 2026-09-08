"""Central construction of deterministic scientific instance graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.configuration.store import ResolvedWorkflow
from nro.engine.bids import discover_raw_runs, matches_filter, matches_selectors
from nro.engine.images import image_source_paths
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE
from nro.orchestration.catalog import module_descriptor, modules_through, normalize_module
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import (
    ParticipantUnavailableError,
    SubjectPlanningContext,
)
from nro.orchestration.registry import RegisteredWorkflow, Registry
from nro.orchestration.selection import discover_bids_participants


@dataclass(frozen=True)
class RequestPlan:
    """One project/workflow/terminal-module request ready for registration."""

    project: str
    registry: Registry
    workflow_id: str
    registered: RegisteredWorkflow
    module: str
    participants: tuple[str, ...]
    instances: tuple[InstanceSpec, ...]
    terminal_keys: tuple[str, ...]


def _minimal_requests(requests: Sequence[RequestPlan]) -> tuple[RequestPlan, ...]:
    """Remove targets covered by other targets in one project and workflow.

    Compare instance dependencies, not module names: a downstream selection
    may consume only some runs or participants. Remaining requests retain only
    their terminal instances and the dependencies needed to produce them.
    """
    instances = {
        instance.key: instance
        for request in requests
        for instance in request.instances
    }

    def closure(keys: Sequence[str]) -> set[str]:
        required: set[str] = set()
        pending = list(keys)
        while pending:
            key = pending.pop()
            if key not in required:
                required.add(key)
                pending.extend(instances[key].dependencies)
        return required

    reachable = closure([key for request in requests for key in request.terminal_keys])
    covered = {
        dependency
        for key in reachable
        for dependency in instances[key].dependencies
    }
    result = []
    for request in requests:
        terminal_keys = tuple(key for key in request.terminal_keys if key not in covered)
        if not terminal_keys:
            continue
        required = closure(terminal_keys)
        participants = {instances[key].participant for key in terminal_keys}
        result.append(replace(
            request,
            terminal_keys=terminal_keys,
            instances=tuple(item for item in request.instances if item.key in required),
            participants=tuple(value for value in request.participants if value in participants),
        ))
    return tuple(result)


@dataclass(frozen=True)
class UnavailableSelection:
    """One participant/module selection omitted because inputs are insufficient."""

    project: str
    participant: str
    module: str
    reason: str


@dataclass(frozen=True)
class PlanningResult:
    """Complete cross-project result of one user selection."""

    requests: tuple[RequestPlan, ...]
    instances: Mapping[str, InstanceSpec]
    matched_participants: Mapping[str, tuple[str, ...]]
    present_participants: frozenset[str]
    unavailable: tuple[UnavailableSelection, ...]

    @property
    def projects(self) -> tuple[str, ...]:
        """Return project IDs represented in matched participant selections."""
        return tuple(dict.fromkeys(request.project for request in self.requests))

    @property
    def participant_count(self) -> int:
        """Return the total number of matched participants across projects."""
        return len(
            {
                (request.project, participant)
                for request in self.requests
                for participant in request.participants
            }
        )


class Planner:
    """Construct complete instance DAGs using the closed module catalog."""

    def __init__(self, registry: Registry, *, bids_root: str | Path) -> None:
        """Bind a registry and resolved BIDS root for subsequent planning requests."""
        self.registry = registry
        self.bids_root = Path(bids_root).expanduser().resolve()

    @staticmethod
    def participants(project_root: Path, requested: Sequence[str]) -> tuple[str, ...]:
        """Return matching source-BIDS participants for one project."""
        available = set(discover_bids_participants(project_root))
        return tuple(
            sorted(available.intersection(requested) if requested else available)
        )

    def plan(
        self,
        *,
        projects: Sequence[str],
        requested_participants: Sequence[str],
        modules: Sequence[str],
        workflows: Mapping[str, ResolvedWorkflow],
        registered_workflows: Mapping[str, RegisteredWorkflow],
        selectors: Mapping[str, tuple[str, ...] | None] | None,
        spaces: tuple[str, ...],
        smoothing_levels: tuple[int, ...],
        memory_gb: int,
        max_memory_gb: int,
        models: Sequence[str] = (),
        model_sets: Sequence[str] | None = None,
    ) -> PlanningResult:
        """Construct requests, dropping targets covered within each workflow."""
        request_plans: list[RequestPlan] = []
        present: set[str] = set()
        unavailable: list[UnavailableSelection] = []
        normalized_modules = tuple(dict.fromkeys(normalize_module(module) for module in modules))

        for project in projects:
            project_registry = Registry.for_project(project, bids_root=self.bids_root)
            project_planner = Planner(project_registry, bids_root=self.bids_root)
            participants = self.participants(
                self.bids_root / project, requested_participants
            )
            if not participants:
                continue
            present.update(participants)
            for workflow_id, workflow in workflows.items():
                registered = registered_workflows[workflow_id]
                workflow_requests: list[RequestPlan] = []
                for module in normalized_modules:
                    group_instances: dict[str, InstanceSpec] = {}
                    terminal_keys: list[str] = []
                    group_participants: list[str] = []
                    for participant in participants:
                        try:
                            planned = project_planner.plan_subject(
                                project=project,
                                participant=participant,
                                module=module,
                                workflow=workflow,
                                registered=registered,
                                selectors=selectors,
                                spaces=spaces,
                                smoothing_levels=smoothing_levels,
                                memory_gb=memory_gb,
                                max_memory_gb=max_memory_gb,
                                models=models,
                                model_sets=model_sets,
                            )
                        except ParticipantUnavailableError as error:
                            unavailable.append(
                                UnavailableSelection(
                                    project=project,
                                    participant=participant,
                                    module=module,
                                    reason=str(error),
                                )
                            )
                            continue
                        group_participants.append(participant)
                        for instance in planned:
                            group_instances[instance.key] = instance
                            if instance.module == module:
                                terminal_keys.append(instance.key)
                    if not terminal_keys:
                        continue
                    unique_participants = tuple(dict.fromkeys(group_participants))
                    workflow_requests.append(
                        RequestPlan(
                            project=project,
                            registry=project_registry,
                            workflow_id=workflow_id,
                            registered=registered,
                            module=module,
                            participants=unique_participants,
                            instances=tuple(group_instances.values()),
                            terminal_keys=tuple(dict.fromkeys(terminal_keys)),
                        )
                    )
                request_plans.extend(_minimal_requests(workflow_requests))
        all_instances: dict[str, InstanceSpec] = {}
        matched: dict[str, list[str]] = {}
        for request in request_plans:
            all_instances.update((instance.key, instance) for instance in request.instances)
            matched.setdefault(request.project, [])
            matched[request.project] = list(dict.fromkeys(
                (*matched[request.project], *request.participants)
            ))
        return PlanningResult(
            requests=tuple(request_plans),
            instances=all_instances,
            matched_participants={
                project: tuple(participants)
                for project, participants in matched.items()
            },
            present_participants=frozenset(present),
            unavailable=tuple(unavailable),
        )

    def register_requests(
        self,
        plan: PlanningResult,
        *,
        selectors: Mapping[str, Any],
        concurrency: int,
        partition: str | None,
    ) -> tuple[str, ...]:
        """Create durable demand for every request group in a plan."""
        return tuple(
            request.registry.create_request(
                registered=request.registered,
                target_module=request.module,
                selectors={**selectors, "participants": list(request.participants)},
                instances=request.instances,
                terminal_instance_keys=request.terminal_keys,
                concurrency=concurrency,
                partition=partition,
            )
            for request in plan.requests
        )

    def plan_subject(
        self,
        *,
        project: str,
        participant: str,
        module: str,
        workflow: ResolvedWorkflow,
        registered: RegisteredWorkflow,
        selectors: Mapping[str, tuple[str, ...] | None] | None = None,
        spaces: tuple[str, ...] | None = None,
        smoothing_levels: tuple[int, ...] | None = None,
        memory_gb: int = 32,
        max_memory_gb: int = 256,
        models: Sequence[str] = (),
        model_sets: Sequence[str] | None = None,
    ) -> tuple[InstanceSpec, ...]:
        """Build the requested logical instance graph for one participant."""
        target = normalize_module(module)
        if (target in {"microparcellation", "networks"} and selectors) or (
            target == "firstlevels" and set(selectors or {}) - {"task"}
        ):
            raise ValueError(
                "Run selectors cannot define a subject-level multirun derivative. "
                "Use the module configuration's input selection instead."
            )

        participant = participant.removeprefix("sub-")
        sub_id = f"sub-{participant}"
        project_root = self.bids_root / project
        subject_dir = project_root / sub_id
        runs = discover_raw_runs(subject_dir)
        target_descriptor = module_descriptor(target)
        task_models = None
        if target_descriptor.select_models is not None:
            task_models = target_descriptor.select_models(tasks=(selectors or {}).get("task", ()), models=models, model_sets=model_sets)
            tasks = {identifier.split("/")[0] for identifier in task_models}
            runs = tuple(run for run in runs if run.entities.get("task") in tasks)
        elif target_descriptor.select_runs is not None:
            runs = target_descriptor.select_runs(
                runs, workflow.configuration(target_descriptor.configuration_class).values, participant,
            )
        if selectors:
            runs = tuple(
                run for run in runs if matches_selectors(run.entities, selectors)
            )
        if target in {"microparcellation", "networks"}:
            aggregate_filter = workflow.configuration("microparcellation").values.get(
                "input_filter", {}
            )
            runs = tuple(
                run for run in runs if matches_filter(run.entities, aggregate_filter)
            )
        if target != "anat" and not runs:
            raise ParticipantUnavailableError(
                f"No raw BOLD runs matched under {subject_dir}"
            )

        requested_spaces = tuple(dict.fromkeys(spaces or (DEFAULT_SPACE,)))
        requested_smoothing = tuple(
            dict.fromkeys(smoothing_levels or (DEFAULT_SMOOTHING_MM,))
        )
        target_pairs: tuple[tuple[str, int], ...] = ()
        if target not in {"anat", "func"}:
            if not requested_spaces:
                raise ValueError("At least one space must be requested")
            if not requested_smoothing or any(value < 0 for value in requested_smoothing):
                raise ValueError("Smoothing levels must be nonnegative integers")
            published_spaces = tuple(
                str(value)
                for value in workflow.configuration("preprocessing").values["func"][
                    "output_spaces"
                ]
            )
            unavailable = tuple(
                space for space in requested_spaces if space not in published_spaces
            )
            if unavailable:
                raise ValueError(
                    "Requested space(s) are not published by preprocessing: "
                    + ", ".join(f"space-{space}" for space in unavailable)
                )
            target_pairs = tuple(
                (space, smoothing)
                for space in requested_spaces
                for smoothing in requested_smoothing
            )

        context = SubjectPlanningContext(
            project=project,
            participant=participant,
            sub_id=sub_id,
            bids_root=self.bids_root,
            project_root=project_root,
            subject_dir=subject_dir,
            workflow=workflow,
            registered=registered,
            registry=self.registry,
            runs=runs,
            aggregate_source_inputs=tuple(
                dict.fromkeys(
                    path for run in runs for path in image_source_paths(run.path)
                )
            ),
            target_pairs=target_pairs,
            memory_gb=memory_gb,
            max_memory_gb=max_memory_gb,
            task_models=task_models,
        )
        descriptors = modules_through(target)
        planned: dict[str, tuple[InstanceSpec, ...]] = {}
        for descriptor in descriptors:
            planned[descriptor.name] = descriptor.plan(context, planned, descriptor)
        return tuple(
            instance
            for descriptor in descriptors
            for instance in planned[descriptor.name]
        )


def build_subject_instances(
    *,
    project: str,
    participant: str,
    module: str,
    workflow: ResolvedWorkflow,
    registered: RegisteredWorkflow,
    registry: Registry,
    selectors: Mapping[str, tuple[str, ...] | None] | None = None,
    spaces: tuple[str, ...] | None = None,
    smoothing_levels: tuple[int, ...] | None = None,
    bids_root: str | Path,
    memory_gb: int = 32,
    max_memory_gb: int = 256,
    models: Sequence[str] = (),
    model_sets: Sequence[str] | None = None,
) -> tuple[InstanceSpec, ...]:
    """Convenience entry point for planning one participant."""
    return Planner(registry, bids_root=bids_root).plan_subject(
        project=project,
        participant=participant,
        module=module,
        workflow=workflow,
        registered=registered,
        selectors=selectors,
        spaces=spaces,
        smoothing_levels=smoothing_levels,
        memory_gb=memory_gb,
        max_memory_gb=max_memory_gb,
        models=models,
        model_sets=model_sets,
    )
