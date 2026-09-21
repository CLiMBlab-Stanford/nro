"""Central construction of deterministic scientific work-item graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.configuration.markup import MarkupStore
from nro.configuration.site import with_site_read_cache
from nro.configuration.store import ResolvedWorkflow, fingerprint
from nro.engine.bids import (
    ENTITY_ORDER,
    discover_raw_runs,
    matches_filter,
    matches_selectors,
    with_bids_metadata_cache,
)
from nro.engine.cli import matches_module_lineage
from nro.engine.image_paths import image_source_paths
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE, supported_output_spaces
from nro.orchestration.catalog import module_descriptor, modules_through, normalize_module
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import (
    ParticipantUnavailableError,
    SubjectPlanningContext,
)
from nro.orchestration.registry import Registry
from nro.orchestration.selection import discover_bids_participants
from nro.orchestration.workflow_registry import RegisteredWorkflow, WorkflowRegistry


@dataclass(frozen=True)
class RequestPlan:
    """One project/workflow/terminal-module request ready for registration."""

    project: str
    workflow_id: str
    registered: RegisteredWorkflow
    module: str
    participants: tuple[str, ...]
    work_items: tuple[WorkItemSpec, ...]
    terminal_keys: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredTarget:
    """One exact registry identity to recompile without expanding selectors."""

    project: str
    participant: str
    module: str
    workflow_id: str
    work_item_key: str
    entities: Mapping[str, Any]
    memory_gb: int | None = None
    max_memory_gb: int | None = None


def _minimal_requests(requests: Sequence[RequestPlan]) -> tuple[RequestPlan, ...]:
    """Remove targets covered by other targets in one project and workflow.

    Compare work_item dependencies, not module names: a downstream selection
    may consume only some runs or participants. Remaining requests retain only
    their terminal work_items and the dependencies needed to produce them.
    """
    groups: dict[tuple[str, str], list[RequestPlan]] = {}
    for request in requests:
        groups.setdefault((request.project, request.workflow_id), []).append(request)
    return tuple(request for group in groups.values() for request in _minimal_request_group(group))


def _minimal_request_group(requests: Sequence[RequestPlan]) -> tuple[RequestPlan, ...]:
    """Remove covered targets within one project and workflow."""
    work_items = {
        work_item.key: work_item for request in requests for work_item in request.work_items
    }

    def closure(keys: Sequence[str]) -> set[str]:
        required: set[str] = set()
        pending = list(keys)
        while pending:
            key = pending.pop()
            if key not in required:
                required.add(key)
                pending.extend(work_items[key].dependencies)
        return required

    reachable = closure([key for request in requests for key in request.terminal_keys])
    covered = {dependency for key in reachable for dependency in work_items[key].dependencies}
    result = []
    for request in requests:
        terminal_keys = tuple(key for key in request.terminal_keys if key not in covered)
        if not terminal_keys:
            continue
        required = closure(terminal_keys)
        participants = {work_items[key].participant for key in terminal_keys}
        result.append(
            replace(
                request,
                terminal_keys=terminal_keys,
                work_items=tuple(item for item in request.work_items if item.key in required),
                participants=tuple(
                    value for value in request.participants if value in participants
                ),
            )
        )
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
    work_items: Mapping[str, WorkItemSpec]
    matched_participants: Mapping[str, tuple[str, ...]]
    present_participants: frozenset[str]
    unavailable: tuple[UnavailableSelection, ...]
    source_digest: str
    site_settings: Mapping[str, Any]

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
    """Construct complete work-item DAGs using the closed module catalog."""

    def __init__(self, registry: WorkflowRegistry, *, bids_root: str | Path) -> None:
        """Bind scientific workflow records and raw BIDS without opening a scheduler."""
        from nro.configuration.site import definitions_roots, settings

        self.registry = registry
        self.bids_root = Path(bids_root).expanduser().resolve()
        self._site_values = settings()[0]
        self._definitions_roots = definitions_roots()
        self._module_plan_cache: dict[tuple[object, ...], tuple[WorkItemSpec, ...]] = {}

    @staticmethod
    def participants(project_root: Path, requested: Sequence[str]) -> tuple[str, ...]:
        """Return matching source-BIDS participants for one project."""
        available = set(discover_bids_participants(project_root))
        return tuple(sorted(available.intersection(requested) if requested else available))

    @with_site_read_cache
    @with_bids_metadata_cache
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
        lineage_ids: Sequence[str] = (),
    ) -> PlanningResult:
        """Construct requests from an explicit user selection."""
        from nro.configuration.site import definitions_root, settings
        from nro.orchestration.source_snapshots import execution_source_root, source_fingerprint

        source_digest = source_fingerprint(execution_source_root())
        site_settings = {**settings()[0], "definitions": str(definitions_root())}
        request_plans: list[RequestPlan] = []
        present: set[str] = set()
        unavailable: list[UnavailableSelection] = []
        normalized_modules = tuple(dict.fromkeys(normalize_module(module) for module in modules))

        for project in projects:
            participants = self.participants(self.bids_root / project, requested_participants)
            if not participants:
                continue
            present.update(participants)
            for workflow_id, workflow in workflows.items():
                registered = registered_workflows[workflow_id]
                workflow_requests: list[RequestPlan] = []
                for module in normalized_modules:
                    directory = registered.directory_for(
                        module_descriptor(module).configuration_class
                    )
                    if not matches_module_lineage(module, directory, lineage_ids):
                        continue
                    group_work_items: dict[str, WorkItemSpec] = {}
                    terminal_keys: list[str] = []
                    group_participants: list[str] = []
                    for participant in participants:
                        try:
                            planned = self.plan_subject(
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
                        for work_item in planned:
                            group_work_items[work_item.key] = work_item
                            if work_item.module == module:
                                terminal_keys.append(work_item.key)
                    if not terminal_keys:
                        continue
                    unique_participants = tuple(dict.fromkeys(group_participants))
                    workflow_requests.append(
                        RequestPlan(
                            project=project,
                            workflow_id=workflow_id,
                            registered=registered,
                            module=module,
                            participants=unique_participants,
                            work_items=tuple(group_work_items.values()),
                            terminal_keys=tuple(dict.fromkeys(terminal_keys)),
                        )
                    )
                request_plans.extend(_minimal_requests(workflow_requests))
        all_work_items: dict[str, WorkItemSpec] = {}
        matched: dict[str, list[str]] = {}
        for request in request_plans:
            all_work_items.update((work_item.key, work_item) for work_item in request.work_items)
            matched.setdefault(request.project, [])
            matched[request.project] = list(
                dict.fromkeys((*matched[request.project], *request.participants))
            )
        return PlanningResult(
            requests=tuple(request_plans),
            work_items=all_work_items,
            matched_participants={
                project: tuple(participants) for project, participants in matched.items()
            },
            present_participants=frozenset(present),
            unavailable=tuple(unavailable),
            source_digest=source_digest,
            site_settings=site_settings,
        )

    @with_site_read_cache
    @with_bids_metadata_cache
    def plan_registered_targets(
        self,
        targets: Sequence[RegisteredTarget],
        *,
        workflows: Mapping[str, ResolvedWorkflow],
        registered_workflows: Mapping[str, RegisteredWorkflow],
        memory_gb: int,
        max_memory_gb: int,
    ) -> PlanningResult:
        """Recompile exact registered identities without broad selector expansion."""
        from nro.configuration.site import definitions_root, settings
        from nro.orchestration.source_snapshots import execution_source_root, source_fingerprint

        source_digest = source_fingerprint(execution_source_root())
        site_settings = {**settings()[0], "definitions": str(definitions_root())}
        requests: list[RequestPlan] = []
        work_items: dict[str, WorkItemSpec] = {}
        present: set[str] = set()
        unavailable: list[UnavailableSelection] = []
        matched: dict[str, list[str]] = {}
        seen: set[tuple[str, str, str, str, str]] = set()
        planned_cache: dict[tuple[object, ...], tuple[WorkItemSpec, ...]] = {}
        for target in targets:
            identity = (
                target.project,
                target.participant,
                target.module,
                target.workflow_id,
                target.work_item_key,
            )
            if identity in seen:
                continue
            seen.add(identity)
            workflow = workflows[target.workflow_id]
            registered = registered_workflows[target.workflow_id]
            entities = dict(target.entities)
            selectors = {key: (value,) for key, value in entities.items() if key in ENTITY_ORDER}
            if target.module in {"anat", "dynconn", "microparcellation", "networks"}:
                selectors = None
            elif target.module == "firstlevels":
                selectors = {"task": (entities["task"],)}
            spaces = (entities["space"],) if "space" in entities else None
            smoothing = (int(entities["smoothing"]),) if "smoothing" in entities else None
            models = (
                (f"{entities['task']}/{entities['model']}",)
                if target.module == "firstlevels"
                else ()
            )
            selected_memory = target.memory_gb if target.memory_gb is not None else memory_gb
            selected_max_memory = (
                target.max_memory_gb if target.max_memory_gb is not None else max_memory_gb
            )
            cache_key = (
                target.project,
                target.participant,
                target.module,
                target.workflow_id,
                tuple(
                    sorted(
                        (key, None if value is None else tuple(value))
                        for key, value in (selectors or {}).items()
                    )
                ),
                spaces,
                smoothing,
                models,
                selected_memory,
                selected_max_memory,
            )
            try:
                planned = planned_cache.get(cache_key)
                if planned is None:
                    planned = self.plan_subject(
                        project=target.project,
                        participant=target.participant,
                        module=target.module,
                        workflow=workflow,
                        registered=registered,
                        selectors=selectors,
                        spaces=spaces,
                        smoothing_levels=smoothing,
                        models=models,
                        model_sets=(),
                        memory_gb=selected_memory,
                        max_memory_gb=selected_max_memory,
                    )
                    planned_cache[cache_key] = planned
            except (FileNotFoundError, ParticipantUnavailableError) as error:
                unavailable.append(
                    UnavailableSelection(
                        target.project,
                        target.participant,
                        target.module,
                        str(error),
                    )
                )
                continue
            terminal = tuple(
                spec.key
                for spec in planned
                if spec.module == target.module and spec.key == target.work_item_key
            )
            if not terminal:
                unavailable.append(
                    UnavailableSelection(
                        target.project,
                        target.participant,
                        target.module,
                        "The registered work item is not produced by the current definitions",
                    )
                )
                continue
            present.add(target.participant)
            requests.append(
                RequestPlan(
                    target.project,
                    target.workflow_id,
                    registered,
                    target.module,
                    (target.participant,),
                    planned,
                    terminal,
                )
            )
        requests = list(_minimal_requests(requests))
        for request in requests:
            work_items.update((spec.key, spec) for spec in request.work_items)
            matched.setdefault(request.project, [])
            matched[request.project] = list(
                dict.fromkeys((*matched[request.project], *request.participants))
            )
        return PlanningResult(
            requests=tuple(requests),
            work_items=work_items,
            matched_participants={key: tuple(value) for key, value in matched.items()},
            present_participants=frozenset(present),
            unavailable=tuple(unavailable),
            source_digest=source_digest,
            site_settings=site_settings,
        )

    def register_requests(
        self,
        plan: PlanningResult,
        *,
        selectors: Mapping[str, Any],
        concurrency: int,
        partition: str | None,
    ) -> tuple[str, ...]:
        """Create demand with captured source and site paths for each request group."""
        from nro.orchestration.execution_cache import cache_lock, cleanup_cache
        from nro.orchestration.execution_pins import capture_execution

        if not isinstance(self.registry, Registry):
            raise ValueError(
                "Branch scientific planning is available, but branch scheduler submission is not enabled yet"
            )
        cleanup_cache(self.registry)
        if not plan.requests:
            return ()
        with cache_lock(self.registry.paths.control):
            source, site = capture_execution(
                self.registry.paths.control,
                self.bids_root,
                expected_source=plan.source_digest,
                site_values=dict(plan.site_settings),
            )
            return tuple(
                Registry.for_project(
                    request.project,
                    bids_root=self.bids_root,
                    registry_path=self.registry.paths.control,
                ).create_request(
                    registered=request.registered,
                    target_module=request.module,
                    selectors={**selectors, "participants": list(request.participants)},
                    work_items=tuple(
                        work_item.evolve(command=source.command(work_item.command, site=site))
                        for work_item in request.work_items
                    ),
                    terminal_work_item_keys=request.terminal_keys,
                    concurrency=concurrency,
                    partition=partition,
                )
                for request in plan.requests
            )

    @with_site_read_cache
    @with_bids_metadata_cache
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
    ) -> tuple[WorkItemSpec, ...]:
        """Build the requested logical work-item graph for one participant."""
        target = normalize_module(module)
        if (target in {"dynconn", "microparcellation", "networks"} and selectors) or (
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
        markup_id = workflow.configuration(target).values.get("markup")
        source_markup = MarkupStore().subject(markup_id, project, subject_dir)
        runs = () if target == "anat" else discover_raw_runs(subject_dir, markup=source_markup)
        target_descriptor = module_descriptor(target)
        task_models = None
        if target_descriptor.select_models is not None:
            task_models = target_descriptor.select_models(
                tasks=(selectors or {}).get("task", ()), models=models, model_sets=model_sets
            )
            tasks = {identifier.split("/")[0] for identifier in task_models}
            runs = tuple(run for run in runs if run.entities.get("task") in tasks)
        elif target_descriptor.select_runs is not None:
            runs = target_descriptor.select_runs(
                runs,
                workflow.configuration(target_descriptor.configuration_class).values,
                participant,
            )
        if selectors:
            runs = tuple(run for run in runs if matches_selectors(run.entities, selectors))
        if target in {"dynconn", "microparcellation", "networks"}:
            filter_class = (
                workflow.configuration("networks").values["connectivity_source"]
                if target == "networks"
                else target
            )
            aggregate_filter = workflow.configuration(filter_class).values.get("input_filter", {})
            runs = tuple(run for run in runs if matches_filter(run.entities, aggregate_filter))
        if target != "anat" and not runs:
            raise ParticipantUnavailableError(f"No raw BOLD runs matched under {subject_dir}")

        requested_spaces = tuple(dict.fromkeys(spaces or (DEFAULT_SPACE,)))
        requested_smoothing = tuple(dict.fromkeys(smoothing_levels or (DEFAULT_SMOOTHING_MM,)))
        target_pairs: tuple[tuple[str, int], ...] = ()
        if target not in {"anat", "func"}:
            if not requested_spaces:
                raise ValueError("At least one space must be requested")
            if not requested_smoothing or any(value < 0 for value in requested_smoothing):
                raise ValueError("Smoothing levels must be nonnegative integers")
            published_spaces = supported_output_spaces(
                str(workflow.configuration("anat").values["fsaverage_template"])
            )
            unavailable = tuple(
                space for space in requested_spaces if space not in published_spaces
            )
            if unavailable:
                raise ValueError(
                    "Requested space(s) are not published by the func module: "
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
                    path
                    for run in runs
                    for path in image_source_paths(run.path, markup=source_markup)
                )
            ),
            target_pairs=target_pairs,
            memory_gb=memory_gb,
            max_memory_gb=max_memory_gb,
            definitions_roots=self._definitions_roots,
            gradient_coefficients_root=Path(self._site_values["gradient_coefficients"]),
            task_models=task_models,
            source_markup=source_markup,
        )
        descriptors = modules_through(target, workflow)
        planned: dict[str, tuple[WorkItemSpec, ...]] = {}
        for descriptor in descriptors:
            upstream_keys = tuple(
                (name, tuple(item.key for item in items)) for name, items in planned.items()
            )
            cache_key = (
                descriptor.name,
                project,
                participant,
                context.registered.lineage_fingerprints[descriptor.name],
                tuple(str(run.path) for run in context.runs),
                context.target_pairs,
                fingerprint(context.task_models or {}),
                fingerprint(context.source_markup.as_dict() if context.source_markup else {}),
                context.memory_gb,
                context.max_memory_gb,
                upstream_keys,
            )
            cached = self._module_plan_cache.get(cache_key)
            if cached is None:
                cached = descriptor.plan(context, planned, descriptor)
                self._module_plan_cache[cache_key] = cached
            planned[descriptor.name] = cached
        return tuple(
            work_item for descriptor in descriptors for work_item in planned[descriptor.name]
        )


def build_subject_work_items(
    *,
    project: str,
    participant: str,
    module: str,
    workflow: ResolvedWorkflow,
    registered: RegisteredWorkflow,
    registry: WorkflowRegistry,
    selectors: Mapping[str, tuple[str, ...] | None] | None = None,
    spaces: tuple[str, ...] | None = None,
    smoothing_levels: tuple[int, ...] | None = None,
    bids_root: str | Path,
    memory_gb: int = 32,
    max_memory_gb: int = 256,
    models: Sequence[str] = (),
    model_sets: Sequence[str] | None = None,
) -> tuple[WorkItemSpec, ...]:
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
