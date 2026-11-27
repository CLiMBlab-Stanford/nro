"""Plan a study analysis and its upstream module graphs as scheduler requests."""

from __future__ import annotations

from typing import Sequence

from nro.analysis.layout import LoadedAnalysis
from nro.modules.analyses.planning import (
    MODULE,
    UpstreamRequest,
    plan_work_items,
    upstream_selections,
    upstream_targets,
)
from nro.orchestration.contracts import WorkItemSpec


def analysis_chain(analysis: LoadedAnalysis) -> tuple[LoadedAnalysis, ...]:
    """Return the analysis and every analysis it reads, upstream ones first.

    Raises ValueError when ``analysis:`` inputs form a cycle.
    """
    from nro.analysis.layout import load_analysis

    ordered: dict[str, LoadedAnalysis] = {}
    active: list[str] = []

    def visit(item: LoadedAnalysis) -> None:
        if item.name in ordered:
            return
        if item.name in active:
            raise ValueError("Analysis inputs form a cycle: " + " -> ".join([*active, item.name]))
        active.append(item.name)
        for spec in item.manifest.inputs.values():
            if spec.analysis is not None:
                upstream = load_analysis(item.source, spec.analysis)
                if upstream.manifest.scope == "project":
                    raise ValueError(
                        f"{item.name} reads {upstream.name}, which has project scope; only "
                        "subject- and run-scope analyses can supply other analyses"
                    )
                visit(upstream)
        active.pop()
        ordered[item.name] = item

    visit(analysis)
    return tuple(ordered.values())


def covered_participants(spec: WorkItemSpec) -> tuple[str, ...]:
    """Return the participants a work item covers: its own, or a project item's list."""
    if spec.is_project_scope:
        return tuple(spec.contract.processing.get("analysis", {}).get("participants", ()))
    return (spec.participant,)


def plan_request(
    planner,
    registry,
    analysis: LoadedAnalysis,
    store,
    *,
    projects: Sequence[str] = (),
    participants: Sequence[str] = (),
    memory_gb: int = 32,
    max_memory_gb: int = 256,
):
    """Plan an analysis and its upstream graphs as one request per member project.

    Registers the analysis's unbound lineage and each upstream workflow in
    ``registry``, then returns a ``PlanningResult`` whose requests have no
    workflow. Each request's graph holds the analysis work items and the
    upstream work items they depend on.
    """
    from nro.orchestration.planner import PlanningResult, RequestPlan, UnavailableSelection
    from nro.site.configuration import definitions_root, settings

    chain = analysis_chain(analysis)
    lineages = {
        item.name: registry.register_unbound_lineage(
            MODULE,
            item.name,
            {"study": item.study.name, "analysis": item.manifest.analysis},
        )
        for item in chain
    }
    members = tuple(projects or analysis.study.projects)
    unknown = sorted(set(members) - set(analysis.study.projects))
    if unknown:
        raise ValueError(f"Study {analysis.study.name} does not include {', '.join(unknown)}")
    if analysis.manifest.scope == "project" and participants:
        raise ValueError(
            f"{analysis.name} has project scope; its participants come from the manifest's "
            "participants selector, so --participant cannot narrow it"
        )
    upstream: list[UpstreamRequest] = []
    graphs: dict[str, WorkItemSpec] = {}
    unavailable: list = []
    selections = tuple(dict.fromkeys(pair for item in chain for pair in upstream_selections(item)))
    for module, workflow_id in selections:
        workflow = store.resolve(workflow_id)
        registered = registry.register_workflow(workflow)
        spaces: tuple = ()
        smoothing: tuple = ()
        for item in chain:
            more_spaces, more_smoothing = upstream_targets(item, module, workflow_id)
            spaces = tuple(dict.fromkeys((*spaces, *more_spaces)))
            smoothing = tuple(dict.fromkeys((*smoothing, *more_smoothing)))
        result = planner.plan(
            projects=members,
            requested_participants=participants,
            modules=(module,),
            workflows={workflow_id: workflow},
            registered_workflows={workflow_id: registered},
            selectors={},
            spaces=spaces,
            smoothing_levels=smoothing,
            memory_gb=memory_gb,
            max_memory_gb=max_memory_gb,
        )
        upstream.append(UpstreamRequest(module, workflow_id, dict(result.work_items)))
        graphs.update(result.work_items)
        unavailable.extend(result.unavailable)
    planned: dict[str, tuple[WorkItemSpec, ...]] = {}
    for item in chain:
        specifications, missing = plan_work_items(
            item,
            lineage=lineages[item.name],
            upstream=upstream,
            bids_root=planner.bids_root,
            projects=members,
            participants=participants,
            memory_gb=memory_gb,
            max_memory_gb=max_memory_gb,
            analysis_specs={name.split("/", 1)[1]: specs for name, specs in planned.items()},
        )
        planned[item.name] = specifications
        graphs.update((spec.key, spec) for spec in specifications)
        if item is analysis:
            unavailable.extend(
                UnavailableSelection(
                    project=project, participant=participant, module=MODULE, reason=reason
                )
                for project, participant, reason in missing
            )
    specifications = planned[analysis.name]
    requests = []
    work_items: dict[str, WorkItemSpec] = {}
    matched: dict[str, tuple[str, ...]] = {}
    for project in members:
        terminals = [spec for spec in specifications if spec.project == project]
        if not terminals:
            continue
        closure: dict[str, WorkItemSpec] = {}
        pending = [spec.key for spec in terminals]
        by_key = {**graphs, **{spec.key: spec for spec in terminals}}
        while pending:
            key = pending.pop()
            if key in closure:
                continue
            closure[key] = by_key[key]
            pending.extend(by_key[key].dependencies)
        participants_matched = tuple(
            dict.fromkeys(
                participant for spec in terminals for participant in covered_participants(spec)
            )
        )
        matched[project] = participants_matched
        work_items.update(closure)
        requests.append(
            RequestPlan(
                project=project,
                workflow_id=None,
                registered=None,
                module=MODULE,
                participants=participants_matched,
                work_items=tuple(closure.values()),
                terminal_keys=tuple(spec.key for spec in terminals),
            )
        )
    return PlanningResult(
        requests=tuple(requests),
        work_items=work_items,
        matched_participants=matched,
        present_participants=frozenset(
            participant for values in matched.values() for participant in values
        ),
        unavailable=tuple(unavailable),
        source_digest=planner._source_digest,
        site_settings={**settings()[0], "definitions": str(definitions_root())},
    )
