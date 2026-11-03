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

    lineage = registry.register_unbound_lineage(
        MODULE,
        analysis.name,
        {"study": analysis.study.name, "analysis": analysis.manifest.analysis},
    )
    members = tuple(projects or analysis.study.projects)
    unknown = sorted(set(members) - set(analysis.study.projects))
    if unknown:
        raise ValueError(f"Study {analysis.study.name} does not include {', '.join(unknown)}")
    upstream: list[UpstreamRequest] = []
    graphs: dict[str, WorkItemSpec] = {}
    unavailable: list = []
    for module, workflow_id in upstream_selections(analysis):
        workflow = store.resolve(workflow_id)
        registered = registry.register_workflow(workflow)
        spaces, smoothing = upstream_targets(analysis, module, workflow_id)
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
    specifications, missing = plan_work_items(
        analysis,
        lineage=lineage,
        upstream=upstream,
        bids_root=planner.bids_root,
        projects=members,
        participants=participants,
        memory_gb=memory_gb,
        max_memory_gb=max_memory_gb,
    )
    unavailable.extend(
        UnavailableSelection(project=project, participant=participant, module=MODULE, reason=reason)
        for project, participant, reason in missing
    )
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
        participants_matched = tuple(dict.fromkeys(spec.participant for spec in terminals))
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
