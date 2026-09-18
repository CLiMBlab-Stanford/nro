"""Compile branch-owned demand and hand it to the central scheduler."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from nro.configuration.site import CHECKOUT, protected_site_fingerprint, settings
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.compiled_request import encode_spec, export_workflow
from nro.orchestration.execution_cache import cache_lock
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.execution_pins import capture_execution

if TYPE_CHECKING:
    from nro.orchestration.contracts import WorkItemSpec
    from nro.orchestration.planner import RequestPlan
    from nro.orchestration.workflow_registry import RegisteredWorkflow


@dataclass
class _RequestGroup:
    """One project/workflow demand record with all independent endpoints."""

    project: str
    registered: RegisteredWorkflow
    work_items: dict[str, WorkItemSpec]
    terminal_keys: list[str]


def _request_groups(requests: Sequence[RequestPlan]) -> tuple[_RequestGroup, ...]:
    """Coalesce one invocation's endpoints by project and workflow."""
    groups: dict[tuple[str, str], _RequestGroup] = {}
    for request in requests:
        key = (request.project, request.workflow_id)
        group = groups.get(key)
        if group is None:
            group = _RequestGroup(request.project, request.registered, {}, [])
            groups[key] = group
        group.work_items.update((spec.key, spec) for spec in request.work_items)
        group.terminal_keys.extend(request.terminal_keys)
    for group in groups.values():
        group.terminal_keys[:] = dict.fromkeys(group.terminal_keys)
    return tuple(groups.values())


def register_requests(
    scientific,
    plan,
    *,
    selectors: dict,
    concurrency: int,
    partition: str | None,
    expected_revisions: dict[str, int],
    demand: bool = True,
    inherit: bool = True,
) -> list[str]:
    """Publish compiled graphs without loading the scheduler schema in this branch.

    The cache lock spans source capture and admission. A concurrent scientific
    edit rejects the graph before submission; an already admitted newer revision
    also rejects a delayed handoff in the central process.
    """
    from nro.orchestration.scheduler_client import command, exchange

    values = settings()[0]
    site_fingerprint = protected_site_fingerprint()
    control = Path(values["registry"])
    branches = BranchStore(control)
    topology = branches.read().topology
    name = topology.registered_checkout(CHECKOUT)
    if name != "main":
        name = topology.require_checkout(CHECKOUT)
    owner = topology.records[name].registry_id
    if scientific.record.registry_id != owner:
        raise ValueError("Scientific registry belongs to another checkout")
    recorded = scientific.record_work_item_graph(
        tuple(plan.work_items.values()),
        expected_revisions={key: expected_revisions.get(key) for key in plan.work_items},
    )
    records = {row.key: row for row in recorded}
    revisions = {key: row.revision for key, row in records.items()}
    paths = BranchPaths(
        name, Path(values["bids"]), Path(values["work"]), Path(values["development"])
    )
    release = None
    if name == "main":
        from nro.orchestration.releases import ReleaseStore

        release = ReleaseStore(branches).require_approved(CHECKOUT)
    with cache_lock(control):
        source, site = capture_execution(
            control,
            paths.bids,
            expected_source=plan.source_digest,
            site_values=dict(plan.site_settings),
        )
        central = command(control, paths.bids)
        entries = []
        for request in _request_groups(plan.requests):
            payload = dict(
                protocol=1,
                branch=name,
                registry_id=owner,
                project=request.project,
                context=ExecutionContext(
                    paths, request.project, request.terminal_keys[0], ()
                ).as_dict(),
                specifications=[encode_spec(spec) for spec in request.work_items.values()],
                revisions={spec.key: revisions[spec.key] for spec in request.work_items.values()},
                contracts={
                    spec.key: records[spec.key].contract for spec in request.work_items.values()
                },
                terminals=list(request.terminal_keys),
                inherit=inherit,
                workflow=export_workflow(scientific, request.registered),
                source=dict(root=str(source.root), digest=source.digest),
                site=str(site),
                site_fingerprint=site_fingerprint,
                python=sys.executable,
                selectors=selectors,
                concurrency=concurrency,
                partition=partition,
                release=release,
                demand=demand,
            )
            entries.append(dict(project=request.project, payload=payload))
        result = exchange(
            central,
            dict(
                operation="admit_many",
                checkout=str(CHECKOUT),
                entries=entries,
            ),
        )
    return result["request_ids"]
