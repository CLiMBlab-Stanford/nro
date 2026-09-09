"""Compile branch-owned demand and hand it to the central scheduler."""

import sys
from pathlib import Path

from nro.configuration.site import CHECKOUT, settings
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.compiled_request import encode_spec, export_workflow
from nro.orchestration.execution_cache import cache_lock
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.execution_pins import capture_execution


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
    control = Path(values["registry"])
    branches = BranchStore(control)
    topology = branches.read().topology
    name = topology.require_checkout(CHECKOUT)
    owner = topology.records[name].registry_id
    if scientific.record.registry_id != owner:
        raise ValueError("Scientific registry belongs to another checkout")
    recorded = scientific.record_graph(
        tuple(plan.instances.values()),
        expected_revisions={key: expected_revisions.get(key) for key in plan.instances},
    )
    revisions = {row.key: row.revision for row in recorded}
    paths = BranchPaths(
        name, Path(values["bids"]), Path(values["work"]), Path(values["development"])
    )
    release = None
    if name == "main":
        from nro.orchestration.releases import ReleaseStore

        release = ReleaseStore(branches).require_approved(CHECKOUT)
    results = []
    with cache_lock(control):
        source, site = capture_execution(
            control,
            paths.bids,
            expected_source=plan.source_digest,
            site_values=dict(plan.site_settings),
        )
        central = command(control, paths.bids)
        for request in plan.requests:
            payload = dict(
                protocol=1,
                branch=name,
                registry_id=owner,
                project=request.project,
                context=ExecutionContext(
                    paths, request.project, request.terminal_keys[0], ()
                ).as_dict(),
                specifications=[encode_spec(spec) for spec in request.instances],
                revisions={spec.key: revisions[spec.key] for spec in request.instances},
                terminals=list(request.terminal_keys),
                inherit=inherit,
                workflow=export_workflow(scientific, request.registered),
                source=dict(root=str(source.root), digest=source.digest),
                site=str(site),
                python=sys.executable,
                selectors=selectors,
                concurrency=concurrency,
                partition=partition,
                release=release,
                demand=demand,
            )
            result = exchange(
                central,
                dict(
                    operation="admit",
                    checkout=str(CHECKOUT),
                    project=request.project,
                    payload=payload,
                ),
            )
            results.append(result["request_id"])
    return results
