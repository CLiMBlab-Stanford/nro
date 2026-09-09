"""Refresh current scientific expectations in the invoking checkout, without demand."""

import json
from pathlib import Path
from types import SimpleNamespace

from nro.configuration.site import CHECKOUT, settings
from nro.configuration.store import ConfigStore
from nro.engine.bids import ENTITY_ORDER
from nro.engine.cli import matches_instance_selectors
from nro.orchestration.branch_requests import register_requests
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.catalog import MODULES, module_descriptor
from nro.orchestration.planner import Planner, RequestPlan, _minimal_requests
from nro.orchestration.planning_context import ParticipantUnavailableError
from nro.orchestration.source_snapshots import source_fingerprint


def refresh(rows: list[dict], selection) -> None:
    """Recompile registered selections before thorough status verification.

    The frontend owns scientific planning. Central admission receives only
    compiled contracts and execution recipes. Historical lineages absent from
    current workflows retain their recorded contracts; this creates no demand.
    """
    if not rows:
        return
    values = settings()[0]
    scientific = BranchStore(Path(values["registry"])).registry_for_checkout(CHECKOUT)
    revisions = {row.key: row.revision for row in scientific.instances()}
    source = source_fingerprint(CHECKOUT)
    store = ConfigStore()
    workflows = {
        path.name.removesuffix("_workflow.yml"): store.resolve(
            path.name.removesuffix("_workflow.yml")
        )
        for path in (store.root / "workflows").glob("*_workflow.yml")
    }
    registered = {}
    planner = Planner(scientific, bids_root=Path(values["bids"]))
    requests, seen = [], set()
    for row in rows:
        if (
            row["module"] not in MODULES
            or selection.projects
            and row["project"] not in selection.projects
            or selection.participants
            and row["participant"] not in selection.participants
            or selection.modules
            and row["module"] not in selection.modules
        ):
            continue
        entities = json.loads(row["entities_json"])
        if not matches_instance_selectors(entities, selection.instance_entities):
            continue
        names = set(filter(None, row.get("workflow_ids", "").split(","))) or set(workflows)
        if selection.workflows:
            names &= set(selection.workflows)
        for name in sorted(names & workflows.keys()):
            if name not in registered:
                registered[name] = scientific.register_workflow(workflows[name])
            registration = registered[name]
            derivative_class = module_descriptor(row["module"]).configuration_class
            if registration.directories[derivative_class] != row["directory_label"]:
                continue
            identity = (
                name,
                row["project"],
                row["participant"],
                row["module"],
                row["entities_json"],
            )
            if identity in seen:
                continue
            seen.add(identity)
            selectors = {
                key: (value,)
                for key, value in entities.items()
                if key in ENTITY_ORDER and key != "sub"
            }
            if row["module"] in {"anat", "dynconn", "microparcellation", "networks"}:
                selectors = None
            elif row["module"] == "firstlevels":
                selectors = {"task": (entities["task"],)}
            try:
                specs = planner.plan_subject(
                    project=row["project"],
                    participant=row["participant"],
                    module=row["module"],
                    workflow=workflows[name],
                    registered=registration,
                    selectors=selectors,
                    spaces=(entities["space"],) if "space" in entities else None,
                    smoothing_levels=(int(entities["smoothing"]),)
                    if "smoothing" in entities
                    else None,
                    models=(f"{entities['task']}/{entities['model']}",)
                    if row["module"] == "firstlevels"
                    else (),
                    model_sets=(),
                    memory_gb=row["memory_gb"],
                    max_memory_gb=row["max_memory_gb"],
                )
            except (FileNotFoundError, ParticipantUnavailableError):
                # Filesystem assessment still reports missing inputs. Missing
                # raw data cannot supply a replacement scientific graph.
                continue
            targets = tuple(spec.key for spec in specs if spec.module == row["module"])
            requests.append(
                RequestPlan(
                    row["project"],
                    name,
                    registration,
                    row["module"],
                    (row["participant"],),
                    specs,
                    targets,
                )
            )
    if requests:
        requests = _minimal_requests(requests)
        plan = SimpleNamespace(
            requests=requests,
            instances={spec.key: spec for request in requests for spec in request.instances},
            source_digest=source,
            site_settings=values,
        )
        register_requests(
            scientific,
            plan,
            selectors={},
            concurrency=1,
            partition=None,
            expected_revisions=revisions,
            demand=False,
        )


def preview(rows: list[dict], visible: set[int], dependencies=()) -> list[dict]:
    """Warn about current processing-policy changes without publishing them."""
    from nro.orchestration.manifests import _current_contract

    result = []
    for row in rows:
        row = dict(row)
        if row["id"] in visible and row["module"] in MODULES:
            _, _, changed = _current_contract(row)
            if changed:
                row.update(
                    artifact_state="stale",
                    artifact_reason="Current checkout processing contract changed",
                )
                if row["status"] == "Success":
                    row["status"] = "Stale"
        result.append(row)
    changed = {row["id"] for row in result if row["artifact_state"] in {"stale", "missing"}}
    while True:
        downstream = {child for child, parent in dependencies if parent in changed}
        if downstream <= changed:
            break
        changed.update(downstream)
    for row in result:
        if row["id"] in visible & changed and row["status"] == "Success":
            row.update(
                status="Stale",
                artifact_state="stale",
                artifact_reason="An upstream derivative is missing or stale",
            )
    return result


def record_observations(rows: list[dict], visible: set[int]) -> None:
    """Save verified scheduler observations beside their scientific contracts."""
    values = settings()[0]
    scientific = BranchStore(Path(values["registry"])).registry_for_checkout(CHECKOUT)
    observations = {}
    for row in rows:
        if (
            row["id"] not in visible
            or not row.get("logical_key")
            or row.get("scientific_revision") is None
        ):
            continue
        observations[row["logical_key"]] = (
            row["scientific_revision"],
            {
                "artifact_state": row["artifact_state"],
                "artifact_reason": row["artifact_reason"],
                "generation": row["current_generation"],
                "artifact_fingerprint": row["artifact_fingerprint"],
            },
        )
    scientific.record_observations(observations)
