"""Preserve PreFreeSurfer while replacing its inferred brain masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport

HOTFIX_ID = "v0301-msmall-source-masks"
SUMMARY = "Rebuild MSMAll descendants with masks derived from nro inputs."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    prefreesurfer_id: str
    invalidated: frozenset[str]


def _nodes(payload: Mapping[str, object], field: str) -> list[dict[str, object]]:
    values = payload.get(field)
    if not isinstance(values, list):
        return []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def _stage_node(nodes: list[dict[str, object]], stage: str) -> dict[str, object] | None:
    suffix = f"/stages/{stage}.complete"
    matches = [
        node
        for node in nodes
        if isinstance(node.get("outputs"), list)
        and any(isinstance(path, str) and path.endswith(suffix) for path in node["outputs"])
    ]
    return matches[0] if len(matches) == 1 else None


def _descendants(topology: list[dict[str, object]], root: str) -> frozenset[str]:
    children: dict[str, set[str]] = {}
    for node in topology:
        node_id = node.get("id")
        dependencies = node.get("dependencies")
        if not isinstance(node_id, str) or not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if isinstance(dependency, str):
                children.setdefault(dependency, set()).add(node_id)
    found: set[str] = set()
    pending = list(children.get(root, ()))
    while pending:
        node_id = pending.pop()
        if node_id in found:
            continue
        found.add(node_id)
        pending.extend(children.get(node_id, ()))
    return frozenset(found)


def _candidate(contract: Path) -> _Candidate | None:
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("module") != "Anatomical Module":
        return None
    if payload.get("signature") == f"hotfix:{HOTFIX_ID}":
        return None
    topology = _nodes(payload, "topology")
    if _stage_node(topology, "prefreesurfer_masks") is not None:
        return None
    prefreesurfer = _stage_node(topology, "prefreesurfer")
    if prefreesurfer is None or not isinstance(prefreesurfer.get("id"), str):
        return None
    prefreesurfer_id = str(prefreesurfer["id"])
    return _Candidate(
        contract=contract,
        payload=payload,
        prefreesurfer_id=prefreesurfer_id,
        invalidated=_descendants(topology, prefreesurfer_id),
    )


def _repair(candidate: _Candidate) -> None:
    payload = dict(candidate.payload)
    repaired: list[dict[str, object]] = []
    for node in _nodes(payload, "nodes"):
        node_id = node.get("id")
        if node_id in candidate.invalidated:
            continue
        if node_id == candidate.prefreesurfer_id:
            # The command did not change. Its shared shell driver gained a
            # separate mask-repair operation for the new downstream step.
            node["accept_relocated_signatures"] = True
        repaired.append(node)
    payload["nodes"] = repaired
    payload["signature"] = f"hotfix:{HOTFIX_ID}"
    atomic_write_json(candidate.contract, payload, sort_keys=True)


def _candidates(control: Path, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    branches = ControlPaths(control).root / "branches"
    found: list[_Candidate] = []
    for project in projects:
        for contract in sorted(branches.glob(f"*/events/{project}/anat/**/runner-contract.json")):
            candidate = _candidate(contract)
            if candidate is not None:
                found.append(candidate)
    return tuple(found)


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Retain valid PreFreeSurfer results and invalidate mask consumers."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    with registry.connection(write=execute) as database:
        if execute:
            placeholders = ",".join("?" for _ in selected)
            active = database.execute(
                f"""SELECT COUNT(*) FROM attempts AS a
                    JOIN work_items AS w ON w.id=a.work_item_id
                    WHERE w.project IN ({placeholders})
                      AND a.state IN ('queued','running','cancel_requested')""",
                selected,
            ).fetchone()[0]
            if active:
                raise ValueError("Hotfix requires selected-project attempts to be stopped")
        candidates = _candidates(registry.paths.control, selected)
        if execute:
            for candidate in candidates:
                _repair(candidate)
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=tuple(candidate.contract for candidate in candidates),
        records=len(candidates),
        applied=execute,
    )
