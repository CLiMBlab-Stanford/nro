"""Invalidate MSMAll work that inferred a mask from nonzero T1w voxels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport

HOTFIX_ID = "v0302-msmall-brain-mask"
SUMMARY = "Rebuild MSMAll stages with the surface-reconstruction brain mask."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    invalidated: frozenset[str]


def _nodes(payload: Mapping[str, object], field: str) -> list[dict[str, object]]:
    values = payload.get(field)
    if not isinstance(values, list):
        return []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def _paths(node: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        value
        for field in ("inputs", "outputs")
        for value in node.get(field, ())
        if isinstance(value, str)
    )


def _stage_node(nodes: list[dict[str, object]], stage: str) -> dict[str, object] | None:
    suffix = f"/stages/{stage}.complete"
    matches = [
        node
        for node in nodes
        if any(isinstance(path, str) and path.endswith(suffix) for path in node.get("outputs", ()))
    ]
    return matches[0] if len(matches) == 1 else None


def _descendants(topology: list[dict[str, object]], roots: set[str]) -> frozenset[str]:
    children: dict[str, set[str]] = {}
    for node in topology:
        node_id = node.get("id")
        dependencies = node.get("dependencies")
        if not isinstance(node_id, str) or not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if isinstance(dependency, str):
                children.setdefault(dependency, set()).add(node_id)
    found = set(roots)
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        for child in children.get(node_id, ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return frozenset(found)


def _is_msmall_node(node: Mapping[str, object]) -> bool:
    name = node.get("name")
    if isinstance(name, str) and "MSMAll" in name:
        return True
    return any(
        "/msmall/" in path or "_space-MSMAll_" in path or "_desc-msmall" in path
        for path in _paths(node)
    )


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
    mask_stage = _stage_node(topology, "prefreesurfer_masks")
    if mask_stage is None:
        return None
    mask_inputs = _paths(mask_stage)
    if not any(path.endswith("_desc-preproc_T1w.nii.gz") for path in mask_inputs):
        return None
    if any(path.endswith("_desc-brain_mask.nii.gz") for path in mask_inputs):
        return None
    roots = {
        node_id
        for node in topology
        for node_id in (node.get("id"),)
        if isinstance(node_id, str) and _is_msmall_node(node)
    }
    invalidated = _descendants(topology, roots)
    if not invalidated:
        return None
    return _Candidate(contract=contract, payload=payload, invalidated=invalidated)


def _candidates(control: Path, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    branches = ControlPaths(control).root / "branches"
    found: list[_Candidate] = []
    for project in projects:
        for contract in sorted(branches.glob(f"*/events/{project}/anat/**/runner-contract.json")):
            candidate = _candidate(contract)
            if candidate is not None:
                found.append(candidate)
    return tuple(found)


def _repair(candidate: _Candidate) -> None:
    payload = dict(candidate.payload)
    payload["nodes"] = [
        node for node in _nodes(payload, "nodes") if node.get("id") not in candidate.invalidated
    ]
    payload["signature"] = f"hotfix:{HOTFIX_ID}"
    atomic_write_json(candidate.contract, payload, sort_keys=True)


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Invalidate old MSMAll records while retaining ordinary anatomy records."""
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
