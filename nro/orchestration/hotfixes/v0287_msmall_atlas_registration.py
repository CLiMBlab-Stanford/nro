"""Preserve valid MSMAll FreeSurfer work across the atlas-registration repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport

HOTFIX_ID = "v0287-msmall-atlas-registration"
SUMMARY = "Invalidate distorted MSMAll atlas products without repeating FreeSurfer."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    freesurfer_id: str
    prefreesurfer_id: str
    masked_atlas_id: str
    prefreesurfer_marker: str


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


def _candidate(contract: Path) -> _Candidate | None:
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) < 3:
        return None
    if payload.get("module") != "Anatomical Module":
        return None
    topology = _nodes(payload, "topology")
    completed = _nodes(payload, "nodes")
    prefreesurfer = _stage_node(topology, "prefreesurfer")
    masked_atlas = _stage_node(topology, "masked_atlas")
    freesurfer = _stage_node(completed, "freesurfer")
    if prefreesurfer is None or masked_atlas is None or freesurfer is None:
        return None
    prefreesurfer_id = prefreesurfer.get("id")
    masked_atlas_id = masked_atlas.get("id")
    freesurfer_id = freesurfer.get("id")
    if not all(
        isinstance(value, str) and value
        for value in (
            prefreesurfer_id,
            masked_atlas_id,
            freesurfer_id,
        )
    ):
        return None
    dependencies = freesurfer.get("dependencies")
    inputs = freesurfer.get("inputs")
    if not isinstance(dependencies, list) or masked_atlas_id not in dependencies:
        return None
    if not isinstance(inputs, list):
        return None
    old_suffix = "/stages/masked_atlas.complete"
    if sum(isinstance(path, str) and path.endswith(old_suffix) for path in inputs) != 1:
        return None
    pref_outputs = prefreesurfer.get("outputs")
    if not isinstance(pref_outputs, list):
        return None
    markers = [
        path
        for path in pref_outputs
        if isinstance(path, str) and path.endswith("/stages/prefreesurfer.complete")
    ]
    if len(markers) != 1:
        return None
    return _Candidate(
        contract=contract,
        payload=payload,
        freesurfer_id=freesurfer_id,
        prefreesurfer_id=prefreesurfer_id,
        masked_atlas_id=masked_atlas_id,
        prefreesurfer_marker=markers[0],
    )


def _descendants(topology: list[dict[str, object]], root: str) -> set[str]:
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
    return found


def _repair(candidate: _Candidate) -> None:
    payload = dict(candidate.payload)
    topology = _nodes(payload, "topology")
    invalidated = _descendants(topology, candidate.masked_atlas_id)
    invalidated.add(candidate.masked_atlas_id)
    invalidated.discard(candidate.freesurfer_id)
    repaired_nodes: list[dict[str, object]] = []
    for node in _nodes(payload, "nodes"):
        node_id = node.get("id")
        if node_id in invalidated:
            continue
        if node_id == candidate.prefreesurfer_id:
            # The historical shell driver was shared by all stages. Its bytes
            # changed only because atlas-registration code elsewhere in that
            # file changed; the PreFreeSurfer command itself is unchanged.
            node["accept_relocated_signatures"] = True
        if node_id == candidate.freesurfer_id:
            inputs = node.get("inputs")
            dependencies = node.get("dependencies")
            assert isinstance(inputs, list) and isinstance(dependencies, list)
            node["inputs"] = [
                candidate.prefreesurfer_marker
                if isinstance(path, str) and path.endswith("/stages/masked_atlas.complete")
                else path
                for path in inputs
            ]
            node["dependencies"] = [
                candidate.prefreesurfer_id
                if dependency == candidate.masked_atlas_id
                else dependency
                for dependency in dependencies
            ]
            # The command is unchanged, but the shared historical driver file
            # also contained the repaired atlas stage. Adopt its current
            # signatures once while retaining the reconstructed surfaces.
            node["accept_relocated_signatures"] = True
        repaired_nodes.append(node)
    payload["nodes"] = repaired_nodes
    # Avoid binding the obsolete topology before the runner records the new
    # one. Node-level evidence above still controls what may be adopted.
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
    """Invalidate only the defective MSMAll atlas branch and its descendants."""
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
