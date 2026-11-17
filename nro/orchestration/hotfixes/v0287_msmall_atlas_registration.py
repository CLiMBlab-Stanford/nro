"""Preserve valid MSMAll FreeSurfer work across the atlas-registration repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nro.engine.io import atomic_write_json
from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.hotfixes._runner_contracts import (
    descendants,
    nodes,
    run_contract_hotfix,
    stage_node,
)

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


def _candidate(contract: Path) -> _Candidate | None:
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) < 3:
        return None
    if payload.get("module") != "Anatomical Module":
        return None
    topology = nodes(payload, "topology")
    completed = nodes(payload, "nodes")
    prefreesurfer = stage_node(topology, "prefreesurfer")
    masked_atlas = stage_node(topology, "masked_atlas")
    freesurfer = stage_node(completed, "freesurfer")
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


def _repair(candidate: _Candidate) -> None:
    payload = dict(candidate.payload)
    topology = nodes(payload, "topology")
    invalidated = set(descendants(topology, (candidate.masked_atlas_id,), include_roots=False))
    invalidated.add(candidate.masked_atlas_id)
    invalidated.discard(candidate.freesurfer_id)
    repaired_nodes: list[dict[str, object]] = []
    for node in nodes(payload, "nodes"):
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


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Invalidate only the defective MSMAll atlas branch and its descendants."""
    return run_contract_hotfix(
        registry,
        projects=projects,
        execute=execute,
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        inspect=_candidate,
        repair=_repair,
    )
