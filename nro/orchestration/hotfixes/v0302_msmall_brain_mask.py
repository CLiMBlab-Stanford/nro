"""Invalidate MSMAll work that inferred a mask from nonzero T1w voxels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.hotfixes._runner_contracts import (
    descendants,
    node_paths,
    nodes,
    rewrite_nodes,
    run_contract_hotfix,
    stage_node,
)

HOTFIX_ID = "v0302-msmall-brain-mask"
SUMMARY = "Rebuild MSMAll stages with the surface-reconstruction brain mask."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    invalidated: frozenset[str]


def _is_msmall_node(node: Mapping[str, object]) -> bool:
    name = node.get("name")
    if isinstance(name, str) and "MSMAll" in name:
        return True
    return any(
        "/msmall/" in path or "_space-MSMAll_" in path or "_desc-msmall" in path
        for path in node_paths(node)
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
    topology = nodes(payload, "topology")
    mask_stage = stage_node(topology, "prefreesurfer_masks")
    if mask_stage is None:
        return None
    mask_inputs = node_paths(mask_stage)
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
    invalidated = descendants(topology, roots, include_roots=True)
    if not invalidated:
        return None
    return _Candidate(contract=contract, payload=payload, invalidated=invalidated)


def _repair(candidate: _Candidate) -> None:
    rewrite_nodes(
        candidate.contract, candidate.payload, HOTFIX_ID, invalidated=candidate.invalidated
    )


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Invalidate old MSMAll records while retaining ordinary anatomy records."""
    return run_contract_hotfix(
        registry,
        projects=projects,
        execute=execute,
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        inspect=_candidate,
        repair=_repair,
    )
