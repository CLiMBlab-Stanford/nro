"""Repair MSMAll inverse-warp domains without repeating surface reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.hotfixes._runner_contracts import (
    descendants,
    nodes,
    rewrite_nodes,
    run_contract_hotfix,
    stage_node,
)

HOTFIX_ID = "v0307-msmall-inverse-warp"
SUMMARY = "Rebuild MSMAll inverse warps on padded subject-space grids."


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
        isinstance(path, str) and "/msmall/" in path
        for field in ("inputs", "outputs")
        for path in node.get(field, ())
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
    atlas = stage_node(topology, "masked_atlas")
    freesurfer = stage_node(topology, "freesurfer")
    if atlas is None or freesurfer is None:
        return None
    outputs = atlas.get("outputs")
    if not isinstance(outputs, list):
        return None
    if any(
        isinstance(path, str) and path.endswith("/xfms/standard2acpc_dc.nii.gz") for path in outputs
    ):
        return None
    atlas_id = atlas.get("id")
    freesurfer_id = freesurfer.get("id")
    if not isinstance(atlas_id, str) or not isinstance(freesurfer_id, str):
        return None
    invalidated = set(descendants(topology, (atlas_id,), include_roots=True))
    invalidated.discard(freesurfer_id)
    return _Candidate(contract=contract, payload=payload, invalidated=frozenset(invalidated))


def _repair(candidate: _Candidate) -> None:
    rewrite_nodes(
        candidate.contract,
        candidate.payload,
        HOTFIX_ID,
        invalidated=candidate.invalidated,
        accept_relocated=_is_msmall_node,
    )


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Invalidate defective inverse warps while retaining independent MSMAll stages."""
    return run_contract_hotfix(
        registry,
        projects=projects,
        execute=execute,
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        inspect=_candidate,
        repair=_repair,
    )
