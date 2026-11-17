"""Preserve PreFreeSurfer while replacing its inferred brain masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.hotfixes._runner_contracts import (
    descendants,
    nodes,
    rewrite_nodes,
    run_contract_hotfix,
    stage_node,
)

HOTFIX_ID = "v0301-msmall-source-masks"
SUMMARY = "Rebuild MSMAll descendants with masks derived from nro inputs."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    prefreesurfer_id: str
    invalidated: frozenset[str]


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
    if stage_node(topology, "prefreesurfer_masks") is not None:
        return None
    prefreesurfer = stage_node(topology, "prefreesurfer")
    if prefreesurfer is None or not isinstance(prefreesurfer.get("id"), str):
        return None
    prefreesurfer_id = str(prefreesurfer["id"])
    return _Candidate(
        contract=contract,
        payload=payload,
        prefreesurfer_id=prefreesurfer_id,
        invalidated=descendants(topology, (prefreesurfer_id,), include_roots=False),
    )


def _repair(candidate: _Candidate) -> None:
    # The PreFreeSurfer command did not change. Its shared shell driver gained a
    # separate mask-repair operation for the new downstream step.
    rewrite_nodes(
        candidate.contract,
        candidate.payload,
        HOTFIX_ID,
        invalidated=candidate.invalidated,
        accept_relocated=lambda node: node.get("id") == candidate.prefreesurfer_id,
    )


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Retain valid PreFreeSurfer results and invalidate mask consumers."""
    return run_contract_hotfix(
        registry,
        projects=projects,
        execute=execute,
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        inspect=_candidate,
        repair=_repair,
    )
