"""Repair MSMAll inverse-warp coverage and dedrift input deletion."""

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

HOTFIX_ID = "v0317-msmall-stage-inputs"
SUMMARY = "Rebuild only MSMAll stages affected by inverse-warp coverage or deleted spheres."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    invalidated: frozenset[str]


def _marker(node: Mapping[str, object], stage: str) -> Path | None:
    suffix = f"/stages/{stage}.complete"
    matches = [
        Path(path)
        for path in node.get("outputs", ())
        if isinstance(path, str) and path.endswith(suffix)
    ]
    return matches[0] if len(matches) == 1 else None


def _is_msmall_node(node: Mapping[str, object]) -> bool:
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
    post = stage_node(topology, "postfreesurfer")
    dedrift = stage_node(topology, "dedrift")
    if atlas is None or post is None or dedrift is None:
        return None
    post_marker = _marker(post, "postfreesurfer")
    dedrift_marker = _marker(dedrift, "dedrift")
    if post_marker is None or dedrift_marker is None or dedrift_marker.exists():
        return None
    if post_marker.exists():
        root = post.get("id")
    else:
        root = atlas.get("id")
    if not isinstance(root, str):
        return None
    return _Candidate(contract, payload, descendants(topology, (root,), include_roots=True))


def _repair(candidate: _Candidate) -> None:
    rewrite_nodes(
        candidate.contract,
        candidate.payload,
        HOTFIX_ID,
        invalidated=candidate.invalidated,
        accept_relocated=_is_msmall_node,
    )


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Retain valid MSMAll checkpoints and remove affected downstream records."""
    return run_contract_hotfix(
        registry,
        projects=projects,
        execute=execute,
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        inspect=_candidate,
        repair=_repair,
    )
