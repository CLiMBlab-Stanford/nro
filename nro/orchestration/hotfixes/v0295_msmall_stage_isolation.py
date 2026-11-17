"""Invalidate MSMAll stages that shared one mutable HCP study tree."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.hotfixes._runner_contracts import (
    descendants,
    node_paths,
    nodes,
    rewrite_nodes,
    run_contract_hotfix,
)

HOTFIX_ID = "v0295-msmall-stage-isolation"
SUMMARY = "Rebuild MSMAll stages with immutable structural inputs."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    payload: dict[str, object]
    invalidated: frozenset[str]


def _legacy_msmall_root(topology: list[dict[str, object]]) -> str | None:
    """Return the shared MSMAll root used before stage isolation."""
    roots: set[str] = set()
    for node in topology:
        outputs = node.get("outputs")
        if not isinstance(outputs, list):
            continue
        paths = tuple(path for path in outputs if isinstance(path, str))
        if not any(path.endswith("/stages/prefreesurfer.complete") for path in paths):
            continue
        restore_paths = [
            path
            for path in paths
            if "/msmall/study/" in path
            and path.endswith(("/T1w_acpc_dc_restore.nii.gz", "/T2w_acpc_dc_restore.nii.gz"))
        ]
        if len(restore_paths) == 2:
            roots.update(path.split("/study/", 1)[0] for path in restore_paths)
    return next(iter(roots)) if len(roots) == 1 else None


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
    msmall_root = _legacy_msmall_root(topology)
    if msmall_root is None:
        return None
    roots = {
        node_id
        for node in topology
        for node_id in (node.get("id"),)
        if isinstance(node_id, str)
        and any(path.startswith(f"{msmall_root}/") for path in node_paths(node))
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
    """Invalidate only contracts that used the shared mutable MSMAll tree."""
    return run_contract_hotfix(
        registry,
        projects=projects,
        execute=execute,
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        inspect=_candidate,
        repair=_repair,
    )
