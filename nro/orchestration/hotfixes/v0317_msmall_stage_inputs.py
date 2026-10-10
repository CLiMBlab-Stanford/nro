"""Repair MSMAll inverse-warp coverage and dedrift input deletion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport

HOTFIX_ID = "v0317-msmall-stage-inputs"
SUMMARY = "Rebuild only MSMAll stages affected by inverse-warp coverage or deleted spheres."


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


def _stage_node(nodes: list[dict[str, object]], stage: str) -> dict[str, object] | None:
    suffix = f"/stages/{stage}.complete"
    matches = [
        node
        for node in nodes
        if any(isinstance(path, str) and path.endswith(suffix) for path in node.get("outputs", ()))
    ]
    return matches[0] if len(matches) == 1 else None


def _marker(node: Mapping[str, object], stage: str) -> Path | None:
    suffix = f"/stages/{stage}.complete"
    matches = [
        Path(path)
        for path in node.get("outputs", ())
        if isinstance(path, str) and path.endswith(suffix)
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
    found = {root}
    pending = [root]
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return frozenset(found)


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
    topology = _nodes(payload, "topology")
    atlas = _stage_node(topology, "masked_atlas")
    post = _stage_node(topology, "postfreesurfer")
    dedrift = _stage_node(topology, "dedrift")
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
    return _Candidate(contract, payload, _descendants(topology, root))


def _repair(candidate: _Candidate) -> None:
    payload = dict(candidate.payload)
    repaired: list[dict[str, object]] = []
    for node in _nodes(payload, "nodes"):
        if node.get("id") in candidate.invalidated:
            continue
        if _is_msmall_node(node):
            node["accept_relocated_signatures"] = True
        repaired.append(node)
    payload["nodes"] = repaired
    payload["signature"] = f"hotfix:{HOTFIX_ID}"
    atomic_write_json(candidate.contract, payload, sort_keys=True)


def _candidates(control: Path, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    branches = ControlPaths(control).root / "branches"
    return tuple(
        candidate
        for project in projects
        for contract in sorted(branches.glob(f"*/events/{project}/anat/**/runner-contract.json"))
        if (candidate := _candidate(contract)) is not None
    )


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Retain valid MSMAll checkpoints and remove affected downstream records."""
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
