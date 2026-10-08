"""Invalidate MSMAll stages that shared one mutable HCP study tree."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport

HOTFIX_ID = "v0295-msmall-stage-isolation"
SUMMARY = "Rebuild MSMAll stages with immutable structural inputs."


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


def _descendants(topology: list[dict[str, object]], roots: set[str]) -> set[str]:
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
    return found


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
    msmall_root = _legacy_msmall_root(topology)
    if msmall_root is None:
        return None
    roots = {
        node_id
        for node in topology
        for node_id in (node.get("id"),)
        if isinstance(node_id, str)
        and any(path.startswith(f"{msmall_root}/") for path in _paths(node))
    }
    invalidated = frozenset(_descendants(topology, roots))
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
    """Invalidate only contracts that used the shared mutable MSMAll tree."""
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
