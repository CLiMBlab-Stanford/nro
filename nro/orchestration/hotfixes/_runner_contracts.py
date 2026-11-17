"""Shared helpers for hotfixes that rewrite anatomical runner contracts.

A runner contract records the step topology and the completed nodes of one
anatomical work item. These hotfixes find contracts produced by a defective
release, remove the affected nodes so that the runner recomputes them, and mark
the contract with the hotfix signature so it is not repaired twice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport


class ContractCandidate(Protocol):
    """A contract selected for repair."""

    contract: Path


Candidate = TypeVar("Candidate", bound=ContractCandidate)


def nodes(payload: Mapping[str, object], field: str) -> list[dict[str, object]]:
    """Return copies of the mapping entries in a contract list such as `topology` or `nodes`."""
    values = payload.get(field)
    if not isinstance(values, list):
        return []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def node_paths(node: Mapping[str, object]) -> tuple[str, ...]:
    """Return a node's input paths followed by its output paths."""
    return tuple(
        value
        for field in ("inputs", "outputs")
        for value in node.get(field, ())
        if isinstance(value, str)
    )


def stage_node(topology: list[dict[str, object]], stage: str) -> dict[str, object] | None:
    """Return the one node that writes `stages/<stage>.complete`, or None if not exactly one."""
    suffix = f"/stages/{stage}.complete"
    matches = [
        node
        for node in topology
        if isinstance(node.get("outputs"), list)
        and any(isinstance(path, str) and path.endswith(suffix) for path in node["outputs"])
    ]
    return matches[0] if len(matches) == 1 else None


def descendants(
    topology: list[dict[str, object]], roots: Iterable[str], *, include_roots: bool
) -> frozenset[str]:
    """Return the IDs of nodes that depend, directly or transitively, on `roots`.

    With `include_roots`, the roots themselves are part of the result.
    """
    children: dict[str, set[str]] = {}
    for node in topology:
        node_id = node.get("id")
        dependencies = node.get("dependencies")
        if not isinstance(node_id, str) or not isinstance(dependencies, list):
            continue
        for dependency in dependencies:
            if isinstance(dependency, str):
                children.setdefault(dependency, set()).add(node_id)
    roots = tuple(roots)
    found: set[str] = set()
    pending = [child for root in roots for child in children.get(root, ())]
    while pending:
        node_id = pending.pop()
        if node_id in found:
            continue
        found.add(node_id)
        pending.extend(children.get(node_id, ()))
    if include_roots:
        found.update(roots)
    return frozenset(found)


def rewrite_nodes(
    contract: Path,
    payload: Mapping[str, object],
    identifier: str,
    *,
    invalidated: frozenset[str],
    accept_relocated: Callable[[dict[str, object]], bool] = lambda _node: False,
) -> None:
    """Drop invalidated completed nodes and sign the contract with the hotfix.

    Retained nodes for which `accept_relocated` is true may adopt their current
    step signatures once. The contract file is replaced atomically.
    """
    repaired = dict(payload)
    kept: list[dict[str, object]] = []
    for node in nodes(payload, "nodes"):
        if node.get("id") in invalidated:
            continue
        if accept_relocated(node):
            node["accept_relocated_signatures"] = True
        kept.append(node)
    repaired["nodes"] = kept
    repaired["signature"] = f"hotfix:{identifier}"
    atomic_write_json(contract, repaired, sort_keys=True)


def anatomical_contracts(control: Path, projects: tuple[str, ...]) -> Iterable[Path]:
    """Yield every branch's anatomical runner contracts for the projects, in sorted order."""
    branches = ControlPaths(control).root / "branches"
    for project in projects:
        yield from sorted(branches.glob(f"*/events/{project}/anat/**/runner-contract.json"))


def run_contract_hotfix(
    registry,
    *,
    projects: tuple[str, ...],
    execute: bool,
    identifier: str,
    summary: str,
    inspect: Callable[[Path], Candidate | None],
    repair: Callable[[Candidate], None],
) -> HotfixReport:
    """Find and, when `execute` is true, repair the selected projects' contracts.

    `inspect` returns a candidate for a contract that needs repair, or None.
    Raises ValueError when no project is given, a project is not in BIDS, or,
    when executing, a selected project still has queued or running attempts.
    """
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
        candidates = tuple(
            candidate
            for contract in anatomical_contracts(registry.paths.control, selected)
            if (candidate := inspect(contract)) is not None
        )
        if execute:
            for candidate in candidates:
                repair(candidate)
    return HotfixReport(
        identifier=identifier,
        summary=summary,
        projects=selected,
        paths=tuple(candidate.contract for candidate in candidates),
        records=len(candidates),
        applied=execute,
    )
