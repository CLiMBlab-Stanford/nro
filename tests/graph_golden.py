"""Record frozen runner graphs and compare them with stored golden files.

Builders are refactored only when their frozen graph contract stays identical,
because a changed step contract can make existing derivatives stale. A golden
record holds, for each step, the contract node, the presentation fields, and
the paths that its action and validator closures can reach.

Signatures normally hash absolute paths, which differ between test runs.
While a graph is built and recorded, the signature function returns the
canonical JSON of its argument instead, and every string under the test root
is rewritten to start with ``<ROOT>``. Step IDs derived from output paths are
recomputed from the rewritten paths.

Set ``UPDATE_GRAPH_GOLDENS=1`` to rewrite the golden files from the current
code. Do that only for an intended and reviewed graph change.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nro.definitions.store import _jsonable
from nro.orchestration import runner_graph

GOLDEN_ROOT = Path(__file__).parent / "fixtures" / "graphs"
_CLOSURE_DEPTH = 6


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _rewrite(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace(root, "<ROOT>")
    if isinstance(value, list):
        return [_rewrite(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite(item, root) for key, item in value.items()}
    return value


def _reachable_paths(value: Any, found: set[str], depth: int, seen: dict[int, Any]) -> None:
    # `seen` keeps each visited object alive, so a temporary object's id
    # cannot be reused by a later one and mistaken for a visited object.
    if depth < 0 or id(value) in seen:
        return
    seen[id(value)] = value
    if isinstance(value, Path):
        found.add(str(value))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _reachable_paths(item, found, depth - 1, seen)
    elif isinstance(value, dict):
        for item in value.values():
            _reachable_paths(item, found, depth - 1, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _reachable_paths(getattr(value, field.name, None), found, depth - 1, seen)
        if type(value).__dataclass_params__.frozen:
            # Frozen layout objects expose derived paths as properties.
            for name, member in vars(type(value)).items():
                if isinstance(member, property):
                    _reachable_paths(getattr(value, name), found, depth - 1, seen)
    elif callable(value):
        for cell in getattr(value, "__closure__", None) or ():
            try:
                contents = cell.cell_contents
            except ValueError:
                continue
            _reachable_paths(contents, found, depth - 1, seen)
        bound = getattr(value, "__self__", None)
        if bound is not None:
            _reachable_paths(bound, found, depth - 1, seen)


def _closure_paths(step: runner_graph.Step) -> list[str]:
    found: set[str] = set()
    for function in (step.action, step.validate, step.prepare, step.finalize):
        if function is not None:
            _reachable_paths(function, found, _CLOSURE_DEPTH, {})
    return sorted(found)


def _stable_ids(graph, root: str) -> dict[str, str]:
    """Map output-derived step IDs to IDs computed from root-neutral paths.

    A step without an explicit ID is named from a hash of its absolute output
    paths. Declared IDs are kept, so a changed declaration still shows.
    """
    stable = {}
    for step in graph.ordered_steps():
        if step.id != runner_graph.RunnerGraph._node_id(step.outputs):
            stable[step.id] = step.id
            continue
        slug = step.id.rsplit("-", 1)[0]
        identity = "\0".join(
            str(runner_graph._declared_path(path)).replace(root, "<ROOT>") for path in step.outputs
        )
        stable[step.id] = f"{slug}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
    return stable


def record_graph(build: Callable[[], Any], root: Path, monkeypatch) -> dict[str, Any]:
    """Build a runner with `build`, freeze its graph, and return a path-neutral record."""
    with monkeypatch.context() as patch:
        patch.setattr(runner_graph, "fingerprint", _canonical)
        graph = build()._graph.freeze()
        ids = _stable_ids(graph, str(root))
        steps = []
        for step in graph.ordered_steps():
            node = graph._contract_step(step)
            node["id"] = ids[node["id"]]
            node["dependencies"] = [ids.get(value, value) for value in node["dependencies"]]
            steps.append(
                {
                    "contract": node,
                    "name": step.name,
                    "resource_class": step.resource_class,
                    "completion_boundary": step.completion_boundary,
                    "closure_paths": _closure_paths(step),
                }
            )
    return _rewrite({"module": graph.module_name, "steps": steps}, str(root))


def assert_matches_golden(name: str, record: dict[str, Any]) -> None:
    """Compare a graph record with its golden file, or rewrite the file when requested."""
    path = GOLDEN_ROOT / f"{name}.json"
    if os.environ.get("UPDATE_GRAPH_GOLDENS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    if not path.is_file():
        pytest.fail(f"Missing golden graph {path}; set UPDATE_GRAPH_GOLDENS=1 to create it")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert record["module"] == expected["module"]
    assert len(record["steps"]) == len(expected["steps"]), "Step count changed"
    for current, golden in zip(record["steps"], expected["steps"], strict=True):
        label = f"{golden['contract']['id']} ({golden['name']})"
        for key in ("contract", "name", "resource_class", "completion_boundary"):
            assert current[key] == golden[key], f"{label}: {key} changed"
        missing = sorted(set(golden["closure_paths"]) - set(current["closure_paths"]))
        assert not missing, f"{label}: action no longer reaches {missing}"
