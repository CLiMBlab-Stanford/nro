"""Plan study-analysis work items, their output paths, and their freshness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.analysis.layout import LoadedAnalysis
from nro.analysis.manifest import bids_label
from nro.analysis.resolve import Resolution, UpstreamIndex, resolve_input

PROVENANCE_SUFFIX = ".provenance.json"
STATUSES = ("ready", "fresh", "not_applicable", "waiting", "error")


def provenance_path(path: Path) -> Path:
    """Return the provenance sidecar of one published output."""
    return path.with_name(path.name + PROVENANCE_SUFFIX)


@dataclass
class AnalysisWorkItem:
    """One planned application of an analysis."""

    analysis: LoadedAnalysis
    project: str
    participant: str
    entities: dict[str, str]
    parameters: dict[str, Any]
    labels: dict[str, str]
    outputs: dict[str, Path] = field(default_factory=dict)
    inputs: dict[str, Resolution] = field(default_factory=dict)
    status: str = "ready"
    message: str = ""
    signature: str | None = None

    @property
    def key(self) -> str:
        """Return a readable work-item key."""
        parts = [self.analysis.name, f"project-{self.project}", f"sub-{self.participant}"]
        if "stem" in self.entities:
            parts.append(self.entities["stem"])
        parts.extend(f"{name}-{label}" for name, label in sorted(self.labels.items()))
        return "/".join(parts)

    def input_values(self) -> dict[str, Any]:
        """Return each input's resolved artifact, artifact set, or ``None``."""
        return {name: resolution.value for name, resolution in self.inputs.items()}

    def is_published(self) -> bool:
        """Return whether every output carries a provenance sidecar for this signature."""
        if self.signature is None:
            return False
        for path in self.outputs.values():
            sidecar = provenance_path(path)
            if not path.is_file() or not sidecar.is_file():
                return False
            try:
                recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("Signature")
            except (OSError, ValueError):
                return False
            if recorded != self.signature:
                return False
        return True


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _input_signature(resolution: Resolution) -> Any:
    value = resolution.value
    if value is None:
        return None
    artifacts = [value] if not hasattr(value, "artifacts") else list(value.artifacts)
    return {
        "lineage": resolution.lineage,
        "files": [[item.path.name, item.signature] for item in artifacts],
    }


def work_item_signature(item: AnalysisWorkItem) -> str:
    """Hash the code, manifest, parameters, entities, and inputs of one work item."""
    analysis = item.analysis
    source = analysis.source
    name = analysis.manifest.analysis
    document = {
        "analysis": analysis.name,
        "version": analysis.manifest.version,
        "manifest": analysis.manifest.digest,
        "code": source.tree_hash(f"analyses/{name}"),
        "common": source.tree_hash("common"),
        "container": analysis.container,
        "parameters": _jsonable(item.parameters),
        "project": item.project,
        "participant": item.participant,
        "entities": dict(sorted(item.entities.items())),
        "inputs": {key: _input_signature(value) for key, value in sorted(item.inputs.items())},
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode("utf-8")).hexdigest()


def _scope_units(
    analysis: LoadedAnalysis, project: str, participants: Sequence[str] | None, bids_root: Path
) -> list[tuple[str, dict[str, str]]]:
    from nro.orchestration.selection import discover_bids_participants

    project_root = bids_root / project
    available = discover_bids_participants(project_root) if project_root.is_dir() else ()
    if participants:
        wanted = {value.removeprefix("sub-") for value in participants}
        available = tuple(value for value in available if value in wanted)
    units: list[tuple[str, dict[str, str]]] = []
    for participant in available:
        if analysis.manifest.scope == "subject":
            units.append((participant, {}))
            continue
        from nro.engine.bids import discover_raw_runs

        for run in discover_raw_runs(project_root / f"sub-{participant}"):
            entities = {str(key): str(value) for key, value in run.entities.items() if key != "sub"}
            entities["stem"] = run.stem
            if run.session:
                entities["session"] = run.session
            units.append((participant, entities))
    return units


def _placeholders(item: AnalysisWorkItem) -> dict[str, str]:
    values = {"project": item.project, "participant": item.participant}
    values.update(item.entities)
    for name, value in item.parameters.items():
        if name in item.labels:
            values[name] = item.labels[name]
        elif isinstance(value, (str, int, float, bool)):
            values[name] = bids_label(value)
    return values


def plan_analysis(
    analysis: LoadedAnalysis,
    *,
    index: UpstreamIndex,
    bids_root: Path,
    projects: Sequence[str] | None = None,
    participants: Sequence[str] | None = None,
    results_root: Path | None = None,
    entities: Mapping[str, str] | None = None,
) -> list[AnalysisWorkItem]:
    """Plan every work item of one analysis across its study's member projects.

    ``entities`` restricts run-scope work items to runs with matching entities.
    """
    manifest = analysis.manifest
    members = analysis.study.projects
    if projects:
        unknown = sorted(set(projects) - set(members))
        if unknown:
            raise ValueError(
                f"Study {analysis.study.name} does not include project(s) {', '.join(unknown)}"
            )
        members = tuple(project for project in members if project in projects)
    root = analysis.results_root if results_root is None else Path(results_root)
    items: list[AnalysisWorkItem] = []
    for project in members:
        for participant, unit in _scope_units(analysis, project, participants, bids_root):
            if entities and any(unit.get(key) != value for key, value in entities.items()):
                continue
            for parameters, labels in manifest.combinations():
                item = AnalysisWorkItem(
                    analysis=analysis,
                    project=project,
                    participant=participant,
                    entities=dict(unit),
                    parameters=parameters,
                    labels=labels,
                )
                directory = root / f"project-{project}" / f"sub-{participant}"
                values = _placeholders(item)
                item.outputs = {
                    slot: directory / pattern.format(**values)
                    for slot, pattern in manifest.outputs.items()
                }
                run_entities = {
                    key: value for key, value in unit.items() if key not in {"stem", "session"}
                }
                if "session" in unit:
                    run_entities["ses"] = unit["session"]
                for name, spec in manifest.inputs.items():
                    item.inputs[name] = resolve_input(
                        spec,
                        project=project,
                        participant=participant,
                        parameters=parameters,
                        index=index,
                        bids_root=bids_root,
                        run_entities=run_entities or None,
                    )
                problems = [value for value in item.inputs.values() if value.status != "ok"]
                if problems:
                    order = {"error": 0, "waiting": 1, "not_applicable": 2}
                    problems.sort(key=lambda value: order[value.status])
                    item.status = problems[0].status
                    item.message = "; ".join(value.message for value in problems)
                else:
                    item.signature = work_item_signature(item)
                    item.status = "fresh" if item.is_published() else "ready"
                items.append(item)
    _reject_collisions(items)
    return items


def _reject_collisions(items: Sequence[AnalysisWorkItem]) -> None:
    owners: dict[Path, str] = {}
    for item in items:
        for path in item.outputs.values():
            previous = owners.setdefault(path, item.key)
            if previous != item.key:
                raise ValueError(
                    f"Work items {previous} and {item.key} both write {path}; "
                    "add the distinguishing entity to the output pattern"
                )
