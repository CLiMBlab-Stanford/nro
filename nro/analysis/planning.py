"""Plan study-analysis work items, their output paths, and their freshness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.analysis.artifacts import Artifact, InputSet, file_entities
from nro.analysis.layout import LoadedAnalysis, load_analysis
from nro.analysis.manifest import InputSpec, ManifestError, bids_label
from nro.analysis.resolve import (
    Resolution,
    UpstreamIndex,
    _matches_run,
    _signature,
    choose,
    resolve_input,
)
from nro.engine.bids import matches_filter

PROVENANCE_SUFFIX = ".provenance.json"
STATUSES = ("ready", "fresh", "not_applicable", "waiting", "error")


def provenance_path(path: Path) -> Path:
    """Return the provenance sidecar of one published output."""
    return path.with_name(path.name + PROVENANCE_SUFFIX)


@dataclass
class AnalysisWorkItem:
    """One planned application of an analysis.

    A project-scope work item has no ``participant``. Its ``participants``
    lists the participants whose inputs it reads, and ``excluded`` lists the
    selected participants it leaves out.
    """

    analysis: LoadedAnalysis
    project: str
    participant: str | None
    entities: dict[str, str]
    parameters: dict[str, Any]
    labels: dict[str, str]
    outputs: dict[str, Path] = field(default_factory=dict)
    inputs: dict[str, Resolution] = field(default_factory=dict)
    status: str = "ready"
    message: str = ""
    signature: str | None = None
    participants: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """Return a readable work-item key."""
        parts = [self.analysis.name, f"project-{self.project}"]
        if self.participant is not None:
            parts.append(f"sub-{self.participant}")
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
    if item.participant is None:
        # Only project-scope signatures enumerate participants, so results of
        # other scopes keep the signatures that earlier releases recorded.
        document["participants"] = list(item.participants)
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode("utf-8")).hexdigest()


def _scope_units(
    analysis: LoadedAnalysis, project: str, participants: Sequence[str] | None, bids_root: Path
) -> list[tuple[str | None, dict[str, str]]]:
    """Return ``(participant, run entities)`` for each work-item unit of one project.

    A project-scope analysis has one unit without a participant.
    """
    if analysis.manifest.scope == "project":
        if participants:
            raise ValueError(
                f"{analysis.name} has project scope; its participants come from the "
                "manifest's participants selector, so --participant cannot narrow it"
            )
        return [(None, {})]
    from nro.orchestration.selection import discover_bids_participants

    project_root = bids_root / project
    available = discover_bids_participants(project_root) if project_root.is_dir() else ()
    if participants:
        wanted = {value.removeprefix("sub-") for value in participants}
        available = tuple(value for value in available if value in wanted)
    units: list[tuple[str | None, dict[str, str]]] = []
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


def selected_participants(
    analysis: LoadedAnalysis, project: str, bids_root: Path
) -> tuple[str, ...]:
    """Expand a project-scope analysis's participant selector for one project."""
    from nro.orchestration.selection import discover_bids_participants

    assert analysis.manifest.participants is not None
    project_root = bids_root / project
    available = discover_bids_participants(project_root) if project_root.is_dir() else ()
    return analysis.manifest.participants.select(available)


def _placeholders(item: AnalysisWorkItem) -> dict[str, str]:
    values = {"project": item.project}
    if item.participant is not None:
        values["participant"] = item.participant
    values.update(item.entities)
    for name, value in item.parameters.items():
        if name in item.labels:
            values[name] = item.labels[name]
        elif isinstance(value, (str, int, float, bool)):
            values[name] = bids_label(value)
    return values


class _Planner:
    """Plan analyses with one upstream index, caching upstream analysis plans."""

    def __init__(self, index: UpstreamIndex, bids_root: Path):
        self.index = index
        self.bids_root = bids_root
        self._upstream: dict[tuple, list[AnalysisWorkItem]] = {}
        self._active: list[str] = []

    def upstream_items(
        self,
        analysis: LoadedAnalysis,
        name: str,
        project: str,
        participants: tuple[str, ...] | None,
    ) -> list[AnalysisWorkItem]:
        """Plan another analysis of the same study at the same commit."""
        upstream = load_analysis(analysis.source, name)
        if upstream.manifest.scope == "project":
            raise ManifestError(
                f"{analysis.name} reads {upstream.name}, which has project scope; "
                "only subject- and run-scope analyses can supply other analyses"
            )
        key = (upstream.name, project, participants)
        if key not in self._upstream:
            if upstream.name in self._active:
                cycle = " -> ".join([*self._active, upstream.name])
                raise ManifestError(f"Analysis inputs form a cycle: {cycle}")
            self._upstream[key] = self.plan(
                upstream, projects=(project,), participants=participants
            )
        return self._upstream[key]

    def resolve(
        self,
        analysis: LoadedAnalysis,
        spec: InputSpec,
        *,
        project: str,
        participant: str,
        parameters: Mapping[str, Any],
        run_entities: Mapping[str, str] | None,
        missing: str = "error",
        upstream_participants: tuple[str, ...] | None = None,
    ) -> Resolution:
        """Resolve one input for one participant."""
        if spec.analysis is None:
            return resolve_input(
                spec,
                project=project,
                participant=participant,
                parameters=parameters,
                index=self.index,
                bids_root=self.bids_root,
                run_entities=run_entities,
                missing=missing,
            )
        try:
            items = self.upstream_items(analysis, spec.analysis, project, upstream_participants)
        except (ManifestError, ValueError) as error:
            return Resolution(spec.name, "error", message=f"input {spec.name}: {error}")
        lineage = f"{analysis.study.name}/{spec.analysis}"
        candidates: list[Artifact] = []
        pending: list[str] = []
        for item in items:
            # An upstream error, such as a participant without a required
            # module output, cannot resolve by waiting, so it supplies nothing.
            if item.participant != participant or item.status in {"not_applicable", "error"}:
                continue
            if not _matches_run(item.entities, run_entities):
                continue
            if item.status != "fresh":
                pending.append(f"{item.key} ({item.status})")
                continue
            for path in item.outputs.values():
                entities = {**item.entities, **file_entities(path)}
                if not matches_filter(entities, spec.select):
                    continue
                candidates.append(
                    Artifact(
                        path=path,
                        entities=entities,
                        module=None,
                        signature=_signature(
                            {"analysis": lineage, "signature": item.signature, "file": path.name}
                        ),
                        project=project,
                        participant=participant,
                        lineage=lineage,
                    )
                )
        return choose(
            spec,
            candidates,
            where=f"sub-{participant} in project {project}",
            pending=pending,
            lineage=lineage,
            missing=missing,
        )

    def plan(
        self,
        analysis: LoadedAnalysis,
        *,
        projects: Sequence[str] | None = None,
        participants: Sequence[str] | None = None,
        results_root: Path | None = None,
        entities: Mapping[str, str] | None = None,
    ) -> list[AnalysisWorkItem]:
        """Plan every work item of one analysis; see ``plan_analysis``."""
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
        self._active.append(analysis.name)
        try:
            items: list[AnalysisWorkItem] = []
            for project in members:
                for participant, unit in _scope_units(
                    analysis, project, participants, self.bids_root
                ):
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
                        if participant is None:
                            self._resolve_project(item)
                        else:
                            self._resolve_participant(item)
                        directory = root / f"project-{project}"
                        if participant is not None:
                            directory = directory / f"sub-{participant}"
                        values = _placeholders(item)
                        item.outputs = {
                            slot: directory / pattern.format(**values)
                            for slot, pattern in manifest.outputs.items()
                        }
                        if item.status == "ready":
                            item.signature = work_item_signature(item)
                            item.status = "fresh" if item.is_published() else "ready"
                        items.append(item)
        finally:
            self._active.pop()
        _reject_collisions(items)
        return items

    def _resolve_participant(self, item: AnalysisWorkItem) -> None:
        assert item.participant is not None
        unit = item.entities
        run_entities = {key: value for key, value in unit.items() if key not in {"stem", "session"}}
        if "session" in unit:
            run_entities["ses"] = unit["session"]
        for name, spec in item.analysis.manifest.inputs.items():
            item.inputs[name] = self.resolve(
                item.analysis,
                spec,
                project=item.project,
                participant=item.participant,
                parameters=item.parameters,
                run_entities=run_entities or None,
            )
        problems = [value for value in item.inputs.values() if value.status != "ok"]
        if problems:
            order = {"error": 0, "waiting": 1, "not_applicable": 2}
            problems.sort(key=lambda value: order[value.status])
            item.status = problems[0].status
            item.message = "; ".join(value.message for value in problems)

    def _resolve_project(self, item: AnalysisWorkItem) -> None:
        """Resolve every input for each selected participant and combine them.

        A participant is included when all its inputs resolve. One that an
        input does not apply to is excluded. A pending participant holds the
        work item or is excluded, as the selector's ``pending`` says. Any
        other problem makes the work item an error.
        """
        analysis = item.analysis
        selection = analysis.manifest.participants
        assert selection is not None
        selected = selected_participants(analysis, item.project, self.bids_root)
        included: list[str] = []
        excluded: list[str] = []
        problems: dict[str, list[str]] = {"error": [], "waiting": []}
        combined: dict[str, list[Artifact]] = {name: [] for name in analysis.manifest.inputs}
        lineages: dict[str, str | None] = {}
        for participant in selected:
            resolutions = {
                name: self.resolve(
                    analysis,
                    spec,
                    project=item.project,
                    participant=participant,
                    parameters=item.parameters,
                    run_entities=None,
                    missing="not_applicable",
                    upstream_participants=selected,
                )
                for name, spec in analysis.manifest.inputs.items()
            }
            statuses = {value.status for value in resolutions.values()}
            if "error" in statuses:
                problems["error"].extend(
                    value.message for value in resolutions.values() if value.status == "error"
                )
                continue
            if "waiting" in statuses and selection.pending == "wait":
                problems["waiting"].append(f"sub-{participant}")
                continue
            if statuses != {"ok"}:
                excluded.append(participant)
                continue
            included.append(participant)
            for name, resolution in resolutions.items():
                lineages.setdefault(name, resolution.lineage)
                value = resolution.value
                if value is None:
                    continue
                combined[name].extend(value.artifacts if isinstance(value, InputSet) else [value])
        item.participants = tuple(included)
        item.excluded = tuple(excluded)
        item.parameters = {
            **item.parameters,
            "participants": list(included),
            "excluded_participants": list(excluded),
        }
        item.inputs = {
            name: Resolution(name, "ok", InputSet(tuple(artifacts)), lineage=lineages.get(name))
            for name, artifacts in combined.items()
        }
        if problems["error"]:
            item.status = "error"
            item.message = "; ".join(problems["error"][:5]) + (
                f" (and {len(problems['error']) - 5} more)" if len(problems["error"]) > 5 else ""
            )
        elif problems["waiting"]:
            waiting = problems["waiting"]
            item.status = "waiting"
            item.message = (
                f"{len(waiting)} selected participants have upstream work that is not fresh: "
                + ", ".join(waiting[:5])
                + (" ..." if len(waiting) > 5 else "")
            )
        elif not included:
            item.status = "not_applicable"
            item.message = f"none of {len(selected)} selected participants has every input"
        else:
            item.message = f"{len(included)} participants included, {len(excluded)} excluded"


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
    ``participants`` narrows subject and run scope; a project-scope analysis
    raises ValueError when it is given. Inputs from another analysis plan that
    analysis too, at the same commit.
    """
    return _Planner(index, Path(bids_root)).plan(
        analysis,
        projects=projects,
        participants=participants,
        results_root=results_root,
        entities=entities,
    )


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
