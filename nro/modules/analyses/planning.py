"""Plan scheduled study-analysis work items and their upstream module graphs."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.analysis.layout import LoadedAnalysis
from nro.analysis.manifest import InputSpec, bids_label
from nro.analysis.planning import AnalysisWorkItem as _Unit
from nro.analysis.planning import _placeholders, _scope_units
from nro.definitions.store import fingerprint
from nro.engine.bids import matches_filter
from nro.engine.paths import study_work_root
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import work_item_key

MODULE = "analyses"
SPECIFICATION_VERSION = 1


@dataclass(frozen=True)
class UpstreamRequest:
    """The upstream module graph that one input's workflow selects."""

    module: str
    workflow: str
    work_items: Mapping[str, WorkItemSpec]


def upstream_selections(analysis: LoadedAnalysis) -> tuple[tuple[str, str], ...]:
    """Return every (module, workflow) pair the analysis's inputs need.

    Expanded parameters can select the workflow, so the pairs cover every
    parameter combination. ``config:`` inputs are not schedulable yet.
    """
    pairs = []
    for parameters, _labels in analysis.manifest.combinations():
        for spec in analysis.manifest.inputs.values():
            if spec.module is None:
                continue
            if spec.config is not None:
                raise ValueError(f"input {spec.name}: scheduled runs need workflow:, not config:")
            assert spec.workflow is not None
            pairs.append((spec.module, spec.workflow.format(**parameters)))
    return tuple(dict.fromkeys(pairs))


def _selected_targets(spec: InputSpec) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return the spaces and smoothing levels that an input's selector names."""

    def values(key: str) -> tuple[str, ...]:
        value = spec.select.get(key)
        if value is None:
            return ()
        return tuple(str(item) for item in (value if isinstance(value, list) else [value]))

    smoothing = tuple(int(item.removesuffix("mm")) for item in values("smoothing"))
    return values("space"), smoothing


def upstream_targets(analysis: LoadedAnalysis, module: str, workflow: str):
    """Return the spaces and smoothing levels to plan for one upstream selection."""
    spaces: list[str] = []
    smoothing: list[int] = []
    for parameters, _labels in analysis.manifest.combinations():
        for spec in analysis.manifest.inputs.values():
            if spec.module == module and (spec.workflow or "").format(**parameters) == workflow:
                selected_spaces, selected_smoothing = _selected_targets(spec)
                spaces.extend(selected_spaces)
                smoothing.extend(selected_smoothing)
    return tuple(dict.fromkeys(spaces)), tuple(dict.fromkeys(smoothing))


def _matches_unit(
    spec: WorkItemSpec, select: Mapping[str, Any], run_entities: Mapping[str, str]
) -> bool:
    """Return whether an upstream work item may supply files for one input.

    Only selector keys and run entities that the work item records are
    compared; files are chosen when the analysis runs.
    """
    recorded = {str(key): str(value) for key, value in spec.entities.items()}
    shared = {key: value for key, value in select.items() if key in recorded}
    if not matches_filter(recorded, shared):
        return False
    return all(recorded.get(key, value) == value for key, value in run_entities.items())


def _write_specification(analysis: LoadedAnalysis, value: dict) -> Path:
    data = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    path = study_work_root(analysis.study.name) / "specifications" / f"{digest}.json"
    if not path.is_file():
        from nro.engine.io import atomic_write_text

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, data)
    return path


def analysis_contract(analysis: LoadedAnalysis, unit: _Unit, inputs: dict) -> dict:
    """Return the scientific contract of one analysis work item."""
    source = analysis.source
    name = analysis.manifest.analysis
    return {
        "study": analysis.study.name,
        "analysis": name,
        "version": analysis.manifest.version,
        "manifest": analysis.manifest.digest,
        "commit": source.commit,
        "code": source.tree_hash(f"analyses/{name}"),
        "common": source.tree_hash("common"),
        "container": analysis.container,
        "entry": analysis.manifest.entry,
        "parameters": json.loads(json.dumps(unit.parameters, sort_keys=True, default=str)),
        "inputs": inputs,
    }


def plan_work_items(
    analysis: LoadedAnalysis,
    *,
    lineage,
    upstream: Sequence[UpstreamRequest],
    bids_root: Path,
    projects: Sequence[str] | None = None,
    participants: Sequence[str] | None = None,
    memory_gb: int = 32,
    max_memory_gb: int = 256,
) -> tuple[tuple[WorkItemSpec, ...], tuple[tuple[str, str, str], ...]]:
    """Plan one work item per participant or run and expanded parameter combination.

    ``lineage`` is the analysis's unbound lineage. Each module input depends on
    every upstream work item of its workflow whose recorded entities agree with
    the selector and the run; the step chooses files from their outputs when it
    runs. Returns the specifications and ``(project, participant, reason)`` for
    each unit that an input cannot supply.
    """
    if source_is_worktree(analysis):
        raise ValueError("Scheduled analyses run from a commit, not the working tree")
    snapshot = analysis.source.snapshot()
    manifest = analysis.manifest
    members = tuple(
        project for project in analysis.study.projects if not projects or project in projects
    )
    by_selection = {(item.module, item.workflow): item for item in upstream}
    results: list[WorkItemSpec] = []
    unavailable: list[tuple[str, str, str]] = []
    for project in members:
        for participant, unit_entities in _scope_units(analysis, project, participants, bids_root):
            run_entities = {
                key: value for key, value in unit_entities.items() if key not in {"stem", "session"}
            }
            if "session" in unit_entities:
                run_entities["ses"] = unit_entities["session"]
            for parameters, labels in manifest.combinations():
                unit = _Unit(
                    analysis, project, participant, dict(unit_entities), parameters, labels
                )
                values = _placeholders(unit)
                output_root = analysis.results_root / f"project-{project}" / f"sub-{participant}"
                outputs = {
                    slot: output_root / pattern.format(**values)
                    for slot, pattern in manifest.outputs.items()
                }
                stem = unit_entities.get("stem", f"sub-{participant}")
                label = "".join(f"_{key}-{value}" for key, value in sorted(labels.items()))
                completion = (
                    output_root / f"{stem}_analysis-{bids_label(manifest.analysis)}{label}"
                    f"_manifest.json"
                )
                dependencies: list[str] = []
                inputs: dict[str, dict] = {}
                contract_inputs: dict[str, dict] = {}
                missing = None
                for name, spec in manifest.inputs.items():
                    entry = {
                        "module": spec.module,
                        "select": dict(spec.select),
                        "many": spec.many,
                        "min": spec.minimum,
                        "max": spec.maximum,
                        "optional": spec.optional,
                        "path": spec.path,
                        "upstream": [],
                    }
                    if spec.module is not None:
                        workflow = (spec.workflow or "").format(**parameters)
                        graph = by_selection[(spec.module, workflow)].work_items
                        matches = [
                            item
                            for item in graph.values()
                            if item.module == spec.module
                            and item.project == project
                            and item.participant == participant
                            and _matches_unit(item, spec.select, run_entities)
                        ]
                        entry["workflow"] = workflow
                        entry["upstream"] = [
                            {
                                "key": item.key,
                                "root": str(item.output_root),
                                "prefix": item.output_prefix,
                                "entities": dict(item.entities),
                            }
                            for item in matches
                        ]
                        dependencies.extend(item.key for item in matches)
                        if not matches and not spec.optional:
                            missing = (
                                f"input {name}: workflow {workflow} plans no {spec.module} work"
                            )
                    inputs[name] = entry
                    contract_inputs[name] = {
                        key: entry[key]
                        for key in (
                            "module",
                            "workflow",
                            "select",
                            "many",
                            "min",
                            "max",
                            "optional",
                            "path",
                        )
                        if key in entry
                    }
                if missing is not None:
                    unavailable.append((project, participant, missing))
                    continue
                entities = {key: value for key, value in run_entities.items()}
                for key, value in labels.items():
                    if key in entities:
                        raise ValueError(f"Expanded parameter {key} collides with a run entity")
                    entities[key] = value
                contract = analysis_contract(analysis, unit, contract_inputs)
                specification = _write_specification(
                    analysis,
                    {
                        "version": SPECIFICATION_VERSION,
                        "analysis": analysis.name,
                        "commit": analysis.source.commit,
                        "code_root": str(snapshot),
                        "entry": manifest.entry,
                        "container": analysis.container,
                        "parameters": parameters,
                        "entities": {
                            "project": project,
                            "participant": participant,
                            **unit_entities,
                        },
                        "run_entities": run_entities,
                        "inputs": inputs,
                        "outputs": {slot: str(path) for slot, path in outputs.items()},
                        "completion": str(completion),
                        "contract": contract,
                    },
                )
                lineage_fingerprint = lineage.lineage_fingerprint
                results.append(
                    WorkItemSpec.create(
                        key=work_item_key(
                            project, MODULE, lineage_fingerprint, participant, entities
                        ),
                        module=MODULE,
                        project=project,
                        participant=participant,
                        entities=entities,
                        scope=manifest.scope,
                        module_lineage_id=lineage.id,
                        config_fingerprint=fingerprint(contract),
                        directory_label=lineage.directory_label,
                        runtime_config=specification,
                        command=(
                            sys.executable,
                            "-m",
                            "nro.modules.analyses",
                            "--specification",
                            str(specification),
                        ),
                        dependencies=tuple(dict.fromkeys(dependencies)),
                        input_paths=(),
                        output_root=output_root,
                        output_prefix=None,
                        resource_class=manifest.resource_class,
                        output_format="Study-analysis output slots and a completion manifest",
                        memory_gb=manifest.memory_gb or memory_gb,
                        max_memory_gb=max(max_memory_gb, manifest.memory_gb or 0),
                        expected_outputs=(completion,),
                        processing={"analysis": contract},
                    )
                )
    keys = [spec.key for spec in results]
    if len(keys) != len(set(keys)):
        raise ValueError("Two analysis work items share an identity; check the output patterns")
    return tuple(results), tuple(unavailable)


def source_is_worktree(analysis: LoadedAnalysis) -> bool:
    """Return whether the analysis was read from uncommitted files."""
    return analysis.source.is_worktree
