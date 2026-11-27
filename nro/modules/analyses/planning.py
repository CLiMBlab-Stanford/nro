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
from nro.orchestration.contracts import PROJECT_SCOPE, WorkItemSpec
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


_CONTRACT_INPUT_KEYS = (
    "module",
    "analysis",
    "workflow",
    "select",
    "many",
    "min",
    "max",
    "optional",
    "path",
)


def _contract_inputs(inputs: Mapping[str, dict]) -> dict[str, dict]:
    """Return the declared form of each input, without the planned upstream items."""
    return {
        name: {key: entry[key] for key in _CONTRACT_INPUT_KEYS if key in entry}
        for name, entry in inputs.items()
    }


def _input_entry(
    spec: InputSpec,
    parameters: Mapping[str, Any],
    *,
    project: str,
    participant: str,
    run_entities: Mapping[str, str],
    by_selection: Mapping[tuple[str, str], UpstreamRequest],
    analysis_specs: Mapping[str, Sequence[WorkItemSpec]],
) -> tuple[dict, list[str], str | None]:
    """Plan one input for one participant.

    Returns the specification entry, the upstream work-item keys it depends on,
    and a reason when no upstream work item can supply it.
    """
    entry: dict[str, Any] = {
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
        candidates = by_selection[(spec.module, workflow)].work_items.values()
        entry["workflow"] = workflow
        source = f"workflow {workflow} plans no {spec.module} work"
    elif spec.analysis is not None:
        candidates = analysis_specs[spec.analysis]
        entry["analysis"] = spec.analysis
        source = f"analysis {spec.analysis} plans no work"
    else:
        return entry, [], None
    matches = [
        item
        for item in candidates
        if (spec.module is None or item.module == spec.module)
        and item.project == project
        and item.participant == participant
        and _matches_unit(item, spec.select, run_entities)
    ]
    entry["upstream"] = [
        {
            "key": item.key,
            "root": str(item.output_root),
            "prefix": item.output_prefix,
            "entities": dict(item.entities),
            "participant": item.participant,
            **({"completion": str(item.expected_outputs[0])} if spec.analysis is not None else {}),
        }
        for item in matches
    ]
    problem = None if matches else f"input {spec.name}: {source} for sub-{participant}"
    return entry, [item.key for item in matches], problem


def _work_item(
    analysis: LoadedAnalysis,
    *,
    lineage,
    project: str,
    participant: str,
    entities: Mapping[str, str],
    contract: dict,
    specification: Path,
    dependencies: Sequence[str],
    output_root: Path,
    completion: Path,
    memory_gb: int,
    max_memory_gb: int,
) -> WorkItemSpec:
    manifest = analysis.manifest
    scope = PROJECT_SCOPE if manifest.scope == "project" else manifest.scope
    return WorkItemSpec.create(
        key=work_item_key(project, MODULE, lineage.lineage_fingerprint, participant, entities),
        module=MODULE,
        project=project,
        participant=participant,
        entities=dict(entities),
        scope=scope,
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


def _plan_project_item(
    analysis: LoadedAnalysis,
    project: str,
    parameters: dict,
    labels: dict,
    *,
    lineage,
    snapshot: Path,
    by_selection,
    analysis_specs,
    bids_root: Path,
    memory_gb: int,
    max_memory_gb: int,
) -> WorkItemSpec | str:
    """Plan one project-scope work item, or return why none can be planned.

    The selector is expanded against the project's BIDS participants. A
    selected participant is included when every required input has planned
    upstream work, and excluded otherwise. Both lists enter the contract, so
    changing either one makes the work item stale. The item depends on the
    upstream work of every included participant.
    """
    from nro.analysis.planning import selected_participants

    manifest = analysis.manifest
    selected = selected_participants(analysis, project, bids_root)
    inputs: dict[str, dict] = {}
    dependencies: list[str] = []
    included: list[str] = []
    excluded: list[str] = []
    for participant in selected:
        entries = {}
        found: list[str] = []
        usable = True
        for name, spec in manifest.inputs.items():
            entry, keys, problem = _input_entry(
                spec,
                parameters,
                project=project,
                participant=participant,
                run_entities={},
                by_selection=by_selection,
                analysis_specs=analysis_specs,
            )
            if problem is not None and not spec.optional:
                usable = False
                break
            entries[name] = entry
            found.extend(keys)
        if not usable:
            excluded.append(participant)
            continue
        included.append(participant)
        dependencies.extend(found)
        for name, entry in entries.items():
            if name in inputs:
                inputs[name]["upstream"].extend(entry["upstream"])
            else:
                inputs[name] = entry
    if not included:
        return f"none of {len(selected)} selected participants has planned upstream work"
    unit = _Unit(analysis, project, None, {}, parameters, labels)
    values = _placeholders(unit)
    output_root = analysis.results_root / f"project-{project}"
    outputs = {
        slot: output_root / pattern.format(**values) for slot, pattern in manifest.outputs.items()
    }
    label = "".join(f"_{key}-{value}" for key, value in sorted(labels.items()))
    completion = (
        output_root / f"project-{project}_analysis-{bids_label(manifest.analysis)}{label}"
        "_manifest.json"
    )
    contract = {
        **analysis_contract(analysis, unit, _contract_inputs(inputs)),
        "participants": list(included),
        "excluded_participants": list(excluded),
    }
    specification = _write_specification(
        analysis,
        {
            "version": SPECIFICATION_VERSION,
            "analysis": analysis.name,
            "scope": "project",
            "commit": analysis.source.commit,
            "code_root": str(snapshot),
            "entry": manifest.entry,
            "container": analysis.container,
            "parameters": parameters,
            "participants": list(included),
            "excluded_participants": list(excluded),
            "entities": {"project": project},
            "run_entities": {},
            "inputs": inputs,
            "outputs": {slot: str(path) for slot, path in outputs.items()},
            "completion": str(completion),
            "contract": contract,
        },
    )
    return _work_item(
        analysis,
        lineage=lineage,
        project=project,
        participant="",
        entities=dict(labels),
        contract=contract,
        specification=specification,
        dependencies=dependencies,
        output_root=output_root,
        completion=completion,
        memory_gb=memory_gb,
        max_memory_gb=max_memory_gb,
    )


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
    analysis_specs: Mapping[str, Sequence[WorkItemSpec]] | None = None,
) -> tuple[tuple[WorkItemSpec, ...], tuple[tuple[str, str, str], ...]]:
    """Plan one work item per participant, run, or project and parameter combination.

    ``analysis_specs`` maps each analysis named by an ``analysis:`` input to the
    work items already planned for it. A project-scope analysis cannot be
    narrowed with ``participants``; its selector decides who it reads.

    ``lineage`` is the analysis's unbound lineage. Each module input depends on
    every upstream work item of its workflow whose recorded entities agree with
    the selector and the run; the step chooses files from their outputs when it
    runs. Returns the specifications and ``(project, participant, reason)`` for
    each unit that an input cannot supply.
    """
    if source_is_worktree(analysis):
        raise ValueError("Scheduled analyses run from a commit, not the working tree")
    if analysis.manifest.scope == "project" and participants:
        raise ValueError(
            f"{analysis.name} has project scope; its participants come from the manifest's "
            "participants selector, so --participant cannot narrow it"
        )
    analysis_specs = analysis_specs or {}
    snapshot = analysis.source.snapshot()
    manifest = analysis.manifest
    members = tuple(
        project for project in analysis.study.projects if not projects or project in projects
    )
    by_selection = {(item.module, item.workflow): item for item in upstream}
    results: list[WorkItemSpec] = []
    unavailable: list[tuple[str, str, str]] = []
    for project in members:
        if manifest.scope == "project":
            for parameters, labels in manifest.combinations():
                spec = _plan_project_item(
                    analysis,
                    project,
                    parameters,
                    labels,
                    lineage=lineage,
                    snapshot=snapshot,
                    by_selection=by_selection,
                    analysis_specs=analysis_specs,
                    bids_root=bids_root,
                    memory_gb=memory_gb,
                    max_memory_gb=max_memory_gb,
                )
                if isinstance(spec, str):
                    unavailable.append((project, "", spec))
                else:
                    results.append(spec)
            continue
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
                missing = None
                for name, spec in manifest.inputs.items():
                    entry, found, problem = _input_entry(
                        spec,
                        parameters,
                        project=project,
                        participant=participant,
                        run_entities=run_entities,
                        by_selection=by_selection,
                        analysis_specs=analysis_specs,
                    )
                    inputs[name] = entry
                    dependencies.extend(found)
                    if problem is not None and not spec.optional:
                        missing = problem
                if missing is not None:
                    unavailable.append((project, participant, missing))
                    continue
                entities = {key: value for key, value in run_entities.items()}
                for key, value in labels.items():
                    if key in entities:
                        raise ValueError(f"Expanded parameter {key} collides with a run entity")
                    entities[key] = value
                contract = analysis_contract(analysis, unit, _contract_inputs(inputs))
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
                results.append(
                    _work_item(
                        analysis,
                        lineage=lineage,
                        project=project,
                        participant=participant,
                        entities=entities,
                        contract=contract,
                        specification=specification,
                        dependencies=dependencies,
                        output_root=output_root,
                        completion=completion,
                        memory_gb=memory_gb,
                        max_memory_gb=max_memory_gb,
                    )
                )
    keys = [spec.key for spec in results]
    if len(keys) != len(set(keys)):
        raise ValueError("Two analysis work items share an identity; check the output patterns")
    return tuple(results), tuple(unavailable)


def source_is_worktree(analysis: LoadedAnalysis) -> bool:
    """Return whether the analysis was read from uncommitted files."""
    return analysis.source.is_worktree
