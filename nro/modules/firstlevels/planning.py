"""Demand-driven work-item construction for participant/task/model GLMs."""

import json
import sys
from pathlib import Path

from nro.engine.bids import matches_filter, resolve_bids_table
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import work_item_key

from .paths import artifact_root, completion_path, work_item_prefix
from .task_models import scientific_model


def refresh_command(command: tuple[str, ...], processing: dict) -> tuple[str, ...]:
    """Update the embedded scientific snapshot when reassessment changes a model."""
    values = list(command)
    if "--model-definition" in values and "task_model" in processing:
        values[values.index("--model-definition") + 1] = json.dumps(
            processing["task_model"], sort_keys=True
        )
    return tuple(values)


def selected_runs(runs, model_id: str, participant: str, input_filter: dict | None = None) -> tuple:
    """Select runs of one task and participant that match the stable input filter."""
    task = model_id.split("/")[0]
    return tuple(
        run
        for run in runs
        if run.entities.get("task") == task
        and run.participant == participant
        and matches_filter(run.entities, input_filter)
    )


def select_model_runs(
    runs, config: dict, participant: str, *, entities: dict | None = None
) -> tuple:
    """Rediscover one registered work item's runs independently of model sets."""
    identifier = f"{entities['task']}/{entities['model']}"
    return selected_runs(runs, identifier, participant, config["input_filter"])


def direct_inputs(
    runs, config: dict, participant: str, entities: dict, *, markup=None
) -> tuple[Path, ...]:
    """Resolve inherited events; model semantics are tracked in the contract."""
    selected = select_model_runs(runs, config, participant, entities=entities)
    return tuple(resolve_bids_table(run.path, suffix="events", markup=markup) for run in selected)


def plan_work_items(context, upstream, descriptor) -> tuple[WorkItemSpec, ...]:
    """Construct each selected model/space/smoothing work item and its func dependencies."""
    values = context.workflow.configuration("firstlevels").values
    func = {work_item.output_prefix: work_item for work_item in upstream["func"]}
    directory = context.registered.directories["firstlevels"]
    result = []
    for model_id, document in context.task_models.items():
        source = scientific_model(document)
        runs = selected_runs(context.runs, model_id, context.participant, values["input_filter"])
        if not runs:
            continue
        for space, smoothing in context.target_pairs:
            root = artifact_root(
                context.project_root,
                directory,
                context.participant,
            )
            prefix = work_item_prefix(context.participant, model_id, space, smoothing)
            entities = {
                "task": model_id.split("/")[0],
                "model": model_id.split("/")[1],
                "space": space,
                "smoothing": str(smoothing),
            }
            result.append(
                WorkItemSpec.create(
                    key=work_item_key(
                        context.project,
                        descriptor.name,
                        context.registered.lineage_fingerprints["firstlevels"],
                        context.participant,
                        entities,
                    ),
                    module=descriptor.name,
                    project=context.project,
                    participant=context.participant,
                    entities=entities,
                    scope=descriptor.scope,
                    module_lineage_id=context.registered.lineages["firstlevels"],
                    config_fingerprint=context.workflow.configuration(
                        "firstlevels"
                    ).scientific_fingerprint,
                    directory_label=directory,
                    runtime_config=context.runtime_config("firstlevels"),
                    command=(
                        sys.executable,
                        "-m",
                        descriptor.execution_module,
                        "-p",
                        context.participant,
                        "-P",
                        context.project,
                        "--model",
                        model_id,
                        "--model-definition",
                        json.dumps(source, sort_keys=True),
                        "-s",
                        space,
                        "-S",
                        str(smoothing),
                    ),
                    dependencies=tuple(
                        dict.fromkeys(
                            [
                                *(work_item.key for work_item in upstream["anat"]),
                                *(func[run.stem].key for run in runs),
                            ]
                        )
                    ),
                    input_paths=direct_inputs(
                        runs,
                        values,
                        context.participant,
                        entities,
                        markup=context.source_markup,
                    ),
                    output_root=root,
                    output_prefix=prefix,
                    output_format=descriptor.output_format,
                    resource_class=descriptor.resource_class,
                    memory_gb=context.memory_gb,
                    max_memory_gb=context.max_memory_gb,
                    expected_outputs=(completion_path(root, prefix),),
                    processing=context.processing_contract(descriptor, task_model=source),
                )
            )
    return tuple(result)
