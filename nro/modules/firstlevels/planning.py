"""Demand-driven instance construction for participant/task/model GLMs."""

import json
import sys
from pathlib import Path

from nro.engine.bids import resolve_bids_table
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import instance_key

from .paths import artifact_root, completion_path, instance_prefix
from .task_models import scientific_model


def refresh_command(command: tuple[str, ...], processing: dict) -> tuple[str, ...]:
    """Update the embedded scientific snapshot when reassessment changes a model."""
    values = list(command)
    if "--model-definition" in values and "task_model" in processing:
        values[values.index("--model-definition") + 1] = json.dumps(
            processing["task_model"], sort_keys=True
        )
    return tuple(values)


def selected_runs(runs, model_id: str, participant: str) -> tuple:
    """Select all runs of the registered task for one participant."""
    task = model_id.split("/")[0]
    return tuple(
        run for run in runs if run.entities.get("task") == task and run.participant == participant
    )


def select_model_runs(
    runs, config: dict, participant: str, *, entities: dict | None = None
) -> tuple:
    """Rediscover one registered instance's runs independently of model sets."""
    identifier = f"{entities['task']}/{entities['model']}"
    return selected_runs(runs, identifier, participant)


def direct_inputs(runs, config: dict, participant: str, entities: dict) -> tuple[Path, ...]:
    """Resolve inherited events; model semantics are tracked in the contract."""
    selected = select_model_runs(runs, config, participant, entities=entities)
    return tuple(resolve_bids_table(run.path, suffix="events") for run in selected)


def plan_instances(context, upstream, descriptor) -> tuple[InstanceSpec, ...]:
    """Construct each selected model/space/smoothing instance and its func dependencies."""
    values = context.workflow.configuration("firstlevels").values
    func = {instance.output_prefix: instance for instance in upstream["func"]}
    directory = context.registered.directories["firstlevels"]
    result = []
    for model_id, document in context.task_models.items():
        source = scientific_model(document)
        runs = selected_runs(context.runs, model_id, context.participant)
        if not runs:
            continue
        for space, smoothing in context.target_pairs:
            root = artifact_root(
                context.project_root,
                directory,
                context.participant,
            )
            prefix = instance_prefix(context.participant, model_id, space, smoothing)
            entities = {
                "task": model_id.split("/")[0],
                "model": model_id.split("/")[1],
                "space": space,
                "smoothing": str(smoothing),
            }
            result.append(
                InstanceSpec.create(
                    key=instance_key(
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
                    configuration_lineage_id=context.registered.lineages["firstlevels"],
                    config_fingerprint=context.workflow.configuration(
                        "firstlevels"
                    ).scientific_fingerprint,
                    directory_label=directory,
                    runtime_config=context.runtime_config("firstlevels"),
                    command=(
                        sys.executable,
                        "-m",
                        "nro.modules.firstlevels",
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
                                *(instance.key for instance in upstream["anat"]),
                                *(func[run.stem].key for run in runs),
                            ]
                        )
                    ),
                    input_paths=direct_inputs(runs, values, context.participant, entities),
                    output_root=root,
                    output_prefix=prefix,
                    output_format=descriptor.output_format,
                    resource_class=descriptor.resource_class,
                    memory_gb=context.memory_gb,
                    max_memory_gb=context.max_memory_gb,
                    expected_outputs=(completion_path(root, prefix),),
                    processing={**descriptor.processing_contract(), "task_model": source},
                )
            )
    return tuple(result)
