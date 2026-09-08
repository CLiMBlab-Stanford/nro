"""Planner-facing construction of cleaned run instances."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.clean.paths import clean_manifest_path
from nro.engine.bids import BidsRun
from nro.engine.targets import smoothing_entity_value
from nro.func.planning import run_arguments
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import SubjectPlanningContext, instance_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def clean_direct_inputs(
    run: BidsRun, clean_config: Mapping[str, object]
) -> tuple[Path, ...]:
    if not clean_config.get("regress_out_task"):
        return ()
    events = run.path.parent / f"{run.stem}_events.tsv"
    return (events,) if events.is_file() else ()


def plan_instances(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[InstanceSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[InstanceSpec, ...]:
    """Construct requested space/smoothing instances for each selected run."""
    func_by_prefix = {instance.output_prefix: instance for instance in upstream["func"]}
    lineage = context.registered.lineages[descriptor.configuration_class]
    directory_label = context.registered.directories[descriptor.configuration_class]
    runtime_config = context.runtime_config(descriptor.configuration_class)
    output_root = (
        context.project_root
        / "derivatives"
        / "clean"
        / directory_label
        / context.sub_id
    )
    clean_values = context.workflow.configuration("clean").values
    result: list[InstanceSpec] = []
    for run in context.runs:
        for space, smoothing in context.target_pairs:
            entities = {**run.entities, "space": space, "smoothing": str(smoothing)}
            prefix = (
                f"{run.stem}_space-{space}_"
                f"smoothing-{smoothing_entity_value(smoothing)}"
            )
            result.append(
                InstanceSpec.create(
                    key=instance_key(
                        context.project,
                        descriptor.name,
                        context.registered.lineage_fingerprints[
                            descriptor.configuration_class
                        ],
                        context.participant,
                        entities,
                    ),
                    module=descriptor.name,
                    project=context.project,
                    participant=context.participant,
                    entities=entities,
                    scope=descriptor.scope,
                    configuration_lineage_id=lineage,
                    config_fingerprint=context.workflow.configuration(
                        descriptor.configuration_class
                    ).scientific_fingerprint,
                    directory_label=directory_label,
                    runtime_config=runtime_config,
                    command=(
                        sys.executable,
                        "-m",
                        "nro.clean",
                        "--participant",
                        context.participant,
                        "--project",
                        context.project,
                        "--space",
                        space,
                        "--smoothing",
                        str(smoothing),
                        *run_arguments(run),
                    ),
                    dependencies=(func_by_prefix[run.stem].key,),
                    input_paths=clean_direct_inputs(run, clean_values),
                    output_root=output_root,
                    output_prefix=prefix,
                    output_format=descriptor.output_format,
                    resource_class=descriptor.resource_class,
                    memory_gb=context.memory_gb,
                    max_memory_gb=context.max_memory_gb,
                    expected_outputs=(
                        clean_manifest_path(
                            context.sub_id,
                            run.stem,
                            project=context.project,
                            clean_id=directory_label,
                            space=space,
                            smoothing_mm=smoothing,
                            ses_id=f"ses-{run.session}" if run.session else None,
                        ),
                    ),
                    processing=descriptor.processing_contract(),
                )
            )
    return tuple(result)
