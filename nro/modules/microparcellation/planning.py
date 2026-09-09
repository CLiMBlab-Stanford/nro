"""Planner-facing construction of microparcellation instances."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.engine.targets import smoothing_entity_value, target_directory_name
from nro.modules.microparcellation.paths import output_paths
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import SubjectPlanningContext, instance_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def plan_instances(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[InstanceSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[InstanceSpec, ...]:
    """Construct one participant instance for each requested target pair."""
    lineage = context.registered.lineages[descriptor.configuration_class]
    directory_label = context.registered.directories[descriptor.configuration_class]
    values = context.workflow.configuration("microparcellation").values
    output_base = (
        Path(
            values.get("output_dir")
            or context.project_root / "derivatives" / "microparcellation" / directory_label
        )
        .expanduser()
        .resolve()
    )
    base_prefix = str(values.get("prefix") or context.sub_id)
    result: list[InstanceSpec] = []
    for space, smoothing in context.target_pairs:
        entities = {"space": space, "smoothing": str(smoothing)}
        output_root = output_base / target_directory_name(space, smoothing) / context.sub_id
        prefix = f"{base_prefix}_space-{space}_smoothing-{smoothing_entity_value(smoothing)}"
        dependencies = tuple(
            instance.key
            for instance in upstream["clean"]
            if instance.entities.get("space") == space
            and instance.entities.get("smoothing") == str(smoothing)
        )
        result.append(
            InstanceSpec.create(
                key=instance_key(
                    context.project,
                    descriptor.name,
                    context.registered.lineage_fingerprints[descriptor.configuration_class],
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
                runtime_config=context.runtime_config(descriptor.configuration_class),
                command=(
                    sys.executable,
                    "-m",
                    "nro.modules.microparcellation",
                    "--participant",
                    context.participant,
                    "--project",
                    context.project,
                    "--space",
                    space,
                    "--smoothing",
                    str(smoothing),
                ),
                dependencies=dependencies,
                input_paths=context.aggregate_source_inputs,
                output_root=output_root,
                output_prefix=prefix,
                output_format=descriptor.output_format,
                resource_class=descriptor.resource_class,
                memory_gb=context.memory_gb,
                max_memory_gb=context.max_memory_gb,
                expected_outputs=tuple(
                    output_paths(output_root, prefix)[name]
                    for name in ("manifest", "quality", "index")
                ),
                processing=descriptor.processing_contract(),
            )
        )
    return tuple(result)
