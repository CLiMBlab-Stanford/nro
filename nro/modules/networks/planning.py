"""Planner-facing construction of individualized-network instances."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.engine.paths import anatomical_manifest_path
from nro.engine.targets import smoothing_entity_value
from nro.modules.networks.labeling import reference_paths
from nro.modules.networks.paths import fixed_output_paths
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import SubjectPlanningContext, instance_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def plan_instances(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[InstanceSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[InstanceSpec, ...]:
    """Construct one network instance for each requested target pair."""
    lineage = context.registered.lineages[descriptor.configuration_class]
    directory_label = context.registered.directories[descriptor.configuration_class]
    values = context.workflow.configuration("networks").values
    output_base = (
        Path(
            values.get("output_dir")
            or context.project_root / "derivatives" / "networks" / directory_label
        )
        .expanduser()
        .resolve()
    )
    base_prefix = str(values.get("prefix") or context.sub_id)
    labeling_enabled = bool((values.get("labeling") or {}).get("enabled", True))
    anat = upstream["anat"][0]
    preprocessing_label = context.registered.directories["preprocessing"]
    result: list[InstanceSpec] = []
    for space, smoothing in context.target_pairs:
        entities = {"space": space, "smoothing": str(smoothing)}
        output_root = output_base / context.sub_id
        prefix = f"{base_prefix}_space-{space}_smoothing-{smoothing_entity_value(smoothing)}"
        micro = next(
            instance for instance in upstream["microparcellation"] if instance.entities == entities
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
                    "nro.modules.networks",
                    "--participant",
                    context.participant,
                    "--project",
                    context.project,
                    "--space",
                    space,
                    "--smoothing",
                    str(smoothing),
                ),
                dependencies=(micro.key, anat.key),
                input_paths=(
                    *context.aggregate_source_inputs,
                    anatomical_manifest_path(
                        context.sub_id,
                        project=context.project,
                        preprocessing_id=preprocessing_label,
                        bids_root=context.bids_root,
                    ),
                    *(reference_paths() if labeling_enabled else ()),
                ),
                output_root=output_root,
                output_prefix=prefix,
                output_format=descriptor.output_format,
                resource_class=descriptor.resource_class,
                memory_gb=context.memory_gb,
                max_memory_gb=context.max_memory_gb,
                expected_outputs=tuple(
                    fixed_output_paths(output_root, prefix)[name] for name in ("manifest", "index")
                ),
                processing=descriptor.processing_contract(),
            )
        )
    return tuple(result)
