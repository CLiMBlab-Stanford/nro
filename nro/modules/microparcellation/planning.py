"""Planner-facing construction of microparcellation work items."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Mapping

from nro.engine.paths import anatomical_manifest_path, module_derivatives_root
from nro.engine.targets import is_fsaverage_space, smoothing_entity_value
from nro.modules.microparcellation.contract import microparcellation_output_paths
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import SubjectPlanningContext, work_item_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct one participant work item for each requested target pair."""
    lineage = context.registered.lineages[descriptor.configuration_class]
    directory_label = context.registered.directories[descriptor.configuration_class]
    output_base = module_derivatives_root(
        "microparcellation",
        directory_label,
        project=context.project,
        bids_root=context.bids_root,
    )
    base_prefix = context.sub_id
    anat = upstream["anat"][0]
    anat_label = context.registered.directories["anat"]
    result: list[WorkItemSpec] = []
    for space, smoothing in context.target_pairs:
        entities = {"space": space, "smoothing": str(smoothing)}
        output_root = output_base / context.sub_id
        prefix = f"{base_prefix}_space-{space}_smoothing-{smoothing_entity_value(smoothing)}"
        clean_dependencies = tuple(
            work_item.key
            for work_item in upstream["clean"]
            if work_item.entities.get("space") == space
            and work_item.entities.get("smoothing") == str(smoothing)
        )
        needs_anatomy = not space.startswith("MNI") and not is_fsaverage_space(space)
        dependencies = (*clean_dependencies, *((anat.key,) if needs_anatomy else ()))
        direct_inputs = list(context.aggregate_source_inputs)
        if needs_anatomy:
            direct_inputs.append(
                anatomical_manifest_path(
                    context.sub_id,
                    project=context.project,
                    anat_id=anat_label,
                    bids_root=context.bids_root,
                )
            )
        result.append(
            WorkItemSpec.create(
                key=work_item_key(
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
                module_lineage_id=lineage,
                config_fingerprint=context.workflow.configuration(
                    descriptor.configuration_class
                ).scientific_fingerprint,
                directory_label=directory_label,
                runtime_config=context.runtime_config(descriptor.configuration_class),
                command=(
                    sys.executable,
                    "-m",
                    descriptor.execution_module,
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
                input_paths=tuple(direct_inputs),
                output_root=output_root,
                output_prefix=prefix,
                output_format=descriptor.output_format,
                resource_class=descriptor.resource_class,
                memory_gb=context.memory_gb,
                max_memory_gb=context.max_memory_gb,
                expected_outputs=tuple(
                    microparcellation_output_paths(output_root, prefix)[name]
                    for name in ("manifest", "quality", "index")
                ),
                processing=context.processing_contract(descriptor),
            )
        )
    return tuple(result)
