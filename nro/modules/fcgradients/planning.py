"""Planner-facing construction of functional-connectivity gradient work items."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Mapping

from nro.engine.paths import anatomical_manifest_path, module_derivatives_root
from nro.engine.targets import is_surface_space, smoothing_entity_value
from nro.modules.dynconn.contract import dynconn_output_paths
from nro.modules.fcgradients.contract import fcgradients_output_paths
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import SubjectPlanningContext, work_item_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct one gradient work item for each requested target pair."""

    lineage = context.registered.lineages[descriptor.configuration_class]
    directory_label = context.registered.directories[descriptor.configuration_class]
    output_base = module_derivatives_root(
        "fcgradients",
        directory_label,
        project=context.project,
        bids_root=context.bids_root,
    )
    anat = upstream["anat"][0]
    anat_manifest = anatomical_manifest_path(
        context.sub_id,
        project=context.project,
        anat_id=context.registered.directories["anat"],
        bids_root=context.bids_root,
    )
    result: list[WorkItemSpec] = []
    for space, smoothing in context.target_pairs:
        entities = {"space": space, "smoothing": str(smoothing)}
        domain = "surface" if is_surface_space(space) else "volume"
        output_root = output_base / context.sub_id
        prefix = f"{context.sub_id}_space-{space}_smoothing-{smoothing_entity_value(smoothing)}"
        source = next(item for item in upstream["dynconn"] if item.entities == entities)
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
                dependencies=(source.key, anat.key),
                input_paths=(
                    *context.aggregate_source_inputs,
                    dynconn_output_paths(source.output_root, source.output_prefix, domain)[
                        "manifest"
                    ],
                    anat_manifest,
                ),
                output_root=output_root,
                output_prefix=prefix,
                output_format=descriptor.output_format,
                resource_class=descriptor.resource_class,
                memory_gb=context.memory_gb,
                max_memory_gb=context.max_memory_gb,
                expected_outputs=tuple(
                    fcgradients_output_paths(output_root, prefix, domain)[name]
                    for name in ("manifest", "index")
                ),
                processing=context.processing_contract(descriptor),
            )
        )
    return tuple(result)
