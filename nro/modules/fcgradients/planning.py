"""Planner-facing construction of functional-connectivity gradient work items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from nro.engine.paths import anatomical_manifest_path, module_derivatives_root
from nro.engine.targets import is_surface_space, smoothing_entity_value
from nro.modules.dynconn.contract import dynconn_output_paths
from nro.modules.fcgradients.contract import fcgradients_output_paths
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import SubjectPlanningContext

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct one gradient work item for each requested target pair."""

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
            context.target_work_item(
                descriptor,
                entities=entities,
                space=space,
                smoothing=smoothing,
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
                expected_outputs=tuple(
                    fcgradients_output_paths(output_root, prefix, domain)[name]
                    for name in ("manifest", "index")
                ),
            )
        )
    return tuple(result)
