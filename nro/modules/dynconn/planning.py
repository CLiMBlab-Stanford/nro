"""Planner-facing construction of dynamic-connectivity work items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from nro.engine.paths import module_derivatives_root
from nro.engine.targets import is_surface_space, smoothing_entity_value
from nro.modules.dynconn.paths import output_paths
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import SubjectPlanningContext

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct one participant work item for each requested target pair."""

    directory_label = context.registered.directories[descriptor.configuration_class]
    output_base = module_derivatives_root(
        "dynconn",
        directory_label,
        project=context.project,
        bids_root=context.bids_root,
    )
    base_prefix = context.sub_id
    result = []
    for space, smoothing in context.target_pairs:
        entities = {"space": space, "smoothing": str(smoothing)}
        output_root = output_base / context.sub_id
        prefix = f"{base_prefix}_space-{space}_smoothing-{smoothing_entity_value(smoothing)}"
        clean = tuple(
            item.key
            for item in upstream["clean"]
            if item.entities.get("space") == space
            and item.entities.get("smoothing") == str(smoothing)
        )
        domain = "surface" if is_surface_space(space) else "volume"
        result.append(
            context.target_work_item(
                descriptor,
                entities=entities,
                space=space,
                smoothing=smoothing,
                dependencies=clean,
                input_paths=context.aggregate_source_inputs,
                output_root=output_root,
                output_prefix=prefix,
                expected_outputs=tuple(
                    output_paths(output_root, prefix, domain)[name]
                    for name in ("manifest", "index")
                ),
            )
        )
    return tuple(result)
