"""Planner-facing construction of cleaned run work items."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.engine.bids import BidsRun, run_arguments
from nro.engine.paths import clean_manifest_path, module_subject_dir
from nro.engine.targets import smoothing_entity_value
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import SubjectPlanningContext

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def clean_direct_inputs(
    run: BidsRun, clean_config: Mapping[str, object], *, markup=None
) -> tuple[Path, ...]:
    """Return task-event inputs that participate in a cleaning contract."""
    if not clean_config.get("regress_out_task"):
        return ()
    events = run.path.parent / f"{run.stem}_events.tsv"
    return (
        (events,) if events.is_file() and (markup is None or not markup.is_excluded(events)) else ()
    )


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct requested space/smoothing work items for each selected run."""
    func_by_prefix = {work_item.output_prefix: work_item for work_item in upstream["func"]}
    anat = upstream["anat"][0]
    directory_label = context.registered.directories[descriptor.configuration_class]
    output_root = module_subject_dir(
        context.sub_id,
        module="clean",
        module_id=directory_label,
        project=context.project,
        bids_root=context.bids_root,
    )
    clean_values = context.workflow.configuration("clean").values
    result: list[WorkItemSpec] = []
    for run in context.runs:
        for space, smoothing in context.target_pairs:
            entities = {**run.entities, "space": space, "smoothing": str(smoothing)}
            prefix = f"{run.stem}_space-{space}_smoothing-{smoothing_entity_value(smoothing)}"
            result.append(
                context.target_work_item(
                    descriptor,
                    entities=entities,
                    space=space,
                    smoothing=smoothing,
                    dependencies=(func_by_prefix[run.stem].key, anat.key),
                    input_paths=clean_direct_inputs(
                        run, clean_values, markup=context.source_markup
                    ),
                    output_root=output_root,
                    output_prefix=prefix,
                    expected_outputs=(
                        clean_manifest_path(
                            context.sub_id,
                            run.stem,
                            project=context.project,
                            clean_id=directory_label,
                            bids_root=context.bids_root,
                            space=space,
                            smoothing_mm=smoothing,
                            ses_id=f"ses-{run.session}" if run.session else None,
                        ),
                    ),
                    extra_arguments=(*run_arguments(run),),
                )
            )
    return tuple(result)
