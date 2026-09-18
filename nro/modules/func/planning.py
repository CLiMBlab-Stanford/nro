"""Planner-facing construction of run-level functional work items."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.configuration.hardware import gradient_unwarping_records
from nro.engine.bids import BidsRun, run_arguments
from nro.engine.paths import functional_manifest_path, module_subject_dir
from nro.modules.func.contract import final_resampling_contract
from nro.modules.func.resolver import (
    ReferenceInventory,
    load_rec,
    load_reference_inventory,
    resolve_func_references,
)
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import SubjectPlanningContext, work_item_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def load_session_inventory(run: BidsRun, *, markup=None) -> ReferenceInventory:
    """Index one session's optional functional references."""
    session_root = run.path.parent.parent
    prefix = f"sub-{run.participant}" + (f"_ses-{run.session}" if run.session else "")
    return load_reference_inventory(
        session_root / "func", session_root / "fmap", prefix, markup=markup
    )


def resolved_func_inputs(
    run: BidsRun,
    *,
    sdc_from_sbref_pair: bool,
    session_inventory: ReferenceInventory,
    markup=None,
) -> tuple[Path, ...]:
    """Mirror functional reference and fieldmap selection for exact inputs."""
    bold = load_rec(run.path, markup=markup)
    result: list[Path] = [bold.img, *bold.metadata_sources]
    explicit_references = bold.metadata.get("NROReferencePolicy") == "explicit"
    use_fmaps = explicit_references or not sdc_from_sbref_pair
    references = resolve_func_references(
        bold=bold,
        sbrefs=session_inventory.sbrefs,
        sidecarless_sbrefs=session_inventory.sidecarless_sbrefs,
        fmaps=session_inventory.fmaps if use_fmaps else (),
        sdc_from_sbref_pair=sdc_from_sbref_pair,
        selection_warning=session_inventory.fieldmap_warning if use_fmaps else None,
        unusable_sbrefs=session_inventory.unusable_sbrefs,
    )
    if references.sbref is not None:
        result.append(references.sbref.img)
        result.extend(references.sbref.metadata_sources)
    if references.pair is not None:
        for record in (references.pair.se1, references.pair.se2):
            result.append(record.img)
            result.extend(record.metadata_sources)
    return tuple(dict.fromkeys(result))


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct one functional work item per selected raw BOLD run."""
    anat = upstream["anat"][0]
    lineage = context.registered.lineages[descriptor.name]
    directory_label = context.registered.directories[descriptor.name]
    runtime_config = context.runtime_config(descriptor.configuration_class)
    output_root = module_subject_dir(
        context.sub_id,
        module="func",
        module_id=directory_label,
        project=context.project,
        bids_root=context.bids_root,
    )
    sdc_from_sbref_pair = bool(context.workflow.configuration("func").values["sdc_from_sbref_pair"])
    gradient_mode = str(context.workflow.configuration("func").values["gradient_unwarping"])
    inventories: dict[Path, ReferenceInventory] = {}
    result: list[WorkItemSpec] = []
    for run in context.runs:
        inventory_key = run.path.parent.parent.resolve()
        if inventory_key not in inventories:
            inventories[inventory_key] = load_session_inventory(run, markup=context.source_markup)
        entities = dict(run.entities)
        direct_inputs = resolved_func_inputs(
            run,
            sdc_from_sbref_pair=sdc_from_sbref_pair,
            session_inventory=inventories[inventory_key],
            markup=context.source_markup,
        )
        images = [path for path in direct_inputs if path.name.endswith((".nii", ".nii.gz"))]
        gradient_records, gradient_resolutions = gradient_unwarping_records(
            images,
            mode=gradient_mode,
            markup=context.source_markup,
            definitions=context.definitions_root,
            coefficient_root=context.gradient_coefficients_root,
        )
        bold_resolution = gradient_resolutions.get(run.path.expanduser().absolute())
        bold_gradient_applied = bool(bold_resolution and bold_resolution.applied)
        result.append(
            WorkItemSpec.create(
                key=work_item_key(
                    context.project,
                    descriptor.name,
                    context.registered.lineage_fingerprints[descriptor.name],
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
                runtime_config=runtime_config,
                command=(
                    sys.executable,
                    "-m",
                    descriptor.execution_module,
                    "--participant",
                    context.participant,
                    "--project",
                    context.project,
                    *run_arguments(run),
                ),
                dependencies=(anat.key,),
                input_paths=direct_inputs,
                output_root=output_root,
                output_prefix=run.stem,
                output_format=descriptor.output_format,
                resource_class=descriptor.resource_class,
                memory_gb=context.memory_gb,
                max_memory_gb=context.max_memory_gb,
                expected_outputs=(
                    functional_manifest_path(
                        context.sub_id,
                        run.stem,
                        project=context.project,
                        func_id=directory_label,
                        bids_root=context.bids_root,
                        ses_id=f"ses-{run.session}" if run.session else None,
                    ),
                ),
                processing={
                    **context.processing_contract(
                        descriptor,
                        gradient_unwarping=gradient_records,
                    ),
                    "final_resampling": final_resampling_contract(
                        gradient_unwarping=bold_gradient_applied
                    ),
                },
            )
        )
    return tuple(result)
