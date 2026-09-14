"""Planner-facing construction of run-level preprocessing instances."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.engine.bids import BidsRun, run_arguments
from nro.engine.paths import functional_manifest_path
from nro.modules.func.resolver import (
    ImageRec,
    load_rec,
    resolve_func_references,
)
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import SubjectPlanningContext, instance_key

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


@dataclass(frozen=True)
class _SessionInventory:
    sbrefs: tuple[ImageRec, ...]
    sidecarless_sbrefs: tuple[Path, ...]
    fmaps: tuple[ImageRec, ...]


def load_session_inventory(run: BidsRun, *, include_fmaps: bool) -> _SessionInventory:
    """Index session inputs once rather than once per BOLD run."""
    session_root = run.path.parent.parent
    func_dir = session_root / "func"
    fmap_dir = session_root / "fmap"
    prefix = f"sub-{run.participant}" + (f"_ses-{run.session}" if run.session else "")
    sbrefs = []
    sidecarless = []
    for path in sorted(func_dir.glob(f"{prefix}_*_sbref.nii*")):
        try:
            sbrefs.append(load_rec(path))
        except FileNotFoundError:
            sidecarless.append(path)
    fmaps = []
    if include_fmaps:
        try:
            fmaps = [load_rec(path) for path in sorted(fmap_dir.glob(f"{prefix}_*_epi.nii*"))]
        except Exception:
            fmaps = []
    return _SessionInventory(tuple(sbrefs), tuple(sidecarless), tuple(fmaps))


def resolved_func_inputs(
    run: BidsRun,
    *,
    sdc_from_sbref_pair: bool,
    session_inventory: _SessionInventory,
) -> tuple[Path, ...]:
    """Mirror functional reference and fieldmap selection for exact inputs."""
    bold = load_rec(run.path)
    result: list[Path] = [bold.img, *bold.metadata_sources]
    fmaps = session_inventory.fmaps
    if bold.metadata.get("NROReferencePolicy") == "explicit" and sdc_from_sbref_pair:
        session_root = run.path.parent.parent
        prefix = f"sub-{run.participant}" + (f"_ses-{run.session}" if run.session else "")
        fmaps = tuple(
            load_rec(path) for path in sorted((session_root / "fmap").glob(f"{prefix}_*_epi.nii*"))
        )
    references = resolve_func_references(
        bold=bold,
        sbrefs=session_inventory.sbrefs,
        sidecarless_sbrefs=session_inventory.sidecarless_sbrefs,
        fmaps=fmaps,
        sdc_from_sbref_pair=sdc_from_sbref_pair,
    )
    if references.sbref is not None:
        result.append(references.sbref.img)
        result.extend(references.sbref.metadata_sources)
    if references.pair is not None:
        for record in (references.pair.se1, references.pair.se2):
            result.append(record.img)
            result.extend(record.metadata_sources)
    return tuple(dict.fromkeys(result))


def plan_instances(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[InstanceSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[InstanceSpec, ...]:
    """Construct one preprocessing instance per selected raw BOLD run."""
    anat = upstream["anat"][0]
    lineage = context.registered.lineages["preprocessing"]
    directory_label = context.registered.directories["preprocessing"]
    runtime_config = context.runtime_config(descriptor.configuration_class)
    output_root = (
        context.project_root / "derivatives" / "preprocessing" / directory_label / context.sub_id
    )
    sdc_from_sbref_pair = bool(context.workflow.configuration("func").values["sdc_from_sbref_pair"])
    inventories: dict[tuple[Path, bool], _SessionInventory] = {}
    result: list[InstanceSpec] = []
    for run in context.runs:
        inventory_key = (run.path.parent.parent.resolve(), not sdc_from_sbref_pair)
        if inventory_key not in inventories:
            inventories[inventory_key] = load_session_inventory(
                run, include_fmaps=not sdc_from_sbref_pair
            )
        entities = dict(run.entities)
        result.append(
            InstanceSpec.create(
                key=instance_key(
                    context.project,
                    descriptor.name,
                    context.registered.lineage_fingerprints["preprocessing"],
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
                    "nro.modules.func",
                    "--participant",
                    context.participant,
                    "--project",
                    context.project,
                    *run_arguments(run),
                ),
                dependencies=(anat.key,),
                input_paths=resolved_func_inputs(
                    run,
                    sdc_from_sbref_pair=sdc_from_sbref_pair,
                    session_inventory=inventories[inventory_key],
                ),
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
                        preprocessing_id=directory_label,
                        bids_root=context.bids_root,
                        ses_id=f"ses-{run.session}" if run.session else None,
                    ),
                ),
                processing=descriptor.processing_contract(),
            )
        )
    return tuple(result)
