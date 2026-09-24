"""Planner-facing construction of anatomical work items."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.configuration.hardware import gradient_unwarping_records
from nro.configuration.markup import SubjectMarkup
from nro.engine.image_paths import image_source_paths
from nro.engine.paths import anat_subject_dir, anatomical_manifest_path
from nro.modules.anat.contract import anatomical_output_contract
from nro.modules.anat.lesion_policy import lesion_reconstruction_contract
from nro.modules.anat.policy import surface_reconstruction_contract
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import (
    ParticipantUnavailableError,
    SubjectPlanningContext,
    work_item_key,
)

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def _canonical_images(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    """Resolve BIDS aliases and retain each physical image exactly once."""
    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.resolve()] = None
    return tuple(unique)


def raw_anatomical_images(
    subject_dir: Path, markup: SubjectMarkup | None = None
) -> tuple[Path, ...]:
    """Return selected T1w and T2w images before adding metadata sources."""
    automatic = []
    for pattern in (
        "anat/*_T1w.nii*",
        "anat/*_T2w.nii*",
        "ses-*/anat/*_T1w.nii*",
        "ses-*/anat/*_T2w.nii*",
    ):
        automatic.extend(sorted(subject_dir.glob(pattern)))
    if markup is None:
        return _canonical_images(automatic)
    automatic = list(markup.filter(automatic))
    by_modality = {
        "T1w": [path for path in automatic if path.name.endswith(("_T1w.nii", "_T1w.nii.gz"))],
        "T2w": [path for path in automatic if path.name.endswith(("_T2w.nii", "_T2w.nii.gz"))],
    }
    selected = []
    for modality, marked in (("T1w", markup.t1w), ("T2w", markup.t2w)):
        paths = markup.filter(marked) if marked else tuple(by_modality[modality])
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Marked {modality} image does not exist: {path}")
            if not path.name.endswith((f"_{modality}.nii", f"_{modality}.nii.gz")):
                raise ValueError(f"Marked {modality} path has the wrong BIDS suffix: {path}")
        selected.extend(paths)
    return _canonical_images(selected)


def raw_anatomical_inputs(
    subject_dir: Path, markup: SubjectMarkup | None = None
) -> tuple[Path, ...]:
    """Return anatomical images and applicable metadata sources for planning."""
    result: list[Path] = []
    for path in raw_anatomical_images(subject_dir, markup):
        result.extend(image_source_paths(path, markup=markup))
    return tuple(dict.fromkeys(result))


def _effective_markup_contract(subject_dir: Path, markup: SubjectMarkup) -> dict[str, object]:
    """Capture only exclusions that can alter automatic anatomical selection."""
    candidates = []
    for pattern in (
        "anat/*_T1w.nii*",
        "anat/*_T2w.nii*",
        "ses-*/anat/*_T1w.nii*",
        "ses-*/anat/*_T2w.nii*",
    ):
        candidates.extend(subject_dir.glob(pattern))
    automatic = [
        path
        for path in candidates
        if (path.name.endswith(("_T1w.nii", "_T1w.nii.gz")) and not markup.t1w)
        or (path.name.endswith(("_T2w.nii", "_T2w.nii.gz")) and not markup.t2w)
    ]
    value = markup.as_dict()
    value["exclude"] = [
        str(root)
        for root in markup.excluded
        if any(path == root or path.is_relative_to(root) for path in automatic)
    ]
    return value


def plan_work_items(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[WorkItemSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[WorkItemSpec, ...]:
    """Construct the one subject-level anatomical work item."""
    del upstream
    lineage = context.registered.lineages[descriptor.name]
    directory_label = context.registered.directories[descriptor.name]
    inputs = raw_anatomical_inputs(context.subject_dir, context.source_markup)
    if not inputs:
        raise ParticipantUnavailableError(f"No T1w or T2w images found under {context.subject_dir}")
    entities: dict[str, str] = {}
    config = context.workflow.configuration(descriptor.configuration_class).values
    surface_engine = str(config["surface_reconstruction_engine"])
    anatomical_images = raw_anatomical_images(context.subject_dir, context.source_markup)
    if (
        surface_engine == "fastsurfer"
        and not context.source_markup.lesion
        and not any(path.name.endswith(("_T1w.nii", "_T1w.nii.gz")) for path in anatomical_images)
    ):
        raise ParticipantUnavailableError("FastSurfer surface reconstruction requires T1w data")
    # Resource-specific runner steps are dispatched independently. The parent
    # anatomical work item always returns to the ordinary CPU pool.
    resource_class = descriptor.resource_class
    gradient_records, _ = gradient_unwarping_records(
        list(anatomical_images),
        mode=str(config["gradient_unwarping"]),
        markup=context.source_markup,
        definitions=context.definitions_roots,
        coefficient_root=context.gradient_coefficients_root,
    )
    processing_values: dict[str, object] = {
        "gradient_unwarping": gradient_records,
        "output_metadata": anatomical_output_contract(lesion=context.source_markup.lesion),
        "source_markup": _effective_markup_contract(context.subject_dir, context.source_markup),
    }
    if context.source_markup.lesion:
        processing_values["lesion_reconstruction"] = lesion_reconstruction_contract()
    processing_values["surface_reconstruction"] = surface_reconstruction_contract(surface_engine)
    return (
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
            ).module_fingerprint(descriptor.name),
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
            ),
            dependencies=(),
            input_paths=inputs,
            output_root=anat_subject_dir(
                context.sub_id,
                project=context.project,
                anat_id=directory_label,
                bids_root=context.bids_root,
            ),
            output_prefix=context.sub_id,
            output_format=descriptor.output_format,
            resource_class=resource_class,
            memory_gb=context.memory_gb,
            max_memory_gb=context.max_memory_gb,
            expected_outputs=(
                anatomical_manifest_path(
                    context.sub_id,
                    project=context.project,
                    anat_id=directory_label,
                    bids_root=context.bids_root,
                ),
            ),
            processing=context.processing_contract(descriptor, **processing_values),
        ),
    )
