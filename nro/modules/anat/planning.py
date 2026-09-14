"""Planner-facing construction of anatomical instances."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.configuration.markup import SubjectMarkup
from nro.engine.images import image_source_paths
from nro.engine.paths import anat_subject_dir, anatomical_manifest_path
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import (
    ParticipantUnavailableError,
    SubjectPlanningContext,
    instance_key,
)

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


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
        return tuple(dict.fromkeys(automatic))
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
    return tuple(dict.fromkeys(selected))


def raw_anatomical_inputs(
    subject_dir: Path, markup: SubjectMarkup | None = None
) -> tuple[Path, ...]:
    """Return anatomical images and applicable metadata sources for planning."""
    result: list[Path] = []
    for path in raw_anatomical_images(subject_dir, markup):
        result.extend(image_source_paths(path, markup=markup))
    return tuple(dict.fromkeys(result))


def plan_instances(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[InstanceSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[InstanceSpec, ...]:
    """Construct the one subject-level anatomical instance."""
    del upstream
    lineage = context.registered.lineages[descriptor.name]
    directory_label = context.registered.directories[descriptor.name]
    inputs = raw_anatomical_inputs(context.subject_dir, context.source_markup)
    if not inputs:
        raise ParticipantUnavailableError(f"No T1w or T2w images found under {context.subject_dir}")
    entities: dict[str, str] = {}
    return (
        InstanceSpec.create(
            key=instance_key(
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
            configuration_lineage_id=lineage,
            config_fingerprint=context.workflow.configuration(
                descriptor.configuration_class
            ).module_fingerprint(descriptor.name),
            directory_label=directory_label,
            runtime_config=context.runtime_config(descriptor.configuration_class),
            command=(
                sys.executable,
                "-m",
                "nro.modules.anat",
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
            resource_class=descriptor.resource_class,
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
            processing=context.processing_contract(descriptor),
        ),
    )
