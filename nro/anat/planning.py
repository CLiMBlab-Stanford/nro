"""Planner-facing construction of anatomical instances."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from nro.engine.images import image_source_paths
from nro.engine.paths import anatomical_manifest_path
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import (
    ParticipantUnavailableError,
    SubjectPlanningContext,
    instance_key,
)

if TYPE_CHECKING:
    from nro.orchestration.catalog import ModuleDescriptor


def raw_anatomical_inputs(subject_dir: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    patterns = (
        "anat/*_T1w.nii*",
        "anat/*_T2w.nii*",
        "ses-*/anat/*_T1w.nii*",
        "ses-*/anat/*_T2w.nii*",
    )
    for pattern in patterns:
        for path in sorted(subject_dir.glob(pattern)):
            result.extend(image_source_paths(path))
    return tuple(dict.fromkeys(result))


def plan_instances(
    context: SubjectPlanningContext,
    upstream: Mapping[str, tuple[InstanceSpec, ...]],
    descriptor: ModuleDescriptor,
) -> tuple[InstanceSpec, ...]:
    """Construct the one subject-level anatomical instance."""
    del upstream
    lineage = context.registered.lineages[descriptor.configuration_class]
    directory_label = context.registered.directories[descriptor.configuration_class]
    inputs = raw_anatomical_inputs(context.subject_dir)
    if not inputs:
        raise ParticipantUnavailableError(
            f"No T1w or T2w images found under {context.subject_dir}"
        )
    entities: dict[str, str] = {}
    return (
        InstanceSpec.create(
            key=instance_key(
                context.project,
                descriptor.name,
                lineage,
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
            ).fingerprint,
            directory_label=directory_label,
            runtime_config=context.runtime_config(descriptor.configuration_class),
            command=(
                sys.executable,
                "-m",
                "nro.anat",
                "--participant",
                context.participant,
                "--project",
                context.project,
            ),
            dependencies=(),
            input_paths=inputs,
            output_root=(
                context.project_root
                / "derivatives"
                / "preprocessing"
                / directory_label
                / context.sub_id
                / "anat"
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
                    preprocessing_id=directory_label,
                ),
            ),
        ),
    )
