"""Shared immutable inputs supplied to module-local work-item planners."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.definitions.markup import SubjectMarkup
from nro.definitions.store import ResolvedWorkflow, fingerprint
from nro.engine.bids import BidsRun
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.resources import memory_tiers
from nro.orchestration.workflow_registry import RegisteredWorkflow, WorkflowRegistry


class ParticipantUnavailableError(RuntimeError):
    """The selected participant cannot support any work item of the target."""


def planning_memory_tier(required_gb: int, initial_gb: int, maximum_gb: int) -> int:
    """Return the smallest schedulable memory tier meeting a planning estimate."""
    for tier in memory_tiers(initial_gb, maximum_gb):
        if tier >= required_gb:
            return tier
    return maximum_gb


def work_item_key(
    project: str,
    module: str,
    module_lineage_fingerprint: str,
    participant: str,
    entities: Mapping[str, str],
) -> str:
    """Return the stable identity key for one logical work item."""
    identity = {
        "project": project,
        "module": module,
        # This serialized label predates the public term ``module lineage``.
        # Keep it stable because vocabulary-only changes must not alter
        # scientific work-item identities.
        "configuration_lineage": module_lineage_fingerprint,
        "participant": participant,
        "entities": dict(sorted(entities.items())),
    }
    return f"{module}:{fingerprint(identity)}"


@dataclass(frozen=True)
class SubjectPlanningContext:
    """Resolved inputs shared by module-local planners for one participant."""

    project: str
    participant: str
    sub_id: str
    bids_root: Path
    project_root: Path
    subject_dir: Path
    workflow: ResolvedWorkflow
    registered: RegisteredWorkflow
    registry: WorkflowRegistry
    runs: tuple[BidsRun, ...]
    aggregate_source_inputs: tuple[Path, ...]
    target_pairs: tuple[tuple[str, int], ...]
    memory_gb: int
    max_memory_gb: int
    definitions_roots: tuple[Path, ...]
    gradient_coefficients_root: Path
    analysis_models: Mapping[str, dict] | None = None
    source_markup: SubjectMarkup | None = None

    def runtime_config(self, configuration_class: str) -> Path:
        """Return the registered runtime path for one configuration class."""
        return self.registry.runtime_config_path(self.registered, configuration_class)

    def target_work_item(
        self,
        descriptor,
        *,
        entities: Mapping[str, str],
        space: str,
        smoothing: int,
        dependencies: tuple[str, ...],
        input_paths: tuple[Path, ...],
        output_root: Path,
        output_prefix: str,
        expected_outputs: tuple[Path, ...],
        extra_arguments: tuple[str, ...] = (),
        memory_gb: int | None = None,
        processing_values: Mapping[str, object] | None = None,
    ) -> WorkItemSpec:
        """Construct one participant work item for a space and smoothing target.

        The key, lineage, configuration fingerprint, runtime configuration,
        resources, and processing contract come from this context and the
        module descriptor. The command runs the module's execution entry point
        with the participant, project, space, smoothing, and `extra_arguments`.
        `memory_gb` defaults to the context's memory tier. `processing_values`
        adds work-item-specific scientific values to the processing contract.
        """
        configuration_class = descriptor.configuration_class
        return WorkItemSpec.create(
            key=work_item_key(
                self.project,
                descriptor.name,
                self.registered.lineage_fingerprints[configuration_class],
                self.participant,
                entities,
            ),
            module=descriptor.name,
            project=self.project,
            participant=self.participant,
            entities=entities,
            scope=descriptor.scope,
            module_lineage_id=self.registered.lineages[configuration_class],
            config_fingerprint=self.workflow.configuration(
                configuration_class
            ).scientific_fingerprint,
            directory_label=self.registered.directories[configuration_class],
            runtime_config=self.runtime_config(configuration_class),
            command=(
                sys.executable,
                "-m",
                descriptor.execution_module,
                "--participant",
                self.participant,
                "--project",
                self.project,
                "--space",
                space,
                "--smoothing",
                str(smoothing),
                *extra_arguments,
            ),
            dependencies=dependencies,
            input_paths=input_paths,
            output_root=output_root,
            output_prefix=output_prefix,
            output_format=descriptor.output_format,
            resource_class=descriptor.resource_class,
            memory_gb=self.memory_gb if memory_gb is None else memory_gb,
            max_memory_gb=self.max_memory_gb,
            expected_outputs=expected_outputs,
            processing=self.processing_contract(descriptor, **(processing_values or {})),
        )

    def processing_contract(self, descriptor, **values) -> dict:
        """Combine module policy with work-item-specific scientific values.

        Source selection is represented by the contract's declared inputs.  A
        whole markup document is neither a scientific input nor stable under
        unrelated edits.  The anatomical planner adds only the effective
        selections that are themselves part of its processing policy.
        """
        processing = dict(descriptor.processing_contract())
        processing.update(values)
        return processing
