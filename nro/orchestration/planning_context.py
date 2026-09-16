"""Shared immutable inputs supplied to module-local work-item planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.configuration.markup import SubjectMarkup
from nro.configuration.store import ResolvedWorkflow, fingerprint
from nro.engine.bids import BidsRun
from nro.orchestration.workflow_registry import RegisteredWorkflow, WorkflowRegistry


class ParticipantUnavailableError(RuntimeError):
    """The selected participant cannot support any work item of the target."""


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
        # Retain the historical field label so a vocabulary-only change does
        # not alter stable work-item keys.
        "module_lineage": module_lineage_fingerprint,
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
    task_models: Mapping[str, dict] | None = None
    source_markup: SubjectMarkup | None = None

    def runtime_config(self, configuration_class: str) -> Path:
        """Return the registered runtime path for one configuration class."""
        return self.registry.runtime_config_path(self.registered, configuration_class)

    def processing_contract(self, descriptor, **values) -> dict:
        """Combine module policy with this subject's captured source markup."""
        processing = dict(descriptor.processing_contract())
        if self.source_markup is not None:
            processing["source_markup"] = self.source_markup.as_dict()
        processing.update(values)
        return processing
