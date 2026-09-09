"""Shared immutable inputs supplied to module-local instance planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.configuration.store import ResolvedWorkflow, fingerprint
from nro.engine.bids import BidsRun
from nro.orchestration.workflow_registry import RegisteredWorkflow, WorkflowRegistry


class ParticipantUnavailableError(RuntimeError):
    """The selected participant cannot support any instance of the target."""


def instance_key(
    project: str,
    module: str,
    configuration_lineage_fingerprint: str,
    participant: str,
    entities: Mapping[str, str],
) -> str:
    """Return the stable identity key for one logical instance."""
    identity = {
        "project": project,
        "module": module,
        "configuration_lineage": configuration_lineage_fingerprint,
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

    def runtime_config(self, configuration_class: str) -> Path:
        """Return the registered runtime configuration path for a derivative class."""
        return self.registry.runtime_config_path(self.registered, configuration_class)
