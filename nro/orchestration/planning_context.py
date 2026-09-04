"""Shared immutable inputs supplied to module-local instance planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.configuration.store import ResolvedWorkflow, fingerprint
from nro.engine.bids import BidsRun
from nro.orchestration.registry import RegisteredWorkflow, Registry


class ParticipantUnavailableError(RuntimeError):
    """The selected participant cannot support any instance of the target."""


def instance_key(
    project: str,
    module: str,
    configuration_lineage_id: int,
    participant: str,
    entities: Mapping[str, str],
) -> str:
    """Return the stable identity key for one logical instance."""
    identity = {
        "project": project,
        "module": module,
        "configuration_lineage_id": configuration_lineage_id,
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
    registry: Registry
    runs: tuple[BidsRun, ...]
    aggregate_source_inputs: tuple[Path, ...]
    target_pairs: tuple[tuple[str, int], ...]
    memory_gb: int
    max_memory_gb: int

    def runtime_config(self, configuration_class: str) -> Path:
        return self.registry.runtime_config_path(self.registered, configuration_class)
