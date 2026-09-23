"""Process-level protocol for runner steps that need another worker class."""

from __future__ import annotations

from dataclasses import dataclass

RESOURCE_HANDOFF_EXIT = 75


@dataclass(frozen=True)
class ResourceHandoff:
    """Describe why a runner yielded before completing its work item."""

    kind: str
    step_id: str | None = None
    resource_class: str | None = None
    reason: str = ""


class ResourceHandoffExit(SystemExit):
    """Exit a module cleanly after writing a durable handoff request."""

    def __init__(self, handoff: ResourceHandoff) -> None:
        """Carry the structured handoff while returning its reserved exit code."""
        super().__init__(RESOURCE_HANDOFF_EXIT)
        self.handoff = handoff
