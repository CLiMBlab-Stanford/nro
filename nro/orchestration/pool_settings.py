"""User-facing schema for mutable scheduler pool settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettingSpec:
    """Describe one setting accepted by the live scheduler interface."""

    values: str
    scope: str
    description: str


SETTABLE_SETTINGS = {
    "concurrency": SettingSpec(
        values="integer >= 1",
        scope="active requests",
        description="Maximum general workers shared by derivative and ingestion work.",
    ),
    "gpu_concurrency": SettingSpec(
        values="integer >= 1",
        scope="persistent scheduler",
        description="Maximum workers executing resource-specific GPU steps.",
    ),
}
