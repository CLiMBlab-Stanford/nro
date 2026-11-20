"""Release-scoped repairs for narrowly identified historical defects."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class HotfixReport:
    """Describe the records one hotfix can repair or has repaired."""

    identifier: str
    summary: str
    projects: tuple[str, ...]
    paths: tuple[Path, ...]
    records: int
    applied: bool


class Hotfix(Protocol):
    """Define the interface implemented by each removable hotfix module."""

    HOTFIX_ID: str
    SUMMARY: str

    def run(self, registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
        """Preview or apply the repair to the selected projects."""
        ...


class PurgeHotfix(Hotfix, Protocol):
    """Define a hotfix that removes defective derivatives instead of repairing records.

    `nro hotfix` passes the owned work-item snapshot rows to `purge_roots` and
    purges the returned work items with all of their downstream consumers.
    """

    def purge_roots(self, rows: list[dict]) -> set[int]:
        """Return the IDs of work items whose derivatives the defect corrupted."""
        ...


def available() -> dict[str, Hotfix]:
    """Load the release-scoped hotfix modules shipped by this version."""
    loaded: dict[str, Hotfix] = {}
    for item in pkgutil.iter_modules(__path__):
        if item.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{item.name}")
        identifier = str(module.HOTFIX_ID)
        if identifier in loaded:
            raise RuntimeError(f"Duplicate hotfix identifier {identifier}")
        loaded[identifier] = module
    return loaded


def apply(registry, *, identifier: str, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Preview or apply one named hotfix against selected projects."""
    try:
        implementation = available()[identifier]
    except KeyError:
        choices = ", ".join(sorted(available())) or "none"
        raise ValueError(f"Unknown hotfix {identifier!r}; available hotfixes: {choices}") from None
    return implementation.run(registry, projects=projects, execute=execute)
