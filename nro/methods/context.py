"""Shared types and helpers for module methods contexts.

Each scientific module that supports methods reporting provides
``nro/modules/<module>/methods.py`` with these attributes:

``TITLE``
    Heading used for the module's part of the methods section.
``CONTEXT_VERSION``
    Integer version of the context schema. Templates declare the version they
    were written for.
``CONTEXT_FIELDS``
    Field schema for ``validate_metadata_fields``.
``CONFIGURATION_FIELDS``
    Mapping from each substantive configuration parameter to the context field
    that reports it.
``METHODS_SILENT``
    Mapping from substantive configuration parameters that methods text omits to
    the reason they are omitted.
``methods(configuration, manifests, *, workflow)``
    Return a ``ModuleMethods`` value for one resolved configuration, the
    resolved values of every class in the workflow, and the
    manifests of its fresh artifacts. ``ModuleMethods.images`` names the
    site container images the lineage ran.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

RUN_ENTITY_COLUMNS = (
    "project",
    "participant",
    "session",
    "task",
    "acquisition",
    "direction",
    "echo",
    "run",
    "space",
    "smoothing",
)


@dataclass(frozen=True)
class ManifestRecord:
    """One fresh artifact manifest with the BIDS entities of its work item."""

    entities: Mapping[str, str]
    data: Mapping[str, Any]


@dataclass(frozen=True)
class RunFact:
    """One row of the run-level supplementary table."""

    entities: Mapping[str, str]
    field: str
    value: str

    def row(self, module: str) -> tuple[str, ...]:
        """Return the table row in ``RUN_ENTITY_COLUMNS`` + module, field, value order."""
        return (
            *(self.entities.get(name, "") for name in RUN_ENTITY_COLUMNS),
            module,
            self.field,
            self.value,
        )


@dataclass(frozen=True)
class SoftwareUse:
    """A tool version observed in manifests, with its citation key if any."""

    software: str
    version: str
    citation: str = ""


@dataclass(frozen=True)
class ModuleMethods:
    """A module's methods context and its supplementary-table contributions."""

    context: Mapping[str, Any]
    run_facts: tuple[RunFact, ...] = ()
    software: tuple[SoftwareUse, ...] = ()
    images: tuple[str, ...] = field(default=())


def lookup(document: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a dotted path from nested mappings, returning ``default`` when absent."""
    value: Any = document
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def tally(
    manifests: Sequence[ManifestRecord], read: Callable[[ManifestRecord], object]
) -> dict[str, int]:
    """Count manifests by the string form of one derived value, in sorted order."""
    counts = Counter(str(read(record)) for record in manifests)
    return dict(sorted(counts.items()))


def participants(manifests: Iterable[ManifestRecord]) -> int:
    """Return the number of distinct project and participant pairs among manifests."""
    return len(
        {
            (record.entities.get("project", ""), record.entities.get("participant", ""))
            for record in manifests
        }
    )


def flatten_keys(values: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Return dotted leaf keys of a nested configuration mapping."""
    keys: list[str] = []
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            keys.extend(flatten_keys(value, name + "."))
        else:
            keys.append(name)
    return keys
