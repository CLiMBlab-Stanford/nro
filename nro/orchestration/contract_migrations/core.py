"""Normalize scientific artifact contracts through restricted migrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

INDETERMINATE = {"nro_historical_value": "indeterminate"}


def _path(value: str | Sequence[str]) -> tuple[str, ...]:
    result = tuple(value.split(".")) if isinstance(value, str) else tuple(value)
    if not result or any(not item for item in result):
        raise ValueError("Contract migration paths must contain nonempty keys")
    return result


def _parent(document: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    current = document
    for key in path[:-1]:
        value = current.get(key)
        if not isinstance(value, dict):
            return None
        current = value
    return current


@dataclass(frozen=True)
class AddField:
    """Add one field with distinct current and historical meanings."""

    path: tuple[str, ...]
    default: Any
    historical: Any

    def __init__(self, path: str | Sequence[str], *, default: Any, historical: Any) -> None:
        """Declare the current default and the meaning of historical absence."""
        object.__setattr__(self, "path", _path(path))
        object.__setattr__(self, "default", deepcopy(default))
        object.__setattr__(self, "historical", deepcopy(historical))

    def apply_historical(self, document: dict[str, Any]) -> None:
        """Impute the declared historical meaning when its parent exists."""
        parent = _parent(document, self.path)
        if parent is not None and self.path[-1] not in parent:
            parent[self.path[-1]] = deepcopy(self.historical)

    def apply_default(self, document: dict[str, Any]) -> None:
        """Impute the current default when its parent exists."""
        parent = _parent(document, self.path)
        if parent is not None and self.path[-1] not in parent:
            parent[self.path[-1]] = deepcopy(self.default)


@dataclass(frozen=True)
class RenameField:
    """Move a historical field without changing its value."""

    source: tuple[str, ...]
    destination: tuple[str, ...]

    def __init__(self, source: str | Sequence[str], destination: str | Sequence[str]) -> None:
        """Declare the old and current locations of one field."""
        object.__setattr__(self, "source", _path(source))
        object.__setattr__(self, "destination", _path(destination))

    def apply_historical(self, document: dict[str, Any]) -> None:
        """Move the source value or leave an absent optional field absent."""
        source_parent = _parent(document, self.source)
        if source_parent is None or self.source[-1] not in source_parent:
            return
        destination_parent = _parent(document, self.destination)
        if destination_parent is None:
            raise ValueError("Contract migration destination parent is absent")
        if self.destination[-1] in destination_parent:
            raise ValueError("Contract migration destination already exists")
        destination_parent[self.destination[-1]] = source_parent.pop(self.source[-1])

    def apply_default(self, document: dict[str, Any]) -> None:
        """Leave current syntax unchanged."""


@dataclass(frozen=True)
class RemoveField:
    """Remove a field declared irrelevant to current scientific identity."""

    path: tuple[str, ...]
    reconstructible: bool

    def __init__(self, path: str | Sequence[str], *, reconstructible: bool) -> None:
        """Declare a removable field whose value no longer carries meaning."""
        if not reconstructible:
            raise ValueError("Removed contract fields must be declared reconstructible")
        object.__setattr__(self, "path", _path(path))
        object.__setattr__(self, "reconstructible", True)

    def apply_historical(self, document: dict[str, Any]) -> None:
        """Remove the retired field when present."""
        parent = _parent(document, self.path)
        if parent is not None:
            parent.pop(self.path[-1], None)

    def apply_default(self, document: dict[str, Any]) -> None:
        """Remove retired syntax from current input as well."""
        self.apply_historical(document)


@dataclass(frozen=True)
class MapValues:
    """Map every admitted historical scalar value to current syntax."""

    path: tuple[str, ...]
    mapping: Mapping[Any, Any]

    def __init__(self, path: str | Sequence[str], mapping: Mapping[Any, Any]) -> None:
        """Declare an exhaustive mapping from historical to current values."""
        if not mapping:
            raise ValueError("Contract value mappings cannot be empty")
        object.__setattr__(self, "path", _path(path))
        object.__setattr__(self, "mapping", deepcopy(dict(mapping)))

    def apply_historical(self, document: dict[str, Any]) -> None:
        """Replace a present value or reject an incomplete mapping."""
        parent = _parent(document, self.path)
        if parent is None or self.path[-1] not in parent:
            return
        value = parent[self.path[-1]]
        try:
            parent[self.path[-1]] = deepcopy(self.mapping[value])
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Contract migration has no mapping for {'.'.join(self.path)}={value!r}"
            ) from error

    def apply_default(self, document: dict[str, Any]) -> None:
        """Leave current values unchanged."""


Operation = AddField | RenameField | RemoveField | MapValues


@dataclass(frozen=True)
class ContractMigration:
    """Describe one module contract transition from its preceding version."""

    destination: int
    summary: str
    contract: tuple[Operation, ...] = ()
    configuration: tuple[Operation, ...] = ()

    def __post_init__(self) -> None:
        if self.destination < 2:
            raise ValueError("Contract migration destinations must follow baseline version 1")
        if not self.summary.strip():
            raise ValueError("Contract migrations require a summary")


@dataclass(frozen=True)
class ContractMigrationChain:
    """Apply one module's contiguous scientific-contract history."""

    module: str
    migrations: tuple[ContractMigration, ...] = ()
    baseline_version: int = 1

    def __post_init__(self) -> None:
        expected = tuple(range(self.baseline_version + 1, self.current_version + 1))
        actual = tuple(item.destination for item in self.migrations)
        if actual != expected:
            raise ValueError(
                f"{self.module} contract migrations must be contiguous after "
                f"{self.baseline_version}: {actual}"
            )

    @property
    def current_version(self) -> int:
        """Return the schema derived from the immutable baseline and chain."""
        return self.baseline_version + len(self.migrations)

    def normalize(
        self,
        contract: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return current canonical views without mutating recorded evidence."""
        result = deepcopy(dict(contract))
        resolved = deepcopy(dict(configuration)) if configuration is not None else None
        raw_version = result.get("contract_schema", self.baseline_version)
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise ValueError("Artifact contract schema must be an integer")
        if not self.baseline_version <= raw_version <= self.current_version:
            raise ValueError(
                f"Unsupported {self.module} artifact contract schema {raw_version}; "
                f"supported range is {self.baseline_version}..{self.current_version}"
            )
        for migration in self.migrations:
            if migration.destination <= raw_version:
                continue
            for operation in migration.contract:
                operation.apply_historical(result)
            if resolved is not None:
                for operation in migration.configuration:
                    operation.apply_historical(resolved)
        for migration in self.migrations:
            for operation in migration.contract:
                operation.apply_default(result)
            if resolved is not None:
                for operation in migration.configuration:
                    operation.apply_default(resolved)
        result["contract_schema"] = self.current_version
        return result, resolved
