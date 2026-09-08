"""Validation primitives for public artifact metadata contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _field(document: Mapping[str, Any], path: str, *, label: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"{label} lacks required field {path}")
        value = value[component]
    return value


def _matches_kind(value: Any, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "nullable_boolean":
        return value is None or isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "nullable_number":
        return value is None or _matches_kind(value, "number")
    if kind == "string":
        return isinstance(value, str)
    if kind == "nullable_string":
        return value is None or isinstance(value, str)
    if kind == "string_list":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if kind == "list":
        return isinstance(value, list)
    if kind == "mapping":
        return isinstance(value, Mapping)
    if kind == "nullable_mapping":
        return value is None or isinstance(value, Mapping)
    raise ValueError(f"Unknown artifact-metadata field kind: {kind}")


def validate_metadata_fields(
    document: Mapping[str, Any],
    fields: Mapping[str, str],
    *,
    label: str = "Artifact metadata",
) -> None:
    """Validate required dotted field paths and their JSON-compatible types."""
    for path, kind in fields.items():
        value = _field(document, path, label=label)
        if not _matches_kind(value, kind):
            raise ValueError(
                f"{label} field {path} must have type {kind}, "
                f"got {type(value).__name__}"
            )
