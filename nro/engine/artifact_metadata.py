"""Validation primitives for public artifact metadata contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_MISSING = object()


def _field_spec(value: object) -> tuple[str, bool, object]:
    if isinstance(value, str):
        return value, False, None
    if isinstance(value, Mapping) and isinstance(value.get("kind"), str):
        unknown = set(value) - {"kind", "default"}
        if unknown:
            raise ValueError("Unknown artifact-metadata field attributes: " + ", ".join(unknown))
        kind = str(value["kind"])
        has_default = "default" in value
        default = value.get("default")
        if has_default and not _matches_kind(default, kind):
            raise ValueError(f"Artifact-metadata default must have type {kind}")
        return kind, has_default, default
    raise ValueError("Artifact-metadata fields require a type name or kind/default mapping")


def _lookup(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
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
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
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
    fields: Mapping[str, object],
    *,
    label: str = "Artifact metadata",
) -> None:
    """Validate known fields, applying declared defaults when a field is absent."""
    for path, specification in fields.items():
        kind, has_default, default = _field_spec(specification)
        value = _lookup(document, path)
        if value is _MISSING:
            if not has_default:
                raise ValueError(f"{label} lacks required field {path}")
            value = default
        if not _matches_kind(value, kind):
            raise ValueError(
                f"{label} field {path} must have type {kind}, got {type(value).__name__}"
            )


def metadata_value(
    document: Mapping[str, Any],
    path: str,
    specification: object,
    *,
    label: str = "Artifact metadata",
) -> Any:
    """Read a declared field or return a copy of its compatibility default."""
    kind, has_default, default = _field_spec(specification)
    value = _lookup(document, path)
    if value is _MISSING:
        if not has_default:
            raise ValueError(f"{label} lacks required field {path}")
        value = deepcopy(default)
    if not _matches_kind(value, kind):
        raise ValueError(f"{label} field {path} must have type {kind}")
    return value


def metadata_contract_compatible(recorded: object, current: object) -> bool:
    """Compare metadata schemas while accepting removed fields and defaulted additions."""
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return recorded == current
    for section, current_value in current.items():
        recorded_value = recorded.get(section, _MISSING)
        if isinstance(current_value, Mapping) and all(
            isinstance(value, (str, Mapping)) for value in current_value.values()
        ):
            if isinstance(recorded_value, list) and all(
                isinstance(value, str) for value in recorded_value
            ):
                recorded_value = {value: "any" for value in recorded_value}
            if not isinstance(recorded_value, Mapping):
                return False
            for path, current_specification in current_value.items():
                try:
                    current_kind, has_default, _default = _field_spec(current_specification)
                except ValueError:
                    return False
                if path not in recorded_value:
                    if not has_default:
                        return False
                    continue
                try:
                    recorded_kind, recorded_has_default, recorded_default = _field_spec(
                        recorded_value[path]
                    )
                except ValueError:
                    return False
                if recorded_kind not in {"any", current_kind}:
                    return False
                if recorded_has_default and not has_default:
                    return False
                if recorded_has_default and has_default and recorded_default != _default:
                    return False
        elif recorded_value is _MISSING or recorded_value != current_value:
            return False
    return True
