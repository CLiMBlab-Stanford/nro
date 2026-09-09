"""Shared manifest access and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from nro.orchestration.runner_graph import Step

from .execution import require_existing_path
from .io import read_json, write_json


def has_nested_key(mapping: dict[str, Any], dotted_path: str) -> bool:
    """Return whether a dotted path is explicitly present in nested mappings."""
    value: object = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def create_json_step(
    *,
    step_name: str,
    path: Path,
    payload: dict[str, Any],
    inputs: Sequence[Path],
    force: bool,
) -> Step:
    """Create an exact, semantically validated JSON-output step."""

    def validate() -> tuple[bool, str]:
        try:
            current = read_json(path)
        except (OSError, ValueError):
            return False, "Metadata is missing or invalid JSON."
        if current != payload:
            return False, "Metadata content does not match the current module configuration."
        return True, "Metadata matches the current module configuration."

    return Step.python(
        name=step_name,
        inputs=tuple(inputs),
        outputs=(path,),
        action=lambda: write_json(path, payload),
        validate=validate,
        force=force,
    )


def require_manifest_output(
    manifest: dict[str, Any],
    key: str,
    *,
    manifest_path: Path,
    manifest_name: str,
) -> Path:
    """Resolve and validate one top-level output recorded in a manifest."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SystemExit(f"{manifest_name} manifest is missing an outputs mapping: {manifest_path}")
    raw = str(outputs.get(key, "")).strip()
    if not raw:
        raise SystemExit(f"{manifest_name} manifest is missing outputs.{key}: {manifest_path}")
    path = Path(raw)
    require_existing_path(path, f"{manifest_name.lower()} output {key}")
    return path


def require_nested_manifest_output(
    manifest: dict[str, Any],
    section: str,
    key: str,
    *,
    manifest_path: Path,
    manifest_name: str,
) -> Path:
    """Resolve and validate an output in a nested manifest section."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SystemExit(f"{manifest_name} manifest is missing an outputs mapping: {manifest_path}")
    nested = outputs.get(section)
    if not isinstance(nested, dict):
        raise SystemExit(f"{manifest_name} manifest is missing outputs.{section}: {manifest_path}")
    raw = str(nested.get(key, "")).strip()
    if not raw:
        raise SystemExit(
            f"{manifest_name} manifest is missing outputs.{section}.{key}: {manifest_path}"
        )
    path = Path(raw)
    require_existing_path(path, f"{manifest_name.lower()} output {section}.{key}")
    return path
