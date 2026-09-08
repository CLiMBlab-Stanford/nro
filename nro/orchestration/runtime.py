"""Resolve a public workflow ID into one private configuration snapshot."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.registry import Registry
from nro.configuration.store import ConfigStore


CONFIGURATION_FINGERPRINT_ENV = "NRO_CONFIGURATION_FINGERPRINT"


def selected_configuration_fingerprint() -> str | None:
    """Return the intrinsic module-configuration fingerprint for this process.

    Workflow identity and upstream directory labels are deliberately excluded:
    this identifies only the named configuration for the current derivative
    module.
    """
    value = os.environ.get(CONFIGURATION_FINGERPRINT_ENV, "").strip()
    return value or None


def load_runtime_workflow_snapshot(runtime_config: str | Path) -> dict:
    """Load the immutable workflow snapshot containing a runtime configuration."""
    runtime = Path(runtime_config).expanduser().resolve()
    suffix = "_runtime"
    if not runtime.parent.name.endswith(suffix):
        raise ValueError(f"Runtime configuration is not inside a workflow snapshot: {runtime}")
    revision = runtime.parent.name[: -len(suffix)]
    snapshot = runtime.parent.parent / f"{revision}_workflow.yml"
    try:
        value = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read workflow snapshot for {runtime}: {snapshot}") from error
    if not isinstance(value, dict) or not isinstance(value.get("configurations"), dict):
        raise ValueError(f"Invalid workflow snapshot: {snapshot}")
    return value


def resolve_workflow_runtime(
    *,
    project: str,
    workflow_id: str,
    derivative_class: str,
    bids_root: str | Path = BIDS_PATH,
) -> Path:
    workflow = ConfigStore().resolve(workflow_id)
    registry = Registry.for_project(project, bids_root=bids_root)
    registered = registry.register_workflow(workflow)
    return registry.runtime_config_path(registered, derivative_class)


def select_runtime_config(
    *,
    project: str,
    workflow_id: str,
    derivative_class: str,
    bids_root: str | Path = BIDS_PATH,
) -> Path:
    """Select a public workflow, or validate an orchestrator-owned snapshot."""
    selected = os.environ.get("NRO_RUNTIME_CONFIG")
    if not selected:
        workflow = ConfigStore().resolve(workflow_id)
        registry = Registry.for_project(project, bids_root=bids_root)
        registered = registry.register_workflow(workflow)
        path = registry.runtime_config_path(registered, derivative_class)
        os.environ["NRO_RUNTIME_CONFIG"] = str(path)
        os.environ[CONFIGURATION_FINGERPRINT_ENV] = workflow.configuration(
            derivative_class
        ).scientific_fingerprint
        return path
    registry = Registry.for_project(project, bids_root=bids_root)
    path = Path(selected).expanduser().resolve()
    root = registry.paths.workflows.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Private runtime configuration is outside the central registry: {path}"
        ) from error
    if not path.is_file():
        raise ValueError(f"Private runtime configuration does not exist: {path}")
    return path
