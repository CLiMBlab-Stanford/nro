"""Resolve workflow-dependent scientific module dependencies."""

from __future__ import annotations

from collections.abc import Mapping

_PRIMARY_DEPENDENCIES: dict[str, str | None] = {
    "anat": None,
    "func": "anat",
    "clean": "func",
    "dynconn": "clean",
    "microparcellation": "clean",
    "networks": None,
    "firstlevels": "func",
}

_DIRECT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "anat": (),
    "func": ("anat",),
    "clean": ("func", "anat"),
    "dynconn": ("clean",),
    "microparcellation": ("clean", "anat"),
    "networks": (),
    "firstlevels": ("func", "anat"),
}


def _network_source(configuration: Mapping[str, object]) -> str:
    source = str(configuration.get("connectivity_source", "microparcellation"))
    if source not in {"microparcellation", "dynconn"}:
        raise ValueError(f"Unsupported networks connectivity source: {source}")
    return source


def primary_dependency(module: str, configuration: Mapping[str, object]) -> str | None:
    """Return the module lineage that defines this module's primary ancestry."""

    if module == "networks":
        return _network_source(configuration)
    try:
        return _PRIMARY_DEPENDENCIES[module]
    except KeyError as error:
        raise ValueError(f"Unknown scientific module: {module}") from error


def direct_dependencies(module: str, configuration: Mapping[str, object]) -> tuple[str, ...]:
    """Return direct work-item dependencies for one resolved module configuration."""

    if module == "networks":
        return _network_source(configuration), "anat"
    try:
        return _DIRECT_DEPENDENCIES[module]
    except KeyError as error:
        raise ValueError(f"Unknown scientific module: {module}") from error
