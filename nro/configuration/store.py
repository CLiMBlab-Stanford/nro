"""Resolve named nro workflows and derivative-class configurations.

Workflow files are named ``<ID>_workflow.yml``. They map derivative classes to
configuration IDs; omitted classes default to ``main``. Configuration files
are named ``<ID>_<CLASS>.yml`` and contain only options local to that class.
Upstream configuration relationships are supplied by the workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from nro.configuration.paths import CONFIG_PATH


DERIVATIVE_CLASSES: tuple[str, ...] = (
    "preprocessing",
    "clean",
    "microparcellation",
    "networks",
)

UPSTREAM_CLASS: dict[str, str | None] = {
    "preprocessing": None,
    "clean": "preprocessing",
    "microparcellation": "clean",
    "networks": "microparcellation",
}

_UPSTREAM_KEYS = {
    "preprocessing": frozenset(),
    "clean": frozenset({"preprocessing_directory"}),
    "microparcellation": frozenset(
        {"preprocessing_directory", "clean_directory"}
    ),
    "networks": frozenset({"microparcellation_directory"}),
}


class WorkflowError(ValueError):
    """A workflow or one of its derivative configurations is invalid."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_config_id(value: str, *, kind: str) -> str:
    identifier = str(value).strip()
    if not _ID_RE.fullmatch(identifier):
        raise WorkflowError(
            f"Invalid {kind} ID {value!r}; IDs may contain only letters, digits, '.', '_', and '-'"
        )
    return identifier


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        location = f"{prefix}.{key}" if prefix else str(key)
        if key not in result:
            raise WorkflowError(f"Unknown configuration option: {location}")
        if isinstance(result[key], Mapping):
            if not isinstance(value, Mapping):
                raise WorkflowError(f"Configuration option {location} must be a mapping")
            result[key] = _deep_merge(result[key], value, location)
        else:
            result[key] = deepcopy(value)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def fingerprint(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedConfiguration:
    derivative_class: str
    config_id: str
    path: Path
    values: dict[str, Any]
    fingerprint: str


@dataclass(frozen=True)
class ResolvedWorkflow:
    workflow_id: str
    path: Path
    selections: dict[str, str]
    configurations: dict[str, ResolvedConfiguration]
    fingerprint: str

    def configuration(self, name: str) -> ResolvedConfiguration:
        try:
            return self.configurations[name]
        except KeyError as error:
            raise WorkflowError(f"Workflow has no derivative class {name!r}") from error


class ConfigStore:
    """Resolve IDs from the repository's single configuration store."""

    def __init__(self) -> None:
        self.root = Path(CONFIG_PATH).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkflowError(f"Configured central store does not exist: {self.root}")

    def _find(self, filename: str, *, category: str) -> Path:
        organized = self.root / category / filename
        if organized.is_file():
            return organized
        raise WorkflowError(
            f"Configuration file {str(Path(category) / filename)!r} was not found in store: {self.root}"
        )

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any]:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise WorkflowError(f"Configuration must contain a mapping: {path}")
        return loaded

    def workflow_path(self, workflow: str) -> tuple[str, Path]:
        suffix = "_workflow.yml"
        workflow_id = validate_config_id(str(workflow), kind="workflow")
        return workflow_id, self._find(f"{workflow_id}{suffix}", category="workflows")

    def configuration_path(self, derivative_class: str, config_id: str) -> Path:
        if derivative_class not in DERIVATIVE_CLASSES:
            raise WorkflowError(f"Unknown derivative class: {derivative_class}")
        config_id = validate_config_id(config_id, kind=f"{derivative_class} configuration")
        return self._find(
            f"{config_id}_{derivative_class}.yml", category=derivative_class
        )

    def load_configuration(
        self, derivative_class: str, config_id: str
    ) -> ResolvedConfiguration:
        path = self.configuration_path(derivative_class, config_id)
        base_path = self.configuration_path(derivative_class, "main")
        base = self._read_mapping(base_path)
        override = {} if path == base_path else self._read_mapping(path)
        forbidden = sorted(
            (set(base) | set(override)) & _UPSTREAM_KEYS[derivative_class]
        )
        if forbidden:
            raise WorkflowError(
                f"{path.name} contains upstream configuration key(s) that belong in a "
                f"workflow: {', '.join(forbidden)}"
            )
        flexible_filter = (
            override.pop("input_filter", None)
            if derivative_class == "microparcellation"
            else None
        )
        values = _deep_merge(base, override)
        if flexible_filter is not None:
            if not isinstance(flexible_filter, Mapping):
                raise WorkflowError("Configuration option input_filter must be a mapping")
            values["input_filter"] = deepcopy(dict(flexible_filter))
        return ResolvedConfiguration(
            derivative_class=derivative_class,
            config_id=config_id,
            path=path,
            values=values,
            fingerprint=fingerprint(
                {
                    "derivative_class": derivative_class,
                    "config_id": config_id,
                    "values": values,
                }
            ),
        )

    def resolve(self, workflow: str | Path = "main") -> ResolvedWorkflow:
        workflow_id, path = self.workflow_path(workflow)
        declared = self._read_mapping(path)
        unknown = sorted(set(declared) - set(DERIVATIVE_CLASSES))
        if unknown:
            raise WorkflowError(
                f"Unknown derivative class(es) in {path.name}: {', '.join(unknown)}"
            )
        selections: dict[str, str] = {}
        configurations: dict[str, ResolvedConfiguration] = {}
        for derivative_class in DERIVATIVE_CLASSES:
            value = declared.get(derivative_class, "main")
            if not isinstance(value, str) or not value.strip():
                raise WorkflowError(
                    f"Workflow selection {derivative_class!r} must be a nonempty configuration ID"
                )
            config_id = value.strip()
            selections[derivative_class] = config_id
            configurations[derivative_class] = self.load_configuration(
                derivative_class, config_id
            )
        resolved = {
            "workflow_id": workflow_id,
            "selections": selections,
            # This key is part of the stable fingerprint serialization.
            "configs": {
                derivative_class: configurations[derivative_class].fingerprint
                for derivative_class in DERIVATIVE_CLASSES
            },
        }
        return ResolvedWorkflow(
            workflow_id=workflow_id,
            path=path,
            selections=selections,
            configurations=configurations,
            fingerprint=fingerprint(resolved),
        )


def main_configuration_value(derivative_class: str, *keys: str) -> Any:
    """Read one value from the central store's main class configuration."""
    value: Any = ConfigStore().load_configuration(derivative_class, "main").values
    for key in keys:
        value = value[key]
    return deepcopy(value)


def main_configuration_factory(
    derivative_class: str,
    *keys: str,
    converter: Callable[[Any], Any] | None = None,
) -> Callable[[], Any]:
    """Build a dataclass default factory backed by the central main config."""
    def factory() -> Any:
        value = main_configuration_value(derivative_class, *keys)
        return converter(value) if converter is not None else value

    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a resolved nro workflow")
    parser.add_argument("workflow", nargs="?", default="main")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    resolved = ConfigStore().resolve(args.workflow)
    rows = [
        {
            "class": derivative_class,
            "configuration": resolved.configurations[derivative_class].config_id,
            "path": str(resolved.configurations[derivative_class].path),
            "fingerprint": resolved.configurations[derivative_class].fingerprint,
        }
        for derivative_class in DERIVATIVE_CLASSES
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "workflow": resolved.workflow_id,
                    "fingerprint": resolved.fingerprint,
                    "classes": rows,
                },
                indent=2,
            )
        )
        return
    print(f"Workflow: {resolved.workflow_id}")
    print(f"Definition: {resolved.path}")
    print(f"Fingerprint: {resolved.fingerprint}")
    print(f"{'DERIVATIVE CLASS':20} {'CONFIGURATION':24} PATH")
    for row in rows:
        print(f"{row['class']:20} {row['configuration']:24} {row['path']}")


if __name__ == "__main__":
    main()
