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

from nro.configuration.site import definitions_root, resolve_resources
from nro.configuration.parsing import DefinitionError, parse_mapping
from nro.configuration.schema import RUNTIME_FIELDS, SCHEMAS, compile_configuration, normalize_fields, scientific_values


DERIVATIVE_CLASSES: tuple[str, ...] = (
    "preprocessing",
    "clean",
    "microparcellation",
    "networks",
    "firstlevels",
)

UPSTREAM_CLASS: dict[str, str | None] = {
    "preprocessing": None,
    "clean": "preprocessing",
    "microparcellation": "clean",
    "networks": "microparcellation",
    "firstlevels": "preprocessing",
}

class WorkflowError(DefinitionError):
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
    """Resolved class configuration with its source ID, values, and fingerprint."""
    derivative_class: str
    config_id: str
    path: Path
    values: dict[str, Any]
    fingerprint: str

    @property
    def scientific_fingerprint(self) -> str:
        """Identify the named scientific settings, excluding execution controls."""
        return configuration_fingerprint(self.derivative_class, self.config_id, self.values, scientific=True)


def configuration_fingerprint(kind: str, identifier: str, values: dict, *, scientific: bool = False) -> str:
    """Hash a named snapshot; optionally compare only its scientific settings."""
    if scientific:
        values = scientific_values(kind, compile_configuration(kind, values))
    return fingerprint({"derivative_class": kind, "config_id": identifier, "values": values})


@dataclass(frozen=True)
class ResolvedWorkflow:
    """Workflow selection and the resolved configurations it references."""
    workflow_id: str
    path: Path
    selections: dict[str, str]
    configurations: dict[str, ResolvedConfiguration]
    fingerprint: str

    def configuration(self, name: str) -> ResolvedConfiguration:
        """Return the resolved configuration for a derivative class; unknown names raise KeyError."""
        try:
            return self.configurations[name]
        except KeyError as error:
            raise WorkflowError(f"Workflow has no derivative class {name!r}") from error


class ConfigStore:
    """Resolve IDs from one external definitions store."""

    def __init__(self, root: Path | None = None) -> None:
        """Use the selected definitions root, or an explicit root for validation and drafts."""
        self.root = Path(root).expanduser().resolve() if root is not None else definitions_root()
        if not self.root.is_dir():
            raise WorkflowError(f"Definitions store does not exist: {self.root}; use nro definitions create")
        if (self.root / '.nro-incomplete').exists():
            raise WorkflowError(f'Definitions publication is incomplete: {self.root}')

    @property
    def configs(self) -> Path:
        """Return the directory containing the derivative-class configurations."""
        return self.root / "configs"

    def _find(self, filename: str, *, category: str) -> Path:
        organized = (self.root if category == "workflows" else self.configs) / category / filename
        if not organized.resolve().is_relative_to(self.root):
            raise WorkflowError(f"Definition escapes the store: {organized}")
        if organized.is_file():
            return organized
        raise WorkflowError(
            f"Configuration file {str(Path(category) / filename)!r} was not found in store: {self.root}"
        )

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any]:
        try:
            return parse_mapping(path.read_text(encoding="utf-8"), source=str(path))
        except DefinitionError as error:
            raise WorkflowError(str(error)) from error

    def workflow_path(self, workflow: str) -> tuple[str, Path]:
        """Validate a workflow selector and return its ID and source file."""
        suffix = "_workflow.yml"
        workflow_id = validate_config_id(str(workflow), kind="workflow")
        return workflow_id, self._find(f"{workflow_id}{suffix}", category="workflows")

    def configuration_path(self, derivative_class: str, config_id: str) -> Path:
        """Return the source file for a validated derivative-class configuration ID."""
        if derivative_class not in DERIVATIVE_CLASSES:
            raise WorkflowError(f"Unknown derivative class: {derivative_class}")
        config_id = validate_config_id(config_id, kind=f"{derivative_class} configuration")
        return self._find(
            f"{config_id}_{derivative_class}.yml", category=derivative_class
        )

    def load_configuration(
        self, derivative_class: str, config_id: str, *, document: Mapping[str, Any] | None = None,
    ) -> ResolvedConfiguration:
        """Load main defaults and merge a named configuration override.

        Resolve site references before fingerprinting. Invalid mappings and unknown
        override keys raise WorkflowError rather than being silently accepted.
        document validates a staged definition without writing it to the store.
        """
        base_path = self.configuration_path(derivative_class, "main")
        config_id = validate_config_id(config_id, kind=f"{derivative_class} configuration")
        path = (self.configuration_path(derivative_class, config_id) if document is None
                else self.configs / derivative_class / f"{config_id}_{derivative_class}.yml")
        if document is not None and not isinstance(document, Mapping):
            raise WorkflowError("Configuration must contain a mapping")
        declared = self._read_mapping(path) if document is None else dict(document)
        forbidden = sorted(set(declared) & RUNTIME_FIELDS[derivative_class].keys())
        if forbidden:
            raise WorkflowError(f"{path}: {', '.join(forbidden)} belong in a workflow")
        if derivative_class == "firstlevels" and set(declared) & {"model", "models", "task", "model_set", "model_documents"}:
            raise WorkflowError(f"{path}: Firstlevels selection keys belong in CLI requests, not configuration")
        try:
            base = compile_configuration(derivative_class, resolve_resources(
                declared if config_id == "main" else self._read_mapping(base_path)))
        except (ValueError, TypeError) as error:
            raise WorkflowError(f"{path if config_id == 'main' else base_path}: {error}") from error
        try:
            override = normalize_fields(SCHEMAS[derivative_class], resolve_resources(
                {} if config_id == "main" else declared), location=derivative_class, complete=False)
        except (ValueError, TypeError) as error:
            raise WorkflowError(f"{path}: {error}") from error
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
        try:
            values = compile_configuration(derivative_class, values)
        except DefinitionError as error:
            raise WorkflowError(f"{path}: {error}") from error
        return ResolvedConfiguration(
            derivative_class=derivative_class,
            config_id=config_id,
            path=path,
            values=values,
            fingerprint=configuration_fingerprint(derivative_class, config_id, values),
        )

    def resolve(
        self, workflow: str | Path = "main", *, document: Mapping[str, Any] | None = None,
    ) -> ResolvedWorkflow:
        """Resolve a workflow and all selected class configurations.

        Return immutable selection metadata and resolved values; missing or invalid
        workflow definitions raise WorkflowError. document validates a staged
        workflow without publishing a file.
        """
        workflow_id = validate_config_id(str(workflow), kind="workflow")
        path = (self.workflow_path(workflow_id)[1] if document is None
                else self.root / "workflows" / f"{workflow_id}_workflow.yml")
        declared = self._read_mapping(path) if document is None else document
        if not isinstance(declared, Mapping) or any(not isinstance(key, str) for key in declared):
            raise WorkflowError(f"{path}: Workflow must contain a mapping with string keys")
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
