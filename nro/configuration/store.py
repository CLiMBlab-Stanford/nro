"""Resolve named nro workflows and module-specific configurations.

Workflow files are named ``<ID>_workflow.yml``. They map configuration classes to
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
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from nro.configuration.parsing import DefinitionError, parse_mapping
from nro.configuration.schema import (
    RUNTIME_FIELDS,
    SCHEMAS,
    compile_configuration,
    normalize_fields,
    scientific_values,
)
from nro.configuration.site import definitions_roots, resolve_resources
from nro.modules import MODULE_NAMES

PACKAGED_CONFIGS = Path(__file__).parent / "starters/configs"

CONFIGURATION_CLASSES = MODULE_NAMES


class WorkflowError(DefinitionError):
    """A workflow or one of its module configurations is invalid."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_config_id(value: str, *, kind: str) -> str:
    """Validate a portable configuration, workflow, or model identifier."""
    identifier = str(value).strip()
    if not _ID_RE.fullmatch(identifier):
        raise WorkflowError(
            f"Invalid {kind} ID {value!r}; IDs may contain only letters, digits, '.', '_', and '-'"
        )
    return identifier


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
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
    """Hash the canonical JSON representation of a definition value."""
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedConfiguration:
    """Resolved class configuration with its source ID, values, and fingerprint."""

    configuration_class: str
    config_id: str
    path: Path
    values: dict[str, Any]
    fingerprint: str

    @property
    def scientific_fingerprint(self) -> str:
        """Identify the named scientific settings, excluding execution controls."""
        return configuration_fingerprint(
            self.configuration_class, self.config_id, self.values, scientific=True
        )

    def module_fingerprint(self, module: str) -> str:
        """Identify settings that can affect one module in a shared class."""
        if self.configuration_class != module:
            raise ValueError(
                f"Configuration class {self.configuration_class!r} does not configure {module!r}"
            )
        return self.scientific_fingerprint


def configuration_fingerprint(
    kind: str,
    identifier: str,
    values: dict,
    *,
    scientific: bool = False,
    complete_snapshot: bool = False,
) -> str:
    """Hash a named snapshot; optionally compare only its scientific settings."""
    if scientific:
        values = scientific_values(
            kind,
            _compile_scientific_snapshot(kind, values, complete=complete_snapshot),
        )
    return fingerprint({"module": kind, "config_id": identifier, "values": values})


@lru_cache(maxsize=None)
def _scientific_defaults(kind: str) -> dict:
    """Compile one package-owned default snapshot once per process."""
    default_path = PACKAGED_CONFIGS / kind / f"main_{kind}.yml"
    return compile_configuration(
        kind,
        resolve_resources(
            parse_mapping(default_path.read_text(encoding="utf-8"), source=str(default_path))
        ),
    )


def _compile_scientific_snapshot(kind: str, values: dict, *, complete: bool = False) -> dict:
    """Normalize a current or historical snapshot against today's declared defaults.

    Configuration files are strict when authored: misspelled and retired keys are
    errors. Stored artifact snapshots need different semantics. Contract migrations
    first reconstruct introduced scientific fields explicitly; this function then
    projects only recognized fields and validates the reconstructed snapshot.
    """

    defaults = _scientific_defaults(kind)

    def project(schema: dict, baseline: dict, snapshot: Mapping[str, Any]) -> dict:
        result = deepcopy(baseline)
        for key, rule in schema.items():
            if key not in snapshot:
                continue
            value = snapshot[key]
            if isinstance(rule, dict) and isinstance(value, Mapping):
                nested_baseline = baseline.get(key, {})
                if not isinstance(nested_baseline, Mapping):
                    nested_baseline = {}
                result[key] = project(rule, nested_baseline, value)
            else:
                result[key] = deepcopy(value)
        return result

    if not isinstance(values, Mapping):
        raise DefinitionError(f"{kind}: configuration snapshot must be a mapping")
    if complete:
        projected = project(SCHEMAS[kind], {}, values)
    else:
        projected = project(SCHEMAS[kind], defaults, values)
    return compile_configuration(kind, projected)


@dataclass(frozen=True)
class ResolvedWorkflow:
    """Workflow selection and the resolved configurations it references."""

    workflow_id: str
    path: Path
    selections: dict[str, str]
    configurations: dict[str, ResolvedConfiguration]
    fingerprint: str

    def configuration(self, name: str) -> ResolvedConfiguration:
        """Return one resolved module configuration or reject an unknown name."""
        try:
            return self.configurations[name]
        except KeyError as error:
            raise WorkflowError(f"Workflow has no configuration class {name!r}") from error


class ConfigStore:
    """Resolve IDs through the active definitions inheritance chain."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        roots: tuple[Path, ...] | None = None,
        site_values: Mapping[str, object] | None = None,
    ) -> None:
        """Use active stores and site values, or explicit ones for staged validation."""
        if root is not None and roots is not None:
            raise ValueError("Specify either root or roots, not both")
        self.roots = (
            tuple(Path(path).expanduser().resolve() for path in roots)
            if roots is not None
            else (Path(root).expanduser().resolve(),)
            if root is not None
            else definitions_roots()
        )
        self.root = self.roots[0]
        self.site_values = dict(site_values) if site_values is not None else None
        from nro.configuration.definition_migrations import validate_store_integrity

        for candidate in self.roots:
            if not candidate.is_dir():
                raise WorkflowError(
                    f"Definitions store does not exist: {candidate}; use nro def init"
                )
            if (candidate / ".nro-incomplete").exists():
                raise WorkflowError(f"Definitions publication is incomplete: {candidate}")
            validate_store_integrity(candidate)

    @property
    def configs(self) -> Path:
        """Return the directory containing module configurations."""
        return self.root / "configs"

    def workflow_ids(self) -> tuple[str, ...]:
        """Return defined workflow IDs with ``main`` first when present."""
        suffix = "_workflow.yml"
        identifiers = {
            path.name[: -len(suffix)]
            for root in self.roots
            for path in (root / "workflows").glob(f"*{suffix}")
        }
        return tuple(sorted(identifiers, key=lambda value: (value != "main", value)))

    def _find(self, filename: str, *, category: str) -> Path:
        relative = (
            Path("workflows") / filename
            if category == "workflows"
            else Path("configs") / category / filename
        )
        for root in self.roots:
            organized = root / relative
            if not organized.resolve().is_relative_to(root):
                raise WorkflowError(f"Definition escapes the store: {organized}")
            if organized.is_file():
                return organized
        raise WorkflowError(
            f"Configuration file {str(Path(category) / filename)!r} was not found in "
            f"definitions chain: {', '.join(map(str, self.roots))}"
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

    def configuration_path(self, configuration_class: str, config_id: str) -> Path:
        """Return an external configuration or the packaged ``main`` default."""
        if configuration_class not in CONFIGURATION_CLASSES:
            raise WorkflowError(f"Unknown configuration class: {configuration_class}")
        config_id = validate_config_id(config_id, kind=f"{configuration_class} configuration")
        filename = f"{config_id}_{configuration_class}.yml"
        relative = Path("configs") / configuration_class / filename
        for root in self.roots:
            external = root / relative
            if not external.resolve().is_relative_to(root):
                raise WorkflowError(f"Definition escapes the store: {external}")
            if external.is_file():
                return external
        if config_id == "main":
            packaged = PACKAGED_CONFIGS / configuration_class / filename
            if packaged.is_file():
                return packaged
        raise WorkflowError(
            f"Configuration file {str(Path(configuration_class) / filename)!r} was not found "
            f"in the definitions chain or packaged defaults: {', '.join(map(str, self.roots))}"
        )

    def _merge_configuration(
        self,
        configuration_class: str,
        base: Mapping[str, Any],
        declared: Mapping[str, Any],
        *,
        path: Path,
    ) -> dict[str, Any]:
        """Validate and merge one partial configuration over resolved values."""
        forbidden = sorted(set(declared) & RUNTIME_FIELDS[configuration_class].keys())
        if forbidden:
            raise WorkflowError(
                f"{path}: {', '.join(forbidden)} are managed by orchestration and cannot "
                f"be set in a {configuration_class} configuration"
            )
        if configuration_class == "firstlevels" and set(declared) & {
            "model",
            "models",
            "task",
            "model_set",
            "model_documents",
        }:
            raise WorkflowError(
                f"{path}: Firstlevels selection keys belong in CLI requests, not configuration"
            )
        try:
            override = normalize_fields(
                SCHEMAS[configuration_class],
                resolve_resources(declared, site_values=self.site_values),
                location=configuration_class,
                complete=False,
            )
        except (ValueError, TypeError) as error:
            raise WorkflowError(f"{path}: {error}") from error
        flexible_filter = (
            override.pop("input_filter", None)
            if configuration_class in {"dynconn", "firstlevels", "microparcellation"}
            else None
        )
        values = _deep_merge(base, override)
        if flexible_filter is not None:
            if not isinstance(flexible_filter, Mapping):
                raise WorkflowError("Configuration option input_filter must be a mapping")
            values["input_filter"] = deepcopy(dict(flexible_filter))
        try:
            return compile_configuration(configuration_class, values)
        except DefinitionError as error:
            raise WorkflowError(f"{path}: {error}") from error

    def load_configuration(
        self,
        configuration_class: str,
        config_id: str,
        *,
        document: Mapping[str, Any] | None = None,
    ) -> ResolvedConfiguration:
        """Load main defaults and merge a named configuration override.

        Resolve site references before fingerprinting. Invalid mappings and unknown
        override keys raise WorkflowError rather than being silently accepted.
        document validates a staged definition without writing it to the store.
        """
        default_path = PACKAGED_CONFIGS / configuration_class / f"main_{configuration_class}.yml"
        if not default_path.is_file():
            raise WorkflowError(f"Packaged defaults are missing: {default_path}")
        config_id = validate_config_id(config_id, kind=f"{configuration_class} configuration")
        main_filename = f"main_{configuration_class}.yml"
        external_main = next(
            (
                root / "configs" / configuration_class / main_filename
                for root in self.roots
                if (root / "configs" / configuration_class / main_filename).is_file()
            ),
            self.configs / configuration_class / main_filename,
        )
        target = self.configs / configuration_class / f"{config_id}_{configuration_class}.yml"
        path = (
            self.configuration_path(configuration_class, config_id) if document is None else target
        )
        if document is not None and not isinstance(document, Mapping):
            raise WorkflowError("Configuration must contain a mapping")
        try:
            base = compile_configuration(
                configuration_class,
                resolve_resources(self._read_mapping(default_path), site_values=self.site_values),
            )
        except (ValueError, TypeError) as error:
            raise WorkflowError(f"{default_path}: {error}") from error
        main_declared = (
            dict(document)
            if config_id == "main" and document is not None
            else self._read_mapping(external_main)
            if external_main.is_file()
            else {}
        )
        values = self._merge_configuration(
            configuration_class,
            base,
            main_declared,
            path=external_main if external_main.is_file() or document is not None else default_path,
        )
        if config_id != "main":
            declared = self._read_mapping(path) if document is None else dict(document)
            values = self._merge_configuration(configuration_class, values, declared, path=path)
        markup_id = values.get("markup")
        if markup_id is not None:
            from nro.configuration.markup import MarkupStore

            try:
                MarkupStore(roots=self.roots).path(markup_id)
            except (FileNotFoundError, ValueError) as error:
                raise WorkflowError(f"{path}: {error}") from error
        return ResolvedConfiguration(
            configuration_class=configuration_class,
            config_id=config_id,
            path=path,
            values=values,
            fingerprint=configuration_fingerprint(configuration_class, config_id, values),
        )

    def resolve(
        self,
        workflow: str | Path = "main",
        *,
        document: Mapping[str, Any] | None = None,
    ) -> ResolvedWorkflow:
        """Resolve a workflow and all selected class configurations.

        Return immutable selection metadata and resolved values; missing or invalid
        workflow definitions raise WorkflowError. document validates a staged
        workflow without publishing a file.
        """
        workflow_id = validate_config_id(str(workflow), kind="workflow")
        path = (
            self.workflow_path(workflow_id)[1]
            if document is None
            else self.root / "workflows" / f"{workflow_id}_workflow.yml"
        )
        declared = self._read_mapping(path) if document is None else document
        if not isinstance(declared, Mapping) or any(not isinstance(key, str) for key in declared):
            raise WorkflowError(f"{path}: Workflow must contain a mapping with string keys")
        unknown = sorted(set(declared) - set(CONFIGURATION_CLASSES))
        if unknown:
            raise WorkflowError(
                f"Unknown configuration class(es) in {path.name}: {', '.join(unknown)}"
            )
        selections: dict[str, str] = {}
        configurations: dict[str, ResolvedConfiguration] = {}
        for configuration_class in CONFIGURATION_CLASSES:
            value = declared.get(configuration_class, "main")
            if not isinstance(value, str) or not value.strip():
                raise WorkflowError(
                    f"Workflow selection {configuration_class!r} must be a nonempty configuration ID"
                )
            config_id = value.strip()
            selections[configuration_class] = config_id
            configurations[configuration_class] = self.load_configuration(
                configuration_class, config_id
            )
        markup_ids = {
            configuration.values.get("markup") for configuration in configurations.values()
        }
        if len(markup_ids) != 1:
            selected = ", ".join(
                f"{name}={configurations[name].values.get('markup')!r}"
                for name in CONFIGURATION_CLASSES
            )
            raise WorkflowError(
                "All module configurations in one workflow must select the same markup "
                f"so they share one source-BIDS view; found {selected}"
            )
        resolved = {
            "workflow_id": workflow_id,
            "selections": selections,
            # This key is part of the stable fingerprint serialization.
            "configs": {
                configuration_class: configurations[configuration_class].fingerprint
                for configuration_class in CONFIGURATION_CLASSES
            },
        }
        return ResolvedWorkflow(
            workflow_id=workflow_id,
            path=path,
            selections=selections,
            configurations=configurations,
            fingerprint=fingerprint(resolved),
        )


def main_configuration_value(configuration_class: str, *keys: str) -> Any:
    """Read one value from the central store's main class configuration."""
    value: Any = ConfigStore().load_configuration(configuration_class, "main").values
    for key in keys:
        value = value[key]
    return deepcopy(value)


def main_configuration_factory(
    configuration_class: str,
    *keys: str,
    converter: Callable[[Any], Any] | None = None,
) -> Callable[[], Any]:
    """Build a dataclass default factory backed by the central main config."""

    def factory() -> Any:
        value = main_configuration_value(configuration_class, *keys)
        return converter(value) if converter is not None else value

    return factory


def build_parser() -> argparse.ArgumentParser:
    """Build the internal workflow-inspection parser."""
    parser = argparse.ArgumentParser(description="Inspect a resolved nro workflow")
    parser.add_argument("workflow", nargs="?", default="main")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Print one resolved workflow as text or JSON."""
    args = build_parser().parse_args(argv)
    resolved = ConfigStore().resolve(args.workflow)
    rows = [
        {
            "class": configuration_class,
            "configuration": resolved.configurations[configuration_class].config_id,
            "path": str(resolved.configurations[configuration_class].path),
            "fingerprint": resolved.configurations[configuration_class].fingerprint,
        }
        for configuration_class in CONFIGURATION_CLASSES
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
    print(f"{'CONFIGURATION CLASS':20} {'CONFIGURATION':24} PATH")
    for row in rows:
        print(f"{row['class']:20} {row['configuration']:24} {row['path']}")


if __name__ == "__main__":
    main()
