"""Parse and validate study analysis manifests and expand their parameters."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import keyword
import re
import string
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml

SCOPES = ("subject", "run", "project", "study")
SUPPORTED_SCOPES = ("subject", "run")
RESOURCE_CLASSES = ("small", "medium", "large")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ANALYSIS_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_MANIFEST_KEYS = {
    "version",
    "entry",
    "environment",
    "scope",
    "inputs",
    "outputs",
    "parameters",
    "resources",
}
_INPUT_KEYS = {"from", "workflow", "config", "select", "many", "min", "max", "optional", "path"}
_RESERVED_PLACEHOLDERS = {"project", "participant", "session", "stem"}


class ManifestError(ValueError):
    """An analysis manifest violates the study-analysis schema."""


def bids_label(value: object) -> str:
    """Return the BIDS label for a scalar value by removing non-alphanumerics."""
    label = re.sub(r"[^A-Za-z0-9]", "", str(value))
    if not label:
        raise ManifestError(f"Value {value!r} has no alphanumeric characters for a BIDS label")
    return label


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


@dataclass(frozen=True)
class InputSpec:
    """One declared analysis input."""

    name: str
    source: str
    workflow: str | None
    config: str | None
    select: Mapping[str, Any]
    many: bool
    minimum: int
    maximum: int | None
    optional: bool
    path: str | None

    @property
    def module(self) -> str | None:
        """Return the upstream module for a ``from: MODULE`` input."""
        if self.source == "raw" or self.source.startswith("analysis:"):
            return None
        return self.source


@dataclass(frozen=True)
class Expansion:
    """One expanded parameter: ordered BIDS labels and their values."""

    name: str
    labels: tuple[str, ...]
    values: tuple[Any, ...]


@dataclass(frozen=True)
class Manifest:
    """A validated analysis manifest."""

    analysis: str
    version: int
    entry: str
    container: str | None
    scope: str
    inputs: Mapping[str, InputSpec]
    outputs: Mapping[str, str]
    parameters: Mapping[str, Any]
    expansions: tuple[Expansion, ...]
    resource_class: str
    memory_gb: int | None
    digest: str
    warnings: tuple[str, ...] = field(default=())

    def combinations(self) -> tuple[tuple[dict[str, Any], dict[str, str]], ...]:
        """Return (parameters, labels) for every expanded parameter combination."""
        result = []
        for choice in itertools.product(*(range(len(item.values)) for item in self.expansions)):
            parameters = dict(self.parameters)
            labels = {}
            for expansion, index in zip(self.expansions, choice):
                parameters[expansion.name] = expansion.values[index]
                labels[expansion.name] = expansion.labels[index]
            result.append((parameters, labels))
        return tuple(result)


def _placeholders(pattern: str, *, where: str) -> set[str]:
    try:
        fields = [
            name
            for _, name, spec, conversion in string.Formatter().parse(pattern)
            if name is not None
        ]
    except ValueError as error:
        raise ManifestError(f"{where} has an invalid pattern {pattern!r}: {error}") from error
    for name in fields:
        if not _NAME.match(name):
            raise ManifestError(f"{where} uses an invalid placeholder {{{name}}}")
    return set(fields)


def _expansion(name: str, value: object) -> Expansion | None:
    if not (isinstance(value, dict) and set(value) == {"expand"}):
        return None
    items = value["expand"]
    if isinstance(items, dict):
        if not items:
            raise ManifestError(f"parameters.{name}.expand cannot be empty")
        labels = tuple(str(label) for label in items)
        for label in labels:
            if not re.fullmatch(r"[A-Za-z0-9]+", label):
                raise ManifestError(
                    f"parameters.{name}.expand label {label!r} must be alphanumeric"
                )
        return Expansion(name, labels, tuple(items.values()))
    if not isinstance(items, list) or not items:
        raise ManifestError(f"parameters.{name}.expand must be a non-empty list or mapping")
    for item in items:
        if not _is_scalar(item):
            raise ManifestError(
                f"parameters.{name}.expand contains the non-scalar value {item!r}; "
                "give labels with the mapping form {expand: {LABEL: VALUE}}"
            )
    labels = tuple(bids_label(item) for item in items)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ManifestError(
            f"parameters.{name}.expand values share the BIDS labels: {', '.join(duplicates)}"
        )
    return Expansion(name, labels, tuple(items))


def _input(name: str, value: object, *, parameters: Mapping[str, Any]) -> InputSpec:
    where = f"inputs.{name}"
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be a mapping")
    unknown = sorted(set(value) - _INPUT_KEYS)
    if unknown:
        raise ManifestError(f"{where} has unknown keys: {', '.join(unknown)}")
    source = value.get("from")
    if not isinstance(source, str) or not source:
        raise ManifestError(f"{where}.from is required")
    if source.startswith("analysis:"):
        raise ManifestError(f"{where}: inputs from other analyses are not supported yet")
    if source != "raw":
        from nro.modules import MODULE_NAMES

        if source not in MODULE_NAMES:
            raise ManifestError(
                f"{where}.from names unknown module {source!r}; choose raw or one of "
                + ", ".join(MODULE_NAMES)
            )
    if "lineage" in value:
        raise ManifestError(f"{where}: name the upstream with workflow or config, not lineage")
    workflow = value.get("workflow")
    config = value.get("config")
    if workflow is not None and config is not None:
        raise ManifestError(f"{where} gives both workflow and config; give at most one")
    if source == "raw":
        if workflow is not None or config is not None:
            raise ManifestError(f"{where}: from: raw takes neither workflow nor config")
    else:
        if value.get("path") is not None:
            raise ManifestError(f"{where}.path is only valid with from: raw")
        if workflow is None and config is None:
            workflow = "main"
    for key, reference in (("workflow", workflow), ("config", config)):
        if reference is None:
            continue
        if not isinstance(reference, str) or not reference:
            raise ManifestError(f"{where}.{key} must be a non-empty string")
        for placeholder in _placeholders(reference, where=f"{where}.{key}"):
            if placeholder not in parameters:
                raise ManifestError(f"{where}.{key} references unknown parameter {{{placeholder}}}")
    select = value.get("select", {})
    if not isinstance(select, dict):
        raise ManifestError(f"{where}.select must be a mapping of entity to value")
    for key, item in select.items():
        if not isinstance(key, str) or not key:
            raise ManifestError(f"{where}.select has an invalid entity {key!r}")
        allowed = (
            item is None
            or _is_scalar(item)
            or (isinstance(item, list) and all(_is_scalar(entry) for entry in item))
        )
        if not allowed:
            raise ManifestError(f"{where}.select.{key} must be a scalar, list, or null")
    many = value.get("many", False)
    optional = value.get("optional", False)
    if not isinstance(many, bool) or not isinstance(optional, bool):
        raise ManifestError(f"{where}.many and {where}.optional must be booleans")
    minimum = value.get("min", 1)
    maximum = value.get("max")
    if not many and ("min" in value or "max" in value):
        raise ManifestError(f"{where}: min and max require many: true")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        raise ManifestError(f"{where}.min must be a non-negative integer")
    if maximum is not None and (
        not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < max(minimum, 1)
    ):
        raise ManifestError(f"{where}.max must be an integer at least min and 1")
    path = value.get("path")
    if path is not None and (not isinstance(path, str) or path.startswith("/") or ".." in path):
        raise ManifestError(f"{where}.path must be a relative pattern inside the project")
    return InputSpec(
        name=name,
        source=source,
        workflow=workflow,
        config=config,
        select=dict(select),
        many=many,
        minimum=minimum,
        maximum=maximum,
        optional=optional,
        path=path,
    )


def parse_manifest(text: str, *, analysis: str) -> Manifest:
    """Validate manifest text for one analysis and return its typed form."""
    if not _ANALYSIS_NAME.match(analysis):
        raise ManifestError(
            f"Analysis name {analysis!r} must start with a letter and use lowercase letters, "
            "digits, and underscores"
        )
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ManifestError(f"analysis.yml is not valid YAML: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError("analysis.yml must be a mapping")
    unknown = sorted(set(value) - _MANIFEST_KEYS)
    if unknown:
        raise ManifestError(f"analysis.yml has unknown keys: {', '.join(unknown)}")
    version = value.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ManifestError("version is required and must be an integer")
    entry = value.get("entry", "run")
    if not isinstance(entry, str) or not entry.isidentifier() or keyword.iskeyword(entry):
        raise ManifestError(f"entry must be a Python function name, not {entry!r}")
    environment = value.get("environment") or {}
    if not isinstance(environment, dict) or set(environment) - {"container"}:
        raise ManifestError("environment accepts only container")
    container = environment.get("container")
    if container is not None and (not isinstance(container, str) or not container):
        raise ManifestError("environment.container must be an image path")
    scope = value.get("scope")
    if scope not in SCOPES:
        raise ManifestError(f"scope must be one of {', '.join(SCOPES)}")
    if scope not in SUPPORTED_SCOPES:
        raise ManifestError(f"scope {scope!r} is not supported yet")

    raw_parameters = value.get("parameters") or {}
    if not isinstance(raw_parameters, dict):
        raise ManifestError("parameters must be a mapping")
    parameters: dict[str, Any] = {}
    expansions = []
    warnings = []
    for name, item in raw_parameters.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ManifestError(f"Parameter name {name!r} must be an identifier")
        if name in _RESERVED_PLACEHOLDERS:
            raise ManifestError(f"Parameter name {name!r} is reserved for work-item entities")
        literal = item.get("expand") if isinstance(item, dict) and set(item) == {"expand"} else None
        if (
            isinstance(literal, list)
            and len(literal) == 1
            and isinstance(literal[0], dict)
            and set(literal[0]) == {"expand"}
        ):
            parameters[name] = literal[0]
            warnings.append(
                f"parameters.{name} passes the literal mapping {{expand: ...}}, not an expansion"
            )
            continue
        expansion = _expansion(name, item)
        if expansion is None:
            parameters[name] = item
        else:
            expansions.append(expansion)
            parameters[name] = None

    raw_inputs = value.get("inputs") or {}
    if not isinstance(raw_inputs, dict):
        raise ManifestError("inputs must be a mapping")
    inputs = {}
    for name, item in raw_inputs.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ManifestError(f"Input name {name!r} must be an identifier")
        inputs[name] = _input(name, item, parameters=parameters)

    raw_outputs = value.get("outputs")
    if not isinstance(raw_outputs, dict) or not raw_outputs:
        raise ManifestError("outputs must map at least one slot name to a filename pattern")
    expanded = {item.name for item in expansions}
    outputs = {}
    filenames = set()
    for slot, pattern in raw_outputs.items():
        where = f"outputs.{slot}"
        if not isinstance(slot, str) or not _NAME.match(slot):
            raise ManifestError(f"Output slot {slot!r} must be an identifier")
        if not isinstance(pattern, str) or not pattern or "/" in pattern or pattern.startswith("."):
            raise ManifestError(f"{where} must be a filename pattern without directories")
        placeholders = _placeholders(pattern, where=where)
        for name in placeholders:
            if name in _RESERVED_PLACEHOLDERS or name in _RUN_PLACEHOLDERS:
                if scope == "subject" and name not in {"project", "participant"}:
                    raise ManifestError(f"{where} uses run placeholder {{{name}}} in subject scope")
                continue
            if name not in parameters:
                raise ManifestError(f"{where} references unknown placeholder {{{name}}}")
            if name not in expanded and not _is_scalar(parameters[name]):
                raise ManifestError(f"{where} uses non-scalar parameter {{{name}}}")
        missing = sorted(expanded - placeholders)
        if missing:
            raise ManifestError(
                f"{where} must include every expanded parameter; missing "
                + ", ".join(f"{{{name}}}" for name in missing)
            )
        if scope == "subject" and "participant" not in placeholders:
            raise ManifestError(f"{where} must include {{participant}}")
        if scope == "run" and "stem" not in placeholders:
            raise ManifestError(f"{where} must include {{stem}}, the run's BIDS stem")
        if pattern in filenames:
            raise ManifestError(f"{where} repeats the pattern of another slot")
        filenames.add(pattern)
        outputs[slot] = pattern

    resources = value.get("resources") or {}
    if not isinstance(resources, dict) or set(resources) - {"class", "memory_gb"}:
        raise ManifestError("resources accepts only class and memory_gb")
    resource_class = resources.get("class", "small")
    if resource_class not in RESOURCE_CLASSES:
        raise ManifestError(f"resources.class must be one of {', '.join(RESOURCE_CLASSES)}")
    memory_gb = resources.get("memory_gb")
    if memory_gb is not None and (not isinstance(memory_gb, int) or memory_gb < 1):
        raise ManifestError("resources.memory_gb must be a positive integer")

    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return Manifest(
        analysis=analysis,
        version=version,
        entry=entry,
        container=container,
        scope=scope,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        expansions=tuple(expansions),
        resource_class=resource_class,
        memory_gb=memory_gb,
        digest=digest,
        warnings=tuple(warnings),
    )


_RUN_PLACEHOLDERS = {"task", "run", "acq", "ses", "dir", "echo", "ce", "rec"}


def check_entry(source: str, entry: str) -> None:
    """Check that ``analysis.py`` defines the entry function without importing it."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ManifestError(f"analysis.py has a syntax error: {error}") from error
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            if isinstance(node, ast.AsyncFunctionDef):
                raise ManifestError(f"analysis.py entry {entry!r} must not be async")
            arguments = node.args
            positional = len(arguments.posonlyargs) + len(arguments.args)
            if positional < 3 and arguments.vararg is None:
                raise ManifestError(
                    f"analysis.py entry {entry!r} must accept (inputs, params, outputs)"
                )
            return
    raise ManifestError(f"analysis.py does not define a top-level function {entry!r}")
