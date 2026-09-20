"""YAML task definitions and execution-only model-set membership."""

import json
import os
import re
import tempfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml

from nro.configuration.parsing import parse_mapping
from nro.configuration.site import definitions_roots
from nro.engine.io import atomic_write_text


def model_path(identifier: str, root: Path | None = None) -> Path:
    """Resolve a registered TASK/VARIANT identifier to its YAML definition."""
    parts = identifier.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", p) for p in parts):
        raise ValueError("Model ID must be TASK/VARIANT using letters, digits and hyphens")
    directories = (
        (Path(root),)
        if root is not None
        else tuple(store / "models" for store in definitions_roots())
    )
    paths = [directory / parts[0] / f"{parts[1]}.yml" for directory in directories]
    for directory, path in zip(directories, paths, strict=True):
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Model escapes the model directory: {path}")
        if path.is_file():
            return path
    return paths[0]


def validate_task_model(value: dict) -> dict:
    """Validate task syntax without reading data or permitting nuisance settings."""
    from .compiler import task_node

    if not isinstance(value, dict):
        raise ValueError("Task model must be a YAML mapping")
    result = deepcopy(value)
    allowed = {
        "description",
        "model_set",
        "conditions",
        "predictors",
        "hrf",
        "hrf_overrides",
        "transformations",
        "contrasts",
        "statsmodels",
    }
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"Unsupported task-model fields: {sorted(unknown)}")
    result["model_set"] = list(model_memberships(result))
    if "description" in result and not isinstance(result["description"], str):
        raise ValueError("description must be text")
    task_node(result)
    return result


def model_memberships(model: dict) -> tuple[str, ...]:
    """Read validated execution membership without interpreting scientific fields."""
    if not isinstance(model, dict):
        raise ValueError("Task model must be a YAML mapping")
    sets = model.get("model_set", [])
    sets = [sets] if isinstance(sets, str) else sets
    if not isinstance(sets, list) or any(
        not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v) for v in sets
    ):
        raise ValueError("model_set must be a name or list of names")
    return tuple(dict.fromkeys(sets))


def load_task_model(identifier: str, root: Path | None = None) -> dict:
    """Load and validate one registered task model, including set membership."""
    path = model_path(identifier, root)
    return validate_task_model(parse_mapping(path.read_text(), source=str(path)))


def scientific_model(model: dict) -> dict:
    """Return a canonical task program, excluding execution metadata.

    Compile authoring shorthand and fill defaults without reading data. Ordered
    predictor and transformation lists remain ordered. A bounded content cache
    avoids recompiling repeated snapshots; callers receive independent copies.
    """
    if not isinstance(model, dict):
        raise ValueError("Task model must be a YAML mapping")
    scientific = {
        key: value for key, value in model.items() if key not in {"model_set", "description"}
    }
    return deepcopy(_canonical_model(json.dumps(scientific, sort_keys=True)))


@lru_cache(maxsize=256)
def _canonical_model(serialized: str) -> dict:
    from .compiler import canonical_model_document, task_node

    source = validate_task_model(json.loads(serialized))
    node = canonical_model_document({"Nodes": [task_node(source)]})["Nodes"][0]
    transformations = node["Transformations"]
    rules = node["Model"]["Software"]["nro"]
    return {
        "predictors": rules["event_predictors"],
        "hrf": rules["default_hrf"],
        "hrf_overrides": rules["hrf_overrides"],
        "statsmodels": {"Transformations": transformations, "Contrasts": node["Contrasts"]},
    }


def canonical_processing(processing: dict) -> dict:
    """Normalize a recorded processing contract without reading the live model."""
    result = deepcopy(processing)
    if "task_model" in result:
        result["task_model"] = scientific_model(result["task_model"])
    return result


def model_contract(entities: dict) -> dict:
    """Return the current scientific definition for one work item's contract."""
    identifier = f"{entities['task']}/{entities['model']}"
    try:
        path = model_path(identifier)
        return {"task_model": scientific_model(parse_mapping(path.read_text(), source=str(path)))}
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        return {"unavailable_model": f"{identifier}: {error}"}


def _selected_models(
    *, tasks=(), models=(), model_sets=None, root: Path | None = None
) -> dict[str, dict]:
    roots = (
        (Path(root),)
        if root is not None
        else tuple(store / "models" for store in definitions_roots())
    )
    if not any(directory.is_dir() for directory in roots):
        raise ValueError(
            "Model directory does not exist in the definitions chain: " + ", ".join(map(str, roots))
        )
    sets = ("main",) if model_sets is None and not models else tuple(model_sets or ())
    selected = {}
    seen = set()
    for directory in roots:
        for path in sorted(directory.glob("*/*.yml")):
            identifier = f"{path.parent.name}/{path.stem}"
            if identifier in seen:
                continue
            seen.add(identifier)
            model_path(identifier, directory)
            if tasks and path.parent.name not in tasks:
                continue
            if models and path.stem not in models and identifier not in models:
                continue
            model = parse_mapping(path.read_text(), source=str(path))
            memberships = model_memberships(model)
            if sets and not set(sets).intersection(memberships):
                continue
            selected[identifier] = model
    return selected


def select_models(
    *, tasks=(), models=(), model_sets=None, root: Path | None = None
) -> dict[str, dict]:
    """Select and validate scientific models for demand planning.

    None defaults model_sets to main unless models are explicit. Empty
    model_sets selects all sets. Values within selectors are alternatives;
    different selectors intersect. Models accept variants or TASK/VARIANT IDs.
    """
    return {
        identifier: validate_task_model(model)
        for identifier, model in _selected_models(
            tasks=tasks, models=models, model_sets=model_sets, root=root
        ).items()
    }


def model_ids_in_sets(sets: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve set filters without validating scientific models during inspection."""
    return tuple(_selected_models(model_sets=sets))


def register_model(identifier: str, source: Path, *, root: Path | None = None) -> Path:
    """Validate and atomically register YAML; an existing identifier is an error."""
    from nro.configuration.site import definition_write

    destination = model_path(identifier, root)
    model = validate_task_model(parse_mapping(Path(source).read_text(), source=str(source)))
    with definition_write(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".register-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "model.yml"
            atomic_write_text(staged, yaml.safe_dump(model, sort_keys=False))
            os.link(staged, destination)
    return destination
