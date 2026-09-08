"""Validation and event sampling for compiled BIDS Stats Models."""

import fnmatch
import re
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd

from .transforms import event_variables, validate_transform


def _keys(value: dict, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unsupported {label} fields: {sorted(unknown)}")


def contrast_weights(contrast: dict) -> np.ndarray:
    """Parse finite linear t/pass weights, including fractional JSON strings."""
    try:
        values = np.array([float(Fraction(str(v))) for v in contrast["Weights"]])
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError("Contrast weights must be a vector of finite numbers or fractions") from error
    if len(values) != len(contrast["ConditionList"]) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Contrast weights must match a nonempty ConditionList")
    if not np.any(values):
        raise ValueError("An all-zero contrast is not a statistical effect")
    return values


def validate_model(document: dict, *, task: str) -> dict:
    """Validate supported Stats Models semantics and return a normalized copy.

    Run GLMs and within-subject meta nodes are supported. Unsupported features
    fail explicitly; this is not a validator for every legal Stats Models file.
    """
    result = deepcopy(document)
    _keys(result, {"Name", "BIDSModelVersion", "Description", "Input", "Nodes", "Edges"}, "model")
    if result.get("BIDSModelVersion") != "1.0.0" or not isinstance(result.get("Name"), str):
        raise ValueError("Provide Name and BIDSModelVersion=1.0.0")
    _keys(result.get("Input"), {"task", "subject", "session", "run", "acquisition", "direction"}, "Input")
    selected_tasks = result["Input"].get("task")
    if selected_tasks not in (task, [task]):
        raise ValueError("Model Input.task must select its registered task exactly")
    for value in result["Input"].values():
        if not isinstance(value, (str, list)) or (isinstance(value, list) and any(not isinstance(v, str) for v in value)):
            raise ValueError("Input selectors must be strings or lists of strings")
    nodes = result.get("Nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("A model needs Nodes")
    names, levels = [], {}
    for node in nodes:
        _keys(node, {"Name", "Level", "GroupBy", "Transformations", "Model", "Contrasts", "DummyContrasts"}, "node")
        name, level = node.get("Name"), node.get("Level")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in names:
            raise ValueError("Node names must be unique path-safe identifiers")
        if level not in {"Run", "Session", "Subject"}:
            raise ValueError("Only Run, Session and Subject levels are supported; no Dataset analyses")
        grouping = node.get("GroupBy")
        allowed_groups = {"subject", "session", "run", "task", "contrast", "direction", "acquisition"}
        if not isinstance(grouping, list) or not set(grouping) <= allowed_groups or "subject" not in grouping:
            raise ValueError("GroupBy must contain subject and only supported BIDS grouping fields")
        if level == "Session" and "session" not in grouping:
            raise ValueError("Session nodes must group by session")
        if level == "Run" and "contrast" in grouping:
            raise ValueError("Run nodes cannot group input images by contrast")
        model = node.get("Model")
        _keys(model, {"Type", "X", "HRF", "Software"}, "Model")
        expected = "glm" if level == "Run" else "meta"
        if model.get("Type") != expected:
            raise ValueError(f"{level} requires Model.Type={expected}; summary-level GLMs are forbidden")
        x = model.get("X")
        if not isinstance(x, list) or not x or any(v != 1 and not isinstance(v, str) for v in x):
            raise ValueError("Model.X must list predictor names/globs or intercept 1")
        if level != "Run" and "HRF" in model:
            raise ValueError("HRFs apply only to run nodes")
        if "HRF" in model:
            _keys(model["HRF"], {"Variables", "Model"}, "HRF")
            if model["HRF"].get("Model") not in {"spm", "glover"}:
                raise ValueError("Supported HRFs: spm, glover")
            variables = model["HRF"].get("Variables")
            if not isinstance(variables, list) or any(not isinstance(v, str) for v in variables):
                raise ValueError("HRF.Variables must list variable names")
        transforms = node.get("Transformations")
        if transforms:
            _keys(transforms, {"Transformer", "Instructions"}, "Transformations")
            if level != "Run" or transforms.get("Transformer") != "pybids-transforms-v1":
                raise ValueError("Only run-level pybids-transforms-v1 instructions are supported")
            for instruction in transforms.get("Instructions", []):
                validate_transform(instruction)
        contrast_names = []
        for contrast in node.get("Contrasts", []):
            _keys(contrast, {"Name", "ConditionList", "Weights", "Test"}, "contrast")
            cname = contrast.get("Name")
            if not isinstance(cname, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cname) or cname in contrast_names:
                raise ValueError("Contrast names must be unique path-safe identifiers")
            if contrast.get("Test") not in {"t", "pass"}:
                raise ValueError("Only t and pass contrasts are supported")
            conditions = contrast.get("ConditionList")
            if not isinstance(conditions, list) or any(v != 1 and not isinstance(v, str) for v in conditions):
                raise ValueError("ConditionList must contain names or intercept 1")
            contrast_weights(contrast)
            contrast_names.append(cname)
        dummy = node.get("DummyContrasts")
        if dummy:
            _keys(dummy, {"Test", "Contrasts"}, "DummyContrasts")
            if dummy.get("Test") not in {"t", "pass"}:
                raise ValueError("DummyContrasts.Test must be t or pass")
            if "Contrasts" in dummy and (not isinstance(dummy["Contrasts"], list) or any(v != 1 and not isinstance(v, str) for v in dummy["Contrasts"])):
                raise ValueError("DummyContrasts.Contrasts must list names or intercept 1")
        if not node.get("Contrasts") and not dummy:
            raise ValueError("Every node must declare contrasts")
        names.append(name)
        levels[name] = level
    if sum(level == "Run" for level in levels.values()) != 1 or nodes[0]["Level"] != "Run":
        raise ValueError("This execution profile requires one root Run node")
    edges = result.setdefault("Edges", [{"Source": a, "Destination": b} for a, b in zip(names, names[1:])])
    parents = {}
    for edge in edges:
        _keys(edge, {"Source", "Destination", "Filter"}, "edge")
        source, dest = edge.get("Source"), edge.get("Destination")
        if source not in names or dest not in names or names.index(source) >= names.index(dest):
            raise ValueError("Edges must connect nodes in topological order")
        if {"Run": 0, "Session": 1, "Subject": 2}[levels[source]] >= {"Run": 0, "Session": 1, "Subject": 2}[levels[dest]]:
            raise ValueError("Edges must increase analysis level")
        if dest in parents:
            raise ValueError("Multiple-parent meta nodes are not supported")
        if "Filter" in edge:
            _keys(edge["Filter"], {"contrast", "subject", "session", "run", "task"}, "edge Filter")
            if any(not isinstance(v, list) for v in edge["Filter"].values()):
                raise ValueError("Edge filter values must be lists")
        parents[dest] = source
    if set(parents) != set(names[1:]):
        raise ValueError("Every meta node needs an incoming edge")
    return result


def validate_run_groups(runs, node: dict, participant: str) -> None:
    """Reject Run GroupBy cells containing multiple images; runs are never concatenated."""
    groups = set()
    aliases = {"subject": "sub", "session": "ses", "direction": "dir", "acquisition": "acq"}
    for run in runs:
        entities = {**run.entities, "sub": participant}
        group = tuple(entities.get(aliases.get(key, key)) for key in node["GroupBy"])
        if group in groups:
            raise ValueError("Run GroupBy must uniquely identify each BOLD run; add session/run/direction as needed")
        groups.add(group)


def expand_columns(patterns: list, available: list[str], *, factor_variables: tuple[str, ...] = ()) -> list[str]:
    """Resolve Model.X globs in order, allowing absent event levels but not typos."""
    result = []
    for pattern in patterns:
        pattern = "intercept" if pattern == 1 else pattern
        matched = [name for name in available if fnmatch.fnmatchcase(name, pattern)]
        if not matched and not any(pattern.startswith(f"{name}.") for name in factor_variables):
            raise ValueError(f"Predictor does not match available variables: {pattern}")
        result.extend(name for name in matched if name not in result)
    return result


def event_design(node: dict, events: pd.DataFrame, confounds: pd.DataFrame, tr: float) -> pd.DataFrame:
    """Construct declared event predictors at full acquisition times before censoring.

    Factor expands categorical events; Convolve or Model.HRF explicitly requests
    a supported canonical HRF. Confound variables are never convolved implicitly.
    """
    from nilearn.glm.first_level.hemodynamic_models import compute_regressor

    if not {"onset", "duration"} <= set(events):
        raise ValueError("Events need onset and duration columns")
    if not np.isfinite(events[["onset", "duration"]].to_numpy(float)).all() or (events.duration < 0).any():
        raise ValueError("Events need finite onsets and nonnegative durations")
    sparse, factor_variables, convolutions = event_variables(events, node.get("Transformations", {}))
    hrf = node["Model"].get("HRF")
    if hrf:
        for pattern in hrf["Variables"]:
            if not any(fnmatch.fnmatchcase(name, pattern) for name in sparse) and not any(pattern.startswith(f"{name}.") for name in factor_variables):
                raise ValueError(f"HRF variable does not match an event variable: {pattern}")
            for name in sparse:
                if fnmatch.fnmatchcase(name, pattern):
                    if name in convolutions:
                        raise ValueError(f"HRF declared twice for {name}")
                    convolutions[name] = hrf["Model"]
    times = np.arange(len(confounds)) * tr
    values = confounds.copy()
    values["intercept"] = 1.0
    for name, amplitudes in sparse.items():
        if name in values:
            raise ValueError(f"Event/confound variable collision: {name}")
        if name in convolutions:
            sampled, _ = compute_regressor(
                np.array([events.onset, events.duration, amplitudes]),
                convolutions[name], times, con_id=name,
            )
            values[name] = sampled[:, 0]
        else:
            sampled = np.zeros(len(times))
            for onset, duration, amplitude in zip(events.onset, events.duration, amplitudes):
                if duration == 0:
                    index = int(np.searchsorted(times, onset))
                    if 0 <= index < len(times):
                        sampled[index] += amplitude
                else:
                    sampled[(times >= onset) & (times < onset + duration)] += amplitude
            values[name] = sampled
    columns = expand_columns(node["Model"]["X"], list(values), factor_variables=tuple(factor_variables))
    return values[columns].astype(float)
