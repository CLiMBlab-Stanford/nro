"""Ordered transformations of event amplitudes before temporal sampling."""

import fnmatch
from copy import deepcopy

import numpy as np
import pandas as pd


def canonical_transform(instruction: dict) -> dict:
    """Make equivalent transform syntax explicit without changing operation order."""
    validate_transform(instruction)
    result = deepcopy(instruction)
    if isinstance(result["Input"], str):
        result["Input"] = [result["Input"]]
    name = result["Name"]
    if name == "Factor":
        result.setdefault("Constraint", "none")
    elif name == "Convolve":
        result.setdefault("Model", "spm")
    elif name == "Scale":
        result.setdefault("Demean", True)
        result.setdefault("Rescale", True)
    if name in {"Demean", "Scale"}:
        result.setdefault("Groupby", [])
    if name == "Sum" and "Weights" in result:
        result["Weights"] = [float(value) for value in result["Weights"]]
    return result


def validate_transform(instruction: dict) -> None:
    """Reject unsupported transformation names, options and argument shapes."""
    options = {"Factor": {"Constraint", "RefLevel"}, "Convolve": {"Model"},
               "Scale": {"Demean", "Rescale", "Output", "Groupby"},
               "Demean": {"Output", "Groupby"},
               "Sum": {"Output", "Weights"}, "Product": {"Output"}, "Select": set()}
    if not isinstance(instruction, dict) or instruction.get("Name") not in options:
        raise ValueError("Supported transformations: Factor, Demean, Scale, Sum, Product, Select, Convolve")
    name = instruction["Name"]
    if set(instruction) - ({"Name", "Input"} | options[name]):
        raise ValueError(f"Unsupported {name} transformation options")
    inputs = instruction.get("Input")
    inputs = [inputs] if isinstance(inputs, str) else inputs
    if not isinstance(inputs, list) or not inputs or any(not isinstance(v, str) or not v for v in inputs):
        raise ValueError("Transformation Input must name event variables")
    if name == "Convolve" and instruction.get("Model", "spm") not in {"spm", "glover"}:
        raise ValueError("Supported HRFs: spm, glover")
    if name == "Factor":
        constraint = instruction.get("Constraint", "none")
        if constraint not in {"none", "drop_one"}:
            raise ValueError("Factor Constraint must be none or drop_one")
        if (constraint == "drop_one") != ("RefLevel" in instruction):
            raise ValueError("Factor drop_one requires an explicit RefLevel; none forbids it")
        if "RefLevel" in instruction and not isinstance(instruction["RefLevel"], (str, int, float)):
            raise ValueError("Factor RefLevel must be a categorical value")
    if name in {"Sum", "Product"} and not isinstance(instruction.get("Output"), str):
        raise ValueError(f"{name} requires one Output name")
    if "Output" in instruction and (not isinstance(instruction["Output"], str) or not instruction["Output"]):
        raise ValueError("Output must be one nonempty variable name")
    if name in {"Scale", "Demean"}:
        if "Output" in instruction and len(inputs) != 1:
            raise ValueError("Scale with Output requires one input")
        for flag in ("Demean", "Rescale"):
            if flag in instruction and not isinstance(instruction[flag], bool):
                raise ValueError(f"Scale {flag} must be boolean")
        if "Groupby" in instruction and (not isinstance(instruction["Groupby"], list) or
                                         any(not isinstance(v, str) for v in instruction["Groupby"])):
            raise ValueError("Transformation Groupby must list event columns")
    if "Weights" in instruction:
        weights = np.asarray(instruction["Weights"], dtype=float)
        if weights.shape != (len(inputs),) or not np.isfinite(weights).all():
            raise ValueError("Sum Weights must match inputs and be finite")


def event_variables(events: pd.DataFrame, transformations: dict) -> tuple[dict, list, dict]:
    """Return sparse amplitudes, factor names and explicit HRFs.

    Arithmetic operates on event rows before convolution. Scale uses sample
    standard deviations within the run or the specified event groups.
    """
    sparse = {name: events[name].to_numpy(float) for name in events
              if name not in {"onset", "duration"} and pd.api.types.is_numeric_dtype(events[name])}
    factors, convolutions = [], {}
    for instruction in transformations.get("Instructions", []):
        validate_transform(instruction)
        operation = instruction["Name"]
        patterns = instruction["Input"]
        patterns = [patterns] if isinstance(patterns, str) else patterns
        if operation == "Factor":
            if convolutions:
                raise ValueError("Event transformations must precede convolution")
            for column in patterns:
                if column not in events or column in {"onset", "duration"}:
                    raise ValueError(f"Missing categorical event variable: {column}")
                factors.append(column)
                sparse.pop(column, None)
                for value in sorted(events[column].dropna().unique(), key=str):
                    if instruction.get("Constraint") == "drop_one" and value == instruction["RefLevel"]:
                        continue
                    name = f"{column}.{value}"
                    if name in sparse:
                        raise ValueError(f"Factor output already exists: {name}")
                    sparse[name] = (events[column] == value).to_numpy(float)
            continue
        names = []
        for pattern in patterns:
            matches = [name for name in sparse if fnmatch.fnmatchcase(name, pattern)]
            if not matches and not any(pattern.startswith(f"{f}.") for f in factors):
                raise ValueError(f"Transformation input does not match an event variable: {pattern}")
            if not matches and not any(c in pattern for c in "*?[") and operation not in {"Convolve", "Select"}:
                sparse[pattern] = np.zeros(len(events))
                matches = [pattern]
            names.extend(matches)
        if operation == "Convolve":
            for name in names:
                if name in convolutions:
                    raise ValueError(f"Repeated HRF convolution: {name}")
                convolutions[name] = instruction.get("Model", "spm")
            continue
        if convolutions:
            raise ValueError("Event transformations must precede convolution")
        if operation == "Select":
            sparse = {name: sparse[name] for name in names}
            continue
        if not names:
            continue
        if operation in {"Scale", "Demean"} and "Output" in instruction and len(names) != 1:
            raise ValueError("An explicit Output requires exactly one expanded input")
        if any(not np.isfinite(sparse[name]).all() for name in names):
            raise ValueError("Transformation inputs contain nonfinite amplitudes")
        if operation in {"Scale", "Demean"}:
            groups = instruction.get("Groupby", [])
            if any(group not in events for group in groups):
                raise ValueError("Transformation Groupby must refer to event columns")
            indices = list(events.groupby(groups, dropna=False).indices.values()) if groups else [np.arange(len(events))]
            for name in names:
                values = sparse[name].copy()
                for rows in indices:
                    source = sparse[name][rows]
                    if operation == "Scale" and len(np.unique(source)) < 2:
                        raise ValueError(f"Cannot Scale a constant event group: {name}; use Demean for centering only")
                    transformed = source - source.mean() if instruction.get("Demean", True) else source.copy()
                    if operation == "Scale" and instruction.get("Rescale", True):
                        scale = source.std(ddof=1) if len(source) > 1 else 0
                        transformed = transformed / scale
                    values[rows] = transformed
                output = instruction.get("Output", name)
                if output != name and output in sparse:
                    raise ValueError(f"Transformation output already exists: {output}")
                sparse[output] = values
        else:
            output = instruction["Output"]
            if output in sparse:
                raise ValueError(f"Transformation output already exists: {output}")
            values = np.stack([sparse[name] for name in names])
            if operation == "Product":
                sparse[output] = values.prod(axis=0)
            else:
                weights = np.asarray(instruction.get("Weights", np.ones(len(names))), dtype=float)
                if weights.shape != (len(names),):
                    raise ValueError("Sum Weights must match expanded inputs")
                sparse[output] = weights @ values
    return sparse, factors, convolutions
