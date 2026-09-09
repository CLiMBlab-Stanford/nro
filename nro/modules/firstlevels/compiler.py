"""Compile task intent and module-owned denoising into executable Stats Models."""

import fnmatch
import re
from copy import deepcopy

from .models import contrast_weights, validate_model

NUISANCE_KEYS = (
    "confounds_regex",
    "temporal_mask_regex",
    "nuisance_variance_explained",
    "minimum_temporal_rank",
    "minimum_temporal_rank_fraction",
    "noise_model",
    "ar_grid",
)


def canonical_model_document(document: dict) -> dict:
    """Normalize compiled syntax while preserving nodes, columns and aliases."""
    from .transforms import canonical_transform

    result = deepcopy(document)
    for node in result["Nodes"]:
        if "Transformations" in node:
            transforms = node["Transformations"]
            transforms.setdefault("Transformer", "pybids-transforms-v1")
            transforms["Instructions"] = [
                canonical_transform(item) for item in transforms["Instructions"]
            ]
        for contrast in node.get("Contrasts", []):
            contrast["Weights"] = [float(value) for value in contrast_weights(contrast)]
    return result


def task_node(source: dict) -> dict:
    """Compile the task-only run node, rejecting conflicting advanced controls."""
    advanced = source.get("statsmodels", {})
    if not isinstance(advanced, dict) or set(advanced) - {"Transformations", "Contrasts"}:
        raise ValueError("statsmodels accepts only Transformations and Contrasts blocks")
    if not isinstance(source.get("transformations", []), list):
        raise ValueError("transformations must be an ordered instruction list")
    if "conditions" in source and "predictors" in source:
        raise ValueError("Use conditions or predictors, not both")
    conditions = source.get("conditions")
    if conditions is not None and (not isinstance(conditions, str) or not conditions):
        raise ValueError("conditions must name a categorical events column")
    predictors = source.get("predictors", [f"{conditions}.*"] if conditions else [])
    if (
        not isinstance(predictors, list)
        or not predictors
        or any(not isinstance(v, str) or not v for v in predictors)
    ):
        raise ValueError("Provide conditions or a nonempty list of event predictors")
    hrf = source.get("hrf", "spm")
    if hrf not in {"spm", "glover", None}:
        raise ValueError("hrf must be spm, glover or null")
    overrides = source.get("hrf_overrides", {})
    if not isinstance(overrides, dict) or any(
        not isinstance(k, str) or v not in {"spm", "glover", None} for k, v in overrides.items()
    ):
        raise ValueError("hrf_overrides maps event predictors to spm, glover or null")
    if "Transformations" in advanced and ("transformations" in source or conditions):
        raise ValueError(
            "Advanced Transformations cannot be combined with conditions or transformations"
        )
    transforms = advanced.get(
        "Transformations",
        {
            "Transformer": "pybids-transforms-v1",
            "Instructions": ([{"Name": "Factor", "Input": conditions}] if conditions else [])
            + source.get("transformations", []),
        },
    )
    if "Contrasts" in advanced and "contrasts" in source:
        raise ValueError("Use contrasts or statsmodels.Contrasts, not both")
    if "Contrasts" in advanced:
        contrasts = deepcopy(advanced["Contrasts"])
        if not isinstance(contrasts, list) or not contrasts:
            raise ValueError("statsmodels.Contrasts must be a nonempty list")
    else:
        declared = source.get("contrasts", {})
        if not isinstance(declared, dict) or not declared:
            raise ValueError("contrasts must map names to predictor-weight mappings")
        contrasts = []
        for name, weights in declared.items():
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"Contrast {name} needs predictor weights")
            if any(not isinstance(v, str) for v in weights):
                raise ValueError("Contrast keys must name event predictors")
            names = [
                f"{conditions}.{v}" if conditions and not v.startswith(f"{conditions}.") else v
                for v in weights
            ]
            contrasts.append(
                {
                    "Name": name,
                    "ConditionList": names,
                    "Weights": list(weights.values()),
                    "Test": "t",
                }
            )
    node = {
        "Name": "run",
        "Level": "Run",
        "GroupBy": ["subject", "session", "task", "run", "direction", "acquisition"],
        "Transformations": deepcopy(transforms),
        "Model": {
            "Type": "glm",
            "X": [*predictors, 1],
            "Software": {
                "nro": {
                    "event_predictors": predictors,
                    "default_hrf": hrf,
                    "hrf_overrides": overrides,
                }
            },
        },
        "Contrasts": contrasts,
    }
    validate_model(
        {"Name": "task", "BIDSModelVersion": "1.0.0", "Input": {"task": "task"}, "Nodes": [node]},
        task="task",
    )
    if any(c["Test"] != "t" for c in contrasts):
        raise ValueError("Task contrasts must use Test=t")
    return node


def compile_model(source: dict, identifier: str, config: dict, *, sessions: bool = True) -> dict:
    """Build run and direct-from-run summary nodes for a registered task model."""
    from .task_models import scientific_model

    source = scientific_model(source)
    node = task_node(source)
    rules = node["Model"]["Software"]["nro"]
    rules["denoising"] = {key: deepcopy(config[key]) for key in NUISANCE_KEYS}
    rules["aggregation_weighting"] = source["aggregation"]["weighting"]
    primitives = list(
        dict.fromkeys(
            v
            for c in node["Contrasts"]
            for v, w in zip(c["ConditionList"], contrast_weights(c))
            if w
        )
    )
    if any(v == 1 or not isinstance(v, str) for v in primitives):
        raise ValueError("Task contrasts must refer to event predictors, not the intercept")
    requested_contrasts = deepcopy(node["Contrasts"])
    reserved = {c["Name"] for c in requested_contrasts}
    primitive_names = {}
    for index, primitive in enumerate(primitives, 1):
        name = f"nroEffect{index}"
        while name in reserved:
            name += "x"
        reserved.add(name)
        primitive_names[primitive] = name
    rules["internal_contrasts"] = list(primitive_names.values())
    node["Contrasts"].extend(
        {"Name": name, "ConditionList": [p], "Weights": [1], "Test": "pass"}
        for p, name in primitive_names.items()
    )
    summary_contrasts = [
        {
            **c,
            "ConditionList": [
                primitive_names[p] for p, w in zip(c["ConditionList"], contrast_weights(c)) if w
            ],
            "Weights": [float(w) for w in contrast_weights(c) if w],
        }
        for c in requested_contrasts
    ]
    nodes, edges = [node], []
    for level in (["Session"] if sessions else []) + ["Subject"]:
        name = level.lower()
        nodes.append(
            {
                "Name": name,
                "Level": level,
                "GroupBy": ["subject", "session"]
                if sessions and level == "Session"
                else ["subject"],
                "Model": {
                    "Type": "meta",
                    "X": list(primitive_names.values()),
                    "Software": {
                        "nro": {"aggregation_weighting": source["aggregation"]["weighting"]}
                    },
                },
                "Contrasts": deepcopy(summary_contrasts),
            }
        )
        edges.append(
            {
                "Source": "run",
                "Destination": name,
                "Filter": {"contrast": list(primitive_names.values())},
            }
        )
    return canonical_model_document(
        validate_model(
            {
                "Name": identifier.replace("/", "-"),
                "BIDSModelVersion": "1.0.0",
                "Input": {"task": identifier.split("/")[0]},
                "Nodes": nodes,
                "Edges": edges,
            },
            task=identifier.split("/")[0],
        )
    )


def realize_run_node(node: dict, events, confounds, config: dict) -> dict:
    """Resolve actual nuisance columns and event HRFs without reading images.

    Event-only selection prevents task models from referring to confounds.
    Exact outlier columns remain separate from continuous nuisance PCA.
    """
    from .models import expand_columns
    from .transforms import event_variables

    node = deepcopy(node)
    rules = node["Model"]["Software"]["nro"]
    denoising = rules["denoising"]
    if denoising != {key: config[key] for key in NUISANCE_KEYS}:
        raise ValueError("Compiled denoising differs from the firstlevels configuration")
    sparse, factors, convolutions = event_variables(events, node.get("Transformations", {}))
    predictors = expand_columns(
        rules["event_predictors"], list(sparse), factor_variables=tuple(factors)
    )
    collisions = set(predictors).intersection(confounds)
    if collisions:
        raise ValueError(f"Event/confound variable collision: {sorted(collisions)}")
    for pattern in rules.get("hrf_overrides", {}):
        if not any(fnmatch.fnmatchcase(name, pattern) for name in predictors) and not any(
            fnmatch.fnmatchcase(pattern, selected) for selected in rules["event_predictors"]
        ):
            raise ValueError(f"HRF override does not match a selected event predictor: {pattern}")
    by_hrf = {}
    for name in predictors:
        matching = [
            value
            for pattern, value in rules.get("hrf_overrides", {}).items()
            if fnmatch.fnmatchcase(name, pattern)
        ]
        if len(matching) > 1:
            raise ValueError(f"Multiple HRF overrides match {name}")
        hrf = matching[0] if matching else rules["default_hrf"]
        if name in convolutions:
            if matching:
                raise ValueError(f"Explicit convolution conflicts with HRF override: {name}")
            continue
        if hrf:
            by_hrf.setdefault(hrf, []).append(name)
    node.setdefault("Transformations", {"Transformer": "pybids-transforms-v1", "Instructions": []})
    node["Transformations"]["Instructions"].extend(
        {"Name": "Convolve", "Input": names, "Model": hrf} for hrf, names in by_hrf.items()
    )
    outliers = [name for name in confounds if re.search(denoising["temporal_mask_regex"], name)]
    nuisance = [
        name
        for name in confounds
        if name not in outliers and re.search(denoising["confounds_regex"], name)
    ]
    if any(name.startswith("global_signal") for name in nuisance):
        raise ValueError("Firstlevels does not remove global signal")
    node["Model"]["X"] = [*predictors, *nuisance, *outliers, 1]
    rules.update(event_columns=predictors, nuisance_columns=nuisance, outlier_columns=outliers)
    return node
