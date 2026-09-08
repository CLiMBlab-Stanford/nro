"""Node-level contrast recipes and bounded-memory evaluation."""

import fnmatch
import re
from collections import defaultdict

import numpy as np

from .io import load_fit
from .models import contrast_weights
from .statistics import Estimate, aggregate, linear_combination


def declared_contrasts(node: dict, names: list[str]) -> list[dict]:
    """Expand dummy contrasts while rejecting duplicate output names."""
    contrasts = list(node.get("Contrasts", []))
    dummy = node.get("DummyContrasts")
    if dummy:
        patterns = dummy.get("Contrasts", node["Model"]["X"])
        candidates = list(names)
        for pattern in patterns:
            name = "intercept" if pattern == 1 else pattern
            if not any(char in name for char in "*?[") and name not in candidates:
                candidates.append(name)
        for name in candidates:
            if any(fnmatch.fnmatchcase(name, "intercept" if p == 1 else p) for p in patterns):
                contrasts.append({"Name": name, "ConditionList": [name], "Weights": [1], "Test": dummy["Test"]})
    if len({c["Name"] for c in contrasts}) != len(contrasts):
        raise ValueError("Explicit and dummy contrasts have duplicate names")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", c["Name"]) for c in contrasts):
        raise ValueError("Contrast names must contain only letters, digits, dots, underscores and hyphens")
    return contrasts


def run_records(node: dict, names: list[str], design, run_key: str, entities: dict) -> tuple[list, list]:
    """Create estimable run contrast recipes and explicit missing-effect records."""
    records, omitted = [], []
    for contrast in declared_contrasts(node, names):
        weights = contrast_weights(contrast)
        conditions = ["intercept" if c == 1 else c for c in contrast["ConditionList"]]
        absent = [name for name, w in zip(conditions, weights) if w != 0 and name not in names]
        vector = np.zeros(len(names))
        for name, weight in zip(conditions, weights):
            if name in names:
                vector[names.index(name)] += weight
        if absent or not design.is_estimable(vector):
            omitted.append({"contrast": contrast["Name"], "entities": entities,
                            "reason": "missing_condition" if absent else "not_estimable",
                            "conditions": absent})
            continue
        records.append({"name": contrast["Name"], "test": contrast["Test"], "entities": entities,
                        "internal": contrast["Name"] in node["Model"].get("Software", {}).get("nro", {}).get("internal_contrasts", []),
                        "recipe": {"kind": "run", "run": run_key, "weights": vector.tolist()}})
    return records, omitted


def meta_records(node: dict, records: list[dict], edge: dict, *, weighting: str) -> tuple[list, list]:
    """Aggregate conditions within GroupBy cells, then form requested contrasts."""
    groups = defaultdict(list)
    for record in records:
        entities = {**record["entities"], "contrast": record["name"]}
        if node["Level"] == "Session" and entities.get("session") is None:
            continue
        if any(entities.get(key) not in allowed for key, allowed in edge.get("Filter", {}).items()):
            continue
        groups[tuple(entities.get(key) for key in node["GroupBy"])].append(record)
    outputs, omitted = [], []
    for grouping, selected in groups.items():
        entities = dict(zip(node["GroupBy"], grouping))
        names = list(dict.fromkeys(record["name"] for record in selected))
        by_name = {}
        if node["Model"]["X"] == [1]:
            if "contrast" not in node["GroupBy"]:
                raise ValueError("An intercept-only meta model must group by contrast")
            by_name["intercept"] = {"kind": weighting, "inputs": [r["recipe"] for r in selected]}
        else:
            for pattern in node["Model"]["X"]:
                if not isinstance(pattern, str):
                    raise ValueError("Meta models use named effects or an intercept-only X=[1]")
                for name in names:
                    if fnmatch.fnmatchcase(name, pattern):
                        by_name[name] = {"kind": weighting, "inputs": [r["recipe"] for r in selected if r["name"] == name]}
        declared = list(by_name)
        for contrast in declared_contrasts(node, declared):
            conditions = ["intercept" if c == 1 else c for c in contrast["ConditionList"]]
            weights = contrast_weights(contrast)
            absent = [name for name, w in zip(conditions, weights) if w and name not in by_name]
            output_name = str(entities.get("contrast", contrast["Name"])) if contrast["Name"] == "intercept" else contrast["Name"]
            if absent:
                omitted.append({"contrast": output_name, "entities": entities, "reason": "missing_condition", "conditions": absent})
                continue
            terms = [(by_name[name], float(w)) for name, w in zip(conditions, weights) if w]
            outputs.append({"name": output_name, "test": contrast["Test"],
                            "entities": {k: v for k, v in entities.items() if k != "contrast"},
                            "recipe": {"kind": "linear", "inputs": [r for r, _ in terms], "weights": [w for _, w in terms]}})
    return outputs, omitted


def evaluate_recipe(recipe: dict, fits: dict) -> Estimate:
    """Resolve a summary recipe on one spatial block of compact source fits."""
    if recipe["kind"] == "run":
        return Estimate({recipe["run"]: np.asarray(recipe["weights"])})
    estimates = [evaluate_recipe(child, fits) for child in recipe["inputs"]]
    if recipe["kind"] == "linear":
        return linear_combination(estimates, recipe["weights"])
    return aggregate(estimates, fits, weighting=recipe["kind"])


def evaluate_maps(recipe: dict, sources: dict, *, block_size: int) -> dict[str, np.ndarray]:
    """Stream original-run covariance into a constant number of scalar-map buffers."""
    first = load_fit(next(iter(sources.values())))
    n_locations = first.beta.shape[1]
    result = {name: np.empty(n_locations, dtype=np.float32) for name in ("effect", "variance", "t", "dof")}
    for start in range(0, n_locations, block_size):
        block = slice(start, min(start + block_size, n_locations))
        fits = {key: load_fit(record, block) for key, record in sources.items()}
        values = evaluate_recipe(recipe, fits).evaluate(fits)
        for name, array in values.items():
            result[name][block] = array
    return result
