"""Semantic publication requirements for first-level statistics."""

from pathlib import Path
import json

from nro.configuration.store import fingerprint
from nro.configuration.schema import scientific_values


def definition_fingerprint(definition: dict) -> str:
    """Hash a model/configuration snapshot and, for summaries, its selected runs."""
    result = fingerprint({"model": definition["model"], "config": definition["config"]})
    if "runs" in definition:
        result = fingerprint({"definition": result, "runs": definition["runs"]})
    return result


def _matches_definition(manifest: dict, expected: dict) -> bool:
    from .compiler import canonical_model_document

    recorded = {**expected, "model": manifest["model_document"], "config": manifest["configuration"]}
    # Verify the saved fingerprint before normalization. For summaries this
    # also checks that the expected run selection matches the recorded fit.
    if definition_fingerprint(recorded) != manifest["definition_fingerprint"]:
        return False
    recorded["model"] = canonical_model_document(recorded["model"])
    expected = {**expected, "model": canonical_model_document(expected["model"])}
    recorded["config"] = scientific_values("firstlevels", recorded["config"])
    expected["config"] = scientific_values("firstlevels", expected["config"])
    return definition_fingerprint(recorded) == definition_fingerprint(expected)


def firstlevels_output_contract() -> dict:
    """Describe the substantive estimator, covariance and omission contracts."""
    return {"layout": "task-level-subject-model-variant", "temporal_filtering": "none",
            "design_export": "compiled-task-model-and-retained-acquisition-rows", "statistics": ["effect", "variance", "t", "dof"],
            "covariance": "original-run-grouped-gls", "degrees_of_freedom": "run-conditional-satterthwaite",
            "input_denoising": "without-aroma", "missing_conditions": "omit-maps-record-reason", "required_manifest_fields":
            ["complete", "model", "node", "configuration", "definition_fingerprint", "public_outputs", "omissions", "records", "source_fits"]}


def validate_completion(path: Path, *, definition: dict | None = None) -> tuple[bool, str]:
    """Validate declared outputs, including compact covariance and node metadata."""
    try:
        value = json.loads(Path(path).read_text())
        expected = firstlevels_output_contract()
        if value.get("output_metadata_contract") != expected or value.get("complete") is not True:
            return False, "Firstlevels publication contract differs"
        for field in expected["required_manifest_fields"]:
            if field not in value:
                return False, f"Missing firstlevels field: {field}"
        if definition is not None and not _matches_definition(value, definition):
            return False, "Firstlevels model or configuration changed"
        if not isinstance(value["records"], list) or not isinstance(value["omissions"], list):
            return False, "Invalid contrast or omission inventory"
        if value["node"] != "module" and value["records"]:
            geometry = value["geometry"]
            fields = {"counts"} if geometry["domain"] == "surface" else {"shape", "affine", "reference_path"}
            if not fields <= set(geometry):
                return False, "Missing firstlevels spatial correspondence metadata"
        for source in value["source_fits"].values():
            if source["dof"] <= 0 or set(source["arrays"]) != {"beta", "residual_variance", "groups", "covariance", "ar_coefficients"}:
                return False, "Invalid compact covariance inventory"
            for output in source["arrays"].values():
                if not Path(output).is_file() or not Path(output).stat().st_size:
                    return False, f"Missing compact fit: {output}"
        for output in value["public_outputs"]:
            item = Path(output)
            if not item.is_file() or not item.stat().st_size:
                return False, f"Missing firstlevels output: {output}"
        return True, "complete"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return False, str(error)


def validate_public_definition(manifest: dict, processing: dict) -> tuple[bool, str]:
    """Check scientific task identity when adopting public outputs after repair."""
    expected = processing.get("task_model")
    from .task_models import scientific_model
    try:
        matches = expected is not None and scientific_model(manifest.get("task_model")) == scientific_model(expected)
    except (ValueError, TypeError, KeyError):
        matches = False
    if not matches:
        return False, "Public firstlevels task model differs from the current definition"
    return True, "Task model matches"
