"""Semantic output contract for functional-connectivity gradient artifacts."""

from collections.abc import Mapping
from pathlib import Path

from nro.engine.artifact_metadata import validate_metadata_fields

BACKENDS = ("workbench_correlation_gradient",)

FCGRADIENTS_MANIFEST_FIELDS = {
    "domain": "string",
    "space": "string",
    "smoothing_fwhm_mm": "integer",
    "backend": "string",
    "backend_details": "mapping",
    "source": "mapping",
    "source.module": "string",
    "source.manifest": "string",
    "source.timeseries": "string",
    "source.representation": "string",
    "source.low_rank": "nullable_mapping",
    "geometry": "mapping",
    "structures": "list",
    "locations": "integer",
    "outputs": "mapping",
    "outputs.gradient": "string",
    "config": "mapping",
    "configuration_fingerprint": "nullable_string",
    "output_metadata_contract": "mapping",
}


def fcgradients_output_paths(directory: Path, prefix: str, domain: str) -> dict[str, Path]:
    """Return the fixed public outputs for one gradient target.

    The volume domain also publishes the gradient as a NIfTI image on the
    dynamic-connectivity grid.
    """

    directory = Path(directory)
    paths = {
        "gradient": directory / f"{prefix}_desc-fcGradient_stat.dscalar.nii",
        "manifest": directory / f"{prefix}_desc-fcGradient_manifest.yaml",
        "index": directory / f"{prefix}_desc-fcGradientIndex_manifest.json",
    }
    if domain == "volume":
        paths["gradient_volume"] = directory / f"{prefix}_desc-fcGradient_stat.nii.gz"
    return paths


def fcgradients_output_contract() -> dict[str, object]:
    """Return the public layout, backends, and required manifest fields."""

    return {
        "layout": "subject-fc-gradients-v1",
        "backends": list(BACKENDS),
        "publication_manifest_fields": dict(FCGRADIENTS_MANIFEST_FIELDS),
    }


def validate_fcgradients_manifest(document: Mapping[str, object]) -> None:
    """Reject a manifest that omits required public metadata."""

    validate_metadata_fields(
        document, FCGRADIENTS_MANIFEST_FIELDS, label="Functional-connectivity gradient manifest"
    )
