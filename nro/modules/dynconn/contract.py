"""Semantic output contract for dynamic-connectivity artifacts."""

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields

DYNCONN_MANIFEST_FIELDS = {
    "domain": "string",
    "space": "string",
    "smoothing_fwhm_mm": "integer",
    "repetition_time_seconds": "number",
    "spatial_shape": "list",
    "source_surfaces": "string_list",
    "concatenated_frames": "integer",
    "functional_runs": "mapping",
    "functional_runs.inclusion_policy": "mapping",
    "functional_runs.minimum_usable_runs": "integer",
    "functional_runs.minimum_aggregate_retained_frames": "integer",
    "functional_runs.included": "list",
    "functional_runs.skipped": "list",
    "outputs": "mapping",
    "outputs.timeseries": "string",
    "outputs.scene": "string",
    "outputs.surfaces": "string_list",
    "config": "mapping",
    "configuration_fingerprint": "nullable_string",
    "output_metadata_contract": "mapping",
}


def dynconn_output_contract() -> dict[str, object]:
    """Return the public layout and required manifest fields."""

    return {
        "layout": "target-subject-dynamic-connectivity-scene-v1",
        "publication_manifest_fields": dict(DYNCONN_MANIFEST_FIELDS),
    }


def validate_dynconn_manifest(document: Mapping[str, object]) -> None:
    """Reject a manifest that omits required public metadata."""

    validate_metadata_fields(
        document, DYNCONN_MANIFEST_FIELDS, label="Dynamic-connectivity manifest"
    )
