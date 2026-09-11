"""Semantic output contract for dynamic-connectivity artifacts."""

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields

DYNCONN_MANIFEST_FIELDS = {
    "domain": "string",
    "space": "string",
    "smoothing_fwhm_mm": "integer",
    "representation": "string",
    "weighting": "string",
    "repetition_time_seconds": "number",
    "series_axis_interpretation": "string",
    "spatial_shape": "list",
    "concatenated_frames": "integer",
    "published_frames": "integer",
    "low_rank": "nullable_mapping",
    "functional_runs": "mapping",
    "functional_runs.inclusion_policy": "mapping",
    "functional_runs.minimum_usable_runs": "integer",
    "functional_runs.minimum_aggregate_retained_frames": "integer",
    "functional_runs.included": "list",
    "functional_runs.skipped": "list",
    "outputs": "mapping",
    "outputs.timeseries": "string",
    "config": "mapping",
    "configuration_fingerprint": "nullable_string",
    "output_metadata_contract": "mapping",
}

DYNCONN_LOW_RANK_FIELDS = {
    "method": "string",
    "requested_dimensions": "integer",
    "dimensions": "integer",
    "oversampling": "integer",
    "power_iterations": "integer",
    "random_seed": "integer",
    "input_frames": "integer",
    "synthetic_frames": "integer",
    "spatial_locations": "integer",
    "valid_locations": "integer",
    "zero_variance_locations": "integer",
    "realized_rank": "integer",
    "retained_variance_fraction": "number",
    "eigenvalues": "list",
    "standardization": "string",
    "synthetic_basis": "string",
}


def dynconn_output_contract() -> dict[str, object]:
    """Return the public layout and required manifest fields."""

    return {
        "layout": "subject-dynamic-connectivity-v5",
        "publication_manifest_fields": dict(DYNCONN_MANIFEST_FIELDS),
        "low_rank_fields": dict(DYNCONN_LOW_RANK_FIELDS),
    }


def validate_dynconn_manifest(document: Mapping[str, object]) -> None:
    """Reject a manifest that omits required public metadata."""

    validate_metadata_fields(
        document, DYNCONN_MANIFEST_FIELDS, label="Dynamic-connectivity manifest"
    )
    representation = document["representation"]
    if representation not in {"full", "low_rank"}:
        raise ValueError(f"Unsupported dynamic-connectivity representation: {representation}")
    if document["weighting"] not in {"equal", "precision"}:
        raise ValueError(f"Unsupported dynamic-connectivity weighting: {document['weighting']}")
    low_rank = document["low_rank"]
    if (representation == "low_rank") != (low_rank is not None):
        raise ValueError("Dynamic-connectivity representation and low-rank metadata disagree")
    if low_rank is not None:
        validate_metadata_fields(
            low_rank,
            DYNCONN_LOW_RANK_FIELDS,
            label="Dynamic-connectivity low-rank metadata",
        )
        if low_rank["synthetic_frames"] != low_rank["dimensions"] + 1:
            raise ValueError("Low-rank synthetic frame count must equal dimensions plus one")
        if low_rank["dimensions"] > low_rank["requested_dimensions"]:
            raise ValueError("Fitted low-rank dimensions exceed the requested ceiling")
        if low_rank["random_seed"] < 0:
            raise ValueError("Low-rank random seed must be nonnegative")
        if document["published_frames"] != low_rank["synthetic_frames"]:
            raise ValueError("Published and low-rank synthetic frame counts disagree")
    elif document["published_frames"] != document["concatenated_frames"]:
        raise ValueError("Full dynamic-connectivity frame counts disagree")
