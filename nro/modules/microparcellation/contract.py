"""Semantic output contract for microparcellation derivatives."""

from __future__ import annotations

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields

MICROPARCELLATION_QUALITY_FIELDS = {
    "metric": "string",
    "source_nodes": "integer",
    "microparcels": "integer",
    "included_runs": "integer",
    "runwise_standardization": "boolean",
    "global_signal_regression": "boolean",
    "weighting": "string",
    "variance_preserved": "number",
    "variance_lost": "number",
    "residual_sum_squares": "number",
    "total_sum_squares": "number",
    "parcel_support": "mapping",
    "parcel_support.minimum_supporting_runs": "integer",
    "parcel_support.mean_supporting_runs": "number",
    "parcel_support.mean_effective_runs": "number",
    "parcel_support.minimum_effective_runs": "number",
    "run_contributions": "list",
    "split_half": "mapping",
    "split_half.method": "string",
    "split_half.allocation": "string",
    "split_half.temporal_block_size": "nullable_number",
    "split_half.first_half_runs": "list",
    "split_half.second_half_runs": "list",
    "split_half.first_half_retained_frames": "integer",
    "split_half.second_half_retained_frames": "integer",
    "split_half.comparable_edges": "integer",
    "split_half.edge_correlation": "number",
    "split_half.spearman_brown_reliability": "number",
    "split_half.mean_absolute_difference": "number",
    "split_half.root_mean_square_difference": "number",
    "connectome": "mapping",
    "connectome.unique_edges": "integer",
    "connectome.off_diagonal_mean": "number",
    "connectome.off_diagonal_standard_deviation": "number",
    "connectome.encoded_quantiles": "mapping",
    "connectome.encoded_positive_fraction": "number",
    "connectome.encoded_negative_fraction": "number",
    "connectome.encoded_zero_fraction": "number",
    "connectome.encoded_saturation_fraction": "number",
    "connectome.encoded_histogram": "mapping",
    "connectome.encoded_histogram.codes": "list",
    "connectome.encoded_histogram.counts": "list",
    "connectome.maximum_asymmetry": "number",
    "connectome.participation_ratio_rank": "number",
    "connectome.dominant_eigenvalue_fraction": "number",
    "connectome.power_iterations": "integer",
    "connectome.power_initialization_seed": "integer",
    "null_baseline": "mapping",
    "null_baseline.method": "string",
    "null_baseline.seed": "integer",
    "null_baseline.parcellations": "list",
    "null_baseline.variance_preserved_mean": "number",
    "null_baseline.variance_preserved_standard_deviation": "number",
    "null_baseline.variance_preserved_minimum": "number",
    "null_baseline.variance_preserved_maximum": "number",
    "null_baseline.fitted_minus_null_mean": "number",
}

MICROPARCELLATION_MANIFEST_FIELDS = {
    "domain": "string",
    "space": "nullable_string",
    "smoothing_fwhm_mm": "integer",
    "n_spatial_nodes": "integer",
    "n_surface_vertices": "nullable_number",
    "hemisphere_vertex_counts": "list",
    "n_active_nodes": "integer",
    "n_active_vertices": "nullable_number",
    "n_gray_matter_voxels": "nullable_number",
    "n_microparcels": "integer",
    "coarsening_steps": "list",
    "source_surfaces": "string_list",
    "source_volume_mask": "nullable_string",
    "volume_mask_resampled": "nullable_boolean",
    "volume_connectivity": "nullable_number",
    "functional_runs": "mapping",
    "functional_runs.inclusion_policy": "mapping",
    "functional_runs.minimum_usable_runs": "integer",
    "functional_runs.minimum_aggregate_retained_frames": "integer",
    "functional_runs.aggregate_retained_frames": "integer",
    "functional_runs.included": "list",
    "functional_runs.skipped": "list",
    "outputs": "mapping",
    "outputs.microparcels": "string",
    "outputs.connectivity": "string",
    "outputs.quality": "string",
    "connectivity_encoding": "mapping",
    "connectivity_encoding.format": "string",
    "connectivity_encoding.dtype": "string",
    "connectivity_encoding.scale": "number",
    "connectivity_encoding.range": "list",
    "connectivity_encoding.diagonal": "number",
    "quality": "mapping",
    "config": "mapping",
    "configuration_fingerprint": "nullable_string",
    "output_metadata_contract": "mapping",
}


def microparcellation_output_contract() -> dict[str, object]:
    """Return the required public metadata schema for substantive freshness comparison."""
    return {
        "layout": "subject-microparcellation-v4",
        "publication_manifest_fields": dict(MICROPARCELLATION_MANIFEST_FIELDS),
        "quality_fields": dict(MICROPARCELLATION_QUALITY_FIELDS),
    }


def validate_microparcellation_quality(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        MICROPARCELLATION_QUALITY_FIELDS,
        label="Microparcellation quality metadata",
    )
    if document["weighting"] not in {"equal", "precision"}:
        raise ValueError(f"Unsupported microparcellation weighting: {document['weighting']}")


def validate_microparcellation_manifest(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        MICROPARCELLATION_MANIFEST_FIELDS,
        label="Microparcellation publication manifest",
    )
