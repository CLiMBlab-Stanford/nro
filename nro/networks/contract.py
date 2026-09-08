"""Semantic output contract for individualized-network derivatives."""

from __future__ import annotations

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields


NETWORK_LABEL_METADATA_FIELDS = {
    "method": "string",
    "candidates_per_reference": "integer",
    "references": "string_list",
    "map_names": "string_list",
}

NETWORK_MANIFEST_FIELDS = {
    "domain": "string",
    "space": "nullable_string",
    "smoothing_fwhm_mm": "integer",
    "n_spatial_nodes": "integer",
    "n_surface_vertices": "nullable_number",
    "hemisphere_vertex_counts": "list",
    "n_active_vertices": "nullable_number",
    "n_gray_matter_voxels": "nullable_number",
    "n_microparcels": "integer",
    "n_edges": "integer",
    "parcellation_strategy": "string",
    "reference_run": "nullable_number",
    "n_reference_networks": "integer",
    "source_surfaces": "string_list",
    "microparcellation_manifest": "string",
    "anatomical_labeling_provenance": "nullable_mapping",
    "outputs": "mapping",
    "outputs.membership": "string",
    "outputs.network_maps": "mapping",
    "outputs.network_labels": "string",
    "outputs.network_labels_metadata": "string",
    "outputs.scene": "string",
    "config": "mapping",
    "configuration_fingerprint": "nullable_string",
    "interpretation": "string",
    "output_metadata_contract": "mapping",
}


def networks_output_contract() -> dict[str, object]:
    """Return the required public metadata schema for substantive freshness comparison."""
    return {
        "layout": "target-subject-scene-v1",
        "publication_manifest_fields": dict(NETWORK_MANIFEST_FIELDS),
        "label_metadata_fields": dict(NETWORK_LABEL_METADATA_FIELDS),
    }


def validate_network_label_metadata(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        NETWORK_LABEL_METADATA_FIELDS,
        label="Network label metadata",
    )


def validate_network_manifest(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        NETWORK_MANIFEST_FIELDS,
        label="Networks publication manifest",
    )
