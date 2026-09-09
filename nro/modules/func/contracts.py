"""Substantive functional-processing policies shared with orchestration."""

from __future__ import annotations

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields

FINAL_RESAMPLING_TOOL = "AFNI 3dNwarpApply"
FINAL_RESAMPLING_INTERPOLATION = "wsinc5"
FINAL_WARP_INTERPOLATION = "linear"

FUNCTIONAL_MANIFEST_FIELDS = {
    "manifest_version": "integer",
    "module": "string",
    "run_stem": "string",
    "complete": "boolean",
    "inputs": "mapping",
    "inputs.epi": "string",
    "inputs.epi_metadata": "string_list",
    "inputs.sbref": "nullable_string",
    "inputs.se1": "nullable_string",
    "inputs.se2": "nullable_string",
    "inputs.anatomical_manifest": "string",
    "options": "mapping",
    "public_outputs": "mapping",
    "public_outputs.clean_inputs": "mapping",
    "public_outputs.confounds_tsv": "string",
    "public_outputs.confounds_json": "string",
    "public_outputs.files": "string_list",
    "registration": "mapping",
    "registration.method": "string",
    "registration.requested_sdc_method": "string",
    "registration.sdc_method": "string",
    "registration.sdc_fallback_reason": "nullable_string",
    "registration.reference_selection": "mapping",
    "registration.static_warp": "string",
    "registration.final_resampling": "mapping",
    "registration.fieldmap_transfer": "nullable_mapping",
    "registration.pe_residual_refinement": "nullable_mapping",
    "denoising": "mapping",
    "denoising.applied": "boolean",
    "denoising.method": "nullable_string",
    "denoising.mode": "nullable_string",
    "denoising.removed_noise_ic_indices": "list",
    "output_metadata_contract": "mapping",
}

FUNCTIONAL_IMAGE_SIDECAR_FIELDS = {
    "Description": "string",
    "Sources": "string_list",
    "SpatialReference": "nullable_string",
    "Registration": "mapping",
    "Denoising": "mapping",
    "Configuration": "mapping",
    "ConfigurationFingerprint": "nullable_string",
}


def final_resampling_contract() -> dict[str, object]:
    """Return the processing contract for final volumetric BOLD resampling."""
    return {
        "tool": FINAL_RESAMPLING_TOOL,
        "data_interpolation": FINAL_RESAMPLING_INTERPOLATION,
        "warp_interpolation": FINAL_WARP_INTERPOLATION,
        "combined_spatial_warp_and_per_volume_motion": True,
        "interpolation_count": 1,
    }


def final_resampling_metadata() -> dict[str, object]:
    """Return the same policy using BIDS-sidecar key conventions."""
    contract = final_resampling_contract()
    return {
        "Tool": contract["tool"],
        "DataInterpolation": contract["data_interpolation"],
        "WarpInterpolation": contract["warp_interpolation"],
        "CombinedSpatialWarpAndPerVolumeMotion": contract[
            "combined_spatial_warp_and_per_volume_motion"
        ],
        "InterpolationCount": contract["interpolation_count"],
    }


def functional_output_contract() -> dict[str, object]:
    """Return the freshness-relevant functional metadata schemas."""
    return {
        "publication_manifest_fields": dict(FUNCTIONAL_MANIFEST_FIELDS),
        "image_sidecar_fields": dict(FUNCTIONAL_IMAGE_SIDECAR_FIELDS),
    }


def validate_functional_manifest(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        FUNCTIONAL_MANIFEST_FIELDS,
        label="Functional publication manifest",
    )


def validate_functional_image_sidecar(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        FUNCTIONAL_IMAGE_SIDECAR_FIELDS,
        label="Functional image sidecar",
    )
