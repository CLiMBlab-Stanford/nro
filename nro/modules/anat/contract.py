"""Semantic output contract for anatomical derivatives."""

from __future__ import annotations

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields
from nro.modules.anat.policy import FASTSURFER_VOXEL_SIZE_MM
from nro.modules.anat.policy import (
    bias_correction_contract as bias_correction_contract,
)
from nro.modules.anat.policy import (
    surface_reconstruction_contract as surface_reconstruction_contract,
)

FASTSURFER_RECONSTRUCTION_VOXEL_SIZE_MM = FASTSURFER_VOXEL_SIZE_MM

ANATOMICAL_MANIFEST_FIELDS = {
    "subject": "string",
    "fs_subject": "string",
    "fsaverage_template": "string",
    "selection_strategy": "string",
    "gradient_unwarping": "mapping",
    "bias_correction": "mapping",
    "surface_reconstruction": "mapping",
    "inputs": "mapping",
    "inputs.t1w": "string_list",
    "inputs.t2w": "string_list",
    "copied_session_files": "string_list",
    "outputs": "mapping",
    "outputs.t1w": "nullable_string",
    "outputs.t2w": "nullable_string",
    "outputs.myelin_map": "nullable_string",
    "outputs.brain_image": "string",
    "outputs.brain_mask": "string",
    "outputs.gray_matter_mask": "string",
    "outputs.cortical_ribbon_mask": "string",
    "outputs.subcortical_masks": "mapping",
    "outputs.surfaces": "mapping",
    "outputs.mni_qc_images": "mapping",
    "outputs.pose_qc": "string",
    "outputs.xfms": "mapping",
    "freesurfer_subjects_dir": "string",
    "mni_template": "string",
    "options": "mapping",
    "options.synthstrip_image": "nullable_string",
    "options.configuration": "mapping",
    "options.configuration_fingerprint": "nullable_string",
    "output_metadata_contract": "mapping",
    "complete": "boolean",
}

LESION_MANIFEST_FIELDS = {
    "lesion": "mapping",
    "lesion.enabled": "boolean",
    "lesion.masker": "mapping",
    "lesion.reconstruction": "mapping",
    "lesion.boundary_margin_mm": "number",
    "outputs.inpainted_t1w": "string",
    "outputs.inpainted_t1w_metadata": "string",
    "outputs.intact_surfaces": "mapping",
    "outputs.lesion_mask": "string",
    "outputs.lesion_metadata": "string",
    "outputs.lesion_probability": "string",
    "outputs.lesion_qc": "string",
    "outputs.lesion_reconstruction_summary": "string",
    "outputs.surface_vertex_mappings": "mapping",
    "outputs.surface_validity": "mapping",
}


def pose_normalization_contract() -> dict[str, object]:
    """Describe how the participant anatomical reference is pose-normalized."""
    return {
        "extent": "anatomical_mask",
        "margin_mm": 5.0,
        "resolution": "selected_anatomical",
        "transform_direction": "moving_to_fixed",
        "intensity_interpolation": "cubic_bspline",
        "mask_interpolation": "nearest_neighbor",
        "post_resampling_mask": True,
    }


def anatomical_output_contract(*, lesion: bool = False) -> dict[str, object]:
    """Return the required public metadata schema for substantive freshness comparison."""
    fields = dict(ANATOMICAL_MANIFEST_FIELDS)
    if lesion:
        fields.update(LESION_MANIFEST_FIELDS)
    return {"publication_manifest_fields": fields}


def validate_anatomical_manifest(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    lesion = document.get("lesion")
    fields = dict(ANATOMICAL_MANIFEST_FIELDS)
    if isinstance(lesion, Mapping) and lesion.get("enabled") is True:
        fields.update(LESION_MANIFEST_FIELDS)
    validate_metadata_fields(
        document,
        fields,
        label="Anatomical publication manifest",
    )
