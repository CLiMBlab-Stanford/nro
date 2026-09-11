"""Semantic output contract for anatomical derivatives."""

from __future__ import annotations

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields

ANATOMICAL_MANIFEST_FIELDS = {
    "subject": "string",
    "fs_subject": "string",
    "fsaverage_template": "string",
    "selection_strategy": "string",
    "inputs": "mapping",
    "inputs.t1w": "string_list",
    "inputs.t2w": "string_list",
    "copied_session_files": "string_list",
    "outputs": "mapping",
    "outputs.subject_t1w": "nullable_string",
    "outputs.subject_t2w": "nullable_string",
    "outputs.myelin_map": "nullable_string",
    "outputs.brain_image": "string",
    "outputs.brain_mask": "string",
    "outputs.gray_matter_mask": "string",
    "outputs.cortical_ribbon_mask": "string",
    "outputs.subcortical_masks": "mapping",
    "outputs.surfaces": "mapping",
    "outputs.mni_qc_images": "mapping",
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


def anatomical_output_contract() -> dict[str, object]:
    """Return the required public metadata schema for substantive freshness comparison."""
    return {"publication_manifest_fields": dict(ANATOMICAL_MANIFEST_FIELDS)}


def validate_anatomical_manifest(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        ANATOMICAL_MANIFEST_FIELDS,
        label="Anatomical publication manifest",
    )
