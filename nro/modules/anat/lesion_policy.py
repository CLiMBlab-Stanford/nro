"""Lightweight scientific policy for lesion-aware anatomy."""

from __future__ import annotations

from nro.engine.freesurfer_templates import FREESURFER_BUILD

from .policy import FREESURFER_VERSION

MASKER_MODEL = "liamchalcroft/synthstroke-synth-plus"
MASKER_REVISION = "e9774354fe78eac705d0b98b56e758d47b363ad3"
MASKER_SOURCE_REVISION = "f627b57b8f9291bfa870cb54cb21ddbeb4941d39"
MASKER_CONFIG_SHA256 = "5ef29a114dbe64ea7c65ea952fdbd6ff867a619baa2a6b5a4c47ae8f02483541"
MASKER_WEIGHTS_SHA256 = "d53d6577c064c86dc466d0144362875c90a1adedc71375cb09711e271432b552"
MASKER_RESOURCES = {
    "config.json": MASKER_CONFIG_SHA256,
    "model.safetensors": MASKER_WEIGHTS_SHA256,
}
MASKER_VOXEL_SIZE_MM = 1.0
MASKER_PATCH_SIZE = 128
MASKER_WINDOW_OVERLAP = 0.5
PROBABILITY_THRESHOLD = 0.5
TEST_TIME_AUGMENTATION = True
SYNTHSR_MODEL = "all-purpose"
BOUNDARY_MARGIN_MM = 0.0


def synthsr_scaffold_contract() -> dict[str, object]:
    """Describe the whole-brain synthetic T1w used as a reconstruction scaffold."""
    return {
        "backend": "FreeSurfer mri_synthsr",
        "version": FREESURFER_VERSION,
        "build": FREESURFER_BUILD,
        "model": SYNTHSR_MODEL,
        "input": "pose_normalized_T1w_before_N4",
        "output_contrast": "synthetic T1w",
        "output_resolution_mm": 1.0,
        "device": "CPU",
        "negative_intensity_policy": "clip_to_zero",
    }


def lesion_reconstruction_contract() -> dict[str, object]:
    """Describe scaffold synthesis and excision around the selected surface backend."""
    return {
        "pipeline": "inpainting_surface_reconstruction_excision",
        "strategy": "whole_brain_synthesis_surface_reconstruction_excision",
        "masker": "SynthStroke",
        "masker_model": MASKER_MODEL,
        "masker_revision": MASKER_REVISION,
        "masker_source_revision": MASKER_SOURCE_REVISION,
        "masker_config_sha256": MASKER_CONFIG_SHA256,
        "masker_weights_sha256": MASKER_WEIGHTS_SHA256,
        "masker_voxel_size_mm": MASKER_VOXEL_SIZE_MM,
        "masker_patch_size": MASKER_PATCH_SIZE,
        "masker_window_overlap": MASKER_WINDOW_OVERLAP,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "test_time_augmentation": TEST_TIME_AUGMENTATION,
        "scaffold_synthesis": synthsr_scaffold_contract(),
        "reconstruction_brain_mask": "whole_brain_reference",
        "boundary_margin_mm": BOUNDARY_MARGIN_MM,
        "face_exclusion": "any_lesion_vertex",
    }
