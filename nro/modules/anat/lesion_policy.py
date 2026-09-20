"""Lightweight scientific policy for lesion-aware anatomy."""

from __future__ import annotations

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
FASTSURFER_VERSION = "2.5.4"
FASTSURFER_SOURCE_REVISION = "cdfccea"
FASTSURFER_OCI_DIGEST = "sha256:8db4881c12961a7d6e2c8ed879f6207fbba82d1668c1d64bef281512cebbe1e5"
NEUROLIT_VERSION = "0.6.1"
NEUROLIT_RECORD = "https://doi.org/10.5281/zenodo.14510136"
NEUROLIT_CHECKPOINTS = {
    "model_axial.pt": "f665c8af703032f4604117c5284f1659a2034fc5916484874fc5633d0c2474d2",
    "model_coronal.pt": "b699ff0c8aa75072fd4c6573247372e7c643bc05035d490009d25aee9975f316",
    "model_sagittal.pt": "7c9dba74c19212f1201ac7fa05654dcf7adad987a91f8b7d08b7efb1852f7efa",
}
BOUNDARY_MARGIN_MM = 0.0


def lesion_reconstruction_contract() -> dict[str, object]:
    """Return the pinned scientific policy activated by lesion source markup."""
    return {
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
        "surface_backend": "FastSurfer-LIT",
        "fastsurfer_version": FASTSURFER_VERSION,
        "fastsurfer_source_revision": FASTSURFER_SOURCE_REVISION,
        "fastsurfer_oci_digest": FASTSURFER_OCI_DIGEST,
        "neurolit_version": NEUROLIT_VERSION,
        "neurolit_record": NEUROLIT_RECORD,
        "neurolit_checkpoint_sha256": NEUROLIT_CHECKPOINTS,
        "boundary_margin_mm": BOUNDARY_MARGIN_MM,
        "face_exclusion": "any_lesion_vertex",
    }
