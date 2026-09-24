"""Pinned scientific policy for anatomical reconstruction backends."""

from __future__ import annotations

FREESURFER_VERSION = "7.4.1"
FREESURFER_BUILD = "freesurfer-linux-centos8_x86_64-7.4.1-20230613-7eb8460"
FASTSURFER_VERSION = "2.5.4"
FASTSURFER_SOURCE_REVISION = "cdfccea"
FASTSURFER_OCI_DIGEST = "sha256:8db4881c12961a7d6e2c8ed879f6207fbba82d1668c1d64bef281512cebbe1e5"
FASTSURFER_FREESURFER_BUILD = "freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460"
FASTSURFER_VOXEL_SIZE_MM = 1.0


def bias_correction_contract() -> dict[str, object]:
    """Describe brain-guided N4 correction of source anatomy."""
    return {
        "method": "N4BiasFieldCorrection",
        "mask_source": "SynthStrip",
        "mask_application": "hard_mask",
        "bias_field_retained": True,
    }


def surface_reconstruction_contract(engine: str = "freesurfer") -> dict[str, object]:
    """Describe the selected surface-reconstruction backend."""
    if engine == "freesurfer":
        return {
            "backend": "FreeSurfer",
            "version": FREESURFER_VERSION,
            "build": FREESURFER_BUILD,
            "skull_stripping": "SynthStrip_external_mask",
            "recon_all_stages": ["autorecon1", "autorecon2", "autorecon3"],
            "external_mask_resampling": "nearest_neighbor",
            "brainmask_intensity_source": "FreeSurfer_normalized_T1",
        }
    if engine != "fastsurfer":
        raise ValueError(f"Unsupported surface-reconstruction engine: {engine}")
    return {
        "backend": "FastSurfer",
        "version": FASTSURFER_VERSION,
        "source_revision": FASTSURFER_SOURCE_REVISION,
        "oci_digest": FASTSURFER_OCI_DIGEST,
        "voxel_size_mm": FASTSURFER_VOXEL_SIZE_MM,
        "mask_reconciliation": "segmentation_union",
        "freesurfer_parcellation": True,
        "options": ["fsaparc", "no_cereb", "no_hypothal"],
    }
