"""Pinned scientific policy for anatomical reconstruction backends."""

from __future__ import annotations

from nro.engine.freesurfer_templates import FREESURFER_BUILD

FREESURFER_VERSION = "7.4.1"
FREESURFER_PRIMARY_SEED = 1234
FREESURFER_TOPOLOGY_FALLBACK_SEEDS = (5678,)
FASTSURFER_VERSION = "2.5.4"
FASTSURFER_SOURCE_REVISION = "cdfccea"
FASTSURFER_OCI_DIGEST = "sha256:8db4881c12961a7d6e2c8ed879f6207fbba82d1668c1d64bef281512cebbe1e5"
FASTSURFER_VOXEL_SIZE_MM = 1.0
T1W_FALLBACK_DEFAULT = "synthesize_from_t2w"


def bias_correction_contract() -> dict[str, object]:
    """Describe brain-guided N4 correction of source anatomy."""
    return {
        "method": "N4BiasFieldCorrection",
        "mask_source": "SynthStrip",
        "mask_application": "hard_mask",
        "bias_field_retained": True,
    }


def t1w_synthesis_contract() -> dict[str, object]:
    """Describe synthesis of a canonical T1w reference from selected T2w data."""
    return {
        "backend": "FreeSurfer mri_synthsr",
        "version": FREESURFER_VERSION,
        "build": FREESURFER_BUILD,
        "source_contrast": "T2w",
        "output_contrast": "synthetic T1w",
        "output_resolution_mm": 1.0,
        "device": "CPU",
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
            "random_seed_policy": {
                "primary": FREESURFER_PRIMARY_SEED,
                "topology_failure_fallbacks": list(FREESURFER_TOPOLOGY_FALLBACK_SEEDS),
            },
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
