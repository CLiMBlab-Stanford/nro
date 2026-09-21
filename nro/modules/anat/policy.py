"""Pinned scientific policy for ordinary anatomical reconstruction."""

from __future__ import annotations

FREESURFER_VERSION = "7.4.1"
FREESURFER_BUILD = "freesurfer-linux-centos8_x86_64-7.4.1-20230613-7eb8460"


def bias_correction_contract() -> dict[str, object]:
    """Describe brain-guided N4 correction of full-head anatomy."""
    return {
        "method": "N4BiasFieldCorrection",
        "mask_source": "SynthStrip",
        "mask_application": "hard_mask",
        "bias_field_retained": True,
    }


def surface_reconstruction_contract() -> dict[str, object]:
    """Describe conventional FreeSurfer reconstruction with an external mask."""
    return {
        "backend": "FreeSurfer",
        "version": FREESURFER_VERSION,
        "build": FREESURFER_BUILD,
        "skull_stripping": "SynthStrip_external_mask",
        "recon_all_stages": ["autorecon1", "autorecon2", "autorecon3"],
    }
