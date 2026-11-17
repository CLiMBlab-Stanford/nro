"""Group the resolved choices and inputs that shape a functional runner graph.

`build_module` resolves these values from the options, the BOLD metadata, the
selected anatomy, and the available fieldmaps before it declares the
registration and resampling steps. Later stages read them from one
`FuncGraphPolicy` instead of from separate local variables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nro.definitions.hardware import GradientUnwarpingResolution


@dataclass(frozen=True)
class FuncGraphPolicy:
    """Resolved choices and inputs for one functional graph."""

    classifier: str
    """ICA classifier: `none`, `ica_aroma`, or `cicada`."""
    denoise_type: str
    """ICA-AROMA denoising mode label used in output names."""
    ica_enabled: bool
    """Whether an ICA classifier runs."""
    epi_metadata_sources: Sequence[Path | None]
    """Sidecars that supplied the BOLD metadata."""
    epi_input_meta: dict
    """Merged BOLD metadata."""
    gradient_resolutions: Mapping[str, GradientUnwarpingResolution]
    """Gradient-unwarping decisions by input kind."""
    gradient_resolution: GradientUnwarpingResolution
    """Gradient-unwarping decision for the BOLD run."""
    subjects_dir: Path
    """FreeSurfer subjects directory of the selected anatomy."""
    fs_subject: str
    """FreeSurfer subject of the selected anatomy."""
    anat_t1: Path
    """Observed T1w image of the selected anatomy."""
    registration_t1: Path
    """T1w image used as the registration target."""
    anat_brain_mask: Path
    """Brain mask of the selected anatomy."""
    anat_mni_template: Path
    """MNI template used by the selected anatomy."""
    t1_to_mni_xfm: Path
    """T1w-to-MNI transform of the selected anatomy."""
    fsnative_surfaces: dict[str, Path]
    """fsnative surfaces of the selected anatomy, by hemisphere and surface."""
    requested_sdc_method: str
    """Susceptibility distortion correction method requested by the configuration."""
    use_fieldmap_sdc: bool
    """Whether a reverse phase-encoding fieldmap pair corrects distortion."""
    sdc_fallback_reason: str | None
    """Why a fieldmapless method fell back, if it did."""
    use_synbold_topup: bool
    """Whether SynBOLD-DisCo supplies the TOPUP reference."""
    use_syn_fallback: bool
    """Whether fieldmapless SyN correction runs."""
    use_synbold_reference: bool
    """Whether the synthetic SynBOLD reference is used."""
    fieldmap_refinement_target: str | None
    """Target of the post-fieldmap refinement, if any."""
    do_refinement: bool
    """Whether the post-fieldmap anatomical refinement runs."""
    resolved_sdc_method: str
    """Distortion correction method recorded in outputs."""
    want_t1: bool
    """Whether T1w-space outputs were requested."""
    want_mni: bool
    """Whether MNI-space outputs were requested."""
    want_fsnative: bool
    """Whether fsnative surface outputs were requested."""
    fsaverage_space: str
    """fsaverage template name used for template surface outputs."""
    want_fsaverage: bool
    """Whether fsaverage surface outputs were requested."""
    need_surface_outputs: bool
    """Whether any surface output was requested."""
