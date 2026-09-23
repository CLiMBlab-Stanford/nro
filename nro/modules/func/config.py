"""Typed functional inputs and canonical configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from nro.configuration.hardware import GradientUnwarpingResolution
from nro.modules.func.contract import MARSS_DIAGNOSTIC_METHOD, final_resampling_contract
from nro.orchestration.runner_support import ContainerSpec


@dataclass(frozen=True)
class Inputs:
    """Resolved BOLD, reference, fieldmap, and anatomy inputs for one functional run."""

    sbref: Optional[Path]
    epi: Path
    se1: Optional[Path]
    se2: Optional[Path]
    se1_json: Optional[Path] = None
    se2_json: Optional[Path] = None
    epi_json: Optional[Path] = None
    sbref_json: Optional[Path] = None
    epi_metadata: Optional[dict[str, Any]] = None
    epi_metadata_sources: tuple[Path, ...] = ()
    sbref_metadata: Optional[dict[str, Any]] = None
    sbref_metadata_sources: tuple[Path, ...] = ()
    se1_metadata: Optional[dict[str, Any]] = None
    se1_metadata_sources: tuple[Path, ...] = ()
    se2_metadata: Optional[dict[str, Any]] = None
    se2_metadata_sources: tuple[Path, ...] = ()
    sbref_metadata_inheritance: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class Options:
    """Functional registration, denoising, output, and execution settings."""

    out_dir: Path
    work_dir: Path
    project: str
    func_id: str
    sub_id: str
    ses_id: Optional[str]
    overwrite: bool
    output_grid: str
    topup_config: str
    ica_aroma_cmd: Optional[Path]
    cicada_cmd: Path
    ica_classifier: str
    ica_regression: str
    cicada_tolerance: int
    cicada_smoothing_retention_mode: str
    use_jacobian: bool
    fieldmap_syn_refine: bool
    syn_base_transform: str
    syn_base_convergence: str
    syn_base_shrink_factors: str
    syn_base_smoothing_sigmas: str
    syn_refine_transform: str
    syn_refine_convergence: str
    syn_refine_shrink_factors: str
    syn_refine_smoothing_sigmas: str
    marss_mode: str
    marss_min_multiband_factor: int
    gradient_unwarping: str
    gradient_unwarp_image: Path
    gradient_unwarp_runtime: str
    fsaverage_template: str
    bbregister_surf: str
    bbregister_init: str
    bbregister_dof: int
    container: Optional[ContainerSpec]
    debug_first_nvols: int
    output_spaces: tuple[str, ...]
    io_chunk_vols: int
    sdc_method: str
    synbold_disco_image: Path
    synbold_disco_license: Path
    synbold_disco_engine: str
    synbold_overlap_erosion_voxels: int
    synbold_min_overlap_voxels: int
    synbold_max_rigid_translation_mm: float
    synbold_max_rigid_rotation_degrees: float
    sbref_max_rigid_displacement_mm: float
    sbref_max_rigid_rotation_degrees: float
    sbref_min_support_overlap: float
    sbref_min_intensity_correlation: float
    anat_id: str


def functional_config_payload(
    opts: Options,
    *,
    gradient_unwarping: GradientUnwarpingResolution | None = None,
) -> dict[str, object]:
    """Return the canonical output-affecting configuration for one run."""
    payload = {
        "output_grid": opts.output_grid,
        "topup_config": opts.topup_config,
        "ica_aroma_cmd": str(opts.ica_aroma_cmd) if opts.ica_aroma_cmd else None,
        "cicada_cmd": str(opts.cicada_cmd),
        "ica_classifier": opts.ica_classifier,
        "ica_regression": opts.ica_regression,
        "cicada_tolerance": int(opts.cicada_tolerance),
        "cicada_smoothing_retention_mode": opts.cicada_smoothing_retention_mode,
        "use_jacobian": bool(opts.use_jacobian),
        "fieldmap_syn_refine": bool(opts.fieldmap_syn_refine),
        "syn_base_transform": opts.syn_base_transform,
        "syn_base_convergence": opts.syn_base_convergence,
        "syn_base_shrink_factors": opts.syn_base_shrink_factors,
        "syn_base_smoothing_sigmas": opts.syn_base_smoothing_sigmas,
        "syn_refine_transform": opts.syn_refine_transform,
        "syn_refine_convergence": opts.syn_refine_convergence,
        "syn_refine_shrink_factors": opts.syn_refine_shrink_factors,
        "syn_refine_smoothing_sigmas": opts.syn_refine_smoothing_sigmas,
        "bbregister_surf": opts.bbregister_surf,
        "bbregister_init": opts.bbregister_init,
        "bbregister_dof": int(opts.bbregister_dof),
        "debug_first_nvols": int(opts.debug_first_nvols),
        "output_spaces": list(opts.output_spaces),
        "fsaverage_template": opts.fsaverage_template,
        "gradient_unwarping": opts.gradient_unwarping,
        "final_resampling": final_resampling_contract(
            gradient_unwarping=bool(gradient_unwarping and gradient_unwarping.applied)
        ),
        "sdc_method": opts.sdc_method,
        "synbold_disco_image": str(opts.synbold_disco_image),
        "synbold_disco_engine": opts.synbold_disco_engine,
        "synbold_overlap_erosion_voxels": int(opts.synbold_overlap_erosion_voxels),
        "synbold_min_overlap_voxels": int(opts.synbold_min_overlap_voxels),
        "synbold_max_rigid_translation_mm": float(opts.synbold_max_rigid_translation_mm),
        "synbold_max_rigid_rotation_degrees": float(opts.synbold_max_rigid_rotation_degrees),
        "sbref_max_rigid_displacement_mm": float(opts.sbref_max_rigid_displacement_mm),
        "sbref_max_rigid_rotation_degrees": float(opts.sbref_max_rigid_rotation_degrees),
        "sbref_min_support_overlap": float(opts.sbref_min_support_overlap),
        "sbref_min_intensity_correlation": float(opts.sbref_min_intensity_correlation),
    }
    if opts.marss_mode != "off":
        payload["marss_mode"] = opts.marss_mode
        payload["marss_diagnostic_method"] = MARSS_DIAGNOSTIC_METHOD
    if opts.marss_mode == "auto":
        payload["marss_min_multiband_factor"] = int(opts.marss_min_multiband_factor)
    return payload


def normalize_output_spaces(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize supported functional output-space aliases."""
    aliases = {
        "t1w": "T1w",
        "fsnative": "fsnative",
        "mni": "MNI152NLin2009cAsym",
        "mni152nlin2009casym": "MNI152NLin2009cAsym",
        "fsaverage": "fsaverage",
        "fsaverage6": "fsaverage6",
    }
    normalized: list[str] = []
    for raw in values:
        key = str(raw).strip()
        if not key:
            continue
        space = aliases.get(key.lower())
        if space is None:
            raise SystemExit(
                "Unknown output space "
                f"{raw!r}. Expected T1w, fsnative, MNI152NLin2009cAsym, or an fsaverage template"
            )
        if space not in normalized:
            normalized.append(space)
    if not normalized:
        raise SystemExit("At least one output space must be requested.")
    return tuple(normalized)
