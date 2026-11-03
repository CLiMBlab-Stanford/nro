"""Methods-report context for functional preprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nro.methods.context import (
    ManifestRecord,
    ModuleMethods,
    RunFact,
    SoftwareUse,
    lookup,
    participants,
)
from nro.modules.func.contract import (
    FINAL_RESAMPLING_INTERPOLATION,
    FINAL_RESAMPLING_TOOL,
    FINAL_WARP_INTERPOLATION,
)

TITLE = "Functional preprocessing"
CONTEXT_VERSION = 1

# Realized ``registration.sdc_method`` values and their context counters.
SDC_COUNTERS = {"topup_sbref_pe": "topup", "synbold_disco": "synbold", "syn": "syn"}

CONTEXT_FIELDS = {
    "observed": "boolean",
    "runs": "integer",
    "participants": "integer",
    "spaces": "string_list",
    "sdc.requested": "string",
    "sdc.from_sbref_pair": "boolean",
    "sdc.fieldmap_syn_refine": "boolean",
    "sdc.topup": "integer",
    "sdc.synbold": "integer",
    "sdc.syn": "integer",
    "sdc.fallback_runs": "integer",
    "sbref.selected_runs": "integer",
    "marss.mode": "string",
    "marss.min_multiband_factor": "integer",
    "marss.applied_runs": "integer",
    "gradient_unwarping.mode": "string",
    "gradient_unwarping.applied_runs": "integer",
    "bbregister.surface": "string",
    "bbregister.init": "string",
    "bbregister.dof": "integer",
    "resampling.tool": "string",
    "resampling.data_interpolation": "string",
    "resampling.warp_interpolation": "string",
    "resampling.output_grid": "string",
    "resampling.use_jacobian": "boolean",
    "ica.classifier": "string",
    "ica.regression": "string",
    "ica.applied_runs": "integer",
    "confounds.fd_threshold_mm": "number",
    "confounds.fd_radius_mm": "number",
    "confounds.dvars_alpha": "number",
    "confounds.dvars_practical_percent": "number",
    "confounds.n_acompcor": "integer",
    "confounds.cosine_high_pass_hz": "number",
    "confounds.cosine_period_s": "number",
    "confounds.nonsteady_max_vols": "integer",
}

CONFIGURATION_FIELDS = {
    "sdc_method": "sdc.requested",
    "sdc_from_sbref_pair": "sdc.from_sbref_pair",
    "fieldmap_syn_refine": "sdc.fieldmap_syn_refine",
    "marss_mode": "marss.mode",
    "marss_min_multiband_factor": "marss.min_multiband_factor",
    "gradient_unwarping": "gradient_unwarping.mode",
    "bbregister_surf": "bbregister.surface",
    "bbregister_init": "bbregister.init",
    "bbregister_dof": "bbregister.dof",
    "output_grid": "resampling.output_grid",
    "use_jacobian": "resampling.use_jacobian",
    "ica_classifier": "ica.classifier",
    "confounds.motion_outlier_fd_thresh": "confounds.fd_threshold_mm",
    "confounds.fd_radius_mm": "confounds.fd_radius_mm",
    "confounds.dvars_statistical_alpha": "confounds.dvars_alpha",
    "confounds.dvars_practical_threshold_percent": "confounds.dvars_practical_percent",
    "confounds.n_acompcor": "confounds.n_acompcor",
    "confounds.cosine_high_pass_hz": "confounds.cosine_high_pass_hz",
    "confounds.nonsteady_max_vols": "confounds.nonsteady_max_vols",
}

_SCHEDULE = "ANTs registration schedule; recorded in the resolved configuration."
_GUARD = "Reference-quality guard; rejections are reported as SBRef use and fallbacks."
_EXECUTION = "Execution environment; images are listed in the software table."

METHODS_SILENT = {
    **{
        f"syn_{stage}_{part}": _SCHEDULE
        for stage in ("base", "refine")
        for part in ("transform", "convergence", "shrink_factors", "smoothing_sigmas")
    },
    **{
        name: _GUARD
        for name in (
            "sbref_max_rigid_displacement_mm",
            "sbref_max_rigid_rotation_degrees",
            "sbref_min_intensity_correlation",
            "sbref_min_support_overlap",
            "synbold_max_rigid_rotation_degrees",
            "synbold_max_rigid_translation_mm",
            "synbold_min_overlap_voxels",
            "synbold_overlap_erosion_voxels",
        )
    },
    **{
        f"container.{name}": _EXECUTION
        for name in ("bind", "cleanenv", "home", "image", "inner_setup", "no_container")
    },
    "confounds.acompcor_max_voxels": "Computational bound on aCompCor voxel sampling.",
    "confounds.aseg_in_epi": "Mask override; default masks come from the anatomy.",
    "confounds.brain_mask_in_epi": "Mask override; default masks come from the anatomy.",
    "confounds.dvars_power": "Afyouni and Nichols test detail; the cited method defines it.",
    "confounds.nonsteady_rel_thresh": "Non-steady-state detector detail.",
    "confounds.nonsteady_stable_run": "Non-steady-state detector detail.",
    "debug_first_nvols": "Debugging control; must be 0 for reported analyses.",
    "markup": "Selects source markup; its effects are reported through observed counts.",
    "synbold_disco_image": "Reported in the software table.",
    "synbold_disco_license": "License file location.",
    "topup_config": "TOPUP configuration file selection.",
}

DEFAULT_SPACES = ("T1w", "fsnative", "MNI152NLin2009cAsym")


def _gradient_applied(data: Mapping[str, Any]) -> bool:
    images = lookup(data, "registration.gradient_unwarping", {}) or {}
    return any(isinstance(item, Mapping) and item.get("applied") for item in images.values())


def methods(
    configuration: Mapping[str, Any],
    manifests: Sequence[ManifestRecord],
    *,
    workflow: Mapping[str, Mapping[str, Any]],
) -> ModuleMethods:
    """Summarize the functional configuration and its fresh run manifests."""
    confounds = configuration["confounds"]
    classifier = str(configuration["ica_classifier"])
    sdc = dict.fromkeys(SDC_COUNTERS.values(), 0)
    fallback = sbref = marss = gradient = ica = 0
    spaces: set[str] = set()
    facts: list[RunFact] = []
    software: set[SoftwareUse] = set()
    for record in manifests:
        data = record.data
        realized = str(lookup(data, "registration.sdc_method", ""))
        if realized in SDC_COUNTERS:
            sdc[SDC_COUNTERS[realized]] += 1
        reason = lookup(data, "registration.sdc_fallback_reason")
        fallback += reason is not None
        selected = lookup(data, "registration.reference_selection.Selected") is True
        sbref += selected
        slice_artifact = lookup(data, "denoising.simultaneous_slice_artifact", {}) or {}
        marss_applied = slice_artifact.get("Applied") is True
        marss += marss_applied
        if marss_applied and slice_artifact.get("OfficialPackageVersion"):
            software.add(
                SoftwareUse(
                    "MARSS", str(slice_artifact["OfficialPackageVersion"]), "tubiolo2024marss"
                )
            )
        gradient_applied = _gradient_applied(data)
        gradient += gradient_applied
        ica_applied = lookup(data, "denoising.applied") is True
        ica += ica_applied
        spaces.update(lookup(data, "options.configuration.output_spaces", []) or [])
        entities = dict(record.entities)
        facts.extend(
            (
                RunFact(entities, "sdc_method", realized),
                RunFact(entities, "sdc_fallback_reason", "" if reason is None else str(reason)),
                RunFact(entities, "sbref_reference", str(selected).lower()),
                RunFact(entities, "marss_applied", str(marss_applied).lower()),
                RunFact(entities, "gradient_unwarping_applied", str(gradient_applied).lower()),
            )
        )
        if classifier != "none":
            removed = lookup(data, "denoising.removed_noise_ic_indices", []) or []
            facts.append(RunFact(entities, "ica_noise_components_removed", str(len(removed))))
    if not spaces:
        spaces = {*DEFAULT_SPACES, str(workflow["anat"]["fsaverage_template"])}
    cosine_hz = float(confounds["cosine_high_pass_hz"])
    context = {
        "observed": bool(manifests),
        "runs": len(manifests),
        "participants": participants(manifests),
        "spaces": sorted(spaces),
        "sdc": {
            "requested": str(configuration["sdc_method"]),
            "from_sbref_pair": bool(configuration["sdc_from_sbref_pair"]),
            "fieldmap_syn_refine": bool(configuration["fieldmap_syn_refine"]),
            **sdc,
            "fallback_runs": fallback,
        },
        "sbref": {"selected_runs": sbref},
        "marss": {
            "mode": str(configuration["marss_mode"]),
            "min_multiband_factor": int(configuration["marss_min_multiband_factor"]),
            "applied_runs": marss,
        },
        "gradient_unwarping": {
            "mode": str(configuration["gradient_unwarping"]),
            "applied_runs": gradient,
        },
        "bbregister": {
            "surface": str(configuration["bbregister_surf"]),
            "init": str(configuration["bbregister_init"]),
            "dof": int(configuration["bbregister_dof"]),
        },
        "resampling": {
            "tool": FINAL_RESAMPLING_TOOL,
            "data_interpolation": FINAL_RESAMPLING_INTERPOLATION,
            "warp_interpolation": FINAL_WARP_INTERPOLATION,
            "output_grid": str(configuration["output_grid"]),
            "use_jacobian": bool(configuration["use_jacobian"]),
        },
        "ica": {
            "classifier": classifier,
            "regression": str(configuration["ica_regression"]),
            "applied_runs": ica,
        },
        "confounds": {
            "fd_threshold_mm": float(confounds["motion_outlier_fd_thresh"]),
            "fd_radius_mm": float(confounds["fd_radius_mm"]),
            "dvars_alpha": float(confounds["dvars_statistical_alpha"]),
            "dvars_practical_percent": float(confounds["dvars_practical_threshold_percent"]),
            "n_acompcor": int(confounds["n_acompcor"]),
            "cosine_high_pass_hz": cosine_hz,
            "cosine_period_s": 1.0 / cosine_hz if cosine_hz > 0 else 0.0,
            "nonsteady_max_vols": int(confounds["nonsteady_max_vols"]),
        },
    }
    images = ["qunex"]
    if sdc["synbold"] or (not manifests and context["sdc"]["requested"] == "synbold_disco"):
        images.append("synbold")
    return ModuleMethods(
        context=context,
        run_facts=tuple(facts),
        software=tuple(sorted(software, key=lambda use: (use.software, use.version))),
        images=tuple(images),
    )
