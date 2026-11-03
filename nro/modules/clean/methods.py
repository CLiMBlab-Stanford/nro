"""Methods-report context for nuisance regression and temporal filtering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nro.methods.context import ManifestRecord, ModuleMethods, participants

TITLE = "Denoising"
CONTEXT_VERSION = 1

DEFAULT_CONFOUNDS_REGEX = r"^(global_signal|framewise_displacement|trans_[xyz]|rot_[xyz]|a_comp_cor_0[0-4]|cosine.*|.*_outlier.*)$"
DEFAULT_TEMPORAL_MASK_REGEX = ".*outlier.*"

CONTEXT_FIELDS = {
    "observed": "boolean",
    "runs": "integer",
    "participants": "integer",
    "spaces": "string_list",
    "smoothing_mm": "string_list",
    "default_confounds": "boolean",
    "confounds_regex": "string",
    "default_temporal_mask": "boolean",
    "temporal_mask_regex": "string",
    "min_trs": "integer",
    "gm_mask_threshold": "number",
    "high_pass_hz": "number",
    "low_pass_hz": "number",
    "detrend": "boolean",
    "standardize": "boolean",
    "regress_out_task": "boolean",
    "nuisance_variance_explained": "number",
    "minimum_temporal_rank": "integer",
    "minimum_temporal_rank_fraction": "number",
}

CONFIGURATION_FIELDS = {
    "confounds_regex": "confounds_regex",
    "temporal_mask_regex": "temporal_mask_regex",
    "min_trs": "min_trs",
    "gm_mask_threshold": "gm_mask_threshold",
    "high_pass": "high_pass_hz",
    "low_pass": "low_pass_hz",
    "detrend": "detrend",
    "standardize": "standardize",
    "regress_out_task": "regress_out_task",
    "nuisance_variance_explained": "nuisance_variance_explained",
    "minimum_temporal_rank": "minimum_temporal_rank",
    "minimum_temporal_rank_fraction": "minimum_temporal_rank_fraction",
}

METHODS_SILENT = {
    **{
        f"container.{name}": "Execution environment; images are listed in the software table."
        for name in ("bind", "cleanenv", "home", "image", "inner_setup", "no_container")
    },
    "markup": "Selects source markup; its effects are reported through observed counts.",
    "wb_command": "Workbench executable name inside the container.",
}


def methods(
    configuration: Mapping[str, Any],
    manifests: Sequence[ManifestRecord],
    *,
    workflow: Mapping[str, Mapping[str, Any]],
) -> ModuleMethods:
    """Summarize the cleaning configuration and its fresh run manifests."""
    del workflow
    spaces = sorted(
        {str(record.data.get("space")) for record in manifests if record.data.get("space")}
    )
    smoothing = [
        f"{value:g}"
        for value in sorted(
            {
                float(record.data["smoothing_fwhm_mm"])
                for record in manifests
                if record.data.get("smoothing_fwhm_mm") is not None
            }
        )
    ]
    runs = {
        (
            record.entities.get("project"),
            record.entities.get("participant"),
            record.data.get("run_stem"),
        )
        for record in manifests
    }
    confounds_regex = str(configuration["confounds_regex"])
    mask_regex = str(configuration["temporal_mask_regex"])
    context = {
        "observed": bool(manifests),
        "runs": len(runs),
        "participants": participants(manifests),
        "spaces": spaces,
        "smoothing_mm": smoothing,
        "default_confounds": confounds_regex == DEFAULT_CONFOUNDS_REGEX,
        "confounds_regex": confounds_regex,
        "default_temporal_mask": mask_regex == DEFAULT_TEMPORAL_MASK_REGEX,
        "temporal_mask_regex": mask_regex,
        "min_trs": int(configuration["min_trs"]),
        "gm_mask_threshold": float(configuration["gm_mask_threshold"]),
        "high_pass_hz": float(configuration["high_pass"]),
        "low_pass_hz": float(configuration["low_pass"]),
        "detrend": bool(configuration["detrend"]),
        "standardize": bool(configuration["standardize"]),
        "regress_out_task": bool(configuration["regress_out_task"]),
        "nuisance_variance_explained": float(configuration["nuisance_variance_explained"]),
        "minimum_temporal_rank": int(configuration["minimum_temporal_rank"]),
        "minimum_temporal_rank_fraction": float(configuration["minimum_temporal_rank_fraction"]),
    }
    return ModuleMethods(context=context, images=("qunex",))
