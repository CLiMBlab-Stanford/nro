"""Methods-report context for pooled dynamic-connectivity series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nro.methods.context import ManifestRecord, ModuleMethods, lookup, participants

TITLE = "Dynamic connectivity"
CONTEXT_VERSION = 1

CONTEXT_FIELDS = {
    "observed": "boolean",
    "participants": "integer",
    "included_runs": "integer",
    "excluded_runs": "integer",
    "fitted_dimensions": "string_list",
    "input_filter": "boolean",
    "minimum_retained_frames": "integer",
    "minimum_retained_fraction": "number",
    "minimum_residual_design_dof": "integer",
    "minimum_participation_effective_rank": "number",
    "maximum_dominant_temporal_variance_fraction": "number",
    "minimum_usable_runs": "integer",
    "minimum_aggregate_retained_frames": "integer",
    "precision_weighting": "boolean",
    "low_rank": "boolean",
    "dimensions": "integer",
    "oversampling": "integer",
    "power_iterations": "integer",
}

CONFIGURATION_FIELDS = {
    "input_filter": "input_filter",
    **{
        f"inclusion.{name}": name
        for name in (
            "minimum_retained_frames",
            "minimum_retained_fraction",
            "minimum_residual_design_dof",
            "minimum_participation_effective_rank",
            "maximum_dominant_temporal_variance_fraction",
            "minimum_usable_runs",
            "minimum_aggregate_retained_frames",
        )
    },
    "weighting": "precision_weighting",
    "low_rank": "low_rank",
    "low_rank_options.dimensions": "dimensions",
    "low_rank_options.oversampling": "oversampling",
    "low_rank_options.power_iterations": "power_iterations",
}

METHODS_SILENT = {
    "markup": "Selects source markup; its effects are reported through observed counts.",
}


def _run_key(record: ManifestRecord, run: Mapping[str, Any]) -> tuple[object, ...]:
    files = run.get("files") or ()
    return (
        record.entities.get("project"),
        record.entities.get("participant"),
        str(files[0]) if files else run.get("run_index"),
    )


def methods(
    configuration: Mapping[str, Any],
    manifests: Sequence[ManifestRecord],
    *,
    workflow: Mapping[str, Mapping[str, Any]],
) -> ModuleMethods:
    """Summarize the pooling configuration and its fresh participant manifests.

    A run counts once per participant even when several spaces or smoothing
    levels pool it. A run excluded in one target and included in another counts
    as included.
    """
    del workflow
    included: set[tuple[object, ...]] = set()
    excluded: set[tuple[object, ...]] = set()
    dimensions: set[int] = set()
    for record in manifests:
        for run in lookup(record.data, "functional_runs.included", None) or ():
            included.add(_run_key(record, run))
        for run in lookup(record.data, "functional_runs.skipped", None) or ():
            excluded.add(_run_key(record, run))
        fitted = lookup(record.data, "low_rank.dimensions")
        if fitted is not None:
            dimensions.add(int(fitted))
    inclusion = configuration["inclusion"]
    options = configuration["low_rank_options"]
    context = {
        "observed": bool(manifests),
        "participants": participants(manifests),
        "included_runs": len(included),
        "excluded_runs": len(excluded - included),
        "fitted_dimensions": [str(value) for value in sorted(dimensions)],
        "input_filter": bool(configuration.get("input_filter")),
        "minimum_retained_frames": int(inclusion["minimum_retained_frames"]),
        "minimum_retained_fraction": float(inclusion["minimum_retained_fraction"]),
        "minimum_residual_design_dof": int(inclusion["minimum_residual_design_dof"]),
        "minimum_participation_effective_rank": float(
            inclusion["minimum_participation_effective_rank"]
        ),
        "maximum_dominant_temporal_variance_fraction": float(
            inclusion["maximum_dominant_temporal_variance_fraction"]
        ),
        "minimum_usable_runs": int(inclusion["minimum_usable_runs"]),
        "minimum_aggregate_retained_frames": int(inclusion["minimum_aggregate_retained_frames"]),
        "precision_weighting": configuration["weighting"] == "precision",
        "low_rank": bool(configuration["low_rank"]),
        "dimensions": int(options["dimensions"]),
        "oversampling": int(options["oversampling"]),
        "power_iterations": int(options["power_iterations"]),
    }
    return ModuleMethods(context=context)
