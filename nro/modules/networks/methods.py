"""Methods-report context for individualized network estimation and labeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nro.methods.context import ManifestRecord, ModuleMethods, participants
from nro.methods.report import MethodsReportError

TITLE = "Individualized networks"
CONTEXT_VERSION = 1

CONTEXT_FIELDS = {
    "observed": "boolean",
    "participants": "integer",
    "spaces": "string_list",
    "smoothing_mm": "string_list",
    "low_rank_source": "boolean",
    "strategy": "string",
    "n_networks": "integer",
    "maximum_dimensions": "integer",
    "reduction_oversampling": "integer",
    "reduction_power_iterations": "integer",
    "ica_max_iterations": "integer",
    "ica_tolerance": "number",
    "ica_upper_quantile": "number",
    "clustering_repetitions": "integer",
    "clustering_n_init": "integer",
    "clustering_max_iterations": "integer",
    "clustering_batch_size": "integer",
    "clustering_max_no_improvement": "integer",
    "clustering_reassignment_ratio": "number",
    "labeling": "boolean",
    "candidates_per_reference": "integer",
}

_SEED = "Fixed seed for reproducibility; recorded in the configuration and manifests."

CONFIGURATION_FIELDS = {
    "parcellation_strategy": "strategy",
    "ica.n_networks": "n_networks",
    "clustering.n_networks": "n_networks",
    "feature_reduction.maximum_dimensions": "maximum_dimensions",
    "feature_reduction.oversampling": "reduction_oversampling",
    "feature_reduction.power_iterations": "reduction_power_iterations",
    "ica.max_iterations": "ica_max_iterations",
    "ica.tolerance": "ica_tolerance",
    "ica.upper_quantile": "ica_upper_quantile",
    "clustering.repetitions": "clustering_repetitions",
    "clustering.n_init": "clustering_n_init",
    "clustering.max_iterations": "clustering_max_iterations",
    "clustering.batch_size": "clustering_batch_size",
    "clustering.max_no_improvement": "clustering_max_no_improvement",
    "clustering.reassignment_ratio": "clustering_reassignment_ratio",
    "labeling.enabled": "labeling",
    "labeling.candidates_per_reference": "candidates_per_reference",
}

METHODS_SILENT = {
    "connectivity_source": "Reported by the upstream connectivity section.",
    "markup": "Selects source markup; its effects are reported through observed counts.",
    "feature_reduction.random_seed": _SEED,
    "ica.random_seed": _SEED,
    "clustering.random_seed": _SEED,
}

SUPPORTED_STRATEGIES = ("ica", "ica_correlation", "clustering")


def methods(
    configuration: Mapping[str, Any],
    manifests: Sequence[ManifestRecord],
    *,
    workflow: Mapping[str, Mapping[str, Any]],
) -> ModuleMethods:
    """Summarize the network estimator, its labeling, and fresh participant manifests.

    Raises MethodsReportError for a strategy that has no methods text.
    """
    strategy = str(configuration["parcellation_strategy"])
    if strategy not in SUPPORTED_STRATEGIES:
        raise MethodsReportError(
            f"Methods text is not yet available for networks strategy {strategy!r}"
        )
    spaces = sorted({str(record.data["space"]) for record in manifests if record.data.get("space")})
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
    reduction = configuration["feature_reduction"]
    ica = configuration["ica"]
    clustering = configuration["clustering"]
    labeling = configuration["labeling"]
    source = workflow.get(str(configuration["connectivity_source"]), {})
    context = {
        "observed": bool(manifests),
        "participants": participants(manifests),
        "spaces": spaces,
        "smoothing_mm": smoothing,
        "low_rank_source": bool(source.get("low_rank", False)),
        "strategy": strategy,
        "n_networks": int((clustering if strategy == "clustering" else ica)["n_networks"]),
        "maximum_dimensions": int(reduction["maximum_dimensions"]),
        "reduction_oversampling": int(reduction["oversampling"]),
        "reduction_power_iterations": int(reduction["power_iterations"]),
        "ica_max_iterations": int(ica["max_iterations"]),
        "ica_tolerance": float(ica["tolerance"]),
        "ica_upper_quantile": float(ica["upper_quantile"]),
        "clustering_repetitions": int(clustering["repetitions"]),
        "clustering_n_init": int(clustering["n_init"]),
        "clustering_max_iterations": int(clustering["max_iterations"]),
        "clustering_batch_size": int(clustering["batch_size"]),
        "clustering_max_no_improvement": int(clustering["max_no_improvement"]),
        "clustering_reassignment_ratio": float(clustering["reassignment_ratio"]),
        "labeling": bool(labeling["enabled"]),
        "candidates_per_reference": int(labeling["candidates_per_reference"]),
    }
    return ModuleMethods(context=context)
