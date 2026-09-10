"""Typed configuration rules for packaged defaults and site overrides."""

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

from .parsing import DefinitionError


@dataclass(frozen=True)
class Field:
    """Validate one setting and identify whether it affects scientific output.

    Numeric types accept equivalent integral/real YAML spellings, but never
    booleans or numeric strings. Lists remain ordered. Bounds are inclusive
    unless their corresponding exclusive flag is set. No field supplies a
    default value.
    """

    kind: str
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    choices: tuple = ()
    item: "Field | None" = None
    nonempty: bool = False
    execution: bool = False

    def normalize(self, value, location: str):
        """Return a canonical value or raise a field-specific DefinitionError."""

        def fail(message):
            raise DefinitionError(f"{location}: {message}")

        if value is None:
            if self.nullable:
                return None
            fail("null is not allowed")
        if self.kind in {"int", "float"}:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                fail(f"must be a finite {self.kind}")
            if self.kind == "int" and int(value) != value:
                fail("must be an integer")
            value = int(value) if self.kind == "int" else float(value)
            if self.minimum is not None and (
                value < self.minimum or self.exclusive_minimum and value == self.minimum
            ):
                fail(f"must be {'>' if self.exclusive_minimum else '>='} {self.minimum}")
            if self.maximum is not None and (
                value > self.maximum or self.exclusive_maximum and value == self.maximum
            ):
                fail(f"must be {'<' if self.exclusive_maximum else '<='} {self.maximum}")
            if value == 0:
                value = 0 if self.kind == "int" else 0.0
        elif self.kind == "bool":
            if not isinstance(value, bool):
                fail("must be a boolean")
        elif self.kind in {"str", "regex"}:
            if not isinstance(value, str) or self.nonempty and not value.strip():
                fail("must be a nonempty string")
            if self.kind == "regex":
                try:
                    re.compile(value)
                except re.error as error:
                    fail(f"invalid regular expression: {error}")
        elif self.kind == "list":
            if not isinstance(value, list) or self.nonempty and not value:
                fail("must be a nonempty list" if self.nonempty else "must be a list")
            value = [
                self.item.normalize(item, f"{location}[{index}]")
                for index, item in enumerate(value)
            ]
        elif self.kind == "filter":
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or not key for key in value
            ):
                fail("must map BIDS entity names to values")
            normalized = {}
            for key, selected in value.items():
                if selected is None:
                    normalized[key] = None
                    continue
                items = selected if isinstance(selected, list) else [selected]
                if any(
                    isinstance(item, bool) or not isinstance(item, (str, int)) for item in items
                ):
                    fail(f"{key} must select a string, integer, list of these, or null")
                normalized[key] = sorted({str(item) for item in items})
            value = normalized
        else:
            raise RuntimeError(f"Unknown schema field kind: {self.kind}")
        if self.choices and value not in self.choices:
            fail(f"must be one of {self.choices}")
        return value


BOOL = Field("bool")
TEXT = Field("str", nonempty=True)
OPTIONAL_TEXT = Field("str", nullable=True, nonempty=True)
OPTIONAL_SEED = Field("int", minimum=0, nullable=True)
SETUP = Field("str", nullable=True)
COUNT = Field("int", minimum=1)
NONNEGATIVE_INT = Field("int", minimum=0)
POSITIVE = Field("float", minimum=0, exclusive_minimum=True)
NONNEGATIVE = Field("float", minimum=0)
FRACTION = Field("float", minimum=0, maximum=1)
POSITIVE_FRACTION = Field("float", minimum=0, maximum=1, exclusive_minimum=True)
STRINGS = Field("list", item=TEXT)
EXEC_BOOL = Field("bool", execution=True)
EXEC_COUNT = Field("int", minimum=1, execution=True)


def enum(*values):
    """Declare a string option with a closed vocabulary."""
    return Field("str", choices=values)


DENOISING = {
    "confounds_regex": Field("regex"),
    "temporal_mask_regex": Field("regex"),
    "nuisance_variance_explained": POSITIVE_FRACTION,
    "minimum_temporal_rank": NONNEGATIVE_INT,
    "minimum_temporal_rank_fraction": FRACTION,
}
CONTAINER = {
    "image": OPTIONAL_TEXT,
    "engine": TEXT,
    "cleanenv": BOOL,
    "bind": STRINGS,
    "home": OPTIONAL_TEXT,
    "inner_setup": SETUP,
    "no_container": BOOL,
}
THREADS = {
    "nthreads": EXEC_COUNT,
    "nthreads_divisor": EXEC_COUNT,
    "nthreads_min": EXEC_COUNT,
    "force": EXEC_BOOL,
    "verbose": EXEC_BOOL,
}

SCHEMAS = {
    "preprocessing": {
        "container": CONTAINER,
        "anat": {
            **THREADS,
            "selection_strategy": enum("first", "robust_average"),
            "mni_template": TEXT,
            "synthstrip_container": TEXT,
            "freesurfer_subjects_dir": OPTIONAL_TEXT,
            "fs_subject": OPTIONAL_TEXT,
        },
        "func": {
            **THREADS,
            "sdc_method": enum("syn", "synbold_disco"),
            "synbold_disco_image": TEXT,
            "synbold_disco_license": TEXT,
            "bbregister_surf": enum("white", "pial"),
            "bbregister_init": enum("coreg", "fsl", "header", "rr"),
            "bbregister_dof": Field("int", choices=(6, 9, 12)),
            "output_grid": enum("t1_native", "t1_epi_vox"),
            "topup_config": TEXT,
            "ica_aroma_cmd": OPTIONAL_TEXT,
            "use_jacobian": BOOL,
            "fieldmap_syn_refine": BOOL,
            "synbold_overlap_erosion_voxels": NONNEGATIVE_INT,
            "synbold_min_overlap_voxels": COUNT,
            "synbold_max_rigid_translation_mm": POSITIVE,
            "synbold_max_rigid_rotation_degrees": POSITIVE,
            "sbref_max_rigid_displacement_mm": POSITIVE,
            "sbref_max_rigid_rotation_degrees": POSITIVE,
            "sbref_min_support_overlap": FRACTION,
            "sbref_min_intensity_correlation": Field("float", minimum=-1, maximum=1),
            **{
                f"syn_{stage}_{key}": TEXT
                for stage in ("base", "refine")
                for key in ("transform", "convergence", "shrink_factors", "smoothing_sigmas")
            },
            "clean_ica_aroma": BOOL,
            "ica_aroma_denoise_type": enum("nonaggr", "aggr", "both"),
            "sdc_from_sbref_pair": BOOL,
            "debug_first_nvols": NONNEGATIVE_INT,
            "io_chunk_vols": EXEC_COUNT,
            "output_spaces": Field("list", item=TEXT, nonempty=True),
        },
        "confounds": {
            "aseg_in_epi": OPTIONAL_TEXT,
            "brain_mask_in_epi": OPTIONAL_TEXT,
            "n_acompcor": NONNEGATIVE_INT,
            "acompcor_max_voxels": COUNT,
            "fd_radius_mm": POSITIVE,
            "motion_outlier_fd_thresh": POSITIVE,
            "nonsteady_max_vols": NONNEGATIVE_INT,
            "nonsteady_rel_thresh": FRACTION,
            "nonsteady_stable_run": COUNT,
        },
    },
    "clean": {
        **DENOISING,
        "min_trs": COUNT,
        "gm_mask_threshold": FRACTION,
        "standardize": BOOL,
        "detrend": BOOL,
        "regress_out_task": BOOL,
        "low_pass": Field("float", nullable=True, minimum=0, exclusive_minimum=True),
        "high_pass": Field("float", nullable=True, minimum=0),
        "force": EXEC_BOOL,
        "verbose": EXEC_BOOL,
        "container": OPTIONAL_TEXT,
        "no_container": BOOL,
        "container_engine": TEXT,
        "container_cleanenv": BOOL,
        "container_bind": STRINGS,
        "container_home": OPTIONAL_TEXT,
        "container_inner_setup": SETUP,
        "wb_command": TEXT,
    },
    "firstlevels": {
        **DENOISING,
        "aggregation_weighting": enum("equal", "precision"),
        "noise_model": enum("ols", "ar1"),
        "ar_grid": Field(
            "list",
            item=Field(
                "float", minimum=-1, maximum=1, exclusive_minimum=True, exclusive_maximum=True
            ),
            nonempty=True,
        ),
        "spatial_block_size": EXEC_COUNT,
        "wb_command": TEXT,
    },
    "microparcellation": {
        "input_filter": Field("filter"),
        "surface": enum("pial", "midthickness", "white", "inflated"),
        "mask": OPTIONAL_TEXT,
        "mask_threshold": Field("float", minimum=0, maximum=1, exclusive_maximum=True),
        "volume_connectivity": Field("int", choices=(6, 18, 26)),
        "output_dir": OPTIONAL_TEXT,
        "prefix": OPTIONAL_TEXT,
        "overwrite": EXEC_BOOL,
        "coarsening": {
            "target_vertices": Field("int", minimum=2),
            "iterations": COUNT,
            "exponential_temperature": POSITIVE,
            "eigenvectors": Field("int", minimum=2),
            "max_levels": COUNT,
            "eigensolver_tolerance": NONNEGATIVE,
        },
        "connectivity": {
            "minimum_retained_frames": Field("int", minimum=2),
            "minimum_retained_fraction": FRACTION,
            "minimum_residual_design_dof": NONNEGATIVE_INT,
            "minimum_participation_effective_rank": NONNEGATIVE,
            "maximum_dominant_temporal_variance_fraction": FRACTION,
            "minimum_usable_runs": COUNT,
            "minimum_aggregate_retained_frames": Field("int", minimum=2),
            "temporal_block_size": EXEC_COUNT,
            "reliability_weighting": BOOL,
            "reliability_vertex_block_size": EXEC_COUNT,
            "global_signal_regression": BOOL,
        },
        "quality": {
            "split_half_block_frames": COUNT,
            "null_parcellations": COUNT,
            "random_seed": NONNEGATIVE_INT,
            "region_growing_attempts": COUNT,
            "connectome_power_iterations": COUNT,
        },
    },
    "dynconn": {
        "input_filter": Field("filter"),
        "surface": enum("pial", "midthickness", "white", "inflated"),
        "output_dir": OPTIONAL_TEXT,
        "prefix": OPTIONAL_TEXT,
        "overwrite": EXEC_BOOL,
        "inclusion": {
            "minimum_retained_frames": Field("int", minimum=2),
            "minimum_retained_fraction": FRACTION,
            "minimum_residual_design_dof": NONNEGATIVE_INT,
            "minimum_participation_effective_rank": NONNEGATIVE,
            "maximum_dominant_temporal_variance_fraction": FRACTION,
            "minimum_usable_runs": COUNT,
            "minimum_aggregate_retained_frames": Field("int", minimum=2),
        },
    },
    "networks": {
        "output_dir": OPTIONAL_TEXT,
        "prefix": OPTIONAL_TEXT,
        "overwrite": EXEC_BOOL,
        "parcellation_strategy": enum("ica", "clustering", "oslom"),
        "connectivity": {
            "transform": enum("clip_positive", "absolute", "square"),
            "minimum_weight": NONNEGATIVE,
            "percentile_cutoff": Field("float", nullable=True, minimum=0, maximum=100),
        },
        "ica": {
            "n_networks": Field("int", minimum=2),
            "random_seed": OPTIONAL_SEED,
            "max_iterations": COUNT,
            "tolerance": POSITIVE,
            "upper_quantile": POSITIVE_FRACTION,
            "svd_oversamples": COUNT,
            "svd_power_iterations": NONNEGATIVE_INT,
        },
        "clustering": {
            "n_networks": Field("int", minimum=2),
            "repetitions": COUNT,
            "random_seed": OPTIONAL_SEED,
            "n_init": COUNT,
            "max_iterations": COUNT,
            "batch_size": COUNT,
            "max_no_improvement": Field("int", nullable=True, minimum=1),
            "reassignment_ratio": FRACTION,
        },
        "oslom": {
            "executable": OPTIONAL_TEXT,
            "initialization": enum("none", "leiden", "file"),
            "initial_partition": OPTIONAL_TEXT,
            "leiden_resolution": POSITIVE,
            "leiden_iterations": COUNT,
            "leiden_seed": OPTIONAL_SEED,
            "weighted": BOOL,
            "directed": Field("bool", choices=(False,)),
            "significance": Field(
                "float", minimum=0, maximum=1, exclusive_minimum=True, exclusive_maximum=True
            ),
            "repetitions": COUNT,
            "internal_runs": COUNT,
            "hierarchical_runs": NONNEGATIVE_INT,
            "extra_args": STRINGS,
            "timeout_seconds": Field(
                "float", nullable=True, minimum=0, exclusive_minimum=True, execution=True
            ),
        },
        "consensus": {
            "assignment_threshold": FRACTION,
            "homeless_threshold": FRACTION,
            "minimum_match_jaccard": FRACTION,
        },
        "labeling": {"enabled": BOOL, "candidates_per_reference": COUNT},
    },
}

RUNTIME_FIELDS = {
    "preprocessing": {},
    "clean": {"preprocessing_directory": TEXT},
    "firstlevels": {"preprocessing_directory": TEXT, "preprocessing_aroma": BOOL},
    "microparcellation": {"preprocessing_directory": TEXT, "clean_directory": TEXT},
    "dynconn": {"preprocessing_directory": TEXT, "clean_directory": TEXT},
    "networks": {"microparcellation_directory": TEXT},
}


def normalize_fields(
    schema: dict, values: dict, *, location: str = "configuration", complete: bool = True
) -> dict:
    """Validate nested fields, requiring all fields for complete configurations."""
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise DefinitionError(f"{location}: must be a mapping with string keys")
    unknown = set(values) - set(schema)
    if unknown:
        raise DefinitionError(
            f"{location}: Unknown configuration option: {', '.join(sorted(unknown))}"
        )
    missing = set(schema) - set(values)
    if complete and missing:
        raise DefinitionError(f"{location}: missing keys: {', '.join(sorted(missing))}")
    result = {}
    for key, value in values.items():
        rule = schema[key]
        name = f"{location}.{key}"
        result[key] = (
            normalize_fields(rule, value, location=name, complete=complete)
            if isinstance(rule, dict)
            else rule.normalize(value, name)
        )
    return result


def _relationships(kind: str, values: dict) -> None:
    if kind == "clean":
        low, high = values["low_pass"], values["high_pass"]
        if low is not None and high is not None and high >= low:
            raise DefinitionError("clean.high_pass must be below clean.low_pass")
    elif kind == "firstlevels":
        if len(set(values["ar_grid"])) != len(values["ar_grid"]):
            raise DefinitionError("firstlevels.ar_grid must contain distinct coefficients")
    elif kind == "microparcellation":
        connectivity = values["connectivity"]
        if connectivity["reliability_weighting"] and connectivity["minimum_retained_frames"] < 8:
            raise DefinitionError(
                "connectivity.minimum_retained_frames must be at least 8 for quarter-split reliability"
            )
    elif kind == "networks":
        oslom = values["oslom"]
        if (
            values["parcellation_strategy"] == "oslom"
            and oslom["initialization"] == "file"
            and oslom["initial_partition"] is None
        ):
            raise DefinitionError(
                "oslom.initial_partition is required when oslom.initialization is file"
            )


def validate_parameters(kind: str, values: dict) -> None:
    """Check module API parameter groups using the configuration field rules.

    Callers supply all groups needed by that module's cross-field checks, but
    omit data inputs and output locations. Paths and tuples are serialized as
    they are in runtime snapshots. This function performs no filesystem checks.
    """
    values = json.loads(json.dumps(values, default=str))
    normalized = normalize_fields(SCHEMAS[kind], values, location=kind, complete=False)
    _relationships(kind, normalized)


@lru_cache(maxsize=256)
def _compile(kind: str, serialized: str, runtime: bool) -> dict:
    schema = SCHEMAS[kind]
    if runtime:
        schema = {**schema, **RUNTIME_FIELDS[kind]}
    result = normalize_fields(schema, json.loads(serialized), location=kind)
    _relationships(kind, result)
    return result


def compile_configuration(kind: str, values: dict, *, runtime: bool = False) -> dict:
    """Validate resolved settings without reading data or applying defaults.

    The bounded cache is keyed by content, not paths or timestamps. Each caller
    receives an independent copy. Runtime snapshots require workflow-injected
    upstream fields; author-written configurations cannot supply them.
    """
    if kind not in SCHEMAS:
        raise DefinitionError(f"Unknown configuration class: {kind}")
    try:
        return deepcopy(_compile(kind, json.dumps(values, sort_keys=True), runtime))
    except (TypeError, ValueError) as error:
        raise DefinitionError(f"{kind}: {error}") from error


def scientific_values(kind: str, values: dict) -> dict:
    """Remove declared execution settings from an already resolved snapshot.

    Preserve unknown fields and ordered lists. This also accepts module runtime
    snapshots whose input/output structure differs from author-written configs.
    """

    def select(schema, mapping):
        result = {}
        for key, value in mapping.items():
            rule = schema.get(key)
            if isinstance(rule, Field) and rule.execution:
                continue
            result[key] = (
                select(rule, value)
                if isinstance(rule, dict) and isinstance(value, dict)
                else deepcopy(value)
            )
        return result

    schema = SCHEMAS[kind]
    if kind in {"dynconn", "microparcellation", "networks"}:
        schema = {
            **schema,
            "output": {"overwrite": EXEC_BOOL, "work_directory": Field("str", execution=True)},
        }
    return select(schema, values)
