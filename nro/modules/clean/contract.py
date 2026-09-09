"""Semantic output contract for cleaned functional derivatives."""

from __future__ import annotations

from collections.abc import Mapping

from nro.engine.artifact_metadata import validate_metadata_fields

# The planner fingerprints this declaration as part of every clean artifact
# contract, and the writer validates the same declaration before publishing.
# Public metadata changes therefore stale affected artifacts without making an
# implementation or package version part of scientific freshness.
CLEAN_SIDECAR_FIELDS: dict[str, str] = {
    "Description": "string",
    "Sources": "string_list",
    "Cleaning.CleanID": "string",
    "Cleaning.PreprocessingID": "string",
    "Cleaning.Space": "string",
    "Cleaning.InputDescription": "string",
    "Cleaning.MinTRs": "integer",
    "Cleaning.SmoothedInput": "nullable_string",
    "Cleaning.SmoothingFWHMMM": "number",
    "Cleaning.ConfoundsRegex": "string",
    "Cleaning.TemporalMaskRegex": "string",
    "Cleaning.TemporalMaskFile": "string",
    "Cleaning.CleaningDefined": "boolean",
    "Cleaning.CleaningUndefinedReason": "nullable_string",
    "Cleaning.OutputDataStatus": "string",
    "Cleaning.TotalFrames": "integer",
    "Cleaning.RetainedFrames": "integer",
    "Cleaning.CensoredFrames": "integer",
    "Cleaning.CensoredFraction": "number",
    "Cleaning.RetainedDurationSeconds": "number",
    "Cleaning.LongestCensoredIntervalFrames": "integer",
    "Cleaning.LongestCensoredIntervalSeconds": "number",
    "Cleaning.TemporalMaskColumnCount": "integer",
    "Cleaning.PassbandBasisDimension": "integer",
    "Cleaning.PassbandBasisRank": "integer",
    "Cleaning.PassbandBasisConditionNumber": "nullable_number",
    "Cleaning.ExactDesignRank": "integer",
    "Cleaning.PostExactTemporalRank": "integer",
    "Cleaning.RegressionDesignRank": "integer",
    "Cleaning.ResidualDesignDegreesOfFreedom": "integer",
    "Cleaning.NuisanceInputColumnCount": "integer",
    "Cleaning.NuisanceUsableColumnCount": "integer",
    "Cleaning.NuisancePCAComponentCount": "integer",
    "Cleaning.NuisancePCAVarianceTarget": "number",
    "Cleaning.NuisancePCAVarianceExplained": "number",
    "Cleaning.NuisancePCAVarianceTargetReached": "boolean",
    "Cleaning.NuisanceSelectionLimitedByTemporalRank": "boolean",
    "Cleaning.MinimumTemporalRank": "integer",
    "Cleaning.MinimumTemporalRankFraction": "number",
    "Cleaning.ProtectedTemporalRank": "integer",
    "Cleaning.TemporalRankFloorSatisfied": "boolean",
    "Cleaning.AlgebraicTemporalRank": "integer",
    "Cleaning.TemporalFilterMethod": "string",
    "Cleaning.Standardize": "boolean",
    "Cleaning.Detrend": "boolean",
    "Cleaning.RegressOutTask": "boolean",
    "Cleaning.LowPassHz": "nullable_number",
    "Cleaning.HighPassHz": "nullable_number",
    "Cleaning.ConfigurationFingerprint": "nullable_string",
    "Cleaning.QualityControl.AlgebraicTemporalRank": "integer",
    "Cleaning.QualityControl.ObservedTemporalRank": "integer",
    "Cleaning.QualityControl.EntropyEffectiveTemporalRank": "number",
    "Cleaning.QualityControl.ParticipationRatioEffectiveTemporalRank": "number",
    "Cleaning.QualityControl.DominantTemporalVarianceFraction": "number",
    "Cleaning.QualityControl.EffectiveRankLocationSampleCount": "integer",
    "Cleaning.QualityControl.EffectiveRankLocationSampling": "string",
    "Cleaning.QualityControl.NonconstantLocationCount": "integer",
    "Cleaning.QualityControl.ConstantLocationCount": "integer",
    "Cleaning.QualityControl.Definitions.AlgebraicTemporalRank": "string",
    "Cleaning.QualityControl.Definitions.ObservedTemporalRank": "string",
    "Cleaning.QualityControl.Definitions.EntropyEffectiveTemporalRank": "string",
    "Cleaning.QualityControl.Definitions.ParticipationRatioEffectiveTemporalRank": "string",
    "Cleaning.QualityControl.Definitions.DominantTemporalVarianceFraction": "string",
}

CLEAN_VOLUME_SIDECAR_FIELDS: dict[str, str] = {
    "Cleaning.GrayMatterMask": "string",
    "Cleaning.GrayMatterMaskThreshold": "number",
}

CLEAN_MANIFEST_FIELDS: dict[str, str] = {
    "manifest_version": "integer",
    "module": "string",
    "run_stem": "string",
    "space": "string",
    "smoothing_fwhm_mm": "integer",
    "source_bold": "string",
    "targets": "list",
    "public_outputs": "string_list",
    "output_metadata_contract": "mapping",
    "configuration": "mapping",
    "configuration_fingerprint": "nullable_string",
    "complete": "boolean",
}


def clean_output_contract() -> dict[str, object]:
    """Return the freshness-relevant public metadata schema."""
    return {
        "publication_manifest_fields": dict(CLEAN_MANIFEST_FIELDS),
        "sidecar_fields": dict(CLEAN_SIDECAR_FIELDS),
        "volume_sidecar_fields": dict(CLEAN_VOLUME_SIDECAR_FIELDS),
    }


def validate_clean_sidecar(
    document: Mapping[str, object],
    *,
    volume: bool,
) -> None:
    """Validate one cleaned BOLD sidecar against its public contract."""
    fields = dict(CLEAN_SIDECAR_FIELDS)
    if volume:
        fields.update(CLEAN_VOLUME_SIDECAR_FIELDS)
    validate_metadata_fields(document, fields, label="Cleaned sidecar")


def validate_clean_manifest(document: Mapping[str, object]) -> None:
    """Validate required metadata field types and structure; reject an incomplete publication contract."""
    validate_metadata_fields(
        document,
        CLEAN_MANIFEST_FIELDS,
        label="Cleaning publication manifest",
    )
