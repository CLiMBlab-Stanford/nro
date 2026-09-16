"""Step support for functional preprocessing."""

import logging
from pathlib import Path
from typing import Any, Optional

from nro.engine.bids import (
    bids_entity,
    bids_readout_time,
)

LOG = logging.getLogger("func")


def _resolve_sdc_reference_policy(
    *,
    requested_sdc_method: str,
    fieldmap_pair_available: bool,
    fieldmap_syn_refine: bool,
) -> tuple[bool, Optional[str]]:
    """Resolve synthetic-reference use and the post-fieldmap refinement target."""
    use_synbold_reference = requested_sdc_method == "synbold_disco" and not fieldmap_pair_available
    fieldmap_refinement_target = (
        "T1wAnatomicalSyN" if fieldmap_pair_available and fieldmap_syn_refine else None
    )
    return use_synbold_reference, fieldmap_refinement_target


def _resolve_fieldmapless_sdc_method(
    requested_sdc_method: str,
    *,
    fieldmap_pair_available: bool,
    bold_metadata: dict[str, Any],
) -> tuple[str, Optional[str]]:
    """Resolve a metadata-compatible fieldmapless SDC method."""
    if requested_sdc_method != "synbold_disco" or fieldmap_pair_available:
        return requested_sdc_method, None
    missing: list[str] = []
    phase_encoding_direction = str(bold_metadata.get("PhaseEncodingDirection", "")).strip()
    if phase_encoding_direction not in {"i", "i-", "j", "j-", "k", "k-"}:
        missing.append("PhaseEncodingDirection")
    try:
        readout_time = float(bids_readout_time(bold_metadata))
        if readout_time <= 0:
            raise ValueError("readout time must be positive")
    except (KeyError, TypeError, ValueError):
        missing.append("TotalReadoutTime or EffectiveEchoSpacing with a phase-encoding matrix size")
    if not missing:
        return requested_sdc_method, None
    return (
        "syn",
        "synbold_disco requested without a usable reverse-PE fieldmap pair, but "
        "the effective BIDS metadata lack "
        + " and ".join(missing)
        + "; using ordinary anatomical SyN",
    )


def _with_suffix(stem: str, suffix: str) -> str:
    if stem.endswith("_bold"):
        return stem[: -len("_bold")] + suffix
    return stem + suffix


def _space_name_for_log(path: Path) -> str:
    return bids_entity(path, "space", default="Unknown") or "Unknown"


def _ica_aroma_output_label(denoise_type: str) -> str:
    mode = str(denoise_type).strip().lower()
    if mode == "aggr":
        return "aromaAgg"
    return "aromaNonAgg"
