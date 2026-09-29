"""Canonical scientific views of inherited source-BIDS metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nro.engine.bids import bids_dataset_root, resolve_bids_metadata

# Acquisition ordering determines which images become participant references and
# which optional functional references precede a BOLD run.
ACQUISITION_ORDER_FIELDS = frozenset(
    {"AcquisitionDateTime", "AcquisitionTime", "SeriesNumber", "AcquisitionNumber"}
)

# These fields are read by functional reference selection, distortion correction,
# temporal processing, or MARSS. Site hardware matching is already represented by
# the path-independent gradient-unwarping record in the work-item contract.
FUNCTIONAL_METADATA_FIELDS = ACQUISITION_ORDER_FIELDS | frozenset(
    {
        "AcquisitionMatrixPE",
        "B0FieldIdentifier",
        "B0FieldSource",
        "EffectiveEchoSpacing",
        "IntendedFor",
        "MultibandAccelerationFactor",
        "NROReferencePolicy",
        "NROSBRef",
        "PhaseEncodingDirection",
        "ReconMatrixPE",
        "RepetitionTime",
        "SliceEncodingDirection",
        "SliceTiming",
        "TotalReadoutTime",
    }
)


def metadata_fields(module: str) -> frozenset[str]:
    """Return source metadata fields declared scientifically relevant by a module."""
    if module == "anat":
        return ACQUISITION_ORDER_FIELDS
    if module == "func":
        return FUNCTIONAL_METADATA_FIELDS
    return frozenset()


def semantic_metadata_values(values: Mapping[str, Any], *, module: str) -> dict[str, Any]:
    """Project an already resolved metadata mapping onto a module declaration."""
    fields = metadata_fields(module)
    return {key: values[key] for key in sorted(fields) if key in values}


def _portable_source(path: Path) -> str:
    root = bids_dataset_root(path)
    return path.expanduser().absolute().relative_to(root).as_posix()


def semantic_metadata_record(
    image: Path,
    *,
    fields: Iterable[str],
    markup=None,
) -> dict[str, Any]:
    """Resolve one image's declared metadata without retaining sidecar identity."""
    try:
        resolved = resolve_bids_metadata(image, markup=markup)
        values = resolved.values
    except FileNotFoundError:
        values = {}
    selected = {key: values[key] for key in sorted(set(fields)) if key in values}
    return {"source": _portable_source(Path(image)), "fields": selected}


def semantic_metadata_snapshot(
    images: Iterable[Path],
    *,
    module: str,
    markup=None,
) -> list[dict[str, Any]]:
    """Return deterministic semantic metadata for selected source images."""
    fields = metadata_fields(module)
    if not fields:
        return []
    unique = sorted({Path(path).expanduser().absolute() for path in images}, key=str)
    return [semantic_metadata_record(image, fields=fields, markup=markup) for image in unique]
