"""Shared CIFTI validation and loading helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from nro.engine.io import atomic_write_json

PCONN_INT8_SCALE = 127.0
INDEXED_CIFTI_SCHEMA = "nro-indexed-cifti-v1"


def _nib():
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("nibabel is required for CIFTI input/output") from error
    return nib


def load_dlabel(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    """Load positive CIFTI labels as zero-based assignments."""
    nib = _nib()
    image = nib.load(str(path))
    label_axis = image.header.get_axis(0)
    brain_axis = image.header.get_axis(1)
    if not isinstance(label_axis, nib.cifti2.LabelAxis) or not isinstance(
        brain_axis, nib.cifti2.BrainModelAxis
    ):
        raise ValueError(f"Expected label-by-brain-model CIFTI: {path}")
    encoded = np.asarray(image.dataobj[0]).reshape(-1)
    if not np.all(np.isfinite(encoded)) or not np.all(encoded == np.rint(encoded)):
        raise ValueError(f"CIFTI labels are not finite integers: {path}")
    counts = []
    arrays = []
    for _, structure_slice, structure_axis in brain_axis.iter_structures():
        values = encoded[structure_slice]
        arrays.append(np.where(values > 0, values - 1, -1).astype(np.int64))
        counts.append(len(structure_axis))
    return np.concatenate(arrays), tuple(counts)


def load_pconn(path: Path) -> np.ndarray:
    """Load and validate a symmetric parcel-connectivity CIFTI matrix."""
    nib = _nib()
    image = nib.load(str(path))
    axes = (image.header.get_axis(0), image.header.get_axis(1))
    if not all(isinstance(axis, nib.cifti2.ParcelsAxis) for axis in axes):
        raise ValueError(f"Expected parcel-by-parcel CIFTI: {path}")
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Parcel connectivity is not square: {path}")
    correlations = np.asarray(image.dataobj, dtype=np.float32)
    for start in range(0, correlations.shape[0], 512):
        stop = min(start + 512, correlations.shape[0])
        if not np.allclose(
            correlations[start:stop],
            correlations[:, start:stop].T,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(f"Parcel connectivity is not symmetric: {path}")
    return correlations


def indexed_cifti_sidecar(path: Path) -> Path:
    """Return the JSON sidecar path for a CIFTI image."""
    path = Path(path)
    if not path.name.endswith(".nii"):
        raise ValueError(f"Expected a CIFTI .nii path: {path}")
    return path.with_name(path.name.removesuffix(".nii") + ".json")


def _reverse_index(
    records: Sequence[Mapping[str, object]], lookup_fields: Sequence[str]
) -> dict[str, list[dict[str, object]]]:
    reverse: dict[str, list[dict[str, object]]] = {}
    for field in lookup_fields:
        groups: list[dict[str, object]] = []
        for index, record in enumerate(records):
            raw = record.get(field)
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                if value is None or not isinstance(value, (str, int, float, bool)):
                    raise ValueError(
                        f"Indexed CIFTI lookup field {field!r} must be scalar or a list"
                    )
                match = next((item for item in groups if item["Value"] == value), None)
                if match is None:
                    groups.append({"Value": value, "Indices": [index]})
                else:
                    indices = match["Indices"]
                    if not isinstance(indices, list):  # pragma: no cover - local invariant
                        raise TypeError("Invalid reverse-index accumulator")
                    indices.append(index)
        reverse[field] = groups
    return reverse


def write_indexed_cifti_sidecar(
    path: Path,
    records: Sequence[Mapping[str, object]],
    *,
    lookup_fields: Sequence[str],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Describe every CIFTI map and write typed reverse indexes for selected fields."""
    path = Path(path)
    image = _nib().load(str(path))
    axis = image.header.get_axis(0)
    if not isinstance(axis, _nib().cifti2.ScalarAxis):
        raise ValueError(f"Expected a dense-scalar CIFTI: {path}")
    records = [dict(record) for record in records]
    names = [str(record.get("Name", "")) for record in records]
    if (
        len(records) != image.shape[0]
        or names != axis.name.tolist()
        or any(not name for name in names)
    ):
        raise ValueError("Indexed CIFTI records must match the ordered scalar-axis names")
    if (
        any(not isinstance(field, str) or not field for field in lookup_fields)
        or len(set(lookup_fields)) != len(lookup_fields)
        or any(field not in record for field in lookup_fields for record in records)
    ):
        raise ValueError("Indexed CIFTI lookup fields must be unique and present on every map")
    sidecar = indexed_cifti_sidecar(path)
    atomic_write_json(
        sidecar,
        {
            "Schema": INDEXED_CIFTI_SCHEMA,
            "CIFTI": path.name,
            "MapAxis": 0,
            "IndexBase": 0,
            "Metadata": dict(metadata or {}),
            "LookupFields": list(lookup_fields),
            "IndexToMetadata": {str(index): record for index, record in enumerate(records)},
            "MetadataToIndices": _reverse_index(records, lookup_fields),
        },
    )
    return sidecar


def validate_indexed_cifti_sidecar(path: Path) -> dict[str, object]:
    """Validate an indexed dense-scalar sidecar against its CIFTI map axis."""
    path = Path(path)
    document = json.loads(indexed_cifti_sidecar(path).read_text(encoding="utf-8"))
    required = {
        "Schema",
        "CIFTI",
        "MapAxis",
        "IndexBase",
        "Metadata",
        "LookupFields",
        "IndexToMetadata",
        "MetadataToIndices",
    }
    if set(document) != required or document["Schema"] != INDEXED_CIFTI_SCHEMA:
        raise ValueError(f"Invalid indexed CIFTI sidecar schema: {path}")
    if document["CIFTI"] != path.name or document["MapAxis"] != 0 or document["IndexBase"] != 0:
        raise ValueError(f"Indexed CIFTI sidecar identifies a different image or axis: {path}")
    lookup_fields = document["LookupFields"]
    index = document["IndexToMetadata"]
    if (
        not isinstance(document["Metadata"], dict)
        or not isinstance(lookup_fields, list)
        or not isinstance(index, dict)
        or not isinstance(document["MetadataToIndices"], dict)
        or any(not isinstance(field, str) or not field for field in lookup_fields)
        or len(set(lookup_fields)) != len(lookup_fields)
    ):
        raise ValueError(f"Invalid indexed CIFTI metadata structure: {path}")
    records = [index.get(str(value)) for value in range(len(index))]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Indexed CIFTI map indices are not contiguous: {path}")
    image = _nib().load(str(path))
    axis = image.header.get_axis(0)
    if not isinstance(axis, _nib().cifti2.ScalarAxis):
        raise ValueError(f"Expected a dense-scalar CIFTI: {path}")
    if (
        image.shape[0] != len(records)
        or [record["Name"] for record in records] != axis.name.tolist()
    ):
        raise ValueError(f"Indexed CIFTI metadata differs from its scalar axis: {path}")
    expected_reverse = _reverse_index(records, lookup_fields)
    if document["MetadataToIndices"] != expected_reverse:
        raise ValueError(f"Indexed CIFTI reverse lookup is inconsistent: {path}")
    return document


def indexed_cifti_indices(path: Path, field: str, value: object) -> tuple[int, ...]:
    """Return all map indices matching one typed metadata value."""
    document = validate_indexed_cifti_sidecar(path)
    try:
        groups = document["MetadataToIndices"][field]  # type: ignore[index]
    except KeyError as error:
        raise KeyError(f"CIFTI sidecar has no lookup field {field!r}: {path}") from error
    for group in groups:
        if group["Value"] == value:
            return tuple(int(index) for index in group["Indices"])
    return ()


def load_indexed_cifti_map(path: Path, field: str, value: object) -> np.ndarray:
    """Load the unique CIFTI map selected by a sidecar metadata value."""
    indices = indexed_cifti_indices(path, field, value)
    if len(indices) != 1:
        raise ValueError(
            f"Expected one CIFTI map for {field}={value!r}, found {len(indices)}: {path}"
        )
    return np.asarray(_nib().load(str(path)).dataobj[indices[0]], dtype=np.float32)
