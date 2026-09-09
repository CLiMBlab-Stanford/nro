"""Shared CIFTI validation and loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PCONN_INT8_SCALE = 127.0


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
