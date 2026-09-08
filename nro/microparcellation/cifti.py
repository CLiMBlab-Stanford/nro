from __future__ import annotations

import colorsys
import logging
import os
from pathlib import Path

import numpy as np

from nibabel.processing import resample_to_output

INT8_SCALE = 127.0
LOG = logging.getLogger(__name__)


def _nib():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for CIFTI input/output") from exc
    return nib


def _structures(surface_paths: tuple[Path, ...]) -> tuple[str, ...]:
    if len(surface_paths) == 2:
        return ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")
    if len(surface_paths) != 1:
        raise ValueError("CIFTI output requires one surface or an ordered left/right pair")
    name = surface_paths[0].name
    side = "RIGHT" if "_hemi-R_" in name or ".R." in name else "LEFT"
    return (f"CIFTI_STRUCTURE_CORTEX_{side}",)


def _parcel_names(count: int) -> np.ndarray:
    width = max(5, len(str(count)))
    return np.asarray([f"microparcel_{index:0{width}d}" for index in range(1, count + 1)])


def _parcel_color(index: int) -> tuple[float, float, float, float]:
    hue = (index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return red, green, blue, 1.0


def _axes(
    labels: np.ndarray,
    vertex_counts: tuple[int, ...],
    surface_paths: tuple[Path, ...],
):
    nib = _nib()
    structures = _structures(surface_paths)
    if len(vertex_counts) != len(structures) or labels.size != sum(vertex_counts):
        raise ValueError("Surface structure and microparcel assignment sizes differ")
    active = labels >= 0
    if not np.any(active):
        raise ValueError("No vertices have microparcel assignments")
    count = int(labels[active].max()) + 1
    if not np.array_equal(np.unique(labels[active]), np.arange(count)):
        raise ValueError("Microparcel assignments must be contiguous and zero-based")

    brain_axis = None
    offset = 0
    for structure, n_vertices in zip(structures, vertex_counts):
        axis = nib.cifti2.BrainModelAxis.from_surface(
            np.arange(n_vertices), n_vertices, name=structure
        )
        brain_axis = axis if brain_axis is None else brain_axis + axis
        offset += n_vertices

    names = _parcel_names(count)
    voxels = np.empty(count, dtype=object)
    vertices = np.empty(count, dtype=object)
    for parcel in range(count):
        voxels[parcel] = np.empty((0, 3), dtype=np.int64)
        vertices[parcel] = {}
    offset = 0
    for structure, n_vertices in zip(structures, vertex_counts):
        hemi_labels = labels[offset:offset + n_vertices]
        active_vertices = np.flatnonzero(hemi_labels >= 0)
        order = np.argsort(hemi_labels[active_vertices], kind="stable")
        sorted_vertices = active_vertices[order]
        sorted_labels = hemi_labels[sorted_vertices]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
        stops = np.r_[starts[1:], len(sorted_labels)]
        for start, stop in zip(starts, stops):
            vertices[int(sorted_labels[start])][structure] = sorted_vertices[start:stop]
        offset += n_vertices
    parcel_axis = nib.cifti2.ParcelsAxis(
        names,
        voxels,
        vertices,
        nvertices=dict(zip(structures, vertex_counts)),
    )
    return brain_axis, parcel_axis, names


def write_dlabel(
    path: Path,
    labels: np.ndarray,
    vertex_counts: tuple[int, ...],
    surface_paths: tuple[Path, ...],
):
    nib = _nib()
    brain_axis, parcel_axis, names = _axes(labels, vertex_counts, surface_paths)
    table = {0: ("unassigned", (0.0, 0.0, 0.0, 0.0))}
    table.update({index: (str(name), _parcel_color(index)) for index, name in enumerate(names, start=1)})
    label_axis = nib.cifti2.LabelAxis(["microparcels"], [table])
    data = np.where(labels >= 0, labels + 1, 0).astype(np.int32)[None, :]
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))
    nib.save(nib.Cifti2Image(data, header=header, dtype=np.int32), str(path))
    return Path(path), parcel_axis


def surface_parcel_axis(
    labels: np.ndarray,
    vertex_counts: tuple[int, ...],
    surface_paths: tuple[Path, ...],
):
    """Construct the parcel axis without rewriting an existing dlabel."""
    return _axes(labels, vertex_counts, surface_paths)[1]


def _is_plumb_affine(affine: np.ndarray, *, tolerance: float = 1e-5) -> bool:
    """Return whether voxel axes are aligned with Workbench's physical axes."""
    linear = np.asarray(affine, dtype=np.float64)[:3, :3]
    dominant_rows = np.argmax(np.abs(linear), axis=0)
    if len(np.unique(dominant_rows)) != 3:
        return False
    off_axis = linear.copy()
    off_axis[dominant_rows, np.arange(3)] = 0.0
    scale = max(1.0, float(np.max(np.abs(linear))))
    return bool(np.max(np.abs(off_axis)) <= tolerance * scale)


def _volume_cifti_grid(
    labels: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Workbench-compatible grid without dropping any parcels."""
    nib = _nib()
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=np.float64)
    if labels.shape != (int(mask.sum()),):
        raise ValueError("Gray-matter mask and microparcel assignment sizes differ")
    if _is_plumb_affine(affine):
        return labels, mask, affine

    # Workbench warns that an oblique CIFTI volume is "not Plumb" and does
    # not render its voxel brain models.  The ordinary label NIfTI remains in
    # the native grid; only the CIFTI brainordinate representation is placed
    # on an axis-aligned grid.  Refine as necessary because nearest-neighbor
    # resampling at the native spacing can erase one-voxel parcels.
    encoded = np.zeros(mask.shape, dtype=np.int32)
    encoded[mask] = labels.astype(np.int32, copy=False) + 1
    image = nib.Nifti1Image(encoded, affine)
    native_spacing = nib.affines.voxel_sizes(affine)
    expected = np.unique(labels[labels >= 0])
    factor = 1.0
    while factor >= 0.25:
        resampled = resample_to_output(
            image,
            voxel_sizes=native_spacing * factor,
            order=0,
        )
        data = np.rint(np.asarray(resampled.dataobj)).astype(np.int64, copy=False)
        display_mask = data > 0
        display_labels = data[display_mask] - 1
        if np.array_equal(np.unique(display_labels), expected):
            LOG.info(
                "Resampled oblique volumetric CIFTI mapping to a plumb grid "
                "at %.3fx native voxel spacing (%d brainordinates)",
                factor,
                int(display_mask.sum()),
            )
            return display_labels, display_mask, np.asarray(resampled.affine)
        factor *= 0.75
    raise RuntimeError(
        "Could not construct a plumb volumetric CIFTI grid without dropping microparcels"
    )


def volume_parcel_axis(labels: np.ndarray, mask: np.ndarray, affine: np.ndarray):
    """Construct the voxel-based parcel axis without writing a CIFTI file."""
    nib = _nib()
    labels, mask, affine = _volume_cifti_grid(labels, mask, affine)
    voxel_indices = np.argwhere(mask).astype(np.int64)
    active = labels >= 0
    if not np.any(active):
        raise ValueError("No gray-matter voxels have microparcel assignments")
    count = int(labels[active].max()) + 1
    if not np.array_equal(np.unique(labels[active]), np.arange(count)):
        raise ValueError("Microparcel assignments must be contiguous and zero-based")
    names = _parcel_names(count)
    voxels = np.empty(count, dtype=object)
    vertices = np.empty(count, dtype=object)
    for parcel in range(count):
        voxels[parcel] = voxel_indices[labels == parcel]
        vertices[parcel] = {}
    return nib.cifti2.ParcelsAxis(
        names,
        voxels,
        vertices,
        affine=affine,
        volume_shape=mask.shape,
    )


def write_volume_dlabel(path: Path, labels: np.ndarray, mask: np.ndarray, affine: np.ndarray):
    """Write compact gray-matter voxel labels and construct their parcel axis."""
    nib = _nib()
    labels, mask, affine = _volume_cifti_grid(labels, mask, affine)
    voxel_indices = np.argwhere(mask).astype(np.int64)
    active = labels >= 0
    if not np.any(active):
        raise ValueError("No gray-matter voxels have microparcel assignments")
    count = int(labels[active].max()) + 1
    if not np.array_equal(np.unique(labels[active]), np.arange(count)):
        raise ValueError("Microparcel assignments must be contiguous and zero-based")

    # Workbench supports voxel-based cortical/gray-matter CIFTI mappings, but
    # the generic OTHER structure is not reliably presented as cortical gray
    # matter in wb_view.  ALL_GREY_MATTER is the intended CIFTI structure for
    # a gray-matter mask that is not split into anatomical substructures.
    brain_axis = nib.cifti2.BrainModelAxis.from_mask(
        mask, name="all_grey_matter", affine=affine
    )
    names = _parcel_names(count)
    parcel_axis = volume_parcel_axis(labels, mask, affine)
    table = {0: ("unassigned", (0.0, 0.0, 0.0, 0.0))}
    table.update({index: (str(name), _parcel_color(index)) for index, name in enumerate(names, start=1)})
    label_axis = nib.cifti2.LabelAxis(["microparcels"], [table])
    data = np.where(labels >= 0, labels + 1, 0).astype(np.int32)[None, :]
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))
    nib.save(nib.Cifti2Image(data, header=header, dtype=np.int32), str(path))
    return Path(path), parcel_axis


def _quantize(correlations: np.ndarray) -> np.ndarray:
    correlations = np.asarray(correlations)
    if correlations.ndim != 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError("Microparcel connectivity must be square")
    quantized = np.empty(correlations.shape, dtype=np.int8)
    block_size = 512
    for start in range(0, correlations.shape[0], block_size):
        stop = min(start + block_size, correlations.shape[0])
        block = correlations[start:stop]
        if not np.all(np.isfinite(block)):
            raise ValueError("Microparcel connectivity contains non-finite values")
        if not np.allclose(block, correlations[:, start:stop].T, rtol=0.0, atol=1e-7):
            raise ValueError("Microparcel connectivity must be symmetric")
        quantized[start:stop] = np.rint(np.clip(block, -1.0, 1.0) * INT8_SCALE).astype(np.int8)
    return quantized


def write_pconn(path: Path, correlations: np.ndarray, parcel_axis) -> Path:
    nib = _nib()
    quantized = _quantize(correlations)
    header = nib.cifti2.Cifti2Header.from_axes((parcel_axis, parcel_axis))
    nib.save(nib.Cifti2Image(quantized, header=header, dtype=np.int8), str(path))

    # NiBabel does not preserve an explicit slope when its source array is
    # already integer, so patch the standard NIfTI-2 scaling fields in place.
    with Path(path).open("r+b") as file:
        nifti_header = nib.Nifti2Header.from_fileobj(file)
        nifti_header["scl_slope"] = 1.0 / INT8_SCALE
        nifti_header["scl_inter"] = 0.0
        file.seek(0)
        nifti_header.write_to(file)
    return Path(path)


def load_dlabel(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
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
        raise ValueError(f"Microparcel labels are not finite integers: {path}")
    counts = []
    arrays = []
    for _, structure_slice, structure_axis in brain_axis.iter_structures():
        values = encoded[structure_slice]
        arrays.append(np.where(values > 0, values - 1, -1).astype(np.int64))
        counts.append(len(structure_axis))
    return np.concatenate(arrays), tuple(counts)


def load_pconn(path: Path) -> np.ndarray:
    nib = _nib()
    image = nib.load(str(path))
    axes = (image.header.get_axis(0), image.header.get_axis(1))
    if not all(isinstance(axis, nib.cifti2.ParcelsAxis) for axis in axes):
        raise ValueError(f"Expected parcel-by-parcel CIFTI: {path}")
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Microparcel connectivity is not square: {path}")
    correlations = np.asarray(image.dataobj, dtype=np.float32)
    for start in range(0, correlations.shape[0], 512):
        stop = min(start + 512, correlations.shape[0])
        if not np.allclose(
            correlations[start:stop], correlations[:, start:stop].T, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"Microparcel connectivity is not symmetric: {path}")
    return correlations


def resolve_wb_command(configured: str | Path) -> str:
    executable = Path(configured).expanduser()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(
            f"Connectome Workbench executable is not available: {executable}"
        )
    return str(executable)
