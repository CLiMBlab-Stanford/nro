"""Volumetric gray-matter graph and functional data handling."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


@dataclass(frozen=True)
class VolumeSpace:
    """Masked volume grid, affine, spatial neighborhoods, and CIFTI structure information."""

    shape: tuple[int, int, int]
    affine: np.ndarray
    mask: np.ndarray
    voxel_indices: np.ndarray
    edges: np.ndarray
    mask_resampled: bool


def _same_grid(image: nib.spatialimages.SpatialImage, shape, affine) -> bool:
    return tuple(image.shape[:3]) == tuple(shape) and np.allclose(
        image.affine, affine, rtol=0.0, atol=1e-5
    )


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int, int], ...]:
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                distance = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and distance != 1:
                    continue
                if connectivity == 18 and distance > 2:
                    continue
                # Retain one direction from each undirected offset pair.
                first_nonzero = next(value for value in (dx, dy, dz) if value)
                if first_nonzero < 0:
                    continue
                offsets.append((dx, dy, dz))
    return tuple(offsets)


def _offset_slices(shape, offset):
    source = []
    neighbor = []
    for size, delta in zip(shape, offset):
        if delta >= 0:
            source.append(slice(0, size - delta))
            neighbor.append(slice(delta, size))
        else:
            source.append(slice(-delta, size))
            neighbor.append(slice(0, size + delta))
    return tuple(source), tuple(neighbor)


def volume_edges(mask: np.ndarray, connectivity: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Return compact masked voxel indices and undirected spatial edges."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Volumetric gray-matter mask must be three-dimensional")
    voxel_indices = np.argwhere(mask).astype(np.int64)
    if len(voxel_indices) < 2:
        raise ValueError("Volumetric gray-matter mask contains fewer than two voxels")
    node_ids = np.full(mask.shape, -1, dtype=np.int64)
    node_ids[tuple(voxel_indices.T)] = np.arange(len(voxel_indices), dtype=np.int64)
    edge_blocks = []
    for offset in _neighbor_offsets(connectivity):
        source_slice, neighbor_slice = _offset_slices(mask.shape, offset)
        source = node_ids[source_slice]
        neighbor = node_ids[neighbor_slice]
        valid = (source >= 0) & (neighbor >= 0)
        if np.any(valid):
            edge_blocks.append(np.column_stack((source[valid], neighbor[valid])))
    if not edge_blocks:
        raise ValueError("Gray-matter mask has no spatially adjacent voxels")
    return voxel_indices, np.concatenate(edge_blocks).astype(np.int64, copy=False)


def load_volume_space(
    reference_path: Path,
    mask_path: Path,
    *,
    threshold: float,
    connectivity: int,
) -> VolumeSpace:
    """Load a functional volume grid and its thresholded gray-matter mask."""
    reference = nib.load(str(reference_path))
    if len(reference.shape) != 4 or reference.shape[3] < 1:
        raise ValueError(f"Volumetric functional run must be nonempty 4D NIfTI: {reference_path}")
    shape = tuple(int(value) for value in reference.shape[:3])
    mask_image = nib.load(str(mask_path))
    resampled = not _same_grid(mask_image, shape, reference.affine)
    if resampled:
        interpolation_order = 1 if "probseg" in mask_path.name else 0
        mask_image = resample_from_to(
            mask_image,
            (shape, reference.affine),
            order=interpolation_order,
        )
    mask_data = np.asarray(mask_image.dataobj, dtype=np.float32)
    if mask_data.ndim == 4 and mask_data.shape[3] == 1:
        mask_data = mask_data[..., 0]
    if mask_data.shape != shape:
        raise ValueError(f"Gray-matter mask is not three-dimensional: {mask_path}")
    mask = np.isfinite(mask_data) & (mask_data > np.float32(threshold))
    voxel_indices, edges = volume_edges(mask, connectivity)
    return VolumeSpace(
        shape=shape,
        affine=np.asarray(reference.affine),
        mask=mask,
        voxel_indices=voxel_indices,
        edges=edges,
        mask_resampled=resampled,
    )


def load_volume_functional(paths: tuple[Path, ...], space: VolumeSpace) -> np.ndarray:
    """Load one complete 4D run, then retain active gray-matter voxels.

    Run-level streaming bounds memory use to one functional run.  Materializing
    the compressed NIfTI once is substantially faster than proxy-slicing each
    volume, which can repeatedly decompress a ``.nii.gz`` stream when
    ``indexed_gzip`` is unavailable.
    """
    if len(paths) != 1:
        raise ValueError("A volumetric functional run must contain exactly one NIfTI file")
    path = paths[0]
    image = nib.load(str(path))
    if len(image.shape) != 4 or image.shape[3] < 1:
        raise ValueError(f"Volumetric functional run must be nonempty 4D NIfTI: {path}")
    if not _same_grid(image, space.shape, space.affine):
        raise ValueError(
            f"Volumetric functional grid differs from the first run: {path}. "
            "Resample cleaned runs to a common space before microparcellation."
        )
    full_data = np.asarray(image.dataobj, dtype=np.float32)
    selection = tuple(space.voxel_indices.T)
    # Advanced indexing creates the compact gray-matter copy.  Transpose to
    # the time-by-node convention used by the surface pathway, then release
    # the full spatial grid before returning to the streaming statistics.
    data = np.asarray(full_data[selection + (slice(None),)].T, dtype=np.float32)
    del full_data
    return data


def write_volume_labels(path: Path, labels: np.ndarray, space: VolumeSpace) -> Path:
    """Write zero-based parcel assignments as a labeled NIfTI volume."""
    labels = np.asarray(labels, dtype=np.int32)
    encoded = np.zeros(space.shape, dtype=np.int32)
    encoded[tuple(space.voxel_indices.T)] = labels + 1
    image = nib.Nifti1Image(encoded, space.affine)
    image.header.set_data_dtype(np.int32)
    image.header.set_xyzt_units("mm")
    count = int(labels.max()) + 1
    width = max(5, len(str(count)))
    label_elements = ['<Label Key="0" Red="0" Green="0" Blue="0" Alpha="0"><![CDATA[???]]></Label>']
    for key in range(1, count + 1):
        hue = (key * 0.6180339887498949) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        name = escape(f"microparcel_{key:0{width}d}")
        label_elements.append(
            f'<Label Key="{key}" Red="{red:.8g}" Green="{green:.8g}" '
            f'Blue="{blue:.8g}" Alpha="1"><![CDATA[{name}]]></Label>'
        )
    caret_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<CaretExtension><VolumeInformation Index="0"><LabelTable>'
        + "".join(label_elements)
        + "</LabelTable><StudyMetaDataLinkSet></StudyMetaDataLinkSet>"
        "<VolumeType><![CDATA[Label]]></VolumeType>"
        "</VolumeInformation></CaretExtension>\n"
    )
    image.header.extensions.append(nib.nifti1.Nifti1Extension(30, caret_xml.encode("utf-8")))
    nib.save(image, str(path))
    return path


def load_volume_labels(path: Path) -> np.ndarray:
    """Load compact zero-based parcels from a Workbench label NIfTI."""
    image = nib.load(str(path))
    encoded = np.asarray(image.dataobj)
    if encoded.ndim == 4 and encoded.shape[3] == 1:
        encoded = encoded[..., 0]
    if encoded.ndim != 3:
        raise ValueError(f"Volumetric microparcel labels must be three-dimensional: {path}")
    active = encoded > 0
    labels = np.rint(encoded[active]).astype(np.int64, copy=False) - 1
    if not len(labels) or not np.all(encoded[active] == np.rint(encoded[active])):
        raise ValueError(f"Volumetric microparcel labels are missing or non-integral: {path}")
    return labels
