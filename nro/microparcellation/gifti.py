from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def _nib():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for GIFTI input/output") from exc
    return nib


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    nib = _nib()
    img = nib.load(str(path))
    coords = img.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    faces = img.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if not coords or not faces:
        raise ValueError(f"Surface {path} must contain POINTSET and TRIANGLE arrays")
    return np.asarray(coords[0].data), np.asarray(faces[0].data, dtype=np.int64)


def load_surfaces(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    loaded = [load_surface(path) for path in paths]
    counts = tuple(len(coords) for coords, _ in loaded)
    offsets = np.cumsum((0,) + counts[:-1])
    coords = np.concatenate([item[0] for item in loaded], axis=0)
    faces = np.concatenate([item[1] + offset for item, offset in zip(loaded, offsets)], axis=0)
    return coords, faces, counts


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    e.sort(axis=1)
    return np.unique(e, axis=0)


def load_mask(paths: tuple[Path, ...] | None, vertex_counts: tuple[int, ...]) -> np.ndarray:
    if paths is None:
        return np.ones(sum(vertex_counts), dtype=bool)
    masks = []
    for path, n_vertices in zip(paths, vertex_counts):
        img = _nib().load(str(path))
        if not img.darrays:
            raise ValueError(f"Mask {path} has no data arrays")
        mask = np.asarray(img.darrays[0].data).reshape(-1).astype(bool)
        if mask.size != n_vertices:
            raise ValueError(f"Mask and surface vertex counts differ for {path}")
        masks.append(mask)
    return np.concatenate(masks)


def iter_func_blocks(paths: tuple[Path, ...], block_size: int) -> Iterator[np.ndarray]:
    """Yield time-by-vertex float32 blocks while retaining only one run in nibabel."""
    matrix = load_functional(paths)
    for start in range(0, matrix.shape[0], block_size):
        yield matrix[start:start + block_size]


def load_functional(paths: tuple[Path, ...]) -> np.ndarray:
    """Load one unilateral or bilateral run as a time-by-vertex float32 matrix."""
    images = [_nib().load(str(path)) for path in paths]
    matrices = [_functional_matrix(img, path) for img, path in zip(images, paths)]
    lengths = {matrix.shape[0] for matrix in matrices}
    if len(lengths) != 1:
        raise ValueError(f"Left/right functional files have different timepoint counts: {paths}")
    return np.concatenate(matrices, axis=1)


def _functional_matrix(img, path: Path) -> np.ndarray:
    if not img.darrays:
        raise ValueError(f"Functional GIFTI run {path} contains an empty file")
    if len(img.darrays) == 1:
        data = np.asarray(img.darrays[0].data, dtype=np.float32)
        if data.ndim == 1:
            return data.reshape(1, -1)
        if data.ndim != 2:
            raise ValueError(f"Functional GIFTI data must be 1D or 2D in {path}")
        if data.shape[0] >= data.shape[1]:
            return data.T
        return data
    return np.stack([np.asarray(x.data, dtype=np.float32).reshape(-1) for x in img.darrays])


def func_shape(paths: tuple[Path, ...]) -> tuple[int, tuple[int, ...]]:
    shapes = []
    for path in paths:
        img = _nib().load(str(path))
        matrix = _functional_matrix(img, path)
        shapes.append(matrix.shape)
    if len({shape[0] for shape in shapes}) != 1:
        raise ValueError(f"Left/right functional files have different timepoint counts: {paths}")
    return shapes[0][0], tuple(shape[1] for shape in shapes)


def write_metric(path: Path, arrays: list[np.ndarray], names: list[str]) -> None:
    nib = _nib()
    if len(arrays) != len(names):
        raise ValueError("arrays and names differ in length")
    out = nib.gifti.GiftiImage()
    for data, name in zip(arrays, names):
        arr = nib.gifti.GiftiDataArray(np.asarray(data, dtype=np.float32), intent="NIFTI_INTENT_SHAPE")
        arr.meta["Name"] = name
        out.add_gifti_data_array(arr)
    nib.save(out, str(path))


def load_label(path: Path) -> np.ndarray:
    """Load a Workbench label GIFTI as zero-based assignments."""
    image = _nib().load(str(path))
    if len(image.darrays) != 1:
        raise ValueError(f"Label GIFTI must contain exactly one data array: {path}")
    encoded = np.asarray(image.darrays[0].data).reshape(-1)
    if not np.all(np.isfinite(encoded)) or not np.all(encoded == np.rint(encoded)):
        raise ValueError(f"Label GIFTI contains non-integral values: {path}")
    return np.where(encoded > 0, encoded - 1, -1).astype(np.int64)


def write_surface_metric(
    output_dir: Path,
    prefix: str,
    stem: str,
    arrays: list[np.ndarray],
    names: list[str],
    vertex_counts: tuple[int, ...],
) -> Path | tuple[Path, Path]:
    if len(vertex_counts) == 1:
        path = output_dir / f"{prefix}_{stem}.shape.gii"
        write_metric(path, arrays, names)
        return path
    split = vertex_counts[0]
    paths = (
        output_dir / f"{prefix}_L_{stem}.shape.gii",
        output_dir / f"{prefix}_R_{stem}.shape.gii",
    )
    write_metric(paths[0], [array[:split] for array in arrays], names)
    write_metric(paths[1], [array[split:] for array in arrays], names)
    return paths
