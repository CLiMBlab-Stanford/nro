"""General image-path, validation, and serialization primitives."""

from __future__ import annotations

import gzip
import shutil
import struct
from pathlib import Path

import numpy as np

from .bids import resolve_bids_metadata
from .io import atomic_output_path


def nifti_stem(path: Path) -> str:
    """Return a NIfTI filename without its simple or compressed suffix."""
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def sidecar_json_path(path: Path) -> Path:
    """Return the conventional JSON sidecar path for an imaging file."""
    path = Path(path)
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name[: -len(".nii.gz")] + ".json")
    for suffix in (".func.gii", ".shape.gii", ".label.gii", ".surf.gii"):
        if path.name.endswith(suffix):
            kind = suffix.removesuffix(".gii")
            return path.with_name(path.name[: -len(suffix)] + kind + ".json")
    return path.with_suffix(".json")


def image_source_paths(path: Path) -> tuple[Path, ...]:
    """Return an image and every applicable BIDS metadata source."""
    path = Path(path)
    try:
        metadata_sources = resolve_bids_metadata(path).sources
    except FileNotFoundError:
        metadata_sources = ()
    return (path, *metadata_sources)


def is_gzip_nifti(path: Path) -> bool:
    """Return whether a path has the compressed NIfTI suffix."""
    return Path(path).name.endswith(".nii.gz")


def uncompressed_nifti_path(path: Path) -> Path:
    """Return the equivalent uncompressed NIfTI path."""
    path = Path(path)
    if is_gzip_nifti(path):
        return path.with_name(path.name[: -len(".nii.gz")] + ".nii")
    return path


def nifti_spatial_shape(path: Path) -> tuple[int, int, int]:
    """Load a NIfTI and return its first three dimensions."""
    try:
        import nibabel as nib
    except Exception as error:  # pragma: no cover - installation failure
        raise SystemExit(f"nibabel is required to inspect {path}: {error}") from error
    shape = nib.load(str(path)).shape
    if len(shape) < 3:
        raise SystemExit(f"Expected a 3D or 4D NIfTI image: {path}")
    return tuple(int(value) for value in shape[:3])


def nifti_volume_count(path: Path) -> int:
    """Read a NIfTI header and return its time-axis length."""
    try:
        import nibabel as nib

        shape = nib.load(str(path)).shape
    except Exception as error:
        raise SystemExit(f"Could not read the NIfTI header for {path}: {error}") from error
    return max(1, int(shape[3])) if len(shape) >= 4 else 1


def read_nifti_header_shape(path: Path) -> tuple[int, int, int]:
    """Read a NIfTI-1 spatial shape without loading its image data."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(348)
    if len(header) < 348:
        raise ValueError(f"Incomplete NIfTI header: {path}")
    little = struct.unpack("<I", header[0:4])[0]
    big = struct.unpack(">I", header[0:4])[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"Unrecognized NIfTI header size: {path}")
    dimensions = struct.unpack(endian + "8h", header[40:56])
    if int(dimensions[0]) < 3:
        raise ValueError(f"Bad NIfTI ndim={int(dimensions[0])}: {path}")
    shape = tuple(int(value) for value in dimensions[1:4])
    if any(value <= 0 for value in shape):
        raise ValueError(f"Bad NIfTI shape={shape}: {path}")
    return shape  # type: ignore[return-value]


def nifti_is_valid(path: Path) -> bool:
    """Return whether a NIfTI path exists and has a readable basic header."""
    path = Path(path)
    if not path.is_file():
        return False
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        return True
    try:
        read_nifti_header_shape(path)
    except Exception:
        return False
    return True


def nifti_zooms_xyz(path: Path) -> tuple[float, float, float]:
    """Return a NIfTI image's three spatial voxel sizes."""
    import nibabel as nib

    zooms = nib.load(str(path)).header.get_zooms()
    if len(zooms) < 3:
        raise ValueError(f"Missing spatial voxel sizes: {path}")
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def copy_or_convert_nifti(source: Path, destination: Path) -> None:
    """Copy a NIfTI, changing gzip representation when suffixes require it."""
    source = Path(source)
    destination = Path(destination)
    if source == destination:
        return
    with atomic_output_path(destination) as staged:
        source_gzip = is_gzip_nifti(source)
        destination_gzip = is_gzip_nifti(staged)
        if source_gzip and not destination_gzip:
            with gzip.open(source, "rb") as source_stream, staged.open("wb") as output_stream:
                shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
        elif not source_gzip and destination_gzip:
            with source.open("rb") as source_stream, gzip.open(staged, "wb") as output_stream:
                shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
        else:
            shutil.copyfile(source, staged)
        if not nifti_is_valid(staged):
            raise RuntimeError(f"Refusing to publish an unreadable NIfTI: {staged}")


def load_gifti_timeseries(path: Path) -> np.ndarray:
    """Load GIFTI data arrays as a time-by-vertex float32 matrix."""
    import nibabel as nib

    image = nib.load(str(path))
    if not getattr(image, "darrays", None):
        raise SystemExit(f"No data arrays found in GIFTI: {path}")
    data = np.vstack([np.asarray(array.data, dtype=np.float32) for array in image.darrays])
    if data.ndim != 2:
        raise SystemExit(f"Expected 2D time x vertex GIFTI data: {path}")
    return data


def load_surface_timeseries(paths: tuple[Path, ...]) -> np.ndarray:
    """Load one unilateral or bilateral run as time by vertex data."""

    import nibabel as nib

    matrices = []
    for path in paths:
        image = nib.load(str(path))
        if not image.darrays:
            raise ValueError(f"Functional GIFTI run contains no data: {path}")
        if len(image.darrays) == 1:
            matrix = np.asarray(image.darrays[0].data, dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            elif matrix.ndim != 2:
                raise ValueError(f"Functional GIFTI data must be one- or two-dimensional: {path}")
            elif matrix.shape[0] >= matrix.shape[1]:
                matrix = matrix.T
        else:
            matrix = np.stack(
                [np.asarray(array.data, dtype=np.float32).reshape(-1) for array in image.darrays]
            )
        matrices.append(matrix)
    if len({matrix.shape[0] for matrix in matrices}) != 1:
        raise ValueError(f"Surface functional files have different frame counts: {paths}")
    return np.concatenate(matrices, axis=1)


def surface_timeseries_shape(paths: tuple[Path, ...]) -> tuple[int, tuple[int, ...]]:
    """Return frame count and ordered vertex counts for surface time series."""

    import nibabel as nib

    shapes = []
    for path in paths:
        image = nib.load(str(path))
        if not image.darrays:
            raise ValueError(f"Functional GIFTI run contains no data: {path}")
        if len(image.darrays) > 1:
            shapes.append((len(image.darrays), int(np.asarray(image.darrays[0].data).size)))
            continue
        shape = np.asarray(image.darrays[0].data).shape
        if len(shape) == 1:
            shapes.append((1, int(shape[0])))
        elif len(shape) == 2:
            shapes.append(
                (int(shape[1]), int(shape[0]))
                if shape[0] >= shape[1]
                else (int(shape[0]), int(shape[1]))
            )
        else:
            raise ValueError(f"Functional GIFTI data must be one- or two-dimensional: {path}")
    if len({shape[0] for shape in shapes}) != 1:
        raise ValueError(f"Surface functional files have different frame counts: {paths}")
    return shapes[0][0], tuple(shape[1] for shape in shapes)


def gifti_vertex_count(path: Path) -> int:
    """Return the length of the first data array in a GIFTI file."""
    import nibabel as nib

    image = nib.load(str(path))
    if not getattr(image, "darrays", None):
        raise SystemExit(f"No data arrays found in GIFTI: {path}")
    return int(np.asarray(image.darrays[0].data).shape[0])


def save_gifti_timeseries(
    template_path: Path,
    data_time_by_vertex: np.ndarray,
    output_path: Path,
) -> None:
    """Save time-by-vertex data using a template GIFTI's metadata."""
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    template = nib.load(str(template_path))
    output = GiftiImage(
        meta=getattr(template, "meta", None),
        labeltable=getattr(template, "labeltable", None),
    )
    for row in data_time_by_vertex:
        output.add_gifti_data_array(GiftiDataArray(np.asarray(row, dtype=np.float32)))
    with atomic_output_path(output_path) as staged:
        nib.save(output, str(staged))


def write_cifti_dense_scalar(
    path: Path,
    reference_dlabel: Path,
    arrays: list[np.ndarray],
    names: list[str],
) -> Path:
    """Write named dense scalars on a reference CIFTI brain model."""
    import nibabel as nib

    reference = nib.load(str(reference_dlabel))
    brain_axis = reference.header.get_axis(1)
    if not isinstance(brain_axis, nib.cifti2.BrainModelAxis):
        raise ValueError(f"Expected a brain-model axis in {reference_dlabel}")
    data = np.vstack([np.asarray(array, dtype=np.float32) for array in arrays])
    if data.shape[1] != len(brain_axis):
        raise ValueError("Dense scalar size does not match the reference brain model")
    scalar_axis = nib.cifti2.ScalarAxis(names)
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    with atomic_output_path(path) as staged:
        nib.save(nib.Cifti2Image(data, header=header, dtype=np.float32), str(staged))
    return path
