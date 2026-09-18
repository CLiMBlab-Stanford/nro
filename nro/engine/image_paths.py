"""Image filename and metadata-source helpers that do not load image libraries."""

from __future__ import annotations

from pathlib import Path

from nro.engine.bids import resolve_bids_metadata


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


def image_source_paths(path: Path, *, markup=None) -> tuple[Path, ...]:
    """Return an image and every applicable BIDS metadata source."""
    path = Path(path)
    try:
        metadata_sources = resolve_bids_metadata(path, markup=markup).sources
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
