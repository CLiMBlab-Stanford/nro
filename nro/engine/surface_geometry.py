"""Resolve display geometry for functional surface targets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from nro.engine.images import surface_timeseries_shape
from nro.engine.targets import is_fsaverage_space
from nro.engine.templates import find_fsaverage_surface


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load coordinates and triangles from a surface GIFTI file."""
    import nibabel as nib

    image = nib.load(str(path))
    coordinates = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    triangles = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if not coordinates or not triangles:
        raise ValueError(f"Surface {path} must contain POINTSET and TRIANGLE arrays")
    return (
        np.asarray(coordinates[0].data),
        np.asarray(triangles[0].data, dtype=np.int64),
    )


def load_surfaces(
    paths: tuple[Path, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Combine surface meshes while offsetting each mesh's vertex indices."""
    loaded = [load_surface(path) for path in paths]
    vertex_counts = tuple(len(coordinates) for coordinates, _ in loaded)
    offsets = np.cumsum((0,) + vertex_counts[:-1])
    coordinates = np.concatenate([item[0] for item in loaded], axis=0)
    triangles = np.concatenate([item[1] + offset for item, offset in zip(loaded, offsets)], axis=0)
    return coordinates, triangles, vertex_counts


def mesh_edges(triangles: np.ndarray) -> np.ndarray:
    """Return the unique undirected edges in a triangular mesh."""
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def load_surface_mask(paths: tuple[Path, ...] | None, vertex_counts: tuple[int, ...]) -> np.ndarray:
    """Load and concatenate masks or return an all-vertex mask."""
    if paths is None:
        return np.ones(sum(vertex_counts), dtype=bool)
    if len(paths) != len(vertex_counts):
        raise ValueError("Mask and surface file counts differ")

    import nibabel as nib

    masks = []
    for path, vertex_count in zip(paths, vertex_counts):
        image = nib.load(str(path))
        if not image.darrays:
            raise ValueError(f"Mask {path} has no data arrays")
        mask = np.asarray(image.darrays[0].data).reshape(-1).astype(bool)
        if mask.size != vertex_count:
            raise ValueError(f"Mask and surface vertex counts differ for {path}")
        masks.append(mask)
    return np.concatenate(masks)


def anatomical_surface_paths(
    anat_path: Path, participant: str, surface: str, space: str
) -> tuple[Path, Path]:
    """Read a native left/right surface pair from an anatomy manifest."""

    if space != "fsnative":
        raise FileNotFoundError(f"Anatomical manifests do not publish space-{space} geometry")
    manifest_path = Path(anat_path) / f"sub-{participant}_desc-preprocessAnat_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing anatomical publication manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = (manifest.get("outputs") or {}).get("surfaces") or {}
    result = []
    for hemisphere in ("lh", "rh"):
        value = published.get(f"{hemisphere}.{surface}")
        if not value:
            raise FileNotFoundError(
                f"Anatomical manifest lacks surfaces.{hemisphere}.{surface}: {manifest_path}"
            )
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing published anatomical surface: {path}")
        result.append(path)
    return tuple(result)  # type: ignore[return-value]


def surface_geometry(
    anat_path: Path,
    participant: str,
    surface: str,
    space: str,
    functional_pair: tuple[Path, ...],
) -> tuple[Path, Path]:
    """Resolve native or fsaverage geometry for one functional surface pair."""

    if space == "fsnative":
        return anatomical_surface_paths(anat_path, participant, surface, space)
    if not is_fsaverage_space(space):
        raise FileNotFoundError(f"Unsupported configured surface space: {space}")
    _, vertex_counts = surface_timeseries_shape(functional_pair)
    hemispheres = tuple(
        match.group(1) if (match := re.search(r"(?:^|_)hemi-([^_]+)", path.name)) else None
        for path in functional_pair
    )
    if hemispheres != ("L", "R"):
        raise ValueError(
            f"Expected deterministic L/R fsaverage functional inputs, got {hemispheres!r}"
        )
    return tuple(
        find_fsaverage_surface(hemi=hemi, surface=surface, n_vertices=count)
        for hemi, count in zip(hemispheres, vertex_counts)
    )  # type: ignore[return-value]
