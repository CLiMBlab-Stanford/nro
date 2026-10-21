"""Load meshes, cortical maps, and seed correlations for the browser viewer.

Every array returned here is per hemisphere and indexed by surface vertex.
Vertices that a CIFTI file does not address, such as the medial wall, are NaN.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

from nro.engine.surface_geometry import load_surface
from nro.viz.catalog import HEMISPHERES, Layer

CORTEX = {"L": "CIFTI_STRUCTURE_CORTEX_LEFT", "R": "CIFTI_STRUCTURE_CORTEX_RIGHT"}


@lru_cache(maxsize=32)
def mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 vertex coordinates and uint32 triangles for one surface."""
    coordinates, triangles = load_surface(path)
    return (
        np.ascontiguousarray(coordinates, dtype="<f4"),
        np.ascontiguousarray(triangles, dtype="<u4"),
    )


def _cortical_models(path: Path) -> dict[str, tuple[slice, np.ndarray, int]]:
    """Map each hemisphere to its CIFTI column slice, vertex indices, and mesh size."""
    import nibabel as nib

    axis = nib.load(str(path)).header.get_axis(1)
    models = {}
    for structure, columns, model in axis.iter_structures():
        for hemisphere, name in CORTEX.items():
            if str(structure) == name:
                models[hemisphere] = (columns, np.asarray(model.vertex), int(model.nvertices[name]))
    return models


def _scatter(
    row: np.ndarray, models: dict[str, tuple[slice, np.ndarray, int]]
) -> dict[str, np.ndarray]:
    hemispheres = {}
    for hemisphere, (columns, vertices, size) in models.items():
        values = np.full(size, np.nan, dtype="<f4")
        values[vertices] = row[columns]
        hemispheres[hemisphere] = values
    return hemispheres


@lru_cache(maxsize=64)
def _layer_map(paths: tuple[tuple[str, Path], ...], index: int) -> dict[str, np.ndarray]:
    import nibabel as nib

    files = dict(paths)
    if "LR" in files:
        image = nib.load(str(files["LR"]))
        row = np.asarray(image.dataobj[index], dtype="<f4")
        return _scatter(row, _cortical_models(files["LR"]))
    return {
        hemisphere: np.asarray(nib.load(str(path)).darrays[index].data, dtype="<f4")
        for hemisphere, path in files.items()
    }


def layer_map(layer: Layer, index: int) -> dict[str, np.ndarray]:
    """Return one map of a metric or label layer for each available hemisphere.

    Raises IndexError when the map index is outside the layer.
    """
    if layer.kind == "connectivity":
        raise ValueError("Connectivity layers are computed from a seed")
    if not 0 <= index < len(layer.maps):
        raise IndexError(f"Layer {layer.label} has no map {index}")
    return _layer_map(tuple(layer.paths.items()), index)


def label_table(layer: Layer, index: int) -> dict[str, dict[str, object]]:
    """Return label names and RGBA colors keyed by label value."""
    import nibabel as nib

    if "LR" in layer.paths:
        axis = nib.load(str(layer.paths["LR"])).header.get_axis(0)
        entries = axis.label[index]
        return {
            str(key): {"name": str(name), "rgba": [float(value) for value in rgba]}
            for key, (name, rgba) in entries.items()
        }
    table = {}
    for path in layer.paths.values():
        for entry in nib.load(str(path)).labeltable.labels:
            table[str(entry.key)] = {
                "name": str(entry.label),
                "rgba": [float(value) for value in entry.rgba],
            }
    return table


class SeedCorrelation:
    """Correlate one cortical vertex with every column of a dense time series.

    The first request for a series reads it completely and normalizes each
    column to zero mean and unit norm. Normalized series stay in memory for
    later requests, least recently used first out, while their total size stays
    within ``budget_bytes``; the newest series is always kept. Series load
    independently, so one slow read does not block seeds on another series.
    Columns with no temporal variance correlate as NaN.
    """

    def __init__(self, budget_bytes: int = 8 * 1024**3) -> None:
        """Start with no series in memory."""
        self.budget_bytes = budget_bytes
        self._lock = threading.Lock()
        self._loading: dict[Path, threading.Lock] = {}
        self._series: OrderedDict[Path, tuple[np.ndarray, dict]] = OrderedDict()

    def resident(self) -> tuple[Path, ...]:
        """Return the series currently in memory, least recently used first."""
        with self._lock:
            return tuple(self._series)

    def prepare(self, path: Path) -> tuple[np.ndarray, dict]:
        """Return a normalized series and its cortical models, loading it if needed."""
        import nibabel as nib

        with self._lock:
            if path in self._series:
                self._series.move_to_end(path)
                return self._series[path]
            loading = self._loading.setdefault(path, threading.Lock())
        with loading:
            with self._lock:
                if path in self._series:
                    self._series.move_to_end(path)
                    return self._series[path]
            series = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
            series -= series.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(series, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                series /= np.where(norms > 0, norms, np.nan)
            entry = (series, _cortical_models(path))
            with self._lock:
                self._series[path] = entry
                self._loading.pop(path, None)
                while len(self._series) > 1 and (
                    sum(item[0].nbytes for item in self._series.values()) > self.budget_bytes
                ):
                    self._series.popitem(last=False)
            return entry

    def correlate(self, layer: Layer, hemisphere: str, vertex: int) -> dict[str, np.ndarray]:
        """Return Pearson correlations with the seed vertex for each hemisphere.

        Raises ValueError when the series does not address the seed vertex.
        """
        series, models = self.prepare(layer.paths["LR"])
        if hemisphere not in models:
            raise ValueError(f"The series has no {hemisphere} cortical vertices")
        columns, vertices, _size = models[hemisphere]
        matches = np.flatnonzero(vertices == vertex)
        if not len(matches):
            raise ValueError(
                f"Vertex {vertex} has no data in this series (for example, medial wall)"
            )
        column = columns.start + int(matches[0])
        seed = series[:, column]
        if not np.isfinite(seed).all():
            raise ValueError(f"Vertex {vertex} has no temporal variance")
        correlations = seed @ series
        return _scatter(np.asarray(correlations, dtype="<f4"), models)


def concatenated(
    values: dict[str, np.ndarray], counts: dict[str, int]
) -> tuple[bytes, dict[str, int]]:
    """Pack hemisphere arrays as little-endian float32, left before right.

    A hemisphere absent from ``values`` is filled with NaN so the client can
    split the buffer by the counts it receives.
    """
    parts = []
    sizes = {}
    for hemisphere in HEMISPHERES:
        if hemisphere in values:
            array = np.ascontiguousarray(values[hemisphere], dtype="<f4")
        elif hemisphere in counts:
            array = np.full(counts[hemisphere], np.nan, dtype="<f4")
        else:
            continue
        sizes[hemisphere] = int(len(array))
        parts.append(array.tobytes())
    return b"".join(parts), sizes
