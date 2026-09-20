"""Load and reduce common spatial feature matrices for network estimators."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy import sparse

from nro.engine.io import atomic_output_path

from .config import FeatureReductionConfig, ModuleConfig

_ROW_BLOCK_SIZE = 4096


def _active_rows(matrix: np.ndarray, *, path: Path) -> np.ndarray:
    """Validate a mapped feature matrix and identify varying, nonzero rows."""

    active = np.zeros(matrix.shape[0], dtype=bool)
    for start in range(0, matrix.shape[0], _ROW_BLOCK_SIZE):
        stop = min(start + _ROW_BLOCK_SIZE, matrix.shape[0])
        block = np.asarray(matrix[start:stop], dtype=np.float32)
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Dynamic-connectivity features contain non-finite values: {path}")
        active[start:stop] = np.any(block != 0, axis=1) & (np.var(block, axis=1) > 0)
    return active


def dynconn_features(path: Path, domain: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return spatial rows, active locations, and structure sizes from dynconn."""

    import nibabel as nib

    image = nib.load(str(path))
    if domain == "surface":
        brain_axis = image.header.get_axis(1)
        if not isinstance(brain_axis, nib.cifti2.BrainModelAxis):
            raise ValueError(f"Expected a CIFTI brain-model axis: {path}")
        matrix = np.asanyarray(image.dataobj).T
        counts = tuple(len(axis) for _, _, axis in brain_axis.iter_structures())
    else:
        if len(image.shape) != 4:
            raise ValueError(f"Expected a four-dimensional dynconn volume: {path}")
        data = np.asanyarray(image.dataobj)
        matrix = data.reshape(-1, data.shape[-1])
        counts = ()
    if matrix.ndim != 2:
        raise ValueError(f"Dynamic-connectivity features are malformed: {path}")
    active = _active_rows(matrix, path=path)
    if not np.any(active):
        raise ValueError(f"Dynamic-connectivity input has no varying spatial locations: {path}")
    return matrix, active, counts


def reduce_features(
    matrix: sparse.spmatrix | np.ndarray,
    active: np.ndarray,
    config: FeatureReductionConfig,
    output: Path,
    *,
    informative_dimensions: int | None = None,
) -> dict[str, object]:
    """Write active spatial features under a configured dimensionality ceiling."""

    from sklearn.utils.extmath import randomized_svd

    temporary: Path | None = None
    try:
        if np.all(active):
            selected = matrix
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                prefix=f".{output.stem}.active-",
                suffix=".npy",
                dir=output.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            selected = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(int(active.sum()), int(matrix.shape[1])),
            )
            destination = 0
            indices = np.flatnonzero(active)
            for start in range(0, len(indices), _ROW_BLOCK_SIZE):
                rows = indices[start : start + _ROW_BLOCK_SIZE]
                selected[destination : destination + len(rows)] = matrix[rows]
                destination += len(rows)
        n_locations, n_columns = selected.shape
        informative = min(n_columns, informative_dimensions or n_columns)
        target = min(config.maximum_dimensions, informative, n_locations)
        if target < 2:
            raise ValueError("Network features have fewer than two informative dimensions")
        total_energy = (
            float(selected.multiply(selected).sum())
            if sparse.issparse(selected)
            else float(np.einsum("ij,ij->", selected, selected, dtype=np.float64))
        )
        if n_columns <= min(config.maximum_dimensions, informative):
            reduced = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
            method = "none"
            retained = 1.0
            effective_seed = None
        else:
            effective_seed = (
                int(np.random.SeedSequence().generate_state(1)[0])
                if config.random_seed is None
                else config.random_seed
            )
            left, singular, _right = randomized_svd(
                selected,
                n_components=target,
                n_oversamples=config.oversampling,
                n_iter=config.power_iterations,
                random_state=effective_seed,
                flip_sign=False,
            )
            reduced = left * singular[np.newaxis, :]
            method = "randomized_svd"
            retained = (
                float(np.square(singular, dtype=np.float64).sum() / total_energy)
                if total_energy > 0
                else 0.0
            )
        reduced = np.asarray(reduced, dtype=np.float32)
        expected_columns = target if method != "none" else n_columns
        if reduced.shape != (n_locations, expected_columns):
            raise RuntimeError("Reduced network feature dimensions are inconsistent")
        if not np.all(np.isfinite(reduced)):
            raise ValueError("Feature reduction produced non-finite values")
        output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(output) as staged, staged.open("wb") as stream:
            np.save(stream, reduced, allow_pickle=False)
        return {
            "method": method,
            "source_dimensions": int(n_columns),
            "informative_dimensions": int(informative),
            "fitted_dimensions": int(reduced.shape[1]),
            "spatial_locations": int(len(active)),
            "active_locations": int(active.sum()),
            "maximum_dimensions": int(config.maximum_dimensions),
            "oversampling": int(config.oversampling),
            "power_iterations": int(config.power_iterations),
            "random_seed": effective_seed,
            "retained_variance_fraction": retained,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_reduced_features(path: Path) -> np.ndarray:
    """Load and validate a saved spatial feature matrix."""

    matrix = np.load(path, allow_pickle=False, mmap_mode="r")
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Reduced network features are malformed: {path}")
    return matrix


def informative_dimensions(cfg: ModuleConfig) -> int | None:
    """Return a declared source rank when dynconn publishes one."""

    if cfg.inputs.source != "dynconn":
        return None
    import yaml

    manifest = yaml.safe_load(cfg.inputs.source_manifest.read_text(encoding="utf-8")) or {}
    low_rank = manifest.get("low_rank")
    return int(low_rank["dimensions"]) if isinstance(low_rank, dict) else None
