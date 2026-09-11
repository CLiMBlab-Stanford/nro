"""Image adapters and compact fit storage; no estimator depends on image format."""

from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.cifti import write_indexed_cifti_sidecar
from nro.engine.images import load_gifti_timeseries
from nro.engine.io import atomic_output_path

from .statistics import RunFit


def read_timeseries(paths: tuple[Path, ...]) -> tuple[np.ndarray, dict]:
    """Load one run as time by location, retaining geometry for scalar maps."""
    if len(paths) == 1:
        image = nib.load(str(paths[0]))
        if not isinstance(image, nib.Nifti1Image):
            raise ValueError("Expected a NIfTI series or paired GIFTI surface series")
        if len(image.shape) != 4:
            raise ValueError("Functional volume must be four-dimensional")
        data = np.asarray(image.dataobj, dtype=np.float32).reshape((-1, image.shape[3])).T
        return data, {"domain": "volume", "reference": image, "shape": image.shape[:3]}
    if len(paths) != 2 or not all(path.name.endswith(".gii") for path in paths):
        raise ValueError("Expected a NIfTI series or paired GIFTI surface series")
    arrays = [load_gifti_timeseries(path) for path in paths]
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("Surface hemispheres have different frame counts")
    return np.concatenate(arrays, axis=1), {
        "domain": "surface",
        "counts": [a.shape[1] for a in arrays],
    }


def _brain_model_axis(geometry: dict):
    if geometry["domain"] == "volume":
        reference = geometry["reference"]
        return nib.cifti2.BrainModelAxis.from_mask(
            np.ones(geometry["shape"], dtype=bool),
            affine=reference.affine,
            name="CIFTI_STRUCTURE_OTHER",
        )
    axis = None
    for structure, count in zip(
        ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"),
        geometry["counts"],
    ):
        candidate = nib.cifti2.BrainModelAxis.from_surface(np.arange(count), count, name=structure)
        axis = candidate if axis is None else axis + candidate
    return axis


def write_statmaps(
    prefix: Path,
    records: list[tuple[dict, dict[str, np.ndarray]]],
    geometry: dict,
    metadata: dict,
) -> list[Path]:
    """Write one indexed CIFTI per available statistic and a sidecar for each."""
    outputs = []
    statistics = tuple(
        name
        for name in ("effect", "variance", "t", "dof")
        if any(name in maps for _, maps in records)
    )
    brain_axis = _brain_model_axis(geometry)
    for statistic in statistics:
        selected = [(record, maps[statistic]) for record, maps in records if statistic in maps]
        map_metadata = [
            {
                "Name": str(record["name"]),
                "Contrast": str(record["name"]),
                "Test": str(record["test"]),
                "Entities": dict(record["entities"]),
                "LinearRecipe": record["recipe"],
            }
            for record, _values in selected
        ]
        names = [record["Name"] for record in map_metadata]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate contrast names in one {statistic} CIFTI")
        values = np.vstack([np.asarray(array, dtype=np.float32) for _record, array in selected])
        scalar_axis = nib.cifti2.ScalarAxis(names)
        image = nib.Cifti2Image(
            values,
            header=nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis)),
            dtype=np.float32,
        )
        path = prefix.with_name(f"{prefix.name}_stat-{statistic}_statmap.dscalar.nii")
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(path) as staged:
            nib.save(image, staged)
        sidecar = write_indexed_cifti_sidecar(
            path,
            map_metadata,
            lookup_fields=("Contrast",),
            metadata={**metadata, "Statistic": statistic},
        )
        outputs.extend((path, sidecar))
    return outputs


def save_fit(prefix: Path, fit: RunFit) -> tuple[dict, list[Path]]:
    """Store mmap-readable coefficients and noise maps, plus small group matrices."""
    paths = {}
    for name in ("beta", "residual_variance", "groups", "covariance", "ar_coefficients"):
        path = prefix.with_name(f"{prefix.name}_desc-{name.replace('_', '')}_fit.npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(path) as staged:
            with staged.open("wb") as stream:
                np.save(stream, getattr(fit, name), allow_pickle=False)
        paths[name] = str(path)
    return {"arrays": paths, "dof": fit.dof}, [Path(path) for path in paths.values()]


def load_fit(record: dict, spatial_slice: slice = slice(None)) -> RunFit:
    """Read only a spatial block of a saved fit using memory-mapped arrays."""
    arrays = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in record["arrays"].items()
    }
    return RunFit(
        arrays["beta"][:, spatial_slice],
        arrays["residual_variance"][spatial_slice],
        arrays["groups"][spatial_slice],
        arrays["covariance"],
        record["dof"],
        arrays["ar_coefficients"],
    )
