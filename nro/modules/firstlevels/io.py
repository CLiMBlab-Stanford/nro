"""Image adapters and compact fit storage; no estimator depends on image format."""

from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.images import load_gifti_timeseries
from nro.engine.io import atomic_output_path, atomic_write_json

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


def write_statmaps(
    prefix: Path, maps: dict[str, np.ndarray], geometry: dict, metadata: dict
) -> list[Path]:
    """Write effect/variance/t/DOF maps and matching sidecars in source geometry."""
    outputs = []
    for statistic, values in maps.items():
        if geometry["domain"] == "volume":
            reference = geometry["reference"]
            header = reference.header.copy()
            header.set_data_dtype(np.float32)
            header.set_intent("none")
            image = nib.Nifti1Image(
                np.asarray(values, np.float32).reshape(geometry["shape"]), reference.affine, header
            )
            targets = [(prefix.with_name(f"{prefix.name}_stat-{statistic}_statmap.nii.gz"), image)]
        else:
            targets = []
            start = 0
            for hemisphere, count in zip(("L", "R"), geometry["counts"]):
                array = nib.gifti.GiftiDataArray(
                    np.asarray(values[start : start + count], np.float32)
                )
                targets.append(
                    (
                        prefix.with_name(
                            f"{prefix.name}_hemi-{hemisphere}_stat-{statistic}_statmap.shape.gii"
                        ),
                        nib.GiftiImage(darrays=[array]),
                    )
                )
                start += count
        for path, image in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_output_path(path) as staged:
                nib.save(image, staged)
            sidecar = path.with_name(
                path.name.removesuffix(".nii.gz").removesuffix(".shape.gii") + ".json"
            )
            atomic_write_json(sidecar, {**metadata, "Statistic": statistic})
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
