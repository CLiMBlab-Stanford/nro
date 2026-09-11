#!/usr/bin/env python3
"""
Assemble a confounds TSV broadly compatible with common fMRIPrep-style regressors.

Compute six MCFLIRT motion parameters, mean white-matter/CSF/global signals,
and the derivatives and squared terms of the nine base signals. Also export
Power framewise displacement, aCompCor components, DVARS diagnostics, and
numbered non-steady-state, extreme-FD, DVARS, and combined motion outliers.
Column availability does not determine clean-regressor selection; the clean
configuration owns that choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from nro.configuration.runtime import SETTINGS
from nro.engine.paths import resolve_project_path

WM_LABELS = frozenset(
    {
        2,  # Left-Cerebral-White-Matter
        41,  # Right-Cerebral-White-Matter
        7,  # Left-Cerebellum-White-Matter
        46,  # Right-Cerebellum-White-Matter
    }
)
CSF_LABELS = frozenset(
    {
        4,  # Left-Lateral-Ventricle
        43,  # Right-Lateral-Ventricle
        5,  # Left-Inf-Lat-Vent
        44,  # Right-Inf-Lat-Vent
        14,  # 3rd-Ventricle
        15,  # 4th-Ventricle
        24,  # CSF
    }
)


def _load_niimg(path: Path):
    try:
        import nibabel as nib
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency: nibabel is required to assemble confounds.\nImport error: {e}"
        )
    return nib.load(str(path))


def _resample_nearest(src_img, ref_img):
    if src_img.shape[:3] == ref_img.shape[:3] and np.allclose(src_img.affine, ref_img.affine):
        return src_img
    try:
        from nibabel.processing import resample_from_to
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Need nibabel.processing.resample_from_to (and typically scipy) to resample aseg to EPI grid.\n"
            f"Import error: {e}"
        )
    return resample_from_to(src_img, (ref_img.shape[:3], ref_img.affine), order=0)


def _derivative1(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x, dtype=np.float64)
    d[1:] = x[1:] - x[:-1]
    return d


def _power2(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) ** 2


def _fd_power(motion: np.ndarray, *, radius_mm: float) -> np.ndarray:
    """
    Framewise displacement (Power et al.): sum(|d(trans)|) + r*sum(|d(rot)|)
    motion: (T,6) where [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    rotations assumed radians, translations mm.
    """
    if motion.ndim != 2 or motion.shape[1] != 6:
        raise ValueError(f"Expected motion shape (T,6), got {motion.shape}")
    d = np.zeros_like(motion, dtype=np.float64)
    d[1:, :] = motion[1:, :] - motion[:-1, :]
    drot = np.abs(d[:, 0:3]) * float(radius_mm)
    dtrans = np.abs(d[:, 3:6])
    return (drot.sum(axis=1) + dtrans.sum(axis=1)).astype(np.float64)


def _infer_mcflirt_order(par: np.ndarray) -> np.ndarray:
    """
    MCFLIRT .par is typically [rotX rotY rotZ transX transY transZ].
    We keep that convention and name columns accordingly.
    """
    if par.ndim != 2 or par.shape[1] != 6:
        raise SystemExit(f"MCFLIRT par file must have 6 columns; got shape {par.shape}")
    return np.asarray(par, dtype=np.float64)


def _mean_signal_dense(
    epi_4d: np.ndarray,
    mask3d: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """
    Fast mean signal from a fully-loaded 4D array (X,Y,Z,T) and a 3D mask.
    Uses a tensordot reduction to avoid materializing (Nvox,T) masked copies.
    """
    if epi_4d.ndim != 4:
        raise ValueError(f"epi_4d must be 4D, got {epi_4d.shape}")
    if mask3d.shape != epi_4d.shape[:3]:
        raise ValueError(f"mask shape {mask3d.shape} does not match epi spatial {epi_4d.shape[:3]}")
    if not np.any(mask3d):
        raise SystemExit(f"{name} mask is empty; cannot compute its mean signal")
    m = mask3d.astype(np.float32, copy=False)
    denom = float(np.sum(m))
    if denom <= 0:
        raise SystemExit(f"{name} mask has zero weight; cannot compute its mean signal")
    # (X,Y,Z) · (X,Y,Z,T) -> (T,)
    signal = np.tensordot(m, epi_4d, axes=([0, 1, 2], [0, 1, 2])) / denom
    return np.asarray(signal, dtype=np.float64)


def _expanded_signal_columns(base: str, signal: np.ndarray) -> dict[str, np.ndarray]:
    derivative = _derivative1(signal)
    return {
        base: np.asarray(signal, dtype=np.float64),
        f"{base}_derivative1": derivative,
        f"{base}_power2": _power2(signal),
        f"{base}_derivative1_power2": _power2(derivative),
    }


def _acompcor_dense(
    *,
    epi_4d: np.ndarray,
    epi_fov: np.ndarray,
    aseg_data: np.ndarray,
    n_components: int,
    max_voxels: int,
) -> np.ndarray:
    """
    Fast aCompCor using a fully-loaded 4D array.
    """
    if epi_4d.ndim != 4:
        raise ValueError(f"epi_4d must be 4D, got {epi_4d.shape}")
    if aseg_data.shape != epi_4d.shape[:3]:
        raise SystemExit(
            f"aseg shape {aseg_data.shape} does not match EPI shape {epi_4d.shape[:3]}"
        )

    tissue_mask = np.isin(aseg_data, sorted(WM_LABELS | CSF_LABELS))
    mask = tissue_mask & epi_fov
    mask_idx = np.flatnonzero(mask.reshape(-1))
    if mask_idx.size < max(100, int(n_components) * 10):
        raise SystemExit(f"WM+CSF mask too small for aCompCor. Mask voxels: {mask_idx.size}.")

    # Pull WM+CSF voxels: (Vmask, T) in float32.
    t = int(epi_4d.shape[3])
    flat = epi_4d.reshape(-1, t)  # view
    X = np.asarray(flat[mask_idx, :], dtype=np.float32)

    # Select a subset by variance over time (to keep PCA stable & fast).
    if int(mask_idx.size) > int(max_voxels):
        var = X.var(axis=1, ddof=1, dtype=np.float64)
        k = int(max_voxels)
        top = np.argpartition(var, -k)[-k:]
        top = top[np.argsort(var[top])[::-1]]
        X = X[top, :]

    # Standardize per voxel (CompCor-style), then PCA across voxels => components over time.
    X = X.astype(np.float32, copy=False)
    X -= X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    std = X.std(axis=1, ddof=1, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    X /= std[:, None]

    mat = X.T  # (T, K)
    n = min(int(n_components), int(mat.shape[0]), int(mat.shape[1]))
    if n <= 0:
        raise SystemExit(f"Invalid aCompCor n_components={n_components} for matrix {mat.shape}")
    try:
        from sklearn.utils.extmath import randomized_svd
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: scikit-learn is required for fast aCompCor randomized SVD.\n"
            f"Import error: {e}"
        )
    u, s, _vt = randomized_svd(mat, n_components=n, n_iter=4, random_state=0)
    comps = u[:, :n] * s[:n]
    return comps.astype(np.float64)


def _nonsteady_spikes(
    gs: np.ndarray, *, max_vols: int, rel_thresh: float, stable_run: int
) -> list[int]:
    t = int(gs.size)
    if t == 0:
        return []
    tail_start = min(max(5, t // 10), t - 1) if t > 1 else 0
    baseline = float(np.median(gs[tail_start:])) if t > 1 else float(gs[0])
    if baseline == 0:
        baseline = float(np.median(gs)) or 1.0
    rel = np.abs(gs - baseline) / abs(baseline)
    spikes: list[int] = []
    stable = 0
    for i in range(min(int(max_vols), t)):
        if rel[i] > float(rel_thresh):
            spikes.append(i)
            stable = 0
        else:
            stable += 1
            if stable >= int(stable_run):
                break
    return spikes


def _spike_regressors(t: int, idxs: list[int], prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for j, i in enumerate(idxs):
        col = np.zeros((t,), dtype=np.float64)
        if 0 <= i < t:
            col[i] = 1.0
        out[f"{prefix}{j:02d}"] = col
    return out


def _dvars_metrics(
    epi_4d: np.ndarray,
    brain_mask: np.ndarray,
    *,
    statistical_alpha: float,
    practical_threshold_percent: float,
    power: float,
) -> dict[str, object]:
    """Compute Afyouni-Nichols DVARS inference without copying the 4D image."""
    from scipy.stats import chi2

    if epi_4d.ndim != 4:
        raise ValueError(f"epi_4d must be 4D, got {epi_4d.shape}")
    if brain_mask.shape != epi_4d.shape[:3]:
        raise ValueError(
            f"brain mask shape {brain_mask.shape} does not match EPI spatial shape "
            f"{epi_4d.shape[:3]}"
        )
    if not 0.0 < float(statistical_alpha) < 1.0:
        raise ValueError("DVARS statistical alpha must be between zero and one")
    if float(practical_threshold_percent) < 0.0:
        raise ValueError("DVARS practical threshold cannot be negative")
    if float(power) <= 0.0:
        raise ValueError("DVARS power transformation must be positive")

    frames = int(epi_4d.shape[3])
    finite_mask = np.asarray(brain_mask, dtype=bool).copy()
    for frame in range(frames):
        finite_mask &= np.isfinite(epi_4d[..., frame])
    voxel_count = int(np.count_nonzero(finite_mask))
    if voxel_count == 0:
        raise SystemExit("Brain mask contains no voxels with a finite complete time course")

    dvars = np.full(frames, np.nan, dtype=np.float64)
    p_values = np.full(frames, np.nan, dtype=np.float64)
    delta_percent = np.full(frames, np.nan, dtype=np.float64)
    if frames < 2:
        return {
            "dvars": dvars,
            "p_values": p_values,
            "delta_percent": delta_percent,
            "outlier_indices": [],
            "voxel_count": voxel_count,
            "null_mean": None,
            "null_sd": None,
            "spatial_degrees_of_freedom": None,
        }

    # DVARS is indexed to the later member of each adjacent frame pair. The
    # cited censoring rule marks both members when a pair is anomalous.
    dvars_squared = np.empty(frames - 1, dtype=np.float64)
    for frame in range(1, frames):
        difference = np.subtract(epi_4d[..., frame], epi_4d[..., frame - 1], dtype=np.float64)
        dvars_squared[frame - 1] = np.mean(np.square(difference[finite_mask]), dtype=np.float64)
    dvars[1:] = np.sqrt(dvars_squared)

    temporal_sum = np.zeros(epi_4d.shape[:3], dtype=np.float64)
    temporal_sum_squares = np.zeros(epi_4d.shape[:3], dtype=np.float64)
    for frame in range(frames):
        values = np.asarray(epi_4d[..., frame], dtype=np.float64)
        temporal_sum += values
        temporal_sum_squares += np.square(values)
    centered_sum_squares = temporal_sum_squares - np.square(temporal_sum) / float(frames)
    average_variance = float(
        np.sum(centered_sum_squares[finite_mask], dtype=np.float64) / float(voxel_count * frames)
    )
    average_variance = max(average_variance, 0.0)

    transformed = np.power(dvars_squared, float(power))
    transformed_median = float(np.median(transformed))
    null_mean = float(np.median(dvars_squared))
    lower_quartile = float(np.percentile(transformed, 25.0))
    transformed_half_iqr_sd = 2.0 * (transformed_median - lower_quartile) / 1.349
    null_sd = (
        float(
            (1.0 / float(power))
            * np.power(transformed_median, 1.0 / float(power) - 1.0)
            * transformed_half_iqr_sd
        )
        if transformed_median > 0.0
        else 0.0
    )

    spatial_dof: float | None = None
    if null_mean > 0.0 and null_sd > 0.0 and np.isfinite(null_sd):
        spatial_dof = 2.0 * np.square(null_mean) / np.square(null_sd)
        scale = 2.0 * null_mean / np.square(null_sd)
        p_values[1:] = chi2.sf(scale * dvars_squared, spatial_dof)
    if average_variance > 0.0 and np.isfinite(average_variance):
        delta_percent[1:] = (dvars_squared - null_mean) / (4.0 * average_variance) * 100.0

    pair_outliers = np.flatnonzero(
        (p_values[1:] < float(statistical_alpha) / float(frames - 1))
        & (delta_percent[1:] > float(practical_threshold_percent))
    )
    outlier_indices = sorted(
        {int(frame) for pair in pair_outliers.tolist() for frame in (pair, pair + 1)}
    )
    return {
        "dvars": dvars,
        "p_values": p_values,
        "delta_percent": delta_percent,
        "outlier_indices": outlier_indices,
        "voxel_count": voxel_count,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "spatial_degrees_of_freedom": spatial_dof,
    }


def get_confounds(
    *,
    epi: Path,
    epi_mean: Path,
    mcflirt_par: Path,
    subjects_dir: Path,
    fs_subject: str,
    out_tsv: Path,
    out_json: Path,
    aseg_in_epi: Optional[Path] = None,
    brain_mask_in_epi: Optional[Path] = None,
    n_acompcor: Optional[int] = None,
    acompcor_max_voxels: Optional[int] = None,
    fd_radius_mm: Optional[float] = None,
    motion_outlier_fd_thresh: Optional[float] = None,
    dvars_statistical_alpha: Optional[float] = None,
    dvars_practical_threshold_percent: Optional[float] = None,
    dvars_power: Optional[float] = None,
    nonsteady_max_vols: Optional[int] = None,
    nonsteady_rel_thresh: Optional[float] = None,
    nonsteady_stable_run: Optional[int] = None,
) -> int:
    """Generate one confounds TSV, metadata sidecar, and outlier indicators."""
    cfg = SETTINGS.get_confounds
    epi = Path(epi)
    epi_mean = Path(epi_mean)
    mcflirt_par = Path(mcflirt_par)
    subjects_dir = Path(subjects_dir)
    out_tsv = Path(out_tsv)
    out_json = Path(out_json)
    aseg_in_epi = Path(aseg_in_epi) if aseg_in_epi is not None else None
    brain_mask_in_epi = Path(brain_mask_in_epi) if brain_mask_in_epi is not None else None

    n_acompcor = int(cfg.n_acompcor if n_acompcor is None else n_acompcor)
    acompcor_max_voxels = int(
        cfg.acompcor_max_voxels if acompcor_max_voxels is None else acompcor_max_voxels
    )
    fd_radius_mm = float(cfg.fd_radius_mm if fd_radius_mm is None else fd_radius_mm)
    motion_outlier_fd_thresh = float(
        cfg.motion_outlier_fd_thresh
        if motion_outlier_fd_thresh is None
        else motion_outlier_fd_thresh
    )
    dvars_statistical_alpha = float(
        cfg.dvars_statistical_alpha if dvars_statistical_alpha is None else dvars_statistical_alpha
    )
    dvars_practical_threshold_percent = float(
        cfg.dvars_practical_threshold_percent
        if dvars_practical_threshold_percent is None
        else dvars_practical_threshold_percent
    )
    dvars_power = float(cfg.dvars_power if dvars_power is None else dvars_power)
    nonsteady_max_vols = int(
        cfg.nonsteady_max_vols if nonsteady_max_vols is None else nonsteady_max_vols
    )
    nonsteady_rel_thresh = float(
        cfg.nonsteady_rel_thresh if nonsteady_rel_thresh is None else nonsteady_rel_thresh
    )
    nonsteady_stable_run = int(
        cfg.nonsteady_stable_run if nonsteady_stable_run is None else nonsteady_stable_run
    )

    epi_img = _load_niimg(epi)
    epi_mean_img = _load_niimg(epi_mean)
    if len(epi_img.shape) != 4:
        raise SystemExit(f"--epi must be 4D, got shape {epi_img.shape}")
    if len(epi_mean_img.shape) != 3:
        raise SystemExit(f"--epi-mean must be 3D, got shape {epi_mean_img.shape}")
    t = int(epi_img.shape[3])

    par = np.loadtxt(str(mcflirt_par), dtype=np.float64)
    if par.ndim == 1:
        par = par.reshape(1, -1)
    par = _infer_mcflirt_order(par)
    if par.shape[0] != t:
        raise SystemExit(f"Motion .par rows ({par.shape[0]}) do not match EPI timepoints ({t}).")

    # Load or derive aseg labels in EPI grid.
    if aseg_in_epi is not None:
        aseg_img = _load_niimg(aseg_in_epi)
        if aseg_img.shape[:3] != epi_mean_img.shape[:3]:
            raise SystemExit(
                f"--aseg-in-epi spatial shape {aseg_img.shape[:3]} does not match --epi-mean {epi_mean_img.shape[:3]}"
            )
        aseg_resampled = np.asanyarray(aseg_img.dataobj, dtype=np.int16)
    else:
        # FreeSurfer aseg provides a robust brain mask in T1 space; resample it to the EPI grid
        # (epi_in_t1 is not explicitly brain-masked in bold_to_t1.py).
        aseg = subjects_dir / fs_subject / "mri" / "aseg.mgz"
        if not aseg.exists():
            raise SystemExit(f"Missing FreeSurfer aseg: {aseg}")
        aseg_img = _load_niimg(aseg)
        aseg_img = _resample_nearest(aseg_img, epi_mean_img)
        aseg_resampled = np.asanyarray(aseg_img.dataobj, dtype=np.int16)

    epi_fov = np.asanyarray(epi_mean_img.dataobj, dtype=np.float32) != 0.0
    if brain_mask_in_epi is not None:
        brain_mask_img = _load_niimg(brain_mask_in_epi)
        if brain_mask_img.shape[:3] != epi_mean_img.shape[:3]:
            raise SystemExit(
                f"--brain-mask-in-epi spatial shape {brain_mask_img.shape[:3]} does not match --epi-mean {epi_mean_img.shape[:3]}"
            )
        brain_mask = np.asanyarray(brain_mask_img.dataobj, dtype=np.float32) > 0.0
        brain_mask &= epi_fov
    else:
        brain_mask = (aseg_resampled != 0) & epi_fov

    # Materialize the registered 4D EPI once. Repeatedly indexing a compressed
    # proxy by timepoint can decompress the entire .nii.gz hundreds of times.
    epi_4d = np.asanyarray(epi_img.dataobj, dtype=np.float32)
    wm_mask = np.isin(aseg_resampled, sorted(WM_LABELS)) & epi_fov
    csf_mask = np.isin(aseg_resampled, sorted(CSF_LABELS)) & epi_fov
    gs = _mean_signal_dense(epi_4d, brain_mask, name="Brain")
    wm = _mean_signal_dense(epi_4d, wm_mask, name="White-matter")
    csf = _mean_signal_dense(epi_4d, csf_mask, name="CSF")
    acomps = _acompcor_dense(
        epi_4d=epi_4d,
        epi_fov=epi_fov,
        aseg_data=aseg_resampled,
        n_components=int(n_acompcor),
        max_voxels=int(acompcor_max_voxels),
    )

    # aCompCor from WM+CSF computed above.

    rot = par[:, 0:3]
    trans = par[:, 3:6]

    cols: dict[str, np.ndarray] = {}
    for i, ax in enumerate(["x", "y", "z"]):
        cols.update(_expanded_signal_columns(f"rot_{ax}", rot[:, i]))
    for i, ax in enumerate(["x", "y", "z"]):
        cols.update(_expanded_signal_columns(f"trans_{ax}", trans[:, i]))

    cols.update(_expanded_signal_columns("white_matter", wm))
    cols.update(_expanded_signal_columns("csf", csf))
    cols.update(_expanded_signal_columns("global_signal", gs))

    for i in range(acomps.shape[1]):
        cols[f"a_comp_cor_{i:02d}"] = acomps[:, i]

    fd = _fd_power(par, radius_mm=float(fd_radius_mm))
    cols["framewise_displacement"] = fd
    dvars_metrics = _dvars_metrics(
        epi_4d,
        brain_mask,
        statistical_alpha=dvars_statistical_alpha,
        practical_threshold_percent=dvars_practical_threshold_percent,
        power=dvars_power,
    )
    cols["dvars"] = np.asarray(dvars_metrics["dvars"], dtype=np.float64)
    cols["dvars_p_value"] = np.asarray(dvars_metrics["p_values"], dtype=np.float64)
    cols["dvars_delta_percent"] = np.asarray(dvars_metrics["delta_percent"], dtype=np.float64)

    nonsteady_idx = _nonsteady_spikes(
        gs,
        max_vols=int(nonsteady_max_vols),
        rel_thresh=float(nonsteady_rel_thresh),
        stable_run=int(nonsteady_stable_run),
    )
    cols.update(_spike_regressors(t, nonsteady_idx, prefix="non_steady_state_outlier"))

    extreme_fd_idx = [int(i) for i in np.flatnonzero(fd > float(motion_outlier_fd_thresh)).tolist()]
    dvars_idx = [int(index) for index in dvars_metrics["outlier_indices"]]
    motion_idx = sorted(set(extreme_fd_idx) | set(dvars_idx))
    cols.update(_spike_regressors(t, extreme_fd_idx, prefix="extreme_fd_outlier"))
    cols.update(_spike_regressors(t, dvars_idx, prefix="dvars_outlier"))
    cols.update(_spike_regressors(t, motion_idx, prefix="motion_outlier"))

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency: pandas is required to write the confounds TSV.\nImport error: {e}"
        )

    # Stable column ordering: 36P signals first, then aCompCor/FD/spikes.
    ordered: list[str] = []
    for base in [f"trans_{a}" for a in "xyz"] + [f"rot_{a}" for a in "xyz"]:
        ordered.append(base)
        ordered.append(base + "_derivative1")
        ordered.append(base + "_power2")
        ordered.append(base + "_derivative1_power2")
    for base in ("white_matter", "csf", "global_signal"):
        ordered.append(base)
        ordered.append(base + "_derivative1")
        ordered.append(base + "_power2")
        ordered.append(base + "_derivative1_power2")
    ordered += [f"a_comp_cor_{i:02d}" for i in range(acomps.shape[1])]
    ordered += ["framewise_displacement", "dvars", "dvars_p_value", "dvars_delta_percent"]
    ordered += sorted([k for k in cols.keys() if k.startswith("non_steady_state_outlier")])
    ordered += sorted([k for k in cols.keys() if k.startswith("extreme_fd_outlier")])
    ordered += sorted([k for k in cols.keys() if k.startswith("dvars_outlier")])
    ordered += sorted([k for k in cols.keys() if k.startswith("motion_outlier")])

    df = pd.DataFrame({k: cols[k] for k in ordered if k in cols})
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a")

    meta = {
        "sources": {
            "epi": str(epi),
            "epi_mean": str(epi_mean),
            "mcflirt_par": str(mcflirt_par),
            "subjects_dir": str(subjects_dir),
            "fs_subject": str(fs_subject),
            "aseg_in_epi": (str(aseg_in_epi) if aseg_in_epi is not None else None),
            "brain_mask_in_epi": (
                str(brain_mask_in_epi) if brain_mask_in_epi is not None else None
            ),
        },
        "parameters": {
            "fd_radius_mm": float(fd_radius_mm),
            "motion_outlier_fd_thresh": float(motion_outlier_fd_thresh),
            "dvars_statistical_alpha": float(dvars_statistical_alpha),
            "dvars_practical_threshold_percent": float(dvars_practical_threshold_percent),
            "dvars_power": float(dvars_power),
            "n_acompcor": int(n_acompcor),
            "acompcor_max_voxels": int(acompcor_max_voxels),
            "nonsteady_max_vols": int(nonsteady_max_vols),
            "nonsteady_rel_thresh": float(nonsteady_rel_thresh),
            "nonsteady_stable_run": int(nonsteady_stable_run),
        },
        "columns": list(df.columns),
        "dvars": {
            "finite_brain_voxels": int(dvars_metrics["voxel_count"]),
            "null_mean_dvars_squared": dvars_metrics["null_mean"],
            "null_sd_dvars_squared": dvars_metrics["null_sd"],
            "spatial_degrees_of_freedom": dvars_metrics["spatial_degrees_of_freedom"],
            "outlier_frames": dvars_idx,
        },
        "notes": [
            "rot_* are in radians as reported by MCFLIRT; trans_* are in mm.",
            "framewise_displacement uses Power-style FD with fixed head radius.",
            "DVARS inference uses the transformed chi-squared, median expected-value, and half-IQR variance estimates described by Afyouni and Nichols (2018).",
            "A significant DVARS difference marks both frames in the adjacent pair.",
            "global_signal is computed within brain_mask_in_epi when provided; otherwise it falls back to an aseg-derived brain mask (aseg!=0) resampled to the EPI grid.",
            "white_matter and csf are mean signals from FreeSurfer aseg tissue labels resampled to the EPI grid.",
            "The six motion and three mean-signal regressors include derivative1, power2, and derivative1_power2 expansions for the Satterthwaite 36-parameter model.",
            "aCompCor is computed from WM+CSF voxels using FreeSurfer aseg labels and nearest-neighbor resampling to EPI grid, using randomized SVD for the top components.",
            "get_confounds loads the full 4D EPI exactly once to avoid repeated .nii.gz decompression.",
        ],
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Build an fMRIPrep-style confounds table for a registered BOLD run."""
    cfg = SETTINGS.get_confounds
    ap = argparse.ArgumentParser(prog="get_confounds.py")
    ap.add_argument(
        "--epi", required=True, type=Path, help="4D EPI (typically a registered BOLD run)"
    )
    ap.add_argument("--epi-mean", required=True, type=Path, help="3D EPI mean")
    ap.add_argument("--mcflirt-par", required=True, type=Path, help="MCFLIRT .par file from -plots")
    ap.add_argument(
        "--subjects-dir", required=True, type=Path, help="FreeSurfer SUBJECTS_DIR (for aseg.mgz)"
    )
    ap.add_argument("--fs-subject", required=True, type=str, help="FreeSurfer subject ID")
    ap.add_argument(
        "--aseg-in-epi",
        default=cfg.aseg_in_epi,
        type=Path,
        help="Optional aseg labels already resampled into the EPI mean grid (skips resampling inside get_confounds).",
    )
    ap.add_argument(
        "--brain-mask-in-epi",
        default=cfg.brain_mask_in_epi,
        type=Path,
        help="Optional brain mask already resampled into the EPI mean grid (used for global signal).",
    )
    ap.add_argument("--out-tsv", required=True, type=Path, help="Output confounds TSV")
    ap.add_argument("--out-json", required=True, type=Path, help="Output confounds JSON sidecar")
    ap.add_argument(
        "--project",
        default=SETTINGS.common.project,
        help="BIDS project name under the configured top-level data directory.",
    )
    ap.add_argument(
        "--n-acompcor",
        type=int,
        default=cfg.n_acompcor,
        help="Number of aCompCor components to compute (default: 10)",
    )
    ap.add_argument(
        "--acompcor-max-voxels",
        type=int,
        default=cfg.acompcor_max_voxels,
        help="Max voxels for PCA (default: 20000)",
    )
    ap.add_argument(
        "--fd-radius-mm",
        type=float,
        default=cfg.fd_radius_mm,
        help="Head radius for FD (mm) (default: 50)",
    )
    ap.add_argument(
        "--motion-outlier-fd-thresh",
        type=float,
        default=cfg.motion_outlier_fd_thresh,
        help="FD threshold in mm for extreme-FD outliers (default: %(default)s)",
    )
    ap.add_argument(
        "--dvars-statistical-alpha",
        type=float,
        default=cfg.dvars_statistical_alpha,
        help="DVARS family-wise statistical alpha (default: %(default)s)",
    )
    ap.add_argument(
        "--dvars-practical-threshold-percent",
        type=float,
        default=cfg.dvars_practical_threshold_percent,
        help="Minimum change in percent D-var (default: %(default)s)",
    )
    ap.add_argument(
        "--dvars-power",
        type=float,
        default=cfg.dvars_power,
        help="DVARS power transformation (default: %(default)s)",
    )
    ap.add_argument(
        "--nonsteady-max-vols",
        type=int,
        default=cfg.nonsteady_max_vols,
        help="Max initial vols for nonsteady detection (default: 20)",
    )
    ap.add_argument(
        "--nonsteady-rel-thresh",
        type=float,
        default=cfg.nonsteady_rel_thresh,
        help="Relative GS deviation threshold (default: 0.05)",
    )
    ap.add_argument(
        "--nonsteady-stable-run",
        type=int,
        default=cfg.nonsteady_stable_run,
        help="Stop after N stable vols (default: 3)",
    )

    args = ap.parse_args(argv)
    project = str(args.project)
    return get_confounds(
        epi=resolve_project_path(args.epi, project=project),
        epi_mean=resolve_project_path(args.epi_mean, project=project),
        mcflirt_par=resolve_project_path(args.mcflirt_par, project=project),
        subjects_dir=resolve_project_path(args.subjects_dir, project=project),
        fs_subject=str(args.fs_subject),
        aseg_in_epi=resolve_project_path(args.aseg_in_epi, project=project),
        brain_mask_in_epi=resolve_project_path(args.brain_mask_in_epi, project=project),
        out_tsv=resolve_project_path(args.out_tsv, project=project),
        out_json=resolve_project_path(args.out_json, project=project),
        n_acompcor=int(args.n_acompcor),
        acompcor_max_voxels=int(args.acompcor_max_voxels),
        fd_radius_mm=float(args.fd_radius_mm),
        motion_outlier_fd_thresh=float(args.motion_outlier_fd_thresh),
        dvars_statistical_alpha=float(args.dvars_statistical_alpha),
        dvars_practical_threshold_percent=float(args.dvars_practical_threshold_percent),
        dvars_power=float(args.dvars_power),
        nonsteady_max_vols=int(args.nonsteady_max_vols),
        nonsteady_rel_thresh=float(args.nonsteady_rel_thresh),
        nonsteady_stable_run=int(args.nonsteady_stable_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
