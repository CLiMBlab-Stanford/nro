from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.modules.func import steps as func_steps
from nro.modules.func.confounds import _nonsteady_spikes


def _reference_detection(
    image_path: Path,
    *,
    max_vols: int,
    rel_thresh: float,
    stable_run: int,
) -> tuple[np.ndarray, int]:
    image = nib.load(image_path)
    total_volumes = int(image.shape[3])
    global_signal = np.zeros(total_volumes, dtype=np.float64)
    for volume_index in range(total_volumes):
        volume = np.asarray(image.dataobj[..., volume_index], dtype=np.float32)
        support = np.isfinite(volume) & (volume != 0)
        global_signal[volume_index] = float(np.median(volume[support])) if support.any() else 0.0
    nonsteady = set(
        _nonsteady_spikes(
            global_signal,
            max_vols=max_vols,
            rel_thresh=rel_thresh,
            stable_run=stable_run,
        )
    )
    dropped = 0
    while dropped in nonsteady:
        dropped += 1
    if dropped >= total_volumes:
        dropped = 0
    return global_signal, dropped


def test_bulk_nonsteady_detection_matches_reference_loop(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    data = rng.integers(-200, 1200, size=(17, 13, 9, 24), dtype=np.int16)
    data[0:2, :, :, :] = 0
    data[..., 0] *= 2
    data[..., 1] = np.where(data[..., 1] != 0, data[..., 1] * 3 // 2, 0)
    image_path = tmp_path / "provisional_mc.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image_path)
    parameters = {"max_vols": 20, "rel_thresh": 0.05, "stable_run": 3}

    reference_signal, reference_dropped = _reference_detection(image_path, **parameters)
    bulk = func_steps._detect_initial_nonsteady_volumes(image_path, **parameters)

    np.testing.assert_array_equal(
        np.asarray(bulk["PerVolumeNonzeroMedianSignal"]),
        reference_signal,
    )
    assert bulk["InitialNonSteadyStateVolumesExcluded"] == reference_dropped
    assert bulk["TotalBOLDVolumes"] == data.shape[3]
