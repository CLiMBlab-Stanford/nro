from __future__ import annotations

import numpy as np
import pandas as pd

from nro.clean.module import _build_cleaning_projection


def test_projection_fit_ignores_censored_signal_but_evaluates_every_frame() -> None:
    n_scans = 80
    time = np.arange(n_scans, dtype=np.float64)
    nuisance = np.sin(2.0 * np.pi * time / 17.0)
    neural = np.cos(2.0 * np.pi * time / 29.0)
    outliers = pd.DataFrame(
        {"motion_outlier00": np.equal(time, 31.0).astype(np.float64)}
    )
    projection = _build_cleaning_projection(
        confounds=pd.DataFrame({"nuisance": nuisance}),
        outliers=outliers,
        tr=1.0,
        detrend=False,
        standardize=False,
        high_pass=None,
        low_pass=None,
    )

    baseline = (4.0 * nuisance + neural)[:, None]
    contaminated = baseline.copy()
    contaminated[31, 0] += 10_000.0
    baseline_clean = projection.transform(baseline)
    contaminated_clean = projection.transform(contaminated)

    np.testing.assert_allclose(
        contaminated_clean[projection.retained],
        baseline_clean[projection.retained],
        atol=1e-5,
    )
    assert contaminated_clean.shape == contaminated.shape
    assert np.all(np.isfinite(contaminated_clean))
    assert contaminated_clean[31, 0] - baseline_clean[31, 0] == 10_000.0


def test_projection_removes_stopband_and_standardizes_from_retained_frames() -> None:
    n_scans = 240
    tr = 1.0
    time = np.arange(n_scans, dtype=np.float64) * tr
    passband = np.sin(2.0 * np.pi * 0.05 * time)
    stopband = (
        2.0 * np.sin(2.0 * np.pi * (1.0 / n_scans) * time)
        + 3.0 * np.cos(2.0 * np.pi * 0.20 * time)
    )
    outlier_values = np.zeros(n_scans, dtype=np.float64)
    outlier_values[[50, 151]] = 1.0
    projection = _build_cleaning_projection(
        confounds=pd.DataFrame(index=np.arange(n_scans)),
        outliers=pd.DataFrame({"dvars_outlier00": outlier_values}),
        tr=tr,
        detrend=False,
        standardize=True,
        high_pass=0.01,
        low_pass=0.10,
    )
    signal = (passband + stopband)[:, None]
    signal[50, 0] += 1_000.0
    signal[151, 0] -= 1_000.0

    cleaned = projection.transform(signal)[:, 0]
    retained_cleaned = cleaned[projection.retained]

    assert cleaned.shape == (n_scans,)
    assert np.all(np.isfinite(cleaned))
    np.testing.assert_allclose(retained_cleaned.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(retained_cleaned.std(ddof=0), 1.0, atol=1e-6)
    assert np.corrcoef(retained_cleaned, passband[projection.retained])[0, 1] > 0.99

