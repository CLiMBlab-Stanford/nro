from __future__ import annotations

import numpy as np
import pandas as pd

from nro.clean.module import _build_cleaning_projection, _cleaned_timecourse_quality


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
    # Missing rows make the passband and stopband non-orthogonal on the retained
    # grid, but the direct passband fit remains dominated by the true component.
    assert np.corrcoef(retained_cleaned, passband[projection.retained])[0, 1] > 0.98


def test_clean_passband_uses_pca_without_stopband_saturation() -> None:
    rng = np.random.default_rng(42)
    n_scans = 445
    tr = 1.08
    time = np.arange(n_scans, dtype=np.float64) * tr
    confounds = pd.DataFrame(
        rng.normal(size=(n_scans, 36)),
        columns=[f"confound_{index:02d}" for index in range(36)],
    )
    task = pd.DataFrame({"task.condition": np.sin(2.0 * np.pi * 0.03 * time)})
    mask = np.zeros(n_scans, dtype=np.float64)
    mask[rng.choice(n_scans, size=88, replace=False)] = 1.0
    projection = _build_cleaning_projection(
        confounds=confounds,
        task=task,
        outliers=pd.DataFrame({"motion_outlier00": mask}),
        tr=tr,
        detrend=True,
        standardize=True,
        high_pass=0.01,
        low_pass=0.10,
        nuisance_variance_explained=0.99,
    )

    assert projection.passband_dimension == 88
    assert projection.passband_rank == 88
    assert 0 < projection.nuisance_pca_components <= 36
    assert 0 < projection.algebraic_temporal_rank < projection.passband_rank

    signal = rng.normal(size=(n_scans, 128)) + 5.0 * task.to_numpy()
    cleaned = projection.transform(signal)
    quality = _cleaned_timecourse_quality(cleaned, projection=projection)

    assert cleaned.shape == signal.shape
    assert np.all(np.isfinite(cleaned))
    assert quality["ObservedTemporalRank"] <= projection.algebraic_temporal_rank
    assert quality["EntropyEffectiveTemporalRank"] <= quality["ObservedTemporalRank"]


def test_nuisance_regression_is_skipped_below_absolute_rank_floor() -> None:
    rng = np.random.default_rng(7)
    n_scans = 100
    projection = _build_cleaning_projection(
        confounds=pd.DataFrame(rng.normal(size=(n_scans, 12))),
        outliers=pd.DataFrame(index=np.arange(n_scans)),
        tr=1.0,
        detrend=True,
        standardize=False,
        high_pass=0.01,
        low_pass=0.10,
        nuisance_variance_explained=0.99,
        minimum_temporal_rank=30,
        minimum_temporal_rank_fraction=0.5,
    )

    assert projection.cleaning_defined
    assert projection.post_exact_temporal_rank < 30
    assert not projection.temporal_rank_floor_satisfied
    assert projection.nuisance_pca_components == 0
    assert projection.algebraic_temporal_rank == projection.post_exact_temporal_rank


def test_unidentifiable_passband_produces_explicit_zero_sentinel() -> None:
    rng = np.random.default_rng(8)
    n_scans = 240
    outlier = np.ones(n_scans, dtype=np.float64)
    outlier[:30] = 0.0
    projection = _build_cleaning_projection(
        confounds=pd.DataFrame(rng.normal(size=(n_scans, 4))),
        outliers=pd.DataFrame({"motion_outlier00": outlier}),
        tr=1.0,
        detrend=True,
        standardize=True,
        high_pass=0.01,
        low_pass=0.10,
        nuisance_variance_explained=0.99,
        minimum_temporal_rank=30,
        minimum_temporal_rank_fraction=0.5,
    )

    assert not projection.cleaning_defined
    assert (
        projection.undefined_reason
        == "passband_basis_not_identifiable_from_retained_frames"
    )
    assert projection.passband_rank < projection.passband_dimension
    assert projection.algebraic_temporal_rank == 0

    cleaned = projection.transform(rng.normal(size=(n_scans, 10)))
    assert cleaned.dtype == np.float32
    assert np.count_nonzero(cleaned) == 0
    quality = _cleaned_timecourse_quality(cleaned, projection=projection)
    assert quality["ObservedTemporalRank"] == 0
    assert quality["NonconstantLocationCount"] == 0
