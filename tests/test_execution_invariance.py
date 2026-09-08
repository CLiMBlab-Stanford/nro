"""Execution tiling must not define scientific groups or change estimators."""

from pathlib import Path

import numpy as np
import pytest

from nro.firstlevels.statistics import fit_glm
from nro.microparcellation.statistics import _moments, local_edge_correlations, parcel_correlations
from nro.microparcellation.cifti import write_pconn, write_volume_dlabel
from nro.networks.adjacency import pconn_to_adjacency


def _execution_fields():
    from nro.configuration.schema import SCHEMAS, Field

    def walk(kind, fields, prefix=()):
        for key, rule in fields.items():
            if isinstance(rule, dict):
                yield from walk(kind, rule, (*prefix, key))
            elif isinstance(rule, Field) and rule.execution:
                yield kind, (*prefix, key)
    return [item for kind, fields in SCHEMAS.items() for item in walk(kind, fields)]


@pytest.mark.parametrize("kind,keys", _execution_fields())
def test_every_declared_execution_setting_is_excluded_from_scientific_identity(kind, keys):
    from copy import deepcopy
    from nro.configuration.store import ConfigStore

    store = ConfigStore()
    original = store.load_configuration(kind, "main")
    values = deepcopy(original.values)
    parent = values
    for key in keys[:-1]:
        parent = parent[key]
    old = parent[keys[-1]]
    parent[keys[-1]] = not old if isinstance(old, bool) else (old or 0) + 1
    changed = store.load_configuration(kind, "main", document=values)
    assert changed.fingerprint != original.fingerprint
    assert changed.scientific_fingerprint == original.scientific_fingerprint


def test_split_half_block_frames_is_scientific():
    from nro.configuration.store import ConfigStore

    store = ConfigStore()
    original = store.load_configuration("microparcellation", "main")
    values = original.values
    before = original.scientific_fingerprint
    values["quality"]["split_half_block_frames"] += 1
    assert store.load_configuration("microparcellation", "main", document=values).scientific_fingerprint != before


@pytest.mark.parametrize("run_count", [1, 2])
@pytest.mark.parametrize("gsr", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_connectivity_and_quality_are_independent_of_execution_tiles(run_count, gsr, weighted):
    rng = np.random.default_rng(827)
    labels = np.repeat(np.arange(6), 2)
    mask = np.ones(len(labels), dtype=bool)
    files = tuple((Path(str(index)),) for index in range(run_count))
    runs = {}
    for index, path in enumerate(files):
        signal = rng.normal(size=(241 - index * 10, 6))
        signal[:, 1] += signal[:, 0]
        runs[path] = (1e4 + signal[:, labels] + rng.normal(scale=.4, size=(len(signal), len(labels)))).astype(np.float32)
    edges = np.column_stack((np.arange(len(labels) - 1), np.arange(1, len(labels))))

    def compute(temporal, spatial):
        options = dict(global_signal_regression=gsr, reliability_weighting=weighted,
                       reliability_vertex_block_size=spatial, load_run=runs.__getitem__)
        local = local_edge_correlations(files, edges, len(labels), temporal, **options)
        parcels = parcel_correlations(files, labels, mask, temporal,
            split_half_block_frames=19, null_partitions=(np.roll(labels, 1),), **options)
        return local, parcels

    expected_local, expected = compute(31, 5)
    for temporal, spatial in ((1, 1), (7, 3), (128, 64), (1024, 1024)):
        local, actual = compute(temporal, spatial)
        np.testing.assert_allclose(local.correlations, expected_local.correlations, atol=3e-6, rtol=2e-5)
        for field in ("correlations", "variance_preserved", "null_variance_preserved", "parcel_reliability_mean",
                      "parcel_effective_runs", "total_sum_squares", "residual_sum_squares"):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=3e-6, rtol=2e-5)
        for field, value in expected.split_half.items():
            if isinstance(value, float):
                np.testing.assert_allclose(actual.split_half[field], value, atol=3e-6, rtol=2e-5)
            else:
                assert actual.split_half[field] == value
        assert actual.split_half["temporal_block_size"] == (19 if run_count == 1 else None)


def test_high_offset_moments_do_not_depend_on_streaming_blocks():
    rng = np.random.default_rng(9)
    data = (1e7 + rng.normal(scale=2, size=(501, 5))).astype(np.float32)
    expected_mean = data.mean(axis=0, dtype=np.float64)
    expected_m2 = np.sum((data.astype(np.float64) - expected_mean) ** 2, axis=0)
    for size in (1, 3, 64, 501):
        mean, m2 = _moments(data, size)
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-7)
        np.testing.assert_allclose(m2, expected_m2, rtol=1e-8)


@pytest.mark.parametrize("grid", [[0.0], [-.3, 0., .3, .6]])
def test_glm_spatial_blocks_preserve_effects_variances_and_noise_groups(grid):
    rng = np.random.default_rng(42)
    retained = np.ones(80, dtype=bool)
    retained[[3, 5, 21, 40]] = False
    design = np.column_stack((np.ones(80), rng.normal(size=(80, 3))))
    data = design @ rng.normal(size=(4, 17)) + rng.normal(size=(80, 17))
    expected = fit_glm(data, design[retained], retained=retained, ar_grid=np.array(grid), block_size=17)
    for size in (1, 4, 128):
        actual = fit_glm(data, design[retained], retained=retained, ar_grid=np.array(grid), block_size=size)
        np.testing.assert_array_equal(actual.groups, expected.groups)
        np.testing.assert_allclose(actual.beta, expected.beta, atol=1e-7)
        np.testing.assert_allclose(actual.residual_variance, expected.residual_variance, atol=1e-7)
        np.testing.assert_array_equal(actual.covariance, expected.covariance)
        assert actual.dof == expected.dof


def test_network_sparsification_blocks_preserve_exact_edges_and_threshold(tmp_path):
    _, parcels = write_volume_dlabel(tmp_path / "labels.dlabel.nii", np.arange(8),
                                     np.ones((2, 2, 2), bool), np.eye(4))
    rng = np.random.default_rng(80)
    weights = rng.uniform(-1, 1, (8, 8))
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)
    path = write_pconn(tmp_path / "connectivity.pconn.nii", weights, parcels)
    for transform in ("absolute", "clip_positive", "square"):
        options = dict(transform=transform, minimum_weight=0, percentile_cutoff=90)
        expected, threshold = pconn_to_adjacency(path, block_size=8, **options)
        for size in (1, 3, 32):
            actual, got_threshold = pconn_to_adjacency(path, block_size=size, **options)
            assert got_threshold == threshold
            np.testing.assert_array_equal(actual.toarray(), expected.toarray())
