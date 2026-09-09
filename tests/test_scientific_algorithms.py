import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nro.configuration.store import ConfigStore
from nro.engine.cifti import load_dlabel, load_pconn
from nro.engine.surface_geometry import load_surfaces
from nro.modules.func.synbold_disco import ensure_image
from nro.modules.microparcellation.cifti import write_dlabel, write_pconn
from nro.modules.microparcellation.coarsen import _variation_edge_costs, loukas_variation_edges
from nro.modules.microparcellation.config import CoarseningConfig
from nro.modules.microparcellation.module import _coarsening_targets, _region_edges
from nro.modules.microparcellation.quality import spatial_null_partitions
from nro.modules.microparcellation.statistics import (
    _accumulate_gram_rows,
    _normalize_symmetric_gram,
    _profile_reliability,
    _quarter_standardized,
    _run_gram_statistics,
    local_edge_correlations,
    make_parcel_mean_loader,
    parcel_correlations,
)
from nro.modules.networks.adjacency import lower_triangular_adjacency, pconn_to_adjacency
from nro.modules.networks.config import OslomConfig
from nro.modules.networks.consensus import membership_stability
from nro.modules.networks.leiden import write_hint
from nro.modules.networks.oslom import parse_tp, resolve_oslom_executable, run_oslom
from nro.orchestration.runner import Runner


class ScientificAlgorithmTests(unittest.TestCase):
    def test_tiled_gram_and_temporal_dual_statistics_match_dense_form(self):
        rng = np.random.default_rng(18)
        timecourses = rng.normal(size=(17, 23)).astype(np.float32)
        expected = timecourses.T @ timecourses
        accumulated = np.zeros_like(expected)

        _accumulate_gram_rows(accumulated, timecourses, block_size=5)
        trace, frobenius = _run_gram_statistics(timecourses)

        np.testing.assert_allclose(accumulated, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(accumulated, accumulated.T)
        self.assertAlmostEqual(trace, float(np.trace(expected, dtype=np.float64)), places=4)
        self.assertAlmostEqual(frobenius, float(np.linalg.norm(expected)), places=4)

        diagonal = np.sqrt(np.maximum(np.diag(expected), 0.0))
        inverse_scale = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal > 0)
        expected_correlations = expected / np.outer(diagonal, diagonal)
        np.fill_diagonal(expected_correlations, 0.0)
        correlations = _normalize_symmetric_gram(accumulated, inverse_scale, block_size=5)
        np.testing.assert_allclose(correlations, expected_correlations, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(correlations, correlations.T)

    def test_network_defaults_come_from_central_defaults(self):
        store = ConfigStore()
        micro_defaults = store.load_configuration("microparcellation", "main").values
        network_defaults = store.load_configuration("networks", "main").values
        self.assertEqual(
            CoarseningConfig().target_vertices,
            micro_defaults["coarsening"]["target_vertices"],
        )
        self.assertEqual(
            CoarseningConfig().iterations,
            micro_defaults["coarsening"]["iterations"],
        )
        self.assertEqual(
            OslomConfig().repetitions,
            network_defaults["oslom"]["repetitions"],
        )

    def test_default_microparcellation_schedule(self):
        config = CoarseningConfig()
        self.assertEqual(config.target_vertices, 40_000)
        self.assertEqual(config.iterations, 3)
        self.assertEqual(
            _coarsening_targets(200_000, config.target_vertices, config.iterations),
            (160_000, 80_000, 40_000),
        )

    def test_synbold_default_comes_from_central_store(self):
        config = ConfigStore().load_configuration("preprocessing", "main").values
        self.assertEqual(
            Path(config["func"]["synbold_disco_image"]),
            Path("/juice6/u/nlp/climblab/apptainer/images/synbold-disco_v1.4.sif"),
        )

    def test_synbold_image_resolution_requires_existing_image(self):
        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "synbold-disco_v1.4.sif"
            with patch(
                "nro.modules.func.synbold_disco.shutil.which", return_value="/usr/bin/singularity"
            ):
                with self.assertRaisesRegex(SystemExit, "Provide it before starting"):
                    ensure_image(image=image, engine="singularity")
                image.write_bytes(b"image")
                self.assertEqual(ensure_image(image=image, engine="singularity"), image)

    def test_bilateral_surface_faces_receive_right_offset(self):
        surfaces = {
            "left": (np.zeros((3, 3)), np.array([[0, 1, 2]])),
            "right": (np.ones((4, 3)), np.array([[0, 2, 3]])),
        }
        with patch(
            "nro.engine.surface_geometry.load_surface", side_effect=lambda path: surfaces[str(path)]
        ):
            coords, faces, counts = load_surfaces((Path("left"), Path("right")))
        self.assertEqual(counts, (3, 4))
        self.assertEqual(coords.shape, (7, 3))
        np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 5, 6]])

    def test_microparcellation_cifti_round_trip(self):
        labels = np.array([0, 0, 1, 1, 2, -1])
        correlations = np.array(
            [[0.0, -0.5, 0.25], [-0.5, 0.0, 1.0], [0.25, 1.0, 0.0]],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dlabel, parcels = write_dlabel(
                root / "microparcels.dlabel.nii",
                labels,
                (3, 3),
                (Path("sub-01_hemi-L_pial.surf.gii"), Path("sub-01_hemi-R_pial.surf.gii")),
            )
            pconn = write_pconn(root / "connectivity.pconn.nii", correlations, parcels)
            loaded_labels, counts = load_dlabel(dlabel)
            loaded_correlations = load_pconn(pconn)
            streamed, percentile = pconn_to_adjacency(
                pconn,
                transform="clip_positive",
                minimum_weight=0.0,
                percentile_cutoff=50.0,
                block_size=2,
            )
            expected_adjacency, expected_percentile = lower_triangular_adjacency(
                np.maximum(loaded_correlations, 0.0),
                minimum_weight=0.0,
                percentile_cutoff=50.0,
            )

            import nibabel as nib

            image = nib.load(str(pconn))
            self.assertEqual(image.get_data_dtype(), np.dtype(np.int8))
            self.assertAlmostEqual(image.dataobj.slope, 1.0 / 127.0)
        np.testing.assert_array_equal(loaded_labels, labels)
        self.assertEqual(counts, (3, 3))
        np.testing.assert_allclose(loaded_correlations, correlations, atol=1.0 / 254.0)
        np.testing.assert_array_equal(np.diag(loaded_correlations), 0.0)
        self.assertAlmostEqual(percentile, expected_percentile)
        np.testing.assert_allclose(streamed.toarray(), expected_adjacency.toarray())

    def test_oslom_resolution_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            configured = root / "configured"
            environment = root / "environment"
            configured.touch()
            environment.touch()
            self.assertEqual(resolve_oslom_executable(configured), configured.resolve())
            with patch("nro.modules.networks.oslom.shutil.which", return_value=str(environment)):
                self.assertEqual(resolve_oslom_executable(None), environment.resolve())

    def test_oslom_resolution_requires_existing_executable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            missing = root / "missing-oslom"
            with patch("nro.modules.networks.oslom.shutil.which", return_value=None):
                with self.assertRaisesRegex(FileNotFoundError, "not found on PATH"):
                    resolve_oslom_executable(None)
                with self.assertRaisesRegex(FileNotFoundError, "Configured OSLOM"):
                    resolve_oslom_executable(missing)

    def test_loukas_exact_count_and_contiguous(self):
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
        sim = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        defaults = CoarseningConfig()
        labels = loukas_variation_edges(
            6,
            edges,
            sim,
            np.ones(6, bool),
            3,
            k=3,
            max_levels=defaults.max_levels,
            eigensolver_tolerance=defaults.eigensolver_tolerance,
        )
        self.assertEqual(len(np.unique(labels)), 3)
        for label in np.unique(labels):
            members = set(np.flatnonzero(labels == label))
            if len(members) > 1:
                self.assertTrue(any(a in members and b in members for a, b in edges))

    def test_coarsening_schedule_uses_target_relative_halving(self):
        self.assertEqual(
            _coarsening_targets(200_000, 20_000, 4),
            (160_000, 80_000, 40_000, 20_000),
        )
        self.assertEqual(_coarsening_targets(100, 10, 3), (40, 20, 10))

    def test_coarsening_schedule_omits_stages_not_below_native_granularity(self):
        self.assertEqual(
            _coarsening_targets(100_000, 20_000, 4),
            (80_000, 40_000, 20_000),
        )
        self.assertEqual(_coarsening_targets(10, 8, 5), (8,))
        self.assertEqual(_coarsening_targets(10, 10, 3), ())
        self.assertEqual(_coarsening_targets(8, 10, 3), ())
        with self.assertRaisesRegex(ValueError, "positive"):
            _coarsening_targets(100, 10, 0)

    def test_region_edges_are_deduplicated_from_original_spatial_graph(self):
        base_edges = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [3, 4], [4, 5]], dtype=np.int64)
        labels = np.array([0, 0, 1, 1, 2, -1], dtype=np.int64)
        np.testing.assert_array_equal(
            _region_edges(base_edges, labels),
            np.array([[0, 1], [1, 2]], dtype=np.int64),
        )

    def test_parcel_mean_loader_recomputes_means_from_original_samples(self):
        data = np.array(
            [[1.0, 3.0, 10.0, 14.0], [2.0, 6.0, 20.0, 24.0]],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        loader, masses = make_parcel_mean_loader(
            labels,
            np.ones(4, dtype=bool),
            load_run=lambda _path: data,
        )
        np.testing.assert_array_equal(masses, [2.0, 2.0])
        np.testing.assert_allclose(
            loader((Path("run"),)),
            np.array([[2.0, 12.0], [4.0, 22.0]], dtype=np.float32),
        )

    def test_vectorized_loukas_cost_matches_matrix_expression(self):
        from scipy import sparse

        w = sparse.csr_matrix(np.array([[0.0, 0.4, 0.2], [0.4, 0.0, 0.7], [0.2, 0.7, 0.0]]))
        degree = np.asarray(w.sum(1)).ravel()
        a = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])
        edges, costs = _variation_edge_costs(w, degree, a)
        p = np.array([[0.5, -0.5], [-0.5, 0.5]])
        expected = []
        for i, j in edges:
            wij = w[i, j]
            local = np.array([[2 * degree[i] - wij, -wij], [-wij, 2 * degree[j] - wij]])
            b = p @ a[[i, j]]
            expected.append(np.linalg.norm(b.T @ local @ b))
        np.testing.assert_allclose(costs, expected, atol=1e-12)

    def test_consensus_overlap_and_homeless(self):
        runs = [[{0, 1}, {2, 3}], [{0, 1}, {2, 3, 4}], [{0, 1, 2}, {2, 3}]]
        stability, homeless, overlap, _ = membership_stability(runs, 6, 0.2)
        self.assertEqual(stability.shape, (6, 2))
        self.assertEqual(homeless[5], 1)
        self.assertAlmostEqual(overlap[2], 1 / 3)

    def test_parse_tp(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tp"
            p.write_text("#module 0 size: 3 bs: 0.1\n0 2 4\n#module 1 size: 2 bs: 0.2\n1 3\n")
            self.assertEqual(parse_tp(p), [{0, 2, 4}, {1, 3}])

    def test_oslom_hint_file_is_passed_to_subprocess(self):
        from nro.modules.networks.config import OslomConfig

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            graph = root / "graph.dat"
            hint = root / "hint.dat"
            graph.write_text("0 1 1\n")
            hint.write_text("0 1\n")
            tp = root / "run" / "graph.dat_oslo_files" / "tp"

            def fake_run_child(_runner, args, *, cwd, **_kwargs):
                command = args[-1]
                self.assertIn("-hint hint.dat", command)
                self.assertIn("-w", command)
                tp.parent.mkdir()
                tp.write_text("#module 0\n0 1\n")

            cfg = OslomConfig(executable=Path("/bin/echo"), initial_partition=hint)
            with patch.object(Runner, "run_child", new=fake_run_child):
                runner = Runner(
                    module_name="Test Module",
                    container=None,
                    binds=(),
                    logger=logging.getLogger("test-oslom"),
                    next_step=iter(range(1, 100)).__next__,
                )
                self.assertEqual(run_oslom(graph, root / "run", cfg, runner=runner), [{0, 1}])

    def test_write_hint_uses_one_community_per_line(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_hint(Path(d) / "hint.dat", [[2, 0], [1]])
            self.assertEqual(path.read_text(), "2 0\n1\n")

    def test_profile_reliability_matches_explicit_split_connectomes(self):
        rng = np.random.default_rng(20260826)
        first = rng.normal(size=(11, 7)).astype(np.float32)
        second = rng.normal(size=(13, 7)).astype(np.float32)
        first = (first - first.mean(axis=0)) / first.std(axis=0, ddof=1)
        second = (second - second.mean(axis=0)) / second.std(axis=0, ddof=1)

        got = _profile_reliability(first, second, vertex_block_size=3)
        first_connectome = np.corrcoef(first, rowvar=False)
        second_connectome = np.corrcoef(second, rowvar=False)
        expected = np.empty(7, dtype=np.float32)
        for vertex in range(7):
            keep = np.arange(7) != vertex
            expected[vertex] = np.clip(
                np.corrcoef(
                    first_connectome[vertex, keep],
                    second_connectome[vertex, keep],
                )[0, 1],
                0.0,
                1.0,
            )
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

        second[:, -1] = 0.0
        got = _profile_reliability(first, second, vertex_block_size=3)
        first_connectome = np.corrcoef(first[:, :-1], rowvar=False)
        second_connectome = np.corrcoef(second[:, :-1], rowvar=False)
        for vertex in range(6):
            keep = np.arange(6) != vertex
            expected[vertex] = np.clip(
                np.corrcoef(
                    first_connectome[vertex, keep],
                    second_connectome[vertex, keep],
                )[0, 1],
                0.0,
                1.0,
            )
        expected[-1] = 0.0
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

    def test_subnormal_quarter_variance_is_treated_as_zero(self):
        data = np.column_stack(
            (
                np.arange(8, dtype=np.float32),
                np.arange(8, dtype=np.float32) * np.float32(1e-20),
            )
        )
        quarters, valid = _quarter_standardized(data, global_signal_regression=False)
        self.assertTrue(all(item[0] and not item[1] for item in valid))
        self.assertTrue(all(np.all(item[:, 1] == 0) for item in quarters))
        self.assertTrue(all(np.all(np.isfinite(item)) for item in quarters))

    def test_local_correlations_use_reliability_weighted_gram(self):
        rng = np.random.default_rng(31)
        runs = {
            "run1": rng.normal(size=(9, 4)).astype(np.float32),
            "run2": rng.normal(size=(15, 4)).astype(np.float32),
        }
        qualities = (
            np.array([1.0, 0.25, 0.7, 0.0], dtype=np.float32),
            np.array([0.2, 0.9, 0.4, 1.0], dtype=np.float32),
        )
        edges = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)

        with (
            patch(
                "nro.modules.microparcellation.statistics.load_functional",
                side_effect=lambda path: runs[str(path[0])],
            ),
            patch(
                "nro.modules.microparcellation.statistics._vertex_reliability",
                side_effect=qualities,
            ),
        ):
            result = local_edge_correlations(
                ((Path("run1"),), (Path("run2"),)),
                edges,
                4,
                3,
                global_signal_regression=False,
                reliability_vertex_block_size=2,
            )
        got = result.correlations
        self.assertEqual(
            result.included_runs,
            ((Path("run1"),), (Path("run2"),)),
        )

        gram = np.zeros((4, 4))
        for data, quality in zip(runs.values(), qualities):
            z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
            weighted = z * np.sqrt(quality)[None, :]
            gram += weighted.T @ weighted
        denominator = np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
        expected = gram / denominator
        np.testing.assert_allclose(got, expected[edges[:, 0], edges[:, 1]], atol=2e-6)

    def test_reliability_weighting_can_be_disabled_in_both_passes(self):
        rng = np.random.default_rng(37)
        runs = {
            "run1": rng.normal(size=(9, 4)).astype(np.float32),
            "run2": rng.normal(size=(13, 4)).astype(np.float32),
        }
        files = tuple((Path(name),) for name in runs)
        edges = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)
        labels = np.array([0, 0, 1, 1])
        mask = np.ones(4, dtype=bool)

        with (
            patch(
                "nro.modules.microparcellation.statistics.load_functional",
                side_effect=lambda path: runs[str(path[0])],
            ),
            patch(
                "nro.modules.microparcellation.statistics._vertex_reliability",
                side_effect=AssertionError("vertex reliability should be skipped"),
            ),
            patch(
                "nro.modules.microparcellation.statistics._parcel_reliability",
                side_effect=AssertionError("parcel reliability should be skipped"),
            ),
        ):
            local = local_edge_correlations(
                files,
                edges,
                4,
                3,
                global_signal_regression=False,
                reliability_vertex_block_size=2,
                reliability_weighting=False,
            ).correlations
            parcels = parcel_correlations(
                files,
                labels,
                mask,
                3,
                split_half_block_frames=128,
                global_signal_regression=False,
                reliability_vertex_block_size=2,
                reliability_weighting=False,
            ).correlations

        vertex_gram = np.zeros((4, 4))
        parcel_gram = np.zeros((2, 2))
        for data in runs.values():
            z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
            vertex_gram += z.T @ z
            parcel_z = np.column_stack((z[:, :2].mean(axis=1), z[:, 2:].mean(axis=1)))
            parcel_z = (parcel_z - parcel_z.mean(axis=0)) / parcel_z.std(axis=0, ddof=1)
            parcel_gram += parcel_z.T @ parcel_z
        vertex_expected = vertex_gram / np.sqrt(
            np.outer(np.diag(vertex_gram), np.diag(vertex_gram))
        )
        parcel_expected = parcel_gram / np.sqrt(
            np.outer(np.diag(parcel_gram), np.diag(parcel_gram))
        )
        np.fill_diagonal(parcel_expected, 0.0)
        np.testing.assert_allclose(local, vertex_expected[edges[:, 0], edges[:, 1]], atol=2e-6)
        np.testing.assert_allclose(parcels, parcel_expected, atol=2e-6)

    def test_parcel_correlations_use_reliability_weighted_gram(self):
        rng = np.random.default_rng(47)
        runs = {
            "run1": rng.normal(size=(10, 6)).astype(np.float32),
            "run2": rng.normal(size=(17, 6)).astype(np.float32),
        }
        qualities = (
            np.array([1.0, 0.3, 0.8], dtype=np.float32),
            np.array([0.2, 1.0, 0.5], dtype=np.float32),
        )
        labels = np.array([0, 0, 1, 1, 2, 2])
        mask = np.ones(6, dtype=bool)

        with (
            patch(
                "nro.modules.microparcellation.statistics.load_functional",
                side_effect=lambda path: runs[str(path[0])],
            ),
            patch(
                "nro.modules.microparcellation.statistics._parcel_reliability",
                side_effect=qualities,
            ),
        ):
            result = parcel_correlations(
                ((Path("run1"),), (Path("run2"),)),
                labels,
                mask,
                4,
                split_half_block_frames=128,
                global_signal_regression=False,
                reliability_vertex_block_size=2,
            )
            got = result.correlations

        gram = np.zeros((3, 3))
        for data, quality in zip(runs.values(), qualities):
            z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
            parcels = np.column_stack([z[:, labels == parcel].mean(axis=1) for parcel in range(3)])
            parcels = (parcels - parcels.mean(axis=0)) / parcels.std(axis=0, ddof=1)
            weighted = parcels * np.sqrt(quality)[None, :]
            gram += weighted.T @ weighted
        denominator = np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
        expected = gram / denominator
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(got, expected, atol=2e-6)
        np.testing.assert_allclose(
            result.parcel_reliability_mean,
            np.mean(qualities, axis=0),
        )
        np.testing.assert_allclose(
            result.parcel_effective_runs,
            np.square(np.sum(qualities, axis=0)) / np.sum(np.square(qualities), axis=0),
            atol=1e-6,
        )
        np.testing.assert_array_equal(result.parcel_supporting_runs, [2, 2, 2])
        self.assertEqual(result.split_half["method"], "whole runs")
        self.assertEqual(result.split_half["first_half_runs"], [1])
        self.assertEqual(result.split_half["second_half_runs"], [2])
        self.assertAlmostEqual(
            sum(item["diagonal_weight_fraction"] for item in result.run_contributions),
            1.0,
        )
        self.assertEqual(result.connectome["unique_edges"], 3)
        self.assertIn("participation_ratio_rank", result.connectome)
        self.assertIn("dominant_eigenvalue_fraction", result.connectome)

    def test_parcel_quality_scores_variance_preserved_and_spatial_null(self):
        time = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        data = np.column_stack((time, time, -time, -time, time**2, time**2)).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 2, 2])
        with patch(
            "nro.modules.microparcellation.statistics.load_functional",
            return_value=data,
        ):
            result = parcel_correlations(
                ((Path("run"),),),
                labels,
                np.ones(6, dtype=bool),
                4,
                split_half_block_frames=128,
                global_signal_regression=False,
                reliability_vertex_block_size=2,
                reliability_weighting=False,
                null_partitions=(np.array([0, 1, 0, 2, 1, 2]),),
            )

        self.assertAlmostEqual(result.variance_preserved, 1.0, places=6)
        self.assertEqual(len(result.null_variance_preserved), 1)
        self.assertLess(result.null_variance_preserved[0], result.variance_preserved)
        self.assertEqual(result.split_half["method"], "alternating temporal blocks")
        self.assertGreater(result.split_half["first_half_retained_frames"], 0)
        self.assertGreater(result.split_half["second_half_retained_frames"], 0)

    def test_spatial_nulls_are_connected_and_retain_fitted_granularity(self):
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        edges = np.column_stack((np.arange(8), np.arange(1, 9)))
        nulls = spatial_null_partitions(
            labels,
            np.ones(9, dtype=bool),
            edges,
            count=5,
            seed=11,
            candidate_attempts=4,
        )

        self.assertEqual(len(nulls), 5)
        for null in nulls:
            self.assertEqual(set(null.labels), {0, 1, 2})
            for parcel in range(3):
                members = np.flatnonzero(null.labels == parcel)
                self.assertTrue(np.all(np.diff(members) == 1))
            self.assertGreaterEqual(null.exactly_matched_fraction, 0.0)
            self.assertLessEqual(null.exactly_matched_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
