from __future__ import annotations

import json
import logging
from dataclasses import replace
from itertools import count
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from nibabel.gifti import GiftiDataArray, GiftiImage

from nro.engine.images import sidecar_json_path
from nro.modules.dynconn.config import (
    InclusionConfig,
    InputsConfig,
    LowRankConfig,
    ModuleConfig,
    OutputConfig,
)
from nro.modules.dynconn.low_rank import fit_low_rank_correlation, pseudo_timeseries_block
from nro.modules.dynconn.module import build_module
from nro.orchestration.runner import Runner


def _sidecar(
    path: Path,
    mask: Path,
    *,
    frames: int,
    retained: int,
    tr: float = 2.0,
    algebraic_rank: int | None = None,
) -> None:
    sidecar_json_path(path).write_text(
        json.dumps(
            {
                "RepetitionTime": tr,
                "Cleaning": {
                    "CleaningDefined": True,
                    "TotalFrames": frames,
                    "RetainedFrames": retained,
                    "CensoredFraction": 1.0 - retained / frames,
                    "ResidualDesignDegreesOfFreedom": retained,
                    "AlgebraicTemporalRank": (
                        retained if algebraic_rank is None else algebraic_rank
                    ),
                    "TemporalMaskFile": str(mask),
                    "TemporalMaskRegex": ".*outlier.*",
                    "QualityControl": {
                        "ParticipationRatioEffectiveTemporalRank": float(retained),
                        "DominantTemporalVarianceFraction": 0.4,
                    },
                },
            }
        )
    )


def _inclusion() -> InclusionConfig:
    return InclusionConfig(2, 0.5, 0, 0.0, 1.0, 2, 4)


def _execute(cfg: ModuleConfig):
    runner = Runner(
        module_name="test",
        container=None,
        binds=(),
        logger=logging.getLogger("dynconn-test"),
        next_step=count(1).__next__,
    )
    result = build_module(cfg, runner)
    with runner.run_context():
        runner.execute()
    return result


def test_surface_module_concatenates_retained_frames(tmp_path: Path) -> None:
    runs, masks = [], []
    for run_index in range(2):
        mask = tmp_path / f"run-{run_index}_desc-confounds_timeseries.tsv"
        mask.write_text("motion_outlier00\n0\n1\n0\n0\n")
        masks.append(mask)
        pair = []
        for hemi in ("L", "R"):
            path = tmp_path / f"run-{run_index}_hemi-{hemi}_desc-clean_bold.func.gii"
            nib.save(
                GiftiImage(
                    darrays=[
                        GiftiDataArray(np.full(3, run_index * 10 + frame, dtype=np.float32))
                        for frame in range(4)
                    ]
                ),
                path,
            )
            _sidecar(path, mask, frames=4, retained=3, algebraic_rank=run_index + 1)
            pair.append(path)
        runs.append(tuple(pair))
    cfg = ModuleConfig(
        InputsConfig(tuple(runs), tuple(masks), "surface", "fsnative", 2),
        OutputConfig(
            tmp_path / "out", tmp_path / "work", "sub-01_space-fsnative_smoothing-2mm", False
        ),
        _inclusion(),
        low_rank=False,
    )
    result = _execute(cfg)
    image = nib.load(str(result["timeseries"]))
    assert image.shape == (6, 6)
    assert image.header.get_axis(0).step == 2.0
    values = np.asarray(image.dataobj)
    np.testing.assert_allclose(values[:3].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(values[3:].mean(axis=0), 0.0, atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(values[:3], axis=0), np.sqrt(1 / 3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(values[3:], axis=0), np.sqrt(2 / 3), atol=1e-6)
    manifest = yaml.safe_load(result["manifest"].read_text())
    assert [record["start_frame"] for record in manifest["functional_runs"]["included"]] == [0, 3]
    assert manifest["representation"] == "full"
    assert manifest["weighting"] == "precision"
    assert manifest["config"]["weighting"] == "precision"
    assert [record["normalized_weight"] for record in manifest["functional_runs"]["included"]] == [
        1 / 3,
        2 / 3,
    ]
    assert manifest["published_frames"] == 6
    assert manifest["low_rank"] is None

    compressed = _execute(
        replace(
            cfg,
            output=replace(
                cfg.output,
                directory=tmp_path / "low-rank-out",
                work_directory=tmp_path / "low-rank-work",
            ),
            low_rank=True,
            low_rank_options=LowRankConfig(2, 2, 1),
        )
    )
    compressed_image = nib.load(str(compressed["timeseries"]))
    assert compressed_image.shape == (3, 6)
    assert compressed_image.header.get_axis(0).step == 1.0
    compressed_manifest = yaml.safe_load(compressed["manifest"].read_text())
    assert compressed_manifest["representation"] == "low_rank"
    assert compressed_manifest["concatenated_frames"] == 6
    assert compressed_manifest["published_frames"] == 3
    assert compressed_manifest["low_rank"]["dimensions"] == 2
    assert compressed_manifest["low_rank"]["requested_dimensions"] == 2
    assert compressed_manifest["low_rank"]["synthetic_frames"] == 3
    assert isinstance(compressed_manifest["low_rank"]["random_seed"], int)


def test_volume_module_writes_uncompressed_four_dimensional_nifti(tmp_path: Path) -> None:
    runs, masks = [], []
    random = np.random.default_rng(9)
    for run_index in range(2):
        mask = tmp_path / f"run-{run_index}_desc-confounds_timeseries.tsv"
        mask.write_text("motion_outlier00\n0\n1\n0\n0\n")
        path = tmp_path / f"run-{run_index}_desc-clean_bold.nii.gz"
        nib.save(
            nib.Nifti1Image(random.normal(size=(2, 2, 2, 4)).astype(np.float32), np.eye(4)),
            path,
        )
        _sidecar(path, mask, frames=4, retained=3)
        runs.append((path,))
        masks.append(mask)
    cfg = ModuleConfig(
        InputsConfig(tuple(runs), tuple(masks), "volume", "MNI152NLin6Asym", 2),
        OutputConfig(
            tmp_path / "out", tmp_path / "work", "sub-01_space-MNI152NLin6Asym_smoothing-2mm", False
        ),
        _inclusion(),
        low_rank=False,
    )
    result = _execute(cfg)
    image = nib.load(str(result["timeseries"]))
    assert result["timeseries"].suffix == ".nii"
    assert image.shape == (2, 2, 2, 6)
    assert image.header.get_zooms()[3] == 2.0

    compressed = _execute(
        replace(
            cfg,
            output=replace(
                cfg.output,
                directory=tmp_path / "low-rank-out",
                work_directory=tmp_path / "low-rank-work",
            ),
            low_rank=True,
            low_rank_options=LowRankConfig(2, 2, 0),
        )
    )
    compressed_image = nib.load(str(compressed["timeseries"]))
    assert compressed_image.shape == (2, 2, 2, 3)
    assert compressed_image.header.get_zooms()[3] == 1.0


def test_pseudo_timeseries_reconstitutes_truncated_correlation() -> None:
    random = np.random.default_rng(12)
    data = random.normal(size=(8, 5)).astype(np.float32)

    def blocks():
        yield data[:3]
        yield data[3:]

    fit = fit_low_rank_correlation(
        blocks,
        n_locations=5,
        dimensions=2,
        oversampling=3,
        power_iterations=0,
        random_seed=4,
    )
    pseudo = pseudo_timeseries_block(fit, 0, 5)
    standardized = data - data.mean(axis=0)
    standardized /= np.sqrt(np.sum(standardized * standardized, axis=0))
    left, singular, _ = np.linalg.svd(standardized.T, full_matrices=False)
    expected_factors = left[:, :2] * singular[:2]
    expected_covariance = expected_factors @ expected_factors.T
    expected_scale = np.sqrt(np.diag(expected_covariance))
    expected_correlation = expected_covariance / np.outer(expected_scale, expected_scale)

    np.testing.assert_allclose(
        np.cov(pseudo, rowvar=False),
        expected_covariance,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.corrcoef(pseudo, rowvar=False),
        expected_correlation,
        atol=2e-5,
    )


def test_system_seed_is_recorded_and_replays_the_fit() -> None:
    random = np.random.default_rng(12)
    data = random.normal(size=(8, 5)).astype(np.float32)

    def blocks():
        yield data

    fit = fit_low_rank_correlation(
        blocks,
        n_locations=5,
        dimensions=10,
        oversampling=2,
        power_iterations=0,
    )
    replay = fit_low_rank_correlation(
        blocks,
        n_locations=5,
        dimensions=10,
        oversampling=2,
        power_iterations=0,
        random_seed=fit.random_seed,
    )

    assert fit.requested_dimensions == 10
    assert fit.dimensions == 5
    np.testing.assert_allclose(
        pseudo_timeseries_block(fit, 0, 5),
        pseudo_timeseries_block(replay, 0, 5),
        atol=1e-6,
    )
