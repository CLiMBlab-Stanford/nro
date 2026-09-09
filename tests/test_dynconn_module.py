from __future__ import annotations

import json
import logging
from itertools import count
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from nibabel.gifti import GiftiDataArray, GiftiImage

from nro.engine.images import sidecar_json_path
from nro.modules.dynconn.config import InclusionConfig, InputsConfig, ModuleConfig, OutputConfig
from nro.modules.dynconn.module import build_module
from nro.orchestration.runner import Runner


def _sidecar(path: Path, mask: Path, *, frames: int, retained: int, tr: float = 2.0) -> None:
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
                    "AlgebraicTemporalRank": retained,
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


def test_surface_module_concatenates_retained_frames_and_packages_scene(tmp_path: Path) -> None:
    surfaces = []
    for hemi in ("L", "R"):
        surfaces.append(tmp_path / f"sub-01_hemi-{hemi}_pial.surf.gii")
        for kind in ("pial", "midthickness", "white", "inflated"):
            image = GiftiImage(
                darrays=[
                    GiftiDataArray(
                        np.zeros((3, 3), dtype=np.float32), intent="NIFTI_INTENT_POINTSET"
                    ),
                    GiftiDataArray(
                        np.array([[0, 1, 2]], dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE"
                    ),
                ]
            )
            nib.save(image, tmp_path / f"sub-01_hemi-{hemi}_{kind}.surf.gii")
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
            _sidecar(path, mask, frames=4, retained=3)
            pair.append(path)
        runs.append(tuple(pair))
    cfg = ModuleConfig(
        InputsConfig(tuple(runs), tuple(masks), "surface", "fsnative", 2, tuple(surfaces)),
        OutputConfig(
            tmp_path / "out", tmp_path / "work", "sub-01_space-fsnative_smoothing-2mm", False
        ),
        _inclusion(),
    )
    result = _execute(cfg)
    image = nib.load(str(result["timeseries"]))
    assert image.shape == (6, 6)
    assert image.header.get_axis(0).step == 2.0
    assert len(result["surfaces"]) == 8
    scene = result["scene"].read_text()
    assert "dynamicConnectivity_bold.dynconn.nii" in scene
    assert str(tmp_path) not in scene
    manifest = yaml.safe_load(result["manifest"].read_text())
    assert [record["start_frame"] for record in manifest["functional_runs"]["included"]] == [0, 3]


def test_volume_module_writes_uncompressed_four_dimensional_nifti(tmp_path: Path) -> None:
    runs, masks = [], []
    for run_index in range(2):
        mask = tmp_path / f"run-{run_index}_desc-confounds_timeseries.tsv"
        mask.write_text("motion_outlier00\n0\n1\n0\n0\n")
        path = tmp_path / f"run-{run_index}_desc-clean_bold.nii.gz"
        nib.save(
            nib.Nifti1Image(np.full((2, 2, 2, 4), run_index, dtype=np.float32), np.eye(4)), path
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
    )
    result = _execute(cfg)
    image = nib.load(str(result["timeseries"]))
    assert result["timeseries"].suffix == ".nii"
    assert image.shape == (2, 2, 2, 6)
    assert image.header.get_zooms()[3] == 2.0
    assert "vol_dynconn" in result["scene"].read_text()
