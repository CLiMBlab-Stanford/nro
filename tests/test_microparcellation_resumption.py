import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.images import sidecar_json_path
from nro.modules.microparcellation.config import (
    CoarseningConfig,
    ConnectivityConfig,
    InputsConfig,
    ModuleConfig,
    OutputConfig,
)
from nro.modules.microparcellation.module import run


def _write_surface(path: Path) -> None:
    image = nib.gifti.GiftiImage()
    image.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                dtype=np.float32,
            ),
            intent="NIFTI_INTENT_POINTSET",
        )
    )
    image.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
        )
    )
    nib.save(image, path)


def _write_functional(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = nib.gifti.GiftiImage()
    for values in rng.normal(size=(12, 4)).astype(np.float32):
        image.add_gifti_data_array(nib.gifti.GiftiDataArray(values))
    nib.save(image, path)
    mask_file = path.parent / "sub-test_desc-confounds_timeseries.tsv"
    mask_file.write_text("motion_outlier00\n" + "0\n" * 12)
    sidecar_json_path(path).write_text(
        json.dumps(
            {
                "Cleaning": {
                    "CleaningDefined": True,
                    "TotalFrames": 12,
                    "RetainedFrames": 12,
                    "CensoredFraction": 0.0,
                    "ResidualDesignDegreesOfFreedom": 10,
                    "AlgebraicTemporalRank": 10,
                    "TemporalMaskFile": str(mask_file),
                    "TemporalMaskRegex": ".*outlier.*",
                    "QualityControl": {
                        "ParticipationRatioEffectiveTemporalRank": 8.0,
                        "DominantTemporalVarianceFraction": 0.2,
                    },
                }
            }
        )
    )


def test_surface_module_uses_runner_and_skips_all_current_stages(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    surfaces = []
    functionals = []
    for index, hemi in enumerate(("L", "R"), start=1):
        surface = tmp_path / f"sub-test_hemi-{hemi}_midthickness.surf.gii"
        functional = tmp_path / f"sub-test_hemi-{hemi}_desc-clean_bold.func.gii"
        for surface_type in ("pial", "midthickness", "white", "inflated"):
            _write_surface(tmp_path / f"sub-test_hemi-{hemi}_{surface_type}.surf.gii")
        _write_functional(functional, index)
        surfaces.append(surface)
        functionals.append(functional)
    cfg = ModuleConfig(
        inputs=InputsConfig(
            functional=(tuple(functionals),),
            temporal_masks=(tmp_path / "sub-test_desc-confounds_timeseries.tsv",),
            domain="surface",
            surface=tuple(surfaces),
        ),
        output=OutputConfig(
            directory=tmp_path / "output",
            work_directory=tmp_path / "work",
            prefix="sub-test",
        ),
        coarsening=CoarseningConfig(
            target_vertices=4,
            iterations=2,
            exponential_temperature=0.1,
            eigenvectors=2,
            max_levels=10,
            eigensolver_tolerance=1e-5,
        ),
        connectivity=ConnectivityConfig(
            minimum_retained_frames=4,
            minimum_retained_fraction=0.0,
            minimum_residual_design_dof=0,
            minimum_participation_effective_rank=0.0,
            maximum_dominant_temporal_variance_fraction=1.0,
            minimum_usable_runs=1,
            minimum_aggregate_retained_frames=4,
            temporal_block_size=4,
            weighting="equal",
            global_signal_regression=False,
        ),
    )

    with caplog.at_level(logging.INFO, logger="nro.modules.microparcellation.module"):
        outputs = run(cfg)
    assert "borders" not in outputs
    mtimes = {
        path: path.stat().st_mtime_ns
        for value in outputs.values()
        for path in ((value,) if isinstance(value, Path) else value)
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a current step unexpectedly recomputed")

    monkeypatch.setattr("nro.modules.microparcellation.module.load_surfaces", forbidden)
    monkeypatch.setattr("nro.modules.microparcellation.module.local_edge_correlations", forbidden)
    monkeypatch.setattr("nro.modules.microparcellation.module.parcel_correlations", forbidden)
    shutil.rmtree(cfg.output.work_directory)
    cfg = replace(
        cfg,
        connectivity=replace(cfg.connectivity, temporal_block_size=6),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nro.modules.microparcellation.module"):
        resumed = run(cfg)

    assert all(path.stat().st_mtime_ns == mtime for path, mtime in mtimes.items())
    assert resumed.keys() == outputs.keys()
    assert "Skipping microparcellation module" in caplog.text
    assert not cfg.output.work_directory.exists()
    assert "Name: Microparcellation Module" in caplog.text
    assert "Status: Success" in caplog.text
    assert "Total Time Elapsed:" in caplog.text
