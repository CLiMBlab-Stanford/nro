from __future__ import annotations

import logging
from itertools import count
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from nro.configuration.runtime import ConfigNode, configure

configure(
    {
        "common": {"qunex_container": "/tmp/qunex.sif"},
        "get_confounds": {
            "aseg_in_epi": None,
            "n_acompcor": 10,
            "acompcor_max_voxels": 20000,
            "fd_radius_mm": 50.0,
            "motion_outlier_fd_thresh": 1.0,
            "dvars_statistical_alpha": 0.05,
            "dvars_practical_threshold_percent": 5.0,
            "dvars_power": 1.0 / 3.0,
            "nonsteady_max_vols": 20,
            "nonsteady_rel_thresh": 0.05,
            "nonsteady_stable_run": 3,
        },
    }
)

from nro.modules.func import confounds as get_confounds_module
from nro.modules.func import steps as func_steps
from nro.orchestration.runner import Runner


@pytest.mark.parametrize("chunk", [1, 2, 7, 128])
def test_temporal_mean_chunk_size_does_not_amplify_cancellation(tmp_path, chunk):
    data = np.tile(np.array([1e8, 1, -1e8], dtype=np.float32), 43).reshape(1, 1, 1, -1)
    source, output = tmp_path / "bold.nii.gz", tmp_path / "mean.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    step = func_steps._create_temporal_mean_step(
        in_4d=source, out_3d=output, env={}, force=False, chunk_vols=chunk
    )
    step.action()
    np.testing.assert_array_equal(
        np.asarray(nib.load(output).dataobj), data.mean(axis=3, dtype=np.float64).astype(np.float32)
    )


class _NoSliceProxy:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.array_calls = 0

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        self.array_calls += 1
        result = np.asarray(self.data, dtype=dtype)
        return result.copy() if copy else result

    def __getitem__(self, _item):
        raise AssertionError("Compressed temporal proxy was sliced instead of loaded once")


class _ProxyImage:
    def __init__(self, data: np.ndarray) -> None:
        self.shape = data.shape
        self.affine = np.eye(4)
        self.header = nib.Nifti1Header()
        self.header.set_data_shape(data.shape)
        self.header.set_data_dtype(data.dtype)
        self.dataobj = _NoSliceProxy(data)


def test_temporal_mean_materializes_proxy_once_before_block_reduction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = np.arange(4 * 3 * 2 * 7, dtype=np.float32).reshape((4, 3, 2, 7))
    image = _ProxyImage(data)
    real_load = nib.load
    monkeypatch.setattr(nib, "load", lambda _path: image)
    runner = Runner(
        module_name="Temporal Mean Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.timecourse.tmean"),
        next_step=count(1).__next__,
    )
    output = tmp_path / "mean.nii.gz"

    runner.add_step(
        func_steps._create_temporal_mean_step(
            in_4d=tmp_path / "input.nii.gz",
            out_3d=output,
            env={},
            force=False,
            chunk_vols=2,
        )
    )
    with runner.run_context():
        runner.execute()

    assert image.dataobj.array_calls == 1
    expected = np.zeros(data.shape[:3], dtype=np.float32)
    for start in range(0, data.shape[3], 2):
        expected += data[..., start : start + 2].sum(axis=3, dtype=np.float32)
    expected /= float(data.shape[3])
    np.testing.assert_array_equal(np.asarray(real_load(output).dataobj), expected)


def test_confounds_loads_epi_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rng = np.random.default_rng(7)
    monkeypatch.setattr(
        get_confounds_module,
        "SETTINGS",
        ConfigNode(
            {
                "get_confounds": {
                    "aseg_in_epi": None,
                    "n_acompcor": 10,
                    "acompcor_max_voxels": 20000,
                    "fd_radius_mm": 50.0,
                    "motion_outlier_fd_thresh": 1.0,
                    "dvars_statistical_alpha": 0.05,
                    "dvars_practical_threshold_percent": 5.0,
                    "dvars_power": 1.0 / 3.0,
                    "nonsteady_max_vols": 20,
                    "nonsteady_rel_thresh": 0.05,
                    "nonsteady_stable_run": 3,
                }
            }
        ),
    )
    epi_data = rng.normal(100.0, 3.0, size=(8, 8, 4, 12)).astype(np.float32)
    epi_image = _ProxyImage(epi_data)
    mean_image = nib.Nifti1Image(epi_data.mean(axis=3), np.eye(4))
    aseg_data = np.full(epi_data.shape[:3], 2, dtype=np.int16)
    aseg_data[:4, :, :] = 4
    aseg_image = nib.Nifti1Image(aseg_data, np.eye(4))
    mask_image = nib.Nifti1Image(np.ones(epi_data.shape[:3], dtype=np.uint8), np.eye(4))
    epi_path = tmp_path / "bold.nii.gz"
    mean_path = tmp_path / "mean.nii.gz"
    aseg_path = tmp_path / "aseg.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    images = {
        epi_path: epi_image,
        mean_path: mean_image,
        aseg_path: aseg_image,
        mask_path: mask_image,
    }
    monkeypatch.setattr(
        get_confounds_module,
        "_load_niimg",
        lambda path: images[Path(path)],
    )
    motion = tmp_path / "motion.par"
    np.savetxt(motion, np.zeros((epi_data.shape[3], 6), dtype=np.float64))
    out_tsv = tmp_path / "confounds.tsv"
    out_json = tmp_path / "confounds.json"

    get_confounds_module.get_confounds(
        epi=epi_path,
        epi_mean=mean_path,
        mcflirt_par=motion,
        subjects_dir=tmp_path,
        fs_subject="subject",
        aseg_in_epi=aseg_path,
        brain_mask_in_epi=mask_path,
        out_tsv=out_tsv,
        out_json=out_json,
        n_acompcor=2,
        acompcor_max_voxels=200,
    )

    assert epi_image.dataobj.array_calls == 1
    assert out_tsv.stat().st_size > 0
    assert out_json.stat().st_size > 0

    confounds = pd.read_csv(out_tsv, sep="\t")
    from nro.configuration.store import ConfigStore

    clean_config_path = ConfigStore().configuration_path("clean", "main")
    clean_config = yaml.safe_load(clean_config_path.read_text(encoding="utf-8"))
    selected = confounds.filter(regex=str(clean_config["confounds_regex"]))

    bases = [f"trans_{axis}" for axis in "xyz"]
    bases += [f"rot_{axis}" for axis in "xyz"]
    bases += ["white_matter", "csf", "global_signal"]
    suffixes = ["", "_derivative1", "_power2", "_derivative1_power2"]
    expected = {f"{base}{suffix}" for base in bases for suffix in suffixes}
    assert set(selected.columns) == expected
    assert selected.shape[1] == 36
    assert not any("outlier" in column for column in selected.columns)
    assert "framewise_displacement" not in selected.columns
    assert not any(column.startswith("a_comp_cor") for column in selected.columns)
    assert {"dvars", "dvars_p_value", "dvars_delta_percent"}.issubset(confounds.columns)


def test_dvars_marks_both_frames_around_an_anomalous_difference() -> None:
    rng = np.random.default_rng(19)
    data = rng.normal(size=(10, 10, 4, 40)).astype(np.float32)
    data[..., 20] += 50.0

    metrics = get_confounds_module._dvars_metrics(
        data,
        np.ones(data.shape[:3], dtype=bool),
        statistical_alpha=0.05,
        practical_threshold_percent=5.0,
        power=1.0 / 3.0,
    )

    assert metrics["outlier_indices"] == [19, 20, 21]
    assert np.isnan(metrics["dvars"][0])
    assert np.isfinite(metrics["dvars"][1:]).all()
