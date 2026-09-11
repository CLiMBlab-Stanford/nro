import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from nro.configuration.store import ConfigStore
from nro.engine.cifti import load_dlabel, load_pconn
from nro.engine.images import sidecar_json_path
from nro.modules.microparcellation.cifti import (
    write_volume_dlabel,
)
from nro.modules.microparcellation.config import (
    CoarseningConfig,
    ConnectivityConfig,
    InputsConfig,
    ModuleConfig,
    OutputConfig,
)
from nro.modules.microparcellation.module import run
from nro.modules.microparcellation.volume import (
    VolumeSpace,
    load_volume_functional,
    load_volume_space,
    volume_edges,
)


def _write_volume_run(
    path: Path,
    data: np.ndarray,
    *,
    cleaning_defined: bool = True,
    undefined_reason: str | None = None,
) -> None:
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)
    frames = int(data.shape[3])
    mask_file = path.with_name(path.name.removesuffix(".nii.gz") + "_confounds.tsv")
    mask_file.write_text("motion_outlier00\n" + "0\n" * frames)
    sidecar_json_path(path).write_text(
        json.dumps(
            {
                "Cleaning": {
                    "CleaningDefined": cleaning_defined,
                    "CleaningUndefinedReason": undefined_reason,
                    "TotalFrames": frames,
                    "RetainedFrames": frames,
                    "CensoredFraction": 0.0,
                    "ResidualDesignDegreesOfFreedom": max(0, frames - 2),
                    "AlgebraicTemporalRank": max(0, frames - 2),
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


def test_oblique_volume_cifti_uses_plumb_grid_without_losing_parcels(tmp_path: Path) -> None:
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:5, 1:5, 1:5] = True
    labels = np.arange(int(mask.sum()), dtype=np.int64)
    angle = np.deg2rad(12.0)
    affine = np.array(
        [
            [2.0 * np.cos(angle), -2.0 * np.sin(angle), 0.0, -8.0],
            [2.0 * np.sin(angle), 2.0 * np.cos(angle), 0.0, -9.0],
            [0.0, 0.0, 2.2, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = tmp_path / "oblique.dlabel.nii"

    _dlabel, parcel_axis = write_volume_dlabel(path, labels, mask, affine)

    loaded, counts = load_dlabel(path)
    brain_axis = nib.load(path).header.get_axis(1)
    linear = brain_axis.affine[:3, :3]
    dominant_rows = np.argmax(np.abs(linear), axis=0)
    off_axis = linear.copy()
    off_axis[dominant_rows, np.arange(3)] = 0.0
    assert counts == (len(loaded),)
    assert np.max(np.abs(off_axis)) < 1e-6
    assert set(loaded) == set(labels)
    assert {index for index, voxels in enumerate(parcel_axis.voxels) if len(voxels)} == set(labels)


def test_six_neighbor_graph_is_restricted_to_mask() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[:2, :2, :2] = True
    indices, edges = volume_edges(mask, connectivity=6)
    assert len(indices) == 8
    assert len(edges) == 12
    assert edges.min() >= 0
    assert edges.max() < 8


def test_default_microparcellation_config_uses_six_volume_neighbors() -> None:
    config = ConfigStore().load_configuration("microparcellation", "main").values
    assert config["volume_connectivity"] == 6
    assert InputsConfig(functional=((Path("run.nii.gz"),),)).volume_connectivity == 6


def test_volume_loader_returns_only_gray_matter_voxels(tmp_path: Path) -> None:
    run = tmp_path / "run.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    data = np.arange(3 * 3 * 3 * 4, dtype=np.float32).reshape(3, 3, 3, 4)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1:, 1:, 1:] = 1
    _write_volume_run(run, data)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

    space = load_volume_space(run, mask_path, threshold=0.5, connectivity=6)
    loaded = load_volume_functional((run,), space)

    assert loaded.shape == (4, 8)
    np.testing.assert_array_equal(loaded, data[mask.astype(bool)].T)


def test_volume_loader_materializes_4d_proxy_once(monkeypatch, tmp_path: Path) -> None:
    data = np.arange(2 * 2 * 2 * 4, dtype=np.float32).reshape(2, 2, 2, 4)

    class CountingProxy:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.array_calls = 0
            self.slice_calls = 0

        def __array__(self, dtype=None, copy=None):
            self.array_calls += 1
            return np.asarray(self.values, dtype=dtype)

        def __getitem__(self, key):
            self.slice_calls += 1
            return self.values[key]

    proxy = CountingProxy(data)

    class FakeImage:
        shape = data.shape
        affine = np.eye(4)
        dataobj = proxy

    monkeypatch.setattr("nro.modules.microparcellation.volume.nib.load", lambda _path: FakeImage())
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    voxel_indices = np.argwhere(mask).astype(np.int64)
    space = VolumeSpace(
        shape=(2, 2, 2),
        affine=np.eye(4),
        mask=mask,
        voxel_indices=voxel_indices,
        edges=np.array([[0, 1]], dtype=np.int64),
        mask_resampled=False,
    )

    loaded = load_volume_functional((tmp_path / "run.nii.gz",), space)

    np.testing.assert_array_equal(loaded, data[mask].T)
    assert proxy.array_calls == 1
    assert proxy.slice_calls == 0


def test_volumetric_module_writes_gray_matter_labels_and_connectivity(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(71)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask_path = tmp_path / "gray-matter.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    runs = []
    for index in range(3):
        path = tmp_path / f"run-{index + 1}.nii.gz"
        data = (
            np.zeros((4, 4, 4, 12), dtype=np.float32)
            if index == 2
            else rng.normal(size=(4, 4, 4, 12))
        )
        _write_volume_run(
            path,
            data,
            cleaning_defined=index != 2,
            undefined_reason=(
                "passband_basis_not_identifiable_from_retained_frames" if index == 2 else None
            ),
        )
        runs.append((path,))

    output = tmp_path / "output"
    cfg = ModuleConfig(
        inputs=InputsConfig(
            functional=tuple(runs),
            temporal_masks=tuple(
                path[0].with_name(path[0].name.removesuffix(".nii.gz") + "_confounds.tsv")
                for path in runs
            ),
            domain="volume",
            mask=mask_path,
            mask_threshold=0.5,
            volume_connectivity=6,
        ),
        output=OutputConfig(directory=output, work_directory=tmp_path / "work", prefix="sub-test"),
        coarsening=CoarseningConfig(
            target_vertices=3,
            iterations=3,
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
    outputs = run(cfg)

    assert outputs["microparcels"].name.endswith("_desc-microparcellation_dseg.dlabel.nii")
    assert outputs["connectivity"].name.endswith("_connectivity.pconn.nii")
    label_volume = np.asarray(nib.load(outputs["microparcels_volume"]).dataobj)
    label_image = nib.load(outputs["microparcels_volume"])
    assert np.all(label_volume[mask == 0] == 0)
    assert np.all(label_volume[mask > 0] > 0)
    assert set(np.unique(label_volume[mask > 0])) == {1, 2, 3}
    assert label_image.header.extensions[0].get_code() == 30
    label_extension = label_image.header.extensions[0].content.decode("utf-8")
    assert "<VolumeType><![CDATA[Label]]></VolumeType>" in label_extension
    assert "microparcel_00001" in label_extension
    compact_labels, counts = load_dlabel(outputs["microparcels"])
    assert counts == (8,)
    assert set(compact_labels) == {0, 1, 2}
    dlabel_image = nib.load(outputs["microparcels"])
    brain_axis = dlabel_image.header.get_axis(1)
    assert set(brain_axis.name) == {"CIFTI_STRUCTURE_ALL_GREY_MATTER"}
    connectivity = load_pconn(outputs["connectivity"])
    assert connectivity.shape == (3, 3)
    pconn_image = nib.load(outputs["connectivity"])
    parcel_axis = pconn_image.header.get_axis(0)
    assert sum(len(voxels) for voxels in parcel_axis.voxels) == 8
    manifest = yaml.safe_load(outputs["manifest"].read_text())
    assert manifest["domain"] == "volume"
    assert manifest["n_gray_matter_voxels"] == 8
    assert [step["target_regions"] for step in manifest["coarsening_steps"]] == [6, 3]
    assert [step["input_regions"] for step in manifest["coarsening_steps"]] == [8, 6]
    quality = manifest["quality"]
    assert quality["included_runs"] == 2
    assert manifest["functional_runs"]["skipped"] == [
        {
            "files": [str(runs[2][0])],
            "reasons": [
                {
                    "reason": "cleaning_undefined",
                    "cleaning_undefined_reasons": [
                        "passband_basis_not_identifiable_from_retained_frames"
                    ],
                }
            ],
        }
    ]
    assert 0.0 <= quality["variance_preserved"] <= 1.0
    assert quality["null_baseline"]["seed"] == 1
    assert len(quality["null_baseline"]["parcellations"]) == 5
    assert outputs["quality"].is_file()
    assert quality["weighting"] == "equal"
    assert quality["parcel_support"]["minimum_supporting_runs"] == 2
    assert len(quality["run_contributions"]) == 2
    assert quality["split_half"]["method"] == "whole runs"
    assert quality["connectome"]["unique_edges"] == 3
    assert len(quality["connectome"]["encoded_histogram"]["counts"]) == 255


def _small_resumable_volume_config(tmp_path: Path) -> tuple[ModuleConfig, Path]:
    rng = np.random.default_rng(91)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask_path = tmp_path / "gray-matter.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    run = tmp_path / "run.nii.gz"
    _write_volume_run(run, rng.normal(size=(4, 4, 4, 12)))
    output = tmp_path / "output"
    output.mkdir()
    (output / "unrelated-user-file.txt").write_text("preserve me")
    return (
        ModuleConfig(
            inputs=InputsConfig(
                functional=((run,),),
                temporal_masks=(
                    run.with_name(run.name.removesuffix(".nii.gz") + "_confounds.tsv"),
                ),
                domain="volume",
                mask=mask_path,
                mask_threshold=0.5,
                volume_connectivity=6,
            ),
            output=OutputConfig(
                directory=output, work_directory=tmp_path / "work", prefix="sub-resume"
            ),
            coarsening=CoarseningConfig(
                target_vertices=3,
                iterations=3,
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
        ),
        run,
    )


def test_module_resumes_at_missing_connectivity_without_recoarsening(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, _run = _small_resumable_volume_config(tmp_path)
    outputs = run(cfg)
    label_mtime = outputs["microparcels"].stat().st_mtime_ns
    checkpoint = cfg.output.work_directory / "pass-01_edge_correlations.npz"
    checkpoint_mtime = checkpoint.stat().st_mtime_ns
    outputs["connectivity"].unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("completed coarsening was unexpectedly recomputed")

    monkeypatch.setattr("nro.modules.microparcellation.module.local_edge_correlations", forbidden)
    monkeypatch.setattr("nro.modules.microparcellation.module.loukas_variation_edges", forbidden)
    resumed = run(cfg)

    assert resumed["connectivity"].is_file()
    assert resumed["microparcels"].stat().st_mtime_ns == label_mtime
    assert checkpoint.stat().st_mtime_ns == checkpoint_mtime
    assert (cfg.output.directory / "unrelated-user-file.txt").read_text() == "preserve me"


def test_module_recomputes_stale_stages_when_functional_input_is_newer(
    tmp_path: Path,
) -> None:
    cfg, run_path = _small_resumable_volume_config(tmp_path)
    outputs = run(cfg)
    watched = {
        "correlations": cfg.output.work_directory / "pass-01_edge_correlations.npz",
        "labels_checkpoint": cfg.output.work_directory / "pass-01_labels.npy",
        "labels": outputs["microparcels"],
        "connectivity": outputs["connectivity"],
        "manifest": outputs["manifest"],
    }
    before = {name: path.stat().st_mtime_ns for name, path in watched.items()}
    time.sleep(0.02)
    run_path.touch()

    run(cfg)

    after = {name: path.stat().st_mtime_ns for name, path in watched.items()}
    assert all(after[name] > before[name] for name in watched)
