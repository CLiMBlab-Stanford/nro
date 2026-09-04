from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest
import yaml

import nro.networks.__main__ as networks_main
from nro.configuration.store import ConfigStore
from nro.microparcellation.cifti import write_pconn, write_volume_dlabel
from nro.networks.config import (
    ConnectivityConfig,
    InputsConfig,
    OslomConfig,
    OutputConfig,
    ModuleConfig,
    LabelingConfig,
)
from nro.networks.labeling import (
    REFERENCE_ATLASES,
    ReferenceAtlas,
    network_map_names,
    project_references_to_cifti,
    rank_reference_candidates,
)
from nro.engine.images import write_cifti_dense_scalar
from nro.networks.module import _load_inputs, run
from nro.networks.scene import write_network_scene
from nro.networks.targets import discover_microparcellation_targets


def _fake_oslom(_graph, directory: Path, _config, *, runner):
    partition = directory / "graph.dat_oslo_files" / "tp"
    partition.parent.mkdir(parents=True, exist_ok=True)
    partition.write_text("#module 0\n0 1\n#module 1\n2\n")
    return [{0, 1}, {2}]


def _manifest(root: Path, space: str, domain: str, smoothing_mm: int = 2) -> Path:
    directory = root
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-01_space-{space}_scale-{smoothing_mm}mm"
    microparcels = directory / f"{prefix}_microparcels.dlabel.nii"
    connectivity = directory / f"{prefix}_microparcel_connectivity.pconn.nii"
    microparcels.touch()
    connectivity.touch()
    surfaces = []
    scene_surfaces = []
    label_volume = None
    if domain == "surface":
        for hemi in ("L", "R"):
            surface = directory / f"sub-01_hemi-{hemi}_pial.surf.gii"
            surface.touch()
            surfaces.append(str(surface))
            for kind in ("pial", "midthickness", "white", "inflated"):
                display = directory / f"{prefix}_hemi-{hemi}_{kind}.surf.gii"
                display.touch()
                scene_surfaces.append(str(display))
    else:
        label_volume = directory / f"{prefix}_microparcels.nii.gz"
        label_volume.touch()
    path = directory / f"{prefix}_manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "domain": domain,
                "space": space,
                "smoothing_fwhm_mm": smoothing_mm,
                "source_surfaces": surfaces,
                "outputs": {
                    "microparcels": str(microparcels),
                    "connectivity": str(connectivity),
                    "scene_surfaces": scene_surfaces,
                    "microparcels_volume": str(label_volume) if label_volume else None,
                },
            }
        )
    )
    return path


def test_discovers_all_current_microparcellation_space_manifests(tmp_path: Path) -> None:
    manifests = (
        _manifest(tmp_path, "fsnative", "surface"),
        _manifest(tmp_path, "T1w", "volume"),
    )

    targets = discover_microparcellation_targets(manifests)

    assert [(target.domain, target.space, target.smoothing_mm) for target in targets] == [
        ("surface", "fsnative", 2),
        ("volume", "T1w", 2),
    ]
    assert len(targets[0].source_surfaces) == 2
    assert len(targets[0].scene_surfaces) == 8
    assert targets[1].source_surfaces == ()
    assert targets[1].label_volume is not None


def test_network_config_uses_shared_subject_output_and_isolated_work_target(
    tmp_path: Path, monkeypatch
) -> None:
    subject = (
        tmp_path
        / "project"
        / "derivatives"
        / "microparcellation"
        / "main"
        / "sub-01"
    )
    manifest = _manifest(subject, "fsnative", "surface")
    monkeypatch.setattr(networks_main, "BIDS_PATH", str(tmp_path))
    config = ConfigStore().load_configuration("networks", "main").values
    config["microparcellation_directory"] = "main"

    target, cfg = networks_main.make_target_config(
        "project", "01", "main", config, micro_manifest=manifest
    )

    assert (target.space, target.smoothing_mm) == ("fsnative", 2)
    assert cfg.output.directory == tmp_path / "project/derivatives/networks/main/sub-01"
    assert cfg.output.work_directory.name == "space-fsnative_smoothing-2mm"
    assert cfg.output.prefix == "sub-01_space-fsnative_scale-2mm"
    assert cfg.inputs.domain == "surface"


def test_volumetric_network_scalars_reuse_microparcellation_brain_model(
    tmp_path: Path,
) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask.flat[:4] = True
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    reference, _ = write_volume_dlabel(
        tmp_path / "microparcels.dlabel.nii",
        labels,
        mask,
        np.eye(4),
    )

    output = write_cifti_dense_scalar(
        tmp_path / "network.dscalar.nii",
        reference,
        [np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)],
        ["network_001_binary"],
    )

    image = nib.load(output)
    assert isinstance(image.header.get_axis(0), nib.cifti2.ScalarAxis)
    assert isinstance(image.header.get_axis(1), nib.cifti2.BrainModelAxis)
    assert image.header.get_axis(0).name.tolist() == ["network_001_binary"]
    np.testing.assert_array_equal(np.asarray(image.dataobj), [[0.0, 1.0, 1.0, 0.0]])


def test_candidate_labels_rank_each_reference_independently() -> None:
    networks = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float32)
    references = {
        atlas.identifier: networks[0].copy()
        for atlas in REFERENCE_ATLASES
    }

    records = rank_reference_candidates(networks, references, 1)
    names = network_map_names(3, records)

    assert len(records) == len(REFERENCE_ATLASES)
    assert {record["network"] for record in records if record["similarity_rank"] == 1} == {1}
    assert "lana001" in names[0] and "aud001" in names[0]
    assert names[2] == "Network 003"


def test_rejects_pconn_with_different_spatial_parcel_mapping(tmp_path: Path) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    dlabel, _ = write_volume_dlabel(tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4))
    _other, wrong_axis = write_volume_dlabel(
        tmp_path / "wrong.dlabel.nii",
        np.array([0, 1, 0, 1, 2, 1, 2, 0], dtype=np.int64),
        mask,
        np.eye(4),
    )
    pconn = write_pconn(tmp_path / "connectivity.pconn.nii", np.eye(3), wrong_axis)
    cfg = ModuleConfig(
        inputs=InputsConfig(dlabel, pconn, domain="volume", space="T1w"),
        output=OutputConfig(tmp_path / "out", tmp_path / "work", "sub-01_space-T1w"),
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    with pytest.raises(ValueError, match="voxel mapping differ"):
        _load_inputs(cfg)


def test_projects_mni_reference_onto_volumetric_cifti(
    tmp_path: Path, monkeypatch
) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.arange(8, dtype=np.int64)
    dlabel, _ = write_volume_dlabel(
        tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4)
    )
    reference_data = np.arange(8, dtype=np.float32).reshape(mask.shape)
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(reference_data, np.eye(4)), reference_path)
    monkeypatch.setattr(
        "nro.networks.labeling.REFERENCE_ATLASES",
        (ReferenceAtlas("test", str(reference_path)),),
    )

    projected = project_references_to_cifti(
        dlabel,
        space="MNI152NLin2009cAsym",
        source_surfaces=(),
        anatomical_reference=None,
        mni_to_t1_transform=None,
    )

    np.testing.assert_array_equal(projected["test"], reference_data[mask])


def test_surface_network_scene_copies_display_surfaces_without_source_references(
    tmp_path: Path,
) -> None:
    source = tmp_path / "micro"
    output = tmp_path / "networks"
    source.mkdir()
    output.mkdir()
    surfaces = []
    for hemi in ("L", "R"):
        for kind in ("pial", "midthickness", "white", "inflated"):
            path = source / f"sub-01_space-fsnative_hemi-{hemi}_{kind}.surf.gii"
            path.write_text(f"{hemi} {kind}\n", encoding="utf-8")
            surfaces.append(path)
    connectivity = source / "source.pconn.nii"
    membership = output / "sub-01_space-fsnative_desc-membership_network.dscalar.nii"
    connectivity.write_bytes(b"pconn")
    membership.write_bytes(b"dscalar")

    scene, assets = write_network_scene(
        output / "sub-01_space-fsnative_networks.scene",
        domain="surface",
        membership=membership,
        connectivity=connectivity,
        scene_surfaces=tuple(surfaces),
    )

    text = scene.read_text(encoding="utf-8")
    assert len(assets) == 9
    assert all(path.is_file() for path in assets)
    assert str(source.resolve()) not in text
    assert "sub-01_space-fsnative_hemi-L_midthickness.surf.gii" in text


def test_volumetric_network_module_writes_dense_network_maps(
    tmp_path: Path, monkeypatch
) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    microparcels, parcel_axis = write_volume_dlabel(
        tmp_path / "microparcels.dlabel.nii",
        labels,
        mask,
        np.eye(4),
    )
    nib.save(
        nib.Nifti1Image(labels.reshape(mask.shape).astype(np.int16), np.eye(4)),
        tmp_path / "microparcels.nii.gz",
    )
    connectivity = write_pconn(
        tmp_path / "connectivity.pconn.nii",
        np.array(
            [[0.0, 0.8, 0.4], [0.8, 0.0, 0.6], [0.4, 0.6, 0.0]],
            dtype=np.float32,
        ),
        parcel_axis,
    )
    monkeypatch.setattr(
        "nro.networks.module.resolve_oslom_executable",
        lambda _configured: Path("/bin/true"),
    )
    monkeypatch.setattr(
        "nro.networks.module.run_oslom",
        _fake_oslom,
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcels=microparcels,
            connectivity=connectivity,
            domain="volume",
            space="T1w",
            label_volume=tmp_path / "microparcels.nii.gz",
        ),
        output=OutputConfig(
            directory=tmp_path / "networks",
            work_directory=tmp_path / "work" / "networks",
            prefix="sub-01_space-T1w",
        ),
        connectivity=ConnectivityConfig(
            transform="clip_positive",
            minimum_weight=0.0,
            percentile_cutoff=None,
            write_matrix=False,
        ),
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    outputs = run(cfg)

    membership = nib.load(outputs["membership"])
    assert membership.shape == (2, 8)
    assert membership.header.get_axis(0).name.tolist() == ["Network 001", "Network 002"]
    assert outputs["scene"].is_file()
    assert all(path.is_file() for path in outputs["scene_assets"])
    scene_text = outputs["scene"].read_text(encoding="utf-8")
    assert str(connectivity.resolve()) not in scene_text
    assert outputs["scene_assets"][0].name in scene_text
    manifest = yaml.safe_load(outputs["manifest"].read_text())
    assert manifest["domain"] == "volume"
    assert manifest["space"] == "T1w"
    assert manifest["n_surface_vertices"] is None
    assert manifest["n_gray_matter_voxels"] == 8

    shutil.rmtree(cfg.output.work_directory)
    monkeypatch.setattr(
        "nro.networks.module._load_inputs",
        lambda _cfg: (_ for _ in ()).throw(
            AssertionError("fresh public outputs must not rematerialize purged WORK")
        ),
    )
    resumed = run(cfg)
    assert resumed["manifest"] == outputs["manifest"]
    assert not cfg.output.work_directory.exists()


def test_missing_public_network_metric_is_rebuilt(
    tmp_path: Path, monkeypatch
) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    microparcels, parcel_axis = write_volume_dlabel(
        tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4)
    )
    nib.save(
        nib.Nifti1Image(labels.reshape(mask.shape).astype(np.int16), np.eye(4)),
        tmp_path / "microparcels.nii.gz",
    )
    connectivity = write_pconn(
        tmp_path / "connectivity.pconn.nii",
        np.array(
            [[0.0, 0.8, 0.4], [0.8, 0.0, 0.6], [0.4, 0.6, 0.0]],
            dtype=np.float32,
        ),
        parcel_axis,
    )
    monkeypatch.setattr(
        "nro.networks.module.resolve_oslom_executable",
        lambda _configured: Path("/bin/true"),
    )
    monkeypatch.setattr(
        "nro.networks.module.run_oslom",
        _fake_oslom,
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcels=microparcels,
            connectivity=connectivity,
            domain="volume",
            space="T1w",
            label_volume=tmp_path / "microparcels.nii.gz",
        ),
        output=OutputConfig(
            directory=tmp_path / "networks",
            work_directory=tmp_path / "work" / "networks",
            prefix="sub-01_space-T1w",
        ),
        connectivity=ConnectivityConfig(
            transform="clip_positive",
            minimum_weight=0.0,
            percentile_cutoff=None,
            write_matrix=False,
        ),
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    outputs = run(cfg)
    missing_metric = outputs["membership"]
    missing_metric.unlink()
    obsolete_metric = (
        cfg.output.directory
        / f"{cfg.output.prefix}_network999_binary.dscalar.nii"
    )
    obsolete_metric.write_text("obsolete result from a prior larger solution")
    untracked_old_file = cfg.output.directory / "unrecognized-old-publication-member.bin"
    untracked_old_file.write_text("belongs to another target in the shared directory")
    tp = (
        cfg.output.work_directory
        / "oslom_runs"
        / "run_001"
        / "graph.dat_oslo_files"
        / "tp"
    )
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text("fixture\n")
    monkeypatch.setattr(
        "nro.networks.module.parse_tp", lambda _path: [{0, 1}, {2}]
    )

    rebuilt = run(cfg)

    assert missing_metric.is_file()
    assert missing_metric == rebuilt["membership"]
    assert not obsolete_metric.exists()
    assert untracked_old_file.exists()
