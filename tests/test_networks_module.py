import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import yaml
from scipy import sparse

import nro.modules.networks.__main__ as networks_main
from nro.configuration.store import ConfigStore
from nro.engine.cifti import (
    indexed_cifti_indices,
    load_indexed_cifti_map,
    write_indexed_cifti_sidecar,
)
from nro.engine.images import write_cifti_dense_scalar
from nro.modules.microparcellation.cifti import write_pconn, write_volume_dlabel
from nro.modules.networks.clustering import clustering_membership
from nro.modules.networks.config import (
    ClusteringConfig,
    ConnectivityConfig,
    IcaConfig,
    InputsConfig,
    LabelingConfig,
    ModuleConfig,
    OslomConfig,
    OutputConfig,
)
from nro.modules.networks.ica import ica_membership
from nro.modules.networks.labeling import (
    REFERENCE_ATLASES,
    ReferenceAtlas,
    network_map_names,
    project_references_to_cifti,
    rank_reference_candidates,
)
from nro.modules.networks.module import _load_inputs, run
from nro.modules.networks.targets import discover_microparcellation_targets


def _fake_oslom(_graph, directory: Path, _config, *, runner):
    partition = directory / "graph.dat_oslo_files" / "tp"
    partition.parent.mkdir(parents=True, exist_ok=True)
    partition.write_text("#module 0\n0 1\n#module 1\n2\n")
    return [{0, 1}, {2}]


def test_ica_membership_is_reproducible_and_normalized() -> None:
    random = np.random.default_rng(4)
    sources = random.laplace(size=(80, 4))
    profiles = sources @ random.normal(size=(4, 30))
    adjacency = profiles @ profiles.T
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    lower = sparse.tril(sparse.csr_matrix(adjacency), k=-1, format="csr")
    config = IcaConfig(n_networks=4, random_seed=7)

    first = ica_membership(lower, config)
    second = ica_membership(lower, config)

    assert first.shape == (80, 4)
    np.testing.assert_array_equal(first, second)
    assert np.all((0 <= first) & (first <= 1))
    np.testing.assert_allclose(first.max(axis=0), 1.0)


def test_clustering_membership_is_reproducible_and_minmax_normalized() -> None:
    adjacency = np.zeros((40, 40), dtype=np.float32)
    adjacency[:20, :20] = 1.0
    adjacency[20:, 20:] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    lower = sparse.tril(sparse.csr_matrix(adjacency), k=-1, format="csr")
    config = ClusteringConfig(
        n_networks=2,
        repetitions=4,
        random_seed=7,
        n_init=1,
        max_iterations=20,
        batch_size=20,
        max_no_improvement=5,
        reassignment_ratio=0.01,
    )

    first, first_inertias = clustering_membership(lower, config)
    second, second_inertias = clustering_membership(lower, config)

    assert first.shape == (40, 2)
    assert first_inertias.shape == (4,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_inertias, second_inertias)
    np.testing.assert_allclose(first.min(axis=0), 0.0)
    np.testing.assert_allclose(first.max(axis=0), 1.0)
    assert np.all(first[:20].argmax(axis=1) == first[0].argmax())
    assert np.all(first[20:].argmax(axis=1) != first[0].argmax())


def _manifest(root: Path, space: str, domain: str, smoothing_mm: int = 2) -> Path:
    directory = root
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-01_space-{space}_smoothing-{smoothing_mm}mm"
    microparcels = directory / f"{prefix}_desc-microparcellation_dseg.dlabel.nii"
    connectivity = directory / f"{prefix}_connectivity.pconn.nii"
    microparcels.touch()
    connectivity.touch()
    surfaces = []
    label_volume = None
    if domain == "surface":
        for hemi in ("L", "R"):
            surface = directory / f"sub-01_hemi-{hemi}_pial.surf.gii"
            surface.touch()
            surfaces.append(str(surface))
    else:
        label_volume = directory / f"{prefix}_microparcels.nii.gz"
        label_volume.touch()
    path = directory / f"{prefix}_desc-microparcellation_manifest.yaml"
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
    assert targets[1].source_surfaces == ()


def test_network_config_uses_shared_subject_output_directory(tmp_path: Path, monkeypatch) -> None:
    subject = tmp_path / "project" / "derivatives" / "microparcellation" / "main" / "sub-01"
    manifest = _manifest(subject, "fsnative", "surface")
    monkeypatch.setattr(networks_main, "BIDS_PATH", str(tmp_path))
    config = ConfigStore().load_configuration("networks", "main").values
    config["microparcellation_directory"] = "main"

    target, cfg = networks_main.make_target_config(
        "project", "01", "main", config, micro_manifest=manifest
    )

    assert (target.space, target.smoothing_mm) == ("fsnative", 2)
    assert cfg.output.directory == tmp_path / "project/derivatives/networks/main/sub-01"
    assert cfg.output.work_directory.parts[-3:] == (
        "main",
        "space-fsnative_smoothing-2mm",
        "sub-01",
    )
    assert cfg.output.prefix == "sub-01_space-fsnative_smoothing-2mm"
    assert cfg.inputs.domain == "surface"
    assert cfg.parcellation_strategy == "ica"
    assert cfg.ica.n_networks == 50


def test_network_config_routes_branch_outputs(tmp_path):
    from nro.orchestration.branches import BranchPaths
    from nro.orchestration.execution_context import ExecutionContext

    paths = BranchPaths(
        "feature/networks", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV"
    )
    context = ExecutionContext(paths, "demo", "networks:1", ())
    manifest = _manifest(tmp_path / "upstream", "T1w", "volume")
    config = ConfigStore().load_configuration("networks", "main").values
    _, cfg = networks_main.make_target_config(
        "demo", "01", "main", config, micro_manifest=manifest, execution_context=context
    )
    context.require_output(cfg.output.directory)
    context.require_output(cfg.output.work_directory)
    assert cfg.inputs.microparcellation_manifest == manifest
    config["output_dir"] = str(paths.bids / "demo/sub-01")
    with pytest.raises(ValueError, match="outside"):
        networks_main.make_target_config(
            "demo", "01", "main", config, micro_manifest=manifest, execution_context=context
        )


def test_network_entry_selects_upstream_branch(tmp_path, monkeypatch):
    import json

    from nro.modules.microparcellation.paths import output_paths
    from nro.orchestration.branches import BranchPaths
    from nro.orchestration.execution_context import ExecutionContext, InputBinding
    from nro.orchestration.runner import Runner

    paths = BranchPaths(
        "feature/networks", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV"
    )
    dev = BranchPaths("dev", paths.bids, paths.work, paths.development)
    relative = Path("derivatives/microparcellation/main/sub-01")
    upstream = dev.output_project("demo") / relative
    manifest = _manifest(upstream, "T1w", "volume")
    prefix = "sub-01_space-T1w_smoothing-2mm"
    index = output_paths(upstream, prefix)["index"]
    index.write_text(
        json.dumps({"space": "T1w", "smoothing_fwhm_mm": 2, "target_manifest": str(manifest)})
    )
    anatomy = paths.source_project("demo") / "derivatives/preprocessing/main/sub-01/anat"
    anatomy.mkdir(parents=True)
    image = anatomy / "sub-01_T1w.nii.gz"
    image.touch()
    (anatomy / "sub-01_desc-preprocessAnat_manifest.json").write_text(
        json.dumps({"outputs": {"brain_image": str(image), "xfms": {"mni_to_t1": str(image)}}})
    )
    context = ExecutionContext(
        paths,
        "demo",
        "networks:test",
        (
            InputBinding(
                "dev", "micro:test", 1, paths.source_project("demo") / relative, upstream, prefix
            ),
            InputBinding("main", "anat:test", 1, anatomy, anatomy, "sub-01"),
        ),
    )
    cfg = ConfigStore().load_configuration("networks", "main").values
    cfg["microparcellation_directory"] = "main"
    cfg["labeling"]["enabled"] = False
    monkeypatch.setattr(
        networks_main, "select_runtime_config", lambda **kw: tmp_path / "config.yml"
    )
    monkeypatch.setattr(networks_main, "load_runtime_configuration", lambda *a: ("main", cfg))
    monkeypatch.setattr(
        networks_main,
        "load_runtime_workflow_snapshot",
        lambda *a: {
            "configurations": {
                "preprocessing": {
                    "directory": "main",
                    "resolved": {"func": {"output_spaces": ["T1w"]}},
                },
                "microparcellation": {"resolved": {}},
            }
        },
    )
    graphs = []

    def initialize_only(self):
        graph = self._graph.freeze()
        graphs.append(graph)
        for step in graph.steps:
            for output in step.outputs:
                context.require_output(output)
        graph.steps[0].action()
        raise RuntimeError("Test stopped after initialization")

    monkeypatch.setattr(Runner, "execute", initialize_only)
    with pytest.raises(RuntimeError, match="Test stopped after initialization"):
        networks_main.main(["-P", "demo", "-p", "01", "-s", "T1w"], execution_context=context)
    assert len(graphs) == 1
    assert any(manifest in step.inputs for step in graphs[0].steps)
    assert not (paths.source_project("demo") / "derivatives/networks").exists()


def test_oslom_workflow_selects_oslom_network_configuration() -> None:
    workflow = ConfigStore().resolve("oslom")

    assert workflow.selections["networks"] == "oslom"
    assert workflow.configuration("networks").values["parcellation_strategy"] == "oslom"


def test_clustering_workflow_selects_clustering_network_configuration() -> None:
    workflow = ConfigStore().resolve("clustering")

    assert workflow.selections["networks"] == "clustering"
    config = workflow.configuration("networks").values
    assert config["parcellation_strategy"] == "clustering"
    assert config["clustering"]["n_networks"] == 50
    assert config["clustering"]["repetitions"] == 100


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
        [
            np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ],
        ["Network 001 | lana002, dna003", "Network 002"],
    )
    write_indexed_cifti_sidecar(
        output,
        [
            {
                "Name": "Network 001 | lana002, dna003",
                "Network": 1,
                "Labels": ["network001", "lana002", "dna003"],
            },
            {"Name": "Network 002", "Network": 2, "Labels": ["network002"]},
        ],
        lookup_fields=("Network", "Labels"),
    )

    image = nib.load(output)
    assert isinstance(image.header.get_axis(0), nib.cifti2.ScalarAxis)
    assert isinstance(image.header.get_axis(1), nib.cifti2.BrainModelAxis)
    assert image.header.get_axis(0).name.tolist() == [
        "Network 001 | lana002, dna003",
        "Network 002",
    ]
    assert indexed_cifti_indices(output, "Labels", "dna003") == (0,)
    np.testing.assert_array_equal(
        load_indexed_cifti_map(output, "Network", 2),
        [1.0, 0.0, 0.0, 1.0],
    )


def test_candidate_labels_rank_each_reference_independently() -> None:
    networks = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float32)
    references = {atlas.identifier: networks[0].copy() for atlas in REFERENCE_ATLASES}

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
    manifest = tmp_path / "microparcellation_manifest.yaml"
    manifest.write_text("{}")
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcellation_manifest=manifest,
            microparcels=dlabel,
            connectivity=pconn,
            domain="volume",
            space="T1w",
        ),
        output=OutputConfig(tmp_path / "out", tmp_path / "work", "sub-01_space-T1w"),
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    with pytest.raises(ValueError, match="voxel mapping differ"):
        _load_inputs(cfg)


def test_projects_mni_reference_onto_volumetric_cifti(tmp_path: Path, monkeypatch) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.arange(8, dtype=np.int64)
    dlabel, _ = write_volume_dlabel(tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4))
    reference_data = np.arange(8, dtype=np.float32).reshape(mask.shape)
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(reference_data, np.eye(4)), reference_path)
    monkeypatch.setattr(
        "nro.modules.networks.labeling.REFERENCE_ATLASES",
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


def test_native_reference_projection_loads_ants_composite_transform(
    tmp_path: Path, monkeypatch
) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.arange(8, dtype=np.int64)
    dlabel, _ = write_volume_dlabel(tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4))
    reference_data = np.arange(8, dtype=np.float32).reshape(mask.shape)
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(reference_data, np.eye(4)), reference_path)
    transform_path = tmp_path / "mni_to_t1.h5"
    transform_path.touch()
    monkeypatch.setattr(
        "nro.modules.networks.labeling.REFERENCE_ATLASES",
        (ReferenceAtlas("test", str(reference_path)),),
    )

    calls: list[tuple[str, str | None]] = []

    def load_transform(filename, fmt="X5"):
        calls.append((filename, fmt))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "nitransforms",
        SimpleNamespace(
            manip=SimpleNamespace(load=load_transform),
            resampling=SimpleNamespace(apply=lambda _transform, moving, _reference: moving),
        ),
    )

    projected = project_references_to_cifti(
        dlabel,
        space="T1w",
        source_surfaces=(),
        anatomical_reference=reference_path,
        mni_to_t1_transform=transform_path,
    )

    assert calls == [(str(transform_path), None)]
    np.testing.assert_array_equal(projected["test"], reference_data[mask])


def test_volumetric_network_module_writes_dense_network_maps(tmp_path: Path, monkeypatch) -> None:
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
        "nro.modules.networks.module.resolve_oslom_executable",
        lambda _configured: Path("/bin/true"),
    )
    monkeypatch.setattr(
        "nro.modules.networks.module.run_oslom",
        _fake_oslom,
    )
    micro_manifest = tmp_path / "microparcellation_manifest.yaml"
    micro_manifest.write_text(
        yaml.safe_dump({"outputs": {"microparcels_volume": str(tmp_path / "microparcels.nii.gz")}})
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcellation_manifest=micro_manifest,
            microparcels=microparcels,
            connectivity=connectivity,
            domain="volume",
            space="T1w",
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
        ),
        parcellation_strategy="oslom",
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    outputs = run(cfg)

    membership = nib.load(outputs["membership"])
    assert membership.shape == (2, 8)
    assert membership.header.get_axis(0).name.tolist() == ["Network 001", "Network 002"]
    manifest = yaml.safe_load(outputs["manifest"].read_text())
    assert manifest["domain"] == "volume"
    assert manifest["space"] == "T1w"
    assert manifest["n_surface_vertices"] is None
    assert manifest["n_gray_matter_voxels"] == 8
    assert "network_maps" not in manifest["outputs"]
    assert Path(manifest["outputs"]["membership_metadata"]) == outputs["membership_metadata"]
    assert indexed_cifti_indices(outputs["membership"], "Network", 1) == (0,)
    assert indexed_cifti_indices(outputs["membership"], "Labels", "network002") == (1,)
    np.testing.assert_array_equal(
        load_indexed_cifti_map(outputs["membership"], "Labels", "network001"),
        np.asarray(membership.dataobj)[0],
    )
    shutil.rmtree(cfg.output.work_directory)
    monkeypatch.setattr(
        "nro.modules.networks.module._load_inputs",
        lambda _cfg: (_ for _ in ()).throw(
            AssertionError("fresh public outputs must not rematerialize purged WORK")
        ),
    )
    resumed = run(cfg)
    assert resumed["manifest"] == outputs["manifest"]
    assert not cfg.output.work_directory.exists()


def test_volumetric_network_module_publishes_ica_pseudo_probabilities(
    tmp_path: Path, monkeypatch
) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    microparcels, parcel_axis = write_volume_dlabel(
        tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4)
    )
    connectivity = write_pconn(
        tmp_path / "connectivity.pconn.nii",
        np.array(
            [[0.0, 0.8, 0.4], [0.8, 0.0, 0.6], [0.4, 0.6, 0.0]],
            dtype=np.float32,
        ),
        parcel_axis,
    )
    label_volume = tmp_path / "microparcels.nii.gz"
    nib.save(
        nib.Nifti1Image(labels.reshape(mask.shape).astype(np.int16), np.eye(4)),
        label_volume,
    )
    micro_manifest = tmp_path / "microparcellation_manifest.yaml"
    micro_manifest.write_text(
        yaml.safe_dump({"outputs": {"microparcels_volume": str(label_volume)}})
    )
    probabilities = np.array([[0.0, 1.0], [0.5, 0.25], [1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        "nro.modules.networks.module.ica_membership",
        lambda _adjacency, _config: probabilities,
    )
    monkeypatch.setattr(
        "nro.modules.networks.module.resolve_oslom_executable",
        lambda _configured: (_ for _ in ()).throw(
            AssertionError("the ICA strategy must not resolve OSLOM")
        ),
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcellation_manifest=micro_manifest,
            microparcels=microparcels,
            connectivity=connectivity,
            domain="volume",
            space="T1w",
        ),
        output=OutputConfig(
            directory=tmp_path / "networks",
            work_directory=tmp_path / "work" / "networks",
            prefix="sub-01_space-T1w",
        ),
        connectivity=ConnectivityConfig(percentile_cutoff=None),
        parcellation_strategy="ica",
        ica=IcaConfig(n_networks=2),
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    outputs = run(cfg)

    membership = np.asarray(nib.load(outputs["membership"]).dataobj)
    np.testing.assert_array_equal(membership, probabilities[labels].T)
    manifest = yaml.safe_load(outputs["manifest"].read_text())
    assert manifest["parcellation_strategy"] == "ica"
    assert manifest["reference_run"] is None
    assert "pseudo-probability" in manifest["interpretation"]
    assert "not posterior probabilities" in manifest["interpretation"]
    assert "stability" not in outputs

    from dataclasses import replace

    mtimes = {
        path: path.stat().st_mtime_ns
        for path in outputs.values()
        if isinstance(path, Path) and path.is_file()
    }
    run(replace(cfg, oslom=replace(cfg.oslom, timeout_seconds=60)))
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in mtimes.items())


def test_volumetric_network_module_publishes_clustering_frequencies(
    tmp_path: Path, monkeypatch
) -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    microparcels, parcel_axis = write_volume_dlabel(
        tmp_path / "microparcels.dlabel.nii", labels, mask, np.eye(4)
    )
    connectivity = write_pconn(
        tmp_path / "connectivity.pconn.nii",
        np.array(
            [[0.0, 0.8, 0.4], [0.8, 0.0, 0.6], [0.4, 0.6, 0.0]],
            dtype=np.float32,
        ),
        parcel_axis,
    )
    label_volume = tmp_path / "microparcels.nii.gz"
    nib.save(
        nib.Nifti1Image(labels.reshape(mask.shape).astype(np.int16), np.eye(4)),
        label_volume,
    )
    micro_manifest = tmp_path / "microparcellation_manifest.yaml"
    micro_manifest.write_text(
        yaml.safe_dump({"outputs": {"microparcels_volume": str(label_volume)}})
    )
    frequencies = np.array([[1.0, 0.0], [0.75, 0.25], [0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(
        "nro.modules.networks.module.clustering_membership",
        lambda _adjacency, _config: (frequencies, np.array([1.0, 2.0])),
    )
    monkeypatch.setattr(
        "nro.modules.networks.module.resolve_oslom_executable",
        lambda _configured: (_ for _ in ()).throw(
            AssertionError("the clustering strategy must not resolve OSLOM")
        ),
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcellation_manifest=micro_manifest,
            microparcels=microparcels,
            connectivity=connectivity,
            domain="volume",
            space="T1w",
        ),
        output=OutputConfig(
            directory=tmp_path / "networks",
            work_directory=tmp_path / "work" / "networks",
            prefix="sub-01_space-T1w",
        ),
        connectivity=ConnectivityConfig(percentile_cutoff=None),
        parcellation_strategy="clustering",
        clustering=ClusteringConfig(n_networks=2, repetitions=2),
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    outputs = run(cfg)

    membership = np.asarray(nib.load(outputs["membership"]).dataobj)
    np.testing.assert_array_equal(membership, frequencies[labels].T)
    manifest = yaml.safe_load(outputs["manifest"].read_text())
    assert manifest["parcellation_strategy"] == "clustering"
    assert "frequencies of assignment" in manifest["interpretation"]
    assert "stability" not in outputs


def test_missing_public_network_metric_is_rebuilt(tmp_path: Path, monkeypatch) -> None:
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
        "nro.modules.networks.module.resolve_oslom_executable",
        lambda _configured: Path("/bin/true"),
    )
    monkeypatch.setattr(
        "nro.modules.networks.module.run_oslom",
        _fake_oslom,
    )
    micro_manifest = tmp_path / "microparcellation_manifest.yaml"
    micro_manifest.write_text(
        yaml.safe_dump({"outputs": {"microparcels_volume": str(tmp_path / "microparcels.nii.gz")}})
    )
    cfg = ModuleConfig(
        inputs=InputsConfig(
            microparcellation_manifest=micro_manifest,
            microparcels=microparcels,
            connectivity=connectivity,
            domain="volume",
            space="T1w",
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
        ),
        parcellation_strategy="oslom",
        oslom=OslomConfig(initialization="none", repetitions=1),
        labeling=LabelingConfig(enabled=False),
    )

    outputs = run(cfg)
    missing_metric = outputs["membership"]
    missing_metric.unlink()
    obsolete_metric = cfg.output.directory / f"{cfg.output.prefix}_network999_binary.dscalar.nii"
    obsolete_metric.write_text("obsolete result from a prior larger solution")
    untracked_old_file = cfg.output.directory / "unrecognized-old-publication-member.bin"
    untracked_old_file.write_text("belongs to another target in the shared directory")
    tp = cfg.output.work_directory / "oslom_runs" / "run_001" / "graph.dat_oslo_files" / "tp"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text("fixture\n")
    monkeypatch.setattr("nro.modules.networks.module.parse_tp", lambda _path: [{0, 1}, {2}])

    rebuilt = run(cfg)

    assert missing_metric.is_file()
    assert missing_metric == rebuilt["membership"]
    assert not obsolete_metric.exists()
    assert untracked_old_file.exists()
