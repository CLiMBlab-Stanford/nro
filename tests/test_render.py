from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import nibabel as nib
import numpy as np
import pytest
import yaml

from nro.bin.render import build_parser
from nro.engine.rendering import (
    _write_dynconn_seed,
    _write_pconn_seed,
    _write_volume_dynconn_seed,
    read_seed_file,
    render_scene,
)


def _brain_axis() -> nib.cifti2.BrainModelAxis:
    return nib.cifti2.BrainModelAxis.from_surface(
        np.arange(3),
        3,
        name="CortexLeft",
    )


def _surfaces() -> dict[str, np.ndarray]:
    return {
        "CIFTI_STRUCTURE_CORTEX_LEFT": np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        )
    }


def _overlay_scene(path: Path) -> None:
    root = ElementTree.Element("Root")
    overlays = ElementTree.SubElement(root, "ObjectArray", Name="m_overlays")
    element = ElementTree.SubElement(overlays, "Element")
    overlay = ElementTree.SubElement(element, "Object", Class="Overlay")
    for name, value in (
        ("m_enabled", "false"),
        ("selectedMapFileNameWithPath", ""),
        ("selectedMapFile", ""),
        ("selectedMapName", ""),
        ("selectedMapIndex", "0"),
    ):
        ElementTree.SubElement(overlay, "Object", Name=name).text = value
    ElementTree.ElementTree(root).write(path, encoding="unicode")


def test_render_parser_uses_shared_selectors_and_render_options(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["-P", "demo", "-p", "01", "-m", "networks", "--output-dir", str(tmp_path)]
    )

    assert args.project == ["demo"]
    assert args.participant == ["01"]
    assert args.module == ["networks"]
    assert args.output_dir == tmp_path


def test_seed_file_uses_project_and_subject_t1w_coordinates(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.yml"
    seed_file.write_text(
        "projects:\n  nptl:\n    sub-t20:\n      - xyz_mm: [-24, -4, -18]\n",
        encoding="utf-8",
    )

    seeds = read_seed_file(seed_file)

    assert tuple(seeds) == (("nptl", "t20"),)
    np.testing.assert_array_equal(seeds[("nptl", "t20")][0], [-24.0, -4.0, -18.0])


def test_seed_file_rejects_nonfinite_coordinates(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.yml"
    seed_file.write_text(
        "projects:\n  nptl:\n    t20:\n      - xyz_mm: [.nan, 0, 0]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        read_seed_file(seed_file)


def test_dynconn_seed_uses_nearest_vertex_and_streams_correlations(tmp_path: Path) -> None:
    source = tmp_path / "timeseries.dtseries.nii"
    output = tmp_path / "seed.dscalar.nii"
    values = np.array(
        [[0, 0, 0], [1, 1, -1], [2, 0, 0], [3, -1, 1]],
        dtype=np.float32,
    )
    nib.save(
        nib.Cifti2Image(
            values,
            header=nib.cifti2.Cifti2Header.from_axes(
                (nib.cifti2.SeriesAxis(0, 1, len(values)), _brain_axis())
            ),
        ),
        source,
    )

    resolved = _write_dynconn_seed(source, output, np.array([5.1, 0.0, 0.0]), _surfaces())

    assert resolved["vertex"] == 1
    assert resolved["distance_mm"] == pytest.approx(0.1)
    correlations = np.asarray(nib.load(output).dataobj[0])
    assert correlations[1] == pytest.approx(1.0)
    assert correlations[2] == pytest.approx(-1.0)


def test_pconn_seed_uses_nearest_parcel(tmp_path: Path) -> None:
    source = tmp_path / "connectivity.pconn.nii"
    output = tmp_path / "seed.pscalar.nii"
    first = nib.cifti2.BrainModelAxis.from_surface(np.array([0, 1]), 3, name="CortexLeft")
    second = nib.cifti2.BrainModelAxis.from_surface(np.array([2]), 3, name="CortexLeft")
    parcels = nib.cifti2.ParcelsAxis.from_brain_models((("first", first), ("second", second)))
    nib.save(
        nib.Cifti2Image(
            np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32),
            header=nib.cifti2.Cifti2Header.from_axes((parcels, parcels)),
        ),
        source,
    )

    resolved = _write_pconn_seed(source, output, np.array([9.9, 0.0, 0.0]), _surfaces())

    assert resolved["parcel"] == "second"
    np.testing.assert_allclose(np.asarray(nib.load(output).dataobj[0]), [0.25, 1.0])


def test_volume_dynconn_seed_uses_nearest_nonzero_voxel(tmp_path: Path) -> None:
    source = tmp_path / "timeseries.nii"
    output = tmp_path / "seed.nii.gz"
    values = np.zeros((2, 2, 2, 4), dtype=np.float32)
    values[0, 0, 0] = [0, 1, 2, 3]
    values[1, 1, 1] = [0, -1, -2, -3]
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(values, affine), source)

    resolved = _write_volume_dynconn_seed(
        source,
        output,
        np.array([1.9, 2.0, 2.0]),
        {},
    )

    assert resolved["voxel_ijk"] == [1, 1, 1]
    assert resolved["distance_mm"] == pytest.approx(0.1)
    correlations = np.asarray(nib.load(output).dataobj)
    assert correlations[1, 1, 1] == pytest.approx(1.0)
    assert correlations[0, 0, 0] == pytest.approx(-1.0)


def test_render_scene_writes_one_image_per_named_map(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "networks.dscalar.nii"
    nib.save(
        nib.Cifti2Image(
            np.ones((2, 3), dtype=np.float32),
            header=nib.cifti2.Cifti2Header.from_axes(
                (nib.cifti2.ScalarAxis(["language", "default"]), _brain_axis())
            ),
        ),
        source,
    )
    source.with_name("networks.dscalar.json").write_text(
        json.dumps(
            {
                "IndexToMetadata": {
                    "0": {"NetworkID": 1},
                    "1": {"NetworkID": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    scene = scene_dir / "selected.scene"
    _overlay_scene(scene)
    (scene_dir / "scene_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "nro-workbench-scene",
                "format_version": 1,
                "scene_id": "selected",
                "project": "demo",
                "participant": "01",
                "space": "fsnative",
                "sources": [
                    {
                        "module": "networks",
                        "role": "derivative",
                        "scene_path": str(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selected_indices = []

    def run(command, **_options):
        assert command[1] == "-scene-capture-image"
        root = ElementTree.parse(command[2]).getroot()
        selected_indices.append(
            next(
                node.text for node in root.iter("Object") if node.get("Name") == "selectedMapIndex"
            )
        )
        Path(command[4]).write_bytes(b"image")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("nro.engine.rendering.subprocess.run", run)
    destination = tmp_path / "paper-images"

    manifest = render_scene(scene, wb_command="wb_command", destination=destination)

    assert manifest == destination / "render_manifest.yaml"
    assert selected_indices == ["0", "1"]
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert [record["map_name"] for record in document["renders"]] == [
        "language",
        "default",
    ]
    assert [record["metadata"]["NetworkID"] for record in document["renders"]] == [1, 2]
    assert len(tuple(destination.glob("*.png"))) == 2


def test_render_scene_refuses_to_replace_unmanaged_output(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    scene = scene_dir / "selected.scene"
    _overlay_scene(scene)
    (scene_dir / "scene_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "nro-workbench-scene",
                "format_version": 1,
                "scene_id": "selected",
                "project": "demo",
                "participant": "01",
                "space": "fsnative",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="unmanaged"):
        render_scene(scene, wb_command="wb_command", destination=destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "user data"
