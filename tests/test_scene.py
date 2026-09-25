from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import yaml

from nro.bin.scene import _collect, _viewer_command, build_parser, scene_id
from nro.engine import viewer_broker
from nro.engine.cli import core_selection
from nro.engine.scenes import (
    SceneSource,
    base_scene,
    build_scene_bundle,
    manifest_lesion_qc_paths,
    manifest_surface_families,
)
from nro.engine.slurm import run_x11


def _surfaces(root: Path) -> tuple[Path, ...]:
    result = []
    for hemi in ("L", "R"):
        for kind in ("pial", "midthickness", "white", "inflated"):
            path = root / f"sub-01_hemi-{hemi}_{kind}.surf.gii"
            path.write_text(f"{hemi} {kind}\n", encoding="utf-8")
            result.append(path)
    return tuple(result)


def test_scene_parser_uses_shared_selectors() -> None:
    args = build_parser().parse_args(
        ["-p", "01", "-P", "demo", "-m", "networks", "-s", "fsnative", "-S", "2"]
    )
    selection = core_selection(args)
    assert selection.participants == ("01",)
    assert selection.modules == ("networks",)
    assert selection.spaces == ("fsnative",)
    assert selection.smoothing == (2,)
    assert scene_id("01", "fsnative", 2, {}, selection).startswith(
        "sub-01_space-fsnative_smoothing-2mm_selection-"
    )


def test_scene_viewer_loads_the_generated_scene_without_a_dialog(tmp_path: Path) -> None:
    scene = tmp_path / "example.scene"

    assert _viewer_command(Path("/opt/workbench/wb_view"), [scene]) == [
        "/opt/workbench/wb_view",
        "-scene-load-hd",
        str(scene),
        "1",
    ]


def test_scene_viewer_requires_one_generated_scene(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one scene"):
        _viewer_command(
            Path("/opt/workbench/wb_view"),
            [tmp_path / "first.scene", tmp_path / "second.scene"],
        )


def test_scene_viewer_uses_the_persistent_broker(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "nro.engine.slurm.open_viewer",
        lambda scene, **options: calls.append((scene, options)),
    )

    run_x11(
        ["/opt/workbench/wb_view", "-scene-load-hd", "/data/example.scene", "1"],
        partition="interactive",
        account="lab",
        control=tmp_path / "control",
    )

    scene, options = calls[0]
    assert scene == Path("/data/example.scene")
    assert options == {
        "viewer": Path("/opt/workbench/wb_view"),
        "partition": "interactive",
        "account": "lab",
        "control": tmp_path / "control",
    }


def test_viewer_broker_launch_requests_twelve_hours_and_fixed_resources(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    viewer = tmp_path / "wb_view"
    viewer.touch()
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.setattr(viewer_broker.shutil, "which", lambda _name: "/usr/bin/srun")
    monkeypatch.setattr(
        viewer_broker.subprocess,
        "Popen",
        lambda command, **options: calls.append((command, options)) or SimpleNamespace(pid=123),
    )

    _process, _token, _log = viewer_broker._launch(
        tmp_path,
        viewer=viewer,
        partition="interactive",
        account="lab",
    )

    command, options = calls[0]
    assert command[:3] == ["/usr/bin/srun", "--x11", "--partition=interactive"]
    assert "--account=lab" in command
    assert "--cpus-per-task=2" in command
    assert "--mem=32G" in command
    assert "--time=12:00:00" in command
    assert command[-2:] == ["--viewer", str(viewer)]
    assert options["start_new_session"] is True


def test_viewer_broker_requires_x11_only_when_starting(monkeypatch, tmp_path: Path) -> None:
    scene = tmp_path / "example.scene"
    viewer = tmp_path / "wb_view"
    scene.touch()
    viewer.touch()
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="DISPLAY"):
        viewer_broker.open_viewer(
            scene,
            viewer=viewer,
            partition="interactive",
            account=None,
            control=tmp_path / "control",
        )


def test_viewer_broker_reuses_a_live_allocation(monkeypatch, tmp_path: Path) -> None:
    scene = tmp_path / "example.scene"
    viewer = tmp_path / "wb_view"
    scene.touch()
    viewer.touch()
    active = {"token": "token", "host": "node", "port": 1234}
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(viewer_broker, "state_directory", lambda _control: tmp_path)
    monkeypatch.setattr(viewer_broker, "_live_active", lambda _root: active)
    monkeypatch.setattr(
        viewer_broker,
        "_exchange",
        lambda selected, payload: {"ok": True, "pid": 456},
    )
    monkeypatch.setattr(
        viewer_broker,
        "_launch",
        lambda *_args, **_kwargs: pytest.fail("live brokers must be reused"),
    )

    assert (
        viewer_broker.open_viewer(
            scene,
            viewer=viewer,
            partition="interactive",
            account=None,
            control=tmp_path / "control",
        )
        == 456
    )


def test_viewer_broker_survives_a_disconnected_client() -> None:
    class DisconnectedStream:
        @staticmethod
        def sendall(_payload):
            raise BrokenPipeError

    viewer_broker._send_response(DisconnectedStream(), {"ok": True})


def test_surface_base_scene_references_existing_geometry(tmp_path: Path) -> None:
    surfaces = _surfaces(tmp_path)
    text = base_scene(scene_id="test", surfaces=surfaces)
    assert str(tmp_path / "sub-01_hemi-L_midthickness.surf.gii") in text
    assert "{{" not in text


def test_surface_base_scene_uses_one_row_for_cerebral_views(tmp_path: Path) -> None:
    root = ElementTree.fromstring(base_scene(scene_id="test", surfaces=_surfaces(tmp_path)))
    configurations = [
        node
        for node in root.iter("Object")
        if node.get("Name", "").startswith("m_cerebralConfiguration[")
    ]

    assert len(configurations) == 3
    for configuration in configurations:
        orientations = [
            node.text
            for node in configuration.iter("Object")
            if node.get("Name") == "m_layoutOrientation"
        ]
        assert orientations == ["ROW_LAYOUT_ORIENTATION"]


def test_anatomical_manifest_selects_only_display_surfaces(tmp_path: Path) -> None:
    surfaces = _surfaces(tmp_path)
    sphere = tmp_path / "sub-01_hemi-L_sphere.surf.gii"
    sphere.write_text("sphere\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"outputs":{"surfaces":{'
        + ",".join(f'"surface{index}":"{path}"' for index, path in enumerate((*surfaces, sphere)))
        + "}}}",
        encoding="utf-8",
    )

    assert manifest_surface_families(manifest) == surfaces


def test_lesion_manifest_selects_diagnostic_volumes_and_intact_surfaces(tmp_path: Path) -> None:
    lesion = tmp_path / "sub-01_space-T1w_desc-lesion_mask.nii.gz"
    inpainted = tmp_path / "sub-01_space-T1w_desc-inpainted_T1w.nii.gz"
    intact = tmp_path / "sub-01_space-fsnative_hemi-L_desc-inpainted_white.surf.gii"
    for path in (lesion, inpainted, intact):
        path.write_bytes(b"data")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"outputs":{'
        f'"lesion_mask":"{lesion}",'
        f'"inpainted_t1w":"{inpainted}",'
        f'"intact_surfaces":{{"lh.white":"{intact}"}}'
        "}}",
        encoding="utf-8",
    )

    assert manifest_lesion_qc_paths(manifest) == (lesion, inpainted, intact)

    data, _surfaces, diagnostics = _collect(
        [
            {
                "status": "Success",
                "module": "anat",
                "project": "demo",
                "participant": "01",
                "directory_label": "main",
                "output_prefix": "anat-main",
                "output_root": str(tmp_path),
                "entities_json": "{}",
                "expected_outputs_json": f'["{manifest}"]',
            }
        ]
    )
    assert not data
    assert tuple(source.path for source in diagnostics) == (lesion, inpainted, intact)
    assert tuple(source.role for source in diagnostics) == (
        "lesion_mask",
        "inpainted_anatomical",
        "intact_surface_scaffold",
    )


def test_published_scene_copies_inputs_and_records_checksums(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "derivative"
    source_root.mkdir()
    data = source_root / "sub-01_space-T1w_desc-map_stat.nii.gz"
    data.write_bytes(b"map")

    def run(command, **_kwargs):
        shutil.copyfile(command[2], command[3])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("nro.engine.scenes.subprocess.run", run)
    destination = tmp_path / "scenes" / "scene-01"
    scene = build_scene_bundle(
        destination,
        scene_id="scene-01",
        sources=(
            SceneSource(
                "firstlevels",
                "derivative",
                data,
                "main",
                "sub-01",
                source_root,
            ),
        ),
        wb_command="wb_command",
        publish=True,
    )

    assert scene.is_file()
    manifest = yaml.safe_load((destination / "scene_manifest.yaml").read_text())
    assert manifest["mode"] == "published"
    assert manifest["sources"][0]["sha256"]
    copied = destination / manifest["sources"][0]["scene_path"]
    assert copied.read_bytes() == b"map"
