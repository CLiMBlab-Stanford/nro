from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nro.bin.scene import build_parser, scene_id
from nro.engine.cli import core_selection
from nro.engine.scenes import (
    SceneSource,
    base_scene,
    build_scene_bundle,
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


def test_scene_viewer_uses_an_x11_slurm_allocation(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))

    run_x11(
        ["/opt/workbench/wb_view", "/data/example.scene"], partition="interactive", account="lab"
    )

    argv, options = calls[0]
    assert argv[:3] == ["/usr/bin/srun", "--x11", "--partition=interactive"]
    assert "--account=lab" in argv
    assert argv[-2:] == ["/opt/workbench/wb_view", "/data/example.scene"]
    assert options == {"check": True}


def test_scene_viewer_reuses_an_existing_slurm_allocation(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(
        shutil, "which", lambda _name: pytest.fail("existing allocations do not invoke srun")
    )
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))

    run_x11(
        ["/opt/workbench/wb_view", "/data/example.scene"],
        partition="interactive",
        account="lab",
    )

    assert calls == [(["/opt/workbench/wb_view", "/data/example.scene"], {"check": True})]


def test_scene_viewer_requires_an_x11_display(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(ValueError, match="DISPLAY"):
        run_x11(["wb_view", "example.scene"], partition="interactive", account=None)


def test_surface_base_scene_references_existing_geometry(tmp_path: Path) -> None:
    surfaces = _surfaces(tmp_path)
    text = base_scene(scene_id="test", surfaces=surfaces)
    assert str(tmp_path / "sub-01_hemi-L_midthickness.surf.gii") in text
    assert "{{" not in text


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
