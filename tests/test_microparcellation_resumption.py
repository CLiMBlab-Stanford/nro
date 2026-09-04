import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.microparcellation.config import (
    CoarseningConfig,
    ConnectivityConfig,
    InputsConfig,
    OutputConfig,
    ModuleConfig,
)
from nro.microparcellation.module import run


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


def test_surface_module_uses_runner_and_skips_all_current_stages(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    surfaces = []
    functionals = []
    for index, hemi in enumerate(("L", "R"), start=1):
        surface = tmp_path / f"sub-test_hemi-{hemi}_midthickness.surf.gii"
        functional = tmp_path / f"sub-test_hemi-{hemi}_desc-clean_bold.func.gii"
        for surface_type in ("pial", "midthickness", "white", "inflated"):
            _write_surface(
                tmp_path / f"sub-test_hemi-{hemi}_{surface_type}.surf.gii"
            )
        _write_functional(functional, index)
        surfaces.append(surface)
        functionals.append(functional)
    cfg = ModuleConfig(
        inputs=InputsConfig(
            functional=(tuple(functionals),),
            domain="surface",
            surface=tuple(surfaces),
        ),
        output=OutputConfig(
            directory=tmp_path / "output",
            work_directory=tmp_path / "work",
            prefix="sub-test",
        ),
        wb_command="/bin/true",
        coarsening=CoarseningConfig(
            target_vertices=4,
            iterations=2,
            exponential_temperature=0.1,
            eigenvectors=2,
            max_levels=10,
            eigensolver_tolerance=1e-5,
        ),
        connectivity=ConnectivityConfig(
            minimum_trs=4,
            temporal_block_size=4,
            reliability_weighting=False,
            reliability_vertex_block_size=8,
            global_signal_regression=False,
        ),
    )

    def fake_borders(dlabel, surface_paths, output_dir, prefix, *, runner, executable=None):
        paths = tuple(
            output_dir / f"{prefix}_hemi-{hemi}_microparcels.border"
            for hemi in ("L", "R")
        )
        for path in paths:
            path.write_text("border")
        return paths

    monkeypatch.setattr("nro.microparcellation.module.write_borders", fake_borders)
    monkeypatch.setattr(
        "nro.microparcellation.module.resolve_wb_command",
        lambda _configured: "/bin/true",
    )
    with caplog.at_level(logging.INFO, logger="nro.microparcellation.module"):
        outputs = run(cfg)
    assert len(outputs["scene_surfaces"]) == 8
    scene = outputs["scene"].read_text()
    for path in outputs["scene_surfaces"]:
        assert path.name in scene
    root = ET.fromstring(scene)
    spec_surfaces = [
        child.text
        for item in root.findall('.//Object[@Class="SpecFileDataFile"]')
        if item.findtext('./Object[@Name="dataFileType"]') == "SURFACE"
        for child in item.findall('./Object[@Name="fileName"]')
    ]
    assert spec_surfaces == [path.name for path in outputs["scene_surfaces"]]
    assert all(
        item.findtext('./Object[@Name="selected"]') == "true"
        for item in root.findall('.//Object[@Class="SpecFileDataFile"]')
        if item.findtext('./Object[@Name="dataFileType"]') == "SURFACE"
    )
    active_surfaces = {
        item.text
        for item in root.findall('.//Object[@Name="m_selectedSurfacePathName"]')
    }
    assert active_surfaces == {
        "sub-test_hemi-L_midthickness.surf.gii",
        "sub-test_hemi-R_midthickness.surf.gii",
    }
    mtimes = {
        path: path.stat().st_mtime_ns
        for value in outputs.values()
        for path in ((value,) if isinstance(value, Path) else value)
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a current step unexpectedly recomputed")

    monkeypatch.setattr("nro.microparcellation.module.load_surfaces", forbidden)
    monkeypatch.setattr("nro.microparcellation.module.local_edge_correlations", forbidden)
    monkeypatch.setattr("nro.microparcellation.module.parcel_correlations", forbidden)
    shutil.rmtree(cfg.output.work_directory)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nro.microparcellation.module"):
        resumed = run(cfg)

    assert all(path.stat().st_mtime_ns == mtime for path, mtime in mtimes.items())
    assert resumed.keys() == outputs.keys()
    assert "Skipping microparcellation module" in caplog.text
    assert not cfg.output.work_directory.exists()
    assert "Name: Microparcellation Module" in caplog.text
    assert "Status: Success" in caplog.text
    assert "Total Time Elapsed:" in caplog.text
