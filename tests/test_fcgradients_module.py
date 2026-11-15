from __future__ import annotations

import logging
import os
import shutil
from itertools import count
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml
from nibabel.gifti import GiftiDataArray, GiftiImage

from nro.engine.artifact_metadata import metadata_contract_compatible
from nro.engine.scenes import public_manifest_paths
from nro.modules.fcgradients.config import (
    InputsConfig,
    ModuleConfig,
    OutputConfig,
    WorkbenchConfig,
    scientific_settings,
    validate_config,
)
from nro.modules.fcgradients.contract import (
    fcgradients_output_contract,
    fcgradients_output_paths,
)
from nro.modules.fcgradients.module import build_module, correlation_gradient_command
from nro.orchestration.runner import Runner
from nro.viz.catalog import build_layers
from nro.viz.data import layer_map


def _workbench() -> str | None:
    from nro.site.paths import WB_COMMAND_PATH

    for candidate in (str(WB_COMMAND_PATH), shutil.which("wb_command")):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


requires_workbench = pytest.mark.skipif(_workbench() is None, reason="wb_command unavailable")


def _options(**changes) -> WorkbenchConfig:
    values = {
        "fisher_z": False,
        "double_correlation": False,
        "fisher_z_first": False,
        "presmooth_fwhm_mm": 0.0,
        "surface_exclude_mm": 0.0,
        "volume_exclude_mm": 0.0,
        "memory_limit_gb": None,
    }
    values.update(changes)
    return WorkbenchConfig(**values)


def _source_manifest(path: Path, timeseries: Path, domain: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "domain": domain,
                "representation": "full",
                "low_rank": None,
                "outputs": {"timeseries": str(timeseries)},
            }
        )
    )
    return path


def _config(tmp_path: Path, inputs: InputsConfig, options: WorkbenchConfig) -> ModuleConfig:
    return ModuleConfig(
        inputs=inputs,
        output=OutputConfig(
            directory=tmp_path / "out",
            work_directory=tmp_path / "work",
            prefix="sub-01_space-test_smoothing-0mm",
            overwrite=False,
        ),
        backend="workbench_correlation_gradient",
        workbench=options,
        wb_command=_workbench() or "wb_command",
    )


def _execute(cfg: ModuleConfig) -> dict[str, Path]:
    runner = Runner(
        module_name="test",
        container=None,
        binds=(),
        logger=logging.getLogger("fcgradients-test"),
        next_step=count(1).__next__,
    )
    result = build_module(cfg, runner)
    with runner.run_context():
        runner.execute()
    return result


def _two_region_series(locations: int, boundary: int, frames: int = 80) -> np.ndarray:
    """Return a series whose locations below ``boundary`` share one signal."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((frames, locations)).astype(np.float32)
    shared = rng.standard_normal((frames, 2)).astype(np.float32) * 3
    data[:, :boundary] += shared[:, :1]
    data[:, boundary:] += shared[:, 1:]
    return data


def test_surface_command_maps_options_to_workbench_flags() -> None:
    command = correlation_gradient_command(
        "wb",
        Path("in.dtseries.nii"),
        Path("out.dscalar.nii"),
        options=_options(
            fisher_z=True,
            double_correlation=True,
            fisher_z_first=True,
            presmooth_fwhm_mm=4.0,
            surface_exclude_mm=10.0,
            volume_exclude_mm=8.0,
            memory_limit_gb=12.5,
        ),
        surfaces=(Path("L.surf.gii"), Path("R.surf.gii")),
    )

    assert command == [
        "wb",
        "-cifti-correlation-gradient",
        "in.dtseries.nii",
        "out.dscalar.nii",
        "-left-surface",
        "L.surf.gii",
        "-right-surface",
        "R.surf.gii",
        "-surface-presmooth",
        "4",
        "-presmooth-fwhm",
        "-fisher-z",
        "-surface-exclude",
        "10",
        "-mem-limit",
        "12.5",
        "-double-correlation",
        "-fisher-z-first",
    ]


def test_volume_command_uses_volume_options_and_omits_disabled_ones() -> None:
    command = correlation_gradient_command(
        "wb",
        Path("in.dtseries.nii"),
        Path("out.dscalar.nii"),
        options=_options(surface_exclude_mm=10.0, volume_exclude_mm=6.0, presmooth_fwhm_mm=2.0),
    )

    assert command == [
        "wb",
        "-cifti-correlation-gradient",
        "in.dtseries.nii",
        "out.dscalar.nii",
        "-volume-presmooth",
        "2",
        "-presmooth-fwhm",
        "-volume-exclude",
        "6",
    ]


def test_memory_limit_is_not_a_scientific_setting(tmp_path: Path) -> None:
    inputs = InputsConfig(
        source_manifest=tmp_path / "m.yaml",
        timeseries=tmp_path / "s.nii",
        representation="full",
        domain="volume",
        space="MNITest",
        smoothing_mm=0,
        mask=tmp_path / "mask.nii",
    )
    small = scientific_settings(_config(tmp_path, inputs, _options(memory_limit_gb=1.0)))
    large = scientific_settings(_config(tmp_path, inputs, _options(memory_limit_gb=64.0)))

    assert small == large
    assert "memory_limit_gb" not in small["workbench"]


def test_null_memory_limit_uses_most_of_the_allocation() -> None:
    from nro.modules.fcgradients.module import effective_options

    assert effective_options(_options(), {"NRO_WORKER_MEMORY_GB": "64"}).memory_limit_gb == 48.0
    assert effective_options(_options(), {"SLURM_MEM_PER_NODE": "32768"}).memory_limit_gb == 24.0
    assert effective_options(_options(), {}).memory_limit_gb is None
    explicit = _options(memory_limit_gb=10.0)
    assert effective_options(explicit, {"NRO_WORKER_MEMORY_GB": "64"}) == explicit


def test_fisher_z_first_matters_only_with_double_correlation(tmp_path: Path) -> None:
    inputs = InputsConfig(
        source_manifest=tmp_path / "m.yaml",
        timeseries=tmp_path / "s.nii",
        representation="full",
        domain="volume",
        space="MNITest",
        smoothing_mm=0,
        mask=tmp_path / "mask.nii",
    )

    def settings(**changes):
        return scientific_settings(_config(tmp_path, inputs, _options(**changes)))

    assert settings(fisher_z_first=True) == settings(fisher_z_first=False)
    assert settings(double_correlation=True, fisher_z_first=True) != settings(
        double_correlation=True, fisher_z_first=False
    )


def test_geometry_is_required_for_each_domain(tmp_path: Path) -> None:
    surface = InputsConfig(
        source_manifest=tmp_path / "m.yaml",
        timeseries=tmp_path / "s.dtseries.nii",
        representation="full",
        domain="surface",
        space="fsnative",
        smoothing_mm=0,
    )
    with pytest.raises(ValueError, match="midthickness"):
        validate_config(_config(tmp_path, surface, _options()))
    volume = InputsConfig(
        source_manifest=tmp_path / "m.yaml",
        timeseries=tmp_path / "s.nii",
        representation="full",
        domain="volume",
        space="MNITest",
        smoothing_mm=0,
    )
    with pytest.raises(ValueError, match="gray-matter mask"):
        validate_config(_config(tmp_path, volume, _options()))


@requires_workbench
def test_volume_gradient_peaks_at_the_connectivity_boundary(tmp_path: Path) -> None:
    shape = (10, 6, 6)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    mask = np.zeros(shape, dtype=np.float32)
    mask[1:9, 1:5, 1:5] = 1.0
    grid = np.zeros((*shape, 80), dtype=np.float32)
    voxels = np.argwhere(mask > 0)
    order = np.argsort(voxels[:, 0], kind="stable")
    voxels = voxels[order]
    series = _two_region_series(len(voxels), int(np.sum(voxels[:, 0] < 5)))
    grid[tuple(voxels.T)] = series.T
    timeseries = tmp_path / "dynconn.nii"
    nib.save(nib.Nifti1Image(grid, affine), timeseries)
    mask_path = tmp_path / "gm_probseg.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine[:, :]), mask_path)
    inputs = InputsConfig(
        source_manifest=_source_manifest(tmp_path / "dynconn.yaml", timeseries, "volume"),
        timeseries=timeseries,
        representation="full",
        domain="volume",
        space="MNITest",
        smoothing_mm=0,
        mask=mask_path,
        mask_threshold=0.5,
    )
    cfg = _config(tmp_path, inputs, _options())

    result = _execute(cfg)

    gradient = np.asarray(nib.load(str(result["gradient_volume"])).dataobj).reshape(shape)
    profile = [float(gradient[x][mask[x] > 0].mean()) for x in range(1, 9)]
    assert int(np.argmax(profile)) + 1 in {4, 5}
    assert np.count_nonzero(gradient[mask == 0]) == 0
    manifest = yaml.safe_load(result["manifest"].read_text())
    assert manifest["domain"] == "volume"
    assert manifest["structures"] == [
        {"name": "CIFTI_STRUCTURE_OTHER", "locations": int(mask.sum())}
    ]
    assert manifest["geometry"]["structure"] == "OTHER"
    assert manifest["backend_details"]["workbench_version"]
    assert metadata_contract_compatible(
        manifest["output_metadata_contract"], fcgradients_output_contract()
    )
    assert set(result) == set(fcgradients_output_paths(tmp_path, "x", "volume")) - {"index"}


def _grid_surface(path: Path, columns: int, rows: int) -> None:
    x, y = np.meshgrid(np.arange(columns, dtype=np.float32), np.arange(rows, dtype=np.float32))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, dtype=np.float32)))
    triangles = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            triangles.append((a, a + 1, a + columns))
            triangles.append((a + 1, a + columns + 1, a + columns))
    image = GiftiImage()
    image.add_gifti_data_array(
        GiftiDataArray(points.astype(np.float32), intent="NIFTI_INTENT_POINTSET")
    )
    image.add_gifti_data_array(
        GiftiDataArray(np.asarray(triangles, dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE")
    )
    nib.save(image, str(path))


@requires_workbench
def test_surface_gradient_peaks_at_the_connectivity_boundary(tmp_path: Path) -> None:
    columns, rows = 12, 6
    vertices = columns * rows
    surfaces = (tmp_path / "L.surf.gii", tmp_path / "R.surf.gii")
    for surface in surfaces:
        _grid_surface(surface, columns, rows)
    column_of = np.tile(np.arange(columns), rows)
    hemisphere = np.zeros((80, vertices), dtype=np.float32)
    order = np.argsort(column_of, kind="stable")
    hemisphere[:, order] = _two_region_series(vertices, int(np.sum(column_of < 6)))
    data = np.concatenate((hemisphere, hemisphere[:, ::-1]), axis=1)
    axis = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(vertices), vertices, name="CIFTI_STRUCTURE_CORTEX_LEFT"
    ) + nib.cifti2.BrainModelAxis.from_surface(
        np.arange(vertices), vertices, name="CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    series = nib.cifti2.SeriesAxis(0.0, 1.0, data.shape[0], unit="SECOND")
    timeseries = tmp_path / "dynconn.dtseries.nii"
    nib.save(
        nib.Cifti2Image(data, header=nib.cifti2.Cifti2Header.from_axes((series, axis))),
        str(timeseries),
    )
    inputs = InputsConfig(
        source_manifest=_source_manifest(tmp_path / "dynconn.yaml", timeseries, "surface"),
        timeseries=timeseries,
        representation="full",
        domain="surface",
        space="fsnative",
        smoothing_mm=0,
        surfaces=surfaces,
    )

    result = _execute(_config(tmp_path, inputs, _options()))

    gradient = np.asarray(nib.load(str(result["gradient"])).dataobj)[0, :vertices]
    interior = (column_of > 0) & (column_of < columns - 1)
    by_column = [
        float(gradient[(column_of == column) & interior].mean()) for column in range(1, 11)
    ]
    assert int(np.argmax(by_column)) + 1 in {5, 6}
    manifest = yaml.safe_load(result["manifest"].read_text())
    assert [item["name"] for item in manifest["structures"]] == [
        "CIFTI_STRUCTURE_CORTEX_LEFT",
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    ]
    assert manifest["geometry"]["surfaces"] == [str(path) for path in surfaces]
    assert "gradient_volume" not in result

    # The viewer reads the map from the manifest's outputs as one cortical metric layer.
    viewable = public_manifest_paths("fcgradients", result["manifest"])
    assert result["gradient"] in viewable
    (layer,) = build_layers([("fcgradients", path, "fsnative") for path in viewable], "x")
    assert layer.kind == "metric"
    assert layer.vertex_counts == {"L": vertices, "R": vertices}
    np.testing.assert_allclose(layer_map(layer, 0)["L"], gradient, rtol=1e-6)
