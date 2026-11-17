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
    from nro.site.paths import wb_command_path

    for candidate in (str(wb_command_path()), shutil.which("wb_command")):
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


def _config(
    tmp_path: Path,
    inputs: InputsConfig,
    options: WorkbenchConfig,
    *,
    surface_mesh: str = "target",
) -> ModuleConfig:
    return ModuleConfig(
        inputs=inputs,
        output=OutputConfig(
            directory=tmp_path / "out",
            work_directory=tmp_path / "work",
            prefix="sub-01_space-test_smoothing-0mm",
            overwrite=False,
        ),
        backend="workbench_correlation_gradient",
        surface_mesh=surface_mesh,
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


def _icosphere(subdivisions: int, radius: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices and triangles of a subdivided icosahedron."""
    t = (1 + 5**0.5) / 2
    points = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    vertices = [np.asarray(point, dtype=float) for point in points]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                midpoints[key] = len(vertices)
                vertices.append((vertices[a] + vertices[b]) / 2)
            return midpoints[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    coordinates = np.asarray(vertices)
    coordinates *= radius / np.linalg.norm(coordinates, axis=1, keepdims=True)
    return coordinates.astype(np.float32), np.asarray(faces, dtype=np.int32)


def _save_surface(path: Path, coordinates: np.ndarray, triangles: np.ndarray) -> Path:
    image = GiftiImage()
    image.add_gifti_data_array(GiftiDataArray(coordinates, intent="NIFTI_INTENT_POINTSET"))
    image.add_gifti_data_array(GiftiDataArray(triangles, intent="NIFTI_INTENT_TRIANGLE"))
    nib.save(image, str(path))
    return path


def _mesh_inputs(tmp_path: Path, **changes) -> InputsConfig:
    values = {
        "source_manifest": tmp_path / "m.yaml",
        "timeseries": tmp_path / "s.dtseries.nii",
        "representation": "full",
        "domain": "surface",
        "space": "fsnative",
        "smoothing_mm": 0,
        "surfaces": (tmp_path / "L.surf.gii", tmp_path / "R.surf.gii"),
    }
    values.update(changes)
    return InputsConfig(**values)


def test_intermediate_mesh_needs_resampling_geometry(tmp_path: Path) -> None:
    pairs = (tmp_path / "L.gii", tmp_path / "R.gii")
    with pytest.raises(ValueError, match="32k fsLR"):
        validate_config(
            _config(tmp_path, _mesh_inputs(tmp_path), _options(), surface_mesh="fsLR32k")
        )
    validate_config(_config(tmp_path, _mesh_inputs(tmp_path), _options(), surface_mesh="target"))
    complete = _mesh_inputs(tmp_path, spheres=pairs, mesh_spheres=pairs, mesh_labels=pairs)
    validate_config(_config(tmp_path, complete, _options(), surface_mesh="fsLR32k"))
    with pytest.raises(ValueError, match="surface gradient mesh"):
        validate_config(_config(tmp_path, complete, _options(), surface_mesh="fsLR59k"))


def test_surface_mesh_is_a_surface_setting(tmp_path: Path) -> None:
    surface = _mesh_inputs(tmp_path)
    assert scientific_settings(_config(tmp_path, surface, _options()))["surface_mesh"] == "target"
    assert (
        scientific_settings(_config(tmp_path, surface, _options(), surface_mesh="fsLR32k"))[
            "surface_mesh"
        ]
        == "fsLR32k"
    )
    volume = InputsConfig(
        source_manifest=tmp_path / "m.yaml",
        timeseries=tmp_path / "s.nii",
        representation="full",
        domain="volume",
        space="MNITest",
        smoothing_mm=0,
        mask=tmp_path / "mask.nii",
    )
    assert scientific_settings(
        _config(tmp_path, volume, _options(), surface_mesh="fsLR32k")
    ) == scientific_settings(_config(tmp_path, volume, _options(), surface_mesh="target"))


@requires_workbench
def test_intermediate_mesh_gradient_returns_to_the_target_layout(tmp_path: Path) -> None:
    target_coordinates, target_triangles = _icosphere(4)
    mesh_coordinates, mesh_triangles = _icosphere(3)
    vertices = len(target_coordinates)
    surfaces, spheres, mesh_spheres, mesh_labels = [], [], [], []
    for hemi in ("L", "R"):
        # On a sphere the midthickness and registration sphere coincide.
        surface = _save_surface(tmp_path / f"{hemi}.surf.gii", target_coordinates, target_triangles)
        surfaces.append(surface)
        spheres.append(surface)
        mesh_spheres.append(
            _save_surface(tmp_path / f"{hemi}.32k.surf.gii", mesh_coordinates, mesh_triangles)
        )
        label = tmp_path / f"{hemi}.label.gii"
        nib.save(
            GiftiImage(darrays=[GiftiDataArray(np.ones(len(mesh_coordinates), np.int32))]),
            str(label),
        )
        mesh_labels.append(label)

    # Two signals meet at the equator; a polar cap is silent like a medial wall.
    rng = np.random.default_rng(5)
    frames = 80
    hemisphere = rng.standard_normal((frames, vertices)).astype(np.float32)
    shared = rng.standard_normal((frames, 2)).astype(np.float32) * 3
    z = target_coordinates[:, 2]
    hemisphere[:, z >= 0] += shared[:, :1]
    hemisphere[:, z < 0] += shared[:, 1:]
    silent = z < -90
    hemisphere[:, silent] = 0
    data = np.concatenate((hemisphere, hemisphere), axis=1)
    axis = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(vertices), vertices, name="CIFTI_STRUCTURE_CORTEX_LEFT"
    ) + nib.cifti2.BrainModelAxis.from_surface(
        np.arange(vertices), vertices, name="CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    series = nib.cifti2.SeriesAxis(0.0, 2.0, frames, unit="SECOND")
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
        surfaces=tuple(surfaces),
        spheres=tuple(spheres),
        mesh_spheres=tuple(mesh_spheres),
        mesh_labels=tuple(mesh_labels),
    )

    result = _execute(_config(tmp_path, inputs, _options(), surface_mesh="fsLR32k"))

    image = nib.load(str(result["gradient"]))
    assert image.header.get_axis(1) == axis
    assert isinstance(image.header.get_axis(0), nib.cifti2.ScalarAxis)
    gradient = np.asarray(image.dataobj)[0, :vertices]
    assert np.all(gradient[silent] == 0)
    band = np.abs(z) < 20
    away = (np.abs(z) > 50) & ~silent
    assert gradient[band].mean() > 2 * gradient[away].mean()
    mesh = yaml.safe_load(result["manifest"].read_text())["geometry"]
    assert mesh["surface_mesh"] == "fsLR32k"
    assert mesh["surface_anatomy"] == "participant"
    assert mesh["intermediate_mesh"]["series_resampling"] == "ADAP_BARY_AREA"
    # The silent cap leaves some 32k vertices without valid target coverage.
    intermediate = tmp_path / "work" / "fsLR32k" / "den-32k_desc-fcGradient_stat.dscalar.nii"
    assert 0 < nib.load(str(intermediate)).shape[1] < 2 * len(mesh_coordinates)


def test_fslr_resampling_geometry_skips_empty_placeholders(tmp_path: Path) -> None:
    from nro.engine.templates import FSLR_32K_VERTEX_COUNT, find_fslr_32k_resampling_geometry

    placeholder, populated = tmp_path / "placeholder", tmp_path / "populated"
    for root in (placeholder, populated):
        (root / "tpl-fsLR").mkdir(parents=True)
    sphere_name = "tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere.surf.gii"
    label_name = "tpl-fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii"
    (placeholder / "tpl-fsLR" / sphere_name).touch()
    (placeholder / "tpl-fsLR" / label_name).touch()
    values = np.ones(FSLR_32K_VERTEX_COUNT, dtype=np.float32)
    for name in (sphere_name, label_name):
        nib.save(GiftiImage(darrays=[GiftiDataArray(values)]), str(populated / "tpl-fsLR" / name))

    assert find_fslr_32k_resampling_geometry(hemi="L", roots=(placeholder, populated)) == (
        populated / "tpl-fsLR" / sphere_name,
        populated / "tpl-fsLR" / label_name,
    )
    with pytest.raises(FileNotFoundError, match="nro setup"):
        find_fslr_32k_resampling_geometry(hemi="L", roots=(placeholder,))
