"""Render finite maps and requested connectivity seeds from Workbench scenes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import numpy as np
import yaml

from nro.engine.cifti import indexed_cifti_sidecar
from nro.engine.io import atomic_output_path, atomic_write_text
from nro.orchestration.registry import RegistryLock

SEED_MODULES = frozenset({"dynconn", "microparcellation"})
FINITE_MAP_MODULES = frozenset({"firstlevels", "microparcellation", "networks"})
RENDER_FORMATS = ("png", "jpg", "tiff")


@dataclass(frozen=True)
class RenderTarget:
    """Identify one file map to display and describe in the render manifest."""

    module: str
    path: Path
    map_index: int
    map_name: str
    metadata: dict[str, object]
    seed: dict[str, object] | None = None
    source_path: Path | None = None


def read_seed_file(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, ...]]:
    """Read subject-specific T1w world coordinates from a YAML seed file."""

    path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read seed file: {path}") from error
    if not isinstance(document, dict) or set(document) != {"projects"}:
        raise ValueError("Seed files must contain one projects mapping")
    projects = document["projects"]
    if not isinstance(projects, dict):
        raise ValueError("Seed-file projects must be a mapping")
    result: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    for project, subjects in projects.items():
        if not isinstance(project, str) or not project or not isinstance(subjects, dict):
            raise ValueError("Each seed-file project must contain a subject mapping")
        for participant, values in subjects.items():
            if not isinstance(participant, str) or not participant or not isinstance(values, list):
                raise ValueError("Each seed-file subject must contain a seed list")
            coordinates = []
            for value in values:
                raw = value.get("xyz_mm") if isinstance(value, dict) else value
                if isinstance(value, dict) and set(value) != {"xyz_mm"}:
                    raise ValueError("Seed records may contain only xyz_mm")
                if (
                    not isinstance(raw, (list, tuple))
                    or len(raw) != 3
                    or any(
                        isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw
                    )
                ):
                    raise ValueError("Each seed must contain three finite xyz_mm coordinates")
                coordinate = np.asarray(raw, dtype=np.float64)
                if not np.all(np.isfinite(coordinate)):
                    raise ValueError("Each seed must contain three finite xyz_mm coordinates")
                coordinates.append(coordinate)
            result[(project, participant.removeprefix("sub-"))] = tuple(coordinates)
    return result


def _scene_path(scene: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else scene.parent / path


def _map_metadata(path: Path, count: int) -> list[dict[str, object]]:
    sidecar = indexed_cifti_sidecar(path) if path.name.endswith(".nii") else None
    if sidecar is not None and sidecar.is_file():
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
            values = document["IndexToMetadata"]
            records = [values[str(index)] for index in range(count)]
            if all(isinstance(record, dict) for record in records):
                return [dict(record) for record in records]
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return [{} for _ in range(count)]


def finite_targets(scene: Path, manifest: dict) -> tuple[RenderTarget, ...]:
    """Enumerate named finite maps while excluding time series and connectivity matrices."""

    import nibabel as nib

    sources = [
        source
        for source in manifest.get("sources", ())
        if isinstance(source, dict) and source.get("role") == "derivative"
    ]
    microparcellation_has_dlabel = any(
        source.get("module") == "microparcellation"
        and str(source.get("scene_path") or "").endswith(".dlabel.nii")
        for source in sources
    )
    targets = []
    for source in sources:
        module = str(source.get("module") or "")
        if module not in FINITE_MAP_MODULES:
            continue
        path = _scene_path(scene, str(source.get("scene_path") or ""))
        if not path.is_file() or path.name.endswith((".pconn.nii", ".dtseries.nii")):
            continue
        if path.name.endswith((".dscalar.nii", ".dlabel.nii")):
            image = nib.load(str(path))
            axis = image.header.get_axis(0)
            if not isinstance(axis, (nib.cifti2.ScalarAxis, nib.cifti2.LabelAxis)):
                continue
            names = [str(name) for name in axis.name]
            metadata = _map_metadata(path, len(names))
        elif path.name.endswith((".func.gii", ".shape.gii", ".label.gii")):
            image = nib.load(str(path))
            names = [
                str(array.meta.get("Name") or f"map-{index + 1}")
                for index, array in enumerate(image.darrays)
            ]
            metadata = [{} for _ in names]
        elif path.name.endswith((".nii", ".nii.gz")) and (
            module == "microparcellation" or "statmap" in path.name
        ):
            if module == "microparcellation" and microparcellation_has_dlabel:
                continue
            image = nib.load(str(path))
            count = image.shape[3] if len(image.shape) == 4 else 1
            names = [f"map-{index + 1}" for index in range(count)]
            metadata = [{} for _ in names]
        else:
            continue
        targets.extend(
            RenderTarget(module, path, index, name, metadata[index])
            for index, name in enumerate(names)
        )
    return tuple(targets)


def _surface_coordinates(scene: Path, manifest: dict) -> dict[str, np.ndarray]:
    import nibabel as nib

    result = {}
    for source in manifest.get("sources", ()):
        if not isinstance(source, dict) or source.get("role") != "display_surface":
            continue
        path = _scene_path(scene, str(source.get("scene_path") or ""))
        if "_midthickness.surf.gii" not in path.name or not path.is_file():
            continue
        structure = (
            "CIFTI_STRUCTURE_CORTEX_LEFT"
            if "_hemi-L_" in path.name
            else ("CIFTI_STRUCTURE_CORTEX_RIGHT" if "_hemi-R_" in path.name else None)
        )
        if structure:
            result[structure] = np.asarray(nib.load(str(path)).darrays[0].data, dtype=np.float64)
    return result


def _nearest_brainordinate(
    axis, xyz: np.ndarray, surfaces: dict[str, np.ndarray]
) -> dict[str, object]:
    import nibabel as nib

    candidates: list[tuple[float, dict[str, object]]] = []
    for structure, structure_slice, model in axis.iter_structures():
        name = str(structure)
        vertices = np.asarray(model.vertex)
        if vertices.size and np.all(vertices >= 0) and name in surfaces:
            coordinates = surfaces[name][vertices]
            local = int(np.argmin(np.sum((coordinates - xyz) ** 2, axis=1)))
            distance = float(np.linalg.norm(coordinates[local] - xyz))
            candidates.append(
                (
                    distance,
                    {
                        "kind": "surface",
                        "structure": name,
                        "brainordinate_index": int(structure_slice.start + local),
                        "vertex": int(vertices[local]),
                        "resolved_xyz_mm": coordinates[local].tolist(),
                    },
                )
            )
        voxels = np.asarray(model.voxel)
        if voxels.ndim == 2 and voxels.shape[1:] == (3,):
            voxels = voxels[np.all(voxels >= 0, axis=1)]
        else:
            voxels = np.empty((0, 3), dtype=np.int64)
        if voxels.size:
            coordinates = nib.affines.apply_affine(axis.affine, voxels)
            local = int(np.argmin(np.sum((coordinates - xyz) ** 2, axis=1)))
            distance = float(np.linalg.norm(coordinates[local] - xyz))
            candidates.append(
                (
                    distance,
                    {
                        "kind": "volume",
                        "structure": name,
                        "brainordinate_index": int(structure_slice.start + local),
                        "voxel_ijk": voxels[local].astype(int).tolist(),
                        "resolved_xyz_mm": coordinates[local].tolist(),
                    },
                )
            )
    if not candidates:
        raise ValueError(
            "Connectivity image has no brainordinates that can be resolved in T1w space"
        )
    distance, record = min(candidates, key=lambda item: item[0])
    return {**record, "distance_mm": distance}


def _nearest_parcel(axis, xyz: np.ndarray, surfaces: dict[str, np.ndarray]) -> dict[str, object]:
    import nibabel as nib

    candidates: list[tuple[float, dict[str, object]]] = []
    for index in range(len(axis)):
        name, voxels, vertices = axis[index]
        for structure, indices in vertices.items():
            structure_name = str(structure)
            indices = np.asarray(indices, dtype=np.int64)
            if indices.size and structure_name in surfaces:
                coordinates = surfaces[structure_name][indices]
                local = int(np.argmin(np.sum((coordinates - xyz) ** 2, axis=1)))
                distance = float(np.linalg.norm(coordinates[local] - xyz))
                candidates.append(
                    (
                        distance,
                        {
                            "kind": "surface",
                            "structure": structure_name,
                            "parcel_index": index,
                            "parcel": str(name),
                            "vertex": int(indices[local]),
                            "resolved_xyz_mm": coordinates[local].tolist(),
                        },
                    )
                )
        voxels = np.asarray(voxels)
        if voxels.ndim == 2 and voxels.shape[1:] == (3,) and len(voxels):
            coordinates = nib.affines.apply_affine(axis.affine, voxels)
            local = int(np.argmin(np.sum((coordinates - xyz) ** 2, axis=1)))
            distance = float(np.linalg.norm(coordinates[local] - xyz))
            candidates.append(
                (
                    distance,
                    {
                        "kind": "volume",
                        "structure": "parcel-volume",
                        "parcel_index": index,
                        "parcel": str(name),
                        "voxel_ijk": voxels[local].astype(int).tolist(),
                        "resolved_xyz_mm": coordinates[local].tolist(),
                    },
                )
            )
    if not candidates:
        raise ValueError("Parcel image has no locations that can be resolved in T1w space")
    distance, record = min(candidates, key=lambda item: item[0])
    return {**record, "distance_mm": distance}


def _seed_name(xyz: np.ndarray) -> str:
    def component(axis: str, value: float) -> str:
        sign = "m" if value < 0 else "p"
        number = f"{abs(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"{axis}{sign}{number}"

    return "seed-" + "_".join(component(axis, value) for axis, value in zip("xyz", xyz))


def _write_dynconn_seed(source: Path, output: Path, xyz: np.ndarray, surfaces) -> dict[str, object]:
    import nibabel as nib

    image = nib.load(str(source))
    brain_axis = image.header.get_axis(1)
    if not isinstance(brain_axis, nib.cifti2.BrainModelAxis):
        raise ValueError(f"Expected a dense time series: {source}")
    resolved = _nearest_brainordinate(brain_axis, xyz, surfaces)
    seed = np.asarray(image.dataobj[:, resolved["brainordinate_index"]], dtype=np.float64)
    seed -= seed.mean()
    seed_norm = float(np.linalg.norm(seed))
    correlations = np.zeros(image.shape[1], dtype=np.float32)
    if seed_norm > 0:
        for start in range(0, image.shape[1], 4096):
            stop = min(start + 4096, image.shape[1])
            block = np.asarray(image.dataobj[:, start:stop], dtype=np.float64)
            block -= block.mean(axis=0, keepdims=True)
            denominator = np.linalg.norm(block, axis=0) * seed_norm
            valid = denominator > 0
            values = correlations[start:stop]
            values[valid] = (seed @ block[:, valid] / denominator[valid]).astype(np.float32)
            correlations[start:stop] = values
    rendered = nib.Cifti2Image(
        correlations[None, :],
        header=nib.cifti2.Cifti2Header.from_axes(
            (nib.cifti2.ScalarAxis([_seed_name(xyz)]), brain_axis)
        ),
        dtype=np.float32,
    )
    with atomic_output_path(output) as staged:
        nib.save(rendered, staged)
    return resolved


def _write_volume_dynconn_seed(
    source: Path, output: Path, xyz: np.ndarray, _surfaces
) -> dict[str, object]:
    import nibabel as nib

    image = nib.load(str(source))
    if len(image.shape) != 4:
        raise ValueError(f"Expected a four-dimensional volume time series: {source}")
    spatial_shape = tuple(int(value) for value in image.shape[:3])
    data = np.asanyarray(image.dataobj)
    voxels = np.indices(spatial_shape, dtype=np.int32).reshape(3, -1).T
    coordinates = nib.affines.apply_affine(image.affine, voxels)
    order = np.argsort(np.sum((coordinates - xyz) ** 2, axis=1))
    selected = None
    for start in range(0, len(order), 256):
        candidates = voxels[order[start : start + 256]]
        series = np.asarray(
            data[candidates[:, 0], candidates[:, 1], candidates[:, 2], :],
            dtype=np.float64,
        )
        series -= series.mean(axis=1, keepdims=True)
        eligible = np.all(np.isfinite(series), axis=1) & (np.linalg.norm(series, axis=1) > 0)
        if np.any(eligible):
            selected = int(order[start + int(np.flatnonzero(eligible)[0])])
            break
    if selected is None:
        raise ValueError(f"Volume time series contains no eligible seed voxels: {source}")
    voxel = voxels[selected]
    resolved_xyz = coordinates[selected]
    seed = np.asarray(data[tuple(voxel) + (slice(None),)], dtype=np.float64)
    seed -= seed.mean()
    seed_norm = float(np.linalg.norm(seed))
    correlations = np.zeros(spatial_shape, dtype=np.float32)
    if seed_norm > 0:
        rows_per_block = max(1, 4096 // spatial_shape[0])
        for z_index in range(spatial_shape[2]):
            for y_start in range(0, spatial_shape[1], rows_per_block):
                y_stop = min(y_start + rows_per_block, spatial_shape[1])
                block = np.asarray(
                    data[:, y_start:y_stop, z_index, :],
                    dtype=np.float64,
                )
                flat = block.reshape(-1, image.shape[3])
                flat -= flat.mean(axis=1, keepdims=True)
                denominator = np.linalg.norm(flat, axis=1) * seed_norm
                values = np.zeros(len(flat), dtype=np.float32)
                valid = denominator > 0
                values[valid] = (flat[valid] @ seed / denominator[valid]).astype(np.float32)
                correlations[:, y_start:y_stop, z_index] = values.reshape(
                    spatial_shape[0], y_stop - y_start
                )
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    with atomic_output_path(output) as staged:
        nib.save(nib.Nifti1Image(correlations, image.affine, header=header), staged)
    return {
        "kind": "volume",
        "structure": "volume",
        "voxel_ijk": voxel.astype(int).tolist(),
        "resolved_xyz_mm": resolved_xyz.tolist(),
        "distance_mm": float(np.linalg.norm(resolved_xyz - xyz)),
    }


def _write_pconn_seed(source: Path, output: Path, xyz: np.ndarray, surfaces) -> dict[str, object]:
    import nibabel as nib

    image = nib.load(str(source))
    axes = image.header.get_axis(0), image.header.get_axis(1)
    if not all(isinstance(axis, nib.cifti2.ParcelsAxis) for axis in axes):
        raise ValueError(f"Expected parcel connectivity: {source}")
    resolved = _nearest_parcel(axes[1], xyz, surfaces)
    values = np.asarray(image.dataobj[int(resolved["parcel_index"]), :], dtype=np.float32)
    rendered = nib.Cifti2Image(
        values[None, :],
        header=nib.cifti2.Cifti2Header.from_axes(
            (nib.cifti2.ScalarAxis([_seed_name(xyz)]), axes[1])
        ),
        dtype=np.float32,
    )
    with atomic_output_path(output) as staged:
        nib.save(rendered, staged)
    return resolved


def seed_targets(
    scene: Path,
    manifest: dict,
    coordinates: Iterable[np.ndarray],
    temporary: Path,
) -> tuple[RenderTarget, ...]:
    """Materialize requested seed maps for dynamic and parcel connectivity."""

    coordinates = tuple(coordinates)
    if not coordinates:
        return ()
    space = str(manifest.get("space") or "")
    if space not in {"fsnative", "T1w"}:
        raise ValueError(
            f"T1w seed coordinates cannot yet be resolved into space-{space}; "
            "select fsnative or T1w"
        )
    surfaces = _surface_coordinates(scene, manifest)
    targets = []
    for source in manifest.get("sources", ()):
        if not isinstance(source, dict) or source.get("role") != "derivative":
            continue
        module = str(source.get("module") or "")
        path = _scene_path(scene, str(source.get("scene_path") or ""))
        if module == "dynconn" and path.name.endswith(".dtseries.nii"):
            writer = _write_dynconn_seed
            suffix = "dscalar.nii"
        elif module == "dynconn" and path.name.endswith((".nii", ".nii.gz")):
            writer = _write_volume_dynconn_seed
            suffix = "nii.gz"
        elif module == "microparcellation" and path.name.endswith(".pconn.nii"):
            writer = _write_pconn_seed
            suffix = "pscalar.nii"
        else:
            continue
        for index, xyz in enumerate(coordinates):
            name = _seed_name(xyz)
            output = temporary / f"{module}_{index + 1}_{name}.{suffix}"
            resolved = writer(path, output, xyz, surfaces)
            seed = {"requested_xyz_mm": xyz.tolist(), **resolved}
            targets.append(RenderTarget(module, output, 0, name, {}, seed, path))
    return tuple(targets)


def _set_overlay(scene: Path, output: Path, target: RenderTarget) -> None:
    root = ElementTree.parse(scene).getroot()
    selected = 0
    for overlays in root.iter("ObjectArray"):
        if overlays.get("Name") != "m_overlays":
            continue
        element = next((child for child in overlays if child.tag == "Element"), None)
        overlay = next(
            (
                child
                for child in (() if element is None else element)
                if child.get("Class") == "Overlay"
            ),
            None,
        )
        if overlay is None:
            continue
        values = {child.get("Name"): child for child in overlay if child.tag == "Object"}
        required = {
            "m_enabled",
            "selectedMapFileNameWithPath",
            "selectedMapFile",
            "selectedMapName",
            "selectedMapIndex",
        }
        if not required <= set(values):
            continue
        values["m_enabled"].text = "true"
        values["selectedMapFileNameWithPath"].text = str(target.path)
        values["selectedMapFile"].text = target.path.name
        values["selectedMapName"].text = target.map_name
        values["selectedMapIndex"].text = str(target.map_index)
        selected += 1
    if not selected:
        raise ValueError(f"Scene contains no display overlay: {scene}")
    output.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ElementTree.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )


def _safe_name(value: str, *, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-") or "map"
    if len(cleaned) <= limit:
        return cleaned
    import hashlib

    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{cleaned[: limit - 11]}-{digest}"


def _replace_render_directory(staging: Path, destination: Path, scene_id: str) -> None:
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace a linked render directory: {destination}")
    if destination.exists():
        manifest = destination / "render_manifest.yaml"
        try:
            document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(
                f"Refusing to replace an unmanaged render directory: {destination}"
            ) from error
        if (
            not isinstance(document, dict)
            or document.get("kind") != "nro-workbench-render"
            or document.get("scene_id") != scene_id
        ):
            raise ValueError(f"Refusing to replace an unmanaged render directory: {destination}")
        shutil.rmtree(destination)
    os.replace(staging, destination)


def render_scene(
    scene: Path,
    *,
    wb_command: str | Path,
    coordinates: Iterable[np.ndarray] = (),
    width: int = 2400,
    image_format: str = "png",
    no_scene_colors: bool = False,
    warn_distance_mm: float = 10.0,
    destination: Path | None = None,
    progress=print,
) -> Path:
    """Render every finite map and requested seed map from one generated scene."""

    if width <= 0:
        raise ValueError("Render width must be positive")
    if image_format not in RENDER_FORMATS:
        raise ValueError(f"Unsupported render format: {image_format}")
    if warn_distance_mm < 0:
        raise ValueError("Seed-distance warning threshold cannot be negative")
    coordinates = tuple(np.asarray(value, dtype=np.float64) for value in coordinates)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in coordinates):
        raise ValueError("Each seed must contain three finite xyz_mm coordinates")
    scene = Path(scene).resolve()
    manifest_path = scene.parent / "scene_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != "nro-workbench-scene":
        raise ValueError(f"Scene manifest is invalid: {manifest_path}")
    scene_id = str(manifest["scene_id"])
    destination = Path(destination).resolve() if destination else scene.parent / "renders"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = RegistryLock(
        destination.parent / f".{destination.name}.lock",
        destination.parent / f".{destination.name}.lock.recovery",
    )
    with lock:
        staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            finite = finite_targets(scene, manifest)
            seeds = seed_targets(scene, manifest, coordinates, staging)
            source_modules = {
                str(source.get("module"))
                for source in manifest.get("sources", ())
                if isinstance(source, dict) and source.get("role") == "derivative"
            }
            anatomy_only = source_modules == {"anat"}
            targets: tuple[RenderTarget | None, ...] = (*finite, *seeds)
            if anatomy_only and not targets:
                targets = (None,)
            skipped = sorted(
                module for module in source_modules.intersection(SEED_MODULES) if not coordinates
            )
            records = []
            for number, target in enumerate(targets, start=1):
                if target is None:
                    label = "anatomy"
                    render_scene_path = scene
                else:
                    label = "-".join(
                        (
                            target.module,
                            _safe_name(target.path.name),
                            str(target.map_index + 1),
                            _safe_name(target.map_name),
                        )
                    )
                    registered = staging / f".registered-{number}.scene"
                    if target.seed is not None:
                        result = subprocess.run(
                            [
                                str(wb_command),
                                "-scene-file-update",
                                str(scene),
                                str(registered),
                                "1",
                                "-error",
                                "-data-file-add",
                                str(target.path),
                            ],
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        if result.returncode:
                            raise RuntimeError((result.stderr or result.stdout).strip())
                        source = registered
                    else:
                        source = scene
                    render_scene_path = staging / f".render-{number}.scene"
                    _set_overlay(source, render_scene_path, target)
                image = staging / f"{number:04d}_{label}.{image_format}"
                command = [
                    str(wb_command),
                    "-scene-capture-image",
                    str(render_scene_path),
                    "1",
                    str(image),
                    "-size-width",
                    str(width),
                    "-renderer",
                    "OSMesa",
                ]
                if no_scene_colors:
                    command.append("-no-scene-colors")
                progress(f"Rendering {number}/{len(targets)}: {label}")
                result = subprocess.run(command, text=True, capture_output=True, check=False)
                if result.returncode or not image.is_file():
                    message = (result.stderr or result.stdout or "no image was created").strip()
                    raise RuntimeError(f"Workbench could not render {label}: {message}")
                record = {
                    "image": image.name,
                    "module": target.module if target else "anat",
                    "source": str(target.source_path or target.path) if target else None,
                    "map_index": target.map_index if target else None,
                    "map_name": target.map_name if target else "anatomy",
                    "metadata": target.metadata if target else {},
                }
                if target and target.seed is not None:
                    record["seed"] = target.seed
                    if float(target.seed["distance_mm"]) > warn_distance_mm:
                        record["warning"] = (
                            f"Resolved seed is {target.seed['distance_mm']:.2f} mm from its requested coordinate"
                        )
                        progress(f"Warning: {record['warning']}")
                records.append(record)
            for path in staging.glob(".*.scene"):
                path.unlink()
            for target in seeds:
                target.path.unlink(missing_ok=True)
            atomic_write_text(
                staging / "render_manifest.yaml",
                yaml.safe_dump(
                    {
                        "kind": "nro-workbench-render",
                        "format_version": 1,
                        "scene_id": scene_id,
                        "source_scene": str(scene),
                        "width_pixels": width,
                        "image_format": image_format,
                        "skipped_seed_modules": skipped,
                        "renders": records,
                    },
                    sort_keys=False,
                ),
            )
            _replace_render_directory(staging, destination, scene_id)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination / "render_manifest.yaml"
