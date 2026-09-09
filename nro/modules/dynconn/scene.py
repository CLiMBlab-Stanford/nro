"""Build relocatable Workbench dynamic-connectivity scenes."""

from __future__ import annotations

import shutil
from pathlib import Path

from nro.engine.io import atomic_write_text
from nro.engine.workbench import SURFACE_TYPES, decode_scene_template, surface_family


def _template(name: str) -> str:
    return decode_scene_template(Path(__file__).with_name(name))


def surface_output_paths(directory: Path, prefix: str) -> tuple[Path, ...]:
    """Return the eight surface files packaged with a surface scene."""

    return tuple(
        Path(directory) / f"{prefix}_hemi-{hemi}_desc-{kind}_surface.surf.gii"
        for hemi in ("L", "R")
        for kind in SURFACE_TYPES
    )


def write_surface_scene(
    directory: Path, prefix: str, timeseries: Path, surfaces: tuple[Path, ...]
) -> tuple[Path, ...]:
    """Copy display geometry and write a relocatable surface scene."""

    if len(surfaces) != 2:
        raise ValueError("A surface scene requires left and right geometry")
    destinations = surface_output_paths(directory, prefix)
    replacements = {
        "{{TIMESERIES}}": timeseries.name,
        "{{DYNCONN}}": timeseries.name.removesuffix(".dtseries.nii") + ".dynconn.nii",
    }
    index = 0
    for hemi, selected in zip(("L", "R"), surfaces):
        family = surface_family(Path(selected))
        for kind in SURFACE_TYPES:
            destination = destinations[index]
            source = family[kind]
            assert source is not None
            shutil.copyfile(source, destination)
            replacements[f"{{{hemi}_{kind.upper()}}}"] = destination.name
            index += 1
    scene = _template("surface.scene.xz.b85")
    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    if "{{" in scene or "}}" in scene:
        raise RuntimeError("Unresolved placeholder in the surface scene template")
    atomic_write_text(Path(directory) / f"{prefix}_desc-dynamicConnectivity_scene.scene", scene)
    return destinations


def write_volume_scene(directory: Path, prefix: str, timeseries: Path) -> None:
    """Write a relocatable voxelwise dynamic-connectivity scene."""

    scene = _template("volume.scene.xz.b85")
    replacements = {
        "{{TIMESERIES}}": timeseries.name,
        "{{DYNCONN}}": timeseries.name.removesuffix(".nii") + ".vol_dynconn",
    }
    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    if "{{" in scene or "}}" in scene:
        raise RuntimeError("Unresolved placeholder in the volume scene template")
    atomic_write_text(Path(directory) / f"{prefix}_desc-dynamicConnectivity_scene.scene", scene)
