"""Construct relocatable Workbench scenes for microparcellation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nro.orchestration.runner import Runner

from nro.engine.workbench import (
    SURFACE_SCENE_TEMPLATE,
    SURFACE_TYPES,
    VOLUME_SCENE_TEMPLATE,
    decode_scene_template,
    package_surface_families,
    packaged_surface_paths,
    surface_family,
    surface_spec_entries,
)


def write_workbench_scene(
    output_dir: Path,
    prefix: str,
    surface_paths: tuple[Path, ...],
    dlabel_path: Path,
    pconn_path: Path,
    *,
    runner: Runner | None = None,
    executable: str | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Package a relocatable montage with four loaded surfaces per hemisphere."""
    scene_path, packaged_surfaces, source_surfaces = surface_scene_output_paths(
        output_dir, prefix, surface_paths
    )
    packaged_surfaces = package_surface_families(
        output_dir,
        prefix,
        surface_paths,
        runner=runner,
        executable=executable,
    )
    packaged = dict(
        zip(
            ((hemisphere, kind) for hemisphere in ("L", "R") for kind in SURFACE_TYPES),
            packaged_surfaces,
        )
    )

    left_midthickness = packaged[("L", "midthickness")]
    right_midthickness = packaged[("R", "midthickness")]

    replacements = {
        "{{LEFT_SURFACE}}": left_midthickness.name,
        "{{LEFT_SURFACE_NAME}}": left_midthickness.name,
        "{{RIGHT_SURFACE}}": right_midthickness.name,
        "{{RIGHT_SURFACE_NAME}}": right_midthickness.name,
        "{{SURFACE_SPEC_ENTRIES}}": surface_spec_entries(packaged),
        "{{DLABEL}}": dlabel_path.name,
        "{{DLABEL_NAME}}": dlabel_path.name,
        "{{PCONN}}": pconn_path.name,
    }
    scene = SURFACE_SCENE_TEMPLATE.read_text()
    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    if "{{" in scene or "}}" in scene:
        raise RuntimeError("Unresolved placeholder in Workbench scene template")

    scene_path.write_text(scene)
    return scene_path, packaged_surfaces


def surface_scene_output_paths(
    output_dir: Path,
    prefix: str,
    surface_paths: tuple[Path, ...],
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    """Return the scene, packaged outputs, and source display surfaces."""
    if len(surface_paths) != 2:
        raise ValueError("Workbench montage output requires left and right surfaces")

    families = tuple(
        surface_family(Path(path), required=("pial", "white")) for path in surface_paths
    )
    source_surfaces = tuple(
        source for family in families for source in family.values() if source is not None
    )
    packaged_surfaces = packaged_surface_paths(output_dir, prefix)
    scene_path = output_dir / f"{prefix}_desc-microparcellation_scene.scene"
    return scene_path, packaged_surfaces, source_surfaces


def write_volume_workbench_scene(
    output_dir: Path,
    prefix: str,
    label_volume_path: Path,
    dlabel_path: Path,
    pconn_path: Path,
) -> Path:
    """Package a relocatable Workbench volume scene with parcel connectivity."""
    scene = decode_scene_template(VOLUME_SCENE_TEMPLATE)
    replacements = {
        "{{VOLUME_LABEL}}": label_volume_path.name,
        "{{DLABEL}}": dlabel_path.name,
        "{{PCONN}}": pconn_path.name,
    }
    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    if "{{" in scene or "}}" in scene:
        raise RuntimeError("Unresolved placeholder in Workbench volume scene template")
    scene_path = output_dir / f"{prefix}_desc-microparcellation_scene.scene"
    scene_path.write_text(scene, encoding="utf-8")
    return scene_path
