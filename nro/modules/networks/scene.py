"""Workbench scenes for multi-map individualized network CIFTIs."""

from __future__ import annotations

import shutil
from pathlib import Path

from nro.engine.workbench import (
    SURFACE_SCENE_TEMPLATE,
    SURFACE_TYPES,
    VOLUME_SCENE_TEMPLATE,
    decode_scene_template,
    surface_inventory,
    surface_spec_entries,
)


def write_network_scene(
    path: Path,
    *,
    domain: str,
    membership: Path,
    connectivity: Path,
    scene_surfaces: tuple[Path, ...] = (),
    label_volume: Path | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Package a relocatable scene that pages named network maps."""
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = path.name.removesuffix("_desc-networks_scene.scene")
    membership_copy = output_dir / membership.name
    if membership.resolve() != membership_copy.resolve():
        shutil.copyfile(membership, membership_copy)
    membership = membership_copy
    connectivity_copy = output_dir / f"{prefix}_connectivity.pconn.nii"
    shutil.copyfile(connectivity, connectivity_copy)
    assets: list[Path] = [membership_copy, connectivity_copy]
    if domain == "surface":
        sources = surface_inventory(scene_surfaces)
        inventory = {
            (hemi, kind): output_dir / f"{prefix}_hemi-{hemi}_desc-{kind}_surface.surf.gii"
            for hemi in ("L", "R")
            for kind in SURFACE_TYPES
        }
        for identity, destination in inventory.items():
            if sources[identity].resolve() != destination.resolve():
                shutil.copyfile(sources[identity], destination)
        assets.extend(inventory.values())
        left = inventory[("L", "midthickness")]
        right = inventory[("R", "midthickness")]
        replacements = {
            "{{LEFT_SURFACE}}": left.name,
            "{{LEFT_SURFACE_NAME}}": left.name,
            "{{RIGHT_SURFACE}}": right.name,
            "{{RIGHT_SURFACE_NAME}}": right.name,
            "{{SURFACE_SPEC_ENTRIES}}": surface_spec_entries(inventory),
            "{{DLABEL}}": membership.name,
            "{{DLABEL_NAME}}": membership.name,
            "{{PCONN}}": connectivity_copy.name,
        }
        scene = SURFACE_SCENE_TEMPLATE.read_text(encoding="utf-8")
    elif domain == "volume":
        if label_volume is None:
            raise ValueError("Volumetric network scene requires its microparcel label volume")
        label_copy = output_dir / f"{prefix}_desc-microparcellation_dseg.nii.gz"
        shutil.copyfile(label_volume, label_copy)
        assets.append(label_copy)
        scene = decode_scene_template(VOLUME_SCENE_TEMPLATE)
        replacements = {
            "{{VOLUME_LABEL}}": label_copy.name,
            "{{DLABEL}}": membership.name,
            "{{PCONN}}": connectivity_copy.name,
        }
    else:
        raise ValueError(f"Unsupported networks scene domain: {domain}")

    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    scene = scene.replace("CONNECTIVITY_DENSE_LABEL", "CONNECTIVITY_DENSE_SCALAR")
    scene = scene.replace("Microparcellation", "Individualized networks")
    if "{{" in scene or "}}" in scene:
        raise RuntimeError("Unresolved placeholder in networks Workbench scene")
    path.write_text(scene, encoding="utf-8")
    return path, tuple(assets)
