"""Workbench scenes for multi-map individualized network CIFTIs."""

from __future__ import annotations

import base64
import lzma
import shutil
from pathlib import Path

from nro.microparcellation.scene import SURFACE_SCENE_TEMPLATE, VOLUME_SCENE_TEMPLATE


def _surface_kind(path: Path) -> str:
    for kind in ("pial", "midthickness", "white", "inflated"):
        if path.name.endswith(f"_{kind}.surf.gii") or f"_desc-{kind}_surface.surf.gii" in path.name:
            return kind
    raise ValueError(f"Unrecognized scene surface: {path}")


def _surface_inventory(paths: tuple[Path, ...]) -> dict[tuple[str, str], Path]:
    inventory = {}
    for path in paths:
        hemi = "L" if "_hemi-L_" in path.name else "R" if "_hemi-R_" in path.name else None
        if hemi is None:
            raise ValueError(f"Scene surface lacks a hemisphere entity: {path}")
        inventory[(hemi, _surface_kind(path))] = path
    expected = {
        (hemi, kind)
        for hemi in ("L", "R")
        for kind in ("pial", "midthickness", "white", "inflated")
    }
    if set(inventory) != expected:
        raise ValueError("Networks surface scene requires four surfaces for each hemisphere")
    return inventory


def _surface_spec_entries(inventory: dict[tuple[str, str], Path]) -> str:
    entries = []
    for index, ((hemi, _kind), path) in enumerate(inventory.items()):
        structure = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
        entries.append(
            f'''                                    <Element Index="{index}">
                                        <Object Type="class" Class="SpecFileDataFile" Name="specFileDataFile" Version="1">
                                            <Object Type="enumeratedType" Name="dataFileType">SURFACE</Object>
                                            <Object Type="enumeratedType" Name="structure">{structure}</Object>
                                            <Object Type="pathName" Name="fileName">{path.name}</Object>
                                            <Object Type="boolean" Name="selected">true</Object>
                                        </Object>
                                    </Element>'''
        )
    return "\n".join(entries)


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
        sources = _surface_inventory(scene_surfaces)
        inventory = {
            (hemi, kind): output_dir / f"{prefix}_hemi-{hemi}_desc-{kind}_surface.surf.gii"
            for hemi in ("L", "R")
            for kind in ("pial", "midthickness", "white", "inflated")
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
            "{{SURFACE_SPEC_ENTRIES}}": _surface_spec_entries(inventory),
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
        encoded = "".join(VOLUME_SCENE_TEMPLATE.read_text(encoding="ascii").split())
        scene = lzma.decompress(base64.b85decode(encoded)).decode("utf-8")
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
