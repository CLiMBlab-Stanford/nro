"""Shared Connectome Workbench scene and executable helpers."""

from __future__ import annotations

import base64
import lzma
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from nro.orchestration.runner import Runner


SURFACE_TYPES = ("pial", "midthickness", "white", "inflated")
SURFACE_SCENE_TEMPLATE = Path(__file__).with_name("workbench_montage.scene.in")
VOLUME_SCENE_TEMPLATE = Path(__file__).with_name("volume_workbench.scene.xz.b85")


def resolve_workbench_command(configured: str | Path) -> str:
    """Return an executable ``wb_command`` path or report its absence."""
    executable = Path(configured).expanduser()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"Connectome Workbench executable is not available: {executable}")
    return str(executable)


def surface_family(
    path: Path,
    *,
    required: Iterable[str] = SURFACE_TYPES,
) -> dict[str, Path | None]:
    """Resolve display surfaces sharing one anatomical filename stem."""
    path = Path(path)
    for kind in SURFACE_TYPES:
        token = f"_{kind}.surf.gii"
        if path.name.endswith(token):
            stem = path.name[: -len(token)]
            break
    else:
        raise ValueError(f"Cannot infer a surface family from {path}")

    family = {kind: path.with_name(f"{stem}_{kind}.surf.gii") for kind in SURFACE_TYPES}
    required_kinds = tuple(required)
    unknown = set(required_kinds).difference(SURFACE_TYPES)
    if unknown:
        raise ValueError(f"Unknown required surface types: {sorted(unknown)}")
    missing = [str(family[kind]) for kind in required_kinds if not family[kind].is_file()]
    if missing:
        raise FileNotFoundError("Missing display surfaces: " + ", ".join(missing))
    return {kind: candidate if candidate.is_file() else None for kind, candidate in family.items()}


def surface_kind(path: Path) -> str:
    """Read a display-surface type from an anatomical or packaged filename."""
    for kind in SURFACE_TYPES:
        if path.name.endswith(f"_{kind}.surf.gii") or (
            f"_desc-{kind}_surface.surf.gii" in path.name
        ):
            return kind
    raise ValueError(f"Unrecognized scene surface: {path}")


def surface_inventory(paths: Iterable[Path]) -> dict[tuple[str, str], Path]:
    """Index a complete bilateral display-surface collection."""
    inventory = {}
    for path in map(Path, paths):
        hemisphere = "L" if "_hemi-L_" in path.name else "R" if "_hemi-R_" in path.name else None
        if hemisphere is None:
            raise ValueError(f"Scene surface lacks a hemisphere entity: {path}")
        key = hemisphere, surface_kind(path)
        if key in inventory:
            raise ValueError(f"Duplicate scene surface for {key}: {path}")
        inventory[key] = path
    expected = {(hemisphere, kind) for hemisphere in ("L", "R") for kind in SURFACE_TYPES}
    if set(inventory) != expected:
        missing = sorted(expected.difference(inventory))
        extra = sorted(set(inventory).difference(expected))
        raise ValueError(
            f"A surface scene requires four surfaces per hemisphere; "
            f"missing={missing}, extra={extra}"
        )
    return inventory


def surface_spec_entries(inventory: dict[tuple[str, str], Path]) -> str:
    """Render Workbench spec-file entries for packaged surfaces."""
    entries = []
    for index, ((hemisphere, _kind), path) in enumerate(inventory.items()):
        structure = "CORTEX_LEFT" if hemisphere == "L" else "CORTEX_RIGHT"
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


def packaged_surface_paths(directory: Path, prefix: str) -> tuple[Path, ...]:
    """Return the ordered bilateral display-surface destinations."""
    return tuple(
        Path(directory) / f"{prefix}_hemi-{hemisphere}_desc-{kind}_surface.surf.gii"
        for hemisphere in ("L", "R")
        for kind in SURFACE_TYPES
    )


def package_surface_families(
    directory: Path,
    prefix: str,
    surface_paths: tuple[Path, ...],
    *,
    runner: Runner | None = None,
    executable: str | None = None,
) -> tuple[Path, ...]:
    """Copy bilateral display surfaces, generating optional surfaces if absent."""
    if len(surface_paths) != 2:
        raise ValueError("Surface packaging requires left and right geometry")
    directory = Path(directory)
    families = tuple(
        surface_family(Path(path), required=("pial", "white")) for path in surface_paths
    )
    destinations = packaged_surface_paths(directory, prefix)
    inventory = dict(
        zip(
            ((hemisphere, kind) for hemisphere in ("L", "R") for kind in SURFACE_TYPES),
            destinations,
        )
    )
    for hemisphere, family in zip(("L", "R"), families):
        for kind, source in family.items():
            if source is not None:
                shutil.copyfile(source, inventory[(hemisphere, kind)])

        if family["midthickness"] is None:
            if runner is None or executable is None:
                raise FileNotFoundError(f"Missing midthickness surface for hemisphere {hemisphere}")
            runner.run_child(
                [
                    executable,
                    "-surface-average",
                    str(inventory[(hemisphere, "midthickness")]),
                    "-surf",
                    str(inventory[(hemisphere, "pial")]),
                    "-surf",
                    str(inventory[(hemisphere, "white")]),
                ]
            )
        if family["inflated"] is None:
            if runner is None or executable is None:
                raise FileNotFoundError(f"Missing inflated surface for hemisphere {hemisphere}")
            very_inflated = (
                directory / f".{prefix}_hemi-{hemisphere}_desc-veryInflated_surface.surf.gii"
            )
            try:
                runner.run_child(
                    [
                        executable,
                        "-surface-generate-inflated",
                        str(inventory[(hemisphere, "midthickness")]),
                        str(inventory[(hemisphere, "inflated")]),
                        str(very_inflated),
                    ]
                )
            finally:
                very_inflated.unlink(missing_ok=True)
    return destinations


def decode_scene_template(path: Path) -> str:
    """Decode an LZMA-compressed, base85-encoded Workbench scene template."""
    encoded = "".join(Path(path).read_text(encoding="ascii").split())
    return lzma.decompress(base64.b85decode(encoded)).decode("utf-8")
