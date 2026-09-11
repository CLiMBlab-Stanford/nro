"""Shared Connectome Workbench scene and executable helpers."""

from __future__ import annotations

import base64
import lzma
import os
from pathlib import Path
from typing import Iterable

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


def decode_scene_template(path: Path) -> str:
    """Decode an LZMA-compressed, base85-encoded Workbench scene template."""
    encoded = "".join(Path(path).read_text(encoding="ascii").split())
    return lzma.decompress(base64.b85decode(encoded)).decode("utf-8")
