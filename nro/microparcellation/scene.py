from __future__ import annotations

import base64
import lzma
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nro.orchestration.runner import Runner


SURFACE_SCENE_TEMPLATE = Path(__file__).with_name("workbench_montage.scene.in")
VOLUME_SCENE_TEMPLATE = Path(__file__).with_name("volume_workbench.scene.xz.b85")
_SURFACE_TYPES = ("pial", "midthickness", "white", "inflated")


def _surface_family(path: Path) -> dict[str, Path | None]:
    """Resolve the four display surfaces associated with one mesh."""
    for surface_type in _SURFACE_TYPES:
        token = f"_{surface_type}.surf.gii"
        if path.name.endswith(token):
            stem = path.name[: -len(token)]
            break
    else:
        raise FileNotFoundError(f"Could not infer anatomical surface family from {path}")

    family = {
        kind: path.with_name(f"{stem}_{kind}.surf.gii") for kind in _SURFACE_TYPES
    }
    for required in ("pial", "white"):
        candidate = family[required]
        assert candidate is not None
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Missing {required} surface needed for the Workbench scene: {candidate}"
            )
    return {
        kind: candidate if candidate.is_file() else None
        for kind, candidate in family.items()
    }


def _surface_spec_entries(packaged: dict[tuple[str, str], Path]) -> str:
    entries = []
    for index, ((hemi, _kind), path) in enumerate(packaged.items()):
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
    families = tuple(_surface_family(Path(path)) for path in surface_paths)
    packaged = {
        (hemi, kind): output_dir / f"{prefix}_hemi-{hemi}_{kind}.surf.gii"
        for hemi in ("L", "R")
        for kind in _SURFACE_TYPES
    }
    for hemi, family in zip(("L", "R"), families):
        for kind, source in family.items():
            if source is None:
                continue
            destination = packaged[(hemi, kind)]
            # These are generated scene artifacts, so their mtimes must reflect
            # this packaging step. Preserving the source mtime makes the step
            # appear perpetually stale relative to its dlabel and pconn inputs.
            shutil.copyfile(source, destination)

        if family["midthickness"] is None:
            if runner is None or executable is None:
                raise FileNotFoundError(
                    f"Missing midthickness surface for hemisphere {hemi} and no Workbench runner was provided"
                )
            runner.run_child(
                [
                    executable,
                    "-surface-average",
                    str(packaged[(hemi, "midthickness")]),
                    "-surf",
                    str(packaged[(hemi, "pial")]),
                    "-surf",
                    str(packaged[(hemi, "white")]),
                ]
            )
        if family["inflated"] is None:
            if runner is None or executable is None:
                raise FileNotFoundError(
                    f"Missing inflated surface for hemisphere {hemi} and no Workbench runner was provided"
                )
            very_inflated = output_dir / f".{prefix}_hemi-{hemi}_veryInflated.surf.gii"
            try:
                runner.run_child(
                    [
                        executable,
                        "-surface-generate-inflated",
                        str(packaged[(hemi, "midthickness")]),
                        str(packaged[(hemi, "inflated")]),
                        str(very_inflated),
                    ]
                )
            finally:
                very_inflated.unlink(missing_ok=True)

    if tuple(packaged.values()) != packaged_surfaces:
        raise RuntimeError("Surface scene output ordering is inconsistent")

    left_midthickness = packaged[("L", "midthickness")]
    right_midthickness = packaged[("R", "midthickness")]

    replacements = {
        "{{LEFT_SURFACE}}": left_midthickness.name,
        "{{LEFT_SURFACE_NAME}}": left_midthickness.name,
        "{{RIGHT_SURFACE}}": right_midthickness.name,
        "{{RIGHT_SURFACE_NAME}}": right_midthickness.name,
        "{{SURFACE_SPEC_ENTRIES}}": _surface_spec_entries(packaged),
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

    families = tuple(_surface_family(Path(path)) for path in surface_paths)
    source_surfaces = tuple(
        source
        for family in families
        for source in family.values()
        if source is not None
    )
    packaged_surfaces = tuple(
        output_dir / f"{prefix}_hemi-{hemi}_{kind}.surf.gii"
        for hemi in ("L", "R")
        for kind in _SURFACE_TYPES
    )
    scene_path = output_dir / f"{prefix}_microparcellation.scene"
    return scene_path, packaged_surfaces, source_surfaces


def write_volume_workbench_scene(
    output_dir: Path,
    prefix: str,
    label_volume_path: Path,
    dlabel_path: Path,
    pconn_path: Path,
) -> Path:
    """Package a relocatable Workbench volume scene with parcel connectivity."""
    encoded = "".join(VOLUME_SCENE_TEMPLATE.read_text(encoding="ascii").split())
    scene = lzma.decompress(base64.b85decode(encoded)).decode("utf-8")
    replacements = {
        "{{VOLUME_LABEL}}": label_volume_path.name,
        "{{DLABEL}}": dlabel_path.name,
        "{{PCONN}}": pconn_path.name,
    }
    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    if "{{" in scene or "}}" in scene:
        raise RuntimeError("Unresolved placeholder in Workbench volume scene template")
    scene_path = output_dir / f"{prefix}_microparcellation.scene"
    scene_path.write_text(scene, encoding="utf-8")
    return scene_path
