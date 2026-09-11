"""Build linked or portable Connectome Workbench scene bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import yaml

from nro.engine.io import atomic_output_path, atomic_write_text
from nro.engine.workbench import (
    SURFACE_SCENE_TEMPLATE,
    SURFACE_TYPES,
    VOLUME_SCENE_TEMPLATE,
    decode_scene_template,
    surface_inventory,
)
from nro.orchestration.registry import RegistryLock

VIEWABLE_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".func.gii",
    ".shape.gii",
    ".label.gii",
    ".surf.gii",
)


@dataclass(frozen=True)
class SceneSource:
    """Describe one derivative or anatomical file loaded by a scene."""

    module: str
    role: str
    path: Path
    directory_label: str = "support"
    output_prefix: str = "support"
    output_root: Path | None = None
    project: str | None = None
    participant: str | None = None
    space: str | None = None


def is_viewable(path: Path) -> bool:
    """Return whether Workbench can load the file as imaging data."""

    name = Path(path).name
    return not name.endswith(".scene") and any(
        name.endswith(suffix) for suffix in VIEWABLE_SUFFIXES
    )


def has_surface_data(paths: tuple[Path, ...]) -> bool:
    """Return whether any selected data file addresses cortical vertices."""

    if any(path.name.endswith((".func.gii", ".shape.gii", ".label.gii")) for path in paths):
        return True
    import nibabel as nib

    for path in paths:
        if not any(
            marker in path.name
            for marker in (".dlabel.nii", ".dscalar.nii", ".dtseries.nii", ".pconn.nii")
        ):
            continue
        try:
            image = nib.load(str(path))
            axes = [image.header.get_axis(index) for index in range(len(image.shape))]
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        for axis in axes:
            vertices = getattr(axis, "vertices", None)
            if isinstance(vertices, dict) and any(len(value) for value in vertices.values()):
                return True
            if isinstance(vertices, (list, tuple)) and any(
                any(len(value) for value in item.values())
                for item in vertices
                if isinstance(item, dict)
            ):
                return True
            try:
                if any("CORTEX" in str(structure) for structure, *_ in axis.iter_structures()):
                    return True
            except AttributeError:
                pass
    return False


def read_manifest(path: Path) -> dict[str, object]:
    """Read a JSON or YAML manifest and require a mapping."""

    try:
        text = Path(path).read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read derivative manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Derivative manifest is not a mapping: {path}")
    return value


def manifest_paths(value: object, manifest: Path) -> tuple[Path, ...]:
    """Return existing files recursively named by a manifest value."""

    found: list[Path] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            try:
                if path.is_file():
                    found.append(path.absolute())
            except OSError:
                pass

    visit(value)
    return tuple(dict.fromkeys(found))


def public_manifest_paths(module: str, manifest: Path) -> tuple[Path, ...]:
    """Read the public imaging inventory from one module manifest."""

    document = read_manifest(manifest)
    field = "public_outputs" if module in {"func", "clean", "firstlevels"} else "outputs"
    return tuple(
        path for path in manifest_paths(document.get(field), manifest) if is_viewable(path)
    )


def manifest_anatomical_images(manifest: Path) -> tuple[Path, ...]:
    """Return anatomical references useful for volume display."""

    document = read_manifest(manifest)
    outputs = document.get("outputs")
    if not isinstance(outputs, dict):
        return ()
    values = (
        outputs.get("brain_image"),
        outputs.get("subject_t1w"),
        outputs.get("subject_t2w"),
    )
    return tuple(
        path for path in manifest_paths(values, manifest) if path.name.endswith((".nii", ".nii.gz"))
    )


def template_surface_family(paths: tuple[Path, ...], space: str) -> tuple[Path, ...]:
    """Resolve template display geometry from cortical imaging outputs."""

    if not space.startswith("fsaverage"):
        return ()
    import nibabel as nib

    from nro.engine.templates import find_fsaverage_surface
    from nro.engine.workbench import surface_family

    counts: dict[str, int] = {}
    for path in paths:
        if path.name.endswith((".func.gii", ".shape.gii", ".label.gii")):
            hemi = "L" if "_hemi-L_" in path.name else "R" if "_hemi-R_" in path.name else None
            if hemi:
                counts[hemi] = int(len(nib.load(str(path)).darrays[0].data))
        elif any(
            marker in path.name for marker in (".dlabel.nii", ".dscalar.nii", ".dtseries.nii")
        ):
            image = nib.load(str(path))
            for index in range(len(image.shape)):
                axis = image.header.get_axis(index)
                try:
                    structures = tuple(axis.iter_structures())
                except AttributeError:
                    continue
                for structure, _slice, model in structures:
                    if "CORTEX_LEFT" in str(structure):
                        counts["L"] = int(model.size)
                    elif "CORTEX_RIGHT" in str(structure):
                        counts["R"] = int(model.size)
        if set(counts) == {"L", "R"}:
            break
    if set(counts) != {"L", "R"}:
        return ()
    result: list[Path] = []
    for hemi in ("L", "R"):
        middle = find_fsaverage_surface(
            hemi=hemi,
            surface="midthickness",
            n_vertices=counts[hemi],
        )
        family = surface_family(middle)
        result.extend(family[kind] for kind in SURFACE_TYPES if family[kind] is not None)
    return tuple(result)


def manifest_surface_families(manifest: Path) -> tuple[Path, ...]:
    """Return complete display-surface families named by a manifest."""

    document = read_manifest(manifest)
    values: list[Path] = []
    source_surfaces = document.get("source_surfaces") or ()
    if isinstance(source_surfaces, (list, tuple)):
        values.extend(Path(str(value)).expanduser() for value in source_surfaces)
    outputs = document.get("outputs")
    if isinstance(outputs, dict) and isinstance(outputs.get("surfaces"), dict):
        values.extend(Path(str(value)).expanduser() for value in outputs["surfaces"].values())
    candidates: list[Path] = []
    expanded: list[Path] = []
    for value in values:
        path = value if value.is_absolute() else manifest.parent / value
        if path.is_file() and path.name.endswith(".surf.gii"):
            if any(path.name.endswith(f"_{kind}.surf.gii") for kind in SURFACE_TYPES):
                stem = next(
                    path.name.removesuffix(f"_{kind}.surf.gii")
                    for kind in SURFACE_TYPES
                    if path.name.endswith(f"_{kind}.surf.gii")
                )
                expanded.extend(path.with_name(f"{stem}_{kind}.surf.gii") for kind in SURFACE_TYPES)
    candidates.extend(dict.fromkeys(path.absolute() for path in expanded if path.is_file()))
    try:
        inventory = surface_inventory(candidates)
    except (ValueError, FileNotFoundError):
        return ()
    return tuple(inventory[(hemi, kind)] for hemi in ("L", "R") for kind in SURFACE_TYPES)


def _surface_spec_entries(inventory: dict[tuple[str, str], Path]) -> str:
    entries = []
    for index, ((hemisphere, _kind), path) in enumerate(inventory.items()):
        structure = "CORTEX_LEFT" if hemisphere == "L" else "CORTEX_RIGHT"
        entries.append(
            f'''                                    <Element Index="{index}">
                                        <Object Type="class" Class="SpecFileDataFile" Name="specFileDataFile" Version="1">
                                            <Object Type="enumeratedType" Name="dataFileType">SURFACE</Object>
                                            <Object Type="enumeratedType" Name="structure">{structure}</Object>
                                            <Object Type="pathName" Name="fileName">{path}</Object>
                                            <Object Type="boolean" Name="selected">true</Object>
                                        </Object>
                                    </Element>'''
        )
    return "\n".join(entries)


def _remove_placeholder_elements(scene: str) -> str:
    root = ElementTree.fromstring(scene)
    changed = True
    while changed:
        changed = False
        for parent in root.iter():
            for child in list(parent):
                descendants = tuple(child.iter())
                nested_elements = tuple(node for node in descendants[1:] if node.tag == "Element")
                if (
                    child.tag == "Element"
                    and any("{{" in (node.text or "") for node in descendants)
                    and not any(
                        "{{" in (node.text or "")
                        for element in nested_elements
                        for node in element.iter()
                    )
                ):
                    parent.remove(child)
                    changed = True
    for array in root.iter("ObjectArray"):
        elements = [child for child in array if child.tag == "Element"]
        array.set("Length", str(len(elements)))
        for index, element in enumerate(elements):
            element.set("Index", str(index))
    for node in root.iter():
        if node.text and "{{" in node.text:
            node.text = ""
    result = ElementTree.tostring(root, encoding="unicode")
    if "{{" in result or "}}" in result:
        raise RuntimeError("Unresolved placeholder in Workbench scene template")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + result


def base_scene(
    *,
    scene_id: str,
    surfaces: tuple[Path, ...] = (),
    data: tuple[Path, ...] = (),
) -> str:
    """Render a base scene with display geometry but no derivative data."""

    if surfaces:
        inventory = surface_inventory(surfaces)
        left = inventory[("L", "midthickness")]
        right = inventory[("R", "midthickness")]
        replacements = {
            "{{LEFT_SURFACE}}": str(left),
            "{{LEFT_SURFACE_NAME}}": left.name,
            "{{RIGHT_SURFACE}}": str(right),
            "{{RIGHT_SURFACE_NAME}}": right.name,
            "{{SURFACE_SPEC_ENTRIES}}": _surface_spec_entries(inventory),
        }
        scene = SURFACE_SCENE_TEMPLATE.read_text(encoding="utf-8")
    else:
        scene = decode_scene_template(VOLUME_SCENE_TEMPLATE)
        volume = next(
            (
                path
                for path in data
                if path.name.endswith((".nii", ".nii.gz"))
                and not any(
                    marker in path.name
                    for marker in (".dlabel.nii", ".dscalar.nii", ".dtseries.nii", ".pconn.nii")
                )
            ),
            None,
        )
        dlabel = next((path for path in data if path.name.endswith(".dlabel.nii")), None)
        replacements = {
            **({"{{VOLUME_LABEL}}": str(volume)} if volume is not None else {}),
            **({"{{DLABEL}}": str(dlabel)} if dlabel is not None else {}),
        }
        if not replacements:
            raise ValueError("A volume scene requires a NIfTI or dense-label anchor")
    for placeholder, value in replacements.items():
        scene = scene.replace(placeholder, value)
    scene = _remove_placeholder_elements(scene)
    return scene.replace("Microparcellation", scene_id)


def _copy_with_digest(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as input_stream, atomic_output_path(destination) as temporary:
        with temporary.open("wb") as output_stream:
            while block := input_stream.read(8 * 1024 * 1024):
                output_stream.write(block)
                digest.update(block)
    shutil.copystat(source, destination)
    return digest.hexdigest()


def _published_path(root: Path, source: SceneSource) -> Path:
    try:
        relative = source.path.relative_to(source.output_root) if source.output_root else None
    except ValueError:
        relative = None
    if relative is None:
        digest = hashlib.sha256(str(source.path.absolute()).encode()).hexdigest()[:12]
        relative = Path(f"{digest}-{source.path.name}")
    return (
        root / "inputs" / source.module / source.directory_label / source.output_prefix / relative
    )


def _replace_generated_directory(staging: Path, destination: Path, scene_id: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError(f"Refusing to replace a linked scene directory: {destination}")
        manifest = destination / "scene_manifest.yaml"
        try:
            document = read_manifest(manifest)
        except ValueError as error:
            raise ValueError(
                f"Refusing to replace an unmanaged scene directory: {destination}"
            ) from error
        if document.get("kind") != "nro-workbench-scene" or document.get("scene_id") != scene_id:
            raise ValueError(f"Refusing to replace an unmanaged scene directory: {destination}")
        shutil.rmtree(destination)
    os.replace(staging, destination)


def build_scene_bundle(
    destination: Path,
    *,
    scene_id: str,
    sources: tuple[SceneSource, ...],
    surfaces: tuple[SceneSource, ...] = (),
    selection: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    wb_command: str | Path,
    publish: bool = False,
) -> Path:
    """Create one linked or self-contained scene bundle and return its scene file."""

    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = RegistryLock(
        destination.parent / f".{scene_id}.lock",
        destination.parent / f".{scene_id}.lock.recovery",
    )
    with lock:
        staging = destination.parent / f".{scene_id}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            all_sources = tuple(dict.fromkeys((*sources, *surfaces)))
            mapped: dict[SceneSource, tuple[Path, str | None]] = {}
            for source in all_sources:
                if publish:
                    target = _published_path(staging, source)
                    mapped[source] = (target, _copy_with_digest(source.path, target))
                else:
                    mapped[source] = (source.path.absolute(), None)
            rendered = {
                source: (path.relative_to(staging) if publish else path)
                for source, (path, _digest) in mapped.items()
            }
            display_surfaces = tuple(rendered[source] for source in surfaces)
            base = staging / ".base.scene"
            atomic_write_text(
                base,
                base_scene(
                    scene_id=scene_id,
                    surfaces=display_surfaces,
                    data=tuple(rendered[source] for source in sources),
                ),
            )
            scene = staging / f"{scene_id}.scene"
            command = [
                str(wb_command),
                "-scene-file-update",
                str(base),
                str(scene),
                "1",
                "-error",
            ]
            for source in sources:
                command.extend(("-data-file-add", str(rendered[source])))
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                cwd=staging,
            )
            if result.returncode:
                message = (result.stderr or result.stdout or "unknown Workbench error").strip()
                raise RuntimeError(f"Workbench could not build the scene: {message}")
            base.unlink(missing_ok=True)
            records = []
            for source in all_sources:
                _stored_path, digest = mapped[source]
                records.append(
                    {
                        "module": source.module,
                        "role": source.role,
                        "source": str(source.path),
                        "scene_path": str(rendered[source]),
                        "size_bytes": source.path.stat().st_size,
                        **({"sha256": digest} if digest is not None else {}),
                    }
                )
            manifest = {
                "kind": "nro-workbench-scene",
                "format_version": 1,
                "scene_id": scene_id,
                "mode": "published" if publish else "linked",
                **(context or {}),
                "selection": selection or {},
                "scene": scene.name,
                "sources": records,
            }
            atomic_write_text(
                staging / "scene_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False)
            )
            _replace_generated_directory(staging, destination, scene_id)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return destination / f"{scene_id}.scene"
