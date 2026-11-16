"""Resolve canonical template data from existing local resources."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from nro.definitions.store import ConfigStore

from .images import gifti_vertex_count

FSAVERAGE_VERTEX_COUNTS = {"fsaverage": 163842, "fsaverage6": 40962}


def templateflow_roots() -> tuple[Path, ...]:
    """Return existing local TemplateFlow trees in preference order."""
    candidates: list[Path] = []
    from nro.site.configuration import settings

    configured = settings()[0]["templates"]
    if configured:
        candidates.append(Path(configured).expanduser())

    mni_template = ConfigStore().load_configuration("anat", "main").values.get("mni_template")
    if mni_template:
        candidates.append(Path(mni_template).expanduser().parent.parent)

    roots: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def find_fsaverage_surface(
    *,
    hemi: str,
    surface: str,
    n_vertices: int,
    roots: tuple[Path, ...] | None = None,
) -> Path:
    """Find a locally available TemplateFlow fsaverage geometry by mesh size."""
    searched: list[Path] = []
    for root in templateflow_roots() if roots is None else roots:
        directory = root / "tpl-fsaverage"
        searched.append(directory)
        for candidate in sorted(
            directory.glob(f"tpl-fsaverage_hemi-{hemi}_den-*_{surface}.surf.gii")
        ):
            try:
                if gifti_vertex_count(candidate) == int(n_vertices):
                    return candidate
            except (OSError, ValueError):
                continue
    locations = ", ".join(str(path) for path in searched) or "no local TemplateFlow root"
    raise FileNotFoundError(
        f"Could not find fsaverage hemisphere-{hemi} {surface} geometry with "
        f"{n_vertices} vertices under {locations}. Set TEMPLATEFLOW_HOME to a populated "
        "TemplateFlow resource directory."
    )


def find_fsaverage_template_surface(
    *,
    template: str,
    hemi: str,
    surface: str,
    roots: tuple[Path, ...] | None = None,
) -> Path:
    """Find geometry for a supported fsaverage template in local TemplateFlow data."""
    try:
        n_vertices = FSAVERAGE_VERTEX_COUNTS[str(template)]
    except KeyError as error:
        raise ValueError(f"Unsupported fsaverage template: {template}") from error
    return find_fsaverage_surface(
        hemi=hemi,
        surface=surface,
        n_vertices=n_vertices,
        roots=roots,
    )


FSLR_32K_VERTEX_COUNT = 32492


def find_fslr_32k_resampling_geometry(
    *,
    hemi: str,
    roots: tuple[Path, ...] | None = None,
) -> tuple[Path, Path]:
    """Find the 32k fsLR sphere in fsaverage coordinates and its cortex label.

    The sphere places the fsLR 32k mesh on the fsaverage registration sphere, so
    data on fsaverage, fsaverage6, or a participant's FreeSurfer ``sphere.reg``
    can be resampled to fsLR 32k directly. The label marks cortex outside the
    medial wall. Empty TemplateFlow placeholders do not count as present.
    """
    searched: list[Path] = []
    for root in templateflow_roots() if roots is None else roots:
        directory = root / "tpl-fsLR"
        searched.append(directory)
        sphere = directory / f"tpl-fsLR_space-fsaverage_hemi-{hemi}_den-32k_sphere.surf.gii"
        label = directory / f"tpl-fsLR_hemi-{hemi}_den-32k_desc-nomedialwall_dparc.label.gii"
        try:
            if all(
                path.is_file()
                and path.stat().st_size > 0
                and gifti_vertex_count(path) == FSLR_32K_VERTEX_COUNT
                for path in (sphere, label)
            ):
                return sphere, label
        except (OSError, ValueError, nib.filebasedimages.ImageFileError):
            continue
    locations = ", ".join(str(path) for path in searched) or "no local TemplateFlow root"
    raise FileNotFoundError(
        f"Could not find the hemisphere-{hemi} 32k fsLR sphere in fsaverage space and "
        f"its medial-wall label under {locations}. Rerun nro setup to install the "
        "pinned TemplateFlow resources."
    )


def find_mni_gray_matter_mask(*, space: str, functional: Path) -> Path:
    """Find the local TemplateFlow GM probability map nearest the functional resolution.

    Preprocessed MNI outputs may legitimately use a cropped field of view, so
    shape and origin are not TemplateFlow-resolution identifiers. Exact grids
    are preferred; otherwise the probability map is selected by voxel size and
    resampled onto the functional grid by the volume loader.
    """
    reference = nib.load(str(functional))
    reference_shape = tuple(reference.shape[:3])
    reference_spacing = nib.affines.voxel_sizes(reference.affine)
    searched: list[Path] = []
    compatible: list[tuple[float, Path]] = []
    for root in templateflow_roots():
        directory = root / f"tpl-{space}"
        searched.append(directory)
        candidates = sorted(directory.glob(f"tpl-{space}*_label-GM_probseg.nii*"))
        for candidate in candidates:
            try:
                image = nib.load(str(candidate))
            except OSError:
                continue
            if tuple(image.shape[:3]) != reference_shape:
                spacing = nib.affines.voxel_sizes(image.affine)
                if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                    score = float(np.max(np.abs(np.log(spacing / reference_spacing))))
                    compatible.append((score, candidate))
                continue
            if np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-4):
                return candidate
            spacing = nib.affines.voxel_sizes(image.affine)
            if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                score = float(np.max(np.abs(np.log(spacing / reference_spacing))))
                compatible.append((score, candidate))
    if compatible:
        return min(compatible, key=lambda item: (item[0], str(item[1])))[1]
    locations = ", ".join(str(path) for path in searched) or "no local TemplateFlow root"
    raise FileNotFoundError(
        f"Could not find a space-{space} TemplateFlow gray-matter probability map for "
        f"{functional} under {locations}. Set TEMPLATEFLOW_HOME to a populated "
        "TemplateFlow resource directory or specify the microparcellation 'mask' option."
    )
