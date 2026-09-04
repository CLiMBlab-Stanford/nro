"""Resolve canonical template data from existing local resources."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.configuration.store import ConfigStore
from .images import gifti_vertex_count


def templateflow_roots() -> tuple[Path, ...]:
    """Return existing local TemplateFlow trees in preference order."""
    candidates: list[Path] = []
    configured = os.environ.get("TEMPLATEFLOW_HOME", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    mni_template = (
        ConfigStore()
        .load_configuration("preprocessing", "main")
        .values["anat"]
        .get("mni_template")
    )
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
