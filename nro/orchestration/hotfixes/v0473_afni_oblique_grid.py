"""Purge functional work resampled on a cardinal grid that AFNI did not use.

Before 0.47.3, nro chose the cardinal ("plumb") grid of an oblique image with
nibabel's `io_orientation`, while AFNI's `3dNwarpApply` chooses it with
`nifti_mat44_to_orientation`. The two rules agree except for grids tilted close
to 45 degrees. For such a grid, the AFNI warp and motion affines that nro wrote
were expressed on the wrong cardinal grid, and the resampled 4D BOLD was moved
far from its registered position. The boldref images, which FSL resamples,
were unaffected.

The defect changed no work-item contract, so the scheduler still treats these
outputs as fresh. This hotfix finds `func` work items whose source BOLD or
T1w output grid falls in the disagreeing range and purges them with all of their
downstream consumers, including their WORK intermediates. The next request
rebuilds them.
"""

from __future__ import annotations

import functools
import itertools
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.images import afni_orientation

HOTFIX_ID = "v0473-afni-oblique-grid"
SUMMARY = "Purge functional runs whose oblique grid AFNI resampled on different cardinal axes."
MANIFEST_SUFFIX = "_desc-preprocessFunc_manifest.json"
READ_THREADS = 32


def _bold_inputs(row: dict) -> Iterable[Path]:
    for value in json.loads(row.get("input_paths_json") or "[]"):
        if str(value).endswith(("_bold.nii.gz", "_bold.nii")):
            yield Path(value)


def _output_grids(row: dict) -> Iterable[Path]:
    """Yield the T1w boldref images, which share the T1w 4D output grid.

    Template grids are cardinal, so only the subject-space grid can be misread.
    """
    for value in json.loads(row.get("expected_outputs_json") or "[]"):
        manifest = Path(value)
        if manifest.name.endswith(MANIFEST_SUFFIX):
            stem = manifest.name[: -len(MANIFEST_SUFFIX)]
            yield manifest.parent / f"{stem}_space-T1w_boldref.nii.gz"


@functools.cache
def grid_was_misread(path: Path) -> bool:
    """Report whether AFNI may have used other cardinal axes than nro assumed for an image.

    nro assumed nibabel's orientation of the image affine. AFNI may have read
    either coded header matrix, so both are compared. A missing file is not
    misread.
    """
    try:
        image = nib.load(str(path))
    except FileNotFoundError:
        return False
    assumed = nib.orientations.io_orientation(image.affine)
    header = image.header
    matrices = [image.affine]
    if isinstance(header, nib.Nifti1Header):
        if int(header["qform_code"]) > 0:
            matrices.append(header.get_qform())
        if int(header["sform_code"]) > 0:
            matrices.append(header.get_sform())
    return any(not np.array_equal(assumed, afni_orientation(matrix)) for matrix in matrices)


def _row_was_misread(row: dict) -> bool:
    grids = itertools.chain(_bold_inputs(row), _output_grids(row))
    return any(grid_was_misread(path) for path in grids)


def purge_roots(rows: list[dict]) -> set[int]:
    """Return the IDs of `func` work items with a misread source or output grid.

    Header reads run on a thread pool because each one waits on shared storage.
    """
    functional = [row for row in rows if row.get("module") == "func"]
    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        flags = pool.map(_row_was_misread, functional)
        return {int(row["id"]) for row, flag in zip(functional, flags, strict=True) if flag}


def run(registry, *, projects: tuple[str, ...], execute: bool):
    """Refuse scheduler-side application; `nro hotfix` applies this hotfix as a purge."""
    raise ValueError(f"{HOTFIX_ID} purges derivatives and must be applied with `nro hotfix`")
