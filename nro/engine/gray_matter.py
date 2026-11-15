"""Resolve the gray-matter mask used by volumetric connectivity modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nro.engine.io import read_public_json
from nro.engine.paths import anatomical_manifest_path
from nro.engine.templates import find_mni_gray_matter_mask
from nro.site.paths import BIDS_PATH

if TYPE_CHECKING:
    from nro.orchestration.execution_context import ExecutionContext


def infer_gray_matter_mask(
    functionals,
    configured_mask,
    *,
    project,
    participant=None,
    anat_directory=None,
    space,
    execution_context: ExecutionContext | None = None,
):
    """Resolve a configured or space-specific gray-matter mask."""
    if configured_mask:
        path = Path(configured_mask).expanduser()
        if not path.is_absolute():
            path = (
                Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
                / project
                / path
            )
        if execution_context is not None:
            path = execution_context.input_path(path)
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured gray-matter mask does not exist: {path}")
        return path

    if space.startswith("MNI"):
        return find_mni_gray_matter_mask(space=space, functional=functionals[0][0])
    if participant is None or anat_directory is None:
        raise ValueError("participant and anat_directory are required for a native-space mask")
    manifest_path = anatomical_manifest_path(
        f"sub-{participant}",
        project=project,
        anat_id=anat_directory,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        manifest_path = execution_context.input_path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing anatomical publication manifest: {manifest_path}")
    manifest = read_public_json(manifest_path)
    value = (manifest.get("outputs") or {}).get("gray_matter_mask")
    if not value:
        raise FileNotFoundError(
            f"Anatomical manifest has no outputs.gray_matter_mask: {manifest_path}"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing published anatomical gray-matter mask: {path}")
    return path.resolve()
