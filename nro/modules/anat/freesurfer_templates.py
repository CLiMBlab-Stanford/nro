"""Publish container-provided FreeSurfer templates without host-specific links."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from nro.engine.io import atomic_write_text

_REQUIRED = (
    "mri/T1.mgz",
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
)


def template_directory(template_root: Path, *, build: str) -> Path:
    """Return one versioned FreeSurfer template in a project's nro namespace."""
    return Path(template_root) / build / "fsaverage"


def container_template_directory(*, build: str) -> Path:
    """Return the mount point reached by a portable subjects-directory link."""
    return Path("/.nro/templates/freesurfer") / build / "fsaverage"


def _valid_template(path: Path) -> bool:
    return path.is_dir() and all(
        (candidate := path / relative).is_file() and candidate.stat().st_size
        for relative in _REQUIRED
    )


def _replace_link(path: Path, target: Path) -> None:
    relative = os.path.relpath(target, path.parent)
    temporary = path.with_name(f".{path.name}.portable-{uuid.uuid4().hex}")
    try:
        temporary.symlink_to(relative, target_is_directory=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_portable_fsaverage(
    *,
    runtime: str,
    image: Path,
    subjects_dir: Path,
    template_root: Path,
    build: str,
    container_source: str,
    execute: Callable[[Sequence[str]], Any],
) -> tuple[Path, bool]:
    """Copy fsaverage once and point ``SUBJECTS_DIR/fsaverage`` to it relatively.

    The copy is shared by every anatomical lineage in one BIDS project. The
    caller supplies the command executor so workers and maintenance operations
    use their existing process-control policy.
    """
    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    target = template_directory(template_root, build=build)
    template_root = Path(template_root)
    template_root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = template_root / ".materialize.lock"
    lock_path.touch(mode=0o664, exist_ok=True)
    created = False
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists() and not _valid_template(target):
            raise RuntimeError(f"Incomplete FreeSurfer template copy: {target}")
        if not target.exists():
            staging = target.with_name(f".{target.name}.stage-{uuid.uuid4().hex}")
            try:
                command = (
                    str(runtime),
                    "exec",
                    "--cleanenv",
                    "--bind",
                    f"{target.parent.resolve()}:/nro-template-output",
                    str(Path(image).resolve()),
                    "cp",
                    "-a",
                    container_source,
                    f"/nro-template-output/{staging.name}",
                )
                execute(command)
                if not _valid_template(staging):
                    raise RuntimeError(
                        f"Container did not provide a complete FreeSurfer template: {image}"
                    )
                atomic_write_text(
                    staging / "nro-template.json",
                    json.dumps(
                        {
                            "Build": build,
                            "ContainerSource": container_source,
                            "Template": "fsaverage",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    mode=0o664,
                )
                os.replace(staging, target)
                created = True
            finally:
                if staging.is_dir() and not staging.is_symlink():
                    shutil.rmtree(staging)
                else:
                    staging.unlink(missing_ok=True)
        link = subjects_dir / "fsaverage"
        expected = os.path.relpath(target, link.parent)
        if link.is_symlink() and os.readlink(link) == expected:
            return target, created
        if link.exists() and not link.is_symlink():
            raise RuntimeError(f"FreeSurfer template link path is occupied: {link}")
        _replace_link(link, target)
    return target, created
