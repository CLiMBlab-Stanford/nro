"""Locate the helpers bundled with nro's managed Apptainer runtime.

This module uses only Python's standard library so the source launcher can
import it before it loads the rest of nro.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

RUNTIME_MANIFEST = ".nro-runtime.json"


def _helper_directory(executable: str | Path) -> str | None:
    """Return a managed runtime's ``bin`` directory when its FUSE helpers can run."""
    path = Path(executable).expanduser()
    if not path.is_absolute() or not (path.parent.parent / RUNTIME_MANIFEST).is_file():
        return None
    if not os.access("/dev/fuse", os.R_OK | os.W_OK):
        return None
    return str(path.parent)


def _append_path(current: str, directory: str) -> str:
    if directory in current.split(os.pathsep):
        return current
    return f"{current}{os.pathsep}{directory}" if current else directory


def expose_managed_runtime_helpers(executable: str | Path) -> None:
    """Let a managed Apptainer and its child processes find its bundled FUSE helpers.

    Apptainer searches ``PATH`` for ``squashfuse_ll`` and ``fuse-overlayfs``.
    Without them it unpacks the complete SIF into a temporary sandbox on every
    launch, and the launch fails where the kernel refuses an unprivileged
    overlay mount. This appends the managed runtime's ``bin`` directory to
    ``PATH`` in the current process, so helpers already on ``PATH`` and other
    host tools keep precedence.

    ``PATH`` is unchanged for external runtimes and when ``/dev/fuse`` is not
    readable and writable. Apptainer does not fall back to unpacking after a
    failed FUSE mount, so hosts without FUSE keep their unpacking behavior.
    """
    directory = _helper_directory(executable)
    if directory is not None:
        os.environ["PATH"] = _append_path(os.environ.get("PATH", ""), directory)


def managed_runtime_environment(
    command: Sequence[str], env: Mapping[str, str] | None
) -> dict[str, str] | None:
    """Return an explicit child environment that can find managed runtime helpers.

    Commands that replace the process environment do not inherit the ``PATH``
    set by :func:`expose_managed_runtime_helpers`. When ``command`` starts the
    managed runtime, the returned copy of ``env`` appends its ``bin`` directory
    to that environment's ``PATH``; all other values are unchanged. ``None``
    still means the child inherits the current process environment.
    """
    if env is None:
        return None
    directory = _helper_directory(command[0]) if command else None
    if directory is None:
        return dict(env)
    return {**env, "PATH": _append_path(env.get("PATH", ""), directory)}
