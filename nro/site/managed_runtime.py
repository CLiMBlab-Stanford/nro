"""Locate the helpers bundled with nro's managed Apptainer runtime.

This module uses only Python's standard library so the source launcher can
import it before it loads the rest of nro.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNTIME_MANIFEST = ".nro-runtime.json"


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
    path = Path(executable).expanduser()
    if not path.is_absolute() or not (path.parent.parent / RUNTIME_MANIFEST).is_file():
        return
    if not os.access("/dev/fuse", os.R_OK | os.W_OK):
        return
    directory = str(path.parent)
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = f"{current}{os.pathsep}{directory}" if current else directory
