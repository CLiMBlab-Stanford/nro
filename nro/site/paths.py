"""Site paths used by supported nro modules.

Each function reads the active site configuration when called, so a process
that changes `NRO_SITE_CONFIG` or its site file sees the new values.
"""

from __future__ import annotations

from pathlib import Path

from nro.site.configuration import bids_root, settings


def bids_path() -> Path:
    """Return the shared BIDS root with symbolic links resolved.

    This matches the root that scheduled attempts receive in their execution
    context, so direct and scheduled runs see the same paths.
    """
    return bids_root()


def work_path() -> Path:
    """Return the configured private work root, without resolving symbolic links."""
    return Path(settings()[0]["work"])


def registry_path() -> Path:
    """Return the configured central control directory."""
    return Path(settings()[0]["registry"])


def wb_command_path() -> Path:
    """Return the configured Connectome Workbench `wb_command` executable."""
    return Path(settings()[0]["workbench"])
