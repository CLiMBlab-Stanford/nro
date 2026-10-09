"""Validate native libraries supplied by the selected Python runtime."""

from __future__ import annotations

import sqlite3
import sys

MINIMUM_SQLITE = (3, 35, 0)


def require_current_runtime() -> None:
    """Reject a Python runtime whose SQLite library cannot execute nro SQL."""
    if sqlite3.sqlite_version_info >= MINIMUM_SQLITE:
        return
    required = ".".join(map(str, MINIMUM_SQLITE))
    raise RuntimeError(
        f"nro requires SQLite {required} or newer; {sys.executable} provides "
        f"{sqlite3.sqlite_version}. Rerun ./install to create the managed Python environment."
    )
