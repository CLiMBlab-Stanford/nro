"""Fixed publication contracts for derivative modules."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a JSON contract last without exposing a partial document."""
    atomic_write_json(path, payload)
