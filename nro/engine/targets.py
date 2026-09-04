"""Analysis-target entities and user-facing defaults."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_SPACE = "fsnative"
DEFAULT_SMOOTHING_MM = 2


def bids_scale_value(smoothing_mm: int) -> str:
    """Encode smoothing for the BIDS ``scale`` filename entity."""
    if smoothing_mm < 0:
        raise ValueError("Smoothing must be a nonnegative integer FWHM in mm")
    return f"{smoothing_mm}mm"


def add_smoothing_entity(path: Path, smoothing_mm: int) -> Path:
    """Add the BIDS ``scale`` encoding of smoothing after a space entity."""
    value = bids_scale_value(smoothing_mm)
    name, replacements = re.subn(
        r"(_space-[^_]+)",
        rf"\1_scale-{value}",
        Path(path).name,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Functional filename lacks a space entity: {path}")
    return Path(path).with_name(name)
