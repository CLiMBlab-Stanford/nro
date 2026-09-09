"""Analysis-target entities and user-facing defaults."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SPACE = "fsnative"
DEFAULT_SMOOTHING_MM = 2


def smoothing_entity_value(smoothing_mm: int) -> str:
    """Encode a smoothing FWHM for nro paths and filenames."""
    if smoothing_mm < 0:
        raise ValueError("Smoothing must be a nonnegative integer FWHM in mm")
    return f"{smoothing_mm}mm"


def add_smoothing_entity(path: Path, smoothing_mm: int) -> Path:
    """Add nro's smoothing entity after a filename's space entity."""
    import re

    value = smoothing_entity_value(smoothing_mm)
    name, replacements = re.subn(
        r"(_space-[^_]+)",
        rf"\1_smoothing-{value}",
        Path(path).name,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Functional filename lacks a space entity: {path}")
    return Path(path).with_name(name)


def target_directory_name(space: str, smoothing_mm: int) -> str:
    """Return the directory name for one space and smoothing target."""
    return f"space-{space}_smoothing-{smoothing_entity_value(smoothing_mm)}"


def target_output_names(base_prefix: str, space: str, smoothing_mm: int) -> tuple[str, str]:
    """Return the shared target directory and entity-decorated output prefix."""
    return (
        target_directory_name(space, smoothing_mm),
        f"{base_prefix}_space-{space}_smoothing-{smoothing_entity_value(smoothing_mm)}",
    )
