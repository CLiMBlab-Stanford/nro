"""Paths published by anatomical preprocessing."""

from pathlib import Path

from nro.configuration.paths import BIDS_PATH


def find_preprocessed_anat_dir(
    project: str, participant: str, preprocessing_id: str
) -> Path:
    """Return the subject-level anatomical derivative directory."""
    return (
        Path(BIDS_PATH)
        / project
        / "derivatives"
        / "preprocessing"
        / preprocessing_id
        / f"sub-{participant.removeprefix('sub-')}"
        / "anat"
    )
