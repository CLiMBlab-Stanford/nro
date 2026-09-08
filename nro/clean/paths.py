"""Deterministic paths for cleaned derivatives and intermediates."""

from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH


def clean_root(*, project: str, clean_id: str) -> Path:
    return Path(BIDS_PATH) / project / "derivatives" / "clean" / clean_id


def clean_work_root(*, project: str, clean_id: str) -> Path:
    return Path(WORK_PATH) / project / "derivatives" / "clean" / clean_id


def clean_subject_dir(sub_id: str, *, project: str, clean_id: str) -> Path:
    return clean_root(project=project, clean_id=clean_id) / sub_id


def clean_session_dir(sub_id: str, ses_id: str, *, project: str, clean_id: str) -> Path:
    return clean_subject_dir(sub_id, project=project, clean_id=clean_id) / ses_id


def clean_manifest_path(
    sub_id: str,
    run_stem: str,
    *,
    project: str,
    clean_id: str,
    space: str,
    smoothing_mm: int,
    ses_id: str | None = None,
) -> Path:
    """Return the public contract for one run, space, and smoothing level."""
    directory = (
        clean_session_dir(sub_id, ses_id, project=project, clean_id=clean_id)
        if ses_id is not None
        else clean_subject_dir(sub_id, project=project, clean_id=clean_id)
    )
    return directory / (
        f"{run_stem}_space-{space}_smoothing-{smoothing_mm}mm_"
        "desc-clean_manifest.json"
    )


def clean_subject_work_dir(sub_id: str, *, project: str, clean_id: str) -> Path:
    return clean_work_root(project=project, clean_id=clean_id) / sub_id


def clean_session_work_dir(sub_id: str, ses_id: str, *, project: str, clean_id: str) -> Path:
    return clean_subject_work_dir(sub_id, project=project, clean_id=clean_id) / ses_id
