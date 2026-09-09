"""Project and derivative path construction."""

from __future__ import annotations

from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH


def project_data_root(project: str, *, bids_root: Path | None = None) -> Path:
    """Resolve a project under the supplied BIDS root or the selected site root."""
    return (Path(BIDS_PATH if bids_root is None else bids_root) / project).resolve()


def project_work_root(project: str) -> Path:
    """Resolve a project's private work root."""
    return (Path(WORK_PATH) / project).resolve()


def preprocessing_derivatives_root(
    *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Locate a preprocessing lineage without creating it."""
    return (
        project_data_root(project, bids_root=bids_root)
        / "derivatives"
        / "preprocessing"
        / preprocessing_id
    )


def preprocessing_work_root(*, project: str, preprocessing_id: str) -> Path:
    """Locate a preprocessing lineage in private work storage."""
    return project_work_root(project) / "derivatives" / "preprocessing" / preprocessing_id


def preprocessing_subject_dir(
    sub_id: str, *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Locate a subject's preprocessing artifacts under the selected root."""
    return (
        preprocessing_derivatives_root(
            project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        / sub_id
    )


def preprocessing_session_dir(
    sub_id: str, ses_id: str, *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Locate one preprocessing session under its subject."""
    return (
        preprocessing_subject_dir(
            sub_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        / ses_id
    )


def preprocessing_subject_anat_dir(
    sub_id: str, *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Locate subject-level anatomical derivatives."""
    return (
        preprocessing_subject_dir(
            sub_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        / "anat"
    )


def preprocessing_session_anat_dir(
    sub_id: str, ses_id: str, *, project: str, preprocessing_id: str
) -> Path:
    """Locate session-level anatomical derivatives."""
    return (
        preprocessing_session_dir(
            sub_id, ses_id, project=project, preprocessing_id=preprocessing_id
        )
        / "anat"
    )


def preprocessing_subject_work_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    """Locate a subject's private preprocessing work."""
    return preprocessing_work_root(project=project, preprocessing_id=preprocessing_id) / sub_id


def preprocessing_session_work_dir(
    sub_id: str, ses_id: str, *, project: str, preprocessing_id: str
) -> Path:
    """Locate a session's private preprocessing work."""
    return (
        preprocessing_subject_work_dir(sub_id, project=project, preprocessing_id=preprocessing_id)
        / ses_id
    )


def preprocess_subject_func_dir(
    sub_id: str, *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Locate functional derivatives for data without session directories."""
    return (
        preprocessing_subject_dir(
            sub_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        / "func"
    )


def preprocess_session_func_dir(
    sub_id: str, ses_id: str, *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Locate functional derivatives for one BIDS session."""
    return (
        preprocessing_session_dir(
            sub_id, ses_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        / "func"
    )


def functional_manifest_path(
    sub_id: str,
    run_stem: str,
    *,
    project: str,
    preprocessing_id: str,
    ses_id: str | None = None,
    bids_root: Path | None = None,
) -> Path:
    """Return the fixed public contract for one preprocessed BOLD run."""
    directory = (
        preprocess_session_func_dir(
            sub_id, ses_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        if ses_id is not None
        else preprocess_subject_func_dir(
            sub_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
    )
    return directory / f"{run_stem}_desc-preprocessFunc_manifest.json"


def preprocess_subject_func_work_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    """Locate subject-level private functional work."""
    return (
        preprocessing_subject_work_dir(sub_id, project=project, preprocessing_id=preprocessing_id)
        / "func"
    )


def preprocess_session_func_work_dir(
    sub_id: str, ses_id: str, *, project: str, preprocessing_id: str
) -> Path:
    """Locate session-level private functional work."""
    return (
        preprocessing_session_work_dir(
            sub_id, ses_id, project=project, preprocessing_id=preprocessing_id
        )
        / "func"
    )


def anatomical_manifest_path(
    sub_id: str, *, project: str, preprocessing_id: str, bids_root: Path | None = None
) -> Path:
    """Return the fixed anatomical completion manifest under the selected root."""
    return (
        preprocessing_subject_anat_dir(
            sub_id, project=project, preprocessing_id=preprocessing_id, bids_root=bids_root
        )
        / f"{sub_id}_desc-preprocessAnat_manifest.json"
    )


def is_bids_session_id(ses_id: str | None) -> bool:
    """Return whether a value includes the BIDS ``ses-`` prefix."""
    return bool(str(ses_id or "").startswith("ses-"))


def resolve_project_path(path: str | Path | None, *, project: str) -> Path | None:
    """Resolve an optional path relative to a project's data root."""
    if path is None:
        return None
    value = Path(path).expanduser()
    return (
        value.resolve() if value.is_absolute() else (project_data_root(project) / value).resolve()
    )


def resolve_project_work_path(path: str | Path | None, *, project: str) -> Path | None:
    """Resolve an optional path relative to a project's private work root."""
    if path is None:
        return None
    value = Path(path).expanduser()
    return (
        value.resolve() if value.is_absolute() else (project_work_root(project) / value).resolve()
    )


def resolve_cwd_path(path: str | Path | None) -> Path | None:
    """Resolve an optional path relative to the current directory."""
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def optional_path(value: object) -> Path | None:
    """Expand an optional path, treating ``auto`` as unspecified."""
    if value in (None, "auto"):
        return None
    return Path(str(value)).expanduser()


def clean_root(*, project: str, clean_id: str, bids_root: Path | None = None) -> Path:
    """Locate a cleaning lineage under a project derivative root."""
    return project_data_root(project, bids_root=bids_root) / "derivatives" / "clean" / clean_id


def clean_work_root(*, project: str, clean_id: str, work_root: Path | None = None) -> Path:
    """Locate a cleaning lineage in private work storage."""
    root = project_work_root(project) if work_root is None else Path(work_root) / project
    return root / "derivatives" / "clean" / clean_id


def clean_subject_dir(
    sub_id: str,
    *,
    project: str,
    clean_id: str,
    bids_root: Path | None = None,
) -> Path:
    """Locate a subject's cleaned derivatives without creating directories."""
    return clean_root(project=project, clean_id=clean_id, bids_root=bids_root) / sub_id


def clean_session_dir(
    sub_id: str,
    ses_id: str,
    *,
    project: str,
    clean_id: str,
    bids_root: Path | None = None,
) -> Path:
    """Locate cleaned derivatives for one BIDS session."""
    return (
        clean_subject_dir(sub_id, project=project, clean_id=clean_id, bids_root=bids_root) / ses_id
    )


def clean_manifest_path(
    sub_id: str,
    run_stem: str,
    *,
    project: str,
    clean_id: str,
    space: str,
    smoothing_mm: int,
    ses_id: str | None = None,
    bids_root: Path | None = None,
) -> Path:
    """Return the public contract for one run, space, and smoothing level."""
    directory = (
        clean_session_dir(
            sub_id,
            ses_id,
            project=project,
            clean_id=clean_id,
            bids_root=bids_root,
        )
        if ses_id is not None
        else clean_subject_dir(sub_id, project=project, clean_id=clean_id, bids_root=bids_root)
    )
    return directory / (
        f"{run_stem}_space-{space}_smoothing-{smoothing_mm}mm_desc-clean_manifest.json"
    )


def clean_subject_work_dir(
    sub_id: str,
    *,
    project: str,
    clean_id: str,
    work_root: Path | None = None,
) -> Path:
    """Locate a subject's private cleaning work."""
    return clean_work_root(project=project, clean_id=clean_id, work_root=work_root) / sub_id


def clean_session_work_dir(
    sub_id: str,
    ses_id: str,
    *,
    project: str,
    clean_id: str,
    work_root: Path | None = None,
) -> Path:
    """Locate a session's private cleaning work."""
    return (
        clean_subject_work_dir(sub_id, project=project, clean_id=clean_id, work_root=work_root)
        / ses_id
    )
