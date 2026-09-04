"""Project and derivative path construction."""

from __future__ import annotations

from pathlib import Path
from nro.configuration.paths import BIDS_PATH, WORK_PATH


def project_data_root(project: str) -> Path:
    return (Path(BIDS_PATH) / project).resolve()


def project_work_root(project: str) -> Path:
    return (Path(WORK_PATH) / project).resolve()


def preprocessing_derivatives_root(*, project: str, preprocessing_id: str) -> Path:
    return project_data_root(project) / "derivatives" / "preprocessing" / preprocessing_id


def preprocessing_work_root(*, project: str, preprocessing_id: str) -> Path:
    return project_work_root(project) / "derivatives" / "preprocessing" / preprocessing_id


def preprocessing_subject_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_derivatives_root(project=project, preprocessing_id=preprocessing_id) / sub_id


def preprocessing_session_dir(sub_id: str, ses_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_subject_dir(sub_id, project=project, preprocessing_id=preprocessing_id) / ses_id


def preprocessing_subject_anat_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_subject_dir(sub_id, project=project, preprocessing_id=preprocessing_id) / "anat"


def preprocessing_session_anat_dir(sub_id: str, ses_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_session_dir(sub_id, ses_id, project=project, preprocessing_id=preprocessing_id) / "anat"


def preprocessing_subject_work_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_work_root(project=project, preprocessing_id=preprocessing_id) / sub_id


def preprocessing_session_work_dir(sub_id: str, ses_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_subject_work_dir(sub_id, project=project, preprocessing_id=preprocessing_id) / ses_id


def preprocess_subject_func_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_subject_dir(sub_id, project=project, preprocessing_id=preprocessing_id) / "func"


def preprocess_session_func_dir(sub_id: str, ses_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_session_dir(sub_id, ses_id, project=project, preprocessing_id=preprocessing_id) / "func"


def functional_manifest_path(
    sub_id: str,
    run_stem: str,
    *,
    project: str,
    preprocessing_id: str,
    ses_id: str | None = None,
) -> Path:
    """Return the fixed public contract for one preprocessed BOLD run."""
    directory = (
        preprocess_session_func_dir(
            sub_id, ses_id, project=project, preprocessing_id=preprocessing_id
        )
        if ses_id is not None
        else preprocess_subject_func_dir(
            sub_id, project=project, preprocessing_id=preprocessing_id
        )
    )
    return directory / f"{run_stem}_desc-preprocessFunc_manifest.json"


def preprocess_subject_func_work_dir(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_subject_work_dir(sub_id, project=project, preprocessing_id=preprocessing_id) / "func"


def preprocess_session_func_work_dir(sub_id: str, ses_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_session_work_dir(sub_id, ses_id, project=project, preprocessing_id=preprocessing_id) / "func"


def anatomical_manifest_path(sub_id: str, *, project: str, preprocessing_id: str) -> Path:
    return preprocessing_subject_anat_dir(sub_id, project=project, preprocessing_id=preprocessing_id) / f"{sub_id}_desc-preprocessAnat_manifest.json"


def is_bids_session_id(ses_id: str | None) -> bool:
    return bool(str(ses_id or "").startswith("ses-"))


def resolve_project_path(path: str | Path | None, *, project: str) -> Path | None:
    if path is None:
        return None
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (project_data_root(project) / value).resolve()


def resolve_project_work_path(path: str | Path | None, *, project: str) -> Path | None:
    if path is None:
        return None
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (project_work_root(project) / value).resolve()


def resolve_cwd_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def optional_path(value: object) -> Path | None:
    """Expand an optional path, treating ``auto`` as unspecified."""
    if value in (None, "auto"):
        return None
    return Path(str(value)).expanduser()
