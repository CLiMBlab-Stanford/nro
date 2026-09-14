"""Project and derivative path construction."""

from __future__ import annotations

from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.modules import MODULE_NAMES


def _directory_component(value: str, *, label: str) -> str:
    """Validate one internally assigned derivative-directory component."""
    value = str(value)
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def project_data_root(project: str, *, bids_root: Path | None = None) -> Path:
    """Resolve a project under the supplied BIDS root or the selected site root."""
    return (Path(BIDS_PATH if bids_root is None else bids_root) / project).resolve()


def project_work_root(project: str) -> Path:
    """Resolve a project's private work root."""
    return (Path(WORK_PATH) / project).resolve()


def module_namespace_root(project_root: Path, module: str) -> Path:
    """Locate one module's public namespace below a selected project root."""
    if module not in MODULE_NAMES:
        raise ValueError(f"Unknown module {module!r}; choose from {', '.join(MODULE_NAMES)}")
    return Path(project_root) / "derivatives" / "nro" / module


def module_artifact_root(project_root: Path, module: str, module_id: str) -> Path:
    """Locate a module artifact root below an already selected project root."""
    module_id = _directory_component(module_id, label="module ID")
    return module_namespace_root(project_root, module) / module_id


def module_derivatives_root(
    module: str, module_id: str, *, project: str, bids_root: Path | None = None
) -> Path:
    """Locate one module configuration's public derivative root."""
    return module_artifact_root(project_data_root(project, bids_root=bids_root), module, module_id)


def module_work_root(
    module: str,
    module_id: str,
    *,
    project: str,
    work_root: Path | None = None,
) -> Path:
    """Locate one module configuration's private work root."""
    project_root = (
        project_work_root(project) if work_root is None else Path(work_root).resolve() / project
    )
    return module_artifact_root(project_root, module, module_id)


def module_subject_dir(
    sub_id: str,
    *,
    module: str,
    module_id: str,
    project: str,
    bids_root: Path | None = None,
) -> Path:
    """Locate one subject within a module derivative root."""
    return module_derivatives_root(module, module_id, project=project, bids_root=bids_root) / sub_id


def module_session_dir(
    sub_id: str,
    ses_id: str,
    *,
    module: str,
    module_id: str,
    project: str,
    bids_root: Path | None = None,
) -> Path:
    """Locate one session within a module's subject directory."""
    return (
        module_subject_dir(
            sub_id,
            module=module,
            module_id=module_id,
            project=project,
            bids_root=bids_root,
        )
        / ses_id
    )


def anat_subject_dir(
    sub_id: str, *, project: str, anat_id: str, bids_root: Path | None = None
) -> Path:
    """Locate subject-level anatomical derivatives."""
    return (
        module_subject_dir(
            sub_id, module="anat", module_id=anat_id, project=project, bids_root=bids_root
        )
        / "anat"
    )


def anat_session_dir(sub_id: str, ses_id: str, *, project: str, anat_id: str) -> Path:
    """Locate session-level anatomical derivatives."""
    return (
        module_session_dir(sub_id, ses_id, module="anat", module_id=anat_id, project=project)
        / "anat"
    )


def anat_subject_work_dir(sub_id: str, *, project: str, anat_id: str) -> Path:
    """Locate a subject's private anatomical work."""
    return module_work_root("anat", anat_id, project=project) / sub_id


def anat_session_work_dir(sub_id: str, ses_id: str, *, project: str, anat_id: str) -> Path:
    """Locate a session's private anatomical work."""
    return anat_subject_work_dir(sub_id, project=project, anat_id=anat_id) / ses_id


def func_subject_dir(
    sub_id: str, *, project: str, func_id: str, bids_root: Path | None = None
) -> Path:
    """Locate functional derivatives for data without session directories."""
    return (
        module_subject_dir(
            sub_id, module="func", module_id=func_id, project=project, bids_root=bids_root
        )
        / "func"
    )


def func_session_dir(
    sub_id: str,
    ses_id: str,
    *,
    project: str,
    func_id: str,
    bids_root: Path | None = None,
) -> Path:
    """Locate functional derivatives for one BIDS session."""
    return (
        module_session_dir(
            sub_id,
            ses_id,
            module="func",
            module_id=func_id,
            project=project,
            bids_root=bids_root,
        )
        / "func"
    )


def functional_manifest_path(
    sub_id: str,
    run_stem: str,
    *,
    project: str,
    func_id: str,
    ses_id: str | None = None,
    bids_root: Path | None = None,
) -> Path:
    """Return the fixed public contract for one preprocessed BOLD run."""
    directory = (
        func_session_dir(sub_id, ses_id, project=project, func_id=func_id, bids_root=bids_root)
        if ses_id is not None
        else func_subject_dir(sub_id, project=project, func_id=func_id, bids_root=bids_root)
    )
    return directory / f"{run_stem}_desc-preprocessFunc_manifest.json"


def func_subject_work_dir(sub_id: str, *, project: str, func_id: str) -> Path:
    """Locate subject-level private functional work."""
    return module_work_root("func", func_id, project=project) / sub_id / "func"


def func_session_work_dir(sub_id: str, ses_id: str, *, project: str, func_id: str) -> Path:
    """Locate session-level private functional work."""
    return module_work_root("func", func_id, project=project) / sub_id / ses_id / "func"


def anatomical_manifest_path(
    sub_id: str, *, project: str, anat_id: str, bids_root: Path | None = None
) -> Path:
    """Return the fixed anatomical completion manifest under the selected root."""
    return (
        anat_subject_dir(sub_id, project=project, anat_id=anat_id, bids_root=bids_root)
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
    return module_derivatives_root("clean", clean_id, project=project, bids_root=bids_root)


def clean_work_root(*, project: str, clean_id: str, work_root: Path | None = None) -> Path:
    """Locate a cleaning lineage in private work storage."""
    return module_work_root("clean", clean_id, project=project, work_root=work_root)


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
