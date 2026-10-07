"""Adopt completed MSMAll driver checkpoints into the runner-level DAG."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.hotfixes import HotfixReport

HOTFIX_ID = "v0286-msmall-runner-stages"
SUMMARY = "Adopt validated legacy MSMAll checkpoints as runner-owned stage outputs."


@dataclass(frozen=True)
class _Candidate:
    contract: Path
    work_root: Path
    completed_stages: tuple[str, ...]


def _legacy_work_root(contract: Path, project: str) -> Path | None:
    """Return the exact legacy MSMAll work root represented by one contract."""
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) < 3:
        return None
    if payload.get("module") != "Anatomical Module":
        return None
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return None
    matches = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("name") == "Estimate MSMAll Registration"
    ]
    if len(matches) != 1:
        return None
    outputs = matches[0].get("outputs")
    if not isinstance(outputs, list):
        return None
    paths = [Path(value) for value in outputs if isinstance(value, str)]
    complete = [path for path in paths if path.name == "complete" and path.parent.name == "msmall"]
    if len(complete) != 1:
        return None
    work_root = complete[0].parent
    if project not in work_root.parts or not (work_root / "configuration.sh").is_file():
        return None
    required_suffixes = {
        "software_versions.txt",
        "sphere.MSMAll.native.surf.gii",
        "sphere.reg.reg_LR.native.surf.gii",
    }
    rendered = "\n".join(str(path) for path in paths)
    if not all(suffix in rendered for suffix in required_suffixes):
        return None
    return work_root


def _completed_stages(work_root: Path) -> tuple[str, ...]:
    """Return legacy stages whose marker and declared evidence remain valid."""
    markers = work_root / "markers"
    if not markers.is_dir():
        return ()
    stages: list[str] = []
    for path in sorted(markers.glob("*.complete")):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        stage = path.name.removesuffix(".complete")
        if _stage_valid(work_root, stage):
            stages.append(stage)
    return tuple(stages)


def _stage_valid(work_root: Path, stage: str) -> bool:
    sessions = [path for path in (work_root / "study").glob("*") if path.is_dir()]
    if len(sessions) != 1 and stage not in {"inventory"}:
        return False
    session = sessions[0] if sessions else work_root / "missing-session"
    native = session / "MNINonLinear/Native"
    results = session / "MNINonLinear/Results"
    evidence: list[Path] = []
    if stage == "inventory":
        evidence = [work_root / "input_manifest.tsv", work_root / "software_versions.txt"]
    elif stage == "prefreesurfer":
        evidence = [
            session / "T1w/T1w_acpc_dc_restore.nii.gz",
            session / "T1w/T2w_acpc_dc_restore.nii.gz",
        ]
    elif stage == "masked_atlas":
        evidence = [
            session / "MNINonLinear/registration_qc.json",
            session / "MNINonLinear/xfms/acpc_dc2standard.nii.gz",
        ]
    elif stage == "freesurfer":
        evidence = [session / f"T1w/{session.name}/surf/{hemi}.white" for hemi in ("lh", "rh")]
    elif stage == "postfreesurfer":
        evidence = [
            native / f"{session.name}.{hemi}.sphere.reg.native.surf.gii" for hemi in ("L", "R")
        ]
    elif stage.startswith("fmri_volume_"):
        run = stage.removeprefix("fmri_volume_")
        evidence = [results / run / f"{run}.nii.gz"]
    elif stage.startswith("fmri_surface_"):
        run = stage.removeprefix("fmri_surface_")
        evidence = [results / run / f"{run}_Atlas.dtseries.nii"]
    elif stage == "multirun_fix":
        evidence = list(
            results.glob("rfMRI_REST_CONCAT/rfMRI_REST_CONCAT_Atlas_hp*_clean.dtseries.nii")
        ) + list(
            results.glob(
                "rfMRI_REST_CONCAT/rfMRI_REST_CONCAT_Atlas_hp*_clean_vn_before_floor.dscalar.nii"
            )
        )
        return len(evidence) == 2 and all(_nonempty(path) for path in evidence)
    elif stage == "prepare_msmall":
        evidence = list(
            results.glob("rfMRI_REST_CONCAT/rfMRI_REST_CONCAT_Atlas_hp*_clean_vn.dscalar.nii")
        )
        return len(evidence) == 1 and _nonempty(evidence[0])
    elif stage == "msmall":
        evidence = [
            next(
                iter(native.glob(f"{session.name}.{hemi}.sphere.*InitialReg*.native.surf.gii")),
                Path(),
            )
            for hemi in ("L", "R")
        ]
    elif stage == "dedrift":
        evidence = [
            native / f"{session.name}.{hemi}.sphere.MSMAll.native.surf.gii" for hemi in ("L", "R")
        ]
    elif stage == "validate":
        evidence = [work_root / "complete"]
    else:
        return False
    return bool(evidence) and all(_nonempty(path) for path in evidence)


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _candidates(control: Path, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    branches = ControlPaths(control).root / "branches"
    found: list[_Candidate] = []
    for project in projects:
        for contract in sorted(branches.glob(f"*/events/{project}/anat/**/runner-contract.json")):
            work_root = _legacy_work_root(contract, project)
            if work_root is None:
                continue
            stages = _completed_stages(work_root)
            if stages:
                found.append(_Candidate(contract, work_root, stages))
    return tuple(found)


def _copy_preserving_time(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
    shutil.copy2(source, destination)
    os.chmod(destination, 0o664)


def _apply(candidate: _Candidate) -> tuple[Path, ...]:
    legacy_configuration = candidate.work_root / "configuration.sh"
    paths: list[Path] = []
    for name in (
        "structural_configuration.sh",
        "surface_configuration.sh",
        "calibration_configuration.sh",
    ):
        destination = candidate.work_root / name
        if not destination.exists():
            _copy_preserving_time(legacy_configuration, destination)
        paths.append(destination)
    for stage in candidate.completed_stages:
        source = candidate.work_root / "markers" / f"{stage}.complete"
        destination = candidate.work_root / "stages" / f"{stage}.complete"
        if not destination.exists():
            _copy_preserving_time(source, destination)
        paths.append(destination)
    payload = json.loads(candidate.contract.read_text(encoding="utf-8"))
    payload["version"] = 2
    atomic_write_json(candidate.contract, payload, sort_keys=True)
    paths.append(candidate.contract)
    return tuple(paths)


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Translate only exact completed contracts from the former shell-owned DAG."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    with registry.connection(write=execute) as database:
        if execute:
            placeholders = ",".join("?" for _ in selected)
            active = database.execute(
                f"""SELECT COUNT(*) FROM attempts AS a
                    JOIN work_items AS w ON w.id=a.work_item_id
                    WHERE w.project IN ({placeholders})
                      AND a.state IN ('queued','running','cancel_requested')""",
                selected,
            ).fetchone()[0]
            if active:
                raise ValueError("Hotfix requires selected-project attempts to be stopped")
        candidates = _candidates(registry.paths.control, selected)
        paths = tuple(
            path
            for candidate in candidates
            for path in (
                candidate.contract,
                candidate.work_root / "structural_configuration.sh",
                candidate.work_root / "surface_configuration.sh",
                candidate.work_root / "calibration_configuration.sh",
                *(
                    candidate.work_root / "stages" / f"{stage}.complete"
                    for stage in candidate.completed_stages
                ),
            )
        )
        if execute:
            paths = tuple(path for candidate in candidates for path in _apply(candidate))
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=tuple(sorted(set(paths))),
        records=len(candidates),
        applied=execute,
    )
