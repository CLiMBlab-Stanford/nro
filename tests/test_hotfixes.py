from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from nro.orchestration.artifact_records import file_record
from nro.orchestration.hotfixes import apply, available
from nro.orchestration.hotfixes.v0278_private_portable_metadata import HOTFIX_ID
from nro.orchestration.hotfixes.v0286_msmall_runner_stages import (
    HOTFIX_ID as MSMALL_HOTFIX_ID,
)


class _Registry:
    def __init__(self, bids_root: Path, database: sqlite3.Connection) -> None:
        self.paths = SimpleNamespace(bids_root=bids_root, control=bids_root.parent / ".nro")
        self.database = database

    @contextmanager
    def connection(self, *, write: bool = False):
        yield self.database
        if write:
            self.database.commit()


def _registry(tmp_path: Path) -> tuple[_Registry, sqlite3.Connection]:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(
        """
        CREATE TABLE work_items (id INTEGER PRIMARY KEY, project TEXT NOT NULL);
        CREATE TABLE attempts (id INTEGER PRIMARY KEY, work_item_id INTEGER, state TEXT);
        CREATE TABLE artifacts (
            id INTEGER PRIMARY KEY,
            direction TEXT NOT NULL,
            path TEXT NOT NULL,
            size INTEGER,
            mtime_ns INTEGER,
            digest_algorithm TEXT,
            digest TEXT
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    return _Registry(tmp_path / "bids", database), database


def _portable_sidecar(tmp_path: Path) -> Path:
    path = tmp_path / "bids/demo/derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "BrainMask": "bids::anat/main/sub-01/anat/sub-01_desc-brain_mask.nii.gz",
                "Sources": ["bids:raw:sub-01/anat/sub-01_T1w.nii.gz"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_private_portable_metadata_hotfix_is_strict_and_idempotent(tmp_path: Path) -> None:
    registry, database = _registry(tmp_path)
    path = _portable_sidecar(tmp_path)
    current = file_record(path)
    database.execute(
        """INSERT INTO artifacts(
               direction,path,size,mtime_ns,digest_algorithm,digest
           ) VALUES ('private',?,?,?,?,?)""",
        (
            str(path.resolve()),
            current["size"] + 7,
            current["mtime_ns"],
            "sha256",
            hashlib.sha256(b"historical absolute metadata").hexdigest(),
        ),
    )

    preview = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=False)
    assert preview.paths == (path,)
    assert preview.records == 1
    assert not preview.applied

    result = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=True)
    assert result.records == 1
    stored = database.execute(
        "SELECT size,mtime_ns,digest FROM artifacts WHERE path=?", (str(path.resolve()),)
    ).fetchone()
    assert dict(stored) == {
        "size": current["size"],
        "mtime_ns": current["mtime_ns"],
        "digest": current["sha256"],
    }
    audit = database.execute(
        "SELECT value FROM metadata WHERE key=?", (f"hotfix:{HOTFIX_ID}:demo",)
    ).fetchone()
    assert json.loads(audit[0])["records"] == 1

    repeated = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=True)
    assert repeated.records == 0
    retained = database.execute(
        "SELECT value FROM metadata WHERE key=?", (f"hotfix:{HOTFIX_ID}:demo",)
    ).fetchone()
    assert json.loads(retained[0])["records"] == 1


def test_private_portable_metadata_hotfix_rejects_changed_timestamp(tmp_path: Path) -> None:
    registry, database = _registry(tmp_path)
    path = _portable_sidecar(tmp_path)
    current = file_record(path)
    database.execute(
        """INSERT INTO artifacts(
               direction,path,size,mtime_ns,digest_algorithm,digest
           ) VALUES ('private',?,?,?,?,?)""",
        (
            str(path.resolve()),
            current["size"] + 7,
            current["mtime_ns"] - 1,
            "sha256",
            "wrong",
        ),
    )

    report = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=False)

    assert report.records == 0


def test_hotfix_registry_discovers_release_scoped_repairs() -> None:
    assert HOTFIX_ID in available()
    assert MSMALL_HOTFIX_ID in available()


def test_msmall_runner_stage_hotfix_adopts_exact_legacy_checkpoints(tmp_path: Path) -> None:
    registry, database = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    work = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01/msmall"
    markers = work / "markers"
    markers.mkdir(parents=True)
    legacy_configuration = work / "configuration.sh"
    legacy_configuration.write_text("subject=sub-01\n")
    (markers / "prefreesurfer.complete").write_text("complete\n")
    (markers / "multirun_fix.complete").write_text("complete\n")
    variance = (
        work / "study/01_msmall/MNINonLinear/Results/rfMRI_REST_CONCAT/"
        "rfMRI_REST_CONCAT_Atlas_hp0.0_clean_vn_before_floor.dscalar.nii"
    )
    variance.parent.mkdir(parents=True)
    variance.write_text("variance")
    (variance.parent / "rfMRI_REST_CONCAT_Atlas_hp0.0_clean.dtseries.nii").write_text("timeseries")
    t1_dir = work / "study/01_msmall/T1w"
    t1_dir.mkdir(parents=True)
    (t1_dir / "T1w_acpc_dc_restore.nii.gz").write_text("T1w")
    (t1_dir / "T2w_acpc_dc_restore.nii.gz").write_text("T2w")
    event.write_text(
        json.dumps(
            {
                "version": 4,
                "module": "Anatomical Module",
                "signature": "work-item",
                "topology": [],
                "nodes": [
                    {
                        "name": "Estimate MSMAll Registration",
                        "outputs": [
                            str(work / "complete"),
                            str(work / "software_versions.txt"),
                            str(
                                work
                                / "study/01_msmall/MNINonLinear/Native/01_msmall.L.sphere.MSMAll.native.surf.gii"
                            ),
                            str(
                                work
                                / "study/01_msmall/MNINonLinear/Native/01_msmall.L.sphere.reg.reg_LR.native.surf.gii"
                            ),
                        ],
                    }
                ],
            }
        )
    )

    preview = apply(
        registry,
        identifier=MSMALL_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert preview.records == 1

    report = apply(
        registry,
        identifier=MSMALL_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert report.records == 1
    assert (work / "structural_configuration.sh").read_text() == "subject=sub-01\n"
    assert (work / "surface_configuration.sh").read_text() == "subject=sub-01\n"
    assert (work / "calibration_configuration.sh").read_text() == "subject=sub-01\n"
    assert (work / "stages/prefreesurfer.complete").is_file()
    assert (work / "stages/multirun_fix.complete").is_file()
    assert json.loads(event.read_text())["version"] == 2

    repeated = apply(
        registry,
        identifier=MSMALL_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert repeated.records == 0
