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


class _Registry:
    def __init__(self, bids_root: Path, database: sqlite3.Connection) -> None:
        self.paths = SimpleNamespace(bids_root=bids_root)
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
