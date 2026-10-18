from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from nro.orchestration import freshen
from nro.orchestration.artifact_records import file_record
from nro.orchestration.assessment import AssessmentReport, AssessmentSnapshot


class _Registry:
    def __init__(self, database: sqlite3.Connection, control: Path):
        self.database = database
        self.paths = SimpleNamespace(control=control)

    @contextmanager
    def connection(self, *, write: bool = False):
        try:
            yield self.database
            if write:
                self.database.commit()
        except BaseException:
            if write:
                self.database.rollback()
            raise


def _database(output: Path, raw: Path) -> sqlite3.Connection:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(
        """
        CREATE TABLE work_items(
            id INTEGER PRIMARY KEY, input_paths_json TEXT, artifact_state TEXT,
            artifact_reason TEXT, updated_at TEXT, current_generation INTEGER
        );
        CREATE TABLE artifacts(
            id INTEGER PRIMARY KEY, work_item_id INTEGER, attempt_id INTEGER,
            direction TEXT, path TEXT, size INTEGER, mtime_ns INTEGER,
            digest_algorithm TEXT, digest TEXT
        );
        CREATE TABLE completions(work_item_id INTEGER PRIMARY KEY, attempt_id INTEGER);
        CREATE TABLE work_item_dependencies(
            work_item_id INTEGER, upstream_work_item_id INTEGER,
            required_generation INTEGER
        );
        CREATE TABLE attempts(id INTEGER PRIMARY KEY, work_item_id INTEGER, state TEXT);
        CREATE TABLE artifact_mutations(work_item_id INTEGER PRIMARY KEY, token TEXT);
        """
    )
    database.execute("INSERT INTO work_items VALUES (1, '[]', 'fresh', '', '', 4)")
    database.execute(
        "INSERT INTO work_items VALUES (2, ?, 'stale', '', '', 2)",
        (json.dumps([str(raw)]),),
    )
    database.execute(
        "INSERT INTO artifacts VALUES (1, 2, 9, 'output', ?, ?, ?, NULL, NULL)",
        (str(output), output.stat().st_size, output.stat().st_mtime_ns),
    )
    database.execute("INSERT INTO completions VALUES (2, 9)")
    database.execute("INSERT INTO work_item_dependencies VALUES (2, 1, 3)")
    database.commit()
    return database


def _approved() -> dict:
    return {
        "branch": "main",
        "items": [{"id": 2, "action": "freshen"}],
        "freshen": 1,
        "retained": 0,
        "rejected": 0,
    }


def test_preview_accepts_only_unchanged_stale_descendants(monkeypatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.nii.gz"
    parent_output = tmp_path / "anat.nii.gz"
    child_output = tmp_path / "func.nii.gz"
    for path, value in ((raw, "raw"), (parent_output, "anat"), (child_output, "func")):
        path.write_text(value)
    work_items = (
        {
            "id": 1,
            "project": "demo",
            "participant": "01",
            "module": "anat",
            "entities_json": "{}",
            "input_paths_json": json.dumps([str(raw)]),
        },
        {
            "id": 2,
            "project": "demo",
            "participant": "01",
            "module": "func",
            "entities_json": "{}",
            "input_paths_json": "[]",
        },
    )
    snapshot = AssessmentSnapshot(
        tmp_path,
        tmp_path / ".nro",
        work_items,
        ({"work_item_id": 2, "upstream_work_item_id": 1},),
        (),
        (
            {
                "work_item_id": 1,
                "inputs": [file_record(raw)],
                "upstream": [],
                "public_outputs": [file_record(parent_output)],
                "private_artifacts": [],
            },
            {
                "work_item_id": 2,
                "inputs": [],
                "upstream": [{"work_item_id": 1, "generation": 1}],
                "public_outputs": [file_record(child_output)],
                "private_artifacts": [],
            },
        ),
        (),
        (),
    )
    report = AssessmentReport(
        snapshot.fingerprint,
        (
            {
                "id": 1,
                "state": "fresh",
                "reason": "valid",
                "contract": None,
                "command": None,
                "inputs": None,
            },
            {
                "id": 2,
                "state": "stale",
                "reason": "Upstream derivative set or generation changed",
                "contract": None,
                "command": None,
                "inputs": None,
            },
        ),
    )
    rows = [
        {
            **row,
            "directory_label": "main-abc",
            "workflow_ids": "main",
            "purge_token": f"token-{row['id']}",
        }
        for row in work_items
    ]
    monkeypatch.setattr(
        freshen,
        "branch_snapshot",
        lambda *_args, **_kwargs: {"branch": "main", "rows": rows},
    )
    monkeypatch.setattr(freshen, "capture_assessment", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(freshen, "evaluate_assessment", lambda *_args, **_kwargs: report)

    result = freshen.preview(
        SimpleNamespace(paths=SimpleNamespace(control=tmp_path / ".nro")),
        checkout=tmp_path,
        site_values={},
        selection={"projects": ["demo"]},
    )

    assert [(item["id"], item["action"]) for item in result["items"]] == [
        (1, "retain"),
        (2, "freshen"),
    ]

    child_output.write_text("evil")
    changed = freshen.preview(
        SimpleNamespace(paths=SimpleNamespace(control=tmp_path / ".nro")),
        checkout=tmp_path,
        site_values={},
        selection={"projects": ["demo"]},
    )
    assert changed["items"][1]["action"] == "reject"
    assert changed["items"][1]["reason"].startswith("artifact modification time changed:")


def test_execute_updates_timestamp_and_generation_binding(monkeypatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.nii.gz"
    output = tmp_path / "func.nii.gz"
    raw.write_text("raw")
    output.write_text("func")
    future = output.stat().st_mtime_ns + 2_000_000_000
    os.utime(raw, ns=(raw.stat().st_atime_ns, future))
    database = _database(output, raw)
    registry = _Registry(database, tmp_path / ".nro")
    approved = _approved()
    monkeypatch.setattr(freshen, "preview", lambda *_args, **_kwargs: approved)
    monkeypatch.setattr(freshen.dependency_state, "synchronize", lambda *_args, **_kwargs: None)

    result = freshen.execute(
        registry,
        checkout=tmp_path,
        site_values={},
        selection={},
        approved=approved,
    )

    assert result["executed"] == 1
    assert output.stat().st_mtime_ns >= raw.stat().st_mtime_ns
    row = database.execute(
        "SELECT artifact_state,artifact_reason FROM work_items WHERE id=2"
    ).fetchone()
    assert tuple(row) == (
        "fresh",
        "Freshened after unchanged-contract validation",
    )
    assert (
        database.execute(
            "SELECT required_generation FROM work_item_dependencies WHERE work_item_id=2"
        ).fetchone()[0]
        == 4
    )


def test_execute_rolls_back_interrupted_timestamp_repair(monkeypatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.nii.gz"
    output = tmp_path / "func.nii.gz"
    raw.write_text("raw")
    output.write_text("func")
    original = output.stat().st_mtime_ns
    os.utime(raw, ns=(raw.stat().st_atime_ns, original + 2_000_000_000))
    database = _database(output, raw)
    registry = _Registry(database, tmp_path / ".nro")
    approved = _approved()
    monkeypatch.setattr(freshen, "preview", lambda *_args, **_kwargs: approved)

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(freshen.dependency_state, "synchronize", interrupt)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        freshen.execute(
            registry,
            checkout=tmp_path,
            site_values={},
            selection={},
            approved=approved,
        )

    assert output.stat().st_mtime_ns == original
    assert database.execute("SELECT mtime_ns FROM artifacts WHERE id=1").fetchone()[0] == original
    assert database.execute("SELECT COUNT(*) FROM artifact_mutations").fetchone()[0] == 0
    assert not list((tmp_path / ".nro/shared/freshening").glob("*.json"))
