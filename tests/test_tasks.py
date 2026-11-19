from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.bin import tasks
from nro.bin.find import find_images


def _file(path: Path, content: bytes = b"image") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _dataset(tmp_path: Path) -> Path:
    bids = tmp_path / "BIDS"
    func = bids / "demo/sub-t12/ses-01/func"
    _file(func / "sub-t12_ses-01_task-Rest_run-01_bold.nii.gz")
    _file(func / "sub-t12_ses-01_task-langloc_run-01_bold.nii.gz")
    _file(func / "sub-t12_ses-01_task-langloc_run-02_bold.nii.gz")
    _file(func / "sub-t12_ses-01_task-langloc_run-01_events.tsv", b"onset\tduration\n0\t1\n")
    _file(func / "sub-t12_ses-01_task-langloc_run-01_sbref.nii.gz")
    _file(bids / "demo/sub-t12/ses-01/anat/sub-t12_ses-01_T1w.nii.gz")
    _file(bids / "demo/sub-t20/func/sub-t20_task-rest_bold.nii.gz")
    _file(bids / "other/task-langloc_events.tsv", b"onset\tduration\n0\t1\n")
    _file(bids / "other/sub-t12/func/sub-t12_task-langloc_bold.nii.gz")
    _file(bids / "demo/derivatives/nro/func/main/sub-t12/sub-t12_task-Rest_bold.nii.gz")
    return bids


def _write_site(monkeypatch, bids: Path) -> None:
    monkeypatch.setattr("nro.site.configuration.bids_root", lambda: bids)


def test_summaries_count_bold_runs_events_participants_and_projects(tmp_path) -> None:
    bids = _dataset(tmp_path)

    summaries = tasks.summarize_tasks(bids, find_images(bids))

    assert [item.as_dict() for item in summaries] == [
        {
            "task": "langloc",
            "runs": 3,
            "runs_with_events": 2,
            "participants": 2,
            "projects": {"demo": 2, "other": 1},
        },
        {
            "task": "Rest",
            "runs": 1,
            "runs_with_events": 0,
            "participants": 1,
            "projects": {"demo": 1},
        },
        {
            "task": "rest",
            "runs": 1,
            "runs_with_events": 0,
            "participants": 1,
            "projects": {"demo": 1},
        },
    ]


def test_table_has_one_header_and_aligned_columns(tmp_path) -> None:
    bids = _dataset(tmp_path)

    table = tasks.render_table(tasks.summarize_tasks(bids, find_images(bids)))

    assert table.splitlines() == [
        "TASK     RUNS  EVENTS  PARTICIPANTS  PROJECTS",
        "langloc     3       2             2  demo:2 other:1",
        "Rest        1       0             1  demo:1",
        "rest        1       0             1  demo:1",
    ]


def test_main_applies_selectors_and_prints_json(tmp_path, monkeypatch, capsys) -> None:
    _write_site(monkeypatch, _dataset(tmp_path))

    tasks.main(["-P", "demo", "--task", "re:(?i)rest", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert [item["task"] for item in output] == ["Rest", "rest"]


def test_main_writes_json_file(tmp_path, monkeypatch) -> None:
    _write_site(monkeypatch, _dataset(tmp_path))
    destination = tmp_path / "out" / "tasks.json"
    destination.parent.mkdir()

    tasks.main(["-p", "t20", "--json", str(destination)])

    assert json.loads(destination.read_text())[0]["task"] == "rest"


def test_main_prints_table_without_pager(tmp_path, monkeypatch, capsys) -> None:
    _write_site(monkeypatch, _dataset(tmp_path))

    tasks.main(["-P", "other", "--no-pager"])

    assert capsys.readouterr().out.splitlines()[1].startswith("langloc")


@pytest.mark.parametrize("selector", ["datatype=anat", "suffix=sbref"])
def test_main_rejects_fixed_image_entities(tmp_path, monkeypatch, selector) -> None:
    _write_site(monkeypatch, _dataset(tmp_path))

    with pytest.raises(SystemExit, match="func BOLD runs only"):
        tasks.main(["-r", selector])


def test_main_reports_empty_table_selection(tmp_path, monkeypatch) -> None:
    _write_site(monkeypatch, _dataset(tmp_path))

    with pytest.raises(SystemExit, match="No source BOLD runs"):
        tasks.main(["--task", "missing", "--no-pager"])
