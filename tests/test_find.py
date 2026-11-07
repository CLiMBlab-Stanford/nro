from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.bin.find import build_parser, find_images, main
from nro.engine.bids import matches_selectors, parse_selectors


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def _dataset(tmp_path: Path) -> Path:
    bids = tmp_path / "BIDS"
    _image(bids / "demo/sub-t12/ses-01/func/sub-t12_ses-01_task-Rest_run-01_bold.nii.gz")
    _image(bids / "demo/sub-t12/ses-02/func/sub-t12_ses-02_task-langloc_run-02_bold.nii.gz")
    _image(bids / "demo/sub-t12/ses-01/anat/sub-t12_ses-01_T1w.nii.gz")
    _image(bids / "demo/sub-t20/func/sub-t20_task-Resting_run-03_bold.nii.gz")
    _image(bids / "other/sub-t12/func/sub-t12_task-Rest_run-04_bold.nii.gz")
    _image(bids / "demo/derivatives/nro/func/main/sub-t12/sub-t12_task-Rest_bold.nii.gz")
    return bids


def test_find_matches_source_images_with_regex_entity_selectors(tmp_path) -> None:
    bids = _dataset(tmp_path)

    matches = find_images(
        bids,
        projects=("demo",),
        participants=("re:t(12|20)",),
        runs=("task=re:Rest.*", "run=0[13]"),
    )

    assert [match.path.name for match in matches] == [
        "sub-t12_ses-01_task-Rest_run-01_bold.nii.gz",
        "sub-t20_task-Resting_run-03_bold.nii.gz",
    ]


def test_find_supports_suffix_datatype_and_absent_entity_selectors(tmp_path) -> None:
    bids = _dataset(tmp_path)

    matches = find_images(
        bids,
        projects=("demo",),
        runs=("datatype=anat", "suffix=T1w", "task="),
    )

    assert [match.path.name for match in matches] == ["sub-t12_ses-01_T1w.nii.gz"]


def test_pattern_alternatives_preserve_group_and_quantifier_commas() -> None:
    selectors = parse_selectors(("task=re:(Rest|langloc),motor", "run=re:0{1,2}", "dir=[A,P]P"))

    assert selectors == {
        "task": ("re:(Rest|langloc)", "motor"),
        "run": ("re:0{1,2}",),
        "dir": ("[A,P]P",),
    }
    assert matches_selectors({"task": "langloc", "run": "00", "dir": "AP"}, selectors)
    assert matches_selectors({"task": "motor", "run": "0", "dir": "PP"}, selectors)
    assert not matches_selectors({"task": "Resting", "run": "00", "dir": "AP"}, selectors)


def test_find_rejects_invalid_regular_expressions(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid regular expression"):
        find_images(tmp_path, runs=("task=re:[",))


@pytest.mark.parametrize("selector", ["space=fsnative", "model=main", "workflow=main"])
def test_find_rejects_derivative_entities_in_run_syntax(tmp_path, selector) -> None:
    with pytest.raises(ValueError, match="not a selectable source BIDS image entity"):
        find_images(tmp_path, runs=(selector,))


@pytest.mark.parametrize(
    "arguments",
    [
        ("--module", "func"),
        ("--workflow", "main"),
        ("--lineage", "main-abc"),
        ("--model", "main"),
        ("--model-set", "main"),
        ("--space", "fsnative"),
        ("--smoothing", "2"),
    ],
)
def test_find_rejects_derivative_only_selectors(arguments) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)

    assert error.value.code == 2


def test_find_cli_prints_paths_or_json(tmp_path, monkeypatch, capsys) -> None:
    bids = _dataset(tmp_path)
    monkeypatch.setattr("nro.site.configuration.bids_root", lambda: bids)

    main(["-P", "other", "-p", "sub-t12", "-r", "task=Rest"])
    plain = capsys.readouterr().out.strip()
    assert plain.endswith("sub-t12_task-Rest_run-04_bold.nii.gz")
    assert Path(plain).is_absolute()

    main(["-P", "other", "-r", "task=Rest", "--json"])
    assert json.loads(capsys.readouterr().out) == [plain]


def test_find_accepts_glob_project_and_participant_selectors(tmp_path) -> None:
    bids = _dataset(tmp_path)

    matches = find_images(bids, projects=("de*",), participants=("t2?",), tasks=("Rest*",))

    assert [match.path.name for match in matches] == ["sub-t20_task-Resting_run-03_bold.nii.gz"]
