from pathlib import Path

import pytest

from nro.engine.bids import (
    discover_raw_runs,
    matches_selectors,
    minimal_selectors,
    parse_selectors,
    resolve_run,
)


def _bold(root: Path, stem: str) -> Path:
    path = root / "sub-01" / "func" / f"{stem}_bold.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_dir_entity_disambiguates_hcp_style_runs(tmp_path: Path) -> None:
    subject = tmp_path / "sub-01"
    _bold(tmp_path, "sub-01_task-rest_run-1_dir-LR")
    _bold(tmp_path, "sub-01_task-rest_run-1_dir-RL")
    runs = discover_raw_runs(subject)

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_run(runs, parse_selectors(("task=rest", "run=1")))
    selected = resolve_run(
        runs, parse_selectors(("task=rest", "run=1", "dir=RL"))
    )

    assert selected.stem == "sub-01_task-rest_run-1_dir-RL"


def test_minimal_selectors_include_only_entities_needed_for_uniqueness(tmp_path: Path) -> None:
    subject = tmp_path / "sub-01"
    for run in ("1", "2"):
        for direction in ("LR", "RL"):
            _bold(tmp_path, f"sub-01_task-rest_run-{run}_dir-{direction}")
    runs = discover_raw_runs(subject)

    selectors = {record.stem: minimal_selectors(record, runs) for record in runs}

    assert selectors["sub-01_task-rest_run-1_dir-LR"] == ("dir=LR", "run=1")
    assert all("task=rest" not in values for values in selectors.values())


def test_missing_entity_can_be_selected_explicitly(tmp_path: Path) -> None:
    subject = tmp_path / "sub-01"
    _bold(tmp_path, "sub-01_task-rest_run-1")
    _bold(tmp_path, "sub-01_task-rest_run-1_dir-LR")
    runs = discover_raw_runs(subject)
    target = next(record for record in runs if record.stem.endswith("run-1"))

    assert minimal_selectors(target, runs) == ("dir=",)
    assert resolve_run(runs, parse_selectors(("dir=",))) == target


def test_run_selectors_accept_comma_delimited_alternatives() -> None:
    selectors = parse_selectors(("task=rest,language", "run=1,2"))

    assert selectors == {"task": ("rest", "language"), "run": ("1", "2")}
    assert matches_selectors({"task": "language", "run": "2"}, selectors)
    assert not matches_selectors({"task": "motor", "run": "2"}, selectors)


def test_repeated_run_entities_merge_alternatives() -> None:
    assert parse_selectors(("task=rest", "task=language,spatial")) == {
        "task": ("rest", "language", "spatial")
    }
