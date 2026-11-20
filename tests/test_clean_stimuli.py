"""Task regression in clean and the stimulus annotation it expands presentations with."""

from types import SimpleNamespace

import numpy as np
import pytest

from nro.definitions.store import ConfigStore
from nro.engine.stimuli import StimulusAnnotationError
from nro.modules.clean.cleaning import _build_task_regressors
from nro.modules.clean.planning import clean_direct_inputs
from nro.orchestration.contract_migrations import current_contract_schema, migrate_contract

WORDS = "onset\tduration\ttext\n0.0\t0\tOnce\n1.0\t0\tupon\n"


def _project(tmp_path, annotations=("stim-a", "stim-b")):
    root = tmp_path / "proj"
    (root / "stimuli").mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.9.0"}')
    for stim_id in annotations:
        (root / "stimuli" / f"{stim_id}_annot-words_events.tsv").write_text(WORDS)
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    run = func / "sub-01_task-story_run-1_bold.nii.gz"
    run.write_bytes(b"")
    events = func / "sub-01_task-story_run-1_events.tsv"
    events.write_text(
        "onset\tduration\tstim_id\ttrial_type\n"
        "10\t20\tstim-a\tstory\n40\t20\tstim-b\tstory\n70\t5\tn/a\tbreak\n"
    )
    return root, SimpleNamespace(path=run, stem="sub-01_task-story_run-1"), events


def test_packaged_default_regresses_presentations_as_blocks():
    assert ConfigStore().load_configuration("clean", "main").values["task_annotation"] == "none"


def test_planned_inputs_follow_the_selected_annotation(tmp_path):
    root, run, events = _project(tmp_path)
    config = {"regress_out_task": True, "task_annotation": "none"}
    assert clean_direct_inputs(run, config) == (events,)
    assert clean_direct_inputs(run, {**config, "task_annotation": "words"}) == (
        events,
        root / "stimuli/stim-a_annot-words_events.tsv",
        root / "stimuli/stim-b_annot-words_events.tsv",
    )
    assert clean_direct_inputs(run, {**config, "regress_out_task": False}) == ()


def test_a_missing_annotation_is_reported_at_planning(tmp_path):
    _, run, _ = _project(tmp_path, annotations=("stim-a",))
    with pytest.raises(StimulusAnnotationError, match="nro stimuli sync"):
        clean_direct_inputs(run, {"regress_out_task": True, "task_annotation": "words"})


def _story(events, annotation):
    regressors, _ = _build_task_regressors(
        events_path=events, n_scans=60, tr=2.0, start_time=0.0, annotation=annotation
    )
    assert [frame.columns[0] for frame in regressors] == ["task.break", "task.story"]
    return regressors[1]["task.story"].to_numpy()


def test_regression_uses_blocks_or_annotation_rows(tmp_path):
    _, _, events = _project(tmp_path)
    blocks, words = _story(events, "none"), _story(events, "words")
    assert np.isfinite(blocks).all() and np.isfinite(words).all()
    assert blocks.max() > 0 and words.max() > 0
    assert not np.allclose(blocks / blocks.max(), words / words.max())


def test_clean_before_task_annotation_regressed_blocks():
    current = ConfigStore().load_configuration("clean", "main").values
    historical = {key: value for key, value in current.items() if key != "task_annotation"}
    contract = {"module": "clean", "configuration": "fingerprint", "processing": {}}
    _, migrated = migrate_contract({**contract, "contract_schema": 4}, historical)
    _, fresh = migrate_contract(
        {**contract, "contract_schema": current_contract_schema("clean")}, current
    )
    assert migrated is not None and fresh is not None
    assert migrated["task_annotation"] == fresh["task_annotation"] == "none"
    assert migrated == fresh
