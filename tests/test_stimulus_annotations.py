"""Stimulus annotation expansion for first-level models."""

import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from test_contrasts import _synthetic_project

from nro.definitions.store import ConfigStore
from nro.engine.bids import discover_raw_runs
from nro.engine.io import read_public_json
from nro.modules.contrasts.authoring import model_draft
from nro.modules.contrasts.compiler import compile_model
from nro.modules.contrasts.contract import validate_completion
from nro.modules.contrasts.module import run_module
from nro.modules.contrasts.planning import _planned_inputs, direct_inputs
from nro.modules.contrasts.stimuli import (
    StimulusAnnotationError,
    annotation_inputs,
    expand_presentations,
)
from nro.modules.contrasts.task_models import scientific_model, validate_task_model

WORDS = {
    "model_set": [],
    "annotation": "words",
    "conditions": "word_class",
    "contrasts": {"content": {"content": 1}, "CvF": {"content": 0.5, "function": -0.5}},
}


def _write(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(path, sep="\t", index=False)
    return path


def _stimuli(root, *, stim="stim-story1", words=None):
    return _write(
        root / "stimuli" / f"{stim}_annot-words_events.tsv",
        words
        or {
            "onset": [0.0, 1.5, 3.0],
            "duration": [0.4, 0.3, 0.5],
            "word_class": ["content", "function", "content"],
        },
    )


def _run_events(root, rows=None):
    return _write(
        root / "sub-01/func/sub-01_task-story_run-1_events.tsv",
        rows
        or {
            "onset": [10.0, 2.0],
            "duration": [20.0, 0.0],
            "trial_type": ["story", "button"],
            "stim_id": ["stim-story1", None],
        },
    )


def test_expansion_shifts_onsets_copies_columns_and_keeps_other_rows(tmp_path):
    path = _stimuli(tmp_path)
    events = pd.read_csv(_run_events(tmp_path), sep="\t")
    expanded, sources = expand_presentations(
        events, annotation="words", dataset_root=tmp_path, source="run.tsv"
    )
    assert expanded["onset"].tolist() == [2.0, 10.0, 11.5, 13.0]
    assert expanded["duration"].tolist() == [0.0, 0.4, 0.3, 0.5]
    assert expanded["trial_type"].tolist() == ["button", "story", "story", "story"]
    assert expanded["word_class"].isna().tolist() == [True, False, False, False]
    assert "stim_id" not in expanded
    assert sources == {
        "stim-story1": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }


def test_repeated_presentations_reuse_one_annotation(tmp_path):
    _stimuli(tmp_path)
    events = pd.DataFrame(
        {"onset": [0.0, 50.0], "duration": [10.0, 10.0], "stim_id": ["stim-story1"] * 2}
    )
    expanded, sources = expand_presentations(events, annotation="words", dataset_root=tmp_path)
    assert expanded["onset"].tolist() == [0.0, 1.5, 3.0, 50.0, 51.5, 53.0]
    assert list(sources) == ["stim-story1"]


@pytest.mark.parametrize(
    "events,message",
    [
        ({"onset": [0.0], "duration": [1.0]}, "no stim_id column"),
        ({"onset": [0.0], "duration": [1.0], "stim_id": [None]}, "no stim_id values"),
        ({"onset": [0.0], "duration": [1.0], "stim_id": ["story1"]}, "stim-<label>"),
        ({"onset": [0.0], "duration": [2.0], "stim_id": ["stim-story1"]}, "end after"),
        (
            {"onset": [0.0], "duration": [9.0], "stim_id": ["stim-story1"], "word_class": ["x"]},
            "occur in both",
        ),
    ],
)
def test_expansion_rejects_invalid_presentations(tmp_path, events, message):
    _stimuli(tmp_path)
    with pytest.raises(ValueError, match=message):
        expand_presentations(pd.DataFrame(events), annotation="words", dataset_root=tmp_path)


def test_zero_duration_presentation_skips_the_length_check(tmp_path):
    _stimuli(tmp_path)
    events = pd.DataFrame({"onset": [5.0], "duration": [0.0], "stim_id": ["stim-story1"]})
    expanded, _ = expand_presentations(events, annotation="words", dataset_root=tmp_path)
    assert expanded["onset"].tolist() == [5.0, 6.5, 8.0]


def test_missing_annotations_and_parts_are_reported_at_planning(tmp_path):
    events = _run_events(tmp_path)
    run = tmp_path / "sub-01/func/sub-01_task-story_run-1_bold.nii.gz"
    _stimuli(tmp_path, stim="stim-story1").rename(
        tmp_path / "stimuli/stim-story1_annot-phonemes_events.tsv"
    )
    with pytest.raises(StimulusAnnotationError) as caught:
        annotation_inputs(events, run, "words")
    message = str(caught.value)
    assert "no annot-words annotation" in message
    assert "Annotations present: phonemes." in message
    assert f"nro stimuli sync {tmp_path.name}" in message
    _stimuli(tmp_path, stim="stim-story1_part-1")
    with pytest.raises(ValueError, match="parts are not supported"):
        annotation_inputs(events, run, "words")


def test_annotation_label_is_validated_and_compiled_only_when_present(tmp_path):
    config = ConfigStore().load_configuration("contrasts", "main").values
    plain = {key: value for key, value in WORDS.items() if key != "annotation"}
    without = compile_model(plain, "story/plain", config)
    assert "annotation" not in without["Nodes"][0]["Model"]["Software"]["nro"]
    with_words = compile_model(WORDS, "story/words", config)
    assert with_words["Nodes"][0]["Model"]["Software"]["nro"]["annotation"] == "words"
    assert "annotation" not in scientific_model(plain)
    assert scientific_model(WORDS)["annotation"] == "words"
    assert scientific_model(WORDS) != scientific_model({**WORDS, "annotation": "phonemes"})
    with pytest.raises(ValueError, match="BIDS label"):
        validate_task_model({**WORDS, "annotation": "word-level"})


def test_direct_inputs_include_annotation_files(tmp_path):
    path = _stimuli(tmp_path)
    events = _run_events(tmp_path)
    bold = tmp_path / "sub-01/func/sub-01_task-story_run-1_bold.nii.gz"
    bold.write_bytes(b"")
    runs = discover_raw_runs(tmp_path / "sub-01")
    paths = direct_inputs(
        runs,
        {"input_filter": None},
        "01",
        {"task": "story", "model": "words"},
        model=WORDS,
    )
    assert paths == (events, path)


def _annotation_project(tmp_path):
    root = _synthetic_project(tmp_path, "volume")
    words = _stimuli(
        root,
        words={
            "onset": np.arange(0.0, 80.0, 2.0),
            "duration": [0.5] * 40,
            "word_class": ["content", "function"] * 20,
        },
    )
    for index in (1, 2):
        stem = f"sub-01_ses-01_task-langlocSN_run-{index:02}"
        _write(
            root / "sub-01/ses-01/func" / f"{stem}_events.tsv",
            {"onset": [10.0], "duration": [80.0], "stim_id": ["stim-story1"]},
        )
    config = ConfigStore().load_configuration("contrasts", "main").values
    config["noise_model"] = "ols"
    kwargs = dict(
        runs=discover_raw_runs(root / "sub-01"),
        participant="01",
        project_root=root,
        func_id="main",
        anat_id="main",
        config_id="main",
        model_id="langlocSN/words",
        model=WORDS,
        config=config,
        space="T1w",
        smoothing=0,
        work_root=tmp_path / "work",
    )
    return root, run_module(**kwargs), words, kwargs


def test_module_fits_an_annotation_model_and_records_sources(tmp_path):
    root, path, words, _ = _annotation_project(tmp_path)
    assert validate_completion(path)[0]
    run_root = path.parents[2] / "ses-01/task-langlocSN/node-run"
    manifest = read_public_json(next(run_root.glob("*run-01_manifest.json")))
    assert manifest["annotation"]["label"] == "words"
    assert manifest["annotation"]["stimuli"]["stim-story1"]["path"] == str(words)
    expanded = pd.read_csv(next(run_root.glob("*run-01_annot-words_events.tsv")), sep="\t")
    assert expanded["onset"].iloc[:2].tolist() == [10.0, 12.0]
    design = json.loads(next(run_root.glob("*run-01_design.json")).read_text())
    assert design["PredictorSources"]["word_class.content"] == "events.tsv+annot-words"


def test_planning_names_project_participant_and_model(tmp_path):
    _run_events(tmp_path)
    (tmp_path / "sub-01/func/sub-01_task-story_run-1_bold.nii.gz").write_bytes(b"")
    runs = discover_raw_runs(tmp_path / "sub-01")
    context = SimpleNamespace(project="demo", participant="01", source_markup=None)
    with pytest.raises(StimulusAnnotationError, match=r"^demo/sub-01, model story/words: .*sync"):
        _planned_inputs(
            context,
            runs,
            {"input_filter": None},
            {"task": "story", "model": "words"},
            "story/words",
            WORDS,
        )


def test_draft_from_annotation_and_report_available_annotations(tmp_path):
    _stimuli(tmp_path)
    events = _run_events(tmp_path)
    lines = []
    draft = yaml.safe_load(
        model_draft(
            [events],
            annotation="words",
            report=lines.append,
            choose=lambda columns: "word_class",
        )
    )
    assert draft["annotation"] == "words"
    assert draft["conditions"] == "word_class"
    assert set(draft["contrasts"]) == {"content", "function"}
    lines.clear()
    plain = yaml.safe_load(model_draft([events], report=lines.append))
    assert plain["conditions"] == "trial_type"
    assert any("stim-story1 annotations: words" in line for line in lines)


def test_annotation_edits_refit_and_manifest_paths_are_portable(tmp_path):
    root, path, words, kwargs = _annotation_project(tmp_path)
    run_root = path.parents[2] / "ses-01/task-langlocSN/node-run"
    raw = json.loads(next(run_root.glob("*run-01_manifest.json")).read_text())
    assert raw["annotation"]["stimuli"]["stim-story1"]["path"] != str(words)
    assert str(tmp_path) not in json.dumps(raw["annotation"])
    before = path.stat().st_mtime_ns
    table = pd.read_csv(words, sep="\t")
    table.loc[0, "word_class"] = "function"
    table.to_csv(words, sep="\t", index=False)
    stamp = before + 10**9
    os.utime(words, ns=(stamp, stamp))
    run_module(**kwargs)
    assert path.stat().st_mtime_ns != before
