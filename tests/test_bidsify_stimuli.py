"""Stimulus annotation snapshots at bidsify review and copies at publication."""

import json
from pathlib import Path

import pytest
from test_bidsify import ingestion, prepared_session, save_decision  # noqa: F401
from test_stimulus_catalog import WORDS, _catalog

from nro.bidsify import review
from nro.bidsify.errors import BidsificationError
from nro.bidsify.publication import publish
from nro.bidsify.stimuli import publish_stimuli, snapshot_stimuli

EVENTS = "onset\tduration\tstim_id\n5\t60\tstim-story1\n"


def _config(tmp_path):
    store = _catalog(tmp_path / "definitions")
    (store / "events").mkdir(exist_ok=True)
    return {"event_store": [str(store / "events")], "event_rules": []}


def test_snapshot_resolves_catalog_annotations(tmp_path):
    snapshot, files = snapshot_stimuli(EVENTS, _config(tmp_path))
    name = "stim-story1_annot-words_events.tsv"
    assert files[name] == WORDS
    assert json.loads(files["stim-story1_annot-words_events.json"]) == {"NROVersion": 1}
    assert snapshot["annotations"][name]["version"] == 1
    assert [row["stimulus_id"] for row in snapshot["stimulus_rows"]] == ["stim-story1"]
    assert snapshot["annotation_rows"][0]["annot_id"] == "words"


def test_snapshot_ignores_events_without_stimuli_and_rejects_unknown(tmp_path):
    config = _config(tmp_path)
    assert snapshot_stimuli("onset\tduration\n1\t1\n", config) == ({}, {})
    with pytest.raises(BidsificationError, match=r"not in the stimulus catalog: \['stim-x'\]"):
        snapshot_stimuli("onset\tduration\tstim_id\n1\t1\tstim-x\n", config)


def _snapshot_record(store, row, tmp_path, *, words=None):
    snapshot, files = snapshot_stimuli(EVENTS, _config(tmp_path))
    if words is not None:
        files["stim-story1_annot-words_events.tsv"] = words
    row["acquisitions"] = [{"id": "bold-1", "stimuli": snapshot}]
    with store.review_session(row["id"]) as token:
        return store.update(
            row,
            expected_revision=row["revision"],
            review_token=token,
            stimulus_files={"bold-1": files},
        )


def test_store_writes_content_addressed_stimulus_snapshots(ingestion, tmp_path):  # noqa: F811
    _registry, store, row = ingestion
    saved = _snapshot_record(store, row, tmp_path)
    files = saved["acquisitions"][0]["stimuli"]["files"]
    table = Path(files["stim-story1_annot-words_events.tsv"])
    assert table.read_text() == WORDS
    assert table.parent == Path(row["config"]["staging"]) / row["id"] / "stimuli"
    with pytest.raises(ValueError, match="does not match the reviewed snapshot"):
        _snapshot_record(store, saved, tmp_path, words=WORDS + "1\t1\tx\n")


def test_publication_copies_stimuli_once_and_keeps_differing_copies(ingestion, tmp_path):  # noqa: F811
    registry, store, row = ingestion
    row = _snapshot_record(store, row, tmp_path)
    row = prepared_session(registry, store, row)
    result = publish(row, registry)
    assert result["stimulus_messages"] == ["Copied stim-story1_annot-words_events.tsv v1"]
    project = registry.paths.bids_root / "demo"
    stimuli = project / "stimuli"
    assert (stimuli / "stim-story1_annot-words_events.tsv").read_text() == WORDS
    sidecar = json.loads((stimuli / "stim-story1_annot-words_events.json").read_text())
    assert sidecar["NROStimulusSource"]["version"] == 1
    assert "stim-story1" in (stimuli / "stimuli.tsv").read_text()
    assert publish(row, registry)["stimulus_messages"] == []
    (stimuli / "stim-story1_annot-words_events.tsv").write_text(WORDS + "9\t1\tlocal\n")
    messages = publish_stimuli(row, project)
    assert len(messages) == 1 and "Kept the existing" in messages[0]
    assert "nro stimuli sync demo" in messages[0]
    assert (stimuli / "stim-story1_annot-words_events.tsv").read_text().endswith("local\n")


def test_publication_rejects_a_changed_snapshot(ingestion, tmp_path):  # noqa: F811
    _registry, store, row = ingestion
    row = _snapshot_record(store, row, tmp_path)
    snapshot = row["acquisitions"][0]["stimuli"]
    Path(snapshot["files"]["stim-story1_annot-words_events.tsv"]).write_text("changed\n")
    with pytest.raises(BidsificationError, match="changed after review"):
        publish_stimuli(row, tmp_path / "project")
    assert not (tmp_path / "project").exists()


def test_review_snapshots_stimuli_and_retries_unknown_ones(tmp_path, monkeypatch):
    config = _config(tmp_path)
    unknown, known = tmp_path / "unknown.tsv", tmp_path / "known.tsv"
    unknown.write_text("onset\tduration\tstim_id\n1\t1\tstim-x\n")
    known.write_text(EVENTS)
    answers = iter([str(unknown), str(known), "y"])
    monkeypatch.setattr(review, "ask", lambda prompt, default="": next(answers))
    item = {"id": "bold-1", "entities": {"task": "story"}, "metadata": {}}
    text, files = review._events({"config": config}, item)
    assert text == EVENTS
    assert set(files) == {
        "stim-story1_annot-words_events.tsv",
        "stim-story1_annot-words_events.json",
    }
    assert review._snapshotted(item) == {"stim-story1"}
