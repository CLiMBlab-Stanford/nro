"""Publication of staged stimulus annotations into the definitions catalog."""

import contextlib
import json
from pathlib import Path

import pytest

from nro.definitions.repository import create_store, validate_store
from nro.definitions.stimuli import ANNOTATION_LEVELS, StimulusCatalog, read_table
from nro.definitions.stimulus_publish import describe, plan_publish, publish_stimuli

STIMULI = "stimulus_id\ttype\tlicense\tpresent\tdescription\n"
WORDS = "onset\tduration\ttext\n0.0\t0\tOnce\n0.5\t0\tupon\n"


def _staged(directory, stories=("story1",), words=WORDS, sidecar=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stimuli.tsv").write_text(
        STIMULI + "".join(f"stim-{s}\taudio\tn/a\tfalse\tStory {s}\n" for s in stories)
    )
    for story in stories:
        (directory / f"stim-{story}_annot-words_events.tsv").write_text(words)
        (directory / f"stim-{story}_annot-words_events.json").write_text(
            json.dumps(sidecar or {"Description": "Words", "text": {"Description": "Word"}})
        )
    return directory


def _publish(staged, root, **kwargs):
    return publish_stimuli(
        staged, root=root, validate=lambda c: validate_store(c, published=root), **kwargs
    )


@pytest.fixture
def root(tmp_path, monkeypatch):
    from nro.site import configuration as site

    store = create_store(tmp_path / "store")
    monkeypatch.setattr(site, "definition_write", lambda path: contextlib.nullcontext())
    monkeypatch.setattr(site, "require_definition_write", lambda path: None)
    return store


def _statuses(plan):
    return {change.name: (change.status, change.version) for change in plan.annotations}


def test_first_publication_assigns_version_one_and_adds_rows(tmp_path, root):
    plan = _publish(_staged(tmp_path / "out", ("story1", "story2")), root)
    assert _statuses(plan) == {
        "stim-story1_annot-words_events": ("new", 1),
        "stim-story2_annot-words_events": ("new", 1),
    }
    assert plan.added_rows == {
        "stimuli.tsv": ("stim-story1", "stim-story2"),
        "annotations.tsv": ("words",),
    }
    assert validate_store(root)["stimulus_annotations"] == 2
    header = (root / "stimuli/stimuli.tsv").read_text().splitlines()[0]
    assert header == "stimulus_id\ttype\tlicense\tpresent\tdescription"
    sidecar = json.loads((root / "stimuli/stim-story1_annot-words_events.json").read_text())
    assert sidecar["NROVersion"] == 1
    assert sidecar["text"] == {"Description": "Word"}


def test_unchanged_publication_writes_nothing(tmp_path, root):
    _publish(_staged(tmp_path / "out"), root)
    plan = _publish(tmp_path / "out", root)
    assert not plan.changed
    assert _statuses(plan) == {"stim-story1_annot-words_events": ("unchanged", 1)}
    assert "Nothing to publish" in describe(plan, dry_run=False)[-1]


def test_changed_values_raise_the_version_and_metadata_keeps_it(tmp_path, root):
    _publish(_staged(tmp_path / "v1"), root)
    changed = _staged(tmp_path / "v2", words=WORDS + "1.0\t0\ta\n")
    plan = _publish(changed, root)
    assert _statuses(plan) == {"stim-story1_annot-words_events": ("changed", 2)}
    assert "(was 1)" in describe(plan, dry_run=False)[0]
    assert StimulusCatalog((root,)).get("stim-story1").annotations["words"].version == 2
    metadata = _staged(
        tmp_path / "v3", words=WORDS + "1.0\t0\ta\n", sidecar={"Description": "Spoken words"}
    )
    plan = _publish(metadata, root)
    assert _statuses(plan) == {"stim-story1_annot-words_events": ("metadata", 2)}
    assert list(plan.updates) == [Path("stimuli/stim-story1_annot-words_events.json")]


def test_rows_merge_with_other_suites_and_new_columns(tmp_path, root):
    _publish(_staged(tmp_path / "first", ("story1",)), root)
    other = tmp_path / "other"
    other.mkdir()
    (other / "stimuli.tsv").write_text(
        "stimulus_id\ttype\tlicense\tpresent\tduration\nstim-passage1\taudio\tn/a\tfalse\t20.5\n"
    )
    (other / "stim-passage1_annot-words_events.tsv").write_text(WORDS)
    (other / "stim-passage1_annot-words_events.json").write_text('{"Description": "Words"}')
    plan = _publish(other, root)
    assert plan.added_rows["stimuli.tsv"] == ("stim-passage1",)
    table = read_table(root / "stimuli/stimuli.tsv")
    assert list(table["stimulus_id"]) == ["stim-story1", "stim-passage1"]
    assert list(table["duration"]) == ["n/a", "20.5"]
    assert list(table["description"]) == ["Story story1", "n/a"]


def test_levels_are_described_by_nro_once(tmp_path, root):
    _publish(_staged(tmp_path / "first", ("story1",)), root)
    assert (root / "stimuli/annotations.tsv").read_text() == (
        f"annot_id\tdescription\nwords\t{ANNOTATION_LEVELS['words']}\n"
    )
    plan = _publish(_staged(tmp_path / "second", ("passage1",)), root)
    assert plan.added_rows["annotations.tsv"] == ()
    assert Path("stimuli/annotations.tsv") not in plan.updates
    assert validate_store(root)["stimulus_annotations"] == 2


def test_dry_run_plans_without_writing(tmp_path, root):
    plan = _publish(_staged(tmp_path / "out"), root, dry_run=True)
    assert plan.changed
    assert not (root / "stimuli/stimuli.tsv").exists()
    assert describe(plan, dry_run=True)[-1].startswith("Dry run: would write 4 files")
    assert Path("stimuli/annotations.tsv") in plan.updates


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda d: (d / "stim-story1_annot-words_events.json").write_text('{"NROVersion": 3}'),
            "remove NROVersion",
        ),
        (lambda d: (d / "stimuli.tsv").write_text(STIMULI), "has no staged stimuli.tsv row"),
        (lambda d: (d / "stimuli.tsv").unlink(), "Missing staged stimuli table"),
        (lambda d: (d / "story1.wav").write_bytes(b""), "Unsupported staged file"),
        (
            lambda d: (d / "annotations.tsv").write_text("annot_id\nwords\n"),
            "nro defines annotation levels",
        ),
        (
            lambda d: (d / "stim-story1_annot-words_events.tsv").rename(
                d / "stim-story1_annot-lemmas_events.tsv"
            ),
            "unknown annotation level 'lemmas'",
        ),
        (lambda d: (d / "stim-story1_annot-words_events.json").unlink(), "needs both"),
        (
            lambda d: (d / "stim-story1_annot-words_events.tsv").write_text("onset\n0\n"),
            "duration",
        ),
    ],
)
def test_invalid_staged_directories_are_refused(tmp_path, root, change, message):
    staged = _staged(tmp_path / "out")
    change(staged)
    with pytest.raises(ValueError, match=message):
        _publish(staged, root)
    assert not (root / "stimuli/stimuli.tsv").exists()


def test_staged_rows_may_not_change_published_rows(tmp_path, root):
    _publish(_staged(tmp_path / "first"), root)
    staged = _staged(tmp_path / "second")
    text = (staged / "stimuli.tsv").read_text().replace("Story story1", "A new description")
    (staged / "stimuli.tsv").write_text(text)
    with pytest.raises(
        ValueError, match=r"stimuli.tsv: stim-story1 differs from the published row"
    ):
        plan_publish(staged, root)


def test_cli_reports_and_publishes(tmp_path, root, monkeypatch, capsys):
    from nro.bin import stimuli as command
    from nro.site import configuration as site

    monkeypatch.setattr(site, "definitions_root", lambda: root)
    monkeypatch.setattr(
        "nro.definitions.repository.store_validator",
        lambda store: lambda candidate: validate_store(candidate, published=store),
    )
    staged = _staged(tmp_path / "out")
    command.main(["publish", str(staged), "--dry-run"])
    output = capsys.readouterr().out
    assert "new        stim-story1_annot-words_events  version 1" in output
    assert not (root / "stimuli/stimuli.tsv").exists()
    command.main(["publish", str(staged)])
    assert "nro stimuli sync PROJECT" in capsys.readouterr().out
    assert validate_store(root)["stimulus_annotations"] == 1
    with pytest.raises(SystemExit, match="does not exist"):
        command.main(["publish", str(tmp_path / "missing")])
