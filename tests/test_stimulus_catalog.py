"""Stimulus annotation catalog, version rules, and project sync."""

import json
import subprocess

import pandas as pd
import pytest

from nro.definitions.migrations import update_store
from nro.definitions.repository import create_store, validate_store
from nro.definitions.stimuli import StimulusCatalog, load_catalog, validate_catalog
from nro.definitions.stimulus_sync import (
    AHEAD,
    CURRENT,
    LOCAL_EDIT,
    MISSING,
    STALE,
    UNMANAGED,
    apply_sync,
    plan_sync,
)

STIMULI = (
    "stimulus_id\ttype\tlicense\tpresent\tdescription\nstim-story1\taudio\tn/a\tfalse\tA story\n"
)
WORDS = "onset\tduration\tword\n0.0\t0.4\tonce\n0.5\t0.3\tupon\n"


def _catalog(root, *, words=WORDS, version=1, extra=None):
    directory = root / "stimuli"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stimuli.tsv").write_text(STIMULI)
    (directory / "annotations.tsv").write_text("annot_id\tdescription\nwords\tWord onsets\n")
    (directory / "stim-story1_annot-words_events.tsv").write_text(words)
    (directory / "stim-story1_annot-words_events.json").write_text(
        json.dumps({"NROVersion": version, **(extra or {})})
    )
    return root


def _project(root, stim_ids=("stim-story1",)):
    func = root / "sub-01/func"
    func.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"onset": [5.0] * len(stim_ids), "duration": [60.0] * len(stim_ids), "stim_id": stim_ids}
    ).to_csv(func / "sub-01_task-story_run-1_events.tsv", sep="\t", index=False)
    return root


def _errors(root, **kwargs):
    return validate_catalog(root, **kwargs)[2]


def test_valid_catalog_loads_rows_annotations_and_versions(tmp_path):
    stimuli, errors = load_catalog(_catalog(tmp_path))
    assert errors == []
    stimulus = stimuli["stim-story1"]
    assert stimulus.row["present"] == "false"
    assert stimulus.annotations["words"].version == 1
    assert stimulus.descriptions["words"]["description"] == "Word onsets"
    assert validate_catalog(tmp_path)[:2] == (1, 1)


def test_store_without_catalog_is_empty(tmp_path):
    assert load_catalog(tmp_path) == ({}, [])


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda d: (d / "stim-story1_annot-words_events.json").unlink(), "JSON sidecar"),
        (
            lambda d: (d / "stim-story1_annot-words_events.json").write_text("{}"),
            "NROVersion must be a positive integer",
        ),
        (
            lambda d: (d / "stim-story1_annot-words_events.json").write_text(
                '{"NROVersion": 1, "NROStimulusSource": {}}'
            ),
            "only in project copies",
        ),
        (lambda d: (d / "stim-story1_audio.wav").write_bytes(b"RIFF"), "media files"),
        (
            lambda d: (d / "stim-story1_part-1_annot-words_events.tsv").write_text(WORDS),
            "Unsupported stimulus catalog file",
        ),
        (
            lambda d: (d / "stimuli.tsv").write_text(STIMULI.replace("false", "true")),
            "present must be false",
        ),
        (
            lambda d: (d / "stimuli.tsv").write_text(STIMULI.split("\n")[0] + "\n"),
            "no stimuli.tsv row",
        ),
        (
            lambda d: (d / "annotations.tsv").write_text("annot_id\tdescription\n"),
            "annot_id words is not listed",
        ),
        (
            lambda d: (d / "stim-story1_annot-words_events.tsv").write_text("onset\n1\n"),
            "onset and duration",
        ),
    ],
)
def test_invalid_catalogs_are_rejected(tmp_path, change, message):
    change(_catalog(tmp_path) / "stimuli")
    assert any(message in error for error in _errors(tmp_path)), _errors(tmp_path)


def test_stimulus_ids_must_be_unique_across_the_chain(tmp_path):
    child, parent = _catalog(tmp_path / "child"), _catalog(tmp_path / "parent")
    assert any("unique across" in e for e in _errors(child, inherited_roots=(parent,)))


@pytest.mark.parametrize(
    "words,version,message",
    [
        (WORDS, 1, None),
        (WORDS + "1.0\t0.2\ta\n", 1, "must be higher than 1"),
        (WORDS + "1.0\t0.2\ta\n", 2, None),
        (WORDS, 2, "must stay 1"),
    ],
)
def test_published_versions_must_follow_content(tmp_path, words, version, message):
    published = _catalog(tmp_path / "published")
    candidate = _catalog(tmp_path / "candidate", words=words, version=version)
    errors = _errors(candidate, published=published)
    assert (errors == []) if message is None else any(message in e for e in errors), errors


def test_new_annotations_start_at_version_one(tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    errors = _errors(_catalog(tmp_path / "candidate", version=3), published=published)
    assert any("must start at NROVersion 1" in e for e in errors)


def _store(tmp_path):
    root = create_store(tmp_path / "store")
    directory = _catalog(tmp_path / "files") / "stimuli"
    update_store(
        root,
        {f"stimuli/{path.name}": path.read_bytes() for path in directory.iterdir()},
        validate=lambda candidate: validate_store(candidate, published=root),
    )
    return root


def test_apply_publishes_catalog_and_enforces_version_bumps(tmp_path):
    root = _store(tmp_path)
    assert validate_store(root)["stimulus_annotations"] == 1
    validate = lambda candidate: validate_store(candidate, published=root)  # noqa: E731
    changed = {"stimuli/stim-story1_annot-words_events.tsv": (WORDS + "1\t1\tx\n").encode()}
    with pytest.raises(ValueError, match="must be higher than 1"):
        update_store(root, changed, validate=validate)
    bumped = {
        **changed,
        "stimuli/stim-story1_annot-words_events.json": b'{"NROVersion": 2}',
    }
    update_store(root, bumped, validate=validate)
    assert StimulusCatalog((root,)).get("stim-story1").annotations["words"].version == 2


def test_direct_edits_are_checked_against_git_head(tmp_path):
    root = _store(tmp_path)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "store"], check=True)
    (root / "stimuli/stim-story1_annot-words_events.tsv").write_text(WORDS + "1\t1\tx\n")
    errors = _errors(root, published=root)
    assert any("must be higher than 1" in e for e in errors)
    (root / "stimuli/stim-story1_annot-words_events.json").write_text('{"NROVersion": 2}')
    assert _errors(root, published=root) == []


def test_direct_edits_without_git_history_cannot_be_checked(tmp_path):
    root = _store(tmp_path)
    (root / "stimuli/stim-story1_annot-words_events.tsv").write_text(WORDS + "1\t1\tx\n")
    assert any("cannot check NROVersion" in e for e in _errors(root, published=root))


def _statuses(plan):
    return {item.name: item.status for item in plan.annotations}


def test_sync_copies_rows_and_annotations_then_reports_current(tmp_path):
    catalog = StimulusCatalog((_catalog(tmp_path / "store"),))
    project = _project(tmp_path / "demo")
    plan = plan_sync(project, catalog)
    assert _statuses(plan) == {"stim-story1_annot-words_events.tsv": MISSING}
    assert [item.name for item in apply_sync(plan)] == ["stim-story1_annot-words_events.tsv"]
    stimuli = project / "stimuli"
    assert (stimuli / "stim-story1_annot-words_events.tsv").read_text() == WORDS
    sidecar = json.loads((stimuli / "stim-story1_annot-words_events.json").read_text())
    assert sidecar["NROVersion"] == 1
    assert sidecar["NROStimulusSource"]["path"] == "stimuli/stim-story1_annot-words_events.tsv"
    assert sidecar["NROStimulusSource"]["version"] == 1
    assert (stimuli / "stimuli.tsv").read_text() == STIMULI
    assert "words\tWord onsets" in (stimuli / "annotations.tsv").read_text()
    again = plan_sync(project, catalog)
    assert _statuses(again) == {"stim-story1_annot-words_events.tsv": CURRENT}
    assert not again.stimulus_rows and not again.annotation_rows
    assert apply_sync(again) == []


def test_sync_replaces_stale_copies_only_on_request(tmp_path):
    project = _project(tmp_path / "demo")
    apply_sync(plan_sync(project, StimulusCatalog((_catalog(tmp_path / "v1"),))))
    newer = StimulusCatalog((_catalog(tmp_path / "v2", words=WORDS + "1\t1\tx\n", version=2),))
    plan = plan_sync(project, newer)
    assert _statuses(plan) == {"stim-story1_annot-words_events.tsv": STALE}
    assert apply_sync(plan) == []
    assert len(apply_sync(plan, replace=True)) == 1
    assert _statuses(plan_sync(project, newer)) == {"stim-story1_annot-words_events.tsv": CURRENT}


def test_sync_never_overwrites_blocked_files(tmp_path):
    project = _project(tmp_path / "demo")
    catalog = StimulusCatalog((_catalog(tmp_path / "v2", version=2),))
    apply_sync(plan_sync(project, catalog))
    table = project / "stimuli/stim-story1_annot-words_events.tsv"
    sidecar = project / "stimuli/stim-story1_annot-words_events.json"
    older = StimulusCatalog((_catalog(tmp_path / "v1"),))
    assert _statuses(plan_sync(project, older)) == {table.name: AHEAD}
    table.write_text(WORDS + "9\t1\tlocal\n")
    plan = plan_sync(project, catalog)
    assert _statuses(plan) == {table.name: LOCAL_EDIT}
    assert apply_sync(plan, replace=True) == []
    assert table.read_text().endswith("local\n")
    sidecar.write_text('{"NROVersion": 2}')
    assert _statuses(plan_sync(project, catalog)) == {table.name: UNMANAGED}
    assert plan_sync(project, catalog).blocked


def test_sync_reports_unknown_stimuli_and_ignores_annotation_files(tmp_path):
    catalog = StimulusCatalog((_catalog(tmp_path / "store"),))
    project = _project(tmp_path / "demo", stim_ids=("stim-story1", "stim-other"))
    apply_sync(plan_sync(project, catalog))
    plan = plan_sync(project, catalog)
    assert plan.unknown == ["stim-other"]
    assert set(plan.references) == {"stim-story1", "stim-other"}


def test_cli_dry_run_and_branch_refusal(tmp_path, monkeypatch, capsys):
    from nro.bin import stimuli as command
    from nro.site import configuration as site

    store = _catalog(tmp_path / "store")
    _project(tmp_path / "bids/demo")
    monkeypatch.setattr(site, "bids_root", lambda: tmp_path / "bids")
    monkeypatch.setattr(site, "definitions_roots", lambda: (store,))
    monkeypatch.setattr(site, "installation_record", lambda: {"mode": "branch"})
    command.main(["sync", "demo", "--dry-run"])
    assert "missing   stim-story1_annot-words_events.tsv v1" in capsys.readouterr().out
    assert not (tmp_path / "bids/demo/stimuli").exists()
    with pytest.raises(SystemExit):
        command.main(["sync", "demo"])
    monkeypatch.setattr(site, "installation_record", lambda: {})
    command.main(["sync", "demo"])
    assert "copied    stim-story1_annot-words_events.tsv v1" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        command.main(["sync", "missing"])


def test_cli_refuses_writes_while_project_attempts_are_active(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import contextmanager

    from nro.bin import stimuli as command
    from nro.orchestration import registry as registry_module
    from nro.site import configuration as site

    database = tmp_path / "registry.sqlite"
    db = sqlite3.connect(database)
    db.executescript(
        """CREATE TABLE work_items (id INTEGER, project TEXT, module TEXT, participant TEXT);
           CREATE TABLE attempts (work_item_id INTEGER, state TEXT);
           INSERT INTO work_items VALUES (1, 'demo', 'contrasts', '01');
           INSERT INTO attempts VALUES (1, 'running');"""
    )
    db.commit()
    locked = []

    class FakeRegistry:
        @classmethod
        def for_project(cls, project, **kwargs):
            return cls()

        def existing_database_path(self):
            return database

        @contextmanager
        def connection(self, *, write=False):
            locked.append(write)
            yield db

    store = _catalog(tmp_path / "store")
    _project(tmp_path / "bids/demo")
    monkeypatch.setattr(site, "bids_root", lambda: tmp_path / "bids")
    monkeypatch.setattr(site, "definitions_roots", lambda: (store,))
    monkeypatch.setattr(site, "installation_record", lambda: {})
    monkeypatch.setattr(registry_module, "Registry", FakeRegistry)
    with pytest.raises(SystemExit, match=r"active derivative attempts \(contrasts:sub-01\)"):
        command.main(["sync", "demo"])
    assert locked == [True]
    assert not (tmp_path / "bids/demo/stimuli").exists()
    command.main(["sync", "demo", "--dry-run"])
    assert locked == [True]
    db.execute("UPDATE attempts SET state='succeeded'")
    command.main(["sync", "demo"])
    assert (tmp_path / "bids/demo/stimuli/stim-story1_annot-words_events.tsv").is_file()
