"""Definition drafts, validation, and guarded publication in isolated stores."""

from pathlib import Path
import fcntl
import shutil
import subprocess

import pytest
import yaml

from nro.bin.create import main as create
from nro.bin.edit import main as edit
from nro.bin.delete import main as delete
from nro.configuration import store as store_module
from nro.configuration.authoring import definition_target, validate_definition
from nro.configuration.store import ConfigStore
from nro.engine import definition_editor
from nro.firstlevels.authoring import discover_event_files, model_draft
from nro.firstlevels.task_models import validate_task_model


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    shutil.copytree(ConfigStore().root, root)
    monkeypatch.setattr(store_module, "definitions_root", lambda: root)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    return ConfigStore()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _events(path, labels, column="trial_type"):
    return _write(path, f"onset\tduration\t{column}\tresponse_time\n" +
                  "".join(f"{i * 2}\t1\t{label}\t0.5\n" for i, label in enumerate(labels)))


def _interactive(monkeypatch, editor, responses):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("VISUAL", "test-editor --wait")
    monkeypatch.setenv("EDITOR", "unused-editor")
    monkeypatch.setattr(definition_editor.subprocess, "run", editor)
    answers = iter(responses)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))


def test_event_discovery_uses_inheritance_and_source_runs(tmp_path):
    bids = tmp_path / "bids"
    root_table = _events(bids / "alpha" / "task-newtask_events.tsv", ["A"])
    _write(bids / "alpha/sub-01/func/sub-01_task-newtask_bold.nii.gz", "x")
    _write(bids / "alpha/sub-02/func/sub-02_task-newtask_bold.nii.gz", "x")
    specific = _events(bids / "beta/sub-01/ses-1/func/sub-01_ses-1_task-newtask_events.tsv", ["B"])
    _write(specific.with_name("sub-01_ses-1_task-newtask_bold.nii.gz"), "x")
    _write(bids / "alpha/sub-01/derivatives/copy/func/sub-01_task-newtask_bold.nii.gz", "x")
    assert discover_event_files("newtask", bids) == (root_table, specific)
    assert discover_event_files("newtask", bids, projects=("alpha",), participants=("sub-02",)) == (root_table,)
    with pytest.raises(ValueError, match="Unknown BIDS project"):
        discover_event_files("newtask", bids, projects=("missing",))
    with pytest.raises(ValueError, match="Unknown participant"):
        discover_event_files("newtask", bids, participants=("99",))
    with pytest.raises(ValueError, match="No source BOLD"):
        discover_event_files("missing", bids)
    root_table.unlink()
    with pytest.raises(FileNotFoundError, match="No applicable BIDS"):
        discover_event_files("newtask", bids)


def test_model_draft_unions_conditions_without_inferring_other_predictors(tmp_path):
    paths = [_events(tmp_path / "one.tsv", ["S", "S", "n/a"]),
             _events(tmp_path / "two.tsv", ["N", "S"])]
    messages = []
    model = validate_task_model(yaml.safe_load(model_draft(paths, report=messages.append)))
    assert model["conditions"] == "trial_type"
    assert model["contrasts"] == {"N": {"N": 1}, "S": {"S": 1}}
    assert model["model_set"] == []
    assert model["hrf"] == "spm"
    assert "predictors" not in model
    assert any("1/2 tables" in line for line in messages)
    assert any("no condition label" in line for line in messages)


def test_draft_requires_resolution_of_ambiguous_condition_columns(tmp_path):
    table = _events(tmp_path / "events.tsv", ["A", "B"], column="condition")
    with pytest.raises(ValueError, match="Choose --conditions"):
        model_draft([table])
    model = yaml.safe_load(model_draft([table], choose=lambda columns: "condition"))
    assert model["conditions"] == "condition"
    with pytest.raises(ValueError, match="must occur in every"):
        model_draft([table], conditions="absent")
    other = _events(tmp_path / "other.tsv", ["C"])
    with pytest.raises(ValueError, match="trial_type is absent"):
        model_draft([table, other])


def test_draft_maps_unsafe_contrast_names_without_losing_conditions(tmp_path):
    table = _events(tmp_path / "events.tsv", ["A B", "A-B", "trial_type.C"])
    model = yaml.safe_load(model_draft([table]))
    assert len(model["contrasts"]) == 3
    assert {key for weights in model["contrasts"].values() for key in weights} == {
        "A B", "A-B", "trial_type.trial_type.C",
    }
    from nro.firstlevels.compiler import task_node
    node = task_node(model)
    assert {name for contrast in node["Contrasts"] for name in contrast["ConditionList"]} == {
        "trial_type.A B", "trial_type.A-B", "trial_type.trial_type.C",
    }


def test_model_draft_uses_compact_shorthand_with_equivalent_compilation(tmp_path):
    from copy import deepcopy
    from nro.firstlevels.compiler import task_node

    text = model_draft([_events(tmp_path / "events.tsv", ["E", "H"])])
    assert "  E: {E: 1}\n" in text
    model = yaml.safe_load(text)
    qualified = deepcopy(model)
    qualified["contrasts"] = {name: {f"trial_type.{key}": weight for key, weight in weights.items()}
                              for name, weights in model["contrasts"].items()}
    assert task_node(model) == task_node(qualified)


@pytest.mark.parametrize("text", [
    "onset\tduration\ttrial_type\n",
    "onset\tduration\ttrial_type\n0\t-1\tA\n",
    "onset\tduration\ttrial_type\nNaN\t1\tA\n",
    "onset\tduration\ttrial_type\ttrial_type\n0\t1\tA\tB\n",
])
def test_malformed_events_do_not_create_partial_drafts(tmp_path, text):
    with pytest.raises(ValueError):
        model_draft([_write(tmp_path / "events.tsv", text)])


def test_create_model_draft_and_register_local_file(store, tmp_path):
    events = _events(tmp_path / "events.tsv", ["A", "B"])
    output = tmp_path / "draft.yml"
    create(["model", "newtask", "--events", str(events), "--output", str(output)])
    target = store.root / "models/newtask/main.yml"
    assert not target.exists()
    create(["model", "newtask", "--file", str(output), "--yes"])
    assert target.read_text() == output.read_text()
    with pytest.raises(SystemExit):
        create(["model", "newtask", "--file", str(output), "--yes"])
    edit(["model", "newtask", "--file", str(output), "--yes"])
    assert yaml.safe_load(target.read_text())["model_set"] == []


def test_config_and_workflow_initialization(store, tmp_path):
    config = tmp_path / "config.yml"
    create(["config", "clean/alternative", "--output", str(config)])
    assert yaml.safe_load(config.read_text()) is None
    assert "# Current main defaults" in config.read_text()
    config.write_text("minimum_temporal_rank: 25\n")
    create(["config", "clean/alternative", "--file", str(config), "--yes"])
    assert store.load_configuration("clean", "alternative").values["minimum_temporal_rank"] == 25
    workflow = _write(tmp_path / "workflow.yml", "clean: alternative\n")
    create(["workflow", "experiment", "--file", str(workflow), "--yes"])
    assert store.resolve("experiment").selections["clean"] == "alternative"
    expanded = tmp_path / "expanded.yml"
    create(["workflow", "copy", "--from", "experiment", "--output", str(expanded)])
    assert yaml.safe_load(expanded.read_text())["clean"] == "alternative"
    assert yaml.safe_load(expanded.read_text())["networks"] == "main"


def test_copy_model_removes_execution_membership(store, tmp_path):
    output = tmp_path / "draft.yml"
    create(["model", "langlocSN/dev", "--from", "langlocSN/main", "--output", str(output)])
    assert yaml.safe_load(output.read_text())["model_set"] == []


@pytest.mark.parametrize("kind,identifier,text", [
    ("model", "newtask", "conditions: trial_type\ncontrasts: {}\n"),
    ("model", "newtask", "conditions: trial_type\nconditions: other\n"),
    ("config", "clean/alternate", "typo: 2\n"),
    ("config", "clean/alternate", "minimum_temporal_rank: wrong\n"),
    ("config", "clean/main", "minimum_temporal_rank: 25\n"),
    ("workflow", "alternate", "clean: absent\n"),
    ("workflow", "alternate", "unknown: main\n"),
])
def test_validation_rejects_invalid_staged_definitions(store, kind, identifier, text):
    with pytest.raises(ValueError):
        validate_definition(store, definition_target(store, kind, identifier), text)


def test_create_existing_routes_to_edit_and_preserves_comments(store, monkeypatch, capsys):
    target = store.configuration_path("clean", "main")
    original = target.read_bytes()

    def editor(command, **kwargs):
        assert command[:2] == ["test-editor", "--wait"]
        assert target.read_bytes() == original
        draft = Path(command[-1])
        assert draft != target
        draft.write_text(draft.read_text() + "\n# Reviewed settings.\n")
        return subprocess.CompletedProcess(command, 0)

    _interactive(monkeypatch, editor, ["y"])
    create(["config", "clean/main"])
    assert target.read_text().endswith("# Reviewed settings.\n")
    assert "already exists; opening for editing" in capsys.readouterr().out


def test_invalid_edit_can_be_corrected_before_publication(store, monkeypatch):
    target = store.root / "workflows/new_workflow.yml"
    calls = []

    def editor(command, **kwargs):
        assert not target.exists()
        calls.append(command)
        Path(command[-1]).write_text("unknown: main\n" if len(calls) == 1 else "clean: main\n")

    _interactive(monkeypatch, editor, ["y", "y"])
    create(["workflow", "new"])
    assert len(calls) == 2
    assert target.read_text() == "clean: main\n"


@pytest.mark.parametrize("outcome", ["cancel", "invalid", "editor_error", "interrupt", "conflict"])
def test_failed_or_cancelled_edits_preserve_store_and_draft(store, monkeypatch, outcome, capsys):
    target = store.workflow_path("main")[1]
    original = target.read_text()
    drafts = []

    def editor(command, **kwargs):
        draft = Path(command[-1])
        drafts.append(draft)
        draft.write_text("unknown: main\n" if outcome == "invalid" else "clean: main\n")
        if outcome == "conflict":
            target.write_text(original + "# Another author's change.\n")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "editor_error":
            raise subprocess.CalledProcessError(1, command)

    _interactive(monkeypatch, editor, ["n"] if outcome in {"cancel", "invalid"} else ["y"])
    if outcome == "cancel":
        edit(["workflow", "main"])
    else:
        with pytest.raises(SystemExit):
            edit(["workflow", "main"])
    expected = original + "# Another author's change.\n" if outcome == "conflict" else original
    assert target.read_text() == expected
    assert drafts[0].is_file()
    assert "Draft retained" in capsys.readouterr().err


def test_missing_edit_and_noninteractive_fallback_are_errors(store):
    for command, argv in ((edit, ["workflow", "absent"]),
                          (create, ["workflow", "main"]),
                          (create, ["workflow", "new"])):
        with pytest.raises(SystemExit):
            command(argv)
    assert not (store.root / "workflows/new_workflow.yml").exists()


@pytest.mark.parametrize("available,expected", [
    ({"vi", "nano"}, "/bin/nano"),
    ({"vi"}, "/bin/vi"),
])
def test_editor_fallback_prefers_nano(store, monkeypatch, available, expected):
    def editor(command, **kwargs):
        assert command[0] == expected

    _interactive(monkeypatch, editor, [])
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr(definition_editor.shutil, "which",
                        lambda name: f"/bin/{name}" if name in available else None)
    edit(["workflow", "main"])


@pytest.mark.parametrize("extra", [["--from", "main"], ["--output", "draft.yml"]])
def test_existing_target_rejects_creation_options(store, monkeypatch, extra):
    _interactive(monkeypatch, lambda *a, **kw: pytest.fail("Editor must not start"), [])
    with pytest.raises(SystemExit):
        create(["workflow", "main", *extra])


def test_guarded_save_rejects_conflicts_and_preserves_modes(tmp_path):
    target = _write(tmp_path / "model.yml", "original")
    target.chmod(0o664)
    with pytest.raises(ValueError, match="changed during editing"):
        definition_editor.save_definition(target, "new", expected=b"different")
    definition_editor.save_definition(target, "new", expected=b"original")
    assert target.read_text() == "new"
    assert target.stat().st_mode & 0o777 == 0o664
    with pytest.raises(ValueError):
        definition_editor.save_definition(target, "overwrite", expected=None)
    link = tmp_path / "link.yml"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        definition_editor.read_definition(link)


def test_guarded_save_respects_another_authors_lock(tmp_path):
    target = _write(tmp_path / "model.yml", "original")
    lock = target.with_name(f".{target.name}.edit.lock")
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another writer"):
            definition_editor.save_definition(target, "new", expected=b"original")
    assert target.read_text() == "original"


def test_local_output_cannot_overwrite_or_bypass_store_publication(store, tmp_path):
    output = _write(tmp_path / "draft.yml", "keep this draft")
    with pytest.raises(SystemExit):
        create(["workflow", "new", "--output", str(output)])
    assert output.read_text() == "keep this draft"
    alias = tmp_path / "store-link"
    alias.symlink_to(store.root, target_is_directory=True)
    with pytest.raises(SystemExit):
        create(["workflow", "new", "--output", str(alias / "workflows/new_workflow.yml")])
    assert not (store.root / "workflows/new_workflow.yml").exists()


def test_delete_model_then_recreate_from_event_defaults(store, tmp_path, capsys):
    target = store.root / "models/newtask/main.yml"
    _write(target, "model_set: main\nconditions: trial_type\ncontrasts:\n  AvB: {A: 1, B: -1}\n")
    original = target.read_bytes()
    unrelated = _write(target.parent / "alternative.yml", "keep this")
    delete(["model", "newtask", "--yes"])
    report = capsys.readouterr().out
    backup = Path(report.split("Recovery copy: ", 1)[1].split(" (temporary", 1)[0])
    assert backup.read_bytes() == original
    assert not target.exists()
    assert unrelated.read_text() == "keep this"
    events = _events(tmp_path / "events.tsv", ["A", "B"])
    draft = tmp_path / "reset.yml"
    create(["model", "newtask", "--events", str(events), "--output", str(draft)])
    create(["model", "newtask", "--file", str(draft), "--yes"])
    model = yaml.safe_load(target.read_text())
    assert model["model_set"] == []
    assert model["contrasts"] == {"A": {"A": 1}, "B": {"B": 1}}


@pytest.mark.parametrize("response", ["n", "y"])
def test_delete_requires_confirmation(store, monkeypatch, response):
    target = _write(store.root / "workflows/experiment_workflow.yml", "{}\n")
    with pytest.raises(SystemExit):
        delete(["workflow", "experiment"])
    assert target.exists()
    _interactive(monkeypatch, lambda *args, **kwargs: pytest.fail("Deletion must not open an editor"), [response])
    delete(["workflow", "experiment"])
    assert target.exists() == (response == "n")


def test_delete_rejects_changed_targets_after_confirmation(store, monkeypatch):
    target = _write(store.root / "workflows/experiment_workflow.yml", "{}\n")
    _interactive(monkeypatch, lambda *args, **kwargs: None, [])

    def confirm(prompt):
        target.write_text("clean: main\n")
        return "y"

    monkeypatch.setattr("builtins.input", confirm)
    with pytest.raises(SystemExit):
        delete(["workflow", "experiment"])
    assert target.read_text() == "clean: main\n"


def test_delete_protects_main_configs_and_warns_about_workflow_references(store, capsys):
    main = store.configuration_path("clean", "main")
    original = main.read_bytes()
    with pytest.raises(SystemExit):
        delete(["config", "clean/main", "--yes"])
    assert main.read_bytes() == original
    alternative = _write(store.configs / "clean/alternative_clean.yml", "{}\n")
    workflow = _write(store.root / "workflows/experiment_workflow.yml", "clean: alternative\n")
    delete(["config", "clean/alternative", "--yes"])
    assert not alternative.exists()
    assert workflow.read_text() == "clean: alternative\n"
    assert "experiment" in capsys.readouterr().out


def test_delete_checks_identity_and_uses_authoring_lock(store):
    target = _write(store.root / "workflows/experiment_workflow.yml", "{}\n")
    for identifier in ("absent", "../experiment"):
        with pytest.raises(SystemExit):
            delete(["workflow", identifier, "--yes"])
    lock = target.with_name(f".{target.name}.edit.lock")
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit):
            delete(["workflow", "experiment", "--yes"])
    assert target.exists()
