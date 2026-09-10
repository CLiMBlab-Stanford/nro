"""External store creation, validation, isolation, and content-based identity."""

import ctypes
import errno
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nro.bin.definitions import main
from nro.configuration import site
from nro.configuration.definitions import _publish, create_store, ensure_store, validate_store
from nro.configuration.store import DERIVATIVE_CLASSES, ConfigStore
from nro.modules.firstlevels.task_models import load_task_model, scientific_model, select_models


def test_create_has_only_generic_starters(tmp_path):
    root = create_store(tmp_path / "store")
    counts = validate_store(root)
    assert counts == dict(configs=2, workflows=3, models=0, event_ids=0, event_tsvs=0, bidsify=1)
    for kind in DERIVATIVE_CLASSES:
        assert not (root / "configs" / kind / f"main_{kind}.yml").exists()
    for kind in set(DERIVATIVE_CLASSES) - {"networks"}:
        assert (root / "configs" / kind / ".gitkeep").is_file()
    assert not (root / ".git").exists()
    assert (root / ".gitignore").is_file()
    assert yaml.safe_load((root / "bidsify/main.yml").read_text())["servers"] == {}
    assert select_models(root=root / "models") == {}


@pytest.mark.parametrize("existing", ["empty", "file", "symlink", "store"])
def test_create_never_overwrites(tmp_path, existing):
    path = tmp_path / "store"
    if existing == "empty":
        path.mkdir()
    elif existing == "file":
        path.write_text("keep")
    elif existing == "symlink":
        path.symlink_to(tmp_path / "absent")
    else:
        create_store(path)
    with pytest.raises(FileExistsError):
        create_store(path)


def test_ensure_reuses_without_changing_bytes(tmp_path):
    root = ensure_store(tmp_path / "store")
    config = root / "configs/clean/alternative_clean.yml"
    config.write_text("standardize: false\n")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert ensure_store(root) == root
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize(
    "relative,text,match",
    [
        ("configs/clean/bad_clean.yml", "typo: 1\n", "typo"),
        ("configs/clean/bad.yaml", "{}\n", "filename"),
        ("workflows/bad_workflow.yml", "clean: absent\n", "absent"),
        ("models/task/bad.yml", "confounds: [motion]\n", "Unsupported"),
        ("events/task/unindexed.tsv", "onset\tduration\n0\t1\n", "Unindexed"),
        ("bidsify/bad.yml", "{}\n", "requires exactly"),
    ],
)
def test_validation_checks_all_variants(tmp_path, relative, text, match):
    root = create_store(tmp_path / "store")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        validate_store(root)


def test_validation_checks_events_and_references(tmp_path):
    root = create_store(tmp_path / "store")
    task = root / "events/task"
    task.mkdir()
    index = {"tasks": ["task"], "files": {"main": {"path": "task/main.tsv", "source_names": []}}}
    (task / "index.yml").write_text(yaml.safe_dump(index))
    with pytest.raises(ValueError, match="missing"):
        validate_store(root)
    (task / "main.tsv").write_text("onset\tduration\n0\t-1\n")
    with pytest.raises(ValueError, match="nonnegative"):
        validate_store(root)
    (task / "main.tsv").write_text("onset\tduration\n0\t1\n")
    assert validate_store(root)["event_tsvs"] == 1


def test_missing_store_fails_but_main_configs_use_packaged_defaults(tmp_path, monkeypatch):
    monkeypatch.setitem(site.DEFAULTS, "definitions", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="does not exist"):
        ConfigStore()
    root = create_store(tmp_path / "store")
    assert ConfigStore(root).resolve().configurations["clean"].path.name == "main_clean.yml"
    assert ConfigStore(root).load_configuration("clean", "main").values["min_trs"] == 50
    assert ensure_store(root) == root


def test_relocation_preserves_compiled_identities(definitions_fixture, tmp_path):
    copied = tmp_path / "relocated"
    shutil.copytree(definitions_fixture, copied)
    first, second = ConfigStore(definitions_fixture), ConfigStore(copied)
    assert first.resolve().fingerprint == second.resolve().fingerprint
    for kind in DERIVATIVE_CLASSES:
        a, b = first.load_configuration(kind, "main"), second.load_configuration(kind, "main")
        assert a.path == b.path
        assert a.fingerprint == b.fingerprint
        assert a.scientific_fingerprint == b.scientific_fingerprint
    assert scientific_model(
        load_task_model("langlocSN/main", copied / "models")
    ) == scientific_model(load_task_model("langlocSN/main", definitions_fixture / "models"))
    (copied / ".git").mkdir()
    (copied / ".git/HEAD").write_text("ref: refs/heads/development\n")
    assert validate_store(copied)["models"] == 1
    assert first.resolve().fingerprint == second.resolve().fingerprint


def test_cli_does_not_select_created_store(tmp_path, capsys):
    selected = site.definitions_root()
    main(["create", str(tmp_path / "store"), "--json"])
    assert json.loads(capsys.readouterr().out)["models"] == 0
    assert site.definitions_root() == selected
    main(["validate", str(tmp_path / "store")])
    assert "Validated" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["validate", str(tmp_path / "missing")])
    assert error.value.code == 1


def test_shared_filesystem_publication(tmp_path, monkeypatch):
    def unsupported(*args):
        ctypes.set_errno(errno.EINVAL)
        return -1

    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: SimpleNamespace(renameat2=unsupported))
    root = create_store(tmp_path / "store")
    assert validate_store(root)["configs"] == 2
    assert not (root / ".nro-incomplete").exists()
    with pytest.raises(FileExistsError):
        _publish(tmp_path / "anything", root)


def test_incomplete_publication_is_rejected(tmp_path, monkeypatch):
    root = create_store(tmp_path / "store")
    (root / ".nro-incomplete").touch()
    monkeypatch.setitem(site.DEFAULTS, "definitions", str(root))
    with pytest.raises(ValueError, match="incomplete"):
        ConfigStore()
    with pytest.raises(ValueError, match="incomplete"):
        validate_store(root)


def test_external_symlink_is_rejected(tmp_path):
    root = create_store(tmp_path / "store")
    external = tmp_path / "external.yml"
    external.write_text("{}\n")
    (root / "configs/clean/external_clean.yml").symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        validate_store(root)


def test_generic_path_without_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "LAB", tmp_path / "missing")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert site.settings()[0]["definitions"] == str(tmp_path / "home/nro/definitions")


def test_setup_creates_selected_store_and_preserves_edits(tmp_path, monkeypatch):
    from nro.bin import setup
    from nro.engine.site_setup import save_settings

    root = tmp_path / "definitions"
    license_file = tmp_path / "license"
    license_file.write_text("test fixture")
    save_settings(site.site_file(), {"definitions": str(root), "license": str(license_file)})
    monkeypatch.setattr("nro.engine.bootstrap.check_workers", lambda path: None)
    for name in ("install_runtime", "install_images", "install_workbench", "install_templates"):
        monkeypatch.setattr(setup, name, lambda **kwargs: None)
    monkeypatch.setattr(setup, "check_installation", lambda **kwargs: [])
    args = [
        "--resources-only",
        "--maintain",
        "--non-interactive",
        "--offline",
        "--without-oslom",
        "--local",
    ]
    setup.main(args)
    assert validate_store(root)["models"] == 0
    path = root / "configs/clean/development_clean.yml"
    path.write_text("standardize: false\n")
    setup.main(args)
    assert path.read_text() == "standardize: false\n"
