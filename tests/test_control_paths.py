"""All site callers share one layout and refuse an implicit live cutover."""

from pathlib import Path

import pytest

from nro.configuration import site
from nro.engine import bootstrap
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import Registry


def test_shared_and_branch_state_locations(tmp_path):
    paths = ControlPaths(tmp_path)
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS", registry_path=tmp_path)
    assert (
        registry.paths.database == paths.database == tmp_path / "shared/scheduler/registry.sqlite3"
    )
    assert registry.paths.events == paths.branch("main") / "events"
    assert registry.paths.workers == paths.scheduler / "workers"
    assert paths.branch("feature/example") == tmp_path / "branches/feature%2Fexample"
    store = BranchStore(tmp_path)
    assert store.path == paths.catalog
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("caller", ["registry", "branches", "install"])
def test_previous_layout_is_not_silently_bootstrapped(tmp_path, caller):
    old = tmp_path / "registry.sqlite3"
    old.write_bytes(b"existing state")
    config = tmp_path / "site.toml"
    config.write_text(f'registry = "{tmp_path}"\n')
    with pytest.raises(ValueError, match="previous layout"):
        if caller == "registry":
            Registry.for_project("demo", registry_path=tmp_path)
        elif caller == "branches":
            BranchStore(tmp_path)
        else:
            bootstrap.check_workers(config)
    assert old.read_bytes() == b"existing state"
    assert not (tmp_path / "shared").exists()


def test_generic_defaults_are_checkout_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "LAB", tmp_path / "absent")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(site, "installation_record", lambda: {})
    for key in site.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "site.toml"
    a = site.settings(path=config)[0]
    monkeypatch.setattr(site, "CHECKOUT", tmp_path / "another-checkout")
    b = site.settings(path=config)[0]
    assert a == b
    assert a["registry"] == str(tmp_path / "home/nro/.nro")
    assert a["bids"] == str(tmp_path / "home/nro/bids")
    config.write_text('registry = "/explicit/shared/control"\n')
    assert site.settings(path=config)[0]["registry"] == "/explicit/shared/control"


def test_branch_cannot_use_production_registry_or_edit_site(tmp_path, monkeypatch):
    from nro.engine import site_setup
    from nro.engine.definition_editor import save_definition

    monkeypatch.setattr(site, "installation_record", lambda: {"mode": "branch"})
    monkeypatch.setattr(site_setup, "installation_record", site.installation_record)
    with pytest.raises(ValueError, match="not permitted"):
        Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    with pytest.raises(ValueError, match="cannot edit"):
        site_setup.edit_settings(["work=/anything"])
    with pytest.raises(ValueError, match="cannot edit"):
        save_definition(tmp_path / "definition.yml", "test", expected=None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "relative",
    ["shared/cache/implementations", "shared/scheduler/registry.sqlite3", "branches/main"],
)
def test_private_paths_cannot_redirect_outside_control(tmp_path, relative):
    control = tmp_path / "control"
    target = control / relative
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    target.symlink_to(outside, target_is_directory=True)
    paths = ControlPaths(control)
    with pytest.raises(ValueError, match="symlink"):
        paths.branch("main") if relative == "branches/main" else paths.require_current_layout()
