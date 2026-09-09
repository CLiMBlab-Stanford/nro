"""Branch administration uses central records without enabling processing."""

import json
from pathlib import Path

import pytest

from nro.bin import branch
from nro.orchestration import branches
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import RegistryPaths


@pytest.fixture
def command(tmp_path, monkeypatch, capsys):
    identities = {}

    def identity(path):
        return Path(path), identities[Path(path)], "a" * 40

    monkeypatch.setattr(branch, "checkout_identity", identity)
    monkeypatch.setattr(branches, "checkout_identity", identity)
    bids = tmp_path / "BIDS"

    def invoke(*args):
        branch.main([*map(str, args), "--bids-root", str(bids), "--json"])
        return json.loads(capsys.readouterr().out)

    return invoke, identities, BranchStore(RegistryPaths.for_project("", bids_root=bids).control)


def test_list_does_not_initialize_store(command):
    invoke, _, store = command
    assert invoke("list") == []
    assert not store.control.exists()


def test_register_attach_and_inspect(command, tmp_path):
    invoke, identities, store = command
    first, second = tmp_path / "first", tmp_path / "second"
    identities.update({first: "dev", second: "dev"})
    initial = invoke("register", "--checkout", first)
    attached = invoke("attach", "--checkout", second)
    assert initial["registry"] == attached["registry"]
    assert initial["registry_id"] == attached["registry_id"]
    assert attached["checkouts"] == [str(first), str(second)]
    assert attached["execution_enabled"] is False
    assert invoke("show", "--checkout", second) == attached
    assert invoke("show", "dev") == attached
    assert not Path(attached["scheduler"]).exists()
    assert set(row["branch"] for row in invoke("list")) == {"dev", "main"}
    assert store.registry("dev").instances() == ()


def test_new_branch_registration_and_retirement(command, tmp_path):
    invoke, identities, store = command
    root = tmp_path / "feature"
    identities[root] = "feature/example"
    record = invoke("register", "--checkout", root)
    assert record["parent"] == "dev"
    assert "feature%2Fexample" in record["registry"]
    assert invoke("retire", "feature/example", "--checkout", root)["retired"]
    with pytest.raises(SystemExit) as error:
        invoke("show", "--checkout", root)
    assert error.value.code == 1
    assert store.registry("feature/example").instances() == ()


def test_duplicate_name_requires_explicit_attach(command, tmp_path):
    invoke, identities, _ = command
    first, second = tmp_path / "first", tmp_path / "second"
    identities.update({first: "experiment", second: "experiment"})
    registered = invoke("register", "--checkout", first)
    with pytest.raises(SystemExit) as error:
        invoke("register", "--checkout", second)
    assert error.value.code == 1
    attached = invoke("attach", "--checkout", second)
    assert attached["registry_id"] == registered["registry_id"]


def test_switched_checkout_is_not_silently_rebound(command, tmp_path):
    invoke, identities, _ = command
    root = tmp_path / "source"
    identities[root] = "dev"
    invoke("register", "--checkout", root)
    identities[root] = "main"
    with pytest.raises(SystemExit) as error:
        invoke("attach", "--checkout", root)
    assert error.value.code == 1


def test_reparent_keeps_registry_location_and_identity(command, tmp_path):
    invoke, identities, _ = command
    first, second = tmp_path / "first", tmp_path / "second"
    identities.update({first: "first", second: "second"})
    invoke("register", "--checkout", first)
    before = invoke("register", "--checkout", second, "--parent", "first")
    after = invoke("reparent", "second", "--parent", "dev", "--checkout", second)
    assert before["registry"] == after["registry"]
    assert before["registry_id"] == after["registry_id"]
    assert before["parent"] == "first" and after["parent"] == "dev"


def test_branch_change_requires_its_own_checkout(command, tmp_path):
    invoke, identities, _ = command
    first, second = tmp_path / "first", tmp_path / "second"
    identities.update({first: "first", second: "second"})
    invoke("register", "--checkout", first)
    invoke("register", "--checkout", second)
    with pytest.raises(SystemExit) as error:
        invoke("retire", "first", "--checkout", second)
    assert error.value.code == 1


def test_select_and_restore_branch_definitions(command, tmp_path, monkeypatch):
    from nro.configuration import site
    from nro.configuration.definitions import create_store

    invoke, identities, _ = command
    root = tmp_path / "source"
    identities[root] = "dev"
    shared = create_store(tmp_path / "shared-definitions")
    private = create_store(tmp_path / "private-definitions")
    values = {**site.settings()[0], "definitions": str(shared)}
    monkeypatch.setattr(site, "settings", lambda: (values, {}))
    invoke("register", "--checkout", root)
    selected = invoke("definitions", "--checkout", root, "--definitions", private)
    assert selected["definitions"] == str(private)
    assert selected["definitions_writable"]
    assert invoke("show", "dev")["definitions"] == str(private)
    restored = invoke("definitions", "--checkout", root, "--shared")
    assert restored["definitions"] == str(shared)
    assert not restored["definitions_writable"]


@pytest.mark.parametrize(
    "args",
    [
        ["register", "name"],
        ["list", "name"],
        ["retire"],
        ["reparent", "name"],
        ["show", "--parent", "dev"],
        ["definitions"],
        ["show", "--shared"],
    ],
)
def test_invalid_command_does_not_create_state(command, args):
    invoke, _, store = command
    with pytest.raises(SystemExit) as error:
        invoke(*args)
    assert error.value.code == 2
    assert not store.control.exists()
