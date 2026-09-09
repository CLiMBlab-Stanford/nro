"""Development definition edits cannot change shared or neighboring stores."""

import pytest

from nro.configuration import branch_definitions, site
from nro.configuration.definitions import create_store
from nro.engine.definition_editor import save_definition
from nro.orchestration import branches
from nro.orchestration.branch_store import BranchStore


@pytest.fixture
def stores(tmp_path, monkeypatch):
    shared = create_store(tmp_path / "shared-definitions")
    private = create_store(tmp_path / "private-definitions")
    store = BranchStore(tmp_path / "control")
    checkout = tmp_path / "source"

    def identity(_):
        return checkout, "dev", "a" * 40

    monkeypatch.setattr(branches, "checkout_identity", identity)
    monkeypatch.setattr(branch_definitions, "checkout_identity", identity)
    initial = store.initialize()
    registered = store.authorize_checkout("dev", checkout, revision=initial.revision)
    record = dict(
        mode="branch",
        ready=True,
        branch="dev",
        checkout=str(checkout),
        registry_id=registered.topology.records["dev"].registry_id,
    )
    values = {**site.settings()[0], "registry": str(store.control), "definitions": str(shared)}
    monkeypatch.setattr(site, "installation_record", lambda: record)
    monkeypatch.setattr(site, "settings", lambda: (values, {}))
    return store, checkout, shared, private, record


def test_default_read_only_and_explicit_private_selection(stores):
    store, checkout, shared, private, record = stores
    assert site.definitions_root() == shared
    with pytest.raises(ValueError, match="cannot edit shared"):
        site.require_definition_write()
    branch_definitions.select_definitions(store, checkout, shared, private)
    assert site.definitions_root() == private
    site.require_definition_write(private / "configs" / "new.yml")
    with pytest.raises(ValueError, match="outside"):
        site.require_definition_write(shared / "configs" / "new.yml")
    # A second installation of this branch reads the same centrally selected path.
    assert branch_definitions.selected_definitions(store.control, dict(record), shared) == private
    branch_definitions.select_definitions(store, checkout, shared, None)
    assert site.definitions_root() == shared


def test_edits_are_published_only_inside_the_private_store(stores):
    store, checkout, shared, private, _ = stores
    branch_definitions.select_definitions(store, checkout, shared, private)
    target = private / "configs" / "example.yml"
    save_definition(target, "example: true\n", expected=None)
    assert target.read_text() == "example: true\n"
    target = shared / "configs" / "example.yml"
    with pytest.raises(ValueError, match="outside"):
        save_definition(target, "example: true\n", expected=None)
    assert not target.exists()
    (private / "escape").symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        save_definition(private / "escape" / "example.yml", "", expected=None)


def test_branch_can_create_a_new_store_without_selecting_it(stores, tmp_path):
    _, _, shared, _, _ = stores
    destination = tmp_path / "another-store"
    create_store(destination)
    assert (destination / "configs").is_dir()
    assert site.definitions_root() == shared
    with pytest.raises(ValueError, match="separate"):
        create_store(shared / "nested")
    assert not (shared / "nested").exists()


@pytest.mark.parametrize(
    "target", ["shared", "shared-child", "shared-parent", "control", "symlink"]
)
def test_selection_rejects_overlapping_or_redirected_roots(stores, tmp_path, target):
    store, checkout, shared, private, _ = stores
    link = tmp_path / "linked"
    link.symlink_to(private, target_is_directory=True)
    path = {
        "shared": shared,
        "shared-child": shared / "configs",
        "shared-parent": shared.parent,
        "control": store.control,
        "symlink": link,
    }[target]
    with pytest.raises(ValueError):
        branch_definitions.select_definitions(store, checkout, shared, path)
    assert site.definitions_root() == shared


def test_other_branches_cannot_select_the_same_store(stores, tmp_path, monkeypatch):
    store, checkout, shared, private, _ = stores
    branch_definitions.select_definitions(store, checkout, shared, private)
    feature = tmp_path / "feature"
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (feature, "feature", "a" * 40))
    store.register("feature", "dev", checkout=feature, revision=store.read().revision)
    with pytest.raises(ValueError, match="overlap"):
        branch_definitions.select_definitions(store, feature, shared, private)


def test_changed_git_branch_or_registration_cannot_use_the_selection(stores, monkeypatch):
    store, checkout, shared, private, record = stores
    branch_definitions.select_definitions(store, checkout, shared, private)
    record["registry_id"] = "b" * 32
    with pytest.raises(ValueError, match="registration"):
        site.definitions_root()


def test_private_selection_does_not_mutate_scientific_contracts(stores):
    store, checkout, shared, private, _ = stores
    registry = store.registry("dev")
    instance = registry.record_instance("test", {"configuration": "same"}, expected_revision=None)
    branch_definitions.select_definitions(store, checkout, shared, private)
    assert registry.instances() == (instance,)


def test_selection_cannot_change_during_publication(stores):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    store, checkout, shared, private, _ = stores
    branch_definitions.select_definitions(store, checkout, shared, private)
    started = Event()

    def reset():
        started.set()
        return branch_definitions.select_definitions(store, checkout, shared, None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with site.definition_write(private / "configs" / "test.yml"):
            future = executor.submit(reset)
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
        assert future.result(timeout=5) == shared


def test_installed_dispatcher_allows_connected_branch_commands(stores, monkeypatch, capsys):
    from nro import cli

    _, _, _, private, _ = stores
    cli.main(["definitions", "validate", str(private), "--json"])
    assert str(private) in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(cli, "_command_main", lambda command: lambda *a, **k: calls.append(command))
    for command in ("create", "edit", "delete", "models", "run", "status", "stop"):
        cli.main([command])
    assert calls == ["create", "edit", "delete", "models", "run", "status", "stop"]
    cli.main(["bidsify"])
    assert calls[-1] == "bidsify"
