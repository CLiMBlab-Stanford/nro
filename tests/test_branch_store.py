"""Durable branch edits use revision checks without touching live registrations."""

import json
from pathlib import Path

import pytest

from nro.orchestration.branch_store import BranchStore


def test_control_permissions_do_not_change_existing_ancestors(tmp_path):
    from nro.orchestration.registry import ensure_shared_directory

    tmp_path.chmod(0o700)
    directory = tmp_path / "custom-control" / "locks"
    ensure_shared_directory(directory)
    assert tmp_path.stat().st_mode & 0o7777 == 0o700
    assert directory.parent.stat().st_mode & 0o7777 == 0o2775
    assert directory.stat().st_mode & 0o7777 == 0o2775
    with pytest.raises(ValueError, match="filesystem root"):
        ensure_shared_directory("/")


def test_store_initialization_and_read_are_separate(tmp_path):
    store = BranchStore(tmp_path / "control")
    assert not store.control.exists()
    with pytest.raises(FileNotFoundError):
        store.read()
    before = store.initialize()
    assert tuple(before.topology.records) == ("dev", "main")
    after = store.register("feature/a", "dev", revision=before.revision)
    assert store.initialize().revision == after.revision
    assert BranchStore(store.control).read().revision == after.revision
    assert not (store.control / "registry.sqlite3").exists()


def test_stale_edit_and_duplicate_name_are_rejected(tmp_path):
    store = BranchStore(tmp_path)
    initial = store.initialize()
    current = store.register("first", "dev", revision=initial.revision)
    with pytest.raises(ValueError, match="changed"):
        store.register("second", "dev", revision=initial.revision)
    with pytest.raises(ValueError, match="already reserved"):
        store.register("first", "dev", revision=current.revision)
    assert store.read().revision == current.revision
    assert "second" not in store.read().topology.records


def test_concurrent_registration_does_not_lose_an_update(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = BranchStore(tmp_path)
    initial = store.initialize()

    def register(name):
        try:
            return store.register(name, "dev", revision=initial.revision)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(register, ["first", "second"]))
    assert sum(result is not None for result in results) == 1
    assert len(store.read().topology.records) == 3


@pytest.mark.parametrize("fail", [False, True])
def test_registry_repair_preserves_branch_authority_and_execution_sources(
    tmp_path, monkeypatch, fail
):
    from nro.orchestration.execution_pins import capture_execution
    from nro.orchestration.registry import Registry

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    store = BranchStore(registry.paths.control)
    initial = store.initialize()
    before = store.register("experiment", "dev", revision=initial.revision)
    science = store.registry("experiment")
    instance = science.record_instance("demo", {"module": "anat"}, expected_revision=None)
    source, site = capture_execution(registry.paths.control, registry.paths.bids_root)
    site_text = site.read_text()
    if fail:

        def reject():
            raise RuntimeError("Test repair failure")

        monkeypatch.setattr(registry, "_initialize_locked", reject)
        with pytest.raises(RuntimeError, match="Test repair failure"):
            registry.reinitialize()
    else:
        registry.reinitialize()
    assert store.read().revision == before.revision
    assert store.registry("experiment").instances() == (instance,)
    source.verify()
    assert site.read_text() == site_text


def test_reparent_and_retirement_preserve_old_snapshot(tmp_path, monkeypatch):
    from nro.orchestration import branches

    store = BranchStore(tmp_path)
    snapshot = store.initialize()
    first, second = tmp_path / "first", tmp_path / "second"
    identities = {first: "first", second: "second"}
    monkeypatch.setattr(
        branches, "checkout_identity", lambda path: (Path(path), identities[Path(path)], "a" * 40)
    )
    snapshot = store.register("first", "dev", revision=snapshot.revision, checkout=first)
    before = store.register("second", "first", revision=snapshot.revision, checkout=second)
    changed = store.reparent("second", "dev", revision=before.revision, checkout=second)
    retired = store.retire("first", revision=changed.revision, checkout=first)
    assert before.topology.ancestors("second") == ("second", "first", "dev", "main")
    assert retired.topology.ancestors("second") == ("second", "dev", "main")
    assert retired.topology.records["first"].retired
    with pytest.raises(ValueError, match="already reserved"):
        store.register("first", "dev", revision=retired.revision)


def test_checkout_authority_is_persisted_and_branch_switch_rejected(tmp_path, monkeypatch):
    from nro.orchestration import branches

    root = tmp_path / "source"
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (root, "dev", "a" * 40))
    store = BranchStore(tmp_path / "control")
    initial = store.initialize()
    store.authorize_checkout("dev", root, revision=initial.revision)
    assert store.read().topology.require_checkout(root) == "dev"
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (root, "main", "a" * 40))
    with pytest.raises(ValueError, match="not authorized"):
        store.read().topology.require_checkout(root)


def test_checkout_from_another_repository_cannot_attach(tmp_path, monkeypatch):
    from nro.orchestration import branches

    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(
        branches,
        "checkout_identity",
        lambda path: (Path(path), "dev", "a" * 40),
    )
    repositories = {first: "origin:one", second: "origin:two"}
    monkeypatch.setattr(branches, "repository_identity", lambda path: repositories[Path(path)])
    store = BranchStore(tmp_path / "control")
    snapshot = store.initialize()
    snapshot = store.authorize_checkout("dev", first, revision=snapshot.revision)
    with pytest.raises(ValueError, match="different Git repository"):
        store.authorize_checkout("dev", second, revision=snapshot.revision)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["dev"].update(retired="false"),
        lambda data: data["dev"].update(checkouts=["relative"]),
        lambda data: data["dev"].update(extra=True),
        lambda data: data["dev"].update(parent="dev"),
    ],
)
def test_corrupt_store_is_not_repaired_silently(tmp_path, change):
    store = BranchStore(tmp_path)
    store.initialize()
    data = json.loads(store.path.read_text())
    change(data)
    store.path.write_text(json.dumps(data))
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.initialize()
    assert store.path.read_bytes() == before
