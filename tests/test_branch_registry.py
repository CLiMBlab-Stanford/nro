"""Branch scientific state is shared by checkouts, not duplicated worker pools."""

import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nro.configuration.store import ConfigStore
from nro.orchestration import branches
from nro.orchestration.branch_registry import BranchRegistry
from nro.orchestration.branch_repair import _repair_records_locked
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.registry import Registry


def test_checkouts_share_one_central_database(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(branches, "checkout_identity", lambda path: (Path(path), "dev", "a" * 40))
    store = BranchStore(tmp_path / "control")
    snapshot = store.initialize()
    snapshot = store.authorize_checkout("dev", first, revision=snapshot.revision)
    store.authorize_checkout("dev", second, revision=snapshot.revision)
    one = store.registry_for_checkout(first)
    two = BranchStore(store.control).registry_for_checkout(second)
    assert one.database == two.database == store.control / "branches/dev/registry.sqlite3"
    assert one.record.registry_id == two.record.registry_id
    instance = one.record_instance("demo", {"scientific_setting": 10}, expected_revision=None)
    assert two.instances() == (instance,)
    assert not first.exists() and not second.exists()
    assert not (store.control / "registry.sqlite3").exists()


def test_branches_have_independent_science_and_no_pool_tables(tmp_path):
    store = BranchStore(tmp_path)
    snapshot = store.initialize()
    store.register("feature/a", "dev", revision=snapshot.revision)
    one, two = store.registry("dev"), store.registry("feature/a")
    contract = {"module": "clean", "config": {"nuisance": True}}
    a = one.record_instance("same-key", contract, expected_revision=None)
    b = two.record_instance("same-key", contract, expected_revision=None)
    assert a.contract_fingerprint == b.contract_fingerprint
    assert one.record.registry_id != two.record.registry_id
    two.record_instance("same-key", {"module": "clean"}, expected_revision=1)
    assert one.instances() == (a,)
    with sqlite3.connect(two.database) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {
        "identity",
        "instances",
        "workflow_revisions",
        "configuration_lineages",
        "configuration_lineage_dependencies",
        "workflow_bindings",
    }


def test_repair_catalog_drops_purged_records_but_keeps_artifact_dependencies(tmp_path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "preprocessing")

    def spec(key, output, dependencies=()):
        return InstanceSpec.create(
            key=key,
            module="anat",
            project="demo",
            participant="01",
            entities={},
            scope="subject",
            configuration_lineage_id=registered.lineages["preprocessing"],
            config_fingerprint="test",
            directory_label="main",
            runtime_config=runtime,
            command=("python", "-m", "nro.modules.anat"),
            dependencies=dependencies,
            input_paths=(),
            output_root=output.parent,
            output_prefix="sub-01",
            expected_outputs=(output,),
            output_format="test",
            resource_class="small",
        )

    parent_output = tmp_path / "outputs" / "parent.txt"
    downstream_output = tmp_path / "outputs" / "downstream.txt"
    orphan_output = tmp_path / "outputs" / "orphan.txt"
    instances = (
        spec("parent", parent_output),
        spec("downstream", downstream_output, ("parent",)),
        spec("orphan", orphan_output),
    )
    downstream_output.parent.mkdir(parents=True)
    downstream_output.write_text("present")
    ids = registry.register_instances(instances)
    store = BranchStore(registry.paths.control)
    owner = store.initialize().topology.records["dev"].registry_id
    with registry.connection(write=True) as db:
        for item in instances:
            contract = json.dumps({"key": item.key})
            db.execute(
                "INSERT INTO branch_instances VALUES (?,?,?,?)",
                (owner, item.key, ids[item.key], contract),
            )
            db.execute(
                "INSERT INTO compiled_revisions VALUES (?,?,?,?)",
                (owner, item.key, 1, item.key),
            )
        repaired = _repair_records_locked(db, branch="dev", registry_id=owner)
        mapped = {
            row[0]
            for row in db.execute(
                "SELECT logical_key FROM branch_instances WHERE registry_id=?", (owner,)
            )
        }
        revisions = {
            row[0]
            for row in db.execute(
                "SELECT logical_key FROM compiled_revisions WHERE registry_id=?", (owner,)
            )
        }

    assert {row["key"] for row in repaired} == {"parent", "downstream"}
    assert mapped == revisions == {"parent", "downstream"}

    legacy_parent_output = tmp_path / "outputs" / "legacy-parent.txt"
    legacy_downstream_output = tmp_path / "outputs" / "legacy-downstream.txt"
    legacy = (
        spec("legacy-parent", legacy_parent_output),
        spec("legacy-downstream", legacy_downstream_output, ("legacy-parent",)),
    )
    downstream_output.unlink()
    legacy_downstream_output.write_text("present")
    legacy_ids = registry.register_instances(legacy)
    main_owner = store.read().topology.records["main"].registry_id
    with registry.connection(write=True) as db:
        repaired = _repair_records_locked(db, branch="main", registry_id=main_owner)
        mapped = {
            row[0]
            for row in db.execute(
                "SELECT logical_key FROM branch_instances WHERE registry_id=?", (main_owner,)
            )
        }

    assert {row["key"] for row in repaired} == {"legacy-parent", "legacy-downstream"}
    assert mapped == {"legacy-parent", "legacy-downstream"}
    assert set(legacy_ids) == mapped


def test_contract_revision_protects_edits_and_observations(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    one, two = store.registry("dev"), store.registry("dev")
    original = one.record_instance("demo", {"a": 1, "b": 2}, expected_revision=None)
    with pytest.raises(ValueError, match="changed"):
        two.record_instance("demo", {"a": 1, "b": 2}, expected_revision=None)
    one.record_observation("demo", {"complete": True}, expected_revision=original.revision)
    same = two.record_instance("demo", {"b": 2, "a": 1}, expected_revision=1)
    assert same.revision == 1 and same.observation == {"complete": True}
    changed = one.record_instance("demo", {"a": 3, "b": 2}, expected_revision=1)
    assert changed.revision == 2 and changed.observation is None
    with pytest.raises(ValueError, match="changed"):
        two.record_instance("demo", {"a": 4}, expected_revision=1)
    with pytest.raises(ValueError, match="changed"):
        two.record_observation("demo", {"complete": True}, expected_revision=1)
    assert two.instances() == (changed,)


def test_observation_batch_rejects_a_superseded_contract(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    registry = store.registry("dev")
    registry.record_instance("demo", {"setting": 1}, expected_revision=None)
    registry.record_instance("demo", {"setting": 2}, expected_revision=1)
    with pytest.raises(ValueError, match="changed"):
        registry.record_observations({"demo": (1, {"artifact_state": "fresh"})})
    assert registry.instances()[0].observation is None


def test_concurrent_scientific_edits_do_not_lose_updates(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    first, second = store.registry("dev"), store.registry("dev")

    def record(pair):
        registry, setting = pair
        try:
            return registry.record_instance("demo", {"setting": setting}, expected_revision=None)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(record, [(first, 1), (second, 2)]))
    assert sum(result is not None for result in results) == 1
    assert first.instances() == tuple(result for result in results if result is not None)


@pytest.mark.parametrize("replacement", ["branch", "scheduler"])
def test_wrong_database_is_not_adopted(tmp_path, replacement):
    store = BranchStore(tmp_path)
    store.initialize()
    dev = store.registry("dev")
    if replacement == "branch":
        shutil.copyfile(store.registry("main").database, dev.database)
    else:
        dev.database.unlink()
        with sqlite3.connect(dev.database) as db:
            db.execute("CREATE TABLE workers (worker_id TEXT)")
    before = dev.database.read_bytes()
    with pytest.raises(ValueError, match="identity|Not an nro branch"):
        store.initialize()
    assert dev.database.read_bytes() == before


def test_missing_registered_database_is_not_recreated(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    dev = store.registry("dev")
    dev.database.unlink()
    with pytest.raises(FileNotFoundError, match="Missing registered"):
        store.initialize()
    assert not dev.database.exists()


@pytest.mark.parametrize("target", ["root", "database"])
def test_redirected_database_is_rejected(tmp_path, target):
    store = BranchStore(tmp_path / "control")
    store.initialize()
    registry = store.registry("dev")
    path = registry.root if target == "root" else registry.database
    elsewhere = tmp_path / "redirected"
    path.rename(elsewhere)
    path.symlink_to(elsewhere, target_is_directory=target == "root")
    with pytest.raises(ValueError, match="symlink"):
        store.registry("dev")


def test_scientific_schema_change_is_local_to_its_branch(tmp_path):
    store = BranchStore(tmp_path)
    snapshot = store.initialize()
    dev = store.registry("dev")
    with sqlite3.connect(dev.database) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="scientific registry schema"):
        dev.instances()
    assert store.registry("main").instances() == ()
    store.register("new", "dev", revision=snapshot.revision)
    assert store.registry("new").instances() == ()


@pytest.mark.parametrize("name", ["registrations.json", "registration-pending.json", "registries"])
def test_branch_names_do_not_collide_with_catalog_files(tmp_path, name):
    store = BranchStore(tmp_path)
    snapshot = store.initialize()
    store.register(name, "dev", revision=snapshot.revision)
    assert store.registry(name).instances() == ()
    assert store.path.is_file()
    assert not store.pending.exists()


@pytest.mark.parametrize(
    "point", ["before_database", "after_database", "before_catalog", "after_catalog"]
)
def test_interrupted_registration_resumes_same_registry_identity(tmp_path, monkeypatch, point):
    from nro.orchestration import branch_store

    store = BranchStore(tmp_path)
    initial = store.initialize()
    original_initialize = BranchRegistry.initialize
    original_write = branch_store.atomic_write_json

    def initialize(registry):
        if registry.record.name == "feature" and point == "before_database":
            raise OSError("simulated interruption")
        original_initialize(registry)
        if registry.record.name == "feature" and point == "after_database":
            raise OSError("simulated interruption")

    def write(path, value, **kwargs):
        if path == store.path and point == "before_catalog":
            raise OSError("simulated interruption")
        original_write(path, value, **kwargs)
        if path == store.path and point == "after_catalog":
            raise OSError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(BranchRegistry, "initialize", initialize)
        patch.setattr(branch_store, "atomic_write_json", write)
        with pytest.raises(OSError, match="simulated"):
            store.register("feature", "dev", revision=initial.revision)
    pending = json.loads(store.pending.read_text())
    expected_id = pending["records"]["feature"]["registry_id"]
    if point != "after_catalog":
        assert "feature" not in store.read().topology.records
    resumed = BranchStore(tmp_path).initialize()
    assert resumed.topology.records["feature"].registry_id == expected_id
    assert store.registry("feature").instances() == ()
    assert not store.pending.exists()
    for name in ("main", "dev"):
        assert resumed.topology.records[name] == initial.topology.records[name]


def test_interrupted_initialization_does_not_publish_partial_catalog(tmp_path, monkeypatch):
    store = BranchStore(tmp_path)
    original = BranchRegistry.initialize

    def interrupt(registry):
        if registry.record.name == "main":
            raise OSError("simulated interruption")
        original(registry)

    with monkeypatch.context() as patch:
        patch.setattr(BranchRegistry, "initialize", interrupt)
        with pytest.raises(OSError, match="simulated"):
            store.initialize()
    assert not store.path.exists()
    ids = {
        name: row["registry_id"]
        for name, row in json.loads(store.pending.read_text())["records"].items()
    }
    assert {
        name: row.registry_id for name, row in store.initialize().topology.records.items()
    } == ids


def test_unregistered_checkout_cannot_supply_a_local_database(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (root, "dev", "a" * 40))
    store = BranchStore(tmp_path / "control")
    store.initialize()
    local = root / ".nro"
    local.mkdir(parents=True)
    shutil.copyfile(store.registry("dev").database, local / "registry.sqlite3")
    with pytest.raises(ValueError, match="not authorized"):
        store.registry_for_checkout(root)
