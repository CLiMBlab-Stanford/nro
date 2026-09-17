"""Branch scientific state is shared by checkouts, not duplicated worker pools."""

import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from nro.configuration.store import CONFIGURATION_CLASSES, ConfigStore
from nro.orchestration import branches
from nro.orchestration.branch_registry import SCHEMA_VERSION, BranchRegistry
from nro.orchestration.branch_repair import (
    _recover_public_work_items,
    _register_current_workflows,
    _repair_records_locked,
)
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.ownership import (
    OWNERSHIP_VERSION,
    lineage_record_path,
    work_item_record_path,
)
from nro.orchestration.planning_context import work_item_key
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
    work_item = one.record_work_item("demo", {"scientific_setting": 10}, expected_revision=None)
    assert two.work_items() == (work_item,)
    assert not first.exists() and not second.exists()
    assert not (store.control / "registry.sqlite3").exists()


def test_branches_have_independent_science_and_no_pool_tables(tmp_path):
    store = BranchStore(tmp_path)
    snapshot = store.initialize()
    store.register("feature/a", "dev", revision=snapshot.revision)
    one, two = store.registry("dev"), store.registry("feature/a")
    contract = {"module": "clean", "config": {"nuisance": True}}
    a = one.record_work_item("same-key", contract, expected_revision=None)
    b = two.record_work_item("same-key", contract, expected_revision=None)
    assert a.contract_fingerprint == b.contract_fingerprint
    assert one.record.registry_id != two.record.registry_id
    two.record_work_item("same-key", {"module": "clean"}, expected_revision=1)
    assert one.work_items() == (a,)
    with sqlite3.connect(two.database) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {
        "identity",
        "work_items",
        "workflow_revisions",
        "module_lineages",
        "module_lineage_dependencies",
        "workflow_bindings",
    }


def test_repair_catalog_drops_purged_records_but_keeps_artifact_dependencies(tmp_path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")

    def spec(key, output, dependencies=()):
        return WorkItemSpec.create(
            key=key,
            module="anat",
            project="demo",
            participant="01",
            entities={},
            scope="subject",
            module_lineage_id=registered.lineages["anat"],
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
    work_items = (
        spec("parent", parent_output),
        spec("downstream", downstream_output, ("parent",)),
        spec("orphan", orphan_output),
    )
    downstream_output.parent.mkdir(parents=True)
    downstream_output.write_text("present")
    ids = registry.register_work_items(work_items)
    store = BranchStore(registry.paths.control)
    owner = store.initialize().topology.records["dev"].registry_id
    with registry.connection(write=True) as db:
        for item in work_items:
            contract = json.dumps({"key": item.key})
            db.execute(
                "INSERT INTO branch_work_items VALUES (?,?,?,?)",
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
                "SELECT logical_key FROM branch_work_items WHERE registry_id=?", (owner,)
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
    legacy_ids = registry.register_work_items(legacy)
    main_owner = store.read().topology.records["main"].registry_id
    with registry.connection(write=True) as db:
        repaired = _repair_records_locked(db, branch="main", registry_id=main_owner)
        mapped = {
            row[0]
            for row in db.execute(
                "SELECT logical_key FROM branch_work_items WHERE registry_id=?", (main_owner,)
            )
        }

    assert {row["key"] for row in repaired} == {"legacy-parent", "legacy-downstream"}
    assert mapped == {"legacy-parent", "legacy-downstream"}
    assert set(legacy_ids) == mapped


def test_branch_repair_recovers_current_public_ownership(tmp_path, monkeypatch):
    from nro.configuration.site import settings

    site_values = settings()[0]
    bids = tmp_path / "BIDS"
    development = tmp_path / "DEV"
    registry = Registry.for_project("", bids_root=bids)
    store = BranchStore(registry.paths.control)
    topology = store.initialize().topology
    owner = topology.records["dev"].registry_id
    scientific = store.registry("dev")
    workflow = ConfigStore().resolve("main")
    registered = scientific.register_workflow(workflow)
    output = development / "dev/BIDS/demo/derivatives/nro/anat/main/sub-01/sub-01_result.txt"
    output.parent.mkdir(parents=True)
    output.write_text("complete")
    runtime = scientific.runtime_config_path(registered, "anat")
    logical_key = work_item_key(
        "demo",
        "anat",
        registered.lineage_fingerprints["anat"],
        "01",
        {},
    )
    spec = WorkItemSpec.create(
        key=logical_key,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directories["anat"],
        runtime_config=runtime,
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(),
        output_root=output.parent,
        output_prefix="sub-01",
        expected_outputs=(output,),
        output_format="test",
        resource_class="small",
    )
    with scientific.connection() as db:
        lineage = dict(
            db.execute(
                "SELECT * FROM module_lineages WHERE id=?",
                (registered.lineages["anat"],),
            ).fetchone()
        )
    project_root = development / "dev/BIDS/demo"
    marker = {
        "record_version": OWNERSHIP_VERSION,
        "owner": "nro",
        "configuration_class": "anat",
        "directory_label": registered.directories["anat"],
        "configuration": {
            "id": lineage["config_id"],
            "fingerprint": lineage["config_fingerprint"],
            "resolved": yaml.safe_load(lineage["resolved_yaml"]),
        },
        "lineage_fingerprint": lineage["lineage_fingerprint"],
        "upstream": [],
        "updated_at": "now",
    }
    receipt = {
        "record_version": OWNERSHIP_VERSION,
        "owner": "nro",
        "work_item_key": spec.key,
        "module": spec.module,
        "project": spec.project,
        "participant": spec.participant,
        "entities": dict(spec.entities),
        "scope": spec.scope,
        "lineage_fingerprint": lineage["lineage_fingerprint"],
        "directory_label": spec.directory_label,
        "artifact_contract": spec.contract.as_dict(spec.identity),
        "scientific_contract": {
            "project": spec.project,
            "participant": spec.participant,
            **spec.contract.as_dict(spec.identity),
        },
        "execution": {
            "command": list(spec.command),
            "runtime_configuration": yaml.safe_load(runtime.read_text()),
        },
        "resources": {
            "resource_class": "small",
            "memory_gb": 32,
            "max_memory_gb": 256,
        },
        "recorded_at": "now",
    }
    marker_path = lineage_record_path(project_root, "anat", spec.directory_label)
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(json.dumps(marker))
    receipt_path = work_item_record_path(
        project_root, "anat", spec.directory_label, "anat", spec.key
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        "nro.configuration.site.settings",
        lambda: (
            {
                **site_values,
                "bids": str(bids),
                "work": str(tmp_path / "WORK"),
                "development": str(development),
            },
            {},
        ),
    )
    assessments = []
    monkeypatch.setattr(
        "nro.orchestration.branch_repair.assess_registry",
        lambda *_args, **kwargs: assessments.append(kwargs),
    )
    monkeypatch.setattr(
        WorkItemSpec,
        "work_item_contract",
        property(lambda _spec: pytest.fail("Recovery recompiled a recorded artifact contract")),
    )

    assert _recover_public_work_items(registry, branch="dev", registry_id=owner) == []
    assert assessments == [{"projects": ("demo",), "compiled": True, "recover_public": True}]
    with registry.connection() as db:
        mapping = db.execute(
            "SELECT logical_key,work_item_id FROM branch_work_items WHERE registry_id=?",
            (owner,),
        ).fetchone()
        row = db.execute(
            "SELECT * FROM work_items WHERE id=?", (mapping["work_item_id"],)
        ).fetchone()
        recovered = _repair_records_locked(db, branch="dev", registry_id=owner)
    assert mapping["logical_key"] == spec.key
    assert row["work_item_key"] == f"{owner}:{spec.key}"
    assert [item["key"] for item in recovered] == [spec.key]

    with sqlite3.connect(scientific.database) as db:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION - 1}")
    from nro.orchestration.scheduler_repair import repair_scientific_schemas

    report = repair_scientific_schemas(registry)
    assert report == [
        {
            "branch": "dev",
            "stored_schema": SCHEMA_VERSION - 1,
            "schema": SCHEMA_VERSION,
            "backup": str(scientific.root / "registry-before-repair.sqlite3"),
            "work_items": 1,
            "unavailable": [],
        }
    ]
    assert [item.key for item in scientific.work_items()] == [spec.key]

    from nro.orchestration.branch_admission import _workflow

    payloads = _register_current_workflows(scientific, store=ConfigStore(workflow.path.parents[1]))
    with registry.connection(write=True) as db:
        for payload in payloads:
            _workflow(db, payload, owner)
    row = next(
        item for item in registry.work_item_rows() if item["work_item_key"].endswith(spec.key)
    )
    assert row["recomputable"] == 1


def test_branch_ownership_rejects_changed_configuration_identity(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    record = {
        "configuration_class": "dynconn",
        "directory_label": "evaluation",
        "configuration": {
            "id": "evaluation",
            "fingerprint": "configuration",
            "resolved": {"low_rank": True},
        },
        "lineage_fingerprint": "scientific-lineage",
        "upstream": [],
    }
    registry.register_owned_lineages([record], branch_registry_id="branch-owner")

    changed = {
        **record,
        "configuration": {**record["configuration"], "id": "other"},
    }
    with pytest.raises(ValueError, match="conflicts with registered lineage"):
        registry.register_owned_lineages([changed], branch_registry_id="branch-owner")

    moved = {**record, "directory_label": "evaluation-2"}
    with pytest.raises(ValueError, match="conflicts with registered lineage"):
        registry.register_owned_lineages([moved], branch_registry_id="branch-owner")


def test_rebuild_seeds_directory_assignments_from_owned_lineages(tmp_path):
    store = BranchStore(tmp_path / "control")
    store.initialize()
    scientific = store.registry("dev")
    workflow = ConfigStore().resolve("main")
    registered = scientific.register_workflow(workflow)
    owned = []
    with scientific.connection() as db:
        for configuration_class in CONFIGURATION_CLASSES:
            lineage_id = registered.lineages[configuration_class]
            row = db.execute("SELECT * FROM module_lineages WHERE id=?", (lineage_id,)).fetchone()
            upstream = [
                {
                    "lineage_fingerprint": parent["lineage_fingerprint"],
                    "role": parent["role"],
                }
                for parent in db.execute(
                    """SELECT parent.lineage_fingerprint,dependency.role
                       FROM module_lineage_dependencies dependency
                       JOIN module_lineages parent
                         ON parent.id=dependency.upstream_module_lineage_id
                       WHERE dependency.module_lineage_id=?""",
                    (lineage_id,),
                )
            ]
            owned.append(
                {
                    "configuration_class": configuration_class,
                    "directory_label": (
                        "preserved-dynconn"
                        if configuration_class == "dynconn"
                        else row["directory_label"]
                    ),
                    "configuration": {
                        "id": row["config_id"],
                        "fingerprint": row["config_fingerprint"],
                        "resolved": yaml.safe_load(row["resolved_yaml"]),
                    },
                    "lineage_fingerprint": row["lineage_fingerprint"],
                    "upstream": upstream,
                    "updated_at": row["created_at"],
                }
            )

    scientific.rebuild([], [], owned_lineages=owned)
    rebuilt = scientific.register_workflow(workflow)

    assert rebuilt.lineage_fingerprints == registered.lineage_fingerprints
    assert rebuilt.directories["dynconn"] == "preserved-dynconn"


def test_contract_revision_protects_edits_and_observations(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    one, two = store.registry("dev"), store.registry("dev")
    original = one.record_work_item("demo", {"a": 1, "b": 2}, expected_revision=None)
    with pytest.raises(ValueError, match="changed"):
        two.record_work_item("demo", {"a": 1, "b": 2}, expected_revision=None)
    one.record_observation("demo", {"complete": True}, expected_revision=original.revision)
    same = two.record_work_item("demo", {"b": 2, "a": 1}, expected_revision=1)
    assert same.revision == 1 and same.observation == {"complete": True}
    changed = one.record_work_item("demo", {"a": 3, "b": 2}, expected_revision=1)
    assert changed.revision == 2 and changed.observation is None
    with pytest.raises(ValueError, match="changed"):
        two.record_work_item("demo", {"a": 4}, expected_revision=1)
    with pytest.raises(ValueError, match="changed"):
        two.record_observation("demo", {"complete": True}, expected_revision=1)
    assert two.work_items() == (changed,)


def test_observation_batch_rejects_a_superseded_contract(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    registry = store.registry("dev")
    registry.record_work_item("demo", {"setting": 1}, expected_revision=None)
    registry.record_work_item("demo", {"setting": 2}, expected_revision=1)
    with pytest.raises(ValueError, match="changed"):
        registry.record_observations({"demo": (1, {"artifact_state": "fresh"})})
    assert registry.work_items()[0].observation is None


def test_concurrent_scientific_edits_do_not_lose_updates(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    first, second = store.registry("dev"), store.registry("dev")

    def record(pair):
        registry, setting = pair
        try:
            return registry.record_work_item("demo", {"setting": setting}, expected_revision=None)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(record, [(first, 1), (second, 2)]))
    assert sum(result is not None for result in results) == 1
    assert first.work_items() == tuple(result for result in results if result is not None)


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
        dev.work_items()
    assert store.registry("main").work_items() == ()
    store.register("new", "dev", revision=snapshot.revision)
    assert store.registry("new").work_items() == ()


def test_scientific_schema_rebuild_does_not_migrate_obsolete_records(tmp_path):
    store = BranchStore(tmp_path)
    store.initialize()
    registry = store.registry("main")
    registry.record_work_item("example", {"module": "anat"}, expected_revision=None)
    registry.record_observation("example", {"artifact_state": "fresh"}, expected_revision=1)
    snapshot = registry.paths.workflows / "legacy/1_workflow.yml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(
        "workflow_id: legacy\n"
        "revision: 1\n"
        "definition_fingerprint: legacy-fingerprint\n"
        "selections: {preprocessing: main}\n"
        "configurations:\n"
        "  preprocessing:\n"
        "    resolved: {anat: {}, func: {}}\n"
    )
    with sqlite3.connect(registry.database) as db:
        db.execute(
            "ALTER TABLE module_lineages RENAME COLUMN configuration_class TO derivative_class"
        )
        db.execute(
            "ALTER TABLE workflow_bindings RENAME COLUMN configuration_class TO derivative_class"
        )
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION - 1}")

    backup = registry.rebuild_schema()

    assert backup == registry.root / "registry-before-repair.sqlite3"
    assert backup.is_file()
    assert registry.work_items() == ()
    assert registry.stored_schema_version() == SCHEMA_VERSION
    with registry.connection() as db:
        assert (
            db.execute(
                "SELECT definition_fingerprint FROM workflow_revisions WHERE workflow_id='legacy'"
            ).fetchone()[0]
            == "legacy-fingerprint"
        )
    with sqlite3.connect(backup) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION - 1


@pytest.mark.parametrize("name", ["registrations.json", "registration-pending.json", "registries"])
def test_branch_names_do_not_collide_with_catalog_files(tmp_path, name):
    store = BranchStore(tmp_path)
    snapshot = store.initialize()
    store.register(name, "dev", revision=snapshot.revision)
    assert store.registry(name).work_items() == ()
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
    assert store.registry("feature").work_items() == ()
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
