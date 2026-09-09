"""A private layout change preserves scientific records and has recoverable boundaries."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from nro.bin import cutover as command
from nro.orchestration import control_cutover as cutover
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import APPLICATION_ID, SCHEMA_SQL, SCHEMA_VERSION, Registry

pytestmark = pytest.mark.integration


@pytest.fixture
def flat_store(tmp_path):
    root = tmp_path / ".nro"
    root.mkdir()
    runtime = root / "snapshots/main_preprocess.yml"
    runtime.parent.mkdir()
    runtime.write_text("setting: scientific\n")
    output = tmp_path / "BIDS/demo/derivatives/preprocessing/main/sub-01/result.nii"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"original scientific data")
    manifest_path = root / "manifests/instance.json"
    manifest_path.parent.mkdir()
    spec = InstanceSpec.create(
        key="anat:example",
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=1,
        config_fingerprint="config",
        directory_label="main",
        resource_class="large",
        runtime_config=runtime,
        command=["python", "-m", "nro.modules.anat", "--runtime-config", str(runtime)],
        input_paths=(),
        output_root=output.parent,
        output_prefix=None,
        expected_outputs=[output],
        dependencies=(),
        output_format="test",
    )
    manifest = dict(
        artifact_contract=spec.instance_contract,
        artifact_fingerprint=spec.contract_fingerprint,
        generation=7,
        software={"command": list(spec.command)},
        runtime_config={
            "path": str(runtime),
            "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        },
        upstream=[{"generation": 6, "manifest": str(root / "manifests/parent.json")}],
    )
    manifest_path.write_text(json.dumps(manifest))
    with sqlite3.connect(root / "registry.sqlite3") as db:
        db.executescript(SCHEMA_SQL)
        db.execute(f"PRAGMA application_id={APPLICATION_ID}")
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [("schema_version", str(SCHEMA_VERSION)), ("registry_uuid", "original-id")],
        )
        db.execute(
            "INSERT INTO configuration_lineages VALUES (1,'preprocessing','main','config','lineage','{}','main','now')"
        )
        row = dict(
            spec.as_record(),
            artifact_state="fresh",
            current_generation=7,
            manifest_path=str(manifest_path),
            created_at="original-date",
            updated_at="original-date",
        )
        db.execute(
            f"INSERT INTO instances ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
            tuple(row.values()),
        )
    return root, spec, manifest, output


def test_cutover_preserves_identity_contracts_and_public_files(flat_store):
    root, spec, manifest, output = flat_store
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (output, root / "snapshots/main_preprocess.yml")
    }
    old_db = (root / "registry.sqlite3").read_bytes()
    plan = cutover.preview(root)
    assert plan.files == 3
    assert not cutover.journal_path(root).exists()
    backup = cutover.execute(root, expected_fingerprint=plan.source_fingerprint)
    assert (backup / "registry.sqlite3").read_bytes() == old_db
    assert output.read_bytes() == before[str(output)][0]
    assert output.stat().st_mtime_ns == before[str(output)][1]
    paths = ControlPaths(root)
    paths.require_current_layout()
    registry = Registry.for_project(
        "demo", bids_root=output.parents[6] / "BIDS", registry_path=root
    )
    with registry.read_connection() as db:
        row = dict(db.execute("SELECT * FROM instances").fetchone())
        assert (
            db.execute("SELECT value FROM metadata WHERE key='registry_uuid'").fetchone()[0]
            == "original-id"
        )
    assert row["artifact_state"] == "fresh" and row["current_generation"] == 7
    assert row["artifact_fingerprint"] == spec.contract_fingerprint
    assert json.loads(row["artifact_contract_json"]) == spec.instance_contract
    assert row["created_at"] == row["updated_at"] == "original-date"
    runtime = paths.branch("main") / "snapshots/main_preprocess.yml"
    assert row["runtime_config_path"] == str(runtime)
    assert json.loads(row["command_json"])[-1] == str(runtime)
    moved = json.loads(Path(row["manifest_path"]).read_text())
    assert moved["artifact_contract"] == manifest["artifact_contract"]
    assert moved["software"] == manifest["software"]
    assert moved["runtime_config"]["sha256"] == hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert runtime.stat().st_mtime_ns == before[str(root / "snapshots/main_preprocess.yml")][1]
    assert moved["upstream"][0]["manifest"] == str(paths.branch("main") / "manifests/parent.json")
    assert not cutover.journal_path(root).exists()


@pytest.mark.parametrize(
    "kind", ["workers", "allocations", "demand", "ingestion", "review", "schema"]
)
def test_quiescence_is_required_even_with_force(flat_store, kind):
    root, _, _, _ = flat_store
    with sqlite3.connect(root / "registry.sqlite3") as db:
        if kind == "workers":
            db.execute(
                "INSERT INTO workers(id,user_name,hostname,pid,resource_class,state,started_at,updated_at) VALUES ('w','user','host',1,'large','running','now','now')"
            )
        elif kind == "allocations":
            db.execute(
                "INSERT INTO scheduler_submissions(intent_token,resource_class,state,created_at) VALUES ('token','large','submitted','now')"
            )
        elif kind == "demand":
            db.execute(
                "INSERT INTO workflow_revisions VALUES (1,'main',1,'definition','workflow.yml','{}','now')"
            )
            db.execute(
                "INSERT INTO requests VALUES ('r','user','demo',1,'anat','{}',1,NULL,'active','now','now')"
            )
        elif kind == "schema":
            db.execute("UPDATE metadata SET value='0' WHERE key='schema_version'")
    if kind in {"ingestion", "review"}:
        path = root / (
            "ingestion/request.json" if kind == "ingestion" else "ingestion/reviews/request.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"state": "running"} if kind == "ingestion" else {"expires": 99999999999})
        )
    with pytest.raises(SystemExit) as error:
        command.main(["--registry", str(root), "--force"])
    assert error.value.code == 1
    assert (root / "registry.sqlite3").exists() and not (root / "shared").exists()
    assert not cutover.journal_path(root).exists()


@pytest.mark.parametrize("kind", ["unknown", "symlink", "changed"])
def test_source_changes_are_not_ignored(flat_store, kind):
    root, _, _, _ = flat_store
    plan = cutover.preview(root)
    if kind == "unknown":
        (root / "unknown").write_text("retain")
    elif kind == "symlink":
        (root / "snapshots/link").symlink_to(root / "registry.sqlite3")
    else:
        (root / "snapshots/main_preprocess.yml").write_text("new setting")
    with pytest.raises(ValueError):
        cutover.execute(root, expected_fingerprint=plan.source_fingerprint)
    assert (root / "registry.sqlite3").is_file()


def test_dry_run_and_cancellation_do_not_write(flat_store, monkeypatch):
    root, _, _, _ = flat_store
    before = cutover._inventory(root)
    command.main(["--registry", str(root), "--dry-run"])
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit) as error:
        command.main(["--registry", str(root)])
    assert error.value.code == 0
    assert cutover._inventory(root) == before
    assert not cutover.journal_path(root).exists()


def test_failed_copy_can_be_rolled_back_without_deletion(flat_store, monkeypatch):
    root, _, _, _ = flat_store
    before = cutover._inventory(root)
    original = cutover._stage

    def fail(*args):
        original(*args)
        raise OSError("interrupted copying")

    with monkeypatch.context() as patch:
        patch.setattr(cutover, "_stage", fail)
        with pytest.raises(OSError, match="interrupted"):
            cutover.execute(root)
    with pytest.raises(ValueError, match="incomplete"):
        ControlPaths(root).require_current_layout()
    with pytest.raises(ValueError, match="roll back"):
        cutover.execute(root)
    retained = cutover.rollback(root)
    assert retained.is_dir()
    assert cutover._inventory(root) == before
    assert not cutover.journal_path(root).exists()


@pytest.mark.parametrize(
    "point", ["before_backup", "after_backup", "after_publication", "before_completion"]
)
def test_publication_can_resume_at_each_rename_boundary(flat_store, monkeypatch, point):
    root, _, _, _ = flat_store
    original = Path.rename

    def rename(source, target):
        target = Path(target)
        if point == "before_backup" and source == root:
            raise OSError("interrupted publication")
        if point == "before_completion" and source == cutover.journal_path(root):
            raise OSError("interrupted publication")
        result = original(source, target)
        if (point == "after_backup" and source == root) or (
            point == "after_publication" and target == root
        ):
            raise OSError("interrupted publication")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", rename)
        with pytest.raises(OSError, match="interrupted"):
            cutover.execute(root)
    with pytest.raises(ValueError, match="incomplete"):
        ControlPaths(root).require_current_layout()
    backup = cutover.execute(root)
    assert (backup / "registry.sqlite3").exists()
    ControlPaths(root).require_current_layout()
    assert not cutover.journal_path(root).exists()


def test_scientific_private_path_dependency_is_not_rewritten(flat_store):
    root, _, _, _ = flat_store
    path = root / "manifests/instance.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_contract"]["inputs"] = [str(root / "snapshots/main_preprocess.yml")]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Scientific field"):
        cutover.execute(root)
    assert json.loads(path.read_text()) == manifest
    cutover.rollback(root)


def test_branch_databases_keep_their_identities(flat_store):
    from nro.orchestration.branch_registry import APPLICATION_ID as BRANCH_APP
    from nro.orchestration.branch_registry import SCHEMA
    from nro.orchestration.branch_registry import SCHEMA_VERSION as BRANCH_VERSION
    from nro.orchestration.branch_store import BranchStore, _encode
    from nro.orchestration.branches import BranchTopology

    root, _, _, _ = flat_store
    records = BranchTopology.reserved()
    parent = root / "branches"
    parent.mkdir()
    (parent / "registrations.json").write_text(json.dumps(_encode(records)))
    for name, record in records.records.items():
        directory = parent / "registries" / name
        directory.mkdir(parents=True)
        with sqlite3.connect(directory / "registry.sqlite3") as db:
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA application_id={BRANCH_APP}")
            db.execute(f"PRAGMA user_version={BRANCH_VERSION}")
            db.executemany(
                "INSERT INTO identity VALUES (?,?)",
                dict(
                    branch=name, registry_id=record.registry_id, scheduler_control=str(root)
                ).items(),
            )
    cutover.execute(root)
    store = BranchStore(root)
    assert store.read().topology.records == records.records
    assert store.registry("dev").instances() == ()


def test_execution_source_cache_remains_byte_identical(flat_store, tmp_path):
    from nro.orchestration.source_snapshots import SourceSnapshot, SourceStore

    root, _, _, _ = flat_store
    source = tmp_path / "source"
    (source / "nro").mkdir(parents=True)
    (source / "nro/__init__.py").write_text(f"PRIVATE = {str(root / 'snapshots/config.yml')!r}\n")
    snapshot = SourceStore(root / "implementations").capture(source)
    digest = snapshot.digest
    cutover.execute(root)
    SourceSnapshot(ControlPaths(root).implementations / digest, digest).verify()


def test_resume_does_not_start_a_new_cutover(flat_store):
    root, _, _, _ = flat_store
    with pytest.raises(SystemExit) as error:
        command.main(["--registry", str(root), "--resume", "--force"])
    assert error.value.code == 1
    assert not cutover.journal_path(root).exists()


def test_json_command_emits_one_result(flat_store, capsys):
    root, _, _, _ = flat_store
    command.main(["--registry", str(root), "--json", "--force"])
    value = json.loads(capsys.readouterr().out)
    assert value["complete"] and Path(value["backup"]).is_dir()


def test_published_pending_cutover_can_roll_back(flat_store, monkeypatch):
    root, _, _, _ = flat_store
    before = cutover._inventory(root)
    original = Path.rename

    def fail(source, target):
        if source == cutover.journal_path(root):
            raise OSError("interrupted finalization")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail)
        with pytest.raises(OSError, match="interrupted"):
            cutover.execute(root)
    retained = cutover.rollback(root)
    assert retained.is_dir()
    assert cutover._inventory(root) == before


def test_changed_staging_is_not_published(flat_store, monkeypatch):
    root, _, _, _ = flat_store
    original = Path.rename

    def fail(source, target):
        if source == root:
            raise OSError("interrupted")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail)
        with pytest.raises(OSError):
            cutover.execute(root)
    journal = json.loads(cutover.journal_path(root).read_text())
    staged = Path(journal["staged"])
    (staged / "shared/cutover.json").write_text("{}")
    with pytest.raises(ValueError, match="Staged state changed"):
        cutover.execute(root)
    assert (root / "registry.sqlite3").exists()
    cutover.rollback(root)


def test_missing_inventory_is_not_treated_as_an_empty_store(tmp_path):
    with pytest.raises(ValueError, match="missing or redirected"):
        cutover._inventory(tmp_path / "missing")


def test_ambiguous_recovery_does_not_replace_any_directory(flat_store):
    root, _, _, _ = flat_store
    token = "a" * 32
    staged = root.with_name(f"{root.name}.cutover-staged-{token}")
    backup = root.with_name(f"{root.name}.cutover-backup-{token}")
    staged.mkdir()
    backup.mkdir()
    cutover._save(
        root,
        dict(root=str(root), token=token, state="prepared", staged=str(staged), backup=str(backup)),
    )
    for operation in (cutover.execute, cutover.rollback):
        with pytest.raises(ValueError, match="Ambiguous"):
            operation(root)
    assert all(path.is_dir() for path in (root, staged, backup))
