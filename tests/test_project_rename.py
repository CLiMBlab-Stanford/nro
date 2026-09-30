import hashlib
import json
from pathlib import Path

import pytest

from nro.configuration.store import fingerprint
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchTopology
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.project_rename import (
    ProjectMove,
    _copy_or_link,
    _project_inventory,
    _replace_project,
    _rewrite_central,
    _rewrite_markup,
    _rewrite_receipts,
    _translate,
    execute,
)
from nro.orchestration.registry import Registry, utcnow


def test_translation_changes_identities_and_paths_without_rewriting_prose():
    mapping = {"anat:old": "anat:new"}
    value = {
        "project": "old",
        "key": "anat:old",
        "path": "/data/old/sub-01/anat",
        "note": "the old project remains historical prose",
    }

    assert _translate(value, "old", "new", mapping) == {
        "project": "new",
        "key": "anat:new",
        "path": "/data/new/sub-01/anat",
        "note": "the old project remains historical prose",
    }
    assert _replace_project("/data/old", "old", "new", {}) == "/data/new"


def test_copy_fallback_preserves_file_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source bytes")
    source.chmod(0o640)

    def fail_link(*_args):
        raise OSError("cross-device link")

    monkeypatch.setattr("nro.orchestration.project_rename.os.link", fail_link)

    _copy_or_link(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == source.stat().st_mode & 0o777
    assert destination.stat().st_mtime_ns == source.stat().st_mtime_ns


def test_project_inventory_fails_closed_on_unreadable_directory(tmp_path, monkeypatch):
    source = tmp_path / "BIDS/old"
    source.mkdir(parents=True)

    def unreadable(_root, *, topdown, followlinks, onerror):
        assert topdown is True and followlinks is False
        onerror(PermissionError(13, "Permission denied", str(source / "private")))
        yield  # pragma: no cover

    monkeypatch.setattr("nro.orchestration.project_rename.os.walk", unreadable)

    with pytest.raises(OSError, match="Cannot inventory project directory.*private"):
        _project_inventory(
            (ProjectMove(source, tmp_path / "BIDS/new"),),
            bids_root=tmp_path / "BIDS",
            old="old",
            new="new",
        )


def test_project_cli_pages_preview_and_confirms_execution(tmp_path, monkeypatch, capsys):
    from nro.bin import project as project_cli

    preview = {
        "old": "old",
        "new": "new",
        "work_items": 2,
        "inventory_entries": 4,
        "ownership_receipts": 1,
        "scene_files": 0,
        "metadata_files": 1,
        "source_symlinks": 0,
        "absolute_symlinks": 0,
        "ingestion_records": 0,
        "definition_files": [],
        "moves": [{"source": "/bids/old", "destination": "/bids/new"}],
        "blockers": [],
    }
    calls = []
    pages = []

    monkeypatch.setattr(
        "nro.configuration.site.settings",
        lambda: ({"registry": str(tmp_path / "control"), "bids": str(tmp_path / "BIDS")}, None),
    )
    monkeypatch.setattr(project_cli, "page_text", pages.append)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    def maintenance(*_args, **fields):
        calls.append(fields["execute"])
        return preview if not fields["execute"] else {**preview, "executed": True}

    monkeypatch.setattr("nro.orchestration.scheduler_client.maintenance", maintenance)

    project_cli.main(["rename", "old", "new"])

    assert calls == [False, True]
    assert pages and "Project rename: old -> new" in pages[0]
    assert "Renamed project old to new." in capsys.readouterr().out


def test_markup_rename_preserves_comments_and_rejects_collisions(tmp_path):
    path = tmp_path / "main_markup.yml"
    path.write_text("# managed definition\nold:\n  sub-01:\n    lesion: false\n")

    changed = _rewrite_markup(path, "old", "new")

    assert changed == "# managed definition\nnew:\n  sub-01:\n    lesion: false\n"
    path.write_text("old: {}\nnew: {}\n")
    try:
        _rewrite_markup(path, "old", "new")
    except ValueError as error:
        assert "already defines" in str(error)
    else:
        raise AssertionError("Expected a destination-project collision")


def test_receipt_rename_uses_the_project_root_and_new_key(tmp_path):
    project = tmp_path / "BIDS/old"
    receipt = project / "derivatives/nro/anat/main-123/.nro/work_items/anat/old-digest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "work_item_key": "anat:old-digest",
                "module": "anat",
                "project": "old",
                "directory_label": "main-123",
                "artifact_contract": {
                    "output": {"root": str(project / "derivatives/nro/anat/main-123")}
                },
            }
        )
    )

    changed, created = _rewrite_receipts(
        (receipt,), "old", "new", {"anat:old-digest": "anat:new-digest"}
    )

    expected = project / "derivatives/nro/anat/main-123/.nro/work_items/anat/new-digest.json"
    assert changed == 1
    assert created == (expected,)
    assert not receipt.exists()
    record = json.loads(expected.read_text())
    assert record["project"] == "new"
    assert record["work_item_key"] == "anat:new-digest"
    assert "/new/" in record["artifact_contract"]["output"]["root"]


def test_central_registry_translation_preserves_work_item_state(tmp_path):
    bids = tmp_path / "BIDS"
    control = tmp_path / "control"
    registry = Registry.for_project("", bids_root=bids, registry_path=control)
    registry.initialize()
    now = utcnow()
    lineage = "lineage-fingerprint"
    old_key = work_item_key("old", "anat", lineage, "01", {})
    contract = {
        "configuration_fingerprint": "config",
        "dependencies": [],
        "inputs": [str(bids / "old/sub-01/anat/sub-01_T1w.nii.gz")],
        "output": {
            "root": str(bids / "old/derivatives/nro/anat/main/sub-01"),
            "prefix": "sub-01",
            "expected": [],
            "format": "directory",
        },
        "processing": {},
    }
    with registry.connection(write=True) as db:
        db.execute(
            "INSERT INTO bids_projects VALUES (?,?,?)",
            ("old", str(bids / "old"), now),
        )
        db.execute(
            "INSERT INTO bids_participants VALUES (?,?,?,?)",
            ("old", "01", str(bids / "old/sub-01"), now),
        )
        cursor = db.execute(
            """INSERT INTO module_lineages
               (configuration_class,config_id,config_fingerprint,lineage_fingerprint,
                resolved_yaml,directory_label,created_at) VALUES (?,?,?,?,?,?,?)""",
            ("anat", "main", "config", lineage, "{}\n", "main", now),
        )
        db.execute(
            """INSERT INTO work_items
               (work_item_key,module,module_lineage_id,project,participant,entities_json,
                scope,artifact_state,artifact_reason,current_generation,resource_class,
                memory_gb,max_memory_gb,revision_fingerprint,artifact_contract_json,
                artifact_fingerprint,command_json,runtime_config_path,input_paths_json,
                output_root,output_prefix,expected_outputs_json,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                old_key,
                "anat",
                cursor.lastrowid,
                "old",
                "01",
                "{}",
                "participant",
                "fresh",
                None,
                1,
                "cpu",
                32,
                256,
                "revision",
                json.dumps(contract),
                fingerprint(contract),
                json.dumps(["nro", "run", "-P", "old"]),
                str(tmp_path / "runtime.yml"),
                json.dumps(contract["inputs"]),
                contract["output"]["root"],
                "sub-01",
                "[]",
                now,
                now,
            ),
        )
        item_id = db.execute("SELECT id FROM work_items").fetchone()[0]
        db.execute(
            """INSERT INTO artifacts
               (work_item_id,attempt_id,direction,path,size,mtime_ns,
                digest_algorithm,digest,metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                item_id,
                None,
                "output",
                str(bids / "old/derivatives/nro/anat/main/sub-01/manifest.json"),
                1,
                1,
                "sha256",
                "old-digest",
                "{}",
            ),
        )

    changed_metadata = bids / "new/derivatives/nro/anat/main/sub-01/manifest.json"
    changed_metadata.parent.mkdir(parents=True)
    changed_metadata.write_text('{"project":"new"}\n')

    with registry.connection(write=True) as db:
        changed, mapping = _rewrite_central(db, "old", "new", changed_metadata=(changed_metadata,))

    assert changed == 1
    new_key = work_item_key("new", "anat", lineage, "01", {})
    assert mapping[old_key] == new_key
    with registry.connection() as db:
        item = db.execute("SELECT * FROM work_items").fetchone()
        assert item["project"] == "new"
        assert item["work_item_key"] == new_key
        assert item["artifact_state"] == "fresh"
        assert "/new/" in item["output_root"]
        assert json.loads(item["command_json"])[-1] == "new"
        assert db.execute("SELECT project FROM bids_projects").fetchone()[0] == "new"
        assert db.execute("SELECT project FROM bids_participants").fetchone()[0] == "new"
        artifact = db.execute("SELECT * FROM artifacts").fetchone()
        assert artifact["path"] == str(changed_metadata)
        assert artifact["digest"] == hashlib.sha256(changed_metadata.read_bytes()).hexdigest()
        assert artifact["mtime_ns"] == changed_metadata.stat().st_mtime_ns


def test_execute_moves_each_managed_project_root(tmp_path, definitions_fixture, monkeypatch):
    bids = tmp_path / "BIDS"
    work = tmp_path / "WORK"
    development = tmp_path / "NRO_DEV"
    control = tmp_path / "control"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry = Registry.for_project("", bids_root=bids, registry_path=control)
    registry.initialize()
    with registry.connection(write=True) as db:
        db.execute(
            "INSERT INTO bids_projects VALUES (?,?,?)",
            ("old", str(bids / "old"), utcnow()),
        )
    store = BranchStore(control)
    topology = store.initialize().topology
    monkeypatch.setattr(
        BranchTopology,
        "registered_checkout",
        lambda self, candidate: "main" if candidate == checkout else None,
    )
    monkeypatch.setattr("nro.orchestration.planner_client.shutdown", lambda _control: None)
    roots = (bids, work, development / "dev/BIDS", development / "dev/WORK")
    for root in roots:
        project = root / "old"
        project.mkdir(parents=True)
        (project / "marker.txt").write_text("present")
    absolute_link = bids / "old/absolute-link"
    absolute_link.symlink_to(bids / "old/marker.txt")
    derivative = bids / "old/derivatives/nro/anat/main/sub-01"
    derivative.mkdir(parents=True)
    derivative_link = derivative / "source-link"
    derivative_link.symlink_to(bids / "old/marker.txt")
    manifest = derivative / "sub-01_manifest.json"
    manifest.write_text(json.dumps({"input": str(bids / "old/marker.txt")}))
    for name in topology.records:
        events = control / "branches" / name / "events/old"
        events.mkdir(parents=True)
        (events / "instance.log").write_text("log")
    values = {
        "bids": str(bids),
        "work": str(work),
        "development": str(development),
        "registry": str(control),
        "definitions": str(definitions_fixture),
    }

    result = execute(registry, checkout=checkout, values=values, old="old", new="new")

    assert result["executed"] is True
    for root in roots:
        assert not (root / "old").exists()
        assert (root / "new/marker.txt").is_file()
    materialized = bids / "new/absolute-link"
    assert materialized.is_file() and not materialized.is_symlink()
    assert materialized.samefile(bids / "new/marker.txt")
    assert (
        bids / "new/derivatives/nro/anat/main/sub-01/source-link"
    ).readlink() == bids / "new/marker.txt"
    assert json.loads(
        (bids / "new/derivatives/nro/anat/main/sub-01/sub-01_manifest.json").read_text()
    )["input"] == str(bids / "new/marker.txt")
    for name in topology.records:
        assert not (control / "branches" / name / "events/old").exists()
        assert (control / "branches" / name / "events/new/instance.log").is_file()
    assert (Path(result["journal"]) / "journal.json").is_file()

    journal = Path(result["journal"]) / "journal.json"
    interrupted = json.loads(journal.read_text())
    interrupted["state"] = "prepared"
    journal.write_text(json.dumps(interrupted))
    with registry.connection(write=True) as db:
        db.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('project_rename_commit',?)",
            (str(journal),),
        )
        db.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('maintenance_mode','project rename')"
        )

    recovered = execute(registry, checkout=checkout, values=values, old="old", new="new")

    assert recovered["executed"] is True
    assert recovered["recovered"] is True
    assert json.loads(journal.read_text())["state"] == "complete"
    with registry.connection() as db:
        assert not db.execute(
            "SELECT 1 FROM metadata WHERE key IN ('maintenance_mode','project_rename_commit')"
        ).fetchall()
