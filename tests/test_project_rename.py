import hashlib
import json
from pathlib import Path

from nro.configuration.store import fingerprint
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchTopology
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.project_rename import (
    ProjectMove,
    _prepare_rename,
    _project_inventory,
    _replace_project,
    _rewrite_central,
    _rewrite_markup,
    _rewrite_receipts,
    _translate,
    decode_preparation,
    encode_preparation,
    execute,
)
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.runner_graph import RunnerGraph, Step


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


def test_project_inventory_does_not_walk_project_data_trees(tmp_path):
    source = tmp_path / "BIDS/old"
    target = tmp_path / "shared.nii.gz"
    target.write_bytes(b"image")
    raw_link = source / "sourcedata/sub-01/ses-01/image.nii.gz"
    raw_link.parent.mkdir(parents=True)
    raw_link.symlink_to(target)
    derivative_link = source / "derivatives/nro/anat/main/sub-01/anat/source.nii.gz"
    derivative_link.parent.mkdir(parents=True)
    derivative_link.symlink_to(target)

    inventory = _project_inventory(
        (ProjectMove(source, tmp_path / "BIDS/new"),),
        old="old",
    )

    assert inventory.scanned == 0


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
        "ingestion_records": 0,
        "definition_files": [],
        "moves": [{"source": "/bids/old", "destination": "/bids/new"}],
        "blockers": [],
        "preparation": "prepared-rename",
    }
    calls = []
    pages = []

    monkeypatch.setattr(
        "nro.configuration.site.settings",
        lambda: (
            {"registry": str(tmp_path / "control"), "bids": str(tmp_path / "BIDS")},
            None,
        ),
    )
    monkeypatch.setattr(project_cli, "page_text", pages.append)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    def maintenance(*_args, **fields):
        calls.append(fields)
        return preview if not fields["execute"] else {**preview, "executed": True}

    monkeypatch.setattr("nro.orchestration.scheduler_client.maintenance", maintenance)

    project_cli.main(["rename", "old", "new"])

    assert [call["execute"] for call in calls] == [False, True]
    assert calls[1]["preparation"] == "prepared-rename"
    assert pages and "Project rename: old -> new" in pages[0]
    assert "Renamed project old to new." in capsys.readouterr().out


def test_project_cli_discards_cancelled_preparation(tmp_path, monkeypatch, capsys):
    from nro.bin import project as project_cli

    preview = {
        "old": "old",
        "new": "new",
        "work_items": 0,
        "inventory_entries": 0,
        "ownership_receipts": 0,
        "scene_files": 0,
        "metadata_files": 0,
        "ingestion_records": 0,
        "definition_files": [],
        "moves": [],
        "blockers": [],
        "preparation": "prepared-rename",
    }
    calls = []
    monkeypatch.setattr(
        "nro.configuration.site.settings",
        lambda: (
            {"registry": str(tmp_path / "control"), "bids": str(tmp_path / "BIDS")},
            None,
        ),
    )
    monkeypatch.setattr(project_cli, "page_text", lambda _text: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    def maintenance(*_args, **fields):
        calls.append(fields)
        return preview if fields["operation"] == "project_rename" else {"cancelled": True}

    monkeypatch.setattr("nro.orchestration.scheduler_client.maintenance", maintenance)

    project_cli.main(["rename", "old", "new"])

    assert [call["operation"] for call in calls] == [
        "project_rename",
        "preparation_cancel",
    ]
    assert calls[1]["preparation"] == "prepared-rename"
    assert "Project rename cancelled." in capsys.readouterr().out


def test_project_cli_force_prints_preview_without_pager_or_prompt(tmp_path, monkeypatch, capsys):
    from nro.bin import project as project_cli

    preview = {
        "old": "old",
        "new": "new",
        "work_items": 0,
        "inventory_entries": 0,
        "ownership_receipts": 0,
        "scene_files": 0,
        "metadata_files": 0,
        "ingestion_records": 0,
        "definition_files": [],
        "moves": [],
        "blockers": [],
        "preparation": "prepared-rename",
    }
    calls = []
    monkeypatch.setattr(
        "nro.configuration.site.settings",
        lambda: (
            {"registry": str(tmp_path / "control"), "bids": str(tmp_path / "BIDS")},
            None,
        ),
    )
    monkeypatch.setattr(
        project_cli,
        "page_text",
        lambda _text: (_ for _ in ()).throw(AssertionError("unexpected pager")),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
    )

    def maintenance(*_args, **fields):
        calls.append(fields)
        return preview if not fields["execute"] else {**preview, "executed": True}

    monkeypatch.setattr("nro.orchestration.scheduler_client.maintenance", maintenance)

    project_cli.main(["rename", "old", "new", "-f"])

    assert [call["execute"] for call in calls] == [False, True]
    assert "Project rename: old -> new" in capsys.readouterr().out


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
    central_lineage = fingerprint({"owner": "main-registry", "lineage": lineage})
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
            ("anat", "main", "config", central_lineage, "{}\n", "main", now),
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
            """INSERT INTO work_item_execution
               (work_item_id,branch,registry_id,logical_key,context_json,
                binding_sources_json,provenance_json,scientific_contract_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (item_id, "main", "main-registry", old_key, "{}", "{}", "{}", "{}"),
        )
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
        old_digest = old_key.split(":", 1)[-1][:16]
        old_log = (
            control / f"branches/main/events/old/anat/sub-01/sub-01/{old_digest}/work-item.log"
        )
        db.execute(
            """INSERT INTO attempts
               (work_item_id,worker_id,state,revision_fingerprint,memory_gb,oom_detected,
                process_group_id,started_at,completed_at,error_type,error_message,log_path,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item_id,
                None,
                "success",
                "revision",
                32,
                0,
                0,
                now,
                now,
                None,
                None,
                str(old_log),
                now,
            ),
        )

    changed_metadata = bids / "new/derivatives/nro/anat/main/sub-01/manifest.json"
    changed_metadata.parent.mkdir(parents=True)
    changed_metadata.write_text('{"project":"new"}\n')

    with registry.connection(write=True) as db:
        changed, mapping = _rewrite_central(
            db,
            "old",
            "new",
            changed_metadata=(changed_metadata,),
            branch_lineages={("main-registry", "anat", "main"): lineage},
        )

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
        assert db.execute("SELECT logical_key FROM work_item_execution").fetchone()[0] == new_key
        attempt_log = db.execute("SELECT log_path FROM attempts").fetchone()[0]
        new_digest = new_key.split(":", 1)[-1][:16]
        assert attempt_log == str(
            control / f"branches/main/events/new/anat/sub-01/sub-01/{new_digest}/work-item.log"
        )


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
    derivative = bids / "old/derivatives/nro/anat/main/sub-01"
    derivative.mkdir(parents=True)
    derivative_link = derivative / "source-link"
    derivative_link.symlink_to("../../../../../marker.txt")
    manifest = derivative / "sub-01_manifest.json"
    manifest.write_text(json.dumps({"input": str(bids / "old/marker.txt")}))
    surface = derivative / "sub-01_hemi-L_white.surf.gii"
    surface.write_text("surface")
    lineage = "lineage-fingerprint"
    old_key = work_item_key("old", "anat", lineage, "01", {})
    now = utcnow()
    with registry.connection(write=True) as db:
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
                "{}",
                fingerprint({}),
                "[]",
                str(tmp_path / "runtime.yml"),
                json.dumps([str(bids / "old/marker.txt")]),
                str(derivative),
                "sub-01",
                json.dumps([str(manifest)]),
                now,
                now,
            ),
        )
    monkeypatch.setattr(
        "nro.orchestration.project_rename._registered_project_files",
        lambda _registry, _project: (derivative_link, manifest),
    )
    for name in topology.records:
        events = control / "branches" / name / "events/old"
        events.mkdir(parents=True)
        (events / "instance.log").write_text("log")
    old_digest = old_key.split(":", 1)[-1][:16]
    old_event = control / f"branches/main/events/old/anat/sub-01/sub-01/{old_digest}"
    old_event.mkdir(parents=True)
    old_graph = RunnerGraph("Anatomical Module")
    old_graph.add(
        Step.command_step(
            (
                "tool",
                "--input",
                str(bids / "old/marker.txt"),
                "--output",
                str(manifest),
            ),
            name="Existing anatomical step",
            inputs=(bids / "old/marker.txt",),
            outputs=(manifest,),
            parameters={"reference": str(bids / "old/marker.txt")},
        )
    )
    old_graph.add(
        Step.command_step(
            ("tool", "--input", str(manifest), "--output", str(surface)),
            name="Dependent surface step",
            inputs=(manifest,),
            outputs=(surface,),
        )
    )
    old_graph.freeze()
    old_graph.reconcile_contract(old_event / "runner-contract.json", signature=old_key)
    values = {
        "bids": str(bids),
        "work": str(work),
        "development": str(development),
        "registry": str(control),
        "definitions": str(definitions_fixture),
    }

    prepared = _prepare_rename(registry, checkout=checkout, values=values, old="old", new="new")
    prepared = decode_preparation(encode_preparation(prepared))
    monkeypatch.setattr(
        "nro.orchestration.project_rename._project_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("execution repeated project discovery")
        ),
    )
    result = execute(
        registry,
        checkout=checkout,
        values=values,
        old="old",
        new="new",
        prepared=prepared,
    )

    assert result["executed"] is True
    for root in roots:
        assert not (root / "old").exists()
        assert (root / "new/marker.txt").is_file()
    assert (bids / "new/derivatives/nro/anat/main/sub-01/source-link").readlink() == Path(
        "../../../../../marker.txt"
    )
    assert (
        bids / "new/derivatives/nro/anat/main/sub-01/source-link"
    ).resolve() == bids / "new/marker.txt"
    assert json.loads(
        (bids / "new/derivatives/nro/anat/main/sub-01/sub-01_manifest.json").read_text()
    )["input"] == str(bids / "new/marker.txt")
    new_key = work_item_key("new", "anat", lineage, "01", {})
    new_digest = new_key.split(":", 1)[-1][:16]
    new_contract = (
        control / f"branches/main/events/new/anat/sub-01/sub-01/{new_digest}/runner-contract.json"
    )
    assert new_contract.is_file()
    new_graph = RunnerGraph("Anatomical Module")
    new_graph.add(
        Step.command_step(
            (
                "tool",
                "--input",
                str(bids / "new/marker.txt"),
                "--output",
                str(bids / "new/derivatives/nro/anat/main/sub-01/sub-01_manifest.json"),
            ),
            name="Existing anatomical step",
            inputs=(bids / "new/marker.txt",),
            outputs=(bids / "new/derivatives/nro/anat/main/sub-01/sub-01_manifest.json",),
            parameters={"reference": str(bids / "new/marker.txt")},
        )
    )
    new_manifest = bids / "new/derivatives/nro/anat/main/sub-01/sub-01_manifest.json"
    new_surface = bids / "new/derivatives/nro/anat/main/sub-01/sub-01_hemi-L_white.surf.gii"
    new_graph.add(
        Step.command_step(
            ("tool", "--input", str(new_manifest), "--output", str(new_surface)),
            name="Dependent surface step",
            inputs=(new_manifest,),
            outputs=(new_surface,),
        )
    )
    new_graph.freeze()
    assert new_graph.step_contract_changes(new_contract, signature=new_key) == {}
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
