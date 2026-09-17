from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from nro.bin.purge import _purge_work_items
from nro.configuration.store import ConfigStore
from nro.orchestration.discovery import register_existing_artifacts
from nro.orchestration.ownership import (
    complete_ownership_records,
    lineage_record_path,
    read_ownership_records,
    work_item_record_path,
    write_work_item_ownership,
)
from nro.orchestration.planner import build_subject_work_items
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.registry import Registry


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_completion_uses_receipts_from_selected_lineage_root(tmp_path: Path) -> None:
    lineage = {
        "lineage_fingerprint": "same-lineage",
        "directory_label": "current",
        "upstream": [],
    }
    previous = ({"lineage_fingerprint": "same-lineage", "directory_label": "previous"}, tmp_path)
    current = ({"lineage_fingerprint": "same-lineage", "directory_label": "current"}, tmp_path)

    lineages, work_items, errors = complete_ownership_records([lineage], [previous, current])

    assert lineages == [lineage]
    assert work_items == [current]
    assert errors == []


def test_discovery_rejects_duplicate_roots_for_one_lineage(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo/sub-01"
    _write(subject / "anat/sub-01_T1w.nii.gz")
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="anat",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    ids = registry.register_work_items(work_items)
    write_work_item_ownership(registry, ids[work_items[0].key])
    original = lineage_record_path(bids / "demo", "anat", registered.directories["anat"])
    duplicate = lineage_record_path(bids / "demo", "anat", "main-2")
    marker = json.loads(original.read_text())
    marker["directory_label"] = "main-2"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(json.dumps(marker))

    lineages, _work_items, errors = read_ownership_records(bids, ["demo"])

    assert lineages == []
    assert any("conflicting derivative roots" in error for error in errors)


def _historical_store(tmp_path: Path) -> ConfigStore:
    root = tmp_path / "configs"
    shutil.copytree(ConfigStore().root, root)
    (root / "configs" / "clean" / "retired_clean.yml").write_text(
        yaml.safe_dump({"standardize": False})
    )
    (root / "workflows" / "retired_workflow.yml").write_text(yaml.safe_dump({"clean": "retired"}))
    store = ConfigStore()
    store.root = root
    return store


def test_work_item_ownership_survives_removed_workflow(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    store = _historical_store(tmp_path)
    workflow = store.resolve("retired")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="clean",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    work_item_ids = registry.register_work_items(work_items)
    for work_item in work_items:
        write_work_item_ownership(registry, work_item_ids[work_item.key])

    clean = next(work_item for work_item in work_items if work_item.module == "clean")
    marker = lineage_record_path(bids / "demo", "clean", registered.directories["clean"])
    receipt = work_item_record_path(
        bids / "demo",
        "clean",
        registered.directories["clean"],
        "clean",
        clean.key,
    )
    assert marker.is_file()
    assert receipt.is_file()
    marker_document = json.loads(marker.read_text())
    assert marker_document["configuration"]["id"] == "retired"
    assert marker_document["upstream"][0]["configuration_class"] == "func"

    (store.root / "workflows" / "retired_workflow.yml").unlink()
    (store.root / "configs" / "clean" / "retired_clean.yml").unlink()
    registry.reinitialize()
    registry.replace_bids_inventory({"demo": ("01",)})
    result = register_existing_artifacts(
        registry,
        bids_root=bids,
        inventory={"demo": ("01",)},
        store=store,
    )

    rows = {row["work_item_key"]: row for row in registry.work_item_rows()}
    assert clean.key in rows
    assert rows[clean.key]["directory_label"] == "retired"
    assert rows[clean.key]["workflow_ids"] is None
    status = {row["work_item_key"]: row for row in registry.work_item_status_snapshot()}
    assert status[clean.key]["status"] == "Unavailable"
    assert not status[clean.key]["recomputable"]
    assert result.artifacts == len(work_items)
    assert result.unavailable == ()

    _purge_work_items(
        [(registry, [rows[clean.key]])],
        work_root=tmp_path / "work",
        dry_run=False,
    )
    assert not receipt.exists()
    assert not marker.exists()


def test_planned_work_item_key_uses_stable_lineage_fingerprint(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    workflow = ConfigStore().resolve("main")
    first = Registry.for_project("demo", bids_root=bids, registry_path=tmp_path / "first" / ".nro")
    first_registered = first.register_workflow(workflow)
    first_work_item = build_subject_work_items(
        project="demo",
        participant="01",
        module="anat",
        workflow=workflow,
        registered=first_registered,
        registry=first,
        bids_root=bids,
    )[0]

    second = Registry.for_project(
        "demo", bids_root=bids, registry_path=tmp_path / "second" / ".nro"
    )
    with second.connection(write=True) as db:
        db.execute(
            """INSERT INTO module_lineages(
                   configuration_class, config_id, config_fingerprint,
                   lineage_fingerprint, resolved_yaml, directory_label, created_at
               ) VALUES ('clean', 'other', 'other', 'other', '{}', 'other', 'now')"""
        )
    second_registered = second.register_workflow(workflow)
    second_work_item = build_subject_work_items(
        project="demo",
        participant="01",
        module="anat",
        workflow=workflow,
        registered=second_registered,
        registry=second,
        bids_root=bids,
    )[0]

    assert first_registered.lineages["anat"] != second_registered.lineages["anat"]
    assert first_work_item.key == second_work_item.key


def test_work_item_key_retains_pre_terminology_identity() -> None:
    assert work_item_key("demo", "clean", "lineage", "01", {"space": "T1w"}) == (
        "clean:c18c0eeff956df04e4199b78d10fd2c356d919f7fcc58a6db5b994fdbf8e216a"
    )
