from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from nro.configuration.store import ConfigStore
from nro.bin.purge import _purge_instances
from nro.orchestration.discovery import register_existing_artifacts
from nro.orchestration.ownership import (
    instance_record_path,
    lineage_record_path,
    write_instance_ownership,
)
from nro.orchestration.planner import build_subject_instances
from nro.orchestration.registry import Registry


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _historical_store(tmp_path: Path) -> ConfigStore:
    root = tmp_path / "configs"
    shutil.copytree(ConfigStore().root, root)
    (root / "configs" / "clean" / "retired_clean.yml").write_text(
        yaml.safe_dump({"standardize": False})
    )
    (root / "workflows" / "retired_workflow.yml").write_text(
        yaml.safe_dump({"clean": "retired"})
    )
    store = ConfigStore()
    store.root = root
    return store


def test_instance_ownership_survives_removed_workflow(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    store = _historical_store(tmp_path)
    workflow = store.resolve("retired")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="clean",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    instance_ids = registry.register_instances(instances)
    for instance in instances:
        write_instance_ownership(registry, instance_ids[instance.key])

    clean = next(instance for instance in instances if instance.module == "clean")
    marker = lineage_record_path(
        bids / "demo", "clean", registered.directories["clean"]
    )
    receipt = instance_record_path(
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
    assert marker_document["upstream"][0]["derivative_class"] == "preprocessing"

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

    rows = {row["instance_key"]: row for row in registry.instance_rows()}
    assert clean.key in rows
    assert rows[clean.key]["directory_label"] == "retired"
    assert rows[clean.key]["workflow_ids"] is None
    status = {
        row["instance_key"]: row
        for row in registry.instance_status_snapshot(expose_nonfresh=True)
    }
    assert status[clean.key]["status"] == "Unavailable"
    assert not status[clean.key]["recomputable"]
    assert result.artifacts == len(instances)
    assert result.unavailable == ()

    _purge_instances(
        [(registry, [rows[clean.key]])],
        work_root=tmp_path / "work",
        dry_run=False,
    )
    assert not receipt.exists()
    assert not marker.exists()


def test_planned_instance_key_uses_stable_lineage_fingerprint(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    workflow = ConfigStore().resolve("main")
    first = Registry.for_project(
        "demo", bids_root=bids, registry_path=tmp_path / "first" / ".nro"
    )
    first_registered = first.register_workflow(workflow)
    first_instance = build_subject_instances(
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
            """INSERT INTO configuration_lineages(
                   derivative_class, config_id, config_fingerprint,
                   lineage_fingerprint, resolved_yaml, directory_label, created_at
               ) VALUES ('clean', 'other', 'other', 'other', '{}', 'other', 'now')"""
        )
    second_registered = second.register_workflow(workflow)
    second_instance = build_subject_instances(
        project="demo",
        participant="01",
        module="anat",
        workflow=workflow,
        registered=second_registered,
        registry=second,
        bids_root=bids,
    )[0]

    assert first_registered.lineages["preprocessing"] != second_registered.lineages[
        "preprocessing"
    ]
    assert first_instance.key == second_instance.key
