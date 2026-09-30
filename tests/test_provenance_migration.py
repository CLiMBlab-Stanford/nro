from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from nro.configuration.store import ConfigStore, fingerprint
from nro.orchestration import scheduler_service
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.completion import record_completion
from nro.orchestration.contract_migrations import current_contract_schema
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.provenance_migration import (
    _candidates,
    _contract_dataset_view,
    _refresh_inventory_locked,
    _scientific_contract_dataset_view,
    _source_candidates,
    migrate_dataset,
)
from nro.orchestration.registry import Registry


def test_dataset_migration_previews_then_rewrites_metadata(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    output = project / "derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"image")
    manifest = output.with_name("sub-01_desc-preprocessAnat_manifest.json")
    manifest.write_text(json.dumps({"outputs": {"t1w": str(output)}}))
    os.utime(manifest, ns=(1_000_000_000, 2_000_000_000))
    original_mtime = manifest.stat().st_mtime_ns
    registry = Registry.for_project("", bids_root=bids)

    preview = migrate_dataset(registry, projects=("demo",), execute=False, version="1.2.3")
    assert preview.errors == ()
    assert set(preview.changed) == {
        manifest,
        project / "derivatives/nro/dataset_description.json",
    }
    assert str(output) in manifest.read_text()

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")
    assert result.errors == ()
    assert json.loads(manifest.read_text())["outputs"]["t1w"] == (
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    )
    assert manifest.stat().st_mtime_ns == original_mtime
    description = json.loads((project / "derivatives/nro/dataset_description.json").read_text())
    assert description["DatasetLinks"] == {"raw": "../.."}

    repeated = migrate_dataset(registry, projects=("demo",), execute=False, version="1.2.3")
    assert repeated.changed == ()


def test_dataset_migration_backup_does_not_copy_extended_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    output = project / "derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"image")
    manifest = output.with_name("manifest.json")
    manifest.write_text(json.dumps({"output": str(output)}))
    registry = Registry.for_project("", bids_root=bids)
    monkeypatch.setattr(
        "nro.orchestration.provenance_migration.shutil.copy2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError(1, "extended metadata is unsupported")
        ),
    )

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert result.errors == ()
    assert json.loads(manifest.read_text())["output"] == (
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    )


def test_dataset_migration_discards_legacy_partial_backup(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    project.mkdir(parents=True)
    registry = Registry.for_project("", bids_root=bids)
    journal = registry.paths.control / "shared/provenance-migrations/interrupted"
    backup = journal / "files/00000000"
    backup.parent.mkdir(parents=True)
    backup.write_text("backup made before the legacy manifest")

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert result.errors == ()
    assert not journal.exists()


def test_inventory_refresh_scans_artifacts_once(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("first")
    second.write_text("second")
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.execute(
        """CREATE TABLE artifacts (
               id INTEGER PRIMARY KEY,
               direction TEXT NOT NULL,
               path TEXT NOT NULL,
               size INTEGER,
               mtime_ns INTEGER,
               digest_algorithm TEXT,
               digest TEXT
           )"""
    )
    database.executemany(
        "INSERT INTO artifacts(direction,path) VALUES('output',?)",
        ((str(first.resolve()),), (str(first.resolve()),), (str(second.resolve()),)),
    )
    statements = []
    database.set_trace_callback(statements.append)

    _refresh_inventory_locked(database, (first, second))

    scans = [
        statement
        for statement in statements
        if statement == "SELECT id,path FROM artifacts WHERE direction='output'"
    ]
    assert len(scans) == 1
    rows = database.execute("SELECT size FROM artifacts ORDER BY id").fetchall()
    assert [row["size"] for row in rows] == [5, 5, 6]


def test_dataset_migration_resumes_registry_phase_without_restoring_files(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    output = project / "derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"image")
    manifest = output.with_name("manifest.json")
    original = json.dumps({"output": str(output)})
    migrated = json.dumps({"output": "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz"})
    manifest.write_text(migrated)
    registry = Registry.for_project("", bids_root=bids)
    journal = registry.paths.control / "shared/provenance-migrations/interrupted"
    backup = journal / "files/00000000"
    backup.parent.mkdir(parents=True)
    backup.write_text(original)
    (journal / "journal.json").write_text(
        json.dumps(
            {
                "format": 2,
                "state": "registry_pending",
                "projects": ["demo"],
                "files": [
                    {
                        "path": str(manifest.resolve()),
                        "backup": "files/00000000",
                        "existed": True,
                    }
                ],
            }
        )
    )

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert result.errors == ()
    assert manifest.read_text() == migrated
    assert not journal.exists()


def test_migration_inventory_prunes_non_bids_and_external_product_trees(
    tmp_path: Path,
) -> None:
    project = tmp_path / "bids/demo"
    raw = project / "sub-01/func/sub-01_task-rest_bold.json"
    excluded = project / "sub-01/func/_excluded/sub-01_task-rest_run-02_bold.json"
    sourcedata = project / "sourcedata/sub-01/func/sub-01_task-rest_bold.json"
    third_party = project / "derivatives/other/sub-01/manifest.json"
    manifest = project / "derivatives/nro/anat/main/sub-01/anat/manifest.json"
    abandoned = manifest.with_name(".manifest.tmp-0123456789abcdef.json")
    external = project / "derivatives/nro/anat/main/code/freesurfer/metadata.json"
    for path in (raw, excluded, sourcedata, third_party, manifest, abandoned, external):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    assert tuple(_source_candidates(project)) == (raw,)
    assert tuple(_candidates(project)) == (manifest,)


def test_dataset_migration_reports_inventory_phases(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    sidecar = project / "sub-01/func/sub-01_task-rest_bold.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}")
    registry = Registry.for_project("", bids_root=bids)
    phases = []

    migrate_dataset(
        registry,
        projects=("demo",),
        execute=False,
        version="1.2.3",
        progress=phases.append,
    )

    assert phases == [
        "Scanning source metadata",
        "Scanning source metadata (1 file)",
        "Scanning derivative metadata",
        "Checking work-item contracts",
    ]


def test_dataset_migration_rewrites_absolute_path_mapping_keys(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    source = project / "sub-01/anat/sub-01_T1w.nii.gz"
    manifest = project / "derivatives/nro/anat/main/sub-01/anat/manifest.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"gradient_unwarping": {str(source): {"applied": False}}}))
    registry = Registry.for_project("", bids_root=bids)

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert result.errors == ()
    assert json.loads(manifest.read_text())["gradient_unwarping"] == {
        "bids:raw:sub-01/anat/sub-01_T1w.nii.gz": {"applied": False}
    }


def test_dataset_migration_removes_obsolete_source_events_file(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    sidecar = project / "sub-01/func/sub-01_task-rest_bold.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "RepetitionTime": 1.5,
                "EventsFile": "/legacy/sub-01_task-rest_events.tsv",
            }
        )
    )
    registry = Registry.for_project("", bids_root=bids)

    preview = migrate_dataset(registry, projects=("demo",), execute=False, version="1.2.3")
    assert sidecar in preview.changed
    assert "EventsFile" in json.loads(sidecar.read_text())

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")
    assert result.errors == ()
    assert json.loads(sidecar.read_text()) == {"RepetitionTime": 1.5}


def test_dataset_contract_migration_replaces_sidecar_identity_with_semantics(
    tmp_path: Path,
) -> None:
    project = tmp_path / "demo"
    project.mkdir()
    (project / "dataset_description.json").write_text("{}")
    image = project / "sub-01/func/sub-01_task-rest_bold.nii.gz"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    sidecar = image.with_name("sub-01_task-rest_bold.json")
    sidecar.write_text(json.dumps({"RepetitionTime": 1.5, "EventsFile": "/legacy/events.tsv"}))
    artifact = {
        "module": "func",
        "inputs": [str(image), str(sidecar)],
        "processing": {},
    }
    scientific = {
        "module": "func",
        "project": "demo",
        "inputs": [{"source": str(image)}, {"source": str(sidecar)}],
        "processing": {},
    }

    migrated_artifact = _contract_dataset_view(artifact, project)
    migrated_scientific = _scientific_contract_dataset_view(scientific, project)

    assert migrated_artifact["inputs"] == [str(image)]
    assert migrated_artifact["contract_schema"] == current_contract_schema("func")
    assert migrated_artifact["processing"]["source_metadata"][0]["fields"] == {
        "RepetitionTime": 1.5
    }
    assert migrated_scientific["inputs"] == [{"source": str(image)}]
    assert "contract_schema" not in migrated_scientific
    assert (
        migrated_scientific["processing"]["source_metadata"]
        == migrated_artifact["processing"]["source_metadata"]
    )


def test_dataset_migration_preserves_completed_generation_and_state(tmp_path: Path) -> None:
    bids = tmp_path / "BIDS"
    project = bids / "demo"
    (project / "dataset_description.json").parent.mkdir(parents=True)
    (project / "dataset_description.json").write_text(
        json.dumps({"Name": "demo", "BIDSVersion": "1.10.0"})
    )
    image = project / "sub-01/func/sub-01_task-rest_bold.nii.gz"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    sidecar = image.with_name("sub-01_task-rest_bold.json")
    sidecar.write_text(json.dumps({"RepetitionTime": 1.5, "EventsFile": "/legacy/events.tsv"}))
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = (
        project
        / "derivatives/nro/func"
        / registered.directories["func"]
        / "sub-01/func/sub-01_task-rest_desc-preprocess_bold.nii.gz"
    )
    output.parent.mkdir(parents=True)
    output.write_bytes(b"derivative")
    key = work_item_key(
        "demo", "func", registered.lineage_fingerprints["func"], "01", {"task": "rest"}
    )
    spec = WorkItemSpec.create(
        key=key,
        module="func",
        project="demo",
        participant="01",
        entities={"task": "rest"},
        scope="run",
        module_lineage_id=registered.lineages["func"],
        config_fingerprint=workflow.configuration("func").scientific_fingerprint,
        directory_label=registered.directories["func"],
        runtime_config=registry.runtime_config_path(registered, "func"),
        command=(sys.executable, "-m", "nro.modules.func"),
        dependencies=(),
        input_paths=(image, sidecar),
        output_root=output.parent,
        output_prefix=output.name.removesuffix(".nii.gz"),
        expected_outputs=(output,),
        resource_class="large",
    )
    registry.create_request(
        registered=registered,
        target_module="func",
        selectors={},
        work_items=(spec,),
        terminal_work_item_keys=(spec.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("worker", resource_class="large")
    claim = registry.claim_ready_work_item("worker", ("large",))
    assert claim is not None
    completed = record_completion(
        registry,
        work_item_id=claim.work_item_id,
        attempt_id=claim.attempt_id,
        outputs=(output,),
    )
    registry.finish_attempt(claim.attempt_id, state="success")

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert result.errors == ()
    row = next(item for item in registry.work_item_rows() if item["id"] == claim.work_item_id)
    assert row["artifact_state"] == "fresh"
    assert row["current_generation"] == completed["generation"] == 1
    contract = json.loads(row["artifact_contract_json"])
    assert contract["inputs"] == [str(image.resolve())]
    assert contract["processing"]["source_metadata"][0]["fields"] == {"RepetitionTime": 1.5}
    with registry.connection() as database:
        completion = database.execute(
            "SELECT generation,artifact_fingerprint FROM completions WHERE work_item_id=?",
            (claim.work_item_id,),
        ).fetchone()
        inputs = database.execute(
            "SELECT path FROM artifacts WHERE work_item_id=? AND direction='input'",
            (claim.work_item_id,),
        ).fetchall()
    assert completion["generation"] == 1
    assert completion["artifact_fingerprint"] == row["artifact_fingerprint"]
    assert [item["path"] for item in inputs] == [str(image.resolve())]
    assert json.loads(sidecar.read_text()) == {"RepetitionTime": 1.5}

    changed_contract = json.loads(json.dumps(contract))
    changed_contract["processing"]["future_policy"] = True
    with registry.connection(write=True) as database:
        database.execute(
            """UPDATE work_items SET artifact_contract_json=?,artifact_fingerprint=?,
                      artifact_state='stale' WHERE id=?""",
            (
                json.dumps(changed_contract, sort_keys=True, separators=(",", ":")),
                fingerprint(changed_contract),
                claim.work_item_id,
            ),
        )

    repeated = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert repeated.errors == ()
    updated = next(item for item in registry.work_item_rows() if item["id"] == claim.work_item_id)
    with registry.connection() as database:
        historical = json.loads(
            database.execute(
                "SELECT artifact_contract_json FROM completions WHERE work_item_id=?",
                (claim.work_item_id,),
            ).fetchone()[0]
        )
    assert updated["artifact_state"] == "stale"
    assert updated["current_generation"] == 1
    assert json.loads(updated["artifact_contract_json"])["processing"]["future_policy"] is True
    assert "future_policy" not in historical["processing"]


def test_dataset_migration_rolls_back_files_after_validation_failure(
    tmp_path: Path, monkeypatch
) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    output = project / "derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"image")
    manifest = output.with_name("sub-01_desc-preprocessAnat_manifest.json")
    original = json.dumps({"outputs": {"t1w": str(output)}})
    manifest.write_text(original)
    registry = Registry.for_project("", bids_root=bids)
    monkeypatch.setattr(
        "nro.orchestration.provenance_migration.read_ownership_records",
        lambda *_args, **_kwargs: ([], [], ["invalid ownership"]),
    )

    with pytest.raises(ValueError, match="invalid ownership"):
        migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert manifest.read_text() == original
    assert not (project / "derivatives/nro/dataset_description.json").exists()
    journal_root = registry.paths.control / "shared/provenance-migrations"
    assert not list(journal_root.iterdir())


def test_dataset_migration_includes_registered_branch_outputs(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "BIDS"
    development = tmp_path / "NRO_DEV"
    work = tmp_path / "WORK"
    registry = Registry.for_project("", bids_root=bids)
    BranchStore(registry.paths.control).initialize()
    manifests = []
    for project in (
        bids / "demo",
        development / "dev/BIDS/demo",
    ):
        output = project / "derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"image")
        manifest = output.with_name("manifest.json")
        manifest.write_text(json.dumps({"output": str(output)}))
        manifests.append(manifest)

    result = migrate_dataset(
        registry,
        projects=("demo",),
        execute=True,
        version="1.2.3",
        site_values={"bids": bids, "work": work, "development": development},
    )

    assert all(manifest in result.changed for manifest in manifests)
    assert [json.loads(manifest.read_text())["output"] for manifest in manifests] == [
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz",
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz",
    ]


def test_dataset_migration_recovers_an_interrupted_transaction(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    output = project / "derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"image")
    manifest = output.with_name("manifest.json")
    original = json.dumps({"output": str(output)})
    manifest.write_text(original)
    registry = Registry.for_project("", bids_root=bids)
    journal = registry.paths.control / "shared/provenance-migrations/interrupted"
    backup = journal / "files/00000000"
    backup.parent.mkdir(parents=True)
    backup.write_text(original)
    manifest.write_text(json.dumps({"output": "bids::wrong"}))
    (journal / "journal.json").write_text(
        json.dumps(
            {
                "format": 1,
                "state": "applying",
                "projects": ["demo"],
                "files": [
                    {
                        "path": str(manifest.resolve()),
                        "backup": "files/00000000",
                        "existed": True,
                    }
                ],
            }
        )
    )

    preview = migrate_dataset(registry, projects=("demo",), execute=False, version="1.2.3")
    assert preview.errors == ()
    assert preview.recovery == (journal,)

    result = migrate_dataset(registry, projects=("demo",), execute=True, version="1.2.3")

    assert manifest in result.changed
    assert json.loads(manifest.read_text())["output"] == (
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    )
    assert not journal.exists()


def test_dataset_migration_recovery_skips_unchanged_file_in_read_only_directory(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    manifest = project / "sub-01/func/_excluded/sub-01_task-rest_bold.json"
    manifest.parent.mkdir(parents=True)
    original = json.dumps({"EventsFile": "/legacy/events.tsv"})
    manifest.write_text(original)
    registry = Registry.for_project("", bids_root=bids)
    journal = registry.paths.control / "shared/provenance-migrations/interrupted"
    backup = journal / "files/00000000"
    backup.parent.mkdir(parents=True)
    backup.write_text(original)
    (journal / "journal.json").write_text(
        json.dumps(
            {
                "format": 1,
                "state": "applying",
                "projects": ["demo"],
                "files": [
                    {
                        "path": str(manifest.resolve()),
                        "backup": "files/00000000",
                        "existed": True,
                    }
                ],
            }
        )
    )
    os.chmod(manifest.parent, 0o555)
    try:
        result = migrate_dataset(
            registry,
            projects=("demo",),
            execute=True,
            version="1.2.3",
        )
    finally:
        os.chmod(manifest.parent, 0o755)

    assert result.errors == ()
    assert manifest.read_text() == original
    assert not journal.exists()


def test_scheduler_routes_dataset_migration(tmp_path: Path) -> None:
    bids = tmp_path / "BIDS"
    work = tmp_path / "WORK"
    development = tmp_path / "NRO_DEV"
    control = tmp_path / "control"
    project = bids / "demo"
    manifest = project / "derivatives/nro/anat/main/sub-01/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"output": str(manifest.parent / "sub-01_T1w.nii.gz")}))
    registry = Registry.for_project("", bids_root=bids, registry_path=control)
    BranchStore(control).initialize()
    values = {
        "bids": str(bids),
        "work": str(work),
        "development": str(development),
    }

    preview = scheduler_service.dispatch(
        registry,
        {
            "operation": "dataset_migration",
            "projects": ["demo"],
            "execute": False,
            "version": "1.2.3",
        },
        values=values,
        message_id="migration-preview",
    )
    before = registry.work_item_rows()
    result = scheduler_service.dispatch(
        registry,
        {
            "operation": "dataset_migration",
            "projects": ["demo"],
            "execute": True,
            "version": "1.2.3",
        },
        values=values,
        message_id="migration-execute",
    )

    assert preview["errors"] == []
    assert preview["recovery"] == []
    assert str(manifest) in preview["changed"]
    assert result["errors"] == []
    assert json.loads(manifest.read_text())["output"] == (
        "bids::anat/main/sub-01/sub-01_T1w.nii.gz"
    )
    assert registry.work_item_rows() == before
