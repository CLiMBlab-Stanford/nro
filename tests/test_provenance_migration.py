from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nro.orchestration import scheduler_service
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.provenance_migration import migrate_public_provenance
from nro.orchestration.registry import Registry


def test_public_provenance_migration_previews_then_rewrites_metadata(tmp_path: Path) -> None:
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

    preview = migrate_public_provenance(
        registry, projects=("demo",), execute=False, version="1.2.3"
    )
    assert preview.errors == ()
    assert set(preview.changed) == {
        manifest,
        project / "derivatives/nro/dataset_description.json",
    }
    assert str(output) in manifest.read_text()

    result = migrate_public_provenance(
        registry, projects=("demo",), execute=True, version="1.2.3"
    )
    assert result.errors == ()
    assert json.loads(manifest.read_text())["outputs"]["t1w"] == (
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    )
    assert manifest.stat().st_mtime_ns == original_mtime
    description = json.loads(
        (project / "derivatives/nro/dataset_description.json").read_text()
    )
    assert description["DatasetLinks"] == {"raw": "../.."}

    repeated = migrate_public_provenance(
        registry, projects=("demo",), execute=False, version="1.2.3"
    )
    assert repeated.changed == ()


def test_public_provenance_migration_rolls_back_files_after_validation_failure(
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
        migrate_public_provenance(
            registry, projects=("demo",), execute=True, version="1.2.3"
        )

    assert manifest.read_text() == original
    assert not (project / "derivatives/nro/dataset_description.json").exists()
    journal_root = registry.paths.control / "shared/provenance-migrations"
    assert not list(journal_root.iterdir())


def test_public_provenance_migration_includes_registered_branch_outputs(
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

    result = migrate_public_provenance(
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


def test_public_provenance_migration_recovers_an_interrupted_transaction(
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

    result = migrate_public_provenance(
        registry, projects=("demo",), execute=True, version="1.2.3"
    )

    assert manifest in result.changed
    assert json.loads(manifest.read_text())["output"] == (
        "bids::anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    )
    assert not journal.exists()


def test_scheduler_routes_public_provenance_migration(tmp_path: Path) -> None:
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
            "operation": "provenance_migration",
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
            "operation": "provenance_migration",
            "projects": ["demo"],
            "execute": True,
            "version": "1.2.3",
        },
        values=values,
        message_id="migration-execute",
    )

    assert preview["errors"] == []
    assert str(manifest) in preview["changed"]
    assert result["errors"] == []
    assert json.loads(manifest.read_text())["output"] == (
        "bids::anat/main/sub-01/sub-01_T1w.nii.gz"
    )
    assert registry.work_item_rows() == before
