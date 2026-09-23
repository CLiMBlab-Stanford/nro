from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.configuration.store import ConfigStore
from nro.orchestration.artifact_ownership import rows_with_public_receipts
from nro.orchestration.garbage_collection import garbage_paths
from nro.orchestration.planner import build_subject_work_items
from nro.orchestration.registry import Registry


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _registry(tmp_path: Path) -> tuple[Path, Path, Registry]:
    bids = tmp_path / "bids"
    work = tmp_path / "work"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    specs = build_subject_work_items(
        project="demo",
        participant="01",
        module="clean",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    registry.register_work_items(specs)
    return bids, work, registry


def _selection(**values) -> dict:
    return {
        "projects": values.get("projects", []),
        "participants": values.get("participants", []),
        "modules": values.get("modules", []),
        "workflows": values.get("workflows", []),
        "lineages": values.get("lineages", []),
        "selectors": values.get("selectors", {}),
    }


def test_gc_preserves_registered_files_and_collects_unclaimed_siblings(tmp_path: Path) -> None:
    bids, work, registry = _registry(tmp_path)
    rows = registry.work_item_rows()
    func = next(row for row in rows if row["module"] == "func")
    entities = json.loads(func["entities_json"])
    owned = _write(
        Path(func["output_root"])
        / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    garbage = _write(Path(func["output_root"]) / "func" / "sub-01_desc-pollutant.txt")

    public, private = garbage_paths(
        bids / "demo",
        rows,
        rows,
        selection=_selection(
            participants=["01"], modules=["func"], selectors={"run": [entities["run"]]}
        ),
        control=registry.paths.control,
        work_root=work,
    )

    assert owned not in public
    # The pollutant lacks the requested run entity, so narrow run selection
    # conservatively leaves it alone.
    assert garbage not in public
    public, private = garbage_paths(
        bids / "demo",
        rows,
        rows,
        selection=_selection(participants=["01"], modules=["func"]),
        control=registry.paths.control,
        work_root=work,
    )
    assert public == (garbage,)
    assert private == ()


def test_gc_treats_anat_and_registered_work_directories_as_owned_trees(tmp_path: Path) -> None:
    bids, work, registry = _registry(tmp_path)
    rows = registry.work_item_rows()
    anat = next(row for row in rows if row["module"] == "anat")
    func = next(row for row in rows if row["module"] == "func")
    inside_anat = _write(Path(anat["output_root"]) / "unlisted-tool-output.dat")
    external_template = tmp_path / "templates" / "fsaverage"
    external_template.mkdir(parents=True)
    inside_anat_symlink = Path(anat["output_root"]) / "fsaverage"
    inside_anat_symlink.symlink_to(external_template, target_is_directory=True)
    func_work = _write(
        work
        / "demo"
        / "derivatives"
        / "nro"
        / "func"
        / func["directory_label"]
        / "sub-01"
        / "func"
        / f"{func['output_prefix']}_bold"
        / "dynamic-scratch.dat"
    )
    garbage_work = _write(
        work / "demo" / "derivatives" / "nro" / "func" / "unknown-lineage" / "sub-01" / "orphan.dat"
    )

    public, private = garbage_paths(
        bids / "demo",
        rows,
        rows,
        selection=_selection(participants=["01"]),
        control=registry.paths.control,
        work_root=work,
    )

    assert inside_anat not in public
    assert inside_anat_symlink not in public
    assert func_work not in private
    assert private == (garbage_work,)


def test_public_receipts_supply_claims_missing_from_registry(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "bids" / "demo"
    receipt = {
        "work_item_key": "demo:clean:abc",
        "module": "clean",
        "project": "demo",
        "participant": "01",
        "entities": {"space": "fsnative", "smoothing": "2"},
        "directory_label": "historical-abc",
        "artifact_contract": {
            "output": {
                "root": str(project / "derivatives/nro/clean/historical-abc/sub-01"),
                "prefix": "sub-01_space-fsnative_smoothing-2mm",
            }
        },
    }
    monkeypatch.setattr(
        "nro.orchestration.ownership.read_ownership_records",
        lambda _root, _projects: ([], [(receipt, tmp_path / "receipt.json")], []),
    )

    rows = rows_with_public_receipts(project, [])

    assert len(rows) == 1
    assert rows[0]["directory_label"] == "historical-abc"
    assert rows[0]["output_prefix"] == "sub-01_space-fsnative_smoothing-2mm"


def test_gc_refuses_invalid_public_ownership_records(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "bids" / "demo"
    monkeypatch.setattr(
        "nro.orchestration.ownership.read_ownership_records",
        lambda _root, _projects: ([], [], ["broken receipt"]),
    )

    with pytest.raises(ValueError, match="invalid ownership records"):
        rows_with_public_receipts(project, [])
