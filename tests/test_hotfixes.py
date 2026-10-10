from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from nro.definitions.store import ConfigStore, fingerprint
from nro.orchestration.artifact_records import file_record
from nro.orchestration.branch_admission import _workflow
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.compiled_request import export_workflow
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.hotfixes import apply, available
from nro.orchestration.hotfixes.v0278_private_portable_metadata import HOTFIX_ID
from nro.orchestration.hotfixes.v0286_msmall_runner_stages import (
    HOTFIX_ID as MSMALL_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0287_msmall_atlas_registration import (
    HOTFIX_ID as MSMALL_ATLAS_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0295_msmall_stage_isolation import (
    HOTFIX_ID as MSMALL_ISOLATION_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0301_msmall_source_masks import (
    HOTFIX_ID as MSMALL_MASK_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0302_msmall_brain_mask import (
    HOTFIX_ID as MSMALL_BRAIN_MASK_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0303_duplicate_ownership_receipts import (
    HOTFIX_ID as DUPLICATE_RECEIPT_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0303_obsolete_pycicada_receipts import (
    HOTFIX_ID as PYCICADA_RECEIPT_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0307_msmall_inverse_warp import (
    HOTFIX_ID as MSMALL_INVERSE_WARP_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0317_msmall_stage_inputs import (
    HOTFIX_ID as MSMALL_STAGE_INPUTS_HOTFIX_ID,
)
from nro.orchestration.hotfixes.v0319_personal_scheduler_adoption import (
    HOTFIX_ID as PERSONAL_SCHEDULER_HOTFIX_ID,
)
from nro.orchestration.ownership import ownership_record_fingerprint
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.runner_graph import RunnerGraph, Step


class _Registry:
    def __init__(self, bids_root: Path, database: sqlite3.Connection) -> None:
        self.paths = SimpleNamespace(bids_root=bids_root, control=bids_root.parent / ".nro")
        self.database = database

    @contextmanager
    def connection(self, *, write: bool = False):
        yield self.database
        if write:
            self.database.commit()


def _registry(tmp_path: Path) -> tuple[_Registry, sqlite3.Connection]:
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(
        """
        CREATE TABLE work_items (id INTEGER PRIMARY KEY, project TEXT NOT NULL);
        CREATE TABLE attempts (id INTEGER PRIMARY KEY, work_item_id INTEGER, state TEXT);
        CREATE TABLE artifacts (
            id INTEGER PRIMARY KEY,
            direction TEXT NOT NULL,
            path TEXT NOT NULL,
            size INTEGER,
            mtime_ns INTEGER,
            digest_algorithm TEXT,
            digest TEXT
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    return _Registry(tmp_path / "bids", database), database


def _portable_sidecar(tmp_path: Path) -> Path:
    path = tmp_path / "bids/demo/derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "BrainMask": "bids::anat/main/sub-01/anat/sub-01_desc-brain_mask.nii.gz",
                "Sources": ["bids:raw:sub-01/anat/sub-01_T1w.nii.gz"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_private_portable_metadata_hotfix_is_strict_and_idempotent(tmp_path: Path) -> None:
    registry, database = _registry(tmp_path)
    path = _portable_sidecar(tmp_path)
    current = file_record(path)
    database.execute(
        """INSERT INTO artifacts(
               direction,path,size,mtime_ns,digest_algorithm,digest
           ) VALUES ('private',?,?,?,?,?)""",
        (
            str(path.resolve()),
            current["size"] + 7,
            current["mtime_ns"],
            "sha256",
            hashlib.sha256(b"historical absolute metadata").hexdigest(),
        ),
    )

    preview = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=False)
    assert preview.paths == (path,)
    assert preview.records == 1
    assert not preview.applied

    result = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=True)
    assert result.records == 1
    stored = database.execute(
        "SELECT size,mtime_ns,digest FROM artifacts WHERE path=?", (str(path.resolve()),)
    ).fetchone()
    assert dict(stored) == {
        "size": current["size"],
        "mtime_ns": current["mtime_ns"],
        "digest": current["sha256"],
    }
    audit = database.execute(
        "SELECT value FROM metadata WHERE key=?", (f"hotfix:{HOTFIX_ID}:demo",)
    ).fetchone()
    assert json.loads(audit[0])["records"] == 1

    repeated = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=True)
    assert repeated.records == 0
    retained = database.execute(
        "SELECT value FROM metadata WHERE key=?", (f"hotfix:{HOTFIX_ID}:demo",)
    ).fetchone()
    assert json.loads(retained[0])["records"] == 1


def test_private_portable_metadata_hotfix_rejects_changed_timestamp(tmp_path: Path) -> None:
    registry, database = _registry(tmp_path)
    path = _portable_sidecar(tmp_path)
    current = file_record(path)
    database.execute(
        """INSERT INTO artifacts(
               direction,path,size,mtime_ns,digest_algorithm,digest
           ) VALUES ('private',?,?,?,?,?)""",
        (
            str(path.resolve()),
            current["size"] + 7,
            current["mtime_ns"] - 1,
            "sha256",
            "wrong",
        ),
    )

    report = apply(registry, identifier=HOTFIX_ID, projects=("demo",), execute=False)

    assert report.records == 0


def test_hotfix_registry_discovers_release_scoped_repairs() -> None:
    assert HOTFIX_ID in available()
    assert MSMALL_HOTFIX_ID in available()
    assert MSMALL_ATLAS_HOTFIX_ID in available()
    assert MSMALL_ISOLATION_HOTFIX_ID in available()
    assert MSMALL_MASK_HOTFIX_ID in available()
    assert MSMALL_BRAIN_MASK_HOTFIX_ID in available()
    assert DUPLICATE_RECEIPT_HOTFIX_ID in available()
    assert PYCICADA_RECEIPT_HOTFIX_ID in available()
    assert PERSONAL_SCHEDULER_HOTFIX_ID in available()


def test_personal_scheduler_hotfix_adopts_failed_legacy_work_item(tmp_path: Path) -> None:
    bids = tmp_path / "BIDS"
    (bids / "demo").mkdir(parents=True)
    registry = Registry.for_project("demo", bids_root=bids)
    registry.initialize()
    branches = BranchStore(registry.paths.control)
    topology = branches.initialize().topology
    owner = topology.records["main"].registry_id
    workflow = ConfigStore().resolve("main")
    legacy = registry.register_workflow(workflow)
    lineage_fingerprint = legacy.lineage_fingerprints["anat"]
    output = bids / "demo/derivatives/nro/anat/main/sub-01/anat/sub-01_T1w.nii.gz"
    key = work_item_key("demo", "anat", lineage_fingerprint, "01", {})
    spec = WorkItemSpec.create(
        key=key,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=legacy.lineages["anat"],
        config_fingerprint="science",
        directory_label=legacy.directories["anat"],
        runtime_config=registry.runtime_config_path(legacy, "anat"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(),
        output_root=output.parent,
        output_prefix="sub-01",
        expected_outputs=(output,),
        resource_class="small",
    )
    work_item_id = registry.register_work_items((spec,))[key]
    with registry.connection(write=True) as database:
        database.execute(
            """INSERT INTO attempts(
                   work_item_id,state,revision_fingerprint,memory_gb,
                   error_type,error_message,created_at
               ) VALUES (?, 'error', ?, 32, 'ContainerError', 'missing runtime', ?)""",
            (work_item_id, spec.revision_fingerprint, utcnow()),
        )

    preview = apply(
        registry,
        identifier=PERSONAL_SCHEDULER_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert preview.records == 1

    result = apply(
        registry,
        identifier=PERSONAL_SCHEDULER_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert result.records == 1
    with registry.connection(write=True) as database:
        row = database.execute(
            """SELECT item.module_lineage_id,lineage.lineage_fingerprint,
                      item.artifact_state,attempt.state,attempt.error_message
               FROM work_items item
               JOIN module_lineages lineage ON lineage.id=item.module_lineage_id
               JOIN attempts attempt ON attempt.work_item_id=item.id
               WHERE item.id=?""",
            (work_item_id,),
        ).fetchone()
        assert row["lineage_fingerprint"] == fingerprint(
            {"owner": owner, "lineage": lineage_fingerprint}
        )
        assert row["artifact_state"] == "missing"
        assert (row["state"], row["error_message"]) == ("error", "missing runtime")
        binding = database.execute(
            """SELECT logical_key,scientific_contract_json FROM branch_work_items
               WHERE registry_id=? AND work_item_id=?""",
            (owner, work_item_id),
        ).fetchone()
        assert binding["logical_key"] == key
        assert (
            database.execute(
                "SELECT 1 FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
                (owner, key),
            ).fetchone()
            is None
        )
        database.execute(
            "INSERT INTO compiled_revisions VALUES (?,?,1,?)",
            (owner, key, fingerprint(json.loads(binding["scientific_contract_json"]))),
        )

    repair_preview = apply(
        registry,
        identifier=PERSONAL_SCHEDULER_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert repair_preview.records == 1
    repair = apply(
        registry,
        identifier=PERSONAL_SCHEDULER_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert repair.records == 1
    with registry.connection(write=True) as database:
        assert (
            database.execute(
                "SELECT 1 FROM compiled_revisions WHERE registry_id=? AND logical_key=?",
                (owner, key),
            ).fetchone()
            is None
        )

        scientific = branches.registry("main")
        current = scientific.register_workflow(workflow)
        _revision, mapping = _workflow(
            database,
            export_workflow(scientific, current),
            owner,
            required_lineages={current.lineages["anat"]},
        )
        assert mapping[current.lineages["anat"]] == row["module_lineage_id"]
        updated = spec.evolve(module_lineage_id=row["module_lineage_id"])
        registry._upsert_work_item_graph_locked(
            database,
            ((updated, updated.as_record()),),
            now=utcnow(),
            owner_branch="main",
        )

    repeated = apply(
        registry,
        identifier=PERSONAL_SCHEDULER_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert repeated.records == 0


def _ownership_receipt(key: str, *, fingerprint: str, output: str = "manifest.json") -> dict:
    payload = {
        "owner": "nro",
        "record_version": 5,
        "project": "demo",
        "module": "func",
        "lineage_fingerprint": fingerprint,
        "participant": "01",
        "entities": {"run": "01", "task": "rest"},
        "directory_label": "main",
        "work_item_key": key,
        "artifact_contract": {"output": {"expected": [output]}},
    }
    payload["record_fingerprint"] = ownership_record_fingerprint(payload)
    return payload


def test_duplicate_ownership_receipt_hotfix_requires_canonical_sibling(tmp_path: Path) -> None:
    registry, _database = _registry(tmp_path)
    root = registry.paths.bids_root / "demo/derivatives/nro/func/main/.nro/work_items/func"
    root.mkdir(parents=True)
    lineage = "lineage"
    expected = work_item_key("demo", "func", lineage, "01", {"run": "01", "task": "rest"})
    obsolete_key = "func:" + "a" * 64
    obsolete = root / (obsolete_key.split(":", 1)[1] + ".json")
    obsolete.write_text(json.dumps(_ownership_receipt(obsolete_key, fingerprint=lineage)))

    missing = apply(
        registry,
        identifier=DUPLICATE_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert missing.records == 0

    canonical = root / (expected.split(":", 1)[1] + ".json")
    canonical.write_text(json.dumps(_ownership_receipt(expected, fingerprint=lineage)))
    preview = apply(
        registry,
        identifier=DUPLICATE_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert preview.paths == (obsolete,)
    assert obsolete.is_file()

    result = apply(
        registry,
        identifier=DUPLICATE_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert result.records == 1
    assert not obsolete.exists()
    assert canonical.is_file()
    assert (
        apply(
            registry,
            identifier=DUPLICATE_RECEIPT_HOTFIX_ID,
            projects=("demo",),
            execute=True,
        ).records
        == 0
    )


def test_duplicate_ownership_receipt_hotfix_requires_same_output_claim(tmp_path: Path) -> None:
    registry, _database = _registry(tmp_path)
    root = registry.paths.bids_root / "demo/derivatives/nro/func/main/.nro/work_items/func"
    root.mkdir(parents=True)
    lineage = "lineage"
    expected = work_item_key("demo", "func", lineage, "01", {"run": "01", "task": "rest"})
    obsolete_key = "func:" + "a" * 64
    obsolete = root / (obsolete_key.split(":", 1)[1] + ".json")
    obsolete.write_text(json.dumps(_ownership_receipt(obsolete_key, fingerprint=lineage)))
    canonical = root / (expected.split(":", 1)[1] + ".json")
    canonical.write_text(
        json.dumps(_ownership_receipt(expected, fingerprint=lineage, output="different.json"))
    )

    report = apply(
        registry,
        identifier=DUPLICATE_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )

    assert report.records == 0


def test_obsolete_pycicada_receipt_hotfix_is_strict_and_idempotent(tmp_path: Path) -> None:
    registry, _database = _registry(tmp_path)
    root = registry.paths.bids_root / "demo/derivatives/nro/func/main/.nro/work_items/func"
    root.mkdir(parents=True)
    receipt = root / "legacy.json"
    payload = {
        "owner": "nro",
        "record_version": 5,
        "execution": {
            "runtime_configuration": {
                "cicada_cmd": "nro-site:pycicada:.",
                "ica_classifier": "none",
            }
        },
    }
    payload["record_fingerprint"] = ownership_record_fingerprint(payload)
    receipt.write_text(json.dumps(payload))

    preview = apply(
        registry,
        identifier=PYCICADA_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert preview.paths == (receipt,)
    assert preview.records == 1
    assert (
        json.loads(receipt.read_text())["execution"]["runtime_configuration"]["cicada_cmd"]
        == "nro-site:pycicada:."
    )

    result = apply(
        registry,
        identifier=PYCICADA_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert result.records == 1
    repaired = json.loads(receipt.read_text())
    assert repaired["execution"]["runtime_configuration"] == {"ica_classifier": "none"}
    assert repaired["record_fingerprint"] == ownership_record_fingerprint(repaired)

    repeated = apply(
        registry,
        identifier=PYCICADA_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert repeated.records == 0


def test_obsolete_pycicada_receipt_hotfix_rejects_damaged_receipts(tmp_path: Path) -> None:
    registry, _database = _registry(tmp_path)
    root = registry.paths.bids_root / "demo/derivatives/nro/func/main/.nro/work_items/func"
    root.mkdir(parents=True)
    receipt = root / "damaged.json"
    receipt.write_text(
        json.dumps(
            {
                "owner": "nro",
                "record_version": 5,
                "record_fingerprint": "wrong",
                "execution": {"runtime_configuration": {"cicada_cmd": "nro-site:pycicada:."}},
            }
        )
    )

    report = apply(
        registry,
        identifier=PYCICADA_RECEIPT_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )

    assert report.records == 0


def test_msmall_runner_stage_hotfix_adopts_exact_legacy_checkpoints(tmp_path: Path) -> None:
    registry, database = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    work = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01/msmall"
    markers = work / "markers"
    markers.mkdir(parents=True)
    legacy_configuration = work / "configuration.sh"
    legacy_configuration.write_text("subject=sub-01\n")
    (markers / "prefreesurfer.complete").write_text("complete\n")
    (markers / "multirun_fix.complete").write_text("complete\n")
    variance = (
        work / "study/01_msmall/MNINonLinear/Results/rfMRI_REST_CONCAT/"
        "rfMRI_REST_CONCAT_Atlas_hp0.0_clean_vn_before_floor.dscalar.nii"
    )
    variance.parent.mkdir(parents=True)
    variance.write_text("variance")
    (variance.parent / "rfMRI_REST_CONCAT_Atlas_hp0.0_clean.dtseries.nii").write_text("timeseries")
    t1_dir = work / "study/01_msmall/T1w"
    t1_dir.mkdir(parents=True)
    (t1_dir / "T1w_acpc_dc_restore.nii.gz").write_text("T1w")
    (t1_dir / "T2w_acpc_dc_restore.nii.gz").write_text("T2w")
    event.write_text(
        json.dumps(
            {
                "version": 4,
                "module": "Anatomical Module",
                "signature": "work-item",
                "topology": [],
                "nodes": [
                    {
                        "name": "Estimate MSMAll Registration",
                        "outputs": [
                            str(work / "complete"),
                            str(work / "software_versions.txt"),
                            str(
                                work
                                / "study/01_msmall/MNINonLinear/Native/01_msmall.L.sphere.MSMAll.native.surf.gii"
                            ),
                            str(
                                work
                                / "study/01_msmall/MNINonLinear/Native/01_msmall.L.sphere.reg.reg_LR.native.surf.gii"
                            ),
                        ],
                    }
                ],
            }
        )
    )

    preview = apply(
        registry,
        identifier=MSMALL_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert preview.records == 1

    report = apply(
        registry,
        identifier=MSMALL_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert report.records == 1
    assert (work / "structural_configuration.sh").read_text() == "subject=sub-01\n"
    assert (work / "surface_configuration.sh").read_text() == "subject=sub-01\n"
    assert (work / "calibration_configuration.sh").read_text() == "subject=sub-01\n"
    assert (work / "stages/prefreesurfer.complete").is_file()
    assert not (work / "stages/multirun_fix.complete").exists()
    assert json.loads(event.read_text())["version"] == 2

    repeated = apply(
        registry,
        identifier=MSMALL_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )
    assert repeated.records == 0


def test_msmall_atlas_hotfix_preserves_freesurfer_and_invalidates_descendants(
    tmp_path: Path,
) -> None:
    registry, database = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    stages = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01/msmall/stages"
    prefree_marker = stages / "prefreesurfer.complete"
    atlas_marker = stages / "masked_atlas.complete"
    freesurfer_marker = stages / "freesurfer.complete"
    post_marker = stages / "postfreesurfer.complete"
    stages.mkdir(parents=True)
    for marker in (prefree_marker, atlas_marker, freesurfer_marker, post_marker):
        marker.write_text("complete\n")

    def node(identifier: str, name: str, inputs: list[Path], outputs: list[Path], deps: list[str]):
        return {
            "id": identifier,
            "name": name,
            "kind": "command",
            "inputs": [str(path) for path in inputs],
            "outputs": [str(path) for path in outputs],
            "dependencies": deps,
            "scientific_signature": f"old-{identifier}",
            "command_signature": f"command-{identifier}",
        }

    prefree = node("prefree", "MSMAll PreFreeSurfer", [], [prefree_marker], [])
    atlas = node(
        "atlas",
        "MSMAll Mask-Aware Atlas Registration",
        [prefree_marker],
        [atlas_marker],
        ["prefree"],
    )
    freesurfer = node(
        "freesurfer",
        "MSMAll FreeSurfer Reconstruction",
        [atlas_marker],
        [freesurfer_marker],
        ["atlas"],
    )
    post = node(
        "post",
        "MSMAll PostFreeSurfer",
        [atlas_marker, freesurfer_marker],
        [post_marker],
        ["atlas", "freesurfer"],
    )
    payload = {
        "version": 4,
        "module": "Anatomical Module",
        "signature": "work-item",
        "topology": [prefree, atlas, freesurfer, post],
        "nodes": [prefree, atlas, freesurfer, post],
    }
    event.write_text(json.dumps(payload), encoding="utf-8")

    report = apply(
        registry,
        identifier=MSMALL_ATLAS_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )

    assert report.records == 1
    repaired = json.loads(event.read_text())
    assert repaired["signature"] == f"hotfix:{MSMALL_ATLAS_HOTFIX_ID}"
    by_id = {item["id"]: item for item in repaired["nodes"]}
    assert set(by_id) == {"prefree", "freesurfer"}
    assert by_id["prefree"]["accept_relocated_signatures"] is True
    assert by_id["freesurfer"]["inputs"] == [str(prefree_marker)]
    assert by_id["freesurfer"]["dependencies"] == ["prefree"]
    assert by_id["freesurfer"]["accept_relocated_signatures"] is True

    current = RunnerGraph("Anatomical Module")
    current.add(
        Step.command_step(
            ("true",),
            id="prefree",
            name="MSMAll PreFreeSurfer",
            outputs=(prefree_marker,),
            parameters={"implementation": "current"},
        )
    )
    current.add(
        Step.command_step(
            ("true",),
            id="atlas",
            name="MSMAll Mask-Aware Atlas Registration",
            inputs=(prefree_marker,),
            outputs=(atlas_marker,),
            parameters={"affine_degrees_of_freedom": 7},
        )
    )
    current.add(
        Step.command_step(
            ("true",),
            id="freesurfer",
            name="MSMAll FreeSurfer Reconstruction",
            inputs=(prefree_marker,),
            outputs=(freesurfer_marker,),
            parameters={"implementation": "current"},
        )
    )
    current.add(
        Step.command_step(
            ("true",),
            id="post",
            name="MSMAll PostFreeSurfer",
            inputs=(atlas_marker, freesurfer_marker),
            outputs=(post_marker,),
            parameters={"implementation": "current"},
        )
    )
    current.freeze()
    assert current.step_contract_changes(event, signature="work-item") == {
        "atlas": "unrecorded_step",
        "post": "unrecorded_step",
    }

    repeated = apply(
        registry,
        identifier=MSMALL_ATLAS_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert repeated.records == 0


def test_msmall_stage_isolation_hotfix_preserves_non_msmall_steps(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    root = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01"

    def node(
        identifier: str,
        name: str,
        inputs: list[Path],
        outputs: list[Path],
        dependencies: list[str],
    ) -> dict[str, object]:
        return {
            "id": identifier,
            "name": name,
            "kind": "command",
            "inputs": [str(path) for path in inputs],
            "outputs": [str(path) for path in outputs],
            "dependencies": dependencies,
        }

    ordinary = node("ordinary", "Canonical anatomy", [], [root / "T1w.nii.gz"], [])
    prefree = node(
        "prefree",
        "MSMAll PreFreeSurfer",
        [root / "T1w.nii.gz"],
        [
            root / "msmall/study/01_msmall/T1w/T1w_acpc_dc_restore.nii.gz",
            root / "msmall/study/01_msmall/T1w/T2w_acpc_dc_restore.nii.gz",
            root / "msmall/stages/prefreesurfer.complete",
        ],
        ["ordinary"],
    )
    post = node(
        "post",
        "MSMAll PostFreeSurfer",
        [root / "msmall/stages/prefreesurfer.complete"],
        [root / "msmall/stages/postfreesurfer.complete"],
        ["prefree"],
    )
    publication = node(
        "publish",
        "Publish MSMAll sphere",
        [root / "msmall/stages/postfreesurfer.complete"],
        [tmp_path / "BIDS/demo/derivatives/nro/anat/main/sub-01/sphere.gii"],
        ["post"],
    )
    payload = {
        "version": 4,
        "module": "Anatomical Module",
        "signature": "work-item",
        "topology": [
            {key: item[key] for key in ("id", "kind", "inputs", "outputs", "dependencies")}
            for item in (ordinary, prefree, post, publication)
        ],
        "nodes": [ordinary, prefree, post, publication],
    }
    event.write_text(json.dumps(payload), encoding="utf-8")

    report = apply(
        registry,
        identifier=MSMALL_ISOLATION_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )

    assert report.records == 1
    repaired = json.loads(event.read_text())
    assert repaired["signature"] == f"hotfix:{MSMALL_ISOLATION_HOTFIX_ID}"
    assert [item["id"] for item in repaired["nodes"]] == ["ordinary"]

    repeated = apply(
        registry,
        identifier=MSMALL_ISOLATION_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert repeated.records == 0


def test_msmall_source_mask_hotfix_preserves_only_prefreesurfer_branch(
    tmp_path: Path,
) -> None:
    registry, _ = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    stages = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01/msmall/stages"

    def node(identifier: str, stage: str, dependencies: list[str]) -> dict[str, object]:
        return {
            "id": identifier,
            "name": stage,
            "kind": "command",
            "inputs": [],
            "outputs": [str(stages / f"{stage}.complete")],
            "dependencies": dependencies,
        }

    ordinary = node("ordinary", "ordinary", [])
    prefreesurfer = node("prefree", "prefreesurfer", ["ordinary"])
    atlas = node("atlas", "masked_atlas", ["prefree"])
    freesurfer = node("freesurfer", "freesurfer", ["prefree"])
    post = node("post", "postfreesurfer", ["atlas", "freesurfer"])
    payload = {
        "version": 4,
        "module": "Anatomical Module",
        "signature": "work-item",
        "topology": [ordinary, prefreesurfer, atlas, freesurfer, post],
        "nodes": [ordinary, prefreesurfer, atlas, freesurfer, post],
    }
    event.write_text(json.dumps(payload), encoding="utf-8")

    report = apply(
        registry,
        identifier=MSMALL_MASK_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )

    assert report.records == 1
    repaired = json.loads(event.read_text())
    assert repaired["signature"] == f"hotfix:{MSMALL_MASK_HOTFIX_ID}"
    assert [item["id"] for item in repaired["nodes"]] == ["ordinary", "prefree"]
    assert repaired["nodes"][1]["accept_relocated_signatures"] is True

    repeated = apply(
        registry,
        identifier=MSMALL_MASK_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert repeated.records == 0


def test_msmall_inverse_warp_hotfix_preserves_independent_stages(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    root = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01"
    stages = root / "msmall/stages"

    def node(
        identifier: str,
        name: str,
        inputs: list[Path],
        outputs: list[Path],
        dependencies: list[str],
    ) -> dict[str, object]:
        return {
            "id": identifier,
            "name": name,
            "kind": "command",
            "inputs": [str(path) for path in inputs],
            "outputs": [str(path) for path in outputs],
            "dependencies": dependencies,
        }

    ordinary = node("ordinary", "Ordinary Anatomy", [], [root / "ordinary.nii.gz"], [])
    prefree = node(
        "prefree",
        "MSMAll PreFreeSurfer",
        [],
        [stages / "prefreesurfer.complete"],
        [],
    )
    atlas = node(
        "atlas",
        "MSMAll Mask-Aware Atlas Registration",
        [stages / "prefreesurfer.complete"],
        [
            root / "msmall/structural/subject/MNINonLinear/registration_qc.json",
            root / "msmall/structural/subject/MNINonLinear/xfms/acpc_dc2standard.nii.gz",
            stages / "masked_atlas.complete",
        ],
        ["prefree"],
    )
    freesurfer = node(
        "freesurfer",
        "MSMAll FreeSurfer Reconstruction",
        [stages / "prefreesurfer.complete"],
        [
            root / "msmall/structural/subject/T1w/subject/surf/lh.white",
            stages / "freesurfer.complete",
        ],
        ["prefree"],
    )
    post = node(
        "post",
        "MSMAll PostFreeSurfer",
        [stages / "masked_atlas.complete", stages / "freesurfer.complete"],
        [stages / "postfreesurfer.complete"],
        ["atlas", "freesurfer"],
    )
    payload = {
        "version": 4,
        "module": "Anatomical Module",
        "signature": "old-work-item",
        "topology": [ordinary, prefree, atlas, freesurfer, post],
        "nodes": [ordinary, prefree, atlas, freesurfer, post],
    }
    event.write_text(json.dumps(payload), encoding="utf-8")

    report = apply(
        registry,
        identifier=MSMALL_INVERSE_WARP_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )

    assert report.records == 1
    repaired = json.loads(event.read_text())
    assert repaired["signature"] == f"hotfix:{MSMALL_INVERSE_WARP_HOTFIX_ID}"
    by_id = {item["id"]: item for item in repaired["nodes"]}
    assert set(by_id) == {"ordinary", "prefree", "freesurfer"}
    assert "accept_relocated_signatures" not in by_id["ordinary"]
    assert by_id["prefree"]["accept_relocated_signatures"] is True
    assert by_id["freesurfer"]["accept_relocated_signatures"] is True

    repeated = apply(
        registry,
        identifier=MSMALL_INVERSE_WARP_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert repeated.records == 0


def test_msmall_stage_inputs_hotfix_preserves_valid_expensive_stages(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)

    for participant, post_complete, expected in (
        ("01", True, {"ordinary", "prefree", "atlas", "freesurfer"}),
        ("02", False, {"ordinary", "prefree", "freesurfer"}),
    ):
        event = (
            registry.paths.control
            / f"branches/main/events/demo/anat/sub-{participant}/sub-{participant}/digest/runner-contract.json"
        )
        event.parent.mkdir(parents=True)
        root = tmp_path / f"WORK/demo/derivatives/nro/anat/main/sub-{participant}/msmall"
        stages = root / "stages"

        def node(identifier: str, stage: str | None, dependencies: list[str]) -> dict:
            outputs = [str(root / f"{identifier}.nii.gz")]
            if stage is not None:
                outputs.append(str(stages / f"{stage}.complete"))
            return {
                "id": identifier,
                "name": identifier,
                "kind": "command",
                "inputs": [],
                "outputs": outputs,
                "dependencies": dependencies,
            }

        ordinary = node("ordinary", None, [])
        prefree = node("prefree", "prefreesurfer", ["ordinary"])
        atlas = node("atlas", "masked_atlas", ["prefree"])
        freesurfer = node("freesurfer", "freesurfer", ["prefree"])
        post = node("post", "postfreesurfer", ["atlas", "freesurfer"])
        dedrift = node("dedrift", "dedrift", ["post"])
        nodes = [ordinary, prefree, atlas, freesurfer, post, dedrift]
        event.write_text(
            json.dumps(
                {
                    "version": 4,
                    "module": "Anatomical Module",
                    "signature": "old",
                    "topology": nodes,
                    "nodes": nodes,
                }
            ),
            encoding="utf-8",
        )
        if post_complete:
            stages.mkdir(parents=True)
            (stages / "postfreesurfer.complete").write_text("complete\n", encoding="utf-8")

        report = apply(
            registry,
            identifier=MSMALL_STAGE_INPUTS_HOTFIX_ID,
            projects=("demo",),
            execute=True,
        )

        assert report.records == 1
        repaired = json.loads(event.read_text(encoding="utf-8"))
        retained = {item["id"] for item in repaired["nodes"]}
        assert retained == expected
        for item in repaired["nodes"]:
            if item["id"] != "ordinary":
                assert item["accept_relocated_signatures"] is True


def test_msmall_brain_mask_hotfix_invalidates_only_msmall_nodes(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    (registry.paths.bids_root / "demo").mkdir(parents=True)
    event = (
        registry.paths.control
        / "branches/main/events/demo/anat/sub-01/sub-01/digest/runner-contract.json"
    )
    event.parent.mkdir(parents=True)
    root = tmp_path / "WORK/demo/derivatives/nro/anat/main/sub-01"
    stages = root / "msmall/stages"

    def node(
        identifier: str,
        name: str,
        inputs: list[Path],
        outputs: list[Path],
        dependencies: list[str],
    ) -> dict[str, object]:
        return {
            "id": identifier,
            "name": name,
            "kind": "command",
            "inputs": [str(path) for path in inputs],
            "outputs": [str(path) for path in outputs],
            "dependencies": dependencies,
        }

    ordinary = node("ordinary", "Ordinary Anatomy", [], [root / "ordinary.nii.gz"], [])
    configuration = node(
        "configuration",
        "Write MSMAll Structural Configuration",
        [],
        [root / "msmall/structural_configuration.sh"],
        [],
    )
    prefreesurfer = node(
        "prefree",
        "MSMAll PreFreeSurfer",
        [],
        [stages / "prefreesurfer.complete"],
        ["configuration"],
    )
    inferred_mask = node(
        "mask",
        "Restore MSMAll Source Brain Masks",
        [root / "anat/sub-01_desc-preproc_T1w.nii.gz"],
        [stages / "prefreesurfer_masks.complete"],
        ["prefree"],
    )
    atlas = node(
        "atlas",
        "MSMAll Mask-Aware Atlas Registration",
        [],
        [stages / "masked_atlas.complete"],
        ["mask"],
    )
    publication = node(
        "publication",
        "Publish MSMAll sphere",
        [],
        [root / "anat/sub-01_space-MSMAll_hemi-L_sphere.surf.gii"],
        ["atlas"],
    )
    nodes = [ordinary, configuration, prefreesurfer, inferred_mask, atlas, publication]
    event.write_text(
        json.dumps(
            {
                "version": 4,
                "module": "Anatomical Module",
                "signature": "work-item",
                "topology": nodes,
                "nodes": nodes,
            }
        ),
        encoding="utf-8",
    )

    report = apply(
        registry,
        identifier=MSMALL_BRAIN_MASK_HOTFIX_ID,
        projects=("demo",),
        execute=True,
    )

    assert report.records == 1
    repaired = json.loads(event.read_text())
    assert repaired["signature"] == f"hotfix:{MSMALL_BRAIN_MASK_HOTFIX_ID}"
    assert [item["id"] for item in repaired["nodes"]] == ["ordinary"]

    repeated = apply(
        registry,
        identifier=MSMALL_BRAIN_MASK_HOTFIX_ID,
        projects=("demo",),
        execute=False,
    )
    assert repeated.records == 0
