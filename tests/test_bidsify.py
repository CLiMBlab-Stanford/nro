"""Ingestion scheduling, approval, privacy boundaries, and offline conversion tests."""

import json
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from nro.bidsify.config import identifier, load_config
from nro.bidsify.images import extract, sanitize_image, secure_directory
from nro.bidsify.publication import approval_snapshot, inventory, publish
from nro.bidsify.review import compatible, validate_events
from nro.bidsify.stages import convert, run_stage
from nro.bidsify.store import IngestionStore
from nro.orchestration.registry import Registry


@pytest.fixture
def ingestion(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "bids")
    config = load_config()
    config["staging"] = str(tmp_path / "staging")
    config["concurrency"] = 2
    store = IngestionStore(registry)
    row = store.create(
        server="cni",
        remote_session="abc123",
        project="demo",
        participant="01",
        session="01",
        config=config,
    )
    return registry, store, row


def worker(registry, name="worker"):
    registry.register_worker(name, resource_class="large", memory_gb=32, slurm_job_id=None)


def save_decision(store, row):
    with store.review_session(row["id"]) as token:
        return store.update(row, expected_revision=row["revision"], review_token=token)


def prepared_session(registry, store, row, *, replace=False):
    root = Path(row["config"]["staging"]) / row["id"] / "bids/sub-01/ses-01/anat"
    root.mkdir(parents=True)
    (root / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"sanitized synthetic output")
    row.update(state="awaiting_approval", stage="publish", replace=replace)
    row["output_hashes"] = inventory(root.parent)
    row["approval"] = approval_snapshot(row, registry)
    row["state"] = "queued"
    row = save_decision(store, row)
    worker(registry)
    return store.claim("worker", 32)


def test_config_servers_come_from_store_and_credentials_are_references():
    config = load_config()
    assert config["servers"]["cni"]["host"] == "cni.example.org"
    assert config["servers"]["lucas"]["host"] == "lucas.example.org"
    assert config["servers"]["cni"]["credential_env"] == "TEST_CNI_KEY"
    assert config["cpus"] == 2


def test_missing_flywheel_points_to_managed_installer(monkeypatch):
    import sys

    from nro.bidsify.errors import BidsificationError
    from nro.bidsify.flywheel import FlywheelSource

    monkeypatch.setenv("TEST_FLYWHEEL_KEY", "test-placeholder")
    monkeypatch.setitem(sys.modules, "flywheel", None)
    with pytest.raises(BidsificationError) as error:
        FlywheelSource({"host": "example.org", "credential_env": "TEST_FLYWHEEL_KEY"})
    message = str(error.value)
    assert "./install --with-bidsify" in message
    assert "./install --maintain --with-bidsify" in message
    assert "From the nro repository" in message
    assert "Stop or drain the worker pool" in message
    assert "managed environment" in message
    assert "pip install" not in message
    assert "test-placeholder" not in message


@pytest.mark.parametrize("value", ["../bad", "/bad", "a/b", "", ".", "x\n"])
def test_staging_identifiers_reject_traversal(value):
    with pytest.raises(ValueError):
        identifier(value)


def test_ingestion_is_not_a_derivative_and_survives_repair(ingestion):
    registry, store, row = ingestion
    assert registry.instance_rows() == []
    registry.reinitialize()
    assert store.get(row["id"]) == row
    assert registry.instance_rows() == []


def test_duplicate_requests_and_concurrent_edits(ingestion):
    registry, store, row = ingestion
    again = store.create(
        server="cni",
        remote_session="abc123",
        project="demo",
        participant="01",
        session="01",
        config=row["config"],
    )
    assert again["id"] == row["id"]
    edited = deepcopy(row)
    edited["state"] = "needs_input"
    with store.review_session(row["id"]) as token:
        store.update(edited, expected_revision=row["revision"], review_token=token)
        with pytest.raises(ValueError, match="concurrently"):
            store.update(row, expected_revision=row["revision"], review_token=token)


def pending_request(store, config, *, remote="pending", session="ex123"):
    return store.create(
        server="cni", remote_session=remote, project="demo", session=session, config=config
    )


def test_pending_requests_deduplicate_by_source_not_missing_labels(ingestion):
    registry, store, original = ingestion
    first = pending_request(store, original["config"])
    assert pending_request(store, original["config"])["id"] == first["id"]
    second = pending_request(store, original["config"], remote="another")
    assert first["id"] != second["id"]
    first["participant"] = "t20"
    first = save_decision(store, first)
    assert pending_request(store, original["config"])["participant"] == "t20"
    second["participant"] = "t20"
    with pytest.raises(ValueError, match="owns this destination"):
        save_decision(store, second)
    assert store.get(second["id"])["participant"] is None


@pytest.mark.parametrize("field", ["participant", "session"])
def test_resolving_identity_requires_lease_and_valid_label(ingestion, field):
    registry, store, original = ingestion
    row = pending_request(store, original["config"], session=None)
    row[field] = "valid"
    with pytest.raises(ValueError, match="Review lease"):
        store.update(row, expected_revision=row["revision"], review_token="invalid")
    row[field] = "../invalid"
    with pytest.raises(ValueError, match="BIDS labels"):
        save_decision(store, row)
    assert store.get(row["id"])[field] is None
    row[field] = "valid"
    row = save_decision(store, row)
    row[field] = "changed"
    with pytest.raises(ValueError, match="resolved BIDS label"):
        save_decision(store, row)


def test_late_identity_checks_existing_bids_destination(ingestion):
    registry, store, original = ingestion
    row = pending_request(store, original["config"])
    (registry.paths.bids_root / "demo/sub-t20/ses-ex123").mkdir(parents=True)
    row["participant"] = "t20"
    with pytest.raises(ValueError, match="already exists"):
        save_decision(store, row)
    assert store.get(row["id"])["participant"] is None


def test_cancelled_unpublished_request_can_be_recreated(ingestion):
    registry, store, original = ingestion
    row = pending_request(store, original["config"])
    row["state"] = "cancelled"
    save_decision(store, row)
    replacement = pending_request(store, original["config"])
    assert replacement["id"] != row["id"]
    assert not replacement["replace"]


@pytest.mark.parametrize("state", ["queued", "needs_input", "cancelled", "published"])
@pytest.mark.parametrize("replace", [False, True])
def test_source_session_cannot_change_destination_project(ingestion, state, replace):
    registry, store, row = ingestion
    with registry.connection(write=True):
        row["state"] = state
        store.write_locked(row)
    with pytest.raises(ValueError, match="belongs to BIDS project"):
        store.create(
            server="cni",
            remote_session=row["remote_session"],
            project="different",
            participant=None,
            session=None,
            config=row["config"],
            replace=replace,
        )
    assert len(store.rows()) == 1


def test_rebidsify_retains_resolved_destination_and_sites_are_distinct(ingestion):
    registry, store, row = ingestion
    with registry.connection(write=True):
        row["state"] = "published"
        store.write_locked(row)
    repeated = store.create(
        server="cni",
        remote_session=row["remote_session"],
        project="demo",
        config=row["config"],
        replace=True,
    )
    assert (repeated["participant"], repeated["session"]) == ("01", "01")
    with pytest.raises(ValueError, match="different BIDS destination"):
        store.create(
            server="cni",
            remote_session=row["remote_session"],
            project="demo",
            participant="changed",
            config=row["config"],
            replace=True,
        )
    other = store.create(
        server="lucas", remote_session=row["remote_session"], project="other", config=row["config"]
    )
    assert other["id"] != repeated["id"]


def test_competing_projects_cannot_claim_same_source(ingestion):
    from concurrent.futures import ThreadPoolExecutor

    registry, store, row = ingestion

    def create(project):
        try:
            return store.create(
                server="cni", remote_session="race", project=project, config=row["config"]
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, ["one", "two"]))
    assert sum(result is not None for result in results) == 1
    assert sum(record["remote_session"] == "race" for record in store.rows()) == 1


def test_store_rejects_publication_state_without_identity(ingestion):
    registry, store, original = ingestion
    row = pending_request(store, original["config"])
    row.update(stage="publish", state="queued")
    with pytest.raises(ValueError, match="Resolve BIDS identity"):
        save_decision(store, row)
    assert store.get(row["id"])["stage"] == "inspect"


@pytest.mark.parametrize("field", ["participant", "session"])
def test_unknown_identity_cannot_organize_or_publish(ingestion, field):
    registry, store, row = ingestion
    row[field] = None
    row["stage"] = "convert"
    result = convert(row, registry)
    assert result["state"] == "needs_input"
    assert any(f"BIDS {field} label" in issue for issue in result["issues"])
    for operation in (approval_snapshot, publish):
        with pytest.raises(ValueError, match=f"BIDS {field} label"):
            operation(row, registry)
    assert not (registry.paths.bids_root / "demo").exists()


def test_image_preparation_does_not_require_identity(ingestion, monkeypatch):
    from nro.bidsify import stages

    registry, store, original = ingestion
    row = pending_request(store, original["config"], session=None)
    row.update(
        stage="prepare",
        acquisitions=[dict(id="a", datatype="anat", suffix="T1w", confirmed=True, entities={})],
    )
    prepared = []

    def prepare(record, item, helpers, source):
        assert record["participant"] is None and record["session"] is None
        path = helpers / item["id"]
        path.mkdir(parents=True)
        (path / ".prepared.json").write_text("{}")
        prepared.append(path)
        item.update(
            datatype="anat",
            suffix="T1w",
            confirmed=True,
            classification={
                "bids_guess": ["anat", "T1w"],
                "source": "dcm2niix",
                "reason": "accepted metadata-derived image type",
            },
        )
        return {"_shape": [2, 2, 2]}

    monkeypatch.setattr(stages, "prepare_image", prepare)
    result = run_stage(row, registry, source=object())
    assert prepared
    assert result["state"] == "needs_input" and result["stage"] == "convert"
    assert len(result["issues"]) == 2
    assert not (registry.paths.bids_root / "demo").exists()


def test_pending_identity_review_resumes_without_preparing_again(ingestion, monkeypatch):
    from nro.bidsify.review import wizard
    from nro.bidsify.status import render

    registry, store, original = ingestion
    row = pending_request(store, original["config"])
    row.update(
        state="needs_input",
        stage="convert",
        acquisitions=[
            dict(id="a", datatype="anat", suffix="T1w", confirmed=True, entities={}, metadata={})
        ],
    )
    row = save_decision(store, row)
    monkeypatch.setattr("builtins.input", lambda _: "")
    with store.review_session(row["id"]) as token:
        row = wizard(store, row, review_token=token)
    assert row["state"] == "needs_input" and row["participant"] is None
    assert "(pending)" in render([row]) and "None" not in render([row])
    assert "participant" in row["issues"][0]
    monkeypatch.setattr("builtins.input", lambda _: "t20")
    with store.review_session(row["id"]) as token:
        row = wizard(store, row, review_token=token)
    assert row["participant"] == "t20" and row["issues"] == []
    assert row["state"] == "queued" and row["stage"] == "convert"
    assert not (registry.paths.bids_root / "demo").exists()


def test_shared_limit_and_worker_supply(ingestion):
    registry, store, row = ingestion
    store.create(
        server="cni",
        remote_session="other",
        project="demo",
        participant="02",
        session="01",
        config=row["config"],
    )
    assert (
        len(
            registry.reserve_worker_submissions(
                request_id=None, resource_class="large", memory_gb=32
            )
        )
        == 2
    )
    assert (
        registry.reserve_worker_submissions(request_id=None, resource_class="large", memory_gb=32)
        == []
    )
    worker(registry, "a")
    worker(registry, "b")
    claimed = store.claim("a", 32)
    assert claimed
    assert registry.set_active_concurrency(1) == 2
    assert store.claim("b", 32) is None
    store.finish(claimed["id"], "a", state="needs_input")
    assert store.claim("b", 32)


def test_waiting_review_consumes_no_worker_supply(ingestion):
    registry, store, row = ingestion
    row["state"] = "needs_input"
    save_decision(store, row)
    assert registry.reserve_worker_submissions(request_id=None, resource_class="large") == []


def test_worker_finishes_stage_without_derivative_manifest(ingestion, monkeypatch):
    from nro.orchestration.execution import ExecutionResult
    from nro.orchestration.worker import Worker

    registry, store, row = ingestion
    worker(registry)
    row = store.claim("worker", 32)

    class Launcher:
        def run(self, envelope, **kwargs):
            assert "-m" in envelope.execution.command
            (store.root / f"{row['id']}.result").write_text(
                json.dumps({"state": "needs_input", "stage": "prepare", "issues": ["review"]})
            )
            kwargs["heartbeat"]()
            return ExecutionResult(0, False, False)

    runner = Worker(registry, resource_class="large", launcher=Launcher())
    runner.worker_id = "worker"
    runner._execute_ingestion(row)
    assert store.get(row["id"])["state"] == "needs_input"
    assert registry.instance_rows() == []


def test_publication_requires_exact_approval(ingestion):
    registry, store, row = ingestion
    row = prepared_session(registry, store, row)
    staged = (
        Path(row["config"]["staging"])
        / row["id"]
        / "bids/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
    )
    staged.write_bytes(b"changed after approval")
    with pytest.raises(ValueError, match="changed"):
        publish(row, registry)
    assert not (registry.paths.bids_root / "demo/sub-01/ses-01").exists()


def test_atomic_publication_and_receipt_recovery(ingestion):
    registry, store, row = ingestion
    row = prepared_session(registry, store, row)
    result = publish(row, registry)
    assert inventory(Path(result["published_path"])) == row["approval"]["outputs"]
    assert publish(row, registry) == result
    store.finish(row["id"], "worker", state="published", changes=result)
    registry.reinitialize()
    assert store.get(row["id"])["state"] == "published"


def test_debug_publication_is_separate_from_raw_scientific_inputs(ingestion, tmp_path):
    from nro.orchestration.branches import BranchPaths

    registry, production, original = ingestion
    paths = BranchPaths("dev", registry.paths.bids_root, tmp_path / "WORK", tmp_path / "NRO_DEV")
    store = IngestionStore(registry, branch_paths=paths)
    row = store.create(
        server=original["server"],
        remote_session=original["remote_session"],
        project="demo",
        participant="01",
        session="01",
        config=original["config"],
    )
    assert row["id"] != original["id"]
    assert [item["id"] for item in production.rows()] == [original["id"]]
    assert Path(row["config"]["staging"]).is_relative_to(store.root)
    staged = Path(row["config"]["staging"]) / row["id"] / "bids/sub-01/ses-01"
    staged.mkdir(parents=True)
    (staged / "test.txt").write_text("debug output")
    row["output_hashes"] = inventory(staged)
    row["approval"] = approval_snapshot(row, registry, branch_paths=paths)
    # Exercise publication ownership checks without admitting unscheduled work.
    row.update(state="running", stage="publish", worker="debug-test")
    with registry.connection(write=True):
        store.write_locked(row)
    with pytest.raises(ValueError, match="different branch"):
        publish(row, registry)
    result = publish(row, registry, branch_paths=paths)
    target = Path(result["published_path"])
    assert target == paths.output_project("demo") / "sub-01/ses-01"
    assert target.is_dir()
    assert not (paths.source_project("demo") / "sub-01").exists()
    assert paths.source_project("demo") == registry.paths.bids_root / "demo"
    assert publish(row, registry, branch_paths=paths) == result
    assert (store.root / "receipts" / f"{row['id']}.json").is_file()
    assert not (production.root / "receipts" / f"{row['id']}.json").exists()


def test_ingestion_rejects_symlinked_destination(ingestion, tmp_path):
    registry, store, row = ingestion
    row = prepared_session(registry, store, row)
    project = registry.paths.bids_root / "demo"
    project.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    project.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        publish(row, registry)
    assert not list(elsewhere.iterdir())


def test_replacement_preserves_unrelated_sessions(ingestion):
    registry, store, row = ingestion
    target = registry.paths.bids_root / "demo/sub-01/ses-01"
    target.mkdir(parents=True)
    (target / "old").write_text("old")
    other = target.parent / "ses-other"
    other.mkdir()
    (other / "keep").write_text("keep")
    row = prepared_session(registry, store, row, replace=True)
    publish(row, registry)
    assert not (target / "old").exists()
    assert (other / "keep").read_text() == "keep"


def test_extract_never_uses_archive_member_paths(tmp_path):
    source = tmp_path / "source"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../../private-name.dcm", b"example")
    extract(source, tmp_path / "dicoms")
    assert [p.name for p in (tmp_path / "dicoms").iterdir()] == ["00000000.dcm"]


def test_staging_symlinks_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        secure_directory(tmp_path / "link/nro")


def test_image_header_sanitization(tmp_path):
    image = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4))
    image.header["descrip"] = "identifying free text"
    image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"private"))
    nib.save(image, tmp_path / "raw.nii.gz")
    sanitize_image(tmp_path / "raw.nii.gz", tmp_path / "safe.nii.gz")
    output = nib.load(tmp_path / "safe.nii.gz")
    assert not output.header.extensions
    assert bytes(output.header["descrip"]).strip(b"\0") == b""


def test_events_validate_without_inventing_timing(tmp_path):
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\n-1\t2\tA\n")
    validate_events(path, duration=10)
    path.write_text("onset\tduration\n10\t2\n")
    with pytest.raises(ValueError):
        validate_events(path, duration=10)


def test_explicit_reference_compatibility():
    metadata = {
        "PhaseEncodingDirection": "j",
        "TotalReadoutTime": 0.03,
        "_shape": [2, 2, 2],
        "_affine": np.eye(4).tolist(),
    }
    a = {"metadata": metadata}
    b = {"metadata": {**metadata, "PhaseEncodingDirection": "j-"}}
    assert compatible(a, b, opposite=True)
    assert not compatible(a, b)
    b["metadata"]["TotalReadoutTime"] = 0.1
    assert not compatible(a, b, opposite=True)


def test_status_lists_pending_without_flywheel(ingestion, capsys):
    from nro.bin.status import main

    registry, store, row = ingestion
    main(["--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["instances"] == []
    assert report["bidsification"][0]["id"] == row["id"]
    main(["-p", "someoneelse", "--json"])
    assert json.loads(capsys.readouterr().out)["bidsification"] == []


def test_inspect_is_metadata_only(ingestion):
    registry, store, row = ingestion
    source = SimpleNamespace(inventory=lambda *_: [{"id": "file", "datatype": None}])
    result = run_stage(row, registry, source=source)
    assert result["state"] == "queued"
    assert result["stage"] == "prepare"
    assert not (registry.paths.bids_root / "demo").exists()


@pytest.mark.parametrize("pending", [False, True])
def test_real_dcm2bids_organizes_sanitized_nifti(ingestion, monkeypatch, pending):
    pytest.importorskip("dcm2bids")
    registry, store, row = ingestion
    row["config"]["validator"] = ["true"]
    if pending:
        row = pending_request(store, row["config"])
    row["stage"] = "convert"
    item = {
        "id": "a",
        "datatype": "func",
        "suffix": "bold",
        "confirmed": True,
        "entities": {"task": "rest", "run": "01"},
        "events": None,
        "sbref": None,
        "fieldmaps": [],
        "metadata": {"RepetitionTime": 2, "_shape": [2, 2, 2, 10]},
    }
    row["acquisitions"] = [item]
    helper = Path(row["config"]["staging"]) / row["id"] / "helpers/a"
    helper.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2, 10), np.float32), np.eye(4)), helper / "image.nii.gz"
    )
    (helper / "image.json").write_text(json.dumps({"RepetitionTime": 2, "SeriesNumber": 1}))
    if pending:
        from nro.bidsify.review import wizard

        # The request keeps prepared images while its participant is unknown.
        before = inventory(helper)
        assert convert(row, registry)["state"] == "needs_input"
        row["state"] = "needs_input"
        row = save_decision(store, row)
        answers = iter(["t20", "none", "none"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        with store.review_session(row["id"]) as token:
            row = wizard(store, row, review_token=token)
        assert inventory(helper) == before
    result = convert(row, registry)
    assert result["state"] == "awaiting_approval"
    assert any(name.endswith("_bold.nii.gz") for name in result["output_hashes"])
    assert not (registry.paths.bids_root / "demo").exists()
    if pending:
        row.update(result)
        row = save_decision(store, row)
        row["approval"] = approval_snapshot(row, registry)
        row.update(state="queued", stage="publish")
        row = save_decision(store, row)
        # The fixture's first request must not take this worker's claim.
        first = next(r for r in store.rows() if r["id"] != row["id"])
        first["state"] = "cancelled"
        save_decision(store, first)
        worker(registry)
        claimed = store.claim("worker", 32)
        assert claimed["id"] == row["id"]
        target = Path(publish(claimed, registry)["published_path"])
        assert target == registry.paths.bids_root / "demo/sub-t20/ses-ex123"
        assert inventory(target) == result["output_hashes"]
        assert not list(registry.paths.bids_root.rglob("*None*"))


def test_raw_anatomy_never_reaches_shared_staging(ingestion, monkeypatch, tmp_path):
    pytest.importorskip("pydicom")
    import pydicom

    from nro.bidsify import images

    registry, store, row = ingestion
    raw_root = tmp_path / "local-anatomy"
    monkeypatch.setattr(images, "temporary_raw", lambda _: raw_root)
    monkeypatch.setattr(
        pydicom, "dcmread", lambda *_args, **_kwargs: SimpleNamespace(Modality="MR")
    )
    item = {
        "id": "a",
        "acquisition": "a",
        "file_token": "hash",
        "bytes": 3,
        "datatype": "anat",
        "suffix": "T1w",
        "source_revision": {},
    }

    def download(_, path):
        assert path.is_relative_to(raw_root)
        path.write_bytes(b"raw")

    def execute(argv, **kwargs):
        if "-b" in argv:
            output = Path(argv[argv.index("-o") + 1])
            nib.save(
                nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)),
                output / "converted.nii.gz",
            )
            (output / "converted.json").write_text(
                json.dumps(
                    {
                        "BidsGuess": ["anat", "_T1w"],
                        "PatientName": "private",
                        "SeriesNumber": 1,
                        "MagneticFieldStrength": 7,
                        "NonlinearGradientCorrection": True,
                    }
                )
            )
        else:
            assert Path(argv[argv.index("-i") + 1]).is_relative_to(raw_root)
            nib.save(
                nib.Nifti1Image(np.full((2, 2, 2), 2, np.float32), np.eye(4)),
                argv[argv.index("-o") + 1],
            )

    monkeypatch.setattr(images, "command", execute)
    shared = tmp_path / "shared"
    result = images.prepare_image(row, item, shared, SimpleNamespace(download=download))
    assert "PatientName" not in result
    assert result["MagneticFieldStrength"] == 7
    assert result["NonlinearGradientCorrection"] is True
    assert not (raw_root / "a").exists()
    assert np.all(np.asanyarray(nib.load(shared / "a/image.nii.gz").dataobj) == 2)
    assert list(shared.rglob("*.dcm")) == []


def test_bids_guess_is_primary_classification_and_rules_refine_it():
    from nro.bidsify.images import classify

    rules = [{"pattern": "(?i)sbref", "datatype": "func", "suffix": "sbref"}]
    bold = classify({"BidsGuess": ["func", "_task-rest_bold"]}, rules)
    assert (bold["datatype"], bold["suffix"], bold["confirmed"]) == (
        "func",
        "bold",
        True,
    )
    sbref = classify(
        {"BidsGuess": ["func", "_task-rest_bold"], "SeriesDescription": "REST_SBRef"},
        rules,
    )
    assert (sbref["datatype"], sbref["suffix"]) == ("func", "sbref")
    assert sbref["classification"]["source"] == "protocol_rule"


@pytest.mark.parametrize(
    "metadata, reason",
    [
        ({"BidsGuess": ["discard", "_localizer"]}, "discard"),
        ({"BidsGuess": ["dwi", "_dwi"]}, "outside nro"),
        ({"SeriesDescription": "T1w"}, "did not provide"),
    ],
)
def test_unusable_bids_guess_proposes_confirmable_ignore(metadata, reason):
    from nro.bidsify.images import classify

    result = classify(
        metadata,
        [{"pattern": "(?i)t1w", "datatype": "anat", "suffix": "T1w"}],
    )
    assert (result["datatype"], result["suffix"]) == ("ignore", "ignore")
    assert result["confirmed"] is False
    assert reason in result["classification"]["reason"]


def test_failed_strip_removes_raw_local_files(ingestion, monkeypatch, tmp_path):
    pytest.importorskip("pydicom")
    from nro.bidsify import images

    registry, store, row = ingestion
    root = tmp_path / "raw"
    monkeypatch.setattr(images, "temporary_raw", lambda _: root)
    item = {
        "id": "a",
        "acquisition": "a",
        "file_token": "hash",
        "bytes": 3,
        "datatype": "anat",
        "suffix": "T1w",
        "source_revision": {},
    }

    def download(_, path):
        path.write_bytes(b"raw")
        raise ValueError("download interrupted")

    with pytest.raises(ValueError):
        images.prepare_image(row, item, tmp_path / "shared", SimpleNamespace(download=download))
    assert not (root / "a").exists()
    assert not (tmp_path / "shared/a/image.nii.gz").exists()


def test_flywheel_revision_pinned_and_labels_not_saved(tmp_path):
    from nro.bidsify.flywheel import FlywheelSource

    file = SimpleNamespace(
        name="private-person-name.dicom.zip",
        type="dicom",
        size=3,
        version=1,
        hash="opaque",
        modified="2026-01-01",
        download=lambda p: Path(p).write_bytes(b"abc"),
    )
    acquisition = SimpleNamespace(id="abc", label="T1w private label", files=[file])
    client = SimpleNamespace(
        get_session=lambda _: SimpleNamespace(
            acquisitions=SimpleNamespace(iter=lambda: iter([acquisition]))
        ),
        get_acquisition=lambda _: acquisition,
    )
    source = FlywheelSource({}, client=client)
    items = source.inventory("session")
    assert "private" not in json.dumps(items)
    source.download(items[0], tmp_path / "download")
    file.version = 2
    with pytest.raises(ValueError, match="source changed"):
        source.download(items[0], tmp_path / "other")
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("answer", ["0", "-1", "3", "", "cat"])
def test_invalid_wizard_indices(answer):
    from nro.bidsify.review import choose_indices

    with pytest.raises(ValueError):
        choose_indices(answer, 2)


def test_wizard_multiple_event_files_resume(ingestion, monkeypatch, tmp_path):
    from nro.bidsify.review import wizard

    registry, store, row = ingestion
    event_path = tmp_path / "events.tsv"
    event_path.write_text("onset\tduration\ttrial_type\n0\t1\tA\n")
    row.update(
        stage="convert",
        state="needs_input",
        acquisitions=[
            {
                "id": f"a{i}",
                "datatype": "func",
                "suffix": "bold",
                "entities": {},
                "events": None,
                "confirmed": True,
                "classification": {
                    "bids_guess": ["func", "bold"],
                    "source": "dcm2niix",
                    "reason": "accepted metadata-derived image type",
                },
                "metadata": {"_shape": [2, 2, 2, 10], "RepetitionTime": 1.0},
            }
            for i in (1, 2)
        ],
    )
    row = save_decision(store, row)
    answers = iter(
        [
            "task=language run=01",
            str(event_path),
            "y",
            "",
            "",
            "task=language run=02",
            str(event_path),
            "y",
            "",
            "",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    with store.review_session(row["id"]) as token:
        result = wizard(store, row, review_token=token)
    assert result["state"] == "queued" and result["stage"] == "convert"
    assert all(
        Path(a["events"]).read_text() == event_path.read_text() for a in result["acquisitions"]
    )


def test_wizard_override_of_proposed_ignore_requeues_preparation(ingestion, monkeypatch):
    from nro.bidsify.review import wizard

    registry, store, row = ingestion
    row.update(
        stage="convert",
        state="needs_input",
        acquisitions=[
            {
                "id": "a",
                "datatype": "ignore",
                "suffix": "ignore",
                "entities": {},
                "events": None,
                "confirmed": False,
                "metadata": {},
                "classification": {
                    "bids_guess": None,
                    "source": "unmatched",
                    "reason": "dcm2niix did not provide a usable BidsGuess",
                },
            }
        ],
    )
    row = save_decision(store, row)
    answers = iter(["anat/T1w", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    with store.review_session(row["id"]) as token:
        result = wizard(store, row, review_token=token)
    item = result["acquisitions"][0]
    assert result["state"] == "queued" and result["stage"] == "prepare"
    assert item["classification_override"] == ["anat", "T1w"]
    assert item["entities_confirmed"] is True


def test_replacement_receipt_cleans_interrupted_backup(ingestion, monkeypatch):
    import nro.bidsify.publication as publication

    registry, store, row = ingestion
    target = registry.paths.bids_root / "demo/sub-01/ses-01"
    target.mkdir(parents=True)
    (target / "old.nii").write_bytes(b"old")
    row = prepared_session(registry, store, row, replace=True)
    real_remove = publication.shutil.rmtree

    def interrupted_remove(path, *args, **kwargs):
        if path.name == f".nro-publish-{row['id']}":
            raise OSError("interrupted cleanup")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(publication.shutil, "rmtree", interrupted_remove)
    with pytest.raises(OSError):
        publish(row, registry)
    temporary = target.parent / f".nro-publish-{row['id']}"
    assert temporary.exists()
    monkeypatch.setattr(publication.shutil, "rmtree", real_remove)
    publish(row, registry)
    assert not temporary.exists()
    assert inventory(target) == row["approval"]["outputs"]


def test_pending_ingestion_allocations_can_be_stopped_by_owner(ingestion):
    import getpass

    registry, store, row = ingestion
    reservations = registry.reserve_worker_submissions(
        request_id=None, resource_class="large", memory_gb=32
    )
    registry.update_submission(reservations[0][0], state="submitted", slurm_job_id="1234")
    assert registry.request_worker_shutdown(user_name="another-user")["job_ids"] == ()
    assert registry.request_worker_shutdown(user_name=getpass.getuser())["job_ids"] == ("1234",)


def test_repair_releases_confirmed_stopped_ingestion(ingestion, monkeypatch):
    from nro.orchestration import worker_control

    registry, store, row = ingestion
    worker(registry)
    store.claim("worker", 32)
    monkeypatch.setattr(worker_control, "wait_for_worker_shutdown", lambda _: None)
    worker_control.stop_worker_pool_for_repair(registry)
    assert store.get(row["id"])["state"] == "interrupted"
    registry.reinitialize()
    assert store.get(row["id"])["state"] == "interrupted"


def test_worker_cancellation_releases_ingestion(ingestion):
    from nro.orchestration.execution import ExecutionResult
    from nro.orchestration.worker import Worker

    registry, store, row = ingestion
    worker(registry)
    claimed = store.claim("worker", 32)
    launcher = SimpleNamespace(run=lambda *_args, **_kwargs: ExecutionResult(-15, True, False))
    runner = Worker(registry, resource_class="large", launcher=launcher)
    runner.worker_id = "worker"
    runner._execute_ingestion(claimed)
    assert store.get(row["id"])["state"] == "interrupted"
    assert store.summary()[0] == 0


def test_review_leases_only_block_the_active_session(ingestion):
    from nro.bidsify.store import ReviewBusyError

    registry, store, first = ingestion
    second = store.create(
        server="cni",
        remote_session="second",
        project="demo",
        participant="02",
        session="01",
        config=first["config"],
    )
    with store.review_session(first["id"]):
        with pytest.raises(ReviewBusyError, match="being reviewed by"):
            IngestionStore(registry).acquire_review(first["id"])
        with IngestionStore(registry).review_session(second["id"]):
            assert store.summary()[1] == 0
        assert store.summary()[1] == 1
        duplicate = store.create(
            server="cni",
            remote_session="abc123",
            project="demo",
            participant="01",
            session="01",
            config=first["config"],
        )
        assert duplicate["id"] == first["id"]
        worker(registry)
        assert store.claim("worker", 32)["id"] == second["id"]
    assert store.claim("worker", 32) is None
    worker(registry, "other-worker")
    assert store.claim("other-worker", 32)["id"] == first["id"]
    with pytest.raises(ReviewBusyError, match="running"):
        store.acquire_review(first["id"])


def test_expired_review_cannot_write_renew_or_release_successor(ingestion, monkeypatch):
    import nro.bidsify.store as storage

    registry, store, row = ingestion
    now = [1000.0]
    monkeypatch.setattr(storage.time, "time", lambda: now[0])
    old = store.acquire_review(row["id"], seconds=10)
    now[0] += 11
    with pytest.raises(ValueError, match="expired"):
        store.renew_review(row["id"], old)
    new = store.acquire_review(row["id"], seconds=10)
    row["acquisitions"] = [{"id": "a"}]
    with pytest.raises(ValueError, match="ownership changed"):
        store.update(
            row,
            expected_revision=row["revision"],
            review_token=old,
            event_files={"a": "unaccepted"},
        )
    assert not (Path(row["config"]["staging"]) / row["id"]).exists()
    store.release_review(row["id"], old)
    assert store._review(row["id"])["token"] == new
    store.release_review(row["id"], new)


def test_review_heartbeat_preserves_decision_revision(ingestion, monkeypatch):
    import threading

    registry, store, row = ingestion
    renewed = threading.Event()
    original = store.renew_review

    def renew(*args, **kwargs):
        original(*args, **kwargs)
        renewed.set()

    monkeypatch.setattr(store, "renew_review", renew)
    with store.review_session(row["id"], seconds=1):
        assert renewed.wait(3)
        assert store.get(row["id"])["revision"] == row["revision"]
    assert store._review(row["id"]) is None


def test_event_updates_are_revision_checked_and_preserve_previous_snapshot(ingestion, monkeypatch):
    registry, store, row = ingestion
    row["acquisitions"] = [{"id": "a", "events": None}]
    with store.review_session(row["id"]) as token:
        saved = store.update(
            row,
            expected_revision=row["revision"],
            review_token=token,
            event_files={"a": "accepted"},
        )
        event_path = Path(saved["acquisitions"][0]["events"])
        assert event_path.read_text() == "accepted"
        with pytest.raises(ValueError, match="concurrently"):
            store.update(
                row,
                expected_revision=row["revision"],
                review_token=token,
                event_files={"a": "rejected"},
            )
        assert list(event_path.parent.iterdir()) == [event_path]

        def fail(_):
            raise OSError("record write failed")

        monkeypatch.setattr(store, "write_locked", fail)
        with pytest.raises(OSError, match="record write failed"):
            store.update(
                saved,
                expected_revision=saved["revision"],
                review_token=token,
                event_files={"a": "new snapshot"},
            )
        assert store.get(row["id"])["acquisitions"][0]["events"] == str(event_path)
        assert event_path.read_text() == "accepted"


@pytest.mark.parametrize("answer", ["skip", "q", "interrupt"])
def test_interactive_exit_releases_only_current_session(ingestion, monkeypatch, answer):
    from nro.bin.bidsify import advance

    registry, store, row = ingestion
    row.update(
        state="needs_input",
        stage="convert",
        acquisitions=[
            {
                "id": "a",
                "datatype": "ignore",
                "suffix": "ignore",
                "confirmed": False,
                "entities": {},
                "events": None,
                "metadata": {},
                "classification": {
                    "bids_guess": ["discard", "localizer"],
                    "source": "dcm2niix",
                    "reason": "dcm2niix classified the acquisition as discard",
                },
            }
        ],
    )
    row = save_decision(store, row)

    def prompt(_):
        assert store._review_active(row["id"])
        if answer == "interrupt":
            raise KeyboardInterrupt
        return answer

    monkeypatch.setattr("builtins.input", prompt)
    if answer == "skip":
        advance(store, row)
    else:
        with pytest.raises(EOFError if answer == "q" else KeyboardInterrupt):
            advance(store, row)
    assert store._review(row["id"]) is None
    assert store.get(row["id"]) == row


def test_multiple_selected_sessions_are_leased_in_turn(ingestion, monkeypatch):
    import nro.bin.bidsify as cli

    registry, store, first = ingestion
    second = store.create(
        server="cni",
        remote_session="second",
        project="demo",
        participant="02",
        session="01",
        config=first["config"],
    )
    rows = []
    for row in (first, second):
        row["state"] = "needs_input"
        rows.append(save_decision(store, row))
    visited = []

    def review(store, row, **kwargs):
        visited.append(row["id"])
        assert store._review_active(row["id"])
        other = next(r for r in rows if r["id"] != row["id"])
        assert not store._review_active(other["id"])
        return row

    monkeypatch.setattr(cli, "_advance", review)
    for row in rows:
        cli.advance(store, row)
        assert store._review(row["id"]) is None
    assert visited == [first["id"], second["id"]]


def test_occupied_review_is_skipped_without_starting_wizard(ingestion, monkeypatch, capsys):
    import nro.bin.bidsify as cli

    registry, store, row = ingestion
    row["state"] = "needs_input"
    row = save_decision(store, row)
    monkeypatch.setattr(
        cli, "_advance", lambda *_args, **_kwargs: pytest.fail("Occupied review started")
    )
    with store.review_session(row["id"]) as token:
        cli.advance(store, row)
        assert store._review(row["id"])["token"] == token
    assert "being reviewed by" in capsys.readouterr().out
