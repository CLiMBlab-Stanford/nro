"""Scientific validators exchange bounded reports with a catalog-independent publisher."""

import json
import sys

import pytest

from nro.configuration.store import ConfigStore, fingerprint
from nro.orchestration import manifests
from nro.orchestration.assessment import (
    AssessmentConflict,
    AssessmentReport,
    AssessmentSnapshot,
    apply_assessment,
    capture_assessment,
)
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.registry import Registry


@pytest.fixture
def graph(tmp_path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    processing = module_descriptor("anat").processing_contract()
    specs = []
    for name, parents in [("root", ()), ("leaf", ("root",)), ("unrelated", ())]:
        output = tmp_path / "outputs" / name / "result_manifest.json"
        output.parent.mkdir(parents=True)
        output.write_text(
            json.dumps(
                {"complete": True, "output_metadata_contract": processing["output_metadata"]}
            )
        )
        specs.append(
            InstanceSpec.create(
                key=name,
                module="anat",
                project="demo",
                participant="01",
                entities={},
                scope="subject",
                configuration_lineage_id=registered.lineages["preprocessing"],
                directory_label="main",
                config_fingerprint=workflow.configuration("preprocessing").scientific_fingerprint,
                runtime_config=registry.runtime_config_path(registered, "preprocessing"),
                command=(sys.executable, "-c", "pass"),
                input_paths=(),
                expected_outputs=(output,),
                output_root=output.parent,
                output_prefix=None,
                dependencies=parents,
                resource_class="large",
                processing=processing,
            )
        )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        instances=specs,
        terminal_instance_keys=("leaf", "unrelated"),
        concurrency=5,
        partition=None,
    )
    ids = registry.instance_ids(tuple(spec.key for spec in specs))
    return registry, specs, ids


def test_snapshot_is_consistent_detached_and_includes_ancestors(graph, monkeypatch):
    registry, _, ids = graph
    snapshot = capture_assessment(registry, instance_ids=[ids["leaf"]])
    assert {row["id"] for row in snapshot.instances} == {ids["root"], ids["leaf"]}
    decoded = AssessmentSnapshot.from_dict(snapshot.as_dict())
    assert decoded == snapshot
    detached = snapshot.as_dict()
    detached["instances"][0]["module"] = "changed"
    assert decoded == snapshot
    before = registry.paths.database.read_bytes()
    monkeypatch.setattr(
        Registry, "connection", lambda *a, **k: pytest.fail("Validator opened registry")
    )
    report = manifests.evaluate_assessment(decoded)
    assert AssessmentReport.from_dict(report.as_dict()) == report
    assert registry.paths.database.read_bytes() == before
    assert not any(registry.paths.manifests.rglob("completion.json"))


def test_global_registry_writes_receipts_inside_the_instance_project(graph):
    from nro.orchestration.ownership import write_instance_ownership

    registry, _, ids = graph
    global_registry = Registry.for_project(
        "", bids_root=registry.paths.bids_root, registry_path=registry.paths.control
    )
    receipt = write_instance_ownership(global_registry, ids["root"])
    assert receipt.is_relative_to(registry.paths.bids_root / "demo" / "derivatives")
    assert not (registry.paths.bids_root / "derivatives").exists()


def test_assessment_does_not_rewrite_ownership_receipts(graph, monkeypatch):
    from nro.orchestration.ownership import write_instance_ownership

    registry, _, ids = graph
    write_instance_ownership(registry, ids["root"])
    monkeypatch.setattr(
        manifests,
        "write_instance_ownership",
        lambda *_args, **_kwargs: pytest.fail("assessment rewrote ownership"),
    )
    assert (
        manifests.assess_registry(registry, instance_ids=[ids["root"]])[ids["root"]][0] == "fresh"
    )


def test_assessment_restores_a_missing_ownership_receipt(graph):
    from nro.orchestration.ownership import instance_record_path

    registry, _, ids = graph
    assert (
        manifests.assess_registry(registry, instance_ids=[ids["root"]])[ids["root"]][0] == "fresh"
    )
    row = next(row for row in registry.instance_rows() if row["id"] == ids["root"])
    receipt = instance_record_path(
        registry.paths.bids_root / "demo",
        "preprocessing",
        "main",
        "anat",
        row["instance_key"],
    )
    assert receipt.is_file()


@pytest.mark.parametrize(
    "mutation",
    ["generation", "contract", "inputs", "graph", "configuration", "claim", "reservation"],
)
def test_obsolete_reports_cannot_overwrite_newer_state(graph, mutation):
    registry, specs, ids = graph
    snapshot = capture_assessment(registry, instance_ids=[ids["leaf"]])
    report = manifests.evaluate_assessment(snapshot)
    with registry.connection(write=True) as db:
        if mutation == "generation":
            db.execute(
                "UPDATE instances SET current_generation=1, artifact_state='fresh' WHERE id=?",
                (ids["root"],),
            )
        elif mutation == "contract":
            db.execute("UPDATE instances SET artifact_fingerprint='new' WHERE id=?", (ids["leaf"],))
        elif mutation == "inputs":
            db.execute(
                "UPDATE instances SET input_paths_json='[\"new-input\"]' WHERE id=?", (ids["leaf"],)
            )
        elif mutation == "graph":
            db.execute(
                "INSERT INTO instance_dependencies VALUES (?, ?, ?, NULL)",
                (ids["leaf"], ids["unrelated"], "input"),
            )
        elif mutation == "configuration":
            db.execute(
                "UPDATE configuration_lineages SET resolved_yaml='changed: true' WHERE id=?",
                (specs[0].configuration_lineage_id,),
            )
        elif mutation == "reservation":
            db.execute("INSERT INTO artifact_mutations VALUES (?, ?)", (ids["root"], "mutation"))
    if mutation == "claim":
        registry.register_worker("test", resource_class="large")
        assert registry.claim_ready_instance("test", ("large",)) is not None
    before = registry.instance_rows()
    with pytest.raises(AssessmentConflict):
        apply_assessment(registry, snapshot, report)
    assert registry.instance_rows() == before


def test_unrelated_work_resources_and_heartbeats_do_not_reject_report(graph):
    registry, _, ids = graph
    snapshot = capture_assessment(registry, instance_ids=[ids["leaf"]])
    report = manifests.evaluate_assessment(snapshot)
    with registry.connection(write=True) as db:
        db.execute("UPDATE instances SET current_generation=8 WHERE id=?", (ids["unrelated"],))
        db.execute("UPDATE instances SET memory_gb=128, max_memory_gb=256")
    registry.register_worker("idle", resource_class="large")
    registry.heartbeat_worker("idle", state="idle")
    states = apply_assessment(registry, snapshot, report)
    assert states[ids["root"]][0] == "fresh"
    assert states[ids["leaf"]][0] == "fresh"
    assert ids["unrelated"] not in states


@pytest.mark.parametrize(
    "change", ["extra", "duplicate", "unknown_field", "fingerprint", "bad_state"]
)
def test_reports_cannot_expand_or_change_their_publication_scope(graph, change):
    registry, _, ids = graph
    snapshot = capture_assessment(registry, instance_ids=[ids["root"]])
    value = manifests.evaluate_assessment(snapshot).as_dict()
    if change == "extra":
        value["updates"].append({**value["updates"][0], "id": ids["unrelated"]})
    elif change == "duplicate":
        value["updates"].append(value["updates"][0])
    elif change == "unknown_field":
        value["updates"][0]["current_generation"] = 10
    elif change == "fingerprint":
        value["snapshot_fingerprint"] = "wrong"
    elif change == "bad_state":
        value["updates"][0]["state"] = "success"
    with pytest.raises(ValueError):
        apply_assessment(registry, snapshot, AssessmentReport.from_dict(value))
    assert all(row["artifact_state"] == "missing" for row in registry.instance_rows())


def test_normal_assessment_retries_a_concurrent_completion(graph, monkeypatch):
    registry, _, ids = graph
    evaluate = manifests.evaluate_assessment
    calls = []

    def concurrent(snapshot):
        report = evaluate(snapshot)
        calls.append(snapshot)
        if len(calls) == 1:
            with registry.connection(write=True) as db:
                db.execute("UPDATE instances SET current_generation=1 WHERE id=?", (ids["root"],))
        return report

    monkeypatch.setattr(manifests, "evaluate_assessment", concurrent)
    result = manifests.assess_registry(registry, instance_ids=[ids["leaf"]])
    assert result[ids["root"]][0] == "fresh"
    assert len(calls) == 2


def test_assessment_cannot_redirect_outputs(graph):
    registry, _, ids = graph
    snapshot = capture_assessment(registry, instance_ids=[ids["root"]])
    value = manifests.evaluate_assessment(snapshot).as_dict()
    contract = json.loads(snapshot.instances[0]["artifact_contract_json"])
    contract["output"]["root"] = "/unrelated"
    value["updates"][0]["contract"] = contract
    with pytest.raises(ValueError, match="cannot change"):
        apply_assessment(registry, snapshot, AssessmentReport.from_dict(value))


def test_worker_defers_contended_assessment_without_stopping(graph, monkeypatch):
    from nro.orchestration import worker

    registry, _, _ = graph

    def conflict(*args, **kwargs):
        raise AssessmentConflict("test contention")

    monkeypatch.setattr(worker, "assess_registry", conflict)
    process = worker.Worker(registry, resource_class="large")
    process._refresh_scheduler_state()
    assert not process.stop_requested
    with registry.connection() as db:
        assert (
            db.execute(
                "SELECT value FROM metadata WHERE key='artifact_assessment_lease_until'"
            ).fetchone()[0]
            == "0"
        )


def test_worker_validates_registered_contract_without_scientific_catalog(graph, monkeypatch):
    import nro.orchestration.catalog as catalog

    registry, specs, ids = graph
    registry.register_worker("test", resource_class="large")
    claim = registry.claim_ready_instance("test", ("large",))
    assert claim.instance_id == ids["root"]
    with registry.connection(write=True) as db:
        contract = json.loads(
            db.execute(
                "SELECT artifact_contract_json FROM instances WHERE id=?", (ids["root"],)
            ).fetchone()[0]
        )
        contract["module"] = "unknown_extension"
        db.execute(
            "UPDATE instances SET module=?,artifact_contract_json=?,artifact_fingerprint=? WHERE id=?",
            ("unknown_extension", json.dumps(contract), fingerprint(contract), ids["root"]),
        )
    monkeypatch.setattr(
        catalog,
        "module_descriptor",
        lambda *_: pytest.fail("Worker consulted a scientific catalog"),
    )
    monkeypatch.setattr(
        catalog, "canonical_contract", lambda *_: pytest.fail("Worker recompiled a contract")
    )
    manifests.record_completion(
        registry,
        instance_id=ids["root"],
        attempt_id=claim.attempt_id,
        outputs=specs[0].expected_outputs,
    )
    assert (
        manifests.assess_registry(registry, instance_ids=[ids["root"]], compiled=True)[ids["root"]][
            0
        ]
        == "fresh"
    )
    with registry.connection(write=True) as db:
        contract["processing"]["new_requirement"] = True
        db.execute(
            "UPDATE instances SET artifact_contract_json=?,artifact_fingerprint=? WHERE id=?",
            (json.dumps(contract), fingerprint(contract), ids["root"]),
        )
    assert (
        manifests.assess_registry(registry, instance_ids=[ids["root"]], compiled=True)[ids["root"]][
            0
        ]
        == "stale"
    )
