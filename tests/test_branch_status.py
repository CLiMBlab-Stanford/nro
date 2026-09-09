"""Checkout-side status compilation stays separate from central bookkeeping."""

from types import SimpleNamespace

from nro.engine.cli import CoreSelection
from nro.orchestration import branch_status


def test_verify_compiles_current_checkout_without_creating_demand(tmp_path, monkeypatch):
    registration = SimpleNamespace(directories={"preprocessing": "main"})
    scientific = SimpleNamespace(instances=lambda: (), register_workflow=lambda _: registration)
    monkeypatch.setattr(
        branch_status,
        "BranchStore",
        lambda _: SimpleNamespace(registry_for_checkout=lambda _: scientific),
    )
    monkeypatch.setattr(
        branch_status,
        "settings",
        lambda: ({"registry": str(tmp_path / "control"), "bids": str(tmp_path / "BIDS")}, {}),
    )
    monkeypatch.setattr(branch_status, "source_fingerprint", lambda _: "captured-source")
    spec = SimpleNamespace(key="subject", module="anat", participant="01", dependencies=())
    compiled = []

    def plan_subject(**kwargs):
        compiled.append(kwargs)
        return (spec,)

    monkeypatch.setattr(
        branch_status, "Planner", lambda *a, **k: SimpleNamespace(plan_subject=plan_subject)
    )
    submitted = []
    monkeypatch.setattr(
        branch_status, "register_requests", lambda *a, **k: submitted.append((a, k))
    )
    rows = [
        dict(
            module="anat",
            project="demo",
            participant="01",
            entities_json="{}",
            workflow_ids="main",
            directory_label="main",
            memory_gb=8,
            max_memory_gb=32,
        )
    ]
    selection = CoreSelection((), (), (), (), {}, (), ())
    branch_status.refresh(rows, selection)
    assert len(compiled) == len(submitted) == 1
    assert compiled[0]["selectors"] is None
    assert submitted[0][1]["demand"] is False
    assert submitted[0][0][1].source_digest == "captured-source"


def test_preview_propagates_current_policy_changes_without_mutation(monkeypatch):
    monkeypatch.setattr(
        "nro.orchestration.manifests._current_contract", lambda row: ({}, "", row["id"] == 1)
    )
    rows = [dict(id=key, module="anat", artifact_state="fresh", status="Success") for key in (1, 2)]
    result = branch_status.preview(rows, {1, 2}, [(2, 1)])
    assert [row["status"] for row in result] == ["Stale", "Stale"]
    assert [row["status"] for row in rows] == ["Success", "Success"]


def test_verified_observations_are_saved_by_logical_key(tmp_path, monkeypatch):
    saved = []
    scientific = SimpleNamespace(record_observations=lambda values: saved.append(values))
    monkeypatch.setattr(
        branch_status,
        "BranchStore",
        lambda _: SimpleNamespace(registry_for_checkout=lambda _: scientific),
    )
    monkeypatch.setattr(
        branch_status,
        "settings",
        lambda: ({"registry": str(tmp_path / "control")}, {}),
    )
    branch_status.record_observations(
        [
            dict(
                id=1,
                logical_key="logical",
                scientific_revision=3,
                artifact_state="fresh",
                artifact_reason="Complete",
                current_generation=2,
                artifact_fingerprint="science",
            ),
            dict(
                id=2,
                logical_key="foreign",
                scientific_revision=4,
                artifact_state="missing",
                artifact_reason="Missing",
                current_generation=0,
                artifact_fingerprint="other",
            ),
        ],
        {1},
    )
    assert saved == [
        {
            "logical": (
                3,
                {
                    "artifact_state": "fresh",
                    "artifact_reason": "Complete",
                    "generation": 2,
                    "artifact_fingerprint": "science",
                },
            )
        }
    ]
