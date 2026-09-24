from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from nro.orchestration import manifests, scheduler_operations, scheduler_service
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import Registry
from nro.orchestration.source_snapshots import SourceSnapshot


def test_batch_verifies_source_and_assesses_projects_once(monkeypatch, tmp_path):
    db = object()
    registry = SimpleNamespace(
        paths=SimpleNamespace(bids_root=tmp_path / "BIDS", control=tmp_path / "control"),
        connection=lambda **_kwargs: nullcontext(db),
    )
    source = {"root": str(tmp_path / "source"), "digest": "a" * 64}
    entries = [
        {"project": project, "payload": {"project": project, "source": source}}
        for project in ("one", "one", "two")
    ]
    verified = []
    assessed = []
    admitted = []

    monkeypatch.setattr(
        SourceSnapshot, "verify_manifest", lambda self: verified.append(self.digest)
    )
    monkeypatch.setattr(
        manifests,
        "assess_registry",
        lambda selected, **kwargs: assessed.append((selected, kwargs)),
    )
    monkeypatch.setattr(
        Registry,
        "for_project",
        lambda project, **kwargs: SimpleNamespace(project=project, kwargs=kwargs),
    )
    topology = object()
    monkeypatch.setattr(
        scheduler_service,
        "BranchStore",
        lambda _control: SimpleNamespace(
            _lock=lambda: nullcontext(), read=lambda: SimpleNamespace(topology=topology)
        ),
    )
    monkeypatch.setattr(
        scheduler_service,
        "candidates_locked",
        lambda _db, project: (f"candidate:{project}",),
    )
    monkeypatch.setattr(scheduler_service, "protected_site_fingerprint", lambda _path: "site")

    def fake_admit(selected, payload, **kwargs):
        admitted.append((selected, payload, kwargs))
        return payload["project"]

    monkeypatch.setattr(scheduler_service, "_admit", fake_admit)

    result = scheduler_service.admit_many(
        registry,
        entries,
        checkout=tmp_path / "checkout",
        site_values={"definitions": str(tmp_path / "definitions")},
    )

    assert result == ["one", "one", "two"]
    assert verified == ["a" * 64]
    assert assessed == [(registry, {"projects": ("one", "two"), "compiled": False})]
    assert all(call[2]["assess"] is False for call in admitted)
    assert all(call[2]["source_verified"] is True for call in admitted)
    assert all(call[2]["expected_site"] == "site" for call in admitted)
    assert [call[2]["locked"] for call in admitted] == [
        (topology, db, ("candidate:one",)),
        (topology, db, ("candidate:one",)),
        (topology, db, ("candidate:two",)),
    ]


def test_batch_rolls_back_every_admission_when_a_later_entry_fails(monkeypatch, tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    source = {"root": str(tmp_path / "source"), "digest": "a" * 64}
    entries = [
        {"project": project, "payload": {"project": project, "source": source}}
        for project in ("one", "two")
    ]
    monkeypatch.setattr(SourceSnapshot, "verify_manifest", lambda self: None)
    monkeypatch.setattr(manifests, "assess_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Registry,
        "for_project",
        lambda project, **kwargs: SimpleNamespace(project=project, kwargs=kwargs),
    )
    topology = object()
    monkeypatch.setattr(
        scheduler_service,
        "BranchStore",
        lambda _control: SimpleNamespace(
            _lock=lambda: nullcontext(), read=lambda: SimpleNamespace(topology=topology)
        ),
    )
    monkeypatch.setattr(scheduler_service, "candidates_locked", lambda _db, _project: ())
    monkeypatch.setattr(scheduler_service, "protected_site_fingerprint", lambda _path: "site")
    calls = 0

    def fake_admit(_registry, _payload, **kwargs):
        nonlocal calls
        calls += 1
        db = kwargs["locked"][1]
        db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('batch-test', 'written')")
        if calls == 2:
            raise RuntimeError("later admission failed")
        return "request-one"

    monkeypatch.setattr(scheduler_service, "_admit", fake_admit)

    with pytest.raises(RuntimeError, match="later admission failed"):
        scheduler_service.admit_many(
            registry,
            entries,
            checkout=tmp_path / "checkout",
            site_values={"definitions": str(tmp_path / "definitions")},
        )

    with registry.connection() as db:
        assert db.execute("SELECT value FROM metadata WHERE key='batch-test'").fetchone() is None


def test_status_uses_mixed_contract_assessment_for_every_checkout(monkeypatch, tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    branches = BranchStore(registry.paths.control)
    main = tmp_path / "main"
    feature = tmp_path / "feature"
    main.mkdir()
    feature.mkdir()
    revision = branches.initialize().revision
    monkeypatch.setattr(
        "nro.orchestration.branches.checkout_identity",
        lambda checkout: (checkout, checkout.name, "revision"),
    )
    branches.authorize_checkout("main", main, revision=revision)
    branches.register("feature", "dev", revision=branches.read().revision, checkout=feature)
    assessed = []
    monkeypatch.setattr(
        manifests,
        "assess_registry",
        lambda selected, **kwargs: assessed.append((selected, kwargs)),
    )

    scheduler_operations.status(registry, checkout=main, mode="verify")
    scheduler_operations.status(registry, checkout=feature, mode="verify")

    assert assessed == [
        (
            registry,
            {"work_item_ids": set(), "compiled": False, "recover_public": True},
        ),
        (
            registry,
            {"work_item_ids": set(), "compiled": False, "recover_public": True},
        ),
    ]
