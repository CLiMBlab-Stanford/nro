from types import SimpleNamespace

from nro.orchestration import manifests, scheduler_service
from nro.orchestration.registry import Registry
from nro.orchestration.source_snapshots import SourceSnapshot


def test_batch_verifies_source_and_assesses_projects_once(monkeypatch, tmp_path):
    registry = SimpleNamespace(
        paths=SimpleNamespace(bids_root=tmp_path / "BIDS", control=tmp_path / "control")
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

    def fake_admit(selected, payload, **kwargs):
        admitted.append((selected, payload, kwargs))
        return payload["project"]

    monkeypatch.setattr(scheduler_service, "admit", fake_admit)

    result = scheduler_service.admit_many(
        registry, entries, checkout=tmp_path / "checkout", site_values={}
    )

    assert result == ["one", "one", "two"]
    assert verified == ["a" * 64]
    assert assessed == [(registry, {"projects": ("one", "two"), "compiled": True})]
    assert all(call[2]["assess"] is False for call in admitted)
    assert all(call[2]["source_verified"] is True for call in admitted)
