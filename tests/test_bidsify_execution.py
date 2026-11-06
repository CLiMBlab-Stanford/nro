"""Ingestion pins accept only the captured source that the launcher verified."""

import pytest

from nro.bidsify import execution
from nro.orchestration.source_snapshots import SourceStore


class ReachedSite(Exception):
    pass


def pinned_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "nro").mkdir(parents=True)
    (root / "nro/__init__.py").write_text("")
    snapshot = SourceStore(tmp_path / "snapshots").capture(root)

    def settings(**_):
        raise ReachedSite

    monkeypatch.setattr(execution, "settings", settings)
    # The user launcher exports the registered checkout to every nro process.
    monkeypatch.setenv("NRO_CHECKOUT", str(root))
    pin = {
        "source_root": str(snapshot.root),
        "source_digest": snapshot.digest,
        "site": str(tmp_path / "site.toml"),
    }
    return snapshot, pin


def test_snapshot_process_accepts_its_pin_despite_launcher_checkout(tmp_path, monkeypatch):
    snapshot, pin = pinned_snapshot(tmp_path, monkeypatch)
    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(snapshot.root))
    with pytest.raises(ReachedSite):
        execution.validate_execution(pin)


def test_checkout_process_rejects_snapshot_pin(tmp_path, monkeypatch):
    _, pin = pinned_snapshot(tmp_path, monkeypatch)
    monkeypatch.delenv("NRO_EXECUTION_SOURCE_ROOT", raising=False)
    with pytest.raises(ValueError, match="captured implementation"):
        execution.validate_execution(pin)
