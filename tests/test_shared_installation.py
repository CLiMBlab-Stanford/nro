"""Shared installation drains execution and publishes tagged main source."""

import json
import sqlite3

import pytest

from nro.engine import shared_installation
from nro.orchestration import worker_control
from nro.orchestration.registry import Registry


def test_pool_drain_requires_confirmation_before_mutation(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    registry.register_worker("worker", resource_class="large")

    with pytest.raises(RuntimeError, match="not changed"):
        shared_installation.prepare_pool(
            registry, checkout=tmp_path / "main", confirm=lambda activity: None
        )

    with registry.connection() as db:
        assert (
            db.execute("SELECT value FROM metadata WHERE key='maintenance_mode'").fetchone() is None
        )
        assert db.execute("SELECT state FROM workers WHERE id='worker'").fetchone()[0] == "idle"


def test_pool_drain_preserves_demand_and_stops_workers(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    registry.register_worker("worker", resource_class="large")
    with sqlite3.connect(registry.paths.database) as db:
        db.execute(
            "INSERT INTO requests VALUES "
            "('request','user','demo',1,'anat','{}',2,NULL,'active','now','now')"
        )
    monkeypatch.setattr(
        shared_installation, "cancel_worker_allocations", lambda registry, shutdown: (0, [])
    )
    monkeypatch.setattr(shared_installation, "wait_for_worker_shutdown", lambda registry: None)

    result = shared_installation.prepare_pool(
        registry, checkout=tmp_path / "main", confirm=lambda activity: "drain"
    )

    assert result["workers"] == 1
    with registry.connection() as db:
        assert db.execute("SELECT state FROM requests WHERE id='request'").fetchone()[0] == "active"
        assert db.execute("SELECT state FROM workers WHERE id='worker'").fetchone()[0] == (
            "terminated"
        )
        assert (
            db.execute("SELECT value FROM metadata WHERE key='maintenance_mode'").fetchone()[0]
            == "installation"
        )
        assert (
            db.execute("SELECT value FROM metadata WHERE key='installation_action'").fetchone()[0]
            == "drain"
        )


def test_pool_stop_interrupts_workers_without_waiting_for_attempts(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    registry.register_worker("worker", resource_class="large", slurm_job_id="101")
    events = []
    original_shutdown = registry.request_worker_shutdown

    def shutdown(**options):
        events.append("shutdown")
        return original_shutdown(**options)

    monkeypatch.setattr(registry, "request_worker_shutdown", shutdown)
    monkeypatch.setattr(shared_installation, "_executing", lambda registry: (1, 0))
    monkeypatch.setattr(
        shared_installation,
        "cancel_worker_allocations",
        lambda registry, request: (events.append("cancel") or 1, []),
    )
    monkeypatch.setattr(
        shared_installation,
        "wait_for_worker_shutdown",
        lambda registry: events.append("confirmed"),
    )

    result = shared_installation.prepare_pool(
        registry, checkout=tmp_path / "main", confirm=lambda activity: "stop"
    )

    assert result["action"] == "stop"
    assert result["stopped_jobs"] == 1
    assert result["finalized"] == {"workers": 1, "attempts": 0, "ingestion": 0}
    assert events == ["shutdown", "cancel", "confirmed"]
    with registry.connection() as db:
        assert db.execute("SELECT state FROM workers WHERE id='worker'").fetchone()[0] == (
            "terminated"
        )
        assert (
            db.execute("SELECT value FROM metadata WHERE key='installation_action'").fetchone()[0]
            == "stop"
        )


def test_legacy_maintenance_barrier_asks_for_an_action_when_resumed(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    checkout = (tmp_path / "main").resolve()
    registry.register_worker("worker", resource_class="large", slurm_job_id="101")
    with registry.connection(write=True) as db:
        db.execute("INSERT INTO metadata VALUES ('maintenance_mode','installation')")
        db.execute("INSERT INTO metadata VALUES ('installation_checkout',?)", (str(checkout),))
    choices = []
    monkeypatch.setattr(shared_installation, "_executing", lambda registry: (1, 0))
    monkeypatch.setattr(
        shared_installation, "cancel_worker_allocations", lambda registry, request: (1, [])
    )
    monkeypatch.setattr(shared_installation, "wait_for_worker_shutdown", lambda registry: None)

    result = shared_installation.prepare_pool(
        registry,
        checkout=checkout,
        confirm=lambda activity: choices.append(activity) or "stop",
    )

    assert result["resuming"] is True
    assert result["action"] == "stop"
    assert len(choices) == 1


def test_worker_cancellation_accepts_an_already_finished_allocation(monkeypatch):
    updates = []

    class RegistryStub:
        def update_submission(self, submission_id, *, state):
            updates.append((submission_id, state))

    monkeypatch.setattr(worker_control, "_slurm_job_terminal", lambda job_id: True)
    monkeypatch.setattr(
        worker_control.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("an absent allocation cannot be cancelled again"),
    )
    stopped, failures = worker_control.cancel_worker_allocations(
        RegistryStub(),
        {
            "submissions": [(7, "101")],
            "job_ids": ("101",),
        },
    )

    assert stopped == 0
    assert failures == []
    assert updates == [(7, "cancelled")]


def test_publish_records_release_in_installation(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="example"\nversion="1.2.3"\n')
    (root / ".gitignore").write_text(".nro-installation.json\nenvironment/\nsite.toml\n")
    import subprocess

    for args in (
        ("init", "-b", "main"),
        ("config", "user.name", "Test Maintainer"),
        ("config", "user.email", "maintainer@example.invalid"),
        ("add", "."),
        ("commit", "-m", "Release"),
        ("tag", "-a", "v1.2.3", "-m", "Release 1.2.3"),
        ("update-ref", "refs/remotes/origin/main", "HEAD"),
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    environment = root / "environment/bin"
    environment.mkdir(parents=True)
    (environment / "python").write_text("")
    site = root / "site.toml"
    site.write_text("")
    installation = {
        "mode": "shared",
        "checkout": str(root),
        "environment": str(environment.parent),
        "site": str(site),
        "ready": True,
    }
    (root / ".nro-installation.json").write_text(json.dumps(installation))
    monkeypatch.setattr(
        "nro.configuration.site.installation_record", lambda checkout=None: installation.copy()
    )
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    monkeypatch.setattr(
        shared_installation,
        "activate",
        lambda registry, checkout, installation_maintenance: {"checkout": str(checkout)},
    )

    result = shared_installation.publish(root, registry)

    saved = json.loads((root / ".nro-installation.json").read_text())
    assert saved["release"] == result["release"]
    assert saved["release"]["version"] == "1.2.3"
