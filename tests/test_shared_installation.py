"""Shared installation drains execution and publishes tagged main source."""

import json
import sqlite3

import pytest

from nro.engine import shared_installation
from nro.orchestration.registry import Registry


def test_pool_drain_requires_confirmation_before_mutation(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    registry.register_worker("worker", resource_class="large")

    with pytest.raises(RuntimeError, match="not changed"):
        shared_installation.prepare_pool(
            registry, checkout=tmp_path / "main", confirm=lambda activity: False
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
        registry, checkout=tmp_path / "main", confirm=lambda activity: True
    )

    assert result["workers"] == 1
    with registry.connection() as db:
        assert db.execute("SELECT state FROM requests WHERE id='request'").fetchone()[0] == "active"
        assert db.execute("SELECT state FROM workers WHERE id='worker'").fetchone()[0] == (
            "shutdown_requested"
        )
        assert (
            db.execute("SELECT value FROM metadata WHERE key='maintenance_mode'").fetchone()[0]
            == "installation"
        )


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
