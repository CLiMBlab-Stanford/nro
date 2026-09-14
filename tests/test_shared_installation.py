"""Shared installation drains execution and publishes tagged main source."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from nro.engine import shared_installation
from nro.orchestration import scheduler_service, worker_control
from nro.orchestration.branch_registry import SCHEMA_VERSION as SCIENTIFIC_SCHEMA_VERSION
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import Registry


def _mock_service(monkeypatch, tmp_path, responses):
    implementation = tmp_path / "implementation.json"
    implementation.write_text("{}")
    operations = []
    values = iter(responses)
    monkeypatch.setattr(
        "nro.orchestration.scheduler_implementation.implementation_path",
        lambda control: implementation,
    )
    monkeypatch.setattr(
        "nro.orchestration.scheduler_client.maintenance",
        lambda *args, operation, **fields: operations.append((operation, fields)) or next(values),
    )
    monkeypatch.setattr(
        "nro.orchestration.scheduler_client.shutdown_service",
        lambda *args, **kwargs: {"shutdown": True},
    )
    monkeypatch.setattr("nro.orchestration.scheduler_bus.read_active", lambda control: None)
    return operations


def test_pool_drain_requires_confirmation_before_mutation(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    operations = _mock_service(
        monkeypatch,
        tmp_path,
        ({"workers": 1, "submissions": 0, "attempts": 0, "ingestion": 0},),
    )

    with pytest.raises(RuntimeError, match="not changed"):
        shared_installation.prepare_pool(
            registry, checkout=tmp_path / "main", confirm=lambda activity: None
        )

    assert [operation for operation, _ in operations] == ["installation_activity"]


def test_pool_drain_preserves_demand_and_stops_workers(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    operations = _mock_service(
        monkeypatch,
        tmp_path,
        (
            {"workers": 1, "submissions": 0, "attempts": 0, "ingestion": 0},
            {
                "workers": 0,
                "submissions": 0,
                "attempts": 0,
                "ingestion": 0,
                "action": "drain",
                "done": True,
                "stopped_jobs": [],
                "failures": [],
            },
        ),
    )

    result = shared_installation.prepare_pool(
        registry, checkout=tmp_path / "main", confirm=lambda activity: "drain"
    )

    assert result["workers"] == 0
    assert [operation for operation, _ in operations] == [
        "installation_activity",
        "installation_prepare",
    ]


def test_installation_repairs_scientific_schema_when_scheduler_is_current(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    branches = BranchStore(registry.paths.control)
    branches.initialize()
    scientific = branches.registry("main")
    scientific.record_instance("example", {"module": "anat"}, expected_revision=None)
    with sqlite3.connect(scientific.database) as db:
        db.execute(f"PRAGMA user_version={SCIENTIFIC_SCHEMA_VERSION - 1}")

    _mock_service(
        monkeypatch,
        tmp_path,
        (
            {"workers": 0, "submissions": 0, "attempts": 0, "ingestion": 0},
            {
                "workers": 0,
                "submissions": 0,
                "attempts": 0,
                "ingestion": 0,
                "action": "drain",
                "done": True,
                "stopped_jobs": [],
                "failures": [],
            },
        ),
    )

    result = shared_installation.prepare_pool(
        registry, checkout=tmp_path / "main", confirm=lambda _activity: "stop"
    )

    assert result["scientific"] == [
        {
            "branch": "main",
            "stored_schema": SCIENTIFIC_SCHEMA_VERSION - 1,
            "schema": SCIENTIFIC_SCHEMA_VERSION,
            "backup": str(scientific.root / "registry-before-repair.sqlite3"),
        }
    ]
    assert scientific.stored_schema_version() == SCIENTIFIC_SCHEMA_VERSION
    assert [(item.key, item.revision) for item in scientific.instances()] == [("example", 1)]


def test_installation_rebuilds_obsolete_schema_before_scheduler_calls(
    tmp_path, monkeypatch, capsys
):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    operations = _mock_service(
        monkeypatch,
        tmp_path,
        (
            {"workers": 0, "submissions": 0, "attempts": 0, "ingestion": 0},
            {
                "workers": 0,
                "submissions": 0,
                "attempts": 0,
                "ingestion": 0,
                "action": "drain",
                "done": True,
                "stopped_jobs": [],
                "failures": [],
            },
        ),
    )
    monkeypatch.setattr(registry, "stored_schema_version", lambda: 17)
    monkeypatch.setattr(
        registry,
        "worker_pool_activity",
        lambda **_options: {"workers": [], "submissions": []},
    )
    repaired = []
    monkeypatch.setattr(
        "nro.orchestration.scheduler_repair.repair_for_installation",
        lambda selected, *, checkout: (
            repaired.append((selected, checkout)) or {"backup": tmp_path / "backup"}
        ),
    )

    result = shared_installation.prepare_pool(
        registry,
        checkout=tmp_path / "main",
        confirm=lambda _activity: "stop",
        rebuild_schema=True,
    )

    assert result["done"]
    assert repaired == [(registry, (tmp_path / "main").resolve())]
    assert [operation for operation, _ in operations] == [
        "installation_activity",
        "installation_prepare",
    ]
    assert "Rebuilt scheduler schema 17" in capsys.readouterr().out


def test_schema_rebuild_stops_the_live_scheduler_before_direct_registry_access(
    tmp_path, monkeypatch
):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    _mock_service(
        monkeypatch,
        tmp_path,
        (
            {"workers": 0, "submissions": 0, "attempts": 0, "ingestion": 0},
            {
                "workers": 0,
                "submissions": 0,
                "attempts": 0,
                "ingestion": 0,
                "action": "drain",
                "done": True,
                "stopped_jobs": [],
                "failures": [],
            },
        ),
    )
    monkeypatch.setattr(registry, "stored_schema_version", lambda: 17)
    scheduler_stopped = False
    active_records = [{"token": "old"}]
    monkeypatch.setattr(
        "nro.orchestration.scheduler_bus.read_active",
        lambda _control: active_records.pop() if active_records else None,
    )

    def stop_scheduler(*_args, **_kwargs):
        nonlocal scheduler_stopped
        scheduler_stopped = True
        return {"stopping": True}

    monkeypatch.setattr("nro.orchestration.scheduler_client.shutdown_service", stop_scheduler)

    def activity(**_options):
        assert scheduler_stopped
        return {"workers": [], "submissions": []}

    monkeypatch.setattr(registry, "worker_pool_activity", activity)
    monkeypatch.setattr(
        "nro.orchestration.scheduler_repair.repair_for_installation",
        lambda *_args, **_kwargs: {"backup": tmp_path / "backup"},
    )

    result = shared_installation.prepare_pool(
        registry,
        checkout=tmp_path / "main",
        confirm=lambda _activity: "stop",
        rebuild_schema=True,
    )

    assert result["done"]
    assert scheduler_stopped


def test_pool_stop_interrupts_workers_without_waiting_for_attempts(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    _mock_service(
        monkeypatch,
        tmp_path,
        (
            {"workers": 1, "submissions": 0, "attempts": 1, "ingestion": 0},
            {
                "workers": 0,
                "submissions": 0,
                "attempts": 0,
                "ingestion": 0,
                "action": "stop",
                "done": True,
                "stopped_jobs": ["101"],
                "failures": [],
            },
        ),
    )

    result = shared_installation.prepare_pool(
        registry, checkout=tmp_path / "main", confirm=lambda activity: "stop"
    )

    assert result["action"] == "stop"
    assert result["stopped_jobs"] == ["101"]
    assert result["done"] is True


def test_installation_progress_is_polled_until_workers_stop(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    checkout = tmp_path / "main"
    _mock_service(
        monkeypatch,
        tmp_path,
        (
            {"workers": 1, "submissions": 0, "attempts": 1, "ingestion": 0},
            {
                "workers": 1,
                "submissions": 0,
                "attempts": 1,
                "ingestion": 0,
                "action": "drain",
                "done": False,
                "stopped_jobs": [],
                "failures": [],
            },
            {
                "workers": 0,
                "submissions": 0,
                "attempts": 0,
                "ingestion": 0,
                "action": "drain",
                "done": True,
                "stopped_jobs": [],
                "failures": [],
            },
        ),
    )

    result = shared_installation.prepare_pool(
        registry,
        checkout=checkout,
        confirm=lambda activity: "drain",
        poll_interval=0,
        report_interval=999,
    )

    assert result["done"] is True


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


def test_installation_stop_waits_for_confirmed_allocation_exit(tmp_path, monkeypatch):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    registry.initialize()
    checkout = tmp_path / "main"
    checkout.mkdir()
    registry.register_worker("worker-1", resource_class="small", slurm_job_id="101")
    with registry.connection(write=True) as db:
        db.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            (
                ("maintenance_mode", "installation"),
                ("installation_checkout", str(checkout.resolve())),
                ("installation_action", "stop"),
            ),
        )

    topology = SimpleNamespace(registered_checkout=lambda _checkout: "main")
    monkeypatch.setattr(
        scheduler_service,
        "BranchStore",
        lambda _control: SimpleNamespace(read=lambda: SimpleNamespace(topology=topology)),
    )
    terminal = False
    monkeypatch.setattr(worker_control, "_slurm_job_terminal", lambda _job_id: terminal)
    monkeypatch.setattr(
        worker_control.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    waiting = scheduler_service.installation_progress(registry, checkout=checkout)

    assert waiting["done"] is False
    assert waiting["workers"] == 1
    with registry.connection() as db:
        assert db.execute("SELECT state FROM workers WHERE id='worker-1'").fetchone()[0] == (
            "shutdown_requested"
        )

    terminal = True
    finished = scheduler_service.installation_progress(registry, checkout=checkout)

    assert finished["done"] is True
    assert finished["workers"] == 0
    with registry.connection() as db:
        assert db.execute("SELECT state FROM workers WHERE id='worker-1'").fetchone()[0] == (
            "terminated"
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
