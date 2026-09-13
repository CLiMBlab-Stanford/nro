"""Test the scheduler's lock-free transport and launch election."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from nro.engine.io import atomic_write_json
from nro.orchestration import scheduler_bus, scheduler_client, scheduler_service
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import Registry


def test_simultaneous_callers_elect_one_controller(tmp_path):
    control = tmp_path / ".nro"

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _index: scheduler_bus.claim_launch(control), range(8)))

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert scheduler_bus.read_launch(control)["token"] == winners[0].token


def test_message_and_response_survive_independent_readers(tmp_path):
    control = tmp_path / ".nro"
    message_id = scheduler_bus.publish_message(control, {"operation": "example"})
    path = scheduler_bus.message_path(control, message_id)

    record = scheduler_bus.consume_message(path)
    scheduler_bus.publish_response(control, message_id, {"result": {"ok": True}})
    scheduler_bus.acknowledge_message(path)

    assert record["payload"] == {"operation": "example"}
    assert scheduler_bus.read_response(control, message_id) == {"result": {"ok": True}}


def test_controller_startup_error_is_scoped_to_launch_token(tmp_path):
    control = tmp_path / ".nro"
    scheduler_bus.prepare(control)
    scheduler_bus.publish_startup_error(control, "current", "database is locked")

    assert scheduler_bus.read_startup_error(control, "current")["error"] == "database is locked"
    assert scheduler_bus.read_startup_error(control, "other") is None


def test_cached_status_neither_starts_service_nor_opens_database(tmp_path, monkeypatch):
    control = tmp_path / ".nro"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(
        BranchStore,
        "read",
        lambda _self: SimpleNamespace(
            topology=SimpleNamespace(registered_checkout=lambda _checkout: "main")
        ),
    )
    scheduler_bus.prepare(control)
    atomic_write_json(
        ControlPaths(control).service_snapshot,
        {
            "protocol": scheduler_bus.PROTOCOL,
            "generation": 4,
            "published_at": "test",
            "service_active": False,
            "workers": [],
            "branches": {
                "main": {
                    "rows": [{"id": 1}],
                    "visible_ids": [1],
                    "ingestion": [],
                    "dependencies": [],
                }
            },
        },
    )
    monkeypatch.setattr(
        scheduler_client,
        "_endpoint",
        lambda *_args: (_ for _ in ()).throw(AssertionError("cached status started service")),
    )

    report = scheduler_client.status(control, tmp_path / "BIDS", checkout=checkout, mode="cached")

    assert report["rows"] == [{"id": 1}]


def test_worker_role_cannot_open_scheduler_database(tmp_path, monkeypatch):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    monkeypatch.setenv("NRO_PROCESS_ROLE", "worker")

    with pytest.raises(RuntimeError, match="cannot open the scheduler registry"):
        with registry.connection():
            pass


def test_worker_event_replay_is_idempotent_and_older_events_are_rejected(tmp_path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    event = {
        "action": "register",
        "worker_id": "worker",
        "worker_token": "token",
        "sequence": 1,
        "resource_class": "large",
        "memory_gb": 32,
        "slurm_job_id": None,
        "user_name": "worker-user",
        "hostname": "worker-host",
        "pid": 12345,
    }

    assert scheduler_service.worker_operation(registry, event) is None
    assert scheduler_service.worker_operation(registry, event) is None
    with registry.connection() as db:
        worker = db.execute(
            "SELECT user_name, hostname, pid FROM workers WHERE id='worker'"
        ).fetchone()
    assert tuple(worker) == ("worker-user", "worker-host", 12345)
    with pytest.raises(ValueError, match="predates"):
        scheduler_service.worker_operation(
            registry,
            {**event, "action": "shutdown_requested", "sequence": 0},
        )
