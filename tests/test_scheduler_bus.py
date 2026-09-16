"""Test scheduler transport, recovery records, and launch election."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from nro.engine.io import atomic_write_json
from nro.orchestration import scheduler_bus, scheduler_client, scheduler_rpc, scheduler_service
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.dependency_state import AttemptInvalidated
from nro.orchestration.registry import Registry
from nro.orchestration.worker_client import WorkerSchedulerClient


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


def test_scheduler_progress_is_atomic_and_transient(tmp_path):
    control = tmp_path / ".nro"
    scheduler_bus.prepare(control)

    scheduler_bus.publish_progress(
        control,
        "request",
        phase="Removing artifact paths",
        completed=25,
        total=100,
    )

    record = scheduler_bus.read_progress(control, "request")
    assert record is not None
    assert record | {"updated_at": None} == {
        "protocol": scheduler_bus.PROTOCOL,
        "id": "request",
        "phase": "Removing artifact paths",
        "completed": 25,
        "total": 100,
        "updated_at": None,
    }
    scheduler_bus.clear_progress(control, "request")
    assert scheduler_bus.read_progress(control, "request") is None


def test_direct_rpc_round_trip() -> None:
    class MemorySocket:
        def __init__(self) -> None:
            self.data = bytearray()

        def sendall(self, data: bytes) -> None:
            self.data.extend(data)

        def recv(self, size: int) -> bytes:
            chunk = self.data[:size]
            del self.data[:size]
            return bytes(chunk)

    connection = MemorySocket()
    record = {"id": "request", "payload": {"operation": "status"}}
    scheduler_rpc.send(connection, record)
    response = scheduler_rpc.receive(connection)

    assert response == record


def test_direct_rpc_rejects_an_obsolete_scheduler_token() -> None:
    envelope = {
        "protocol": scheduler_bus.PROTOCOL,
        "token": "old",
        "generation": 3,
        "record": {},
    }

    with pytest.raises(ValueError, match="endpoint is obsolete"):
        scheduler_rpc.validate_request(envelope, token="current")


def test_worker_heartbeat_uses_direct_only_rpc(monkeypatch) -> None:
    client = object.__new__(WorkerSchedulerClient)
    client.endpoint = SimpleNamespace()
    client.worker_id = "worker"
    client.token = "token"
    client.sequence = 0
    calls = []
    monkeypatch.setattr(
        "nro.orchestration.worker_client.exchange",
        lambda _endpoint, message, **options: calls.append((message, options)),
    )

    client.heartbeat_worker("worker", state="running")

    assert calls[0][0]["action"] == "heartbeat"
    assert calls[0][1]["durable"] is False
    assert calls[0][1]["require_service"] is True


def test_worker_output_visibility_uses_direct_only_rpc(monkeypatch) -> None:
    client = object.__new__(WorkerSchedulerClient)
    client.endpoint = SimpleNamespace()
    calls = []
    monkeypatch.setattr(
        "nro.orchestration.worker_client.exchange",
        lambda _endpoint, message, **options: calls.append((message, options)) or True,
    )

    assert client.outputs_visible((Path("/tmp/output"),))
    assert calls == [
        (
            {"operation": "output_visibility", "paths": ["/tmp/output"]},
            {"timeout": 60.0, "require_service": True, "durable": False},
        )
    ]


def test_worker_preserves_completion_invalidation_across_scheduler_rpc(monkeypatch) -> None:
    client = object.__new__(WorkerSchedulerClient)

    def invalidated(*_args, **_kwargs):
        raise scheduler_client.SchedulerError(
            "Attempt changed before completion",
            error_type="AttemptInvalidated",
        )

    monkeypatch.setattr(client, "_call", invalidated)

    with pytest.raises(AttemptInvalidated, match="Attempt changed before completion"):
        client.record_completion(work_item_id=1, attempt_id=2, outputs=(Path("output"),))


def test_scheduler_response_preserves_service_error_type() -> None:
    with pytest.raises(scheduler_client.SchedulerError) as raised:
        scheduler_client._response_result(
            {"error": "Attempt changed before completion", "error_type": "AttemptInvalidated"}
        )

    assert raised.value.error_type == "AttemptInvalidated"


def test_scheduler_checks_output_visibility_without_registry(tmp_path) -> None:
    output = tmp_path / "output"
    output.write_text("complete")

    assert scheduler_service.dispatch(
        None,
        {"operation": "output_visibility", "paths": [str(output)]},
        values={},
        message_id="probe",
    )
    assert not scheduler_service.dispatch(
        None,
        {"operation": "output_visibility", "paths": [str(tmp_path / "missing")]},
        values={},
        message_id="probe",
    )


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
