"""Test scheduler transport, recovery records, and launch election."""

import json
import threading
import time
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


def test_controller_requests_resources_for_threaded_service(tmp_path):
    source = SimpleNamespace(command=lambda command, *, site: list(command))

    script = scheduler_bus.write_controller_script(
        tmp_path,
        bids_root=tmp_path / "BIDS",
        token="launch",
        source=source,
        site=tmp_path / "site.toml",
        python=Path("/usr/bin/python3"),
        partition="scheduler",
        account="nlp",
    )

    text = script.read_text(encoding="utf-8")
    assert f"#SBATCH --cpus-per-task={scheduler_bus.SCHEDULER_CPUS}\n" in text
    assert "#SBATCH --mem=4G\n" in text


def test_controller_accepts_explicit_resource_request(tmp_path):
    source = SimpleNamespace(command=lambda command, *, site: list(command))

    script = scheduler_bus.write_controller_script(
        tmp_path,
        bids_root=tmp_path / "BIDS",
        token="launch",
        source=source,
        site=tmp_path / "site.toml",
        python=Path("/usr/bin/python3"),
        partition="interactive",
        account="nlp",
        time_hours=12,
        memory_gb=8,
        cpus=6,
    )

    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --time=12:00:00\n" in text
    assert "#SBATCH --mem=8G\n" in text
    assert "#SBATCH --cpus-per-task=6\n" in text


def test_scheduler_session_shares_one_connection_across_registry_handles(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    project = Registry.for_project(
        "demo",
        bids_root=tmp_path / "BIDS",
        registry_path=registry.paths.control,
    )

    with registry.scheduler_session() as owned:
        with registry.connection(write=True) as database:
            assert database is owned
            database.execute("INSERT INTO metadata(key,value) VALUES ('session-test','1')")
        with project.read_connection() as database:
            assert database is owned
            assert (
                database.execute("SELECT value FROM metadata WHERE key='session-test'").fetchone()[
                    0
                ]
                == "1"
            )

    with registry.connection() as database:
        assert database is not owned


def test_scheduler_session_serializes_transactions_from_service_threads(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    projects = [
        Registry.for_project(
            f"project-{index}",
            bids_root=tmp_path / "BIDS",
            registry_path=registry.paths.control,
        )
        for index in range(16)
    ]

    def write_metadata(index):
        with projects[index].connection(write=True) as database:
            database.execute(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                (f"thread-{index}", str(index)),
            )

    with registry.scheduler_session():
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_metadata, range(len(projects))))
        with registry.connection() as database:
            rows = database.execute(
                "SELECT key,value FROM metadata WHERE key LIKE 'thread-%'"
            ).fetchall()

    assert {row["key"]: row["value"] for row in rows} == {
        f"thread-{index}": str(index) for index in range(len(projects))
    }


def test_scheduler_session_can_bound_transport_waits(tmp_path) -> None:
    from nro.orchestration.registry import RegistryLockTimeout

    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    entered = threading.Event()
    release = threading.Event()

    def hold_connection() -> None:
        with registry.connection():
            entered.set()
            release.wait()

    with registry.scheduler_session():
        thread = threading.Thread(target=hold_connection)
        thread.start()
        assert entered.wait(timeout=2)
        with pytest.raises(RegistryLockTimeout, match="connection is busy"):
            with registry.connection(session_timeout=0.01):
                pass
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_durable_request_response_is_replayed_from_registry(tmp_path):
    from nro.orchestration.scheduler_requests import RequestCoordinator

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    record = scheduler_bus.create_message({"operation": "example"})
    calls = []
    coordinator = RequestCoordinator(
        registry,
        lambda received: calls.append(received["id"]) or {"result": {"ok": True}},
    )

    assert coordinator.run(record) == {"result": {"ok": True}}
    assert coordinator.run(record) == {"result": {"ok": True}}
    assert calls == [record["id"]]


def test_durable_request_retries_scheduler_lock_contention(tmp_path):
    from nro.orchestration.registry import RegistryLockTimeout
    from nro.orchestration.scheduler_requests import RequestCoordinator

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    record = scheduler_bus.create_message({"operation": "example"})
    calls = 0

    def operation(_received):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RegistryLockTimeout("busy")
        return {"result": "done"}

    coordinator = RequestCoordinator(registry, operation)
    with pytest.raises(RegistryLockTimeout, match="busy"):
        coordinator.run(record)
    assert coordinator.run(record) == {"result": "done"}
    assert calls == 2


def test_interrupted_durable_request_is_recovered(tmp_path):
    from nro.orchestration.scheduler_requests import prepare, register

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    record = scheduler_bus.create_message({"operation": "example"})
    assert register(registry, record) is None
    with registry.connection(write=True) as db:
        db.execute("UPDATE scheduler_requests SET state='running' WHERE id=?", (record["id"],))

    assert prepare(registry) == (record,)
    with registry.connection() as db:
        assert (
            db.execute(
                "SELECT state FROM scheduler_requests WHERE id=?", (record["id"],)
            ).fetchone()[0]
            == "pending"
        )


def test_installation_cancels_requests_from_a_dead_scheduler(tmp_path):
    from nro.orchestration.scheduler_requests import (
        RequestCoordinator,
        cancel_orphaned_for_maintenance,
        register,
    )

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    pending = scheduler_bus.create_message({"operation": "pending"})
    running = scheduler_bus.create_message({"operation": "running"})
    completed = scheduler_bus.create_message({"operation": "completed"})
    assert register(registry, pending) is None
    assert register(registry, running) is None
    RequestCoordinator(registry, lambda _record: {"result": "done"}).run(completed)
    with registry.connection(write=True) as db:
        db.execute("UPDATE scheduler_requests SET state='running' WHERE id=?", (running["id"],))

    assert cancel_orphaned_for_maintenance(registry) == 2

    with registry.connection() as db:
        rows = {
            row["id"]: (row["state"], row["response_json"])
            for row in db.execute(
                "SELECT id,state,response_json FROM scheduler_requests ORDER BY id"
            )
        }
    assert rows[completed["id"]] == ("completed", '{"result":"done"}')
    for identifier in (pending["id"], running["id"]):
        state, encoded = rows[identifier]
        assert state == "completed"
        assert json.loads(encoded) == {
            "error": "Scheduler request was cancelled for shared installation maintenance",
            "error_type": "SchedulerError",
        }


def test_installation_preserves_requests_during_a_scheduler_launch(tmp_path, monkeypatch):
    from nro.orchestration.scheduler_requests import cancel_orphaned_for_maintenance, register

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    record = scheduler_bus.create_message({"operation": "pending"})
    assert register(registry, record) is None
    monkeypatch.setattr(scheduler_bus, "read_launch", lambda _control: {"job_id": "101"})
    monkeypatch.setattr(scheduler_bus, "_launch_abandoned", lambda _record: False)

    assert cancel_orphaned_for_maintenance(registry) == 0

    with registry.connection() as db:
        state = db.execute(
            "SELECT state FROM scheduler_requests WHERE id=?", (record["id"],)
        ).fetchone()[0]
    assert state == "pending"


def test_one_shot_installation_replaces_requests_from_a_dead_epoch(tmp_path, monkeypatch):
    from nro.orchestration.scheduler_requests import cancel_orphaned_for_maintenance, register

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    current = scheduler_bus.create_message({"operation": "installation_activity"})
    orphaned = scheduler_bus.create_message({"operation": "project_rename"})
    assert register(registry, current) is None
    assert register(registry, orphaned) is None
    monkeypatch.setattr(scheduler_bus, "read_active", lambda _control: {"token": "dead-scheduler"})
    monkeypatch.setattr(
        scheduler_bus,
        "read_launch",
        lambda _control: {"token": "maintenance", "job_id": "local-101"},
    )

    assert (
        cancel_orphaned_for_maintenance(
            registry,
            launch_token="maintenance",
            preserve_ids=(current["id"],),
        )
        == 1
    )

    with registry.connection() as db:
        states = {
            row["id"]: row["state"] for row in db.execute("SELECT id,state FROM scheduler_requests")
        }
    assert states == {current["id"]: "pending", orphaned["id"]: "completed"}


def test_prepare_prunes_only_expired_completed_requests(tmp_path):
    from nro.orchestration.scheduler_requests import RequestCoordinator, prepare

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    coordinator = RequestCoordinator(registry, lambda _received: {"result": "done"})
    old = scheduler_bus.create_message({"operation": "old"})
    recent = scheduler_bus.create_message({"operation": "recent"})
    coordinator.run(old)
    coordinator.run(recent)
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE scheduler_requests SET updated_at='2020-01-01T00:00:00+00:00' WHERE id=?",
            (old["id"],),
        )

    assert prepare(registry) == ()
    with registry.connection() as db:
        identifiers = {row[0] for row in db.execute("SELECT id FROM scheduler_requests")}
    assert identifiers == {recent["id"]}


def test_concurrent_durable_retries_share_one_execution(tmp_path):
    from nro.orchestration.scheduler_requests import RequestCoordinator

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    record = scheduler_bus.create_message({"operation": "example"})
    calls = []
    coordinator = RequestCoordinator(
        registry,
        lambda received: calls.append(received["id"]) or {"result": "done"},
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = coordinator.submit(record, pool)
        second = coordinator.submit(record, pool)
        assert first.result() == {"result": "done"}
        assert second.result() == {"result": "done"}
    assert calls == [record["id"]]


def test_request_coordinator_reports_inflight_work(tmp_path) -> None:
    from nro.orchestration.scheduler_requests import RequestCoordinator

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    record = scheduler_bus.create_message({"operation": "example"})
    entered = threading.Event()
    release = threading.Event()

    def operation(_record):
        entered.set()
        release.wait()
        return {"result": "done"}

    coordinator = RequestCoordinator(registry, operation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = coordinator.submit(record, pool)
        assert entered.wait(timeout=2)
        assert coordinator.has_inflight()
        release.set()
        assert future.result(timeout=2) == {"result": "done"}
    assert not coordinator.has_inflight()


def test_scheduler_background_work_yields_to_requests() -> None:
    coordinator = SimpleNamespace(has_inflight=lambda: True)
    assert not scheduler_service._background_ready(
        coordinator,
        last_request_activity=0.0,
        now=scheduler_service.BACKGROUND_QUIET_SECONDS + 1.0,
    )

    coordinator.has_inflight = lambda: False
    assert not scheduler_service._background_ready(
        coordinator,
        last_request_activity=time.monotonic(),
        now=time.monotonic(),
    )
    assert scheduler_service._background_ready(
        coordinator,
        last_request_activity=0.0,
        now=scheduler_service.BACKGROUND_QUIET_SECONDS,
    )


def test_scheduler_separates_polling_from_maintenance_execution() -> None:
    executors = {
        name: object() for name in ("poll", "worker", "completion", "command", "maintenance")
    }
    worker = {"payload": {"operation": "worker"}}
    completion = {"payload": {"operation": "worker", "action": "record_completion"}}
    purge = {"payload": {"operation": "purge"}}

    assert scheduler_service._executor_for(worker, executors, durable=False) is executors["poll"]
    assert scheduler_service._executor_for(worker, executors) is executors["worker"]
    assert scheduler_service._executor_for(completion, executors) is executors["completion"]
    assert scheduler_service._executor_for(purge, executors) is executors["maintenance"]


def test_scheduler_keeps_durable_identity_pending_when_registration_lock_is_busy() -> None:
    from nro.orchestration.registry import RegistryLockTimeout

    class BusyCoordinator:
        def submit(self, _record, _executor):
            raise RegistryLockTimeout("busy")

    record = scheduler_bus.create_message({"operation": "example"})

    assert scheduler_service._submit_durable(
        record,
        BusyCoordinator(),
        object(),
        scheduler_service._ActivitySignal(),
    ) == {"pending": record["id"]}


def _capacity_event(worker_id: str, sequence: int, *, profile: dict | None = None) -> dict:
    return {
        "action": "request_capacity",
        "kind": "expand",
        "worker_id": worker_id,
        "sequence": sequence,
        "profile": profile,
    }


def test_equivalent_capacity_requests_coalesce_durably(tmp_path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()

    scheduler_service._queue_pool_expansion(registry, _capacity_event("first", 4))
    scheduler_service._queue_pool_expansion(registry, _capacity_event("second", 9))

    pending = scheduler_service._pending_pool_expansions(registry)
    assert len(pending) == 1
    assert pending[0][2] == {"profile": None}
    assert '"revision":"second:9"' in pending[0][1]
    assert scheduler_service._registry_busy(registry)


def test_capacity_requests_remain_distinct_by_profile(tmp_path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()

    scheduler_service._queue_pool_expansion(
        registry, _capacity_event("cpu", 1, profile={"partition": "cpu"})
    )
    scheduler_service._queue_pool_expansion(
        registry, _capacity_event("gpu", 1, profile={"partition": "gpu"})
    )

    assert {
        row[2]["profile"]["partition"]
        for row in scheduler_service._pending_pool_expansions(registry)
    } == {"cpu", "gpu"}


def test_capacity_drain_preserves_concurrent_newer_request(tmp_path, monkeypatch) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    scheduler_service._queue_pool_expansion(registry, _capacity_event("first", 1))
    calls = []

    def supply(_registry, payload):
        calls.append(payload)
        if len(calls) == 1:
            scheduler_service._queue_pool_expansion(registry, _capacity_event("second", 2))
        return []

    monkeypatch.setattr(scheduler_service, "_supply_requested_pool", supply)

    assert scheduler_service._drain_pool_expansions(registry) == 2
    assert calls == [{"profile": None}, {"profile": None}]
    assert scheduler_service._pending_pool_expansions(registry) == []
    assert not scheduler_service._registry_busy(registry)


def test_failed_capacity_drain_leaves_request_for_retry(tmp_path, monkeypatch) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    scheduler_service._queue_pool_expansion(registry, _capacity_event("worker", 1))

    def fail(_registry, _payload):
        raise RuntimeError("transient failure")

    monkeypatch.setattr(scheduler_service, "_supply_requested_pool", fail)

    assert scheduler_service._drain_pool_expansions(registry) == 0
    assert len(scheduler_service._pending_pool_expansions(registry)) == 1


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


def test_scheduler_client_rpc_attempt_does_not_block_wait_reporting(monkeypatch) -> None:
    release = threading.Event()

    def request(_active, _record, **_options):
        release.wait()
        return {"result": "done"}

    monkeypatch.setattr(scheduler_rpc, "request", request)
    attempt = scheduler_client._start_direct_attempt(
        {"token": "scheduler", "port": 1234},
        {"id": "request"},
        timeout=60.0,
        durable=True,
    )

    assert attempt.outcome() is None
    release.set()
    for _ in range(100):
        outcome = attempt.outcome()
        if outcome is not None:
            break
        time.sleep(0.01)
    assert outcome == ({"result": "done"}, None)


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
    assert calls[0][1]["timeout"] == 60.0


@pytest.mark.parametrize("operation", ("dataset_migration", "hotfix", "project_rename"))
def test_long_running_maintenance_requires_scheduler_service(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    endpoint = SimpleNamespace()
    calls = []
    monkeypatch.setattr(
        scheduler_client,
        "_endpoint",
        lambda *_args, **_kwargs: endpoint,
    )
    monkeypatch.setattr(
        scheduler_client,
        "exchange",
        lambda selected, message, **options: calls.append((selected, message, options)) or {},
    )

    scheduler_client.maintenance(
        tmp_path / "control",
        tmp_path / "BIDS",
        checkout=tmp_path / "checkout",
        operation=operation,
        projects=["demo"],
        execute=False,
        version="1.2.3",
    )

    assert calls[0][0] is endpoint
    assert calls[0][1]["operation"] == operation
    assert calls[0][2]["require_service"] is True
    assert calls[0][2]["start_epoch"] is True
    assert calls[0][2]["timeout"] is None


@pytest.mark.parametrize(
    ("method", "arguments", "expected"),
    (
        ("heartbeat_worker", ("worker",), None),
        ("worker_shutdown_requested", ("worker",), False),
        ("attempt_cancel_requested", (17,), False),
        ("attempt_summary", (17,), "attempt state unavailable while the scheduler is busy"),
        ("required_memory_above", (32,), None),
    ),
)
def test_worker_polls_survive_scheduler_transport_delays(
    monkeypatch, capsys, method, arguments, expected
) -> None:
    client = object.__new__(WorkerSchedulerClient)
    client._last_poll_warning = 0.0

    def delayed(*_args, **_kwargs):
        raise scheduler_client.SchedulerError("Central scheduler did not respond within 60 seconds")

    monkeypatch.setattr(client, "_call", delayed)
    keywords = {"state": "running"} if method == "heartbeat_worker" else {}

    assert getattr(client, method)(*arguments, **keywords) == expected
    assert "scheduler poll" in capsys.readouterr().err


def test_worker_poll_preserves_remote_errors(monkeypatch) -> None:
    client = object.__new__(WorkerSchedulerClient)
    client._last_poll_warning = 0.0

    def rejected(*_args, **_kwargs):
        raise scheduler_client.SchedulerError("Worker token is obsolete", error_type="ValueError")

    monkeypatch.setattr(client, "_call", rejected)

    with pytest.raises(scheduler_client.SchedulerError, match="token is obsolete"):
        client.worker_shutdown_requested("worker")


def test_worker_graph_signature_uses_durable_rpc(monkeypatch) -> None:
    client = object.__new__(WorkerSchedulerClient)
    client.endpoint = SimpleNamespace()
    client.worker_id = "worker"
    client.token = "token"
    client.sequence = 0
    calls = []
    monkeypatch.setattr(
        "nro.orchestration.worker_client.exchange",
        lambda _endpoint, message, **options: calls.append((message, options)) or "signature",
    )

    assert client.runner_graph_signature(17) == "signature"
    assert calls[0][0]["action"] == "runner_graph_signature"
    assert calls[0][0]["work_item_id"] == 17
    assert calls[0][1]["durable"] is True
    assert calls[0][1]["require_service"] is True
    assert calls[0][1]["timeout"] is None


def test_worker_resource_step_claim_uses_bound_worker_identity(monkeypatch) -> None:
    client = object.__new__(WorkerSchedulerClient)
    client.worker_id = "gpu-worker"
    calls = []
    monkeypatch.setattr(
        client,
        "_call",
        lambda action, **fields: calls.append((action, fields)),
    )

    assert client.claim_resource_step("gpu-worker", resource_class="gpu", memory_gb=32) is None
    assert calls == [("claim_resource_step", {"resource_class": "gpu", "memory_gb": 32})]
    with pytest.raises(ValueError, match="identity differs"):
        client.claim_resource_step("other-worker", resource_class="gpu", memory_gb=32)


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


def test_controller_startup_error_is_scoped_to_launch_token(tmp_path):
    control = tmp_path / ".nro"
    scheduler_bus.prepare(control)
    scheduler_bus.publish_startup_error(control, "current", "database is locked")

    assert scheduler_bus.read_startup_error(control, "current")["error"] == "database is locked"
    assert scheduler_bus.read_startup_error(control, "other") is None


def test_controller_startup_progress_is_scoped_and_cleared_on_activation(tmp_path):
    control = tmp_path / ".nro"
    scheduler_bus.prepare(control)
    claim = scheduler_bus.claim_launch(control)
    assert claim is not None
    scheduler_bus.publish_startup_progress(control, claim.token, "Opening scheduler registry")

    record = scheduler_bus.read_startup_progress(control, claim.token)
    assert record is not None
    assert record["phase"] == "Opening scheduler registry"
    assert scheduler_bus.read_startup_progress(control, "other") is None

    scheduler_bus.activate(control, claim.token, 1, host="scheduler.example", port=23001)
    assert scheduler_bus.read_startup_progress(control, claim.token) is None


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


def test_component_status_snapshot_reuses_static_graph(tmp_path):
    control = tmp_path / ".nro"
    scheduler_bus.prepare(control)

    def snapshot(generation, status, *, participant="01"):
        return {
            "protocol": scheduler_bus.PROTOCOL,
            "generation": generation,
            "published_at": f"generation-{generation}",
            "service_active": True,
            "workers": [{"id": "worker", "state": status.lower()}],
            "submissions": [],
            "branches": {
                "main": {
                    "rows": [
                        {
                            "id": 1,
                            "project": "demo",
                            "participant": participant,
                            "module": "anat",
                            "status": status,
                            "artifact_state": "fresh" if status == "Success" else "missing",
                        }
                    ],
                    "visible_ids": [1],
                    "ingestion": [],
                    "dependencies": [],
                }
            },
        }

    first = snapshot(4, "Running")
    scheduler_bus.publish_snapshot(control, first)
    assert scheduler_bus.read_snapshot(control) == first
    static_paths = tuple(ControlPaths(control).service.glob("status-static-*.json"))
    assert len(static_paths) == 1

    second = snapshot(5, "Success")
    scheduler_bus.publish_snapshot(control, second)
    assert scheduler_bus.read_snapshot(control) == second
    assert tuple(ControlPaths(control).service.glob("status-static-*.json")) == static_paths

    changed = snapshot(6, "Success", participant="02")
    scheduler_bus.publish_snapshot(control, changed)
    assert scheduler_bus.read_snapshot(control) == changed
    assert len(tuple(ControlPaths(control).service.glob("status-static-*.json"))) == 2


def test_component_status_snapshot_survives_repeated_publication_races(tmp_path, monkeypatch):
    control = tmp_path / ".nro"
    expected = {
        "protocol": scheduler_bus.PROTOCOL,
        "generation": 1,
        "published_at": "now",
        "service_active": True,
        "workers": [],
        "submissions": [],
        "branches": {},
    }
    scheduler_bus.publish_snapshot(control, expected)
    assemble = scheduler_bus._assemble_status_snapshot
    attempts = 0

    def racing(*args):
        nonlocal attempts
        attempts += 1
        if attempts < 9:
            raise ValueError("publication advanced")
        return assemble(*args)

    monkeypatch.setattr(scheduler_bus, "_assemble_status_snapshot", racing)

    assert scheduler_bus.read_snapshot(control) == expected
    assert attempts == 9


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
