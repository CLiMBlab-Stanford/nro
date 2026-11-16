"""Durable worker requests share one registry transaction per batch."""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path

import pytest

from nro.orchestration import scheduler_bus, scheduler_service
from nro.orchestration.registry import Registry
from nro.orchestration.scheduler_requests import RequestCoordinator


def _record(action: str, **payload) -> dict:
    return scheduler_bus.create_message(
        {"operation": "worker", "action": action, "worker_id": "worker", **payload},
        kind="worker",
    )


@contextlib.contextmanager
def _session(tmp_path: Path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()
    with registry.scheduler_session() as connection:
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        try:
            yield registry, statements
        finally:
            connection.set_trace_callback(None)


def _commits(statements: list[str]) -> int:
    return sum(1 for statement in statements if statement.strip().upper() == "COMMIT")


def test_batched_requests_register_apply_and_respond_in_one_commit(tmp_path: Path) -> None:
    release = threading.Event()

    def operation(record: dict) -> dict:
        release.wait(5)
        with registry.connection(write=True) as database:
            database.execute(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                (f"applied:{record['id']}", "1"),
            )
        return {"result": record["payload"]["action"]}

    with _session(tmp_path) as (registry, statements):
        batcher = scheduler_service._DurableBatcher(registry, interval=0.2)
        coordinator = RequestCoordinator(registry, operation)
        try:
            records = [_record("attempt_process", attempt_id=index) for index in range(5)]
            statements.clear()
            futures = [coordinator.submit(record, batcher) for record in records]
            release.set()
            assert [future.result(timeout=5) for future in futures] == [
                {"result": "attempt_process"}
            ] * 5
        finally:
            batcher.shutdown()
        assert _commits(statements) == 1
        with registry.connection() as database:
            states = database.execute(
                "SELECT state, response_json FROM scheduler_requests ORDER BY id"
            ).fetchall()
            applied = database.execute(
                "SELECT COUNT(*) FROM metadata WHERE key LIKE 'applied:%'"
            ).fetchone()[0]
        assert [row["state"] for row in states] == ["completed"] * 5
        assert applied == 5


def test_failed_request_does_not_undo_its_batch(tmp_path: Path) -> None:
    def operation(record: dict) -> dict:
        if record["payload"]["attempt_id"] == 1:
            with registry.connection(write=True) as database:
                database.execute("INSERT INTO metadata(key,value) VALUES ('partial','1')")
                raise ValueError("rejected")
        with registry.connection(write=True) as database:
            database.execute(
                "INSERT INTO metadata(key,value) VALUES (?, '1')",
                (f"applied:{record['payload']['attempt_id']}",),
            )
        return {"result": "ok"}

    with _session(tmp_path) as (registry, _statements):
        batcher = scheduler_service._DurableBatcher(registry, interval=0.2)
        coordinator = RequestCoordinator(registry, operation)
        records = [_record("attempt_process", attempt_id=index) for index in range(3)]
        try:
            futures = [coordinator.submit(record, batcher) for record in records]
            results = [future.result(timeout=5) for future in futures]
        finally:
            batcher.shutdown()
        assert results[0] == results[2] == {"result": "ok"}
        assert results[1] == {"error": "rejected", "error_type": "ValueError"}
        with registry.connection() as database:
            keys = {row[0] for row in database.execute("SELECT key FROM metadata")}
            row = database.execute(
                "SELECT state, response_json FROM scheduler_requests WHERE id=?",
                (records[1]["id"],),
            ).fetchone()
        assert {"applied:0", "applied:2"} <= keys
        assert "partial" not in keys
        assert row["state"] == "completed"
        assert json.loads(row["response_json"])["error_type"] == "ValueError"


def test_failed_commit_fails_every_request_in_the_batch() -> None:
    class FailingRegistry:
        @contextlib.contextmanager
        def connection(self, *, write: bool = False):
            yield None
            raise OSError("disk full")

    batcher = scheduler_service._DurableBatcher(FailingRegistry(), interval=0.2)
    try:
        futures = [batcher.submit(lambda value=value: value) for value in range(3)]
        for future in futures:
            with pytest.raises(OSError, match="disk full"):
                future.result(timeout=5)
    finally:
        batcher.shutdown()


def test_batch_stops_taking_requests_after_its_budget() -> None:
    batches: list[int] = []

    class CountingRegistry:
        @contextlib.contextmanager
        def connection(self, *, write: bool = False):
            batches.append(0)
            yield None

    batcher = scheduler_service._DurableBatcher(CountingRegistry(), interval=0.2, budget=0.0)
    try:
        futures = [batcher.submit(lambda value=value: value) for value in range(3)]
        assert [future.result(timeout=5) for future in futures] == [0, 1, 2]
    finally:
        batcher.shutdown()
    assert len(batches) == 3


@pytest.mark.parametrize(
    ("payload", "batched"),
    (
        ({"operation": "worker", "action": "claim"}, True),
        ({"operation": "worker", "action": "finish_attempt"}, True),
        ({"operation": "worker", "action": "request_capacity", "kind": "expand"}, True),
        ({"operation": "worker", "action": "request_capacity", "kind": "successor"}, False),
        ({"operation": "worker", "action": "record_completion"}, False),
        ({"operation": "worker", "action": "claim_ingestion"}, False),
    ),
)
def test_only_short_worker_requests_are_batched(payload, batched) -> None:
    executors = {name: object() for name in ("worker", "poll", "completion", "batch")}
    chosen = scheduler_service._executor_for({"payload": payload}, executors)
    assert (chosen is executors["batch"]) is batched
    poll = scheduler_service._executor_for({"payload": payload}, executors, durable=False)
    assert poll is executors["poll"]


@pytest.mark.parametrize(
    ("action", "batched"),
    (("shutdown_requested", True), ("check_in", True), ("required_memory", False)),
)
def test_worker_polls_share_the_batch_thread(action, batched) -> None:
    import socket
    from concurrent.futures import Future

    from nro.orchestration import scheduler_rpc

    calls = []

    class Recording:
        def submit(self, function, *args, **kwargs):
            calls.append((self, function, kwargs.get("expires_at")))
            future: Future = Future()
            future.set_result({"result": "served"})
            return future

    executors = {name: Recording() for name in ("worker", "poll", "completion", "batch")}

    class Checkins:
        def submit(self, record, *, expires_at=None):
            return Recording().submit(None, expires_at=expires_at)

    server, client = socket.socketpair()
    try:
        scheduler_service._dispatch_connection(
            server,
            _record(action),
            False,
            object(),
            values={},
            coordinator=None,
            executors=executors,
            checkins=Checkins(),
            changed=scheduler_service._ActivitySignal(),
            expires_at=123.0,
        )
        assert scheduler_rpc.receive(client) == {"result": "served"}
    finally:
        client.close()
    executor, function, expires_at = calls[0]
    assert (executor is executors["batch"]) is batched
    assert expires_at == 123.0
    if batched:
        assert function is scheduler_service._nondurable_response
