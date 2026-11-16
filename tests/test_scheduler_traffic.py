import time

import pytest

from nro.orchestration import scheduler_bus, scheduler_client, scheduler_rpc, scheduler_service
from nro.orchestration.worker_client import WorkerSchedulerClient


def _heartbeat(worker_id: str) -> dict:
    return scheduler_bus.create_message(
        {"operation": "worker", "action": "heartbeat", "worker_id": worker_id},
        kind="worker",
    )


def _poll(action: str) -> dict:
    return scheduler_bus.create_message(
        {"operation": "worker", "action": action, "worker_id": "worker"},
        kind="worker",
    )


def _envelope(**extra) -> dict:
    return {
        "protocol": scheduler_bus.PROTOCOL,
        "token": "token",
        "generation": 1,
        "durable": False,
        "record": _heartbeat("worker"),
        **extra,
    }


def test_request_expiry_is_limited_to_non_durable_requests():
    record, durable = scheduler_rpc.validate_request(_envelope(expires_at=1.0), token="token")
    assert durable is False and record["payload"]["action"] == "heartbeat"
    with pytest.raises(ValueError, match="non-durable"):
        scheduler_rpc.validate_request(_envelope(durable=True, expires_at=1.0), token="token")
    with pytest.raises(ValueError, match="non-durable"):
        scheduler_rpc.validate_request(_envelope(expires_at="soon"), token="token")


def test_expired_poll_skips_the_registry(monkeypatch):
    def unreachable(*_args, **_kwargs):
        raise AssertionError("an expired poll must not be served")

    monkeypatch.setattr(scheduler_service, "_message_response", unreachable)
    response = scheduler_service._nondurable_response(
        object(), _heartbeat("worker"), values={}, expires_at=time.time() - 1
    )
    assert response["error_type"] == "RequestExpired"


def test_live_poll_is_served(monkeypatch):
    monkeypatch.setattr(
        scheduler_service, "_message_response", lambda *_args, **_kwargs: {"result": "ok"}
    )
    response = scheduler_service._nondurable_response(
        object(), _poll("shutdown_requested"), values={}, expires_at=time.time() + 60
    )
    assert response == {"result": "ok"}


def test_checkin_batcher_answers_expired_events_without_applying_them(monkeypatch):
    batches = []

    def apply(_registry, messages):
        batches.append([message["worker_id"] for message in messages])
        return [{"result": message["worker_id"]} for message in messages]

    monkeypatch.setattr(scheduler_service, "_apply_worker_checkin_batch", apply)
    batcher = scheduler_service._WorkerCheckinBatcher(object(), interval=0.1)
    try:
        expired = batcher.submit(_heartbeat("late"), expires_at=time.time() - 1)
        live = batcher.submit(_heartbeat("live"), expires_at=time.time() + 60)
        unbounded = batcher.submit(_heartbeat("unbounded"))
        assert expired.result(timeout=1)["error_type"] == "RequestExpired"
        assert live.result(timeout=1) == {"result": "live"}
        assert unbounded.result(timeout=1) == {"result": "unbounded"}
    finally:
        batcher.close()
    assert batches == [["live", "unbounded"]]


def test_worker_poll_treats_expiry_as_a_delay(monkeypatch, capsys):
    client = object.__new__(WorkerSchedulerClient)
    client._last_poll_warning = 0.0

    def expired(*_args, **_kwargs):
        raise scheduler_client.SchedulerError("late", error_type="RequestExpired")

    monkeypatch.setattr(client, "_call", expired)
    assert client.worker_shutdown_requested("worker") is False
    assert "scheduler poll" in capsys.readouterr().err


def test_worker_poll_still_raises_service_errors(monkeypatch):
    client = object.__new__(WorkerSchedulerClient)
    client._last_poll_warning = 0.0

    def failed(*_args, **_kwargs):
        raise scheduler_client.SchedulerError("fenced", error_type="ValueError")

    monkeypatch.setattr(client, "_call", failed)
    with pytest.raises(scheduler_client.SchedulerError):
        client.worker_shutdown_requested("worker")


def test_batched_checkin_uses_the_checkin_logic(monkeypatch):
    applied = []

    def apply(_registry, messages):
        applied.append(messages)
        return [{"result": {"shutdown_requested": False}}]

    monkeypatch.setattr(scheduler_service, "_apply_worker_checkin_batch", apply)
    response = scheduler_service._nondurable_response(
        object(), _poll("check_in"), values={}, expires_at=time.time() + 60
    )
    assert response == {"result": {"shutdown_requested": False}}
    assert applied[0][0]["action"] == "check_in"
