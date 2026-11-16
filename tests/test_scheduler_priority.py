"""Request threads take the scheduler registry lock before background work."""

from __future__ import annotations

import threading
import time

from nro.orchestration import scheduler_bus, scheduler_service
from nro.orchestration.registry import SessionMutex


def _contend(mutex: SessionMutex, names: list[str], order: list[str]) -> list[threading.Thread]:
    def take(name: str) -> None:
        mutex.acquire()
        order.append(name)
        mutex.release()

    threads = [threading.Thread(target=take, args=(name,), name=name) for name in names]
    for thread in threads:
        thread.start()
        time.sleep(0.05)
    return threads


def test_waiting_request_goes_before_background_work() -> None:
    mutex = SessionMutex(defer_seconds=5.0)
    order: list[str] = []
    mutex.acquire()
    threads = _contend(mutex, ["scheduler-maintenance_0", "scheduler-batch"], order)
    mutex.release()
    for thread in threads:
        thread.join(5)
    assert order == ["scheduler-batch", "scheduler-maintenance_0"]


def test_background_work_is_deferred_only_for_a_bounded_time() -> None:
    mutex = SessionMutex(defer_seconds=0.2)
    order: list[str] = []
    gate = threading.Event()

    def request() -> None:
        mutex.acquire()
        gate.wait(5)
        mutex.release()

    mutex.acquire()
    holder = threading.Thread(target=request, name="scheduler-command_0")
    holder.start()
    time.sleep(0.05)
    started = time.monotonic()
    background = _contend(mutex, ["scheduler-capacity_0"], order)
    mutex.release()
    time.sleep(0.05)
    # The request now holds the lock; background waits for it, not beyond.
    gate.set()
    holder.join(5)
    for thread in background:
        thread.join(5)
    assert order == ["scheduler-capacity_0"]
    assert time.monotonic() - started < 2.0


def test_background_thread_takes_a_free_lock_immediately() -> None:
    mutex = SessionMutex(defer_seconds=5.0)
    acquired = []
    thread = threading.Thread(
        target=lambda: acquired.append(mutex.acquire(timeout=1.0)),
        name="scheduler-reconciliation_0",
    )
    started = time.monotonic()
    thread.start()
    thread.join(5)
    assert acquired == [True] and time.monotonic() - started < 1.0


def test_lock_is_reentrant_and_rejects_foreign_release() -> None:
    mutex = SessionMutex()
    mutex.acquire()
    mutex.acquire()
    errors = []

    def foreign() -> None:
        try:
            mutex.release()
        except RuntimeError as error:
            errors.append(error)

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert errors
    mutex.release()
    assert mutex.held_since is not None
    mutex.release()
    assert mutex.held_since is None


def _call(action: str) -> tuple:
    record = scheduler_bus.create_message(
        {"operation": "worker", "action": action, "worker_id": action}, kind="worker"
    )
    return (lambda *_args: action, (object(), record, None), {}, None)


def test_batch_runs_releases_before_claims(monkeypatch) -> None:
    order: list[str] = []

    class Registry:
        import contextlib

        @contextlib.contextmanager
        def connection(self, *, write: bool = False):
            yield None

    batcher = object.__new__(scheduler_service._DurableBatcher)
    batcher.registry = Registry()
    batcher.budget = 10.0
    batcher._carry = []
    pending = []
    for action in ("claim", "attempt_process", "finish_attempt", "claim", "close"):
        function, args, kwargs, _future = _call(action)

        class Future:
            def __init__(self, action=action):
                self.action = action

            def set_result(self, _result):
                order.append(self.action)

            def set_exception(self, error):
                raise error

        pending.append((function, args, kwargs, Future()))
    batcher._run_batch(pending)
    assert order == ["finish_attempt", "close", "attempt_process", "claim", "claim"]


def test_background_work_is_not_starved_by_a_stream_of_requests() -> None:
    mutex = SessionMutex(defer_seconds=0.2)
    stop = threading.Event()
    got_lock: list[float] = []

    def request_stream(index: int) -> None:
        while not stop.is_set():
            mutex.acquire()
            time.sleep(0.005)
            mutex.release()

    streams = [
        threading.Thread(target=request_stream, args=(index,), name=f"scheduler-worker_{index}")
        for index in range(4)
    ]
    for thread in streams:
        thread.start()
    time.sleep(0.05)
    started = time.monotonic()

    def background() -> None:
        mutex.acquire()
        got_lock.append(time.monotonic() - started)
        mutex.release()

    waiter = threading.Thread(target=background, name="scheduler-maintenance_0")
    waiter.start()
    waiter.join(5)
    stop.set()
    for thread in streams:
        thread.join(5)
    assert got_lock and 0.15 <= got_lock[0] < 2.0
