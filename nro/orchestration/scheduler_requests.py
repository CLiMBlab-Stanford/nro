"""Persist retryable scheduler requests without using a filesystem mailbox."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Executor, Future
from typing import Callable

from nro.configuration.store import fingerprint
from nro.orchestration.registry import utcnow
from nro.orchestration.scheduler_bus import validate_message


def _encoded(record: dict) -> tuple[str, str]:
    payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
    return payload, fingerprint(record)


def prepare(registry) -> tuple[dict, ...]:
    """Recover interrupted requests and return records awaiting execution."""
    with registry.connection(write=True) as db:
        from nro.orchestration.request_plans import compact_terminal_plans

        compact_terminal_plans(db)
        db.execute(
            "UPDATE scheduler_requests SET state='pending',updated_at=? WHERE state='running'",
            (utcnow(),),
        )
        db.execute(
            "DELETE FROM scheduler_requests "
            "WHERE state='completed' AND datetime(updated_at) < datetime('now','-1 hour')"
        )
        rows = db.execute(
            "SELECT record_json FROM scheduler_requests WHERE state='pending' ORDER BY created_at,id"
        ).fetchall()
    return tuple(validate_message(json.loads(row[0])) for row in rows)


def register(registry, record: dict) -> dict | None:
    """Register one request or return its previously committed response."""
    encoded, digest = _encoded(record)
    now = utcnow()
    with registry.connection() as db:
        row = db.execute(
            "SELECT record_fingerprint,state,response_json FROM scheduler_requests WHERE id=?",
            (record["id"],),
        ).fetchone()
    if row is None:
        with registry.connection(write=True) as db:
            row = db.execute(
                "SELECT record_fingerprint,state,response_json FROM scheduler_requests WHERE id=?",
                (record["id"],),
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO scheduler_requests "
                    "(id,kind,record_fingerprint,record_json,state,response_json,created_at,updated_at) "
                    "VALUES (?,?,?,?, 'pending',NULL,?,?)",
                    (record["id"], record["kind"], digest, encoded, now, now),
                )
                return None
    if row["record_fingerprint"] != digest:
        raise ValueError("Scheduler request identity was reused with different content")
    if row["state"] == "completed":
        return json.loads(row["response_json"])
    return None


def _execute(registry, record: dict, operation: Callable[[dict], dict]) -> dict:
    """Claim, execute, and commit one durable request."""
    completed = register(registry, record)
    if completed is not None:
        return completed
    with registry.connection(write=True) as db:
        row = db.execute(
            "SELECT state,response_json FROM scheduler_requests WHERE id=?", (record["id"],)
        ).fetchone()
        if row["state"] == "completed":
            return json.loads(row["response_json"])
        db.execute(
            "UPDATE scheduler_requests SET state='running',updated_at=? WHERE id=?",
            (utcnow(), record["id"]),
        )
    try:
        response = operation(record)
    except BaseException as error:
        response = {"error": str(error), "error_type": type(error).__name__}
    encoded = json.dumps(response, separators=(",", ":"), sort_keys=True)
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE scheduler_requests SET state='completed',response_json=?,updated_at=? WHERE id=?",
            (encoded, utcnow(), record["id"]),
        )
        if record["kind"] == "command":
            db.execute(
                "DELETE FROM scheduler_requests "
                "WHERE state='completed' AND datetime(updated_at) < datetime('now','-1 hour')"
            )
    from nro.orchestration.scheduler_bus import clear_progress

    clear_progress(registry.paths.control, str(record["id"]))
    return response


class RequestCoordinator:
    """Deduplicate durable requests and route each identity to one executor."""

    def __init__(self, registry, operation: Callable[[dict], dict]) -> None:
        """Bind the registry and validated request operation."""
        self.registry = registry
        self.operation = operation
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}

    @staticmethod
    def completed(response: dict) -> Future:
        """Return a resolved future containing a committed response."""
        future: Future = Future()
        future.set_result(response)
        return future

    def submit(self, record: dict, executor: Executor) -> Future:
        """Return the existing request future or schedule its first execution."""
        identifier = str(record["id"])
        with self._lock:
            existing = self._inflight.get(identifier)
            if existing is not None:
                return existing
        response = register(self.registry, record)
        if response is not None:
            return self.completed(response)
        with self._lock:
            existing = self._inflight.get(identifier)
            if existing is not None:
                return existing
            future = executor.submit(_execute, self.registry, record, self.operation)
            self._inflight[identifier] = future

            def release(_future: Future) -> None:
                with self._lock:
                    if self._inflight.get(identifier) is _future:
                        self._inflight.pop(identifier, None)

            future.add_done_callback(release)
            return future

    def run(self, record: dict) -> dict:
        """Execute one request synchronously for a fenced one-shot coordinator."""
        return _execute(self.registry, record, self.operation)
