"""Exchange framed scheduler requests over a direct TCP connection."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAX_MESSAGE_BYTES = 128 * 1024 * 1024
_HEADER = struct.Struct("!Q")


def open_listener() -> socket.socket:
    """Bind a reusable TCP listener and return it in nonblocking mode."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", 0))
    listener.listen(128)
    listener.setblocking(False)
    return listener


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Scheduler connection closed during a message")
        chunks.extend(chunk)
    return bytes(chunks)


def receive(connection: socket.socket) -> dict[str, Any]:
    """Read one bounded, length-prefixed JSON object."""
    size = _HEADER.unpack(_receive_exact(connection, _HEADER.size))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("Scheduler message exceeds the transport limit")
    value = json.loads(_receive_exact(connection, size))
    if not isinstance(value, dict):
        raise ValueError("Scheduler message must be a JSON object")
    return value


def send(connection: socket.socket, value: dict[str, Any]) -> None:
    """Write one bounded, length-prefixed JSON object."""
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Scheduler message exceeds the transport limit")
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def request(
    active: dict[str, Any],
    record: dict[str, Any],
    *,
    timeout: float,
    durable: bool,
) -> dict:
    """Send one request to the active scheduler and return its response."""
    address = (str(active["host"]), int(active["port"]))
    with socket.create_connection(address, timeout=timeout) as connection:
        connection.settimeout(timeout)
        send(
            connection,
            {
                "protocol": int(active["protocol"]),
                "token": str(active["token"]),
                "generation": int(active["generation"]),
                "durable": bool(durable),
                "record": record,
            },
        )
        return receive(connection)


def validate_request(envelope: dict[str, Any], *, token: str) -> tuple[dict[str, Any], bool]:
    """Validate one direct request against the active scheduler incarnation."""
    from nro.orchestration.scheduler_bus import PROTOCOL, validate_message

    if (
        envelope.get("protocol") != PROTOCOL
        or envelope.get("token") != token
        or not isinstance(envelope.get("generation"), int)
        or not isinstance(envelope.get("durable"), bool)
    ):
        raise ValueError("Scheduler endpoint is obsolete")
    return validate_message(envelope.get("record")), envelope["durable"]
