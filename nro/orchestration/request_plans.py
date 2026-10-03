"""Retain complete plans only while requests may need reconciliation."""

from __future__ import annotations

import json
import sqlite3
import zlib

_COMPRESSED = b"NROZ1\0"
_COMPRESSION_THRESHOLD = 4096


def decode_plan(value: str | bytes) -> dict:
    """Decode a legacy JSON or compressed request reconciliation plan."""
    if isinstance(value, bytes) and value.startswith(_COMPRESSED):
        value = zlib.decompress(value[len(_COMPRESSED) :])
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Stored request plan must be a mapping")
    return payload


def encode_plan(payload: dict) -> str | bytes:
    """Encode large machine-only plans compactly without changing their meaning."""
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if len(encoded) < _COMPRESSION_THRESHOLD:
        return encoded.decode()
    return _COMPRESSED + zlib.compress(encoded, level=6)


def terminal_plan(payload: dict) -> str | bytes:
    """Encode the terminal identities needed for later publication."""
    terminals = payload.get("terminals")
    if not isinstance(terminals, list) or not all(isinstance(value, str) for value in terminals):
        raise ValueError("Stored request plan has invalid terminal identities")
    return encode_plan({"terminals": terminals})


def compact_terminal_plans(database: sqlite3.Connection) -> int:
    """Discard execution recipes that cannot require inheritance reconciliation.

    Terminal requests retain only their selected terminal identities for later
    publication. Active requests need the complete recipe only while they use
    an inherited artifact that may have to be replaced locally.
    """
    changed = 0
    cursor = database.execute(
        """SELECT plan.request_id,plan.payload_json,request.state
           FROM request_plans plan JOIN requests request ON request.id=plan.request_id
           WHERE request.state='active'
              OR CASE WHEN typeof(plan.payload_json)='text' THEN NOT (
                     json_valid(plan.payload_json)
                     AND json_type(plan.payload_json,'$.terminals')='array'
                     AND json_remove(plan.payload_json,'$.terminals')='{}'
                 ) ELSE 1 END"""
    )
    while rows := cursor.fetchmany(64):
        updates = []
        for row in rows:
            payload = decode_plan(row["payload_json"])
            if row["state"] == "active" and payload.get("inherited"):
                encoded = encode_plan(payload)
            else:
                encoded = terminal_plan(payload)
            if encoded != row["payload_json"]:
                updates.append((encoded, row["request_id"]))
        database.executemany(
            "UPDATE request_plans SET payload_json=? WHERE request_id=?",
            updates,
        )
        changed += len(updates)
    return changed
