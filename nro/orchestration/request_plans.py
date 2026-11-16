"""Retain complete plans only while requests may need reconciliation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib

_COMPRESSED = b"NROZ1\0"
_COMPRESSION_THRESHOLD = 4096
# An active request's plan can hold tens of thousands of specifications.
# Decoding and re-encoding it on every maintenance pass held the registry
# lock for tens of seconds, although stored plans never change. These caches
# are keyed by a digest of the stored value, so they stay correct for any
# registry and need no invalidation.
_PLAN_CACHE_LIMIT = 64
_CANONICAL_PLANS: dict[tuple[bool, bytes], None] = {}
_PLAN_SUMMARIES: dict[bytes, dict] = {}


def _plan_digest(value: str | bytes) -> bytes:
    data = value if isinstance(value, bytes) else value.encode()
    return hashlib.blake2b(data, digest_size=16).digest()


def _remember(cache: dict, key, value) -> None:
    if len(cache) >= _PLAN_CACHE_LIMIT:
        cache.clear()
    cache[key] = value


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


def plan_reconciliation_summary(value: str | bytes) -> dict:
    """Return the fields branch reconciliation reads before deciding to replay.

    The result has ``has_specifications``, ``branch``, ``registry_id``, and
    ``inherited``. Callers must treat it as read-only.
    """
    digest = _plan_digest(value)
    summary = _PLAN_SUMMARIES.get(digest)
    if summary is None:
        payload = decode_plan(value)
        summary = {
            "has_specifications": "specifications" in payload,
            "branch": payload.get("branch"),
            "registry_id": payload.get("registry_id"),
            "inherited": tuple(payload.get("inherited", ())),
        }
        _remember(_PLAN_SUMMARIES, digest, summary)
    return summary


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
            active = row["state"] == "active"
            stored = row["payload_json"]
            if (active, _plan_digest(stored)) in _CANONICAL_PLANS:
                continue
            payload = decode_plan(stored)
            if active and payload.get("inherited"):
                encoded = encode_plan(payload)
            else:
                encoded = terminal_plan(payload)
            # Both encodings are deterministic, so the stored or rewritten
            # value is canonical for this request state from now on.
            _remember(_CANONICAL_PLANS, (active, _plan_digest(encoded)), None)
            if encoded != stored:
                updates.append((encoded, row["request_id"]))
        database.executemany(
            "UPDATE request_plans SET payload_json=? WHERE request_id=?",
            updates,
        )
        changed += len(updates)
    return changed
