"""Retain complete plans only while requests may need reconciliation."""

from __future__ import annotations

import json
import sqlite3


def terminal_plan(payload: dict) -> str:
    """Encode the terminal identities needed for later publication."""
    terminals = payload.get("terminals")
    if not isinstance(terminals, list) or not all(isinstance(value, str) for value in terminals):
        raise ValueError("Stored request plan has invalid terminal identities")
    return json.dumps({"terminals": terminals}, separators=(",", ":"), sort_keys=True)


def compact_terminal_plans(database: sqlite3.Connection) -> int:
    """Discard execution recipes from requests that can no longer be reconciled."""
    changed = 0
    rows = database.execute(
        """SELECT plan.request_id,plan.payload_json
           FROM request_plans plan JOIN requests request ON request.id=plan.request_id
           WHERE request.state!='active'"""
    ).fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"])
        encoded = terminal_plan(payload)
        if encoded == row["payload_json"]:
            continue
        database.execute(
            "UPDATE request_plans SET payload_json=? WHERE request_id=?",
            (encoded, row["request_id"]),
        )
        changed += 1
    return changed
