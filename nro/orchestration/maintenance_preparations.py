"""Persist expensive maintenance previews until execution or cancellation."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from nro.engine.io import atomic_write_json
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryLock, ensure_shared_directory, utcnow

FORMAT = 1
_TOKEN = re.compile(r"[0-9a-f]{32}")


def _path(control: Path, token: str) -> Path:
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ValueError("Invalid maintenance preparation token")
    return ControlPaths(control).maintenance_preparations / f"{token}.json"


def _lock(control: Path) -> RegistryLock:
    paths = ControlPaths(control)
    ensure_shared_directory(paths.shared)
    return RegistryLock(
        paths.shared / "maintenance-preparations.lock",
        paths.shared / "maintenance-preparations.recovery-lock",
        timeout=120,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            "Prepared maintenance preview is unavailable; run the preview again"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid maintenance preparation: {path}") from error
    required = {"format", "token", "kind", "scope", "created_at", "payload"}
    if (
        not isinstance(record, dict)
        or set(record) != required
        or record["format"] != FORMAT
        or not isinstance(record["kind"], str)
        or not isinstance(record["scope"], str)
        or not isinstance(record["created_at"], str)
        or not isinstance(record["payload"], dict)
        or record["token"] != path.stem
    ):
        raise ValueError(f"Invalid maintenance preparation: {path}")
    return record


def retain(control: Path, *, kind: str, scope: str, payload: dict[str, Any]) -> str:
    """Persist one preparation and supersede older previews for the same scope."""
    if not kind or not scope or not isinstance(payload, dict):
        raise ValueError("Maintenance preparation kind, scope, and payload are required")
    paths = ControlPaths(control)
    root = paths.maintenance_preparations
    token = uuid.uuid4().hex
    with _lock(control):
        ensure_shared_directory(root)
        destination = _path(control, token)
        atomic_write_json(
            destination,
            {
                "format": FORMAT,
                "token": token,
                "kind": kind,
                "scope": scope,
                "created_at": utcnow(),
                "payload": payload,
            },
            sort_keys=True,
            mode=0o660,
            durable=True,
        )
        for candidate in root.glob("*.json"):
            if candidate == destination:
                continue
            try:
                record = _read(candidate)
            except ValueError:
                continue
            if record["kind"] == kind and record["scope"] == scope:
                candidate.unlink(missing_ok=True)
    return token


def load(control: Path, *, kind: str, token: str) -> dict[str, Any]:
    """Load a retained preparation without consuming it."""
    with _lock(control):
        record = _read(_path(control, token))
        if record["kind"] != kind:
            raise ValueError("Prepared maintenance preview does not match this operation")
        return record["payload"]


def discard(control: Path, *, kind: str, token: str) -> None:
    """Delete one matching preparation after success or explicit cancellation."""
    with _lock(control):
        path = _path(control, token)
        record = _read(path)
        if record["kind"] != kind:
            raise ValueError("Prepared maintenance preview does not match this operation")
        path.unlink()
