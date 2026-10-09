"""Remove obsolete pycicada paths from historical ownership receipts."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import atomic_write_json
from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.ownership import OWNERSHIP_VERSION, ownership_record_fingerprint

HOTFIX_ID = "v0303-obsolete-pycicada-receipts"
SUMMARY = "Remove the retired pycicada site path from historical ownership receipts."
_OBSOLETE_REFERENCE = "nro-site:pycicada:."


@dataclass(frozen=True)
class _Candidate:
    path: Path
    payload: dict[str, object]


def _candidate(path: Path) -> _Candidate | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("owner") != "nro" or payload.get("record_version") != OWNERSHIP_VERSION:
        return None
    if payload.get("record_fingerprint") != ownership_record_fingerprint(payload):
        return None
    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        return None
    runtime = execution.get("runtime_configuration")
    if not isinstance(runtime, Mapping) or runtime.get("cicada_cmd") != _OBSOLETE_REFERENCE:
        return None
    return _Candidate(path, payload)


def _candidates(bids_root: Path, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    found: list[_Candidate] = []
    for project in projects:
        ownership = bids_root / project / "derivatives/nro"
        pattern = "*/*/.nro/work_items/*/*.json"
        for path in sorted(ownership.glob(pattern)):
            candidate = _candidate(path)
            if candidate is not None:
                found.append(candidate)
    return tuple(found)


def _repair(candidate: _Candidate) -> None:
    payload = dict(candidate.payload)
    execution = dict(payload["execution"])
    runtime = dict(execution["runtime_configuration"])
    del runtime["cicada_cmd"]
    execution["runtime_configuration"] = runtime
    payload["execution"] = execution
    payload["record_fingerprint"] = ownership_record_fingerprint(payload)
    mode = stat.S_IMODE(candidate.path.stat().st_mode)
    atomic_write_json(candidate.path, payload, sort_keys=True, mode=mode, durable=True)


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Repair intact receipts containing only the retired pycicada command path."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    candidates = _candidates(registry.paths.bids_root, selected)
    if execute:
        for candidate in candidates:
            _repair(candidate)
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=tuple(candidate.path for candidate in candidates),
        records=len(candidates),
        applied=execute,
    )
