"""Remove obsolete ownership receipts shadowed by canonical replacements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nro.engine.io import sync_directory
from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.ownership import OWNERSHIP_VERSION, ownership_record_fingerprint
from nro.orchestration.planning_context import work_item_key

HOTFIX_ID = "v0303-duplicate-ownership-receipts"
SUMMARY = "Remove invalid ownership receipts shadowed by canonical replacements."
_IDENTITY_FIELDS = (
    "project",
    "module",
    "lineage_fingerprint",
    "participant",
    "entities",
    "directory_label",
)


@dataclass(frozen=True)
class _Candidate:
    obsolete: Path
    canonical: Path


def _document(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_file():
        return None
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
    return payload


def _expected_key(record: Mapping[str, object]) -> str | None:
    entities = record.get("entities")
    required = ("project", "module", "lineage_fingerprint", "participant")
    if not isinstance(entities, Mapping) or any(
        not isinstance(record.get(key), str) for key in required
    ):
        return None
    return work_item_key(
        str(record["project"]),
        str(record["module"]),
        str(record["lineage_fingerprint"]),
        str(record["participant"]),
        {str(key): str(value) for key, value in entities.items()},
    )


def _same_claim(obsolete: Mapping[str, object], canonical: Mapping[str, object]) -> bool:
    if any(obsolete.get(field) != canonical.get(field) for field in _IDENTITY_FIELDS):
        return False
    old_contract = obsolete.get("artifact_contract")
    new_contract = canonical.get("artifact_contract")
    if not isinstance(old_contract, Mapping) or not isinstance(new_contract, Mapping):
        return False
    return old_contract.get("output") == new_contract.get("output")


def _candidate(path: Path) -> _Candidate | None:
    obsolete = _document(path)
    if obsolete is None:
        return None
    expected = _expected_key(obsolete)
    stored = obsolete.get("work_item_key")
    if not isinstance(stored, str) or stored == expected or ":" not in stored or expected is None:
        return None
    if path.stem != stored.split(":", 1)[1]:
        return None
    canonical_path = path.with_name(expected.split(":", 1)[1] + ".json")
    canonical = _document(canonical_path)
    if canonical is None or canonical.get("work_item_key") != expected:
        return None
    if canonical_path.stem != expected.split(":", 1)[1]:
        return None
    if not _same_claim(obsolete, canonical):
        return None
    return _Candidate(path, canonical_path)


def _candidates(bids_root: Path, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    found: list[_Candidate] = []
    for project in projects:
        ownership = bids_root / project / "derivatives/nro"
        for path in sorted(ownership.glob("*/*/.nro/work_items/*/*.json")):
            candidate = _candidate(path)
            if candidate is not None:
                found.append(candidate)
    return tuple(found)


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Delete invalid receipts only when an equivalent canonical receipt exists."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    candidates = _candidates(registry.paths.bids_root, selected)
    if execute:
        for candidate in candidates:
            candidate.obsolete.unlink()
            sync_directory(candidate.obsolete.parent)
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=tuple(candidate.obsolete for candidate in candidates),
        records=len(candidates),
        applied=execute,
    )
