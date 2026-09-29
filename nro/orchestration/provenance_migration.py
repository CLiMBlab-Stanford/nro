"""Transactional conversion of nro public metadata to portable references."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from nro.configuration.store import fingerprint
from nro.engine.bids import bids_suffix
from nro.engine.io import atomic_write_json, atomic_write_text
from nro.engine.references import (
    configured_reference_roots,
    derivative_dataset_description,
    encode_path_values,
    nro_derivative_root,
    portable_public_payload,
    public_document_roots,
    resolve_path_values,
)
from nro.engine.source_metadata import semantic_metadata_snapshot
from nro.orchestration.artifact_records import file_record
from nro.orchestration.branches import BranchPaths
from nro.orchestration.ownership import (
    LEGACY_OWNERSHIP_VERSION,
    LINEAGE_RECORD_NAME,
    OWNERSHIP_DIRECTORY,
    OWNERSHIP_VERSION,
    convert_legacy_ownership_record,
    ownership_record_fingerprint,
    read_ownership_records,
)

_IMAGING_SIDECAR_SUFFIXES = frozenset(
    {
        "T1w",
        "T2w",
        "PDw",
        "FLAIR",
        "angio",
        "asl",
        "bold",
        "dwi",
        "epi",
        "fieldmap",
        "m0scan",
        "magnitude",
        "magnitude1",
        "magnitude2",
        "phase1",
        "phase2",
        "phasediff",
        "sbref",
    }
)


@dataclass(frozen=True)
class DatasetMigrationReport:
    """Preview or outcome of one coordinated dataset conversion."""

    scanned: int
    changed: tuple[Path, ...]
    contracts: int
    errors: tuple[str, ...]


def _bids_roots(registry, site_values: Mapping[str, object] | None) -> tuple[Path, ...]:
    """Return main and registered branch-owned public BIDS roots."""
    roots = [Path(registry.paths.bids_root)]
    if site_values is None:
        return tuple(roots)
    required = ("bids", "work", "development")
    if not all(str(site_values.get(key, "")).strip() for key in required):
        return tuple(roots)
    from nro.orchestration.branch_store import BranchStore

    try:
        topology = BranchStore(registry.paths.control).read().topology
    except (FileNotFoundError, ValueError):
        return tuple(roots)
    for name in sorted(topology.records):
        if name == "main":
            continue
        paths = BranchPaths(name, *(Path(str(site_values[key])) for key in required))
        roots.append(paths.output_bids)
    return tuple(dict.fromkeys(path.absolute() for path in roots))


def _journal_root(registry) -> Path:
    # Retain the established private location so an interrupted migration from
    # an earlier release cannot be orphaned during this interface expansion.
    return Path(registry.paths.control) / "shared" / "provenance-migrations"


def _is_source_sidecar(path: Path, project_root: Path) -> bool:
    """Return whether a JSON file belongs to raw BIDS rather than derivatives."""
    try:
        relative = path.expanduser().absolute().relative_to(project_root.expanduser().absolute())
    except ValueError:
        return False
    return path.suffix.lower() == ".json" and "derivatives" not in relative.parts


def _is_source_imaging_sidecar(path: Path, project_root: Path) -> bool:
    """Return whether a raw-project JSON file describes an imaging suffix."""
    return _is_source_sidecar(path, project_root) and bids_suffix(path) in _IMAGING_SIDECAR_SUFFIXES


def _contract_dataset_view(contract: Mapping[str, object], project_root: Path) -> dict:
    """Replace source-sidecar identity with resolved semantic metadata."""
    from nro.orchestration.contract_migrations import current_contract_schema

    result = json.loads(json.dumps(contract))
    module = str(result.get("module", ""))
    inputs = [Path(str(value)).expanduser().absolute() for value in result.get("inputs", ())]
    retained = [path for path in inputs if not _is_source_sidecar(path, project_root)]
    images = [path for path in retained if path.name.endswith((".nii", ".nii.gz"))]
    result["inputs"] = sorted({str(path) for path in retained})
    processing = dict(result.get("processing") or {})
    snapshot = semantic_metadata_snapshot(images, module=module)
    if snapshot:
        processing["source_metadata"] = list(snapshot)
    else:
        processing.pop("source_metadata", None)
    if processing:
        result["processing"] = processing
    result["contract_schema"] = current_contract_schema(module)
    return result


def _scientific_contract_dataset_view(contract: Mapping[str, object], project_root: Path) -> dict:
    """Convert the location-independent branch form of a scientific contract."""
    result = json.loads(json.dumps(contract))
    retained = []
    images = []
    for item in result.get("inputs", ()):
        if not isinstance(item, Mapping) or "source" not in item:
            retained.append(item)
            continue
        source = Path(str(item["source"])).expanduser().absolute()
        if _is_source_sidecar(source, project_root):
            continue
        retained.append(item)
        if source.name.endswith((".nii", ".nii.gz")):
            images.append(source)
    result["inputs"] = sorted(retained, key=fingerprint)
    module = str(result.get("module", ""))
    processing = dict(result.get("processing") or {})
    snapshot = semantic_metadata_snapshot(images, module=module)
    if snapshot:
        processing["source_metadata"] = snapshot
    else:
        processing.pop("source_metadata", None)
    if processing:
        result["processing"] = processing
    return result


def _ownership_dataset_view(path: Path, value: object, source_project_root: Path) -> object:
    """Upgrade one ownership receipt to semantic source metadata."""
    if OWNERSHIP_DIRECTORY not in path.parts or path.name == LINEAGE_RECORD_NAME:
        return value
    if not isinstance(value, dict):
        raise ValueError("ownership document is not a mapping")
    derivative_project_root = next(
        parent.parent.parent
        for parent in path.parents
        if parent.name == "nro" and parent.parent.name == "derivatives"
    )
    roots = configured_reference_roots(
        source_project_root,
        derivative_root=nro_derivative_root(derivative_project_root),
    )
    decoded = resolve_path_values(value, roots)
    artifact_contract = decoded.get("artifact_contract")
    if isinstance(artifact_contract, Mapping):
        decoded["artifact_contract"] = _contract_dataset_view(
            artifact_contract, source_project_root
        )
    scientific_contract = decoded.get("scientific_contract")
    if isinstance(scientific_contract, Mapping):
        decoded["scientific_contract"] = _scientific_contract_dataset_view(
            scientific_contract, source_project_root
        )
    encoded = encode_path_values(decoded, roots, public=True)
    encoded["record_fingerprint"] = ownership_record_fingerprint(encoded)
    return encoded


def _refresh_inventory(registry, paths: Iterable[Path]) -> None:
    """Align private integrity evidence with restored or converted metadata."""
    with registry.connection(write=True) as database:
        _refresh_inventory_locked(database, paths)


def _refresh_inventory_locked(database, paths: Iterable[Path]) -> None:
    """Refresh integrity evidence inside the caller's transaction."""
    for path in paths:
        resolved = str(path.resolve())
        if not path.is_file():
            database.execute(
                "DELETE FROM artifacts WHERE direction='output' AND path=?", (resolved,)
            )
            continue
        record = file_record(path)
        database.execute(
            """UPDATE artifacts SET size=?,mtime_ns=?,digest_algorithm=?,digest=?
               WHERE direction='output' AND path=?""",
            (
                record["size"],
                record["mtime_ns"],
                "sha256" if "sha256" in record else None,
                record.get("sha256"),
                resolved,
            ),
        )


def _verify_recorded_source_metadata(registry, paths: Iterable[Path]) -> tuple[str, ...]:
    """Reject cleanup when a recorded sidecar no longer matches completion evidence."""
    errors = []
    with registry.connection() as database:
        for path in paths:
            resolved = str(path.resolve())
            rows = database.execute(
                """SELECT size,mtime_ns,digest_algorithm,digest FROM artifacts
                   WHERE direction='input' AND path=?""",
                (resolved,),
            ).fetchall()
            if not rows:
                continue
            current = file_record(path)
            for row in rows:
                same = int(row["size"]) == current["size"]
                if row["digest_algorithm"] == "sha256" and row["digest"] is not None:
                    same = same and row["digest"] == current.get("sha256")
                else:
                    same = same and int(row["mtime_ns"]) == current["mtime_ns"]
                if not same:
                    errors.append(
                        f"{path}: current content differs from recorded completion evidence"
                    )
                    break
    return tuple(errors)


def _migrate_registry_contracts(registry, projects: tuple[str, ...]) -> None:
    """Align live and completed scheduler contracts with semantic metadata."""
    with registry.connection(write=True) as database:
        _migrate_registry_contracts_locked(registry, database, projects)


def _migrate_registry_contracts_locked(registry, database, projects: tuple[str, ...]) -> None:
    """Migrate scheduler contracts inside the caller's transaction."""
    selected = set(projects)
    rows = database.execute("SELECT id,project,artifact_contract_json FROM work_items").fetchall()
    for row in rows:
        if row["project"] not in selected:
            continue
        project_root = Path(registry.paths.bids_root) / str(row["project"])
        contract = _contract_dataset_view(json.loads(row["artifact_contract_json"]), project_root)
        rendered = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        digest = fingerprint(contract)
        inputs = json.dumps(contract.get("inputs", ()))
        database.execute(
            """UPDATE work_items SET artifact_contract_json=?,artifact_fingerprint=?,
                      input_paths_json=? WHERE id=?""",
            (rendered, digest, inputs, row["id"]),
        )
        completion = database.execute(
            "SELECT artifact_contract_json FROM completions WHERE work_item_id=?",
            (row["id"],),
        ).fetchone()
        if completion is not None:
            completed_contract = _contract_dataset_view(
                json.loads(completion["artifact_contract_json"]), project_root
            )
            database.execute(
                """UPDATE completions SET artifact_contract_json=?,artifact_fingerprint=?
                   WHERE work_item_id=?""",
                (
                    json.dumps(completed_contract, sort_keys=True, separators=(",", ":")),
                    fingerprint(completed_contract),
                    row["id"],
                ),
            )
        for table in ("work_item_execution", "branch_work_items"):
            stored = database.execute(
                f"SELECT scientific_contract_json FROM {table} WHERE work_item_id=?",
                (row["id"],),
            ).fetchall()
            for item in stored:
                scientific = _scientific_contract_dataset_view(
                    json.loads(item["scientific_contract_json"]), project_root
                )
                database.execute(
                    f"""UPDATE {table} SET scientific_contract_json=?
                        WHERE work_item_id=? AND scientific_contract_json=?""",
                    (
                        json.dumps(scientific, sort_keys=True, separators=(",", ":")),
                        row["id"],
                        item["scientific_contract_json"],
                    ),
                )
        source_prefix = str(project_root.resolve()) + os.sep
        database.execute(
            """DELETE FROM artifacts WHERE work_item_id=? AND direction='input'
               AND path LIKE ? ESCAPE '\\' AND path LIKE '%.json'""",
            (row["id"], source_prefix.replace("%", "\\%").replace("_", "\\_") + "%"),
        )


def _migrate_branch_contracts(registry, projects: tuple[str, ...]) -> None:
    """Align reconstructible branch-scientific records with durable receipts."""
    from nro.orchestration.branch_store import BranchStore

    selected = set(projects)
    try:
        store = BranchStore(registry.paths.control)
        topology = store.read().topology
    except (FileNotFoundError, ValueError):
        return
    for name in sorted(topology.records):
        scientific = store.registry(name)
        with scientific._connection(write=True) as database:
            for row in database.execute(
                "SELECT work_item_key,contract_json FROM work_items"
            ).fetchall():
                contract = json.loads(row["contract_json"])
                project = str(contract.get("project", ""))
                if project not in selected:
                    continue
                migrated = _scientific_contract_dataset_view(
                    contract, Path(registry.paths.bids_root) / project
                )
                database.execute(
                    """UPDATE work_items SET contract_json=?,contract_fingerprint=?
                       WHERE work_item_key=?""",
                    (
                        json.dumps(migrated, sort_keys=True, separators=(",", ":")),
                        fingerprint(migrated),
                        row["work_item_key"],
                    ),
                )


def _registry_contract_change_count(registry, projects: tuple[str, ...]) -> int:
    """Count scheduler contracts whose dataset representation would change."""
    selected = set(projects)
    changed = 0
    with registry.connection() as database:
        rows = database.execute("SELECT project,artifact_contract_json FROM work_items").fetchall()
    for row in rows:
        if row["project"] not in selected:
            continue
        current = json.loads(row["artifact_contract_json"])
        migrated = _contract_dataset_view(
            current, Path(registry.paths.bids_root) / str(row["project"])
        )
        changed += migrated != current
    return changed


def _restore_journal(registry, journal: Path, record: dict) -> None:
    """Roll back one prepared or interrupted public-metadata transaction."""
    restored = []
    for entry in reversed(record.get("files", [])):
        path = Path(entry["path"])
        if entry["existed"]:
            backup = journal / entry["backup"]
            if not backup.is_file():
                raise RuntimeError(f"Portable provenance backup is missing: {backup}")
            shutil.copy2(backup, path)
        else:
            path.unlink(missing_ok=True)
        restored.append(path)
    _refresh_inventory(registry, restored)
    atomic_write_json(
        journal / "journal.json",
        {**record, "state": "rolled_back"},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )


def _recover_interrupted(registry) -> None:
    """Restore any conversion that did not reach its durable commit marker."""
    root = _journal_root(registry)
    if not root.is_dir():
        return
    for journal in sorted(path for path in root.iterdir() if path.is_dir()):
        marker = journal / "journal.json"
        if not marker.is_file():
            raise RuntimeError(f"Portable provenance journal is incomplete: {journal}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        state = record.get("state")
        if state == "committing":
            projects = tuple(str(value) for value in record.get("projects", ()))
            if _registry_contract_change_count(registry, projects) == 0:
                _migrate_branch_contracts(registry, projects)
                atomic_write_json(
                    marker,
                    {**record, "state": "complete"},
                    sort_keys=True,
                    mode=0o664,
                    durable=True,
                )
            else:
                _restore_journal(registry, journal, record)
        elif state not in {"complete", "rolled_back"}:
            _restore_journal(registry, journal, record)
        shutil.rmtree(journal)


def _unfinished_journals(registry) -> tuple[Path, ...]:
    """Return migration journals that require execution-time recovery."""
    root = _journal_root(registry)
    if not root.is_dir():
        return ()
    unfinished = []
    for journal in sorted(path for path in root.iterdir() if path.is_dir()):
        marker = journal / "journal.json"
        if not marker.is_file():
            unfinished.append(journal)
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("state") not in {"complete", "rolled_back"}:
            unfinished.append(journal)
    return tuple(unfinished)


def _document(path: Path) -> tuple[object, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text), "json"
    return yaml.safe_load(text), "yaml"


def _serialized(value: object, kind: str) -> str:
    if kind == "json":
        return json.dumps(value, indent=2, sort_keys=True) + "\n"
    return yaml.safe_dump(value, sort_keys=False)


def _portable_document(path: Path, value: object) -> object:
    if path.name == "dataset_description.json":
        return value
    if OWNERSHIP_DIRECTORY in path.parts:
        if not isinstance(value, dict):
            raise ValueError("ownership document is not a mapping")
        version = value.get("record_version")
        if version == OWNERSHIP_VERSION:
            return value
        if version != LEGACY_OWNERSHIP_VERSION:
            raise ValueError(f"unsupported ownership record version {version!r}")
        configuration_class = None
        if path.name == LINEAGE_RECORD_NAME:
            configuration_class = str(value.get("configuration_class") or "")
        return convert_legacy_ownership_record(
            value,
            roots=public_document_roots(path),
            configuration_class=configuration_class,
        )
    return portable_public_payload(path, value)


def _candidates(project_root: Path) -> tuple[Path, ...]:
    root = project_root / "derivatives" / "nro"
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
        )
    )


def migrate_dataset(
    registry,
    *,
    projects: Iterable[str],
    execute: bool = False,
    version: str,
    site_values: Mapping[str, object] | None = None,
) -> DatasetMigrationReport:
    """Preview or apply source and derivative metadata normalization together.

    Representation-only source fields are removed only while matching durable
    work-item contracts are converted to semantic metadata snapshots. Scientific
    generations, states, and dependency edges remain unchanged.
    """
    replacements: dict[Path, str] = {}
    source_replacements: set[Path] = set()
    errors: list[str] = []
    scanned = 0
    selected = tuple(sorted(set(projects)))
    if execute:
        with registry.connection() as database:
            active = int(
                database.execute(
                    """SELECT COUNT(*) FROM attempts
                       WHERE state IN ('queued','running','cancel_requested')"""
                ).fetchone()[0]
            )
        if active:
            raise ValueError("Dataset migration requires all attempts to be stopped")
        _recover_interrupted(registry)
    else:
        errors.extend(
            f"Interrupted migration requires --execute recovery: {path}"
            for path in _unfinished_journals(registry)
        )
    bids_roots = _bids_roots(registry, site_values)
    project_roots = tuple(root / project for root in bids_roots for project in selected)
    source_project_roots = tuple(Path(registry.paths.bids_root) / project for project in selected)
    for source_project_root in source_project_roots:
        if not source_project_root.is_dir():
            continue
        for path in sorted(source_project_root.rglob("*.json")):
            if not _is_source_imaging_sidecar(path, source_project_root):
                continue
            scanned += 1
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{path}: {error}")
                continue
            if not isinstance(value, dict) or "EventsFile" not in value:
                continue
            converted = dict(value)
            converted.pop("EventsFile")
            replacements[path] = _serialized(converted, "json")
            source_replacements.add(path)
    for project_root in project_roots:
        derivative_root = project_root / "derivatives/nro"
        if not derivative_root.is_dir():
            continue
        for path in _candidates(project_root):
            scanned += 1
            try:
                value, kind = _document(path)
                converted = _portable_document(path, value)
                converted = _ownership_dataset_view(
                    path,
                    converted,
                    Path(registry.paths.bids_root) / project_root.name,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
                errors.append(f"{path}: {error}")
                continue
            if converted != value:
                replacements[path] = _serialized(converted, kind)
        description = project_root / "derivatives/nro/dataset_description.json"
        try:
            expected_description = derivative_dataset_description(project_root, version=version)
        except ValueError as error:
            message = f"{description}: {error}"
            if message not in errors:
                errors.append(message)
            continue
        current_description = None
        if description.is_file():
            try:
                current_description = json.loads(description.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{description}: {error}")
        if current_description != expected_description:
            replacements[description] = _serialized(expected_description, "json")
    errors.extend(_verify_recorded_source_metadata(registry, source_replacements))
    contract_changes = 0
    try:
        contract_changes = _registry_contract_change_count(registry, selected)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"Could not prepare scheduler contracts: {error}")
    if errors or not execute:
        return DatasetMigrationReport(scanned, tuple(replacements), contract_changes, tuple(errors))
    if not replacements:
        _migrate_registry_contracts(registry, selected)
        _migrate_branch_contracts(registry, selected)
        return DatasetMigrationReport(scanned, (), contract_changes, ())
    root = _journal_root(registry)
    root.mkdir(parents=True, exist_ok=True, mode=0o2775)
    journal = root / uuid.uuid4().hex
    backups = journal / "files"
    backups.mkdir(parents=True, mode=0o2775)
    entries = []
    for index, path in enumerate(replacements):
        relative = f"files/{index:08d}"
        existed = path.is_file()
        if existed:
            shutil.copy2(path, journal / relative)
        entries.append({"path": str(path.resolve()), "backup": relative, "existed": existed})
    journal_record = {
        "format": 1,
        "state": "prepared",
        "projects": list(selected),
        "files": entries,
    }
    atomic_write_json(
        journal / "journal.json",
        journal_record,
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    try:
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "applying"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
        for path, rendered in replacements.items():
            existed = path.is_file()
            stat = path.stat() if existed else None
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
            atomic_write_text(
                path,
                rendered,
                mode=(stat.st_mode & 0o777) if stat is not None else 0o664,
                durable=True,
            )
            if stat is not None:
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        validation_errors = []
        for bids_root in bids_roots:
            represented = tuple(
                project
                for project in selected
                if (bids_root / project / "derivatives/nro").is_dir()
            )
            if not represented:
                continue
            lineages, receipts, found = read_ownership_records(
                bids_root,
                represented,
                source_bids_root=registry.paths.bids_root,
            )
            _ = (lineages, receipts)
            validation_errors.extend(found)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        with registry.connection(write=True) as database:
            _migrate_registry_contracts_locked(registry, database, selected)
            _refresh_inventory_locked(database, replacements)
            atomic_write_json(
                journal / "journal.json",
                {**journal_record, "state": "committing"},
                sort_keys=True,
                mode=0o664,
                durable=True,
            )
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "complete"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
        _migrate_branch_contracts(registry, selected)
    except Exception:
        _recover_interrupted(registry)
        raise
    finally:
        marker = journal / "journal.json"
        if marker.is_file():
            state = json.loads(marker.read_text(encoding="utf-8")).get("state")
            if state in {"complete", "rolled_back"}:
                shutil.rmtree(journal)
    return DatasetMigrationReport(scanned, tuple(replacements), contract_changes, ())
