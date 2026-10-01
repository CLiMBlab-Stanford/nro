"""Transactional conversion of nro public metadata to portable references."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, TypeVar

import yaml

from nro.configuration.store import fingerprint
from nro.engine.bids import bids_suffix
from nro.engine.freesurfer_templates import (
    FASTSURFER_FREESURFER_BUILD,
    FASTSURFER_FSAVERAGE_SOURCE,
    FREESURFER_BUILD,
    FREESURFER_FSAVERAGE_SOURCE,
    LEGACY_FSAVERAGE_SOURCES,
    QUNEX_FREESURFER_BUILD,
    QUNEX_FSAVERAGE_SOURCE,
    ensure_portable_fsaverage,
    template_directory,
)
from nro.engine.io import atomic_output_path, atomic_write_json, atomic_write_text
from nro.engine.references import (
    ReferenceRoots,
    configured_reference_roots,
    derivative_dataset_description,
    encode_path_values,
    nro_derivative_root,
    omit_private_path_values,
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
    normalize_portable_ownership_record,
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
_METADATA_READ_WORKERS = 8
_METADATA_READ_BATCH = 256
_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True)
class DatasetMigrationReport:
    """Preview or outcome of one coordinated dataset conversion."""

    scanned: int
    changed: tuple[Path, ...]
    contracts: int
    errors: tuple[str, ...]
    recovery: tuple[Path, ...] = ()
    templates: tuple[Path, ...] = ()
    source_links: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DatasetMigrationPreparation:
    """Validated metadata rewrites retained between preview and confirmation."""

    report: DatasetMigrationReport
    replacements: Mapping[Path, str]
    source_replacements: frozenset[Path]
    projects: tuple[str, ...]
    bids_roots: tuple[Path, ...]
    source_digests: Mapping[Path, str | None]
    template_symlinks: Mapping[Path, str]
    source_symlinks: Mapping[Path, str]


def _copy_or_link(source: str | Path, destination: str | Path) -> str:
    """Hard-link one file when possible, otherwise preserve its metadata."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def _materialize_source_symlink(path: Path) -> None:
    """Replace one raw-data link with project-owned bytes transactionally."""
    target = path.resolve(strict=True)
    temporary = path.with_name(f".{path.name}.materialize-{uuid.uuid4().hex}")
    saved = path.with_name(f".{path.name}.symlink-{uuid.uuid4().hex}")
    try:
        if target.is_dir():
            shutil.copytree(target, temporary, symlinks=False, copy_function=_copy_or_link)
        elif target.is_file():
            _copy_or_link(target, temporary)
        else:
            raise ValueError(f"Unsupported raw BIDS symbolic-link target: {path} -> {target}")
        os.replace(path, saved)
        try:
            os.replace(temporary, path)
        except BaseException:
            os.replace(saved, path)
            raise
        saved.unlink()
    finally:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        else:
            temporary.unlink(missing_ok=True)
        saved.unlink(missing_ok=True)


def _restore_source_symlink(path: Path, target: str) -> None:
    """Restore a raw-data link after an interrupted migration."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    path.symlink_to(target)


def _source_symlinks(project_root: Path) -> tuple[dict[Path, str], tuple[str, ...]]:
    """Inventory raw-data links without entering the derivatives namespace."""
    links: dict[Path, str] = {}
    errors = []

    def failed(error: OSError) -> None:
        errors.append(f"Cannot inspect raw BIDS path {error.filename}: {error.strerror}")

    for parent, directories, files in os.walk(
        project_root, topdown=True, followlinks=False, onerror=failed
    ):
        directory = Path(parent)
        if directory == project_root and "derivatives" in directories:
            directories.remove("derivatives")
        for name in tuple(directories):
            path = directory / name
            if path.is_symlink():
                directories.remove(name)
                links[path] = os.readlink(path)
        for name in files:
            path = directory / name
            if path.is_symlink():
                links[path] = os.readlink(path)
    for path in links:
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            errors.append(f"Cannot materialize raw BIDS link {path}: {error}")
            continue
        if not target.is_file() and not target.is_dir():
            errors.append(f"Unsupported raw BIDS link target: {path} -> {target}")
    return links, tuple(errors)


def _template_spec(target: str, site_values: Mapping[str, object]) -> tuple[str, Path]:
    """Resolve a legacy container link to the configured image that owns it."""
    if target == FREESURFER_FSAVERAGE_SOURCE:
        return FREESURFER_BUILD, Path(str(site_values["freesurfer"]))
    if target == FASTSURFER_FSAVERAGE_SOURCE:
        return FASTSURFER_FREESURFER_BUILD, Path(str(site_values["fastsurfer"]))
    if target == QUNEX_FSAVERAGE_SOURCE:
        return QUNEX_FREESURFER_BUILD, Path(str(site_values["qunex"]))
    raise ValueError(f"Unsupported FreeSurfer template target: {target}")


def _template_root(link: Path) -> Path:
    """Return the project-shared template root for one anatomical code link."""
    for parent in link.parents:
        if parent.name == "nro" and parent.parent.name == "derivatives":
            return parent / ".nro/templates/freesurfer"
    raise ValueError(f"FreeSurfer template link is outside nro derivatives: {link}")


def _legacy_template_links(project_root: Path) -> tuple[dict[Path, str], tuple[str, ...]]:
    """Find repairable container links without walking reconstruction products."""
    links: dict[Path, str] = {}
    errors = []
    anat_root = project_root / "derivatives/nro/anat"
    if not anat_root.is_dir():
        return links, ()
    resolved_project = project_root.resolve()
    for link in sorted(anat_root.glob("*/code/freesurfer/fsaverage")):
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        if Path(target).is_absolute():
            if target in LEGACY_FSAVERAGE_SOURCES:
                links[link] = target
            else:
                errors.append(f"Unsupported absolute nro derivative link: {link} -> {target}")
            continue
        try:
            resolved = (link.parent / target).resolve(strict=False)
        except RuntimeError as error:
            errors.append(f"Cannot resolve nro derivative link {link}: {error}")
            continue
        if not resolved.is_relative_to(resolved_project):
            errors.append(f"nro derivative link escapes its BIDS project: {link} -> {target}")
    return links, tuple(errors)


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


def _ownership_dataset_view(
    path: Path,
    value: object,
    source_project_root: Path,
    *,
    roots: ReferenceRoots | None = None,
) -> object:
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
    if roots is None:
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
    targets = {str(path.resolve()): path for path in paths}
    if not targets:
        return
    artifact_ids: dict[str, list[int]] = {}
    for row in database.execute("SELECT id,path FROM artifacts WHERE direction='output'"):
        if row["path"] in targets:
            artifact_ids.setdefault(row["path"], []).append(int(row["id"]))
    updates = []
    removals = []
    for resolved, path in targets.items():
        ids = artifact_ids.get(resolved, ())
        if not ids:
            continue
        if not path.is_file():
            removals.extend((artifact_id,) for artifact_id in ids)
            continue
        record = file_record(path)
        updates.extend(
            (
                record["size"],
                record["mtime_ns"],
                "sha256" if "sha256" in record else None,
                record.get("sha256"),
                artifact_id,
            )
            for artifact_id in ids
        )
    database.executemany(
        """UPDATE artifacts SET size=?,mtime_ns=?,digest_algorithm=?,digest=?
           WHERE id=?""",
        updates,
    )
    database.executemany("DELETE FROM artifacts WHERE id=?", removals)


def _verify_recorded_source_metadata(registry, paths: Iterable[Path]) -> tuple[str, ...]:
    """Reject cleanup when a recorded sidecar no longer matches completion evidence."""
    targets = {str(path.resolve()): path for path in paths}
    if not targets:
        return ()
    errors = []
    with registry.connection() as database:
        recorded: dict[str, list] = {}
        for row in database.execute(
            """SELECT path,size,mtime_ns,digest_algorithm,digest FROM artifacts
               WHERE direction='input'"""
        ):
            if row["path"] in targets:
                recorded.setdefault(row["path"], []).append(row)
        for resolved, path in targets.items():
            rows = recorded.get(resolved, ())
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
    if not selected:
        return
    placeholders = ",".join("?" for _ in selected)
    rows = database.execute(
        f"""SELECT id,project,artifact_contract_json FROM work_items
            WHERE project IN ({placeholders})""",
        tuple(sorted(selected)),
    ).fetchall()
    selected_ids = {int(row["id"]) for row in rows}
    branch_contracts: dict[int, list] = {}
    for item in database.execute(
        "SELECT rowid,work_item_id,scientific_contract_json FROM branch_work_items"
    ):
        work_item_id = int(item["work_item_id"])
        if work_item_id in selected_ids:
            branch_contracts.setdefault(work_item_id, []).append(item)
    for row in rows:
        work_item_id = int(row["id"])
        project_root = Path(registry.paths.bids_root) / str(row["project"])
        contract = _contract_dataset_view(json.loads(row["artifact_contract_json"]), project_root)
        rendered = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        digest = fingerprint(contract)
        inputs = json.dumps(contract.get("inputs", ()))
        database.execute(
            """UPDATE work_items SET artifact_contract_json=?,artifact_fingerprint=?,
                      input_paths_json=? WHERE id=?""",
            (rendered, digest, inputs, work_item_id),
        )
        completion = database.execute(
            "SELECT artifact_contract_json FROM completions WHERE work_item_id=?",
            (work_item_id,),
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
                    work_item_id,
                ),
            )
        execution = database.execute(
            """SELECT scientific_contract_json FROM work_item_execution
               WHERE work_item_id=?""",
            (work_item_id,),
        ).fetchone()
        if execution is not None:
            scientific = _scientific_contract_dataset_view(
                json.loads(execution["scientific_contract_json"]), project_root
            )
            database.execute(
                """UPDATE work_item_execution SET scientific_contract_json=?
                   WHERE work_item_id=?""",
                (
                    json.dumps(scientific, sort_keys=True, separators=(",", ":")),
                    work_item_id,
                ),
            )
        for item in branch_contracts.get(work_item_id, ()):
            scientific = _scientific_contract_dataset_view(
                json.loads(item["scientific_contract_json"]), project_root
            )
            database.execute(
                "UPDATE branch_work_items SET scientific_contract_json=? WHERE rowid=?",
                (
                    json.dumps(scientific, sort_keys=True, separators=(",", ":")),
                    item["rowid"],
                ),
            )
        source_prefix = str(project_root.resolve()) + os.sep
        database.execute(
            """DELETE FROM artifacts WHERE work_item_id=? AND direction='input'
               AND path LIKE ? ESCAPE '\\' AND path LIKE '%.json'""",
            (work_item_id, source_prefix.replace("%", "\\%").replace("_", "\\_") + "%"),
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
    for entry in reversed(record.get("source_links", [])):
        path = Path(entry["path"])
        if path.parent.is_dir():
            _restore_source_symlink(path, str(entry["old_target"]))
    for entry in reversed(record.get("templates", [])):
        path = Path(entry["path"])
        if path.parent.is_dir():
            path.unlink(missing_ok=True)
            path.symlink_to(str(entry["old_target"]))
        directory = Path(entry["template_directory"])
        if not entry.get("template_directory_preexisting") and directory.is_dir():
            shutil.rmtree(directory)
    for entry in reversed(record.get("files", [])):
        path = Path(entry["path"])
        if entry["existed"]:
            backup = journal / entry["backup"]
            if not backup.is_file():
                raise RuntimeError(f"Portable provenance backup is missing: {backup}")
            if path.is_file() and path.stat().st_size == backup.stat().st_size:
                if path.read_bytes() == backup.read_bytes():
                    # Avoid replacing unchanged files. Besides reducing shared
                    # filesystem traffic, this permits recovery to pass safely
                    # through read-only excluded source directories that the
                    # legacy inventory incorrectly included.
                    continue
            backup_stat = backup.stat()
            with atomic_output_path(path) as temporary:
                shutil.copyfile(backup, temporary)
            os.chmod(path, int(entry.get("mode", backup_stat.st_mode & 0o777)))
            os.utime(
                path,
                ns=(
                    int(entry.get("atime_ns", backup_stat.st_atime_ns)),
                    int(entry.get("mtime_ns", backup_stat.st_mtime_ns)),
                ),
            )
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


def _finish_registry_update(registry, journal: Path, record: dict) -> None:
    """Commit the restartable registry phase after public files validate."""
    projects = tuple(str(value) for value in record.get("projects", ()))
    paths = tuple(Path(entry["path"]) for entry in record.get("files", ()))
    with registry.connection(write=True) as database:
        _migrate_registry_contracts_locked(registry, database, projects)
        _refresh_inventory_locked(database, paths)
    marker = journal / "journal.json"
    atomic_write_json(
        marker,
        {**record, "state": "committing"},
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    _migrate_branch_contracts(registry, projects)
    atomic_write_json(
        marker,
        {**record, "state": "complete"},
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
            # Releases before format 2 created backups before publishing a
            # manifest. Such a directory cannot describe applied changes, so
            # it is safe to discard rather than permanently blocking recovery.
            shutil.rmtree(journal)
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        state = record.get("state")
        if state == "preparing":
            # No public file is written until every backup is durable and the
            # journal advances to prepared.
            pass
        elif state == "registry_pending":
            _finish_registry_update(registry, journal, record)
        elif state == "committing":
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


def _parallel_map(function: Callable[[_T], _R], values: Iterable[_T]) -> Iterator[_R]:
    """Map bounded batches concurrently while preserving filesystem order."""
    with ThreadPoolExecutor(max_workers=_METADATA_READ_WORKERS) as executor:
        batch: list[_T] = []
        for value in values:
            batch.append(value)
            if len(batch) < _METADATA_READ_BATCH:
                continue
            yield from executor.map(function, batch)
            batch.clear()
        if batch:
            yield from executor.map(function, batch)


def _source_document_update(path: Path) -> tuple[Path, str | None, str | None]:
    """Return a normalized raw sidecar update without writing the file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return path, None, f"{path}: {error}"
    if not isinstance(value, dict) or "EventsFile" not in value:
        return path, None, None
    converted = dict(value)
    converted.pop("EventsFile")
    return path, _serialized(converted, "json"), None


def _derivative_document_update(
    request: tuple[Path, Path, ReferenceRoots],
) -> tuple[Path, str | None, str | None]:
    """Return a portable derivative-metadata update without writing the file."""
    path, source_project_root, roots = request
    try:
        value, kind = _document(path)
        converted = _portable_document(path, value, roots=roots)
        converted = _ownership_dataset_view(
            path,
            converted,
            source_project_root,
            roots=roots,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        return path, None, f"{path}: {error}"
    if converted == value:
        return path, None, None
    return path, _serialized(converted, kind), None


def _portable_document(path: Path, value: object, *, roots: ReferenceRoots | None = None) -> object:
    if path.name == "dataset_description.json":
        return value
    if OWNERSHIP_DIRECTORY in path.parts:
        if not isinstance(value, dict):
            raise ValueError("ownership document is not a mapping")
        version = value.get("record_version")
        if version == OWNERSHIP_VERSION:
            configuration_class = None
            if path.name == LINEAGE_RECORD_NAME:
                configuration_class = str(value.get("configuration_class") or "")
            return normalize_portable_ownership_record(
                value,
                roots=roots or public_document_roots(path),
                configuration_class=configuration_class,
            )
        if version != LEGACY_OWNERSHIP_VERSION:
            raise ValueError(f"unsupported ownership record version {version!r}")
        configuration_class = None
        if path.name == LINEAGE_RECORD_NAME:
            configuration_class = str(value.get("configuration_class") or "")
        return convert_legacy_ownership_record(
            value,
            roots=roots or public_document_roots(path),
            configuration_class=configuration_class,
        )
    if roots is None:
        roots = public_document_roots(path)
    return encode_path_values(omit_private_path_values(value, roots), roots, public=True)


def _metadata_files(root: Path, *, prune_code_products: bool = False) -> Iterator[Path]:
    """Yield structured metadata without statting every scientific output."""
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        names.sort()
        files.sort()
        current = Path(directory)
        relative = current.relative_to(root)
        if prune_code_products and len(relative.parts) == 3 and relative.parts[-1] == "code":
            names.clear()
            continue
        for name in files:
            if name.startswith(".") and ".tmp-" in name:
                # A killed atomic writer can leave its unpublished sibling
                # behind. It is garbage, not part of the public metadata set.
                continue
            path = current / name
            if path.suffix.lower() in {".json", ".yaml", ".yml"}:
                yield path


def _source_candidates(project_root: Path) -> Iterator[Path]:
    """Yield imaging sidecars from raw BIDS subject trees only."""
    for subject_root in sorted(project_root.glob("sub-*")):
        if not subject_root.is_dir():
            continue
        for path in _metadata_files(subject_root):
            if "_excluded" in path.relative_to(subject_root).parts:
                continue
            if _is_source_imaging_sidecar(path, project_root):
                yield path


def _candidates(project_root: Path) -> Iterator[Path]:
    """Yield nro-owned public metadata while pruning external code products."""
    root = project_root / "derivatives" / "nro"
    if not root.is_dir():
        return
    yield from _metadata_files(root, prune_code_products=True)


def _apply_dataset_migration(
    registry,
    prepared: DatasetMigrationPreparation,
    *,
    site_values: Mapping[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> DatasetMigrationReport:
    """Apply one retained migration preview without rediscovering its files."""
    report = prepared.report
    if report.errors:
        return report
    with registry.connection() as database:
        active = int(
            database.execute(
                """SELECT COUNT(*) FROM attempts
                   WHERE state IN ('queued','running','cancel_requested')"""
            ).fetchone()[0]
        )
    if active:
        raise ValueError("Project migration requires all attempts to be stopped")
    if report.recovery:
        _recover_interrupted(registry)
        raise ValueError("Interrupted migration recovered; preview the project migration again")
    _recover_interrupted(registry)
    replacements = dict(prepared.replacements)
    template_symlinks = dict(prepared.template_symlinks)
    source_symlinks = dict(prepared.source_symlinks)
    selected = prepared.projects
    bids_roots = prepared.bids_roots
    if _registry_contract_change_count(registry, selected) != report.contracts:
        raise ValueError("Work-item contracts changed after the project migration preview")

    def notify(phase: str, count: int | None = None) -> None:
        if progress is None:
            return
        if count is None:
            progress(phase)
        else:
            unit = "file" if count == 1 else "files"
            progress(f"{phase} ({count:,} {unit})" if count else phase)

    for path, expected in prepared.source_digests.items():
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if current != expected:
            raise ValueError(f"Migration input changed after preview: {path}")
    for path, expected in source_symlinks.items():
        if not path.is_symlink() or os.readlink(path) != expected:
            raise ValueError(f"Raw BIDS link changed after migration preview: {path}")
    if not replacements and not template_symlinks and not source_symlinks:
        notify("Updating work-item contracts")
        _migrate_registry_contracts(registry, selected)
        _migrate_branch_contracts(registry, selected)
        return DatasetMigrationReport(report.scanned, (), report.contracts, ())
    if template_symlinks and site_values is None:
        raise ValueError("FreeSurfer template migration requires site configuration")
    notify("Backing up migration metadata", 0)
    root = _journal_root(registry)
    root.mkdir(parents=True, exist_ok=True, mode=0o2775)
    journal = root / uuid.uuid4().hex
    entries = []
    for index, path in enumerate(replacements):
        relative = f"files/{index:08d}"
        existed = path.is_file()
        stat = path.stat() if existed else None
        entry = {"path": str(path.resolve()), "backup": relative, "existed": existed}
        if stat is not None:
            entry.update(
                mode=stat.st_mode & 0o777,
                atime_ns=stat.st_atime_ns,
                mtime_ns=stat.st_mtime_ns,
            )
        entries.append(entry)
    backups = journal / "files"
    backups.mkdir(parents=True, mode=0o2775)
    template_entries = []
    for path, old_target in template_symlinks.items():
        build, _image = _template_spec(old_target, site_values or {})
        directory = template_directory(_template_root(path), build=build)
        template_entries.append(
            {
                "path": str(path),
                "old_target": old_target,
                "template_directory": str(directory),
                "template_directory_preexisting": directory.exists(),
            }
        )
    journal_record = {
        "format": 3,
        "state": "preparing",
        "projects": list(selected),
        "files": entries,
        "templates": template_entries,
        "source_links": [
            {"path": str(path), "old_target": target} for path, target in source_symlinks.items()
        ],
    }
    atomic_write_json(
        journal / "journal.json",
        journal_record,
        sort_keys=True,
        mode=0o664,
        durable=True,
    )
    try:
        for index, entry in enumerate(entries):
            if entry["existed"]:
                with atomic_output_path(journal / entry["backup"]) as temporary:
                    shutil.copyfile(Path(entry["path"]), temporary)
            notify("Backing up migration metadata", index + 1)
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "prepared"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "applying"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
        if source_symlinks:
            notify("Materializing raw BIDS links", 0)
        for index, path in enumerate(source_symlinks):
            _materialize_source_symlink(path)
            notify("Materializing raw BIDS links", index + 1)
        if template_symlinks:
            notify("Repairing FreeSurfer template links", 0)
        for index, (path, old_target) in enumerate(template_symlinks.items()):
            build, image = _template_spec(old_target, site_values or {})
            ensure_portable_fsaverage(
                runtime=str((site_values or {})["runtime"]),
                image=image,
                subjects_dir=path.parent,
                template_root=_template_root(path),
                build=build,
                container_source=old_target,
                execute=lambda command: subprocess.run(command, check=True),
            )
            notify("Repairing FreeSurfer template links", index + 1)
        notify("Writing portable metadata", 0)
        for index, (path, rendered) in enumerate(replacements.items()):
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
            notify("Writing portable metadata", index + 1)
        notify("Validating migrated ownership")
        validation_errors = []
        for bids_root in bids_roots:
            represented = tuple(
                project
                for project in selected
                if (bids_root / project / "derivatives/nro").is_dir()
            )
            if not represented:
                continue
            _lineages, _receipts, found = read_ownership_records(
                bids_root,
                represented,
                source_bids_root=registry.paths.bids_root,
            )
            validation_errors.extend(found)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        notify("Updating work-item contracts")
        atomic_write_json(
            journal / "journal.json",
            {**journal_record, "state": "registry_pending"},
            sort_keys=True,
            mode=0o664,
            durable=True,
        )
        _finish_registry_update(registry, journal, journal_record)
    except Exception:
        _recover_interrupted(registry)
        raise
    finally:
        marker = journal / "journal.json"
        if marker.is_file():
            state = json.loads(marker.read_text(encoding="utf-8")).get("state")
            if state in {"complete", "rolled_back"}:
                shutil.rmtree(journal)
    return DatasetMigrationReport(
        report.scanned,
        tuple(replacements),
        report.contracts,
        (),
        templates=tuple(template_symlinks),
        source_links=tuple(source_symlinks),
    )


def migrate_dataset(
    registry,
    *,
    projects: Iterable[str],
    execute: bool = False,
    version: str,
    site_values: Mapping[str, object] | None = None,
    prepared: DatasetMigrationPreparation | None = None,
    retain_preparation: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DatasetMigrationReport | DatasetMigrationPreparation:
    """Preview or apply source and derivative metadata normalization together.

    Representation-only source fields are removed only while matching durable
    work-item contracts are converted to semantic metadata snapshots. Scientific
    generations, states, and dependency edges remain unchanged.
    """
    if prepared is not None:
        if not execute:
            raise ValueError("A retained project migration may only be executed")
        return _apply_dataset_migration(
            registry, prepared, site_values=site_values, progress=progress
        )
    replacements: dict[Path, str] = {}
    source_replacements: set[Path] = set()
    errors: list[str] = []
    source_symlinks: dict[Path, str] = {}
    scanned = 0
    last_progress = -1
    last_notice = None

    def report(phase: str, count: int | None = None, *, force: bool = False) -> None:
        nonlocal last_notice, last_progress
        if progress is None:
            return
        if count is None:
            notice = phase
        else:
            unit = "file" if count == 1 else "files"
            notice = f"{phase} ({count:,} {unit})" if count else phase
        if notice == last_notice or (
            count is not None and not force and count - last_progress < 250
        ):
            return
        progress(notice)
        last_notice = notice
        last_progress = count if count is not None else -1

    selected = tuple(sorted(set(projects)))
    recovery = _unfinished_journals(registry)
    if execute:
        with registry.connection() as database:
            active = int(
                database.execute(
                    """SELECT COUNT(*) FROM attempts
                       WHERE state IN ('queued','running','cancel_requested')"""
                ).fetchone()[0]
            )
        if active:
            raise ValueError("Project migration requires all attempts to be stopped")
        _recover_interrupted(registry)
    bids_roots = _bids_roots(registry, site_values)
    project_roots = tuple(root / project for root in bids_roots for project in selected)
    source_project_roots = tuple(Path(registry.paths.bids_root) / project for project in selected)
    source_scanned = 0
    report("Scanning source metadata", source_scanned, force=True)
    for source_project_root in source_project_roots:
        if not source_project_root.is_dir():
            continue
        found_links, link_errors = _source_symlinks(source_project_root)
        source_symlinks.update(found_links)
        errors.extend(link_errors)
        for path, rendered, error in _parallel_map(
            _source_document_update,
            _source_candidates(source_project_root),
        ):
            scanned += 1
            source_scanned += 1
            report("Scanning source metadata", source_scanned)
            if error is not None:
                errors.append(error)
                continue
            if rendered is None:
                continue
            replacements[path] = rendered
            source_replacements.add(path)
    report("Scanning source metadata", source_scanned, force=True)
    derivative_scanned = 0
    template_symlinks: dict[Path, str] = {}
    last_progress = -1
    report("Scanning derivative metadata", derivative_scanned, force=True)
    for project_root in project_roots:
        derivative_root = project_root / "derivatives/nro"
        if not derivative_root.is_dir():
            continue
        found_templates, template_errors = _legacy_template_links(project_root)
        template_symlinks.update(found_templates)
        errors.extend(template_errors)
        source_project_root = Path(registry.paths.bids_root) / project_root.name
        roots = configured_reference_roots(
            source_project_root,
            derivative_root=derivative_root,
        )
        requests = ((path, source_project_root, roots) for path in _candidates(project_root))
        for path, rendered, error in _parallel_map(
            _derivative_document_update,
            requests,
        ):
            scanned += 1
            derivative_scanned += 1
            report("Scanning derivative metadata", derivative_scanned)
            if error is not None:
                errors.append(error)
                continue
            if rendered is not None:
                replacements[path] = rendered
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
    report("Scanning derivative metadata", derivative_scanned, force=True)
    report("Checking work-item contracts", force=True)
    errors.extend(_verify_recorded_source_metadata(registry, source_replacements))
    contract_changes = 0
    try:
        contract_changes = _registry_contract_change_count(registry, selected)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"Could not prepare scheduler contracts: {error}")
    migration_report = DatasetMigrationReport(
        scanned,
        tuple(replacements),
        contract_changes,
        tuple(errors),
        recovery if not execute else (),
        tuple(template_symlinks),
        tuple(source_symlinks),
    )
    preparation = DatasetMigrationPreparation(
        migration_report,
        dict(replacements),
        frozenset(source_replacements),
        selected,
        bids_roots,
        {
            path: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for path in replacements
        },
        dict(template_symlinks),
        dict(source_symlinks),
    )
    if retain_preparation:
        return preparation
    if not execute:
        return migration_report
    return _apply_dataset_migration(
        registry, preparation, site_values=site_values, progress=progress
    )
