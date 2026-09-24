"""Durable ownership records stored with nro derivatives."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import yaml

from nro.configuration.store import CONFIGURATION_CLASSES, fingerprint
from nro.engine.io import atomic_write_json, atomic_write_text
from nro.engine.paths import module_artifact_root, module_namespace_root
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.dependencies import primary_dependency
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.workflow_registry import lineage_directory_label

if TYPE_CHECKING:
    from nro.orchestration.registry import Registry


OWNERSHIP_VERSION = 4
OWNERSHIP_DIRECTORY = ".nro"
LINEAGE_RECORD_NAME = "lineage.json"


def lineage_root(project_root: Path, configuration_class: str, directory_label: str) -> Path:
    """Return the public root assigned to one module lineage."""
    return module_artifact_root(project_root, configuration_class, directory_label)


def lineage_record_path(project_root: Path, configuration_class: str, directory_label: str) -> Path:
    """Return the ownership record for one module lineage."""
    return lineage_root(project_root, configuration_class, directory_label) / (
        f"{OWNERSHIP_DIRECTORY}/{LINEAGE_RECORD_NAME}"
    )


def work_item_record_path(
    project_root: Path,
    configuration_class: str,
    directory_label: str,
    module: str,
    key: str,
) -> Path:
    """Return the ownership receipt path for one work item."""
    digest = key.split(":", 1)[-1]
    return (
        lineage_root(project_root, configuration_class, directory_label)
        / OWNERSHIP_DIRECTORY
        / "work_items"
        / module
        / f"{digest}.json"
    )


def remove_empty_ownership_root(
    project_root: Path, configuration_class: str, directory_label: str
) -> None:
    """Remove empty ownership and lineage directories after their final receipt."""
    lineage = lineage_root(project_root, configuration_class, directory_label)
    control = lineage / OWNERSHIP_DIRECTORY
    work_items = control / "work_items"
    if work_items.is_dir():
        for module_directory in work_items.iterdir():
            if module_directory.is_dir():
                try:
                    module_directory.rmdir()
                except OSError:
                    pass
        try:
            work_items.rmdir()
        except OSError:
            return
    elif any((control / "work_items").glob("*")):
        return
    (control / LINEAGE_RECORD_NAME).unlink(missing_ok=True)
    try:
        control.rmdir()
    except OSError:
        return
    boundary = module_namespace_root(project_root, configuration_class)
    current = lineage
    while current != boundary:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _lineage_rows(registry: "Registry", lineage_id: int) -> tuple[dict, list[dict]]:
    with registry.connection() as db:
        lineage = dict(
            db.execute("SELECT * FROM module_lineages WHERE id=?", (lineage_id,)).fetchone()
        )
        upstream = [
            dict(row)
            for row in db.execute(
                """
                SELECT dependency.role, parent.configuration_class,
                       parent.config_id, parent.lineage_fingerprint,
                       parent.directory_label
                FROM module_lineage_dependencies dependency
                JOIN module_lineages parent
                  ON parent.id=dependency.upstream_module_lineage_id
                WHERE dependency.module_lineage_id=?
                ORDER BY dependency.role, parent.configuration_class
                """,
                (lineage_id,),
            )
        ]
    return lineage, upstream


def write_work_item_ownership(
    registry: "Registry", work_item_id: int, *, attempt_id: int | None = None
) -> Path:
    """Store enough public metadata to recover a work item after registry repair."""
    with registry.connection() as db:
        work_item = dict(
            db.execute(
                """
                SELECT item.*, lineage.configuration_class, lineage.config_id,
                       lineage.config_fingerprint, lineage.lineage_fingerprint,
                       lineage.resolved_yaml, lineage.directory_label
                FROM work_items item
                JOIN module_lineages lineage
                  ON lineage.id=item.module_lineage_id
                WHERE item.id=?
                """,
                (work_item_id,),
            ).fetchone()
        )
        logical_dependencies = [
            str(row[0])
            for row in db.execute(
                """SELECT COALESCE(execution.logical_key, upstream.work_item_key)
                   FROM work_item_dependencies dependency
                   JOIN work_items upstream ON upstream.id=dependency.upstream_work_item_id
                   LEFT JOIN work_item_execution execution
                     ON execution.work_item_id=upstream.id
                   WHERE dependency.work_item_id=?
                   ORDER BY COALESCE(execution.logical_key, upstream.work_item_key)""",
                (work_item_id,),
            )
        ]
    lineage, upstream = _lineage_rows(registry, int(work_item["module_lineage_id"]))
    project_root = registry.paths.bids_root / str(work_item["project"])
    provenance = None
    with registry.connection() as db:
        if attempt_id is None:
            execution = db.execute(
                """SELECT context_json, provenance_json, logical_key, branch,
                          scientific_contract_json
                   FROM work_item_execution WHERE work_item_id=?""",
                (work_item_id,),
            ).fetchone()
        else:
            execution = db.execute(
                """SELECT e.context_json,e.provenance_json
                   FROM attempt_execution e
                JOIN attempts a ON a.id=e.attempt_id WHERE e.attempt_id=? AND a.work_item_id=?""",
                (attempt_id, work_item_id),
            ).fetchone()
            if execution is not None:
                current = db.execute(
                    """SELECT logical_key,scientific_contract_json,branch
                       FROM work_item_execution WHERE work_item_id=?""",
                    (work_item_id,),
                ).fetchone()
                execution = dict(execution)
                if current is not None:
                    execution.update(dict(current))
    if execution is not None and not isinstance(execution, dict):
        execution = dict(execution)
    if execution is not None:
        from nro.orchestration.execution_context import ExecutionContext

        context = ExecutionContext.from_dict(json.loads(execution["context_json"]))
        project_root = context.paths.output_project(str(work_item["project"]))
        context.require_output(Path(work_item["output_root"]))
        provenance = json.loads(execution["provenance_json"])
        from nro.orchestration.branch_store import BranchStore

        scientific = BranchStore(registry.paths.control).registry(str(execution["branch"]))
        with scientific.connection() as db:
            local = db.execute(
                """SELECT id FROM module_lineages
                   WHERE configuration_class=? AND directory_label=?""",
                (work_item["configuration_class"], work_item["directory_label"]),
            ).fetchone()
        if local is None:
            raise ValueError("Branch scientific registry lacks the completed module lineage")
        lineage, upstream = _lineage_rows(scientific, int(local["id"]))
    now = datetime.now(timezone.utc).isoformat()
    root_record = {
        "record_version": OWNERSHIP_VERSION,
        "owner": "nro",
        "configuration_class": lineage["configuration_class"],
        "directory_label": lineage["directory_label"],
        "configuration": {
            "id": lineage["config_id"],
            "fingerprint": lineage["config_fingerprint"],
            "resolved": yaml.safe_load(lineage["resolved_yaml"]) or {},
        },
        "lineage_fingerprint": lineage["lineage_fingerprint"],
        "upstream": upstream,
        "updated_at": now,
    }
    if provenance is not None:
        root_record["implementation"] = provenance
    root_path = lineage_record_path(
        project_root,
        str(lineage["configuration_class"]),
        str(lineage["directory_label"]),
    )
    root_path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
    atomic_write_json(root_path, root_record, sort_keys=True, mode=0o664, durable=True)

    runtime_path = Path(work_item["runtime_config_path"])
    runtime_configuration = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
    logical_key = str(
        (execution.get("logical_key") if execution is not None else None)
        or work_item["work_item_key"]
    )
    artifact_contract = json.loads(work_item["artifact_contract_json"])
    artifact_contract["dependencies"] = logical_dependencies
    scientific_contract = (
        json.loads(execution["scientific_contract_json"])
        if execution is not None and execution.get("scientific_contract_json") is not None
        else {
            "project": work_item["project"],
            "participant": work_item["participant"],
            **artifact_contract,
        }
    )
    receipt = {
        "record_version": OWNERSHIP_VERSION,
        "owner": "nro",
        "work_item_key": logical_key,
        "module": work_item["module"],
        "project": work_item["project"],
        "participant": work_item["participant"],
        "entities": json.loads(work_item["entities_json"]),
        "scope": work_item["scope"],
        "lineage_fingerprint": lineage["lineage_fingerprint"],
        "directory_label": lineage["directory_label"],
        "artifact_contract": artifact_contract,
        "scientific_contract": scientific_contract,
        "execution": {
            "command": json.loads(work_item["command_json"]),
            "runtime_configuration": runtime_configuration,
        },
        "resources": {
            "resource_class": work_item["resource_class"],
            "memory_gb": int(work_item["memory_gb"]),
            "max_memory_gb": int(work_item["max_memory_gb"]),
        },
        "recorded_at": now,
    }
    if provenance is not None:
        receipt["implementation"] = provenance
    receipt_path = work_item_record_path(
        project_root,
        str(lineage["configuration_class"]),
        str(lineage["directory_label"]),
        str(work_item["module"]),
        logical_key,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
    atomic_write_json(receipt_path, receipt, sort_keys=True, mode=0o664, durable=True)
    return receipt_path


def missing_work_item_ownership(
    registry: "Registry", work_item_ids: Iterable[int]
) -> tuple[int, ...]:
    """Return fresh work items whose public recovery records are incomplete."""
    selected = tuple(sorted(set(work_item_ids)))
    if not selected:
        return ()
    rows = []
    with registry.connection() as db:
        for offset in range(0, len(selected), 500):
            batch = selected[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                dict(row)
                for row in db.execute(
                    f"""SELECT i.id,i.work_item_key,i.module,i.project,
                               c.configuration_class,c.directory_label,e.context_json,e.logical_key
                        FROM work_items i
                        JOIN module_lineages c ON c.id=i.module_lineage_id
                        LEFT JOIN work_item_execution e ON e.work_item_id=i.id
                        WHERE i.id IN ({placeholders})""",
                    batch,
                )
            )
    missing = []
    lineages: dict[Path, bool] = {}
    for row in rows:
        project_root = registry.paths.bids_root / row["project"]
        if row["context_json"]:
            from nro.orchestration.execution_context import ExecutionContext

            context = ExecutionContext.from_dict(json.loads(row["context_json"]))
            project_root = context.paths.output_project(row["project"])
        lineage = lineage_record_path(
            project_root, row["configuration_class"], row["directory_label"]
        )
        logical_key = str(row["logical_key"] or row["work_item_key"])
        receipt = work_item_record_path(
            project_root,
            row["configuration_class"],
            row["directory_label"],
            row["module"],
            logical_key,
        )
        lineage_exists = lineages.get(lineage)
        if lineage_exists is None:
            lineage_exists = lineage.is_file()
            lineages[lineage] = lineage_exists
        if not lineage_exists or not receipt.is_file():
            missing.append(row["id"])
    return tuple(missing)


def read_ownership_records(
    bids_root: Path, projects: Iterable[str]
) -> tuple[list[dict], list[tuple[dict, Path]], list[str]]:
    """Read current-format ownership records from the selected projects."""
    lineages: dict[tuple[str, str], dict] = {}
    ambiguous: set[tuple[str, str]] = set()
    work_items: list[tuple[dict, Path]] = []
    errors: list[str] = []
    for project in projects:
        project_root = Path(bids_root) / project
        for configuration_class in CONFIGURATION_CLASSES:
            class_root = module_namespace_root(project_root, configuration_class)
            for marker_path in sorted(
                class_root.glob(f"*/{OWNERSHIP_DIRECTORY}/{LINEAGE_RECORD_NAME}")
            ):
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    _validate_lineage_record(marker, marker_path, configuration_class)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    errors.append(f"{marker_path}: {error}")
                    continue
                identity = (configuration_class, str(marker["lineage_fingerprint"]))
                previous = lineages.get(identity)
                if previous is not None and str(previous["directory_label"]) != str(
                    marker["directory_label"]
                ):
                    errors.append(
                        "Stored lineage "
                        f"{marker['lineage_fingerprint']} has conflicting derivative roots: "
                        f"{previous['directory_label']} and {marker['directory_label']}"
                    )
                    ambiguous.add(identity)
                    lineages.pop(identity, None)
                elif identity not in ambiguous and (
                    previous is None or str(marker["updated_at"]) > str(previous["updated_at"])
                ):
                    lineages[identity] = marker
                work_item_root = marker_path.parent / "work_items"
                for receipt_path in sorted(work_item_root.glob("*/*.json")):
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        _validate_work_item_record(
                            receipt,
                            receipt_path,
                            project=project,
                            configuration_class=configuration_class,
                            marker=marker,
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                        errors.append(f"{receipt_path}: {error}")
                        continue
                    work_items.append((receipt, receipt_path))
    return list(lineages.values()), work_items, errors


def complete_ownership_records(
    lineage_records: list[dict],
    work_item_records: list[tuple[dict, Path]],
) -> tuple[list[dict], list[tuple[dict, Path]], list[str]]:
    """Select complete lineage roots and their matching work-item receipts."""
    selected_directories = {
        str(record["lineage_fingerprint"]): str(record["directory_label"])
        for record in lineage_records
    }
    known = {str(record["lineage_fingerprint"]) for record in lineage_records}
    incomplete = {
        str(record["lineage_fingerprint"])
        for record in lineage_records
        if any(str(parent["lineage_fingerprint"]) not in known for parent in record["upstream"])
    }
    while True:
        downstream = {
            str(record["lineage_fingerprint"])
            for record in lineage_records
            if str(record["lineage_fingerprint"]) not in incomplete
            and any(
                str(parent["lineage_fingerprint"]) in incomplete for parent in record["upstream"]
            )
        }
        if not downstream:
            break
        incomplete.update(downstream)
    errors = [
        f"Stored lineage {value} lacks a complete upstream lineage chain"
        for value in sorted(incomplete)
    ]
    return (
        [
            record
            for record in lineage_records
            if str(record["lineage_fingerprint"]) not in incomplete
        ],
        [
            record
            for record in work_item_records
            if str(record[0]["lineage_fingerprint"]) in selected_directories
            and str(record[0]["lineage_fingerprint"]) not in incomplete
            and str(record[0]["directory_label"])
            == selected_directories[str(record[0]["lineage_fingerprint"])]
        ],
        errors,
    )


def _validate_lineage_record(
    record: Mapping[str, object], path: Path, configuration_class: str
) -> None:
    if record.get("record_version") != OWNERSHIP_VERSION or record.get("owner") != "nro":
        raise ValueError("unsupported ownership record")
    if record.get("configuration_class") != configuration_class:
        raise ValueError("configuration class does not match its directory")
    if record.get("directory_label") != path.parent.parent.name:
        raise ValueError("directory label does not match its directory")
    configuration = record.get("configuration")
    if not isinstance(configuration, Mapping) or not isinstance(
        configuration.get("resolved"), Mapping
    ):
        raise ValueError("configuration snapshot is missing")
    expected_configuration = fingerprint(
        {
            "module": configuration_class,
            "config_id": configuration.get("id"),
            "values": configuration.get("resolved"),
        }
    )
    if configuration.get("fingerprint") != expected_configuration:
        raise ValueError("configuration fingerprint does not match its snapshot")
    upstream = record.get("upstream")
    if not isinstance(upstream, list):
        raise ValueError("upstream lineage list is missing")
    expected_parent = primary_dependency(configuration_class, configuration["resolved"])
    parent_fingerprints = [
        str(item.get("lineage_fingerprint")) for item in upstream if isinstance(item, Mapping)
    ]
    if expected_parent is None and parent_fingerprints:
        raise ValueError("root lineage unexpectedly declares an upstream lineage")
    if expected_parent is not None and len(parent_fingerprints) != 1:
        raise ValueError("lineage must declare exactly one upstream lineage")
    if expected_parent is not None:
        parent = upstream[0]
        if (
            not isinstance(parent, Mapping)
            or parent.get("configuration_class") != expected_parent
            or parent.get("role") != expected_parent
        ):
            raise ValueError("upstream lineage has the wrong class or role")
    expected = fingerprint(
        {
            "module": configuration_class,
            "config_id": configuration.get("id"),
            "upstream": parent_fingerprints[0] if parent_fingerprints else None,
        }
    )
    if record.get("lineage_fingerprint") != expected:
        raise ValueError("lineage fingerprint does not match its semantic identity")
    expected_directory = lineage_directory_label(str(configuration.get("id")), expected)
    if record.get("directory_label") != expected_directory:
        raise ValueError(f"lineage directory is nondeterministic; expected {expected_directory}")


def _validate_work_item_record(
    record: Mapping[str, object],
    path: Path,
    *,
    project: str,
    configuration_class: str,
    marker: Mapping[str, object],
) -> None:
    if record.get("record_version") != OWNERSHIP_VERSION or record.get("owner") != "nro":
        raise ValueError("unsupported work-item ownership record")
    if record.get("project") != project:
        raise ValueError("project does not match its derivative tree")
    module = str(record.get("module"))
    from nro.orchestration.catalog import module_descriptor

    if module_descriptor(module).configuration_class != configuration_class:
        raise ValueError("module does not match the recorded configuration class")
    if record.get("lineage_fingerprint") != marker.get("lineage_fingerprint"):
        raise ValueError("work-item and root lineage fingerprints differ")
    entities = record.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("work-item entities are missing")
    contract = record.get("artifact_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("artifact contract is missing")
    scientific_contract = record.get("scientific_contract")
    if not isinstance(scientific_contract, Mapping):
        raise ValueError("scientific contract is missing")
    identity_fingerprint = str(record["lineage_fingerprint"])
    expected_key = work_item_key(
        project,
        module,
        identity_fingerprint,
        str(record.get("participant")),
        {str(key): str(value) for key, value in entities.items()},
    )
    if record.get("work_item_key") != expected_key:
        raise ValueError("work-item key does not match its semantic identity")
    if path.stem != expected_key.split(":", 1)[-1]:
        raise ValueError("work-item record filename does not match its key")
    if contract.get("module") != module or contract.get("entities") != dict(
        sorted(entities.items())
    ):
        raise ValueError("artifact contract does not match the work-item identity")
    if (
        scientific_contract.get("module") != module
        or scientific_contract.get("project") != project
        or scientific_contract.get("participant") != record.get("participant")
        or scientific_contract.get("entities") != dict(sorted(entities.items()))
    ):
        raise ValueError("scientific contract does not match the work-item identity")
    output = contract.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("artifact output contract is missing")
    root = Path(str(output.get("root", ""))).resolve()
    configuration_root = path.parents[3].resolve()
    try:
        root.relative_to(configuration_root)
    except ValueError as error:
        raise ValueError("artifact output root lies outside its lineage root") from error
    if not isinstance(record.get("execution"), Mapping):
        raise ValueError("execution snapshot is missing")
    if not isinstance(record.get("resources"), Mapping):
        raise ValueError("resource request is missing")


def materialize_work_item_specs(
    registry: "Registry",
    records: Iterable[tuple[dict, Path]],
    lineage_ids: Mapping[str, int],
    *,
    namespace: str | None = None,
) -> tuple[list[WorkItemSpec], list[str]]:
    """Recreate work-item specifications without the originating workflow.

    Namespace recovered runtime snapshots when records belong to a development
    branch. Module-lineage identity excludes mutable configuration content, so
    separate branches cannot safely share that private execution path.
    """
    specs: list[WorkItemSpec] = []
    errors: list[str] = []
    for record, source in records:
        try:
            lineage_fingerprint = str(record["lineage_fingerprint"])
            lineage_id = lineage_ids[lineage_fingerprint]
            contract = record["artifact_contract"]
            output = contract["output"]
            execution = record["execution"]
            resources = record["resources"]
            runtime_path = (
                registry.paths.snapshots
                / "owned-configurations"
                / (namespace or "main")
                / lineage_fingerprint[:16]
                / f"{contract['configuration']}.yml"
            )
            runtime_path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
            atomic_write_text(
                runtime_path,
                yaml.safe_dump(execution["runtime_configuration"], sort_keys=False),
                mode=0o664,
                durable=True,
            )
            specs.append(
                WorkItemSpec.create(
                    key=str(record["work_item_key"]),
                    module=str(record["module"]),
                    project=str(record["project"]),
                    participant=str(record["participant"]),
                    entities={str(key): str(value) for key, value in record["entities"].items()},
                    scope=str(record["scope"]),
                    module_lineage_id=lineage_id,
                    config_fingerprint=str(contract["configuration"]),
                    directory_label=str(record["directory_label"]),
                    runtime_config=runtime_path,
                    command=tuple(str(value) for value in execution["command"]),
                    dependencies=tuple(str(value) for value in contract.get("dependencies", ())),
                    input_paths=tuple(Path(value) for value in contract.get("inputs", ())),
                    output_root=Path(output["root"]),
                    output_prefix=output.get("prefix"),
                    expected_outputs=tuple(Path(value) for value in output.get("expected", ())),
                    output_format=str(output["format"]),
                    resource_class=str(resources["resource_class"]),
                    memory_gb=int(resources["memory_gb"]),
                    max_memory_gb=int(resources["max_memory_gb"]),
                    processing=contract.get("processing", {}),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{source}: {error}")
    return specs, errors
