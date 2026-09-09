"""Durable ownership records stored with nro derivatives."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import yaml

from nro.configuration.store import DERIVATIVE_CLASSES, UPSTREAM_CLASS, fingerprint
from nro.engine.io import atomic_write_json, atomic_write_text
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planning_context import instance_key

if TYPE_CHECKING:
    from nro.orchestration.registry import Registry


OWNERSHIP_VERSION = 1
OWNERSHIP_DIRECTORY = ".nro"
LINEAGE_RECORD_NAME = "lineage.json"


def lineage_root(project_root: Path, derivative_class: str, directory_label: str) -> Path:
    """Return the public root assigned to one configuration lineage."""
    return Path(project_root) / "derivatives" / derivative_class / directory_label


def lineage_record_path(project_root: Path, derivative_class: str, directory_label: str) -> Path:
    """Return the ownership record for one configuration lineage."""
    return lineage_root(project_root, derivative_class, directory_label) / (
        f"{OWNERSHIP_DIRECTORY}/{LINEAGE_RECORD_NAME}"
    )


def instance_record_path(
    project_root: Path,
    derivative_class: str,
    directory_label: str,
    module: str,
    key: str,
) -> Path:
    """Return the ownership receipt path for one module instance."""
    digest = key.split(":", 1)[-1]
    return (
        lineage_root(project_root, derivative_class, directory_label)
        / OWNERSHIP_DIRECTORY
        / "instances"
        / module
        / f"{digest}.json"
    )


def remove_empty_ownership_root(
    project_root: Path, derivative_class: str, directory_label: str
) -> None:
    """Remove a lineage marker after its final instance receipt is purged."""
    control = lineage_root(project_root, derivative_class, directory_label) / OWNERSHIP_DIRECTORY
    instances = control / "instances"
    if instances.is_dir():
        for module_directory in instances.iterdir():
            if module_directory.is_dir():
                try:
                    module_directory.rmdir()
                except OSError:
                    pass
        try:
            instances.rmdir()
        except OSError:
            return
    elif any((control / "instances").glob("*")):
        return
    (control / LINEAGE_RECORD_NAME).unlink(missing_ok=True)
    try:
        control.rmdir()
    except OSError:
        pass


def _lineage_rows(registry: "Registry", lineage_id: int) -> tuple[dict, list[dict]]:
    with registry.connection() as db:
        lineage = dict(
            db.execute("SELECT * FROM configuration_lineages WHERE id=?", (lineage_id,)).fetchone()
        )
        upstream = [
            dict(row)
            for row in db.execute(
                """
                SELECT dependency.role, parent.derivative_class,
                       parent.config_id, parent.lineage_fingerprint,
                       parent.directory_label
                FROM configuration_lineage_dependencies dependency
                JOIN configuration_lineages parent
                  ON parent.id=dependency.upstream_configuration_lineage_id
                WHERE dependency.configuration_lineage_id=?
                ORDER BY dependency.role, parent.derivative_class
                """,
                (lineage_id,),
            )
        ]
    return lineage, upstream


def write_instance_ownership(
    registry: "Registry", instance_id: int, *, attempt_id: int | None = None
) -> Path:
    """Store enough public metadata to recover an instance after registry repair."""
    with registry.connection() as db:
        instance = dict(
            db.execute(
                """
                SELECT item.*, lineage.derivative_class, lineage.config_id,
                       lineage.config_fingerprint, lineage.lineage_fingerprint,
                       lineage.resolved_yaml, lineage.directory_label
                FROM instances item
                JOIN configuration_lineages lineage
                  ON lineage.id=item.configuration_lineage_id
                WHERE item.id=?
                """,
                (instance_id,),
            ).fetchone()
        )
    lineage, upstream = _lineage_rows(registry, int(instance["configuration_lineage_id"]))
    project_root = registry.paths.bids_root / str(instance["project"])
    provenance = None
    with registry.connection() as db:
        if attempt_id is None:
            execution = db.execute(
                "SELECT context_json, provenance_json FROM instance_execution WHERE instance_id=?",
                (instance_id,),
            ).fetchone()
        else:
            execution = db.execute(
                """SELECT e.context_json,e.provenance_json FROM attempt_execution e
                JOIN attempts a ON a.id=e.attempt_id WHERE e.attempt_id=? AND a.instance_id=?""",
                (attempt_id, instance_id),
            ).fetchone()
    if execution is not None:
        from nro.orchestration.execution_context import ExecutionContext

        context = ExecutionContext.from_dict(json.loads(execution["context_json"]))
        project_root = context.paths.output_project(str(instance["project"]))
        context.require_output(Path(instance["output_root"]))
        provenance = json.loads(execution["provenance_json"])
    now = datetime.now(timezone.utc).isoformat()
    root_record = {
        "record_version": OWNERSHIP_VERSION,
        "owner": "nro",
        "derivative_class": lineage["derivative_class"],
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
        str(lineage["derivative_class"]),
        str(lineage["directory_label"]),
    )
    root_path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
    atomic_write_json(root_path, root_record, sort_keys=True, mode=0o664, durable=True)

    runtime_path = Path(instance["runtime_config_path"])
    runtime_configuration = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
    receipt = {
        "record_version": OWNERSHIP_VERSION,
        "owner": "nro",
        "instance_key": instance["instance_key"],
        "module": instance["module"],
        "project": instance["project"],
        "participant": instance["participant"],
        "entities": json.loads(instance["entities_json"]),
        "scope": instance["scope"],
        "lineage_fingerprint": lineage["lineage_fingerprint"],
        "directory_label": lineage["directory_label"],
        "artifact_contract": json.loads(instance["artifact_contract_json"]),
        "execution": {
            "command": json.loads(instance["command_json"]),
            "runtime_configuration": runtime_configuration,
        },
        "resources": {
            "resource_class": instance["resource_class"],
            "memory_gb": int(instance["memory_gb"]),
            "max_memory_gb": int(instance["max_memory_gb"]),
        },
        "recorded_at": now,
    }
    if provenance is not None:
        receipt["implementation"] = provenance
    receipt_path = instance_record_path(
        project_root,
        str(lineage["derivative_class"]),
        str(lineage["directory_label"]),
        str(instance["module"]),
        str(instance["instance_key"]),
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
    atomic_write_json(receipt_path, receipt, sort_keys=True, mode=0o664, durable=True)
    return receipt_path


def read_ownership_records(
    bids_root: Path, projects: Iterable[str]
) -> tuple[list[dict], list[tuple[dict, Path]], list[str]]:
    """Read current-format ownership records from the selected projects."""
    lineages: dict[tuple[str, str], dict] = {}
    instances: list[tuple[dict, Path]] = []
    errors: list[str] = []
    for project in projects:
        project_root = Path(bids_root) / project
        for derivative_class in DERIVATIVE_CLASSES:
            class_root = project_root / "derivatives" / derivative_class
            for marker_path in sorted(
                class_root.glob(f"*/{OWNERSHIP_DIRECTORY}/{LINEAGE_RECORD_NAME}")
            ):
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    _validate_lineage_record(marker, marker_path, derivative_class)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    errors.append(f"{marker_path}: {error}")
                    continue
                identity = (derivative_class, str(marker["lineage_fingerprint"]))
                previous = lineages.get(identity)
                if previous is None or str(marker["updated_at"]) > str(previous["updated_at"]):
                    lineages[identity] = marker
                instance_root = marker_path.parent / "instances"
                for receipt_path in sorted(instance_root.glob("*/*.json")):
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        _validate_instance_record(
                            receipt,
                            receipt_path,
                            project=project,
                            derivative_class=derivative_class,
                            marker=marker,
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                        errors.append(f"{receipt_path}: {error}")
                        continue
                    instances.append((receipt, receipt_path))
    return list(lineages.values()), instances, errors


def _validate_lineage_record(
    record: Mapping[str, object], path: Path, derivative_class: str
) -> None:
    if record.get("record_version") != OWNERSHIP_VERSION or record.get("owner") != "nro":
        raise ValueError("unsupported ownership record")
    if record.get("derivative_class") != derivative_class:
        raise ValueError("derivative class does not match its directory")
    if record.get("directory_label") != path.parent.parent.name:
        raise ValueError("directory label does not match its directory")
    configuration = record.get("configuration")
    if not isinstance(configuration, Mapping) or not isinstance(
        configuration.get("resolved"), Mapping
    ):
        raise ValueError("configuration snapshot is missing")
    expected_configuration = fingerprint(
        {
            "derivative_class": derivative_class,
            "config_id": configuration.get("id"),
            "values": configuration.get("resolved"),
        }
    )
    if configuration.get("fingerprint") != expected_configuration:
        raise ValueError("configuration fingerprint does not match its snapshot")
    upstream = record.get("upstream")
    if not isinstance(upstream, list):
        raise ValueError("upstream lineage list is missing")
    expected_parent = UPSTREAM_CLASS[derivative_class]
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
            or parent.get("derivative_class") != expected_parent
            or parent.get("role") != expected_parent
        ):
            raise ValueError("upstream lineage has the wrong class or role")
    expected = fingerprint(
        {
            "derivative_class": derivative_class,
            "config_id": configuration.get("id"),
            "upstream": parent_fingerprints[0] if parent_fingerprints else None,
        }
    )
    if record.get("lineage_fingerprint") != expected:
        raise ValueError("lineage fingerprint does not match its semantic identity")


def _validate_instance_record(
    record: Mapping[str, object],
    path: Path,
    *,
    project: str,
    derivative_class: str,
    marker: Mapping[str, object],
) -> None:
    if record.get("record_version") != OWNERSHIP_VERSION or record.get("owner") != "nro":
        raise ValueError("unsupported instance ownership record")
    if record.get("project") != project:
        raise ValueError("project does not match its derivative tree")
    module = str(record.get("module"))
    from nro.orchestration.catalog import module_descriptor

    if module_descriptor(module).configuration_class != derivative_class:
        raise ValueError("module does not belong to the recorded derivative class")
    if record.get("lineage_fingerprint") != marker.get("lineage_fingerprint"):
        raise ValueError("instance and root lineage fingerprints differ")
    entities = record.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("instance entities are missing")
    expected_key = instance_key(
        project,
        module,
        str(record["lineage_fingerprint"]),
        str(record.get("participant")),
        {str(key): str(value) for key, value in entities.items()},
    )
    if record.get("instance_key") != expected_key:
        raise ValueError("instance key does not match its semantic identity")
    if path.stem != expected_key.split(":", 1)[-1]:
        raise ValueError("instance record filename does not match its key")
    if not isinstance(record.get("artifact_contract"), Mapping):
        raise ValueError("artifact contract is missing")
    contract = record["artifact_contract"]
    if contract.get("module") != module or contract.get("entities") != dict(
        sorted(entities.items())
    ):
        raise ValueError("artifact contract does not match the instance identity")
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


def materialize_instance_specs(
    registry: "Registry",
    records: Iterable[tuple[dict, Path]],
    lineage_ids: Mapping[str, int],
) -> tuple[list[InstanceSpec], list[str]]:
    """Recreate planner records without requiring the originating workflow."""
    specs: list[InstanceSpec] = []
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
                InstanceSpec.create(
                    key=str(record["instance_key"]),
                    module=str(record["module"]),
                    project=str(record["project"]),
                    participant=str(record["participant"]),
                    entities={str(key): str(value) for key, value in record["entities"].items()},
                    scope=str(record["scope"]),
                    configuration_lineage_id=lineage_id,
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
