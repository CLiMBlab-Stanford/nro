"""Typed contracts exchanged by the planner, registry, and workers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.configuration.store import fingerprint


INSTANCE_CONTRACT_VERSION = 3


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


@dataclass(frozen=True)
class InstanceIdentity:
    """The fields that identify one schedulable module instance."""

    key: str
    module: str
    project: str
    participant: str
    entities: Mapping[str, str]
    configuration_lineage_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "module": self.module,
            "configuration_lineage_id": self.configuration_lineage_id,
            "participant": self.participant,
            "entities": dict(sorted(self.entities.items())),
        }


@dataclass(frozen=True)
class OutputContract:
    """The public output boundary promised by an instance."""

    root: Path
    prefix: str | None
    expected: tuple[Path, ...]
    format: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": _normalized_path(self.root),
            "prefix": self.prefix,
            "expected": sorted({_normalized_path(path) for path in self.expected}),
            "format": self.format,
        }


@dataclass(frozen=True)
class InstanceContract:
    """The freshness-relevant promise made by one instance.

    Identity supplies the module and applicable entities when the contract is
    serialized. Execution commands and resource requests deliberately do not
    belong here.
    """

    configuration_fingerprint: str
    dependencies: tuple[str, ...]
    inputs: tuple[Path, ...]
    output: OutputContract
    processing: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, identity: InstanceIdentity) -> dict[str, Any]:
        result: dict[str, Any] = {
            "module": identity.module,
            "configuration": self.configuration_fingerprint,
            "entities": dict(sorted(identity.entities.items())),
            "dependencies": sorted(set(self.dependencies)),
            "inputs": sorted({_normalized_path(path) for path in self.inputs}),
            "output": self.output.as_dict(),
        }
        if self.processing:
            result["processing"] = dict(self.processing)
        return result


@dataclass(frozen=True)
class ExecutionRecipe:
    """Non-semantic instructions for executing an instance."""

    command: tuple[str, ...]
    runtime_config: Path


@dataclass(frozen=True)
class ResourceRequest:
    """Scheduler requirements that do not affect derivative freshness."""

    resource_class: str
    memory_gb: int = 32
    max_memory_gb: int = 256


@dataclass(frozen=True)
class InstanceSpec:
    """Complete planner specification for one logical instance."""

    identity: InstanceIdentity
    contract: InstanceContract
    execution: ExecutionRecipe
    resources: ResourceRequest
    scope: str
    directory_label: str

    @classmethod
    def create(
        cls,
        *,
        key: str,
        module: str,
        project: str,
        participant: str,
        entities: Mapping[str, str],
        scope: str,
        configuration_lineage_id: int,
        config_fingerprint: str,
        directory_label: str,
        runtime_config: Path,
        command: Sequence[str],
        dependencies: Sequence[str],
        input_paths: Sequence[Path],
        output_root: Path,
        output_prefix: str | None,
        resource_class: str,
        output_format: str | None = None,
        memory_gb: int = 32,
        max_memory_gb: int = 256,
        expected_outputs: Sequence[Path] = (),
        processing: Mapping[str, Any] | None = None,
    ) -> "InstanceSpec":
        """Construct the hierarchy from fields convenient for module planners."""
        if output_format is None:
            from nro.orchestration.catalog import module_descriptor

            output_format = module_descriptor(module).output_format
        return cls(
            identity=InstanceIdentity(
                key=key,
                module=module,
                project=project,
                participant=participant,
                entities=dict(entities),
                configuration_lineage_id=configuration_lineage_id,
            ),
            contract=InstanceContract(
                configuration_fingerprint=config_fingerprint,
                dependencies=tuple(dependencies),
                inputs=tuple(Path(path) for path in input_paths),
                output=OutputContract(
                    root=Path(output_root),
                    prefix=output_prefix,
                    expected=tuple(Path(path) for path in expected_outputs),
                    format=output_format,
                ),
                processing=dict(processing or {}),
            ),
            execution=ExecutionRecipe(
                command=tuple(str(value) for value in command),
                runtime_config=Path(runtime_config),
            ),
            resources=ResourceRequest(
                resource_class=resource_class,
                memory_gb=memory_gb,
                max_memory_gb=max_memory_gb,
            ),
            scope=scope,
            directory_label=directory_label,
        )

    @property
    def key(self) -> str:
        return self.identity.key

    @property
    def module(self) -> str:
        return self.identity.module

    @property
    def project(self) -> str:
        return self.identity.project

    @property
    def participant(self) -> str:
        return self.identity.participant

    @property
    def entities(self) -> Mapping[str, str]:
        return self.identity.entities

    @property
    def configuration_lineage_id(self) -> int:
        return self.identity.configuration_lineage_id

    @property
    def config_fingerprint(self) -> str:
        return self.contract.configuration_fingerprint

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.contract.dependencies

    @property
    def input_paths(self) -> tuple[Path, ...]:
        return self.contract.inputs

    @property
    def output_root(self) -> Path:
        return self.contract.output.root

    @property
    def output_prefix(self) -> str | None:
        return self.contract.output.prefix

    @property
    def expected_outputs(self) -> tuple[Path, ...]:
        return self.contract.output.expected

    @property
    def command(self) -> tuple[str, ...]:
        return self.execution.command

    @property
    def runtime_config(self) -> Path:
        return self.execution.runtime_config

    @property
    def resource_class(self) -> str:
        return self.resources.resource_class

    @property
    def memory_gb(self) -> int:
        return self.resources.memory_gb

    @property
    def max_memory_gb(self) -> int:
        return self.resources.max_memory_gb

    @property
    def instance_contract(self) -> dict[str, Any]:
        """Return the normalized semantic contract used for freshness."""
        return self.contract.as_dict(self.identity)

    @property
    def contract_fingerprint(self) -> str:
        return fingerprint(self.instance_contract)

    @property
    def revision_fingerprint(self) -> str:
        return fingerprint(
            {
                "contract_version": INSTANCE_CONTRACT_VERSION,
                "module": self.module,
                "config": self.config_fingerprint,
                "entities": self.entities,
            }
        )

    def as_record(self) -> dict[str, Any]:
        """Serialize to the current registry storage representation."""
        contract = self.instance_contract
        return {
            "instance_key": self.key,
            "module": self.module,
            "project": self.project,
            "participant": self.participant,
            "entities_json": json.dumps(dict(self.entities), sort_keys=True),
            "scope": self.scope,
            "configuration_lineage_id": self.configuration_lineage_id,
            "resource_class": self.resource_class,
            "memory_gb": self.memory_gb,
            "max_memory_gb": self.max_memory_gb,
            "revision_fingerprint": self.revision_fingerprint,
            # Registry storage uses "artifact" for the concrete filesystem
            # evidence implementing this conceptual instance contract.
            "artifact_contract_json": json.dumps(
                contract, sort_keys=True, separators=(",", ":")
            ),
            "artifact_fingerprint": self.contract_fingerprint,
            "command_json": json.dumps(self.command),
            "runtime_config_path": str(self.runtime_config),
            "input_paths_json": json.dumps(contract["inputs"]),
            "output_root": contract["output"]["root"],
            "output_prefix": self.output_prefix,
            "expected_outputs_json": json.dumps(contract["output"]["expected"]),
        }

    def evolve(self, **changes: Any) -> "InstanceSpec":
        """Return a changed specification while preserving its hierarchy."""
        identity_fields = {
            "key",
            "module",
            "project",
            "participant",
            "entities",
            "configuration_lineage_id",
        }
        contract_fields = {
            "config_fingerprint": "configuration_fingerprint",
            "dependencies": "dependencies",
            "input_paths": "inputs",
            "processing": "processing",
        }
        output_fields = {
            "output_root": "root",
            "output_prefix": "prefix",
            "expected_outputs": "expected",
            "output_format": "format",
        }
        execution_fields = {
            "command": "command",
            "runtime_config": "runtime_config",
        }
        resource_fields = {
            "resource_class": "resource_class",
            "memory_gb": "memory_gb",
            "max_memory_gb": "max_memory_gb",
        }
        top_level = {name for name in ("scope", "directory_label") if name in changes}
        recognized = (
            identity_fields
            | set(contract_fields)
            | set(output_fields)
            | set(execution_fields)
            | set(resource_fields)
            | top_level
        )
        unknown = set(changes) - recognized
        if unknown:
            raise TypeError(f"Unknown InstanceSpec field(s): {', '.join(sorted(unknown))}")

        identity = replace(
            self.identity,
            **{name: changes[name] for name in identity_fields if name in changes},
        )
        output = replace(
            self.contract.output,
            **{
                target: changes[source]
                for source, target in output_fields.items()
                if source in changes
            },
        )
        contract = replace(
            self.contract,
            output=output,
            **{
                target: changes[source]
                for source, target in contract_fields.items()
                if source in changes
            },
        )
        execution = replace(
            self.execution,
            **{
                target: changes[source]
                for source, target in execution_fields.items()
                if source in changes
            },
        )
        resources = replace(
            self.resources,
            **{
                target: changes[source]
                for source, target in resource_fields.items()
                if source in changes
            },
        )
        return replace(
            self,
            identity=identity,
            contract=contract,
            execution=execution,
            resources=resources,
            **{name: changes[name] for name in top_level},
        )


@dataclass(frozen=True)
class ExecutionEnvelope:
    """Typed work claimed by a worker for one attempt."""

    instance_id: int
    attempt_id: int
    instance_key: str
    module: str
    project: str
    participant: str
    entities: Mapping[str, str]
    scope: str
    manifest_path: Path
    revision_fingerprint: str
    config_fingerprint: str
    instance_contract: Mapping[str, Any]
    contract_fingerprint: str
    execution: ExecutionRecipe
    input_paths: tuple[Path, ...]
    output_root: Path
    output_prefix: str | None
    expected_outputs: tuple[Path, ...]
    log_path: Path

    @classmethod
    def from_registry_row(cls, row: Mapping[str, Any]) -> "ExecutionEnvelope":
        """Decode the registry's storage representation at its boundary."""
        return cls(
            instance_id=int(row["id"]),
            attempt_id=int(row["attempt_id"]),
            instance_key=str(row["instance_key"]),
            module=str(row["module"]),
            project=str(row["project"]),
            participant=str(row["participant"]),
            entities=json.loads(row["entities_json"]),
            scope=str(row["scope"]),
            manifest_path=Path(row["manifest_path"]),
            revision_fingerprint=str(row["revision_fingerprint"]),
            config_fingerprint=str(row["config_fingerprint"]),
            instance_contract=json.loads(row["artifact_contract_json"]),
            contract_fingerprint=str(row["artifact_fingerprint"]),
            execution=ExecutionRecipe(
                command=tuple(str(value) for value in json.loads(row["command_json"])),
                runtime_config=Path(row["runtime_config_path"]),
            ),
            input_paths=tuple(Path(value) for value in json.loads(row["input_paths_json"])),
            output_root=Path(row["output_root"]),
            output_prefix=row["output_prefix"],
            expected_outputs=tuple(
                Path(value) for value in json.loads(row["expected_outputs_json"])
            ),
            log_path=Path(row["log_path"]),
        )
