"""Exchange scientific freshness results without importing a module catalog."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping

from nro.orchestration import dependency_state

PROTOCOL_VERSION = 1


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class AssessmentConflict(RuntimeError):
    """The graph changed while scientific validation was reading its artifacts."""


@dataclass(frozen=True)
class AssessmentSnapshot:
    """One consistent registry view for a scientific validator.

    Rows include the selected instances and all their ancestors. Mutable worker
    heartbeats and resource limits are excluded from the comparison token.
    This record grants no authority to update the represented instances.
    """

    bids_root: Path
    control: Path
    instances: tuple[dict, ...]
    dependencies: tuple[dict, ...]
    configurations: tuple[dict, ...]
    attempts: tuple[dict, ...]
    mutations: tuple[dict, ...]

    @property
    def paths(self) -> SimpleNamespace:
        """Supply filesystem context without a database handle to scientific checks."""
        return SimpleNamespace(bids_root=self.bids_root, control=self.control)

    def as_dict(self) -> dict:
        """Serialize a detached JSON-compatible snapshot for a validator process."""
        return json.loads(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "bids_root": str(self.bids_root),
                    "control": str(self.control),
                    "instances": self.instances,
                    "dependencies": self.dependencies,
                    "configurations": self.configurations,
                    "attempts": self.attempts,
                    "mutations": self.mutations,
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, value: Mapping) -> AssessmentSnapshot:
        """Decode a supported snapshot; reject a malformed transport envelope."""
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "protocol",
                "bids_root",
                "control",
                "instances",
                "dependencies",
                "configurations",
                "attempts",
                "mutations",
            }
            or type(value["protocol"]) is not int
            or value["protocol"] != PROTOCOL_VERSION
        ):
            raise ValueError("Unsupported assessment snapshot")
        for key in ("bids_root", "control"):
            if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
                raise ValueError("Assessment paths must be absolute")
        for key in ("instances", "dependencies", "configurations", "attempts", "mutations"):
            if not isinstance(value[key], list) or not all(
                isinstance(row, dict) for row in value[key]
            ):
                raise ValueError(f"Invalid assessment {key}")
        detached = json.loads(json.dumps(value, allow_nan=False))
        return cls(
            Path(detached["bids_root"]),
            Path(detached["control"]),
            *(
                tuple(detached[key])
                for key in ("instances", "dependencies", "configurations", "attempts", "mutations")
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Identify the captured state for compare-and-swap publication, not freshness."""
        return _fingerprint(self.as_dict())


@dataclass(frozen=True)
class AssessmentReport:
    """Scientific decisions tied to one captured registry view.

    The scheduler validates this transport and checks the snapshot before writing.
    It does not infer scientific equivalence or interpret module configuration.
    """

    snapshot_fingerprint: str
    updates: tuple[dict, ...]

    def as_dict(self) -> dict:
        """Serialize decisions without transferring a database connection."""
        return json.loads(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "snapshot_fingerprint": self.snapshot_fingerprint,
                    "updates": self.updates,
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, value: Mapping) -> AssessmentReport:
        """Decode only supported, bounded update fields and valid artifact states."""
        if (
            not isinstance(value, Mapping)
            or set(value) != {"protocol", "snapshot_fingerprint", "updates"}
            or type(value["protocol"]) is not int
            or value["protocol"] != PROTOCOL_VERSION
            or not isinstance(value["snapshot_fingerprint"], str)
            or not isinstance(value["updates"], list)
        ):
            raise ValueError("Unsupported assessment report")
        seen = set()
        for update in value["updates"]:
            if not isinstance(update, dict) or set(update) != {
                "id",
                "state",
                "reason",
                "contract",
                "command",
                "inputs",
            }:
                raise ValueError("Invalid assessment update fields")
            if type(update["id"]) is not int or update["id"] < 1 or update["id"] in seen:
                raise ValueError("Assessment instance IDs must be distinct positive integers")
            seen.add(update["id"])
            if update["state"] not in ("fresh", "missing", "stale") or not isinstance(
                update["reason"], str
            ):
                raise ValueError("Invalid assessment artifact state")
            if update["contract"] is not None and not isinstance(update["contract"], dict):
                raise ValueError("Invalid assessment contract")
            for key in ("command", "inputs"):
                item = update[key]
                if item is not None and (
                    not isinstance(item, list) or not all(isinstance(v, str) for v in item)
                ):
                    raise ValueError(f"Invalid assessment {key}")
            if update["command"] is not None and update["contract"] is None:
                raise ValueError("Assessment command changes require a changed contract")
        detached = json.loads(json.dumps(value, allow_nan=False))
        return cls(detached["snapshot_fingerprint"], tuple(detached["updates"]))


def _capture_locked(registry, db, *, instance_ids=None, projects=None) -> AssessmentSnapshot:
    instances = [
        dict(row)
        for row in db.execute("""SELECT i.*, e.branch AS execution_branch
        FROM instances i LEFT JOIN instance_execution e ON e.instance_id=i.id ORDER BY i.id""")
    ]
    edges = [
        dict(row)
        for row in db.execute(
            "SELECT * FROM instance_dependencies ORDER BY instance_id, upstream_instance_id, role"
        )
    ]
    requested = None if instance_ids is None else set(instance_ids)
    project_set = None if projects is None else set(projects)
    selected = {
        row["id"]
        for row in instances
        if (requested is None or row["id"] in requested)
        and (project_set is None or row["project"] in project_set)
    }
    parents = {}
    for edge in edges:
        parents.setdefault(edge["instance_id"], set()).add(edge["upstream_instance_id"])
    pending = list(selected)
    while pending:
        for parent in parents.get(pending.pop(), set()) - selected:
            selected.add(parent)
            pending.append(parent)
    # Resources and heartbeats can change while validation is in progress without
    # changing its conclusions. Generation, graph, recipe, and attempt changes cannot.
    instances = [
        {
            key: value
            for key, value in row.items()
            if key not in {"memory_gb", "max_memory_gb", "updated_at"}
        }
        for row in instances
        if row["id"] in selected
    ]
    lineages = {row["configuration_lineage_id"] for row in instances}
    configs = tuple(
        dict(row)
        for row in db.execute(
            "SELECT id, config_fingerprint, resolved_yaml FROM configuration_lineages ORDER BY id"
        )
        if row["id"] in lineages
    )
    attempts = tuple(
        dict(row)
        for row in db.execute(
            "SELECT id, instance_id, state FROM attempts WHERE state IN "
            "('queued', 'running', 'cancel_requested') ORDER BY id"
        )
        if row["instance_id"] in selected
    )
    mutations = tuple(
        dict(row)
        for row in db.execute("SELECT * FROM artifact_mutations ORDER BY instance_id")
        if row["instance_id"] in selected
    )
    return AssessmentSnapshot(
        registry.paths.bids_root,
        registry.paths.control,
        tuple(instances),
        tuple(edge for edge in edges if edge["instance_id"] in selected),
        configs,
        attempts,
        mutations,
    )


def capture_assessment(
    registry, *, instance_ids: Iterable[int] | None = None, projects: Iterable[str] | None = None
) -> AssessmentSnapshot:
    """Read selected records and their dependency closure in one locked transaction."""
    with registry.connection() as db:
        return _capture_locked(registry, db, instance_ids=instance_ids, projects=projects)


def apply_assessment(
    registry, snapshot: AssessmentSnapshot, report: AssessmentReport
) -> dict[int, tuple[str, str]]:
    """Publish a report atomically only while its captured registry state still matches.

    Reject the whole report on conflict, including changes to an upstream outside
    the original selection. Caller authorization and validator selection precede
    this operation; neither a snapshot nor its fingerprint grants write access.
    """
    from nro.orchestration.registry import utcnow

    report = AssessmentReport.from_dict(report.as_dict())
    selected = {row["id"] for row in snapshot.instances}
    if (
        report.snapshot_fingerprint != snapshot.fingerprint
        or {update["id"] for update in report.updates} != selected
    ):
        raise ValueError("Assessment report does not match its captured selection")
    now = utcnow()
    by_id = {row["id"]: row for row in snapshot.instances}
    for update in report.updates:
        if update["contract"] is not None:
            original = json.loads(by_id[update["id"]]["artifact_contract_json"])

            def fixed(contract):
                return {
                    key: value
                    for key, value in contract.items()
                    if key not in {"configuration", "processing"}
                }

            if fixed(original) != fixed(update["contract"]):
                raise ValueError(
                    "Assessment cannot change instance identity, dependencies, or output paths"
                )
        if update["inputs"] is not None and any(
            not Path(path).is_absolute() or ".." in Path(path).parts for path in update["inputs"]
        ):
            raise ValueError("Assessment input paths must be normalized and absolute")
    with registry.connection(write=True) as db:
        current = _capture_locked(registry, db, instance_ids=selected)
        if current.fingerprint != snapshot.fingerprint:
            raise AssessmentConflict(
                "Registry state changed during assessment; retry with a new snapshot"
            )
        for update in report.updates:
            instance_id = update["id"]
            if update["contract"] is not None:
                contract = json.dumps(update["contract"], sort_keys=True, separators=(",", ":"))
                # Scientific code has already normalized the contract. This hash
                # checks its serialized identity without importing that code.
                contract_fingerprint = _fingerprint(update["contract"])
                db.execute(
                    """UPDATE instances SET artifact_state=?, artifact_reason=?,
                              artifact_contract_json=?, artifact_fingerprint=?,
                              command_json=COALESCE(?, command_json),
                              input_paths_json=COALESCE(?, input_paths_json), updated_at=? WHERE id=?""",
                    (
                        update["state"],
                        update["reason"],
                        contract,
                        contract_fingerprint,
                        json.dumps(update["command"]) if update["command"] is not None else None,
                        json.dumps(update["inputs"]) if update["inputs"] is not None else None,
                        now,
                        instance_id,
                    ),
                )
                db.execute(
                    """UPDATE attempts SET state='cancel_requested', error_type='InstanceGraphChanged',
                              error_message='Instance contract changed while work was active'
                              WHERE instance_id=? AND state IN ('queued', 'running')""",
                    (instance_id,),
                )
            else:
                db.execute(
                    """UPDATE instances SET artifact_state=?, artifact_reason=?,
                              input_paths_json=COALESCE(?, input_paths_json), updated_at=? WHERE id=?""",
                    (
                        update["state"],
                        update["reason"],
                        json.dumps(update["inputs"]) if update["inputs"] is not None else None,
                        now,
                        instance_id,
                    ),
                )
            if update["state"] == "fresh":
                db.executemany(
                    """UPDATE instance_dependencies SET required_generation=?
                                  WHERE instance_id=? AND upstream_instance_id=?""",
                    [
                        (
                            by_id[edge["upstream_instance_id"]]["current_generation"],
                            instance_id,
                            edge["upstream_instance_id"],
                        )
                        for edge in snapshot.dependencies
                        if edge["instance_id"] == instance_id
                    ],
                )
        dependency_state.synchronize(db, now=now)
        return {
            row["id"]: (row["artifact_state"], row["artifact_reason"])
            for row in db.execute("SELECT id, artifact_state, artifact_reason FROM instances")
            if row["id"] in selected
        }
