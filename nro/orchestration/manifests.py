"""Instance completion certificates and filesystem-authoritative freshness checks.

The orchestration layer validates direct inputs, upstream instance generations, and
public artifacts below an instance's derivative output root.  It also records
private step artifacts when available: a changed private artifact invalidates
the instance, while an absent one is tolerated because the module can recreate it
when it next needs to run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import yaml

from nro.configuration.store import fingerprint
from nro.engine.bids import discover_raw_runs, matches_filter
from nro.engine.io import atomic_write_json
from nro.orchestration import dependency_state
from nro.orchestration.assessment import (
    AssessmentConflict,
    AssessmentReport,
    AssessmentSnapshot,
    apply_assessment,
    capture_assessment,
)
from nro.orchestration.ownership import write_instance_ownership
from nro.orchestration.registry import Registry, ensure_shared_directory, utcnow

MANIFEST_VERSION = 3
SMALL_DIGEST_LIMIT = 1024 * 1024


def file_record(path: str | Path) -> dict:
    """Describe one completed file for later integrity checks."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            "Completion manifests may record only files; directory-producing instances "
            f"must expose a final completion breadcrumb: {resolved}"
        )
    stat = resolved.stat()
    record = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= SMALL_DIGEST_LIMIT:
        record["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return record


def inventory(paths: Iterable[str | Path]) -> list[dict]:
    """Return deterministic integrity records for unique resolved files."""
    return [file_record(path) for path in sorted({Path(path).resolve() for path in paths})]


def _same_file(record: dict) -> tuple[bool, str | None]:
    path = Path(record["path"])
    if not path.is_file():
        return False, f"Missing file: {path}"
    stat = path.stat()
    if stat.st_size != record.get("size"):
        return False, f"File size changed: {path}"
    expected_digest = record.get("sha256")
    if expected_digest is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_digest:
            return False, f"File content changed: {path}"
    elif stat.st_mtime_ns != record.get("mtime_ns"):
        return False, f"File modification time changed: {path}"
    return True, None


def _same_existing_private_file(record: dict) -> tuple[bool, str | None]:
    """Validate a private artifact only when it still exists.

    WORK cleanup is normal and must not schedule a completed derivative.  An
    existing file whose recorded state changes, however, is direct evidence of
    interference and should make the owning instance stale.
    """
    if not Path(record["path"]).is_file():
        return True, None
    return _same_file(record)


def _is_orchestration_control_artifact(path: Path, registry: Registry) -> bool:
    """Control records describe execution; they are never instance inputs.

    In particular, the completion manifest is written after the worker's step
    ledger has been updated.  Recording it as a private artifact would make a
    certificate invalidate itself on its next rewrite.
    """
    try:
        path.resolve().relative_to(registry.paths.control.resolve())
        return True
    except ValueError:
        return False


def _read_manifest(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _current_contract(row: dict) -> tuple[dict, str, bool]:
    """Overlay the current code-defined processing policy on a stored contract."""
    from nro.orchestration.catalog import canonical_contract, module_descriptor

    certificate = _read_manifest(Path(row["manifest_path"])) if row.get("manifest_path") else None
    configuration = certificate.get("configuration") if certificate else None
    contract = json.loads(row["artifact_contract_json"])
    try:
        contract = canonical_contract(contract, configuration)
    except (ValueError, TypeError, KeyError):
        return contract, fingerprint(contract), True
    current = module_descriptor(str(row["module"])).processing_for(json.loads(row["entities_json"]))
    changed = contract.get("processing", {}) != current
    if current:
        contract["processing"] = deepcopy(current)
    else:
        contract.pop("processing", None)
    return contract, fingerprint(contract), changed


def _certificate_matches_contract(
    certificate: dict, expected_fingerprint: str, *, compiled: bool = False
) -> bool:
    """Compare recorded specifications after module-specific normalization."""
    try:
        recorded = certificate["artifact_contract"]
        if compiled:
            return (
                certificate.get("artifact_fingerprint")
                == fingerprint(recorded)
                == expected_fingerprint
            )
        from nro.orchestration.catalog import canonical_contract

        return (
            certificate.get("artifact_fingerprint") == fingerprint(recorded)
            and fingerprint(canonical_contract(recorded, certificate.get("configuration")))
            == expected_fingerprint
        )
    except (KeyError, TypeError, ValueError):
        return False


def _shallow_public_output_error(outputs: object) -> tuple[str, str] | None:
    """Check only declaration shape, existence, and nonempty file size."""
    if not isinstance(outputs, list) or not outputs:
        return "missing", "Completion certificate contains no public artifacts"
    for item in outputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            return "stale", "Completion certificate has an invalid public artifact record"
        path = Path(item["path"])
        try:
            present = path.is_file() and path.stat().st_size > 0
        except OSError as error:
            return "stale", f"Could not inspect declared public artifact {path}: {error}"
        if not present:
            return "missing", f"Public artifact is missing or empty: {path}"
    return None


def preview_registry(
    registry: Registry,
    *,
    projects: Iterable[str] | None = None,
    compiled: bool = False,
) -> dict[int, tuple[str, str]]:
    """Project cheap filesystem and current-policy evidence without writing.

    This deliberately does not rediscover source datasets, fingerprint files,
    inspect private WORK artifacts, or promote a nonfresh registry record to
    fresh.  It is a fast warning layer, not an alternative registry assessor.
    """
    instances = registry.instance_rows(read_only=True)
    dependencies = registry.instance_dependencies(read_only=True)
    if projects is not None:
        selected_projects = set(projects)
        selected = {int(row["id"]) for row in instances if str(row["project"]) in selected_projects}
        instances = [row for row in instances if int(row["id"]) in selected]
        dependencies = [
            (instance_id, upstream_id)
            for instance_id, upstream_id in dependencies
            if instance_id in selected and upstream_id in selected
        ]

    by_id = {int(row["id"]): row for row in instances}
    upstream: dict[int, list[int]] = {}
    for instance_id, upstream_id in dependencies:
        upstream.setdefault(instance_id, []).append(upstream_id)

    states: dict[int, tuple[str, str]] = {}
    remaining = set(by_id)
    while remaining:
        progressed = False
        for instance_id in tuple(remaining):
            parents = upstream.get(instance_id, [])
            if any(parent in remaining for parent in parents):
                continue
            row = by_id[instance_id]
            if compiled:
                current_fingerprint = row["artifact_fingerprint"]
                contract_changed = False
            else:
                contract, current_fingerprint, contract_changed = _current_contract(row)
            if contract_changed:
                state = ("stale", "Current module processing contract changed")
            elif any(states[parent][0] != "fresh" for parent in parents):
                state = ("stale", "An upstream derivative is missing or stale")
            elif row["artifact_state"] != "fresh":
                state = (
                    str(row["artifact_state"]),
                    str(row.get("artifact_reason") or "Registry record is not fresh"),
                )
            else:
                missing_input = next(
                    (
                        Path(value)
                        for value in json.loads(row["input_paths_json"])
                        if not Path(value).is_file()
                    ),
                    None,
                )
                if missing_input is not None:
                    state = ("stale", f"Direct input is missing: {missing_input}")
                else:
                    completion = _read_manifest(Path(row["manifest_path"]))
                    if completion is not None:
                        if not _certificate_matches_contract(
                            completion, current_fingerprint, compiled=compiled
                        ):
                            state = ("stale", "Completion certificate contract changed")
                        else:
                            output_error = _shallow_public_output_error(
                                completion.get("public_outputs")
                            )
                            state = output_error or (
                                "fresh",
                                "Declared public artifacts are present",
                            )
                    else:
                        try:
                            recovered, reason = _public_derivative_completion(
                                row, registry, compiled=compiled
                            )
                        except _PublicDerivativeContractMismatch as error:
                            state = ("stale", str(error))
                        except OSError as error:
                            recovered, reason = None, f"Could not inspect public outputs: {error}"
                            state = ("missing", reason)
                        else:
                            state = (
                                ("missing", reason)
                                if recovered is None
                                else (
                                    "fresh",
                                    "Public derivative outputs are present; deep provenance was not checked",
                                )
                            )
            states[instance_id] = state
            remaining.remove(instance_id)
            progressed = True
        if not progressed:
            for instance_id in remaining:
                states[instance_id] = ("stale", "Instance dependency cycle detected")
            break
    return states


@dataclass(frozen=True)
class _PublicDerivativeCompletion:
    evidence: tuple[Path, ...]
    outputs: tuple[Path, ...]

    @property
    def oldest_completion_ns(self) -> int:
        """Return the oldest completion-evidence modification time."""
        return min(path.stat().st_mtime_ns for path in self.evidence)

    @property
    def newest_completion_ns(self) -> int:
        """Return the newest completion-evidence modification time."""
        return max(path.stat().st_mtime_ns for path in self.evidence)


class _PublicDerivativeContractMismatch(ValueError):
    """Public completion evidence does not implement the current output contract."""


def _referenced_files(value, *, base: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()

    def visit(item) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str) and item.strip():
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = base / path
            paths.add(path.resolve())

    visit(value)
    return tuple(sorted(paths))


def _read_yaml_mapping(path: Path) -> dict | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _public_derivative_completion(
    row: dict, registry: Registry, *, compiled: bool = False
) -> tuple[_PublicDerivativeCompletion | None, str]:
    """Validate the instance's fixed public derivative contract.

    The expected root manifests are fixed when the instance DAG is constructed.
    Their inventories may describe a variable-cardinality directory product,
    but neither this fallback nor freshness assessment is allowed to discover
    files by walking a derivatives directory.
    """
    module = str(row["module"])
    if not compiled:
        from nro.orchestration.catalog import module_descriptor
    root = Path(row["output_root"]).resolve()
    try:
        evidence = [
            Path(value).expanduser().resolve() for value in json.loads(row["expected_outputs_json"])
        ]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        return None, f"Public {module} output contract is invalid: {error}"
    if not evidence:
        return None, f"Public {module} output contract is empty"
    outputs: set[Path] = set()
    declared_metadata_contracts: list[object] = []

    pending = list(evidence)
    visited: set[Path] = set()
    while pending:
        artifact = pending.pop()
        try:
            artifact.relative_to(root)
        except ValueError:
            return None, f"Public {module} output contract escapes its derivative root: {artifact}"
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            return None, f"Public {module} completion manifest is missing or empty: {artifact}"
        outputs.add(artifact)
        if artifact in visited or not artifact.name.endswith(
            ("_manifest.json", "_manifest.yaml", "_manifest.yml")
        ):
            continue
        visited.add(artifact)
        value = (
            _read_manifest(artifact) if artifact.suffix == ".json" else _read_yaml_mapping(artifact)
        )
        if value is None:
            return None, f"Public {module} completion manifest is invalid: {artifact}"
        if value.get("complete") is False:
            return None, f"Public {module} completion manifest is incomplete: {artifact}"
        if "output_metadata_contract" in value:
            declared_metadata_contracts.append(value["output_metadata_contract"])
        validate_definition = (
            None if compiled else module_descriptor(module).validate_public_definition
        )
        if validate_definition is not None:
            valid, reason = validate_definition(
                value, json.loads(row["artifact_contract_json"]).get("processing", {})
            )
            if not valid:
                raise _PublicDerivativeContractMismatch(reason)
        inventory_value = value.get("public_outputs")
        referenced = _referenced_files(inventory_value, base=artifact.parent)
        for path in referenced:
            try:
                path.relative_to(root)
            except ValueError:
                # External paths in a public manifest are provenance inputs,
                # never discovered instance outputs.
                continue
            outputs.add(path)
            if path.name.endswith(("_manifest.json", "_manifest.yaml", "_manifest.yml")):
                pending.append(path)

    if not outputs:
        return None, f"Public {module} completion evidence records no derivative outputs"
    missing = [path for path in sorted(outputs) if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return None, f"Public {module} output is missing or empty: {missing[0]}"
    current_metadata_contract = (
        json.loads(row["artifact_contract_json"]).get("processing", {}).get("output_metadata")
        if compiled
        else module_descriptor(module).processing_contract().get("output_metadata")
    )
    if current_metadata_contract is not None:
        if not declared_metadata_contracts:
            raise _PublicDerivativeContractMismatch(
                f"Public {module} completion evidence does not declare an output metadata contract"
            )
        if current_metadata_contract not in declared_metadata_contracts:
            raise _PublicDerivativeContractMismatch(
                f"Public {module} output metadata does not implement the current contract"
            )
    return _PublicDerivativeCompletion(tuple(sorted(evidence)), tuple(sorted(outputs))), ""


def assess_registry(
    registry: Registry,
    *,
    instance_ids: Iterable[int] | None = None,
    projects: Iterable[str] | None = None,
    compiled: bool = False,
) -> dict[int, tuple[str, str]]:
    """Assess a consistent graph outside the lock, then publish if it is unchanged.

    Set compiled=True in workers to check registered expectations without
    importing module code or recompiling scientific contracts.
    Retry up to three times when concurrent planning or completion changes the
    captured state. Persistent contention raises AssessmentConflict rather than
    overwriting a newer result with an outdated assessment.
    """
    instance_ids = None if instance_ids is None else tuple(instance_ids)
    projects = None if projects is None else tuple(projects)
    for attempt in range(3):
        snapshot = capture_assessment(registry, instance_ids=instance_ids, projects=projects)
        report = (
            evaluate_assessment(snapshot, compiled=True)
            if compiled
            else evaluate_assessment(snapshot)
        )
        try:
            states = apply_assessment(registry, snapshot, report)
            break
        except AssessmentConflict:
            if attempt == 2:
                raise
    for instance_id, (state, _reason) in states.items():
        if state != "fresh":
            continue
        write_instance_ownership(registry, instance_id)
    return states


def evaluate_assessment(
    snapshot: AssessmentSnapshot, *, compiled: bool = False
) -> AssessmentReport:
    """Validate filesystem evidence against current or already compiled contracts.

    Branch-side callers may refresh their scientific expectations. Workers use
    compiled=True and never import a scientific catalog. Neither mode opens a
    registry or writes ownership receipts. Source provenance is not evidence of
    freshness.
    """
    if not compiled:
        from nro.modules.anat.planning import raw_anatomical_inputs
        from nro.modules.clean.planning import clean_direct_inputs
        from nro.modules.func.planning import load_session_inventory, resolved_func_inputs
        from nro.orchestration.catalog import module_descriptor
    instances = deepcopy(snapshot.instances)
    dependencies = [
        (edge["instance_id"], edge["upstream_instance_id"]) for edge in snapshot.dependencies
    ]
    registry = snapshot
    by_id = {int(row["id"]): row for row in instances}
    contract_updates: dict[int, tuple[str, str]] = {}
    changed_contracts: set[int] = set()
    command_updates: dict[int, str] = {}
    for instance_id, row in by_id.items():
        if compiled or row.get("execution_branch") not in (None, "main"):
            continue
        contract, current_fingerprint, changed = _current_contract(row)
        if changed:
            changed_contracts.add(instance_id)
        if changed or fingerprint(json.loads(row["artifact_contract_json"])) != current_fingerprint:
            row["artifact_contract_json"] = json.dumps(
                contract, sort_keys=True, separators=(",", ":")
            )
            row["artifact_fingerprint"] = current_fingerprint
            contract_updates[instance_id] = (row["artifact_contract_json"], current_fingerprint)
            refresh_command = module_descriptor(row["module"]).refresh_command
            if changed and refresh_command is not None:
                row["command_json"] = json.dumps(
                    refresh_command(tuple(json.loads(row["command_json"])), contract["processing"])
                )
                command_updates[instance_id] = row["command_json"]
    config_records = {int(row["id"]): dict(row) for row in snapshot.configurations}
    config_values = {
        lineage_id: yaml.safe_load(record["resolved_yaml"]) or {}
        for lineage_id, record in config_records.items()
    }
    upstream: dict[int, list[int]] = {}
    for instance_id, upstream_id in dependencies:
        upstream.setdefault(instance_id, []).append(upstream_id)

    states: dict[int, tuple[str, str]] = {}
    input_updates: dict[int, str] = {}
    raw_runs_by_participant: dict[tuple[str, str], tuple] = {}
    session_inventories: dict[tuple[Path, bool], object] = {}
    completion_bounds: dict[int, tuple[int, int]] = {}
    remaining = set(by_id)
    while remaining:
        progressed = False
        for instance_id in tuple(remaining):
            parents = upstream.get(instance_id, [])
            if any(parent in remaining for parent in parents):
                continue
            row = by_id[instance_id]
            registered_only = compiled or row.get("execution_branch") not in (None, "main")
            project = str(row["project"])
            participant = str(row["participant"])
            subject_dir = registry.paths.bids_root / project / f"sub-{participant}"
            participant_key = (project, participant)
            command = [str(value) for value in json.loads(row["command_json"])]
            expected_module = {
                "anat": "nro.modules.anat",
                "func": "nro.modules.func",
                "clean": "nro.modules.clean",
            }.get(str(row["module"]))
            managed_direct_inputs = not registered_only and bool(
                expected_module and expected_module in command
            )
            direct_universe_error: str | None = None
            if managed_direct_inputs and row["module"] in {"func", "clean"}:
                try:
                    if participant_key not in raw_runs_by_participant:
                        raw_runs_by_participant[participant_key] = discover_raw_runs(subject_dir)
                    runs = raw_runs_by_participant[participant_key]
                    entities = json.loads(row["entities_json"])
                    source_entities = {
                        key: value
                        for key, value in entities.items()
                        if key not in {"space", "smoothing"}
                    }
                    matches = [run for run in runs if dict(run.entities) == source_entities]
                    if len(matches) != 1:
                        raise ValueError(
                            f"expected one raw run for {entities}, found {len(matches)}"
                        )
                    config = config_values[int(row["configuration_lineage_id"])]
                    if row["module"] == "func":
                        sdc_from_sbref_pair = bool(config["func"]["sdc_from_sbref_pair"])
                        inventory_key = (
                            matches[0].path.parent.parent.resolve(),
                            not sdc_from_sbref_pair,
                        )
                        if inventory_key not in session_inventories:
                            session_inventories[inventory_key] = load_session_inventory(
                                matches[0],
                                include_fmaps=not sdc_from_sbref_pair,
                            )
                        expected_paths = resolved_func_inputs(
                            matches[0],
                            sdc_from_sbref_pair=sdc_from_sbref_pair,
                            session_inventory=session_inventories[inventory_key],
                        )
                    else:
                        expected_paths = clean_direct_inputs(matches[0], config)
                    recorded_paths = {
                        str(Path(value).resolve()) for value in json.loads(row["input_paths_json"])
                    }
                    current_paths = {str(Path(value).resolve()) for value in expected_paths}
                    if current_paths != recorded_paths:
                        input_updates[instance_id] = json.dumps(sorted(current_paths))
                        direct_universe_error = (
                            "Selected direct input set changed: expected "
                            f"{len(current_paths)} file(s), instance records {len(recorded_paths)}"
                        )
                except (KeyError, OSError, ValueError) as error:
                    direct_universe_error = f"Could not reassess selected direct inputs: {error}"
            elif managed_direct_inputs and row["module"] == "anat":
                expected_paths = {
                    str(path.resolve()) for path in raw_anatomical_inputs(subject_dir)
                }
                recorded_paths = {
                    str(Path(value).resolve()) for value in json.loads(row["input_paths_json"])
                }
                if expected_paths != recorded_paths:
                    input_updates[instance_id] = json.dumps(sorted(expected_paths))
                    direct_universe_error = (
                        "Selected anatomical input set changed: expected "
                        f"{len(expected_paths)} file(s), instance records {len(recorded_paths)}"
                    )
            multirun_error: str | None = None
            descriptor = (
                SimpleNamespace(direct_inputs=None, select_runs=None)
                if registered_only
                else module_descriptor(row["module"])
            )
            package = (
                f"nro.modules.{row['module']}"
                if row["module"]
                in {"anat", "func", "clean", "dynconn", "microparcellation", "networks"}
                else f"nro.{row['module']}"
            )
            if descriptor.direct_inputs is not None and package in command:
                try:
                    if participant_key not in raw_runs_by_participant:
                        raw_runs_by_participant[participant_key] = discover_raw_runs(subject_dir)
                    paths = descriptor.direct_inputs(
                        raw_runs_by_participant[participant_key],
                        config_values[int(row["configuration_lineage_id"])],
                        participant,
                        json.loads(row["entities_json"]),
                    )
                    current_paths = {str(path.resolve()) for path in paths}
                    recorded_paths = {
                        str(Path(path).resolve()) for path in json.loads(row["input_paths_json"])
                    }
                    if current_paths != recorded_paths:
                        input_updates[instance_id] = json.dumps(sorted(current_paths))
                        direct_universe_error = "Selected direct input set changed"
                except (KeyError, OSError, ValueError) as error:
                    direct_universe_error = f"Could not reassess selected direct inputs: {error}"
            if not registered_only and (
                row["module"] in {"dynconn", "microparcellation"}
                or descriptor.select_runs is not None
            ):
                try:
                    config = config_values[int(row["configuration_lineage_id"])]
                    if participant_key not in raw_runs_by_participant:
                        raw_runs_by_participant[participant_key] = discover_raw_runs(subject_dir)
                    expected_runs = raw_runs_by_participant[participant_key]
                    if descriptor.select_runs is not None:
                        expected_runs = descriptor.select_runs(
                            expected_runs,
                            config,
                            row["participant"],
                            entities=json.loads(row["entities_json"]),
                        )
                    input_filter = config.get("input_filter", {})
                    expected_entities = {
                        json.dumps(dict(run.entities), sort_keys=True)
                        for run in expected_runs
                        if matches_filter(run.entities, input_filter)
                    }
                    recorded_entities = {
                        json.dumps(
                            {
                                key: value
                                for key, value in json.loads(by_id[parent]["entities_json"]).items()
                                if key not in {"space", "smoothing"}
                            },
                            sort_keys=True,
                        )
                        for parent in parents
                        if by_id[parent]["module"] == descriptor.upstream_modules[0]
                    }
                    if expected_entities != recorded_entities:
                        multirun_error = (
                            "Selected raw run universe changed: expected "
                            f"{len(expected_entities)} run(s), dependency graph records "
                            f"{len(recorded_entities)}; submit python -m "
                            "nro.bin.run to plan the "
                            "new upstream instance set"
                        )
                except (KeyError, OSError, ValueError) as error:
                    multirun_error = f"Could not reassess selected raw run universe: {error}"
            manifest_path = Path(row["manifest_path"])
            manifest = _read_manifest(manifest_path)
            certificate_bounds: tuple[int, int] | None = None
            if instance_id in changed_contracts:
                state = ("stale", "Current module processing contract changed")
            elif manifest is None:
                if registered_only and row["artifact_state"] != "fresh":
                    state = (row["artifact_state"], row["artifact_reason"])
                elif direct_universe_error:
                    state = ("stale", direct_universe_error)
                elif multirun_error:
                    state = ("stale", multirun_error)
                elif any(states[parent][0] != "fresh" for parent in parents):
                    state = ("stale", "An upstream derivative is missing or stale")
                else:
                    public_contract_mismatch = False
                    try:
                        recovered, recovery_error = _public_derivative_completion(
                            row, registry, compiled=registered_only
                        )
                    except _PublicDerivativeContractMismatch as error:
                        recovered, recovery_error = None, str(error)
                        public_contract_mismatch = True
                    if recovered is None:
                        state = (
                            "stale" if public_contract_mismatch else "missing",
                            recovery_error,
                        )
                    else:
                        try:
                            direct_paths = [
                                Path(value).resolve()
                                for value in json.loads(row["input_paths_json"])
                            ]
                            missing_input = next(
                                (path for path in direct_paths if not path.is_file()), None
                            )
                            if missing_input is not None:
                                state = ("stale", f"Direct input is missing: {missing_input}")
                            else:
                                newest_direct = max(
                                    (path.stat().st_mtime_ns for path in direct_paths),
                                    default=0,
                                )
                                newest_upstream = max(
                                    (completion_bounds[parent][1] for parent in parents),
                                    default=0,
                                )
                                if newest_direct > recovered.oldest_completion_ns:
                                    state = (
                                        "stale",
                                        "A direct input is newer than the public completion evidence",
                                    )
                                elif newest_upstream > recovered.oldest_completion_ns:
                                    state = (
                                        "stale",
                                        "An upstream derivative is newer than the public completion evidence",
                                    )
                                else:
                                    state = (
                                        "fresh",
                                        "Public derivative outputs validate; private orchestration provenance is unavailable",
                                    )
                                    completion_bounds[instance_id] = (
                                        recovered.oldest_completion_ns,
                                        recovered.newest_completion_ns,
                                    )
                        except (OSError, ValueError) as error:
                            state = (
                                "stale",
                                f"Could not validate direct input timestamps: {error}",
                            )
            elif manifest.get("manifest_version") != MANIFEST_VERSION:
                state = ("stale", "Completion manifest version changed")
            else:
                # The instance key binds the module, entities, and immutable
                # instance-level configuration instance.  Implementation hashes
                # are retained as provenance only: editing the scheduler,
                # runner, or code after a derivative completed is not direct
                # evidence that the derivative on disk is stale.  Direct
                # inputs, upstream generations, and artifact fingerprints
                # below remain the authoritative freshness contract.
                configuration_compatible = (manifest.get("configuration") or {}).get(
                    "fingerprint"
                ) == config_records[int(row["configuration_lineage_id"])]["config_fingerprint"]
                recorded_configuration = manifest.get("configuration") or {}
                if not registered_only and isinstance(recorded_configuration.get("resolved"), dict):
                    from nro.configuration.store import configuration_fingerprint

                    descriptor = module_descriptor(row["module"])
                    try:
                        current_values = config_values[int(row["configuration_lineage_id"])]
                        identifier = recorded_configuration["id"]
                        configuration_compatible = configuration_fingerprint(
                            descriptor.configuration_class,
                            identifier,
                            recorded_configuration["resolved"],
                            scientific=True,
                        ) == configuration_fingerprint(
                            descriptor.configuration_class,
                            identifier,
                            current_values,
                            scientific=True,
                        )
                    except (ValueError, KeyError, TypeError):
                        configuration_compatible = False
                if not _certificate_matches_contract(
                    manifest, row["artifact_fingerprint"], compiled=registered_only
                ):
                    state = ("stale", "Instance contract changed")
                elif (
                    manifest.get("revision_fingerprint") != row["revision_fingerprint"]
                    and not configuration_compatible
                ):
                    state = ("stale", "Instance contract or resolved configuration changed")
                # The runtime snapshot is orchestration metadata, not a derivative
                # input.  A missing snapshot must not make otherwise valid public
                # derivatives stale.  A present-but-changed snapshot is covered by
                # the instance/configuration revision check above.
                elif direct_universe_error:
                    state = ("stale", direct_universe_error)
                elif multirun_error:
                    state = ("stale", multirun_error)
                elif any(states[parent][0] != "fresh" for parent in parents):
                    state = ("stale", "An upstream derivative is missing or stale")
                else:
                    recorded_upstream = {
                        int(item["instance_id"]): int(item["generation"])
                        for item in manifest.get("upstream", [])
                    }
                    current_upstream = {
                        parent: int(by_id[parent]["current_generation"]) for parent in parents
                    }
                    if recorded_upstream != current_upstream:
                        state = ("stale", "Upstream derivative set or generation changed")
                    else:
                        input_error: str | None = None
                        try:
                            current_inputs = inventory(json.loads(row["input_paths_json"]))
                        except (OSError, ValueError) as error:
                            current_inputs = []
                            input_error = f"Direct input is missing or unreadable: {error}"
                        recorded_inputs = manifest.get("inputs", [])
                        if input_error:
                            state = ("stale", input_error)
                        elif current_inputs != recorded_inputs:
                            state = ("stale", "Direct input set or fingerprint changed")
                        else:
                            public_outputs = manifest.get("public_outputs", [])
                            if not public_outputs:
                                state = (
                                    "missing",
                                    "Completion certificate contains no public artifacts",
                                )
                            else:
                                output_error = next(
                                    (
                                        reason
                                        for okay, reason in (
                                            _same_file(item) for item in public_outputs
                                        )
                                        if not okay
                                    ),
                                    None,
                                )
                                private_error = next(
                                    (
                                        reason
                                        for okay, reason in (
                                            _same_existing_private_file(item)
                                            for item in manifest.get("private_artifacts", [])
                                            if not _is_orchestration_control_artifact(
                                                Path(str(item.get("path", ""))), registry
                                            )
                                        )
                                        if not okay
                                    ),
                                    None,
                                )
                                if output_error:
                                    state = ("stale", output_error)
                                elif private_error:
                                    state = ("stale", f"Private artifact changed: {private_error}")
                                else:
                                    state = ("fresh", "Completion certificate artifacts validate")
                                    # The certificate's public artifacts have
                                    # just been stat'ed/validated.  Their live
                                    # mtimes are the completion bounds needed
                                    # by descendants, so do not recursively
                                    # rescan the public derivative tree.
                                    output_mtimes = [
                                        Path(str(item["path"])).stat().st_mtime_ns
                                        for item in public_outputs
                                    ]
                                    certificate_bounds = (min(output_mtimes), max(output_mtimes))
            if state[0] == "fresh" and instance_id not in completion_bounds:
                if certificate_bounds is not None:
                    completion_bounds[instance_id] = certificate_bounds
                else:
                    try:
                        recovered, _recovery_error = _public_derivative_completion(
                            row, registry, compiled=registered_only
                        )
                    except _PublicDerivativeContractMismatch:
                        recovered = None
                    if recovered is not None:
                        completion_bounds[instance_id] = (
                            recovered.oldest_completion_ns,
                            recovered.newest_completion_ns,
                        )
                    elif manifest_path.is_file():
                        stamp = manifest_path.stat().st_mtime_ns
                        completion_bounds[instance_id] = (stamp, stamp)
            states[instance_id] = state
            remaining.remove(instance_id)
            progressed = True
        if not progressed:
            for instance_id in remaining:
                states[instance_id] = ("stale", "Instance dependency cycle detected")
            break

    return AssessmentReport(
        snapshot.fingerprint,
        tuple(
            {
                "id": instance_id,
                "state": state,
                "reason": reason,
                "contract": json.loads(contract_updates[instance_id][0])
                if instance_id in contract_updates
                else None,
                "command": json.loads(command_updates[instance_id])
                if instance_id in command_updates
                else None,
                "inputs": json.loads(input_updates[instance_id])
                if instance_id in input_updates
                else None,
            }
            for instance_id, (state, reason) in sorted(states.items())
        ),
    )


def _private_step_artifacts(
    registry: Registry,
    *,
    attempt_id: int,
    output_root: Path,
) -> list[dict]:
    """Inventory absolute, persistent step outputs outside the derivative root.

    Runner ledgers are intentionally best-effort diagnostics, so a malformed
    or relative output entry is simply not eligible for instance-level validation.
    Module-level staleness still owns all such details whenever the module is
    invoked.
    """
    with registry.connection() as db:
        row = db.execute("SELECT log_path FROM attempts WHERE id=?", (attempt_id,)).fetchone()
    if row is None or not row["log_path"]:
        return []
    ledger = Path(str(row["log_path"])).parent / "current-steps.json"
    value = _read_manifest(ledger)
    if value is None:
        return []
    public_root = output_root.resolve()
    paths: set[Path] = set()
    for step in value.values():
        if not isinstance(step, dict):
            continue
        cwd = step.get("cwd")
        for item in step.get("outputs", []):
            if not isinstance(item, str) or not item.strip():
                continue
            path = Path(item).expanduser()
            if not path.is_absolute():
                # Relative output paths are only meaningful when the ledger
                # recorded a concrete working directory.
                if not cwd:
                    continue
                path = Path(str(cwd)).expanduser() / path
            try:
                resolved = path.resolve()
                resolved.relative_to(public_root)
            except ValueError:
                if resolved.is_file() and not _is_orchestration_control_artifact(
                    resolved, registry
                ):
                    paths.add(resolved)
    return inventory(paths)


def record_completion(
    registry: Registry,
    *,
    instance_id: int,
    attempt_id: int,
    outputs: Iterable[str | Path],
) -> dict:
    """Write a completion manifest last, then advance the registry generation."""
    with registry.connection() as db:
        instance = dict(
            db.execute(
                """SELECT t.*, ci.config_id, ci.config_fingerprint,
                          ci.lineage_fingerprint, ci.resolved_yaml
                   FROM instances t JOIN configuration_lineages ci ON ci.id=t.configuration_lineage_id
                   WHERE t.id=?""",
                (instance_id,),
            ).fetchone()
        )
        dependency_state.check_completion(db, instance, attempt_id)
        execution = db.execute(
            "SELECT provenance_json,command_json FROM attempt_execution WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        parents = [
            dict(row)
            for row in db.execute(
                """
                SELECT t.id, t.current_generation, t.manifest_path
                FROM instance_dependencies td JOIN instances t ON t.id=td.upstream_instance_id
                WHERE td.instance_id=? ORDER BY t.id
                """,
                (instance_id,),
            )
        ]
    output_records = inventory(outputs)
    if not output_records:
        raise RuntimeError(
            f"Instance produced no discoverable outputs under {instance['output_root']}"
        )
    private_records = _private_step_artifacts(
        registry,
        attempt_id=attempt_id,
        output_root=Path(instance["output_root"]),
    )
    input_records = inventory(json.loads(instance["input_paths_json"]))
    generation = int(instance["current_generation"]) + 1
    runtime_config = file_record(instance["runtime_config_path"])
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "instance_id": instance_id,
        "instance_key": instance["instance_key"],
        "module": instance["module"],
        "project": instance["project"],
        "participant": instance["participant"],
        "entities": json.loads(instance["entities_json"]),
        "configuration_lineage_id": instance["configuration_lineage_id"],
        "revision_fingerprint": instance["revision_fingerprint"],
        "artifact_contract": json.loads(instance["artifact_contract_json"]),
        "artifact_fingerprint": instance["artifact_fingerprint"],
        "configuration": {
            "id": instance["config_id"],
            "fingerprint": instance["config_fingerprint"],
            "lineage_fingerprint": instance["lineage_fingerprint"],
            "resolved": yaml.safe_load(instance["resolved_yaml"]) or {},
        },
        "runtime_config": runtime_config,
        "software": {
            "name": "nro",
            "python": sys.version,
            "executable": sys.executable,
            "command": json.loads(instance["command_json"]),
        },
        "generation": generation,
        "attempt_id": attempt_id,
        "completed_at": utcnow(),
        "inputs": input_records,
        "upstream": [
            {
                "instance_id": int(parent["id"]),
                "generation": int(parent["current_generation"]),
                "manifest": parent["manifest_path"],
            }
            for parent in parents
        ],
        "public_outputs": output_records,
        "private_artifacts": private_records,
    }
    path = Path(instance["manifest_path"])
    if execution is not None:
        manifest["implementation"] = json.loads(execution["provenance_json"])
        manifest["software"]["command"] = json.loads(execution["command_json"])
    ensure_shared_directory(path.parent)
    write_instance_ownership(registry, instance_id, attempt_id=attempt_id)
    with registry.connection(write=True) as db:
        dependency_state.check_completion(db, instance, attempt_id)
        atomic_write_json(path, manifest, sort_keys=True, mode=0o664, durable=True)
        db.execute(
            """
            UPDATE instances SET artifact_state='fresh', artifact_reason='Completed successfully',
                current_generation=?, updated_at=? WHERE id=?
            """,
            (generation, utcnow(), instance_id),
        )
        db.executemany(
            """UPDATE instance_dependencies SET required_generation=?
                          WHERE instance_id=? AND upstream_instance_id=?""",
            [(parent["current_generation"], instance_id, parent["id"]) for parent in parents],
        )
        for item in input_records:
            db.execute(
                """INSERT INTO artifacts(instance_id, attempt_id, direction, path, size, mtime_ns,
                   digest_algorithm, digest, metadata_json) VALUES (?, ?, 'input', ?, ?, ?, ?, ?, '{}')""",
                (
                    instance_id,
                    attempt_id,
                    item["path"],
                    item["size"],
                    item["mtime_ns"],
                    "sha256" if "sha256" in item else None,
                    item.get("sha256"),
                ),
            )
        for item in output_records:
            db.execute(
                """INSERT INTO artifacts(instance_id, attempt_id, direction, path, size, mtime_ns,
                   digest_algorithm, digest, metadata_json) VALUES (?, ?, 'output', ?, ?, ?, ?, ?, '{}')""",
                (
                    instance_id,
                    attempt_id,
                    item["path"],
                    item["size"],
                    item["mtime_ns"],
                    "sha256" if "sha256" in item else None,
                    item.get("sha256"),
                ),
            )
    return manifest
