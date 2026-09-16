"""Assess artifact freshness from completion certificates and the filesystem.

The orchestration layer validates direct inputs, upstream work-item generations,
and public artifacts below a work item's derivative output root. Private
intermediates can invalidate an artifact when changed, but their absence is
tolerated because a module can recreate them when needed.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from nro.configuration.store import fingerprint
from nro.engine.bids import discover_raw_runs, matches_filter
from nro.orchestration.assessment import (
    AssessmentConflict,
    AssessmentReport,
    AssessmentSnapshot,
    apply_assessment,
    capture_assessment,
)
from nro.orchestration.ownership import missing_work_item_ownership, write_work_item_ownership
from nro.orchestration.registry import Registry

MANIFEST_VERSION = 5
SMALL_DIGEST_LIMIT = 1024 * 1024
COMPLETION_VISIBILITY_TIMEOUT = 30.0
COMPLETION_VISIBILITY_POLL_INTERVAL = 0.25


def file_record(path: str | Path) -> dict:
    """Describe one completed file for later integrity checks."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if resolved.exists():
            raise ValueError(f"Completion artifact is not a regular file: {resolved}")
        raise ValueError(f"Completion artifact is missing: {resolved}")
    stat = resolved.stat()
    record = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= SMALL_DIGEST_LIMIT:
        record["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return record


def inventory(paths: Iterable[str | Path]) -> list[dict]:
    """Return deterministic integrity records for unique resolved files."""
    return [file_record(path) for path in sorted({Path(path).resolve() for path in paths})]


def _completion_output_inventory(paths: Iterable[str | Path]) -> list[dict]:
    """Read completed outputs after bounded shared-filesystem visibility retries.

    A worker publishes each file atomically and checks it before asking the scheduler
        to certify the work item. Some shared filesystems can nevertheless expose a newly
    replaced directory entry to another node only after a short delay. Retry missing
    paths here, at the authoritative scheduler read, while rejecting visible
    non-files immediately.
    """
    resolved = tuple(sorted({Path(path).expanduser().resolve() for path in paths}))
    deadline = time.monotonic() + COMPLETION_VISIBILITY_TIMEOUT
    while True:
        try:
            return inventory(resolved)
        except ValueError:
            if any(path.exists() and not path.is_file() for path in resolved):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(COMPLETION_VISIBILITY_POLL_INTERVAL)


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
    interference and should make the owning work item stale.
    """
    if not Path(record["path"]).is_file():
        return True, None
    return _same_file(record)


def _is_orchestration_control_artifact(path: Path, registry: Registry) -> bool:
    """Control records describe execution; they are never work-item inputs.

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
    descriptor = module_descriptor(str(row["module"]))
    current = descriptor.processing_for(json.loads(row["entities_json"]))
    recorded_processing = contract.get("processing", {})
    if "source_markup" in recorded_processing:
        current["source_markup"] = recorded_processing["source_markup"]
    for key in descriptor.dynamic_processing_keys:
        if key in recorded_processing:
            current[key] = recorded_processing[key]
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
    work_items = registry.work_item_rows(read_only=True)
    dependencies = registry.work_item_dependencies(read_only=True)
    if projects is not None:
        selected_projects = set(projects)
        selected = {
            int(row["id"]) for row in work_items if str(row["project"]) in selected_projects
        }
        work_items = [row for row in work_items if int(row["id"]) in selected]
        dependencies = [
            (work_item_id, upstream_id)
            for work_item_id, upstream_id in dependencies
            if work_item_id in selected and upstream_id in selected
        ]

    by_id = {int(row["id"]): row for row in work_items}
    upstream: dict[int, list[int]] = {}
    for work_item_id, upstream_id in dependencies:
        upstream.setdefault(work_item_id, []).append(upstream_id)

    states: dict[int, tuple[str, str]] = {}
    remaining = set(by_id)
    while remaining:
        progressed = False
        for work_item_id in tuple(remaining):
            parents = upstream.get(work_item_id, [])
            if any(parent in remaining for parent in parents):
                continue
            row = by_id[work_item_id]
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
            states[work_item_id] = state
            remaining.remove(work_item_id)
            progressed = True
        if not progressed:
            for work_item_id in remaining:
                states[work_item_id] = ("stale", "Work-item dependency cycle detected")
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
    """Validate the work item's fixed public derivative contract.

    The expected root manifests are fixed when the work item DAG is constructed.
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
                # never discovered work item outputs.
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
    work_item_ids: Iterable[int] | None = None,
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
    work_item_ids = None if work_item_ids is None else tuple(work_item_ids)
    projects = None if projects is None else tuple(projects)
    for attempt in range(3):
        snapshot = capture_assessment(registry, work_item_ids=work_item_ids, projects=projects)
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
    fresh = [work_item_id for work_item_id, (state, _reason) in states.items() if state == "fresh"]
    for work_item_id in missing_work_item_ownership(registry, fresh):
        write_work_item_ownership(registry, work_item_id)
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
        from nro.configuration.hardware import gradient_unwarping_records
        from nro.configuration.markup import MarkupStore
        from nro.modules.anat.planning import raw_anatomical_inputs
        from nro.modules.clean.planning import clean_direct_inputs
        from nro.modules.func.contract import final_resampling_contract
        from nro.modules.func.planning import load_session_inventory, resolved_func_inputs
        from nro.orchestration.catalog import module_descriptor
    work_items = deepcopy(snapshot.work_items)
    dependencies = [
        (edge["work_item_id"], edge["upstream_work_item_id"]) for edge in snapshot.dependencies
    ]
    registry = snapshot
    by_id = {int(row["id"]): row for row in work_items}
    contract_updates: dict[int, tuple[str, str]] = {}
    changed_contracts: set[int] = set()
    command_updates: dict[int, str] = {}

    def uses_registered_contract(row: dict) -> bool:
        recovered = bool(row.get("branch_owned")) and row.get("execution_branch") is None
        return compiled or recovered or row.get("execution_branch") not in (None, "main")

    for work_item_id, row in by_id.items():
        if uses_registered_contract(row):
            continue
        contract, current_fingerprint, changed = _current_contract(row)
        if changed:
            changed_contracts.add(work_item_id)
        if changed or fingerprint(json.loads(row["artifact_contract_json"])) != current_fingerprint:
            row["artifact_contract_json"] = json.dumps(
                contract, sort_keys=True, separators=(",", ":")
            )
            row["artifact_fingerprint"] = current_fingerprint
            contract_updates[work_item_id] = (row["artifact_contract_json"], current_fingerprint)
            refresh_command = module_descriptor(row["module"]).refresh_command
            if changed and refresh_command is not None:
                row["command_json"] = json.dumps(
                    refresh_command(tuple(json.loads(row["command_json"])), contract["processing"])
                )
                command_updates[work_item_id] = row["command_json"]
    config_records = {int(row["id"]): dict(row) for row in snapshot.configurations}
    config_values = {
        lineage_id: yaml.safe_load(record["resolved_yaml"]) or {}
        for lineage_id, record in config_records.items()
    }
    upstream: dict[int, list[int]] = {}
    for work_item_id, upstream_id in dependencies:
        upstream.setdefault(work_item_id, []).append(upstream_id)

    states: dict[int, tuple[str, str]] = {}
    input_updates: dict[int, str] = {}
    raw_runs_by_participant: dict[tuple[str, str, str | None], tuple] = {}
    markups: dict[tuple[str, str, str | None], object] = {}
    session_inventories: dict[tuple[Path, bool, str | None], object] = {}
    completion_bounds: dict[int, tuple[int, int]] = {}
    remaining = set(by_id)
    while remaining:
        progressed = False
        for work_item_id in tuple(remaining):
            parents = upstream.get(work_item_id, [])
            if any(parent in remaining for parent in parents):
                continue
            row = by_id[work_item_id]
            registered_only = uses_registered_contract(row)
            project = str(row["project"])
            participant = str(row["participant"])
            subject_dir = registry.paths.bids_root / project / f"sub-{participant}"
            config = config_values[int(row["module_lineage_id"])]
            module_config = (
                config[row["module"]] if isinstance(config.get(row["module"]), dict) else config
            )
            markup_key = (project, participant, module_config.get("markup"))
            if not registered_only and markup_key not in markups:
                markups[markup_key] = MarkupStore().subject(
                    module_config.get("markup"), project, subject_dir
                )
            source_markup = None if registered_only else markups[markup_key]
            run_key = markup_key
            command = [str(value) for value in json.loads(row["command_json"])]
            descriptor = None if registered_only else module_descriptor(row["module"])
            managed_direct_inputs = descriptor is not None and bool(
                descriptor.execution_module in command
            )
            expected_paths: tuple[Path, ...] = ()
            direct_universe_error: str | None = None
            if managed_direct_inputs and row["module"] in {"func", "clean"}:
                try:
                    if run_key not in raw_runs_by_participant:
                        raw_runs_by_participant[run_key] = discover_raw_runs(
                            subject_dir, markup=source_markup
                        )
                    runs = raw_runs_by_participant[run_key]
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
                    if row["module"] == "func":
                        sdc_from_sbref_pair = bool(module_config["sdc_from_sbref_pair"])
                        inventory_key = (
                            matches[0].path.parent.parent.resolve(),
                            not sdc_from_sbref_pair,
                            module_config.get("markup"),
                        )
                        if inventory_key not in session_inventories:
                            session_inventories[inventory_key] = load_session_inventory(
                                matches[0], markup=source_markup
                            )
                        expected_paths = resolved_func_inputs(
                            matches[0],
                            sdc_from_sbref_pair=sdc_from_sbref_pair,
                            session_inventory=session_inventories[inventory_key],
                            markup=source_markup,
                        )
                    else:
                        expected_paths = clean_direct_inputs(
                            matches[0], module_config, markup=source_markup
                        )
                    recorded_paths = {
                        str(Path(value).resolve()) for value in json.loads(row["input_paths_json"])
                    }
                    current_paths = {str(Path(value).resolve()) for value in expected_paths}
                    if current_paths != recorded_paths:
                        input_updates[work_item_id] = json.dumps(sorted(current_paths))
                        direct_universe_error = (
                            "Selected direct input set changed: expected "
                            f"{len(current_paths)} file(s), work item records {len(recorded_paths)}"
                        )
                except (KeyError, OSError, ValueError) as error:
                    direct_universe_error = f"Could not reassess selected direct inputs: {error}"
            elif managed_direct_inputs and row["module"] == "anat":
                expected_paths = {
                    str(path.resolve())
                    for path in raw_anatomical_inputs(subject_dir, source_markup)
                }
                recorded_paths = {
                    str(Path(value).resolve()) for value in json.loads(row["input_paths_json"])
                }
                if expected_paths != recorded_paths:
                    input_updates[work_item_id] = json.dumps(sorted(expected_paths))
                    direct_universe_error = (
                        "Selected anatomical input set changed: expected "
                        f"{len(expected_paths)} file(s), work item records {len(recorded_paths)}"
                    )
            if (
                not registered_only
                and managed_direct_inputs
                and expected_paths
                and row["module"] in {"anat", "func"}
            ):
                images = [
                    Path(path)
                    for path in expected_paths
                    if Path(path).name.endswith((".nii", ".nii.gz"))
                ]
                records, resolutions = gradient_unwarping_records(
                    images,
                    mode=str(module_config.get("gradient_unwarping", "off")),
                    markup=source_markup,
                )
                expected_dynamic = {"gradient_unwarping": records}
                if row["module"] == "func":
                    bold_path = matches[0].path.expanduser().absolute()
                    bold_resolution = resolutions.get(bold_path)
                    expected_dynamic["final_resampling"] = final_resampling_contract(
                        gradient_unwarping=bool(bold_resolution and bold_resolution.applied)
                    )
                contract = json.loads(row["artifact_contract_json"])
                processing = dict(contract.get("processing", {}))
                if any(processing.get(key) != value for key, value in expected_dynamic.items()):
                    processing.update(expected_dynamic)
                    contract["processing"] = processing
                    current_fingerprint = fingerprint(contract)
                    row["artifact_contract_json"] = json.dumps(
                        contract, sort_keys=True, separators=(",", ":")
                    )
                    row["artifact_fingerprint"] = current_fingerprint
                    contract_updates[work_item_id] = (
                        row["artifact_contract_json"],
                        current_fingerprint,
                    )
                    changed_contracts.add(work_item_id)
            multirun_error: str | None = None
            if (
                descriptor is not None
                and descriptor.direct_inputs is not None
                and descriptor.execution_module in command
            ):
                try:
                    if run_key not in raw_runs_by_participant:
                        raw_runs_by_participant[run_key] = discover_raw_runs(
                            subject_dir, markup=source_markup
                        )
                    paths = descriptor.direct_inputs(
                        raw_runs_by_participant[run_key],
                        module_config,
                        participant,
                        json.loads(row["entities_json"]),
                        markup=source_markup,
                    )
                    current_paths = {str(path.resolve()) for path in paths}
                    recorded_paths = {
                        str(Path(path).resolve()) for path in json.loads(row["input_paths_json"])
                    }
                    if current_paths != recorded_paths:
                        input_updates[work_item_id] = json.dumps(sorted(current_paths))
                        direct_universe_error = "Selected direct input set changed"
                except (KeyError, OSError, ValueError) as error:
                    direct_universe_error = f"Could not reassess selected direct inputs: {error}"
            if not registered_only and (
                row["module"] in {"dynconn", "microparcellation"}
                or descriptor.select_runs is not None
            ):
                try:
                    if run_key not in raw_runs_by_participant:
                        raw_runs_by_participant[run_key] = discover_raw_runs(
                            subject_dir, markup=source_markup
                        )
                    expected_runs = raw_runs_by_participant[run_key]
                    if descriptor.select_runs is not None:
                        expected_runs = descriptor.select_runs(
                            expected_runs,
                            module_config,
                            row["participant"],
                            entities=json.loads(row["entities_json"]),
                        )
                    input_filter = module_config.get("input_filter", {})
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
                            "new upstream work item set"
                        )
                except (KeyError, OSError, ValueError) as error:
                    multirun_error = f"Could not reassess selected raw run universe: {error}"
            manifest_path = Path(row["manifest_path"])
            manifest = _read_manifest(manifest_path)
            certificate_bounds: tuple[int, int] | None = None
            if work_item_id in changed_contracts:
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
                                    completion_bounds[work_item_id] = (
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
                # The work item key binds the module, entities, and immutable
                # work-item-level configuration. Implementation hashes
                # are retained as provenance only: editing the scheduler,
                # runner, or code after a derivative completed is not direct
                # evidence that the derivative on disk is stale.  Direct
                # inputs, upstream generations, and artifact fingerprints
                # below remain the authoritative freshness contract.
                configuration_compatible = (manifest.get("configuration") or {}).get(
                    "fingerprint"
                ) == config_records[int(row["module_lineage_id"])]["config_fingerprint"]
                recorded_configuration = manifest.get("configuration") or {}
                if not registered_only and isinstance(recorded_configuration.get("resolved"), dict):
                    from nro.configuration.store import configuration_fingerprint

                    descriptor = module_descriptor(row["module"])
                    try:
                        current_values = config_values[int(row["module_lineage_id"])]
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
                    state = ("stale", "Work-item contract changed")
                elif (
                    manifest.get("revision_fingerprint") != row["revision_fingerprint"]
                    and not configuration_compatible
                ):
                    state = ("stale", "Work-item contract or resolved configuration changed")
                # The runtime snapshot is orchestration metadata, not a derivative
                # input.  A missing snapshot must not make otherwise valid public
                # derivatives stale.  A present-but-changed snapshot is covered by
                # the work item/configuration revision check above.
                elif direct_universe_error:
                    state = ("stale", direct_universe_error)
                elif multirun_error:
                    state = ("stale", multirun_error)
                elif any(states[parent][0] != "fresh" for parent in parents):
                    state = ("stale", "An upstream derivative is missing or stale")
                else:
                    recorded_upstream = {
                        int(item["work_item_id"]): int(item["generation"])
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
            if state[0] == "fresh" and work_item_id not in completion_bounds:
                if certificate_bounds is not None:
                    completion_bounds[work_item_id] = certificate_bounds
                else:
                    try:
                        recovered, _recovery_error = _public_derivative_completion(
                            row, registry, compiled=registered_only
                        )
                    except _PublicDerivativeContractMismatch:
                        recovered = None
                    if recovered is not None:
                        completion_bounds[work_item_id] = (
                            recovered.oldest_completion_ns,
                            recovered.newest_completion_ns,
                        )
                    elif manifest_path.is_file():
                        stamp = manifest_path.stat().st_mtime_ns
                        completion_bounds[work_item_id] = (stamp, stamp)
            states[work_item_id] = state
            remaining.remove(work_item_id)
            progressed = True
        if not progressed:
            for work_item_id in remaining:
                states[work_item_id] = ("stale", "Work-item dependency cycle detected")
            break

    return AssessmentReport(
        snapshot.fingerprint,
        tuple(
            {
                "id": work_item_id,
                "state": state,
                "reason": reason,
                "contract": json.loads(contract_updates[work_item_id][0])
                if work_item_id in contract_updates
                else None,
                "command": json.loads(command_updates[work_item_id])
                if work_item_id in command_updates
                else None,
                "inputs": json.loads(input_updates[work_item_id])
                if work_item_id in input_updates
                else None,
            }
            for work_item_id, (state, reason) in sorted(states.items())
        ),
    )
