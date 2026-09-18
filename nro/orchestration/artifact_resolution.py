"""Compare scientific graphs independently of branch-owned storage locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from nro.configuration.store import fingerprint
from nro.orchestration.branches import BranchTopology
from nro.orchestration.contracts import WorkItemSpec


def _prefixes_overlap(first: str | None, second: str | None) -> bool:
    """Return whether two filename-prefix claims can select the same file."""
    if first is None or second is None:
        return True
    return first == second or first.startswith(second + "_") or second.startswith(first + "_")


def output_contracts_overlap(
    first_root: Path,
    first_prefix: str | None,
    first_expected: Sequence[Path],
    second_root: Path,
    second_prefix: str | None,
    second_expected: Sequence[Path],
) -> bool:
    """Return whether two public-output contracts can select the same path."""
    first_root = first_root.expanduser().resolve()
    second_root = second_root.expanduser().resolve()
    first_outputs = {path.expanduser().resolve() for path in first_expected}
    second_outputs = {path.expanduser().resolve() for path in second_expected}
    if first_outputs & second_outputs:
        return True
    if first_root != second_root:
        return False
    if first_prefix is not None and second_prefix is not None:
        return _prefixes_overlap(first_prefix, second_prefix)
    if first_prefix is not None:
        return any(
            path.parent == first_root
            and (path.name == first_prefix or path.name.startswith(first_prefix + "_"))
            for path in second_outputs
        )
    if second_prefix is not None:
        return any(
            path.parent == second_root
            and (path.name == second_prefix or path.name.startswith(second_prefix + "_"))
            for path in first_outputs
        )
    return False


def output_claims_overlap(first: WorkItemSpec, second: WorkItemSpec) -> bool:
    """Return whether two work items claim any common public output namespace."""
    return output_contracts_overlap(
        first.output_root,
        first.output_prefix,
        first.expected_outputs,
        second.output_root,
        second.output_prefix,
        second.expected_outputs,
    )


def validate_output_ownership(work_items: Sequence[WorkItemSpec]) -> None:
    """Reject graphs in which distinct vertices can publish the same path."""
    ordered = sorted(work_items, key=lambda item: item.key)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if output_claims_overlap(first, second):
                raise ValueError(
                    "Work-item output claims overlap: "
                    f"{first.key} ({first.output_root}, prefix={first.output_prefix!r}) and "
                    f"{second.key} ({second.output_root}, prefix={second.output_prefix!r})"
                )


def scientific_contracts(work_items: Sequence[WorkItemSpec]) -> dict[str, dict]:
    """Compile location-independent contracts for a complete work-item graph.

    Output members retain their relative paths. Direct source/resource paths
    remain exact. Paths inside a declared upstream artifact become references
    to that producer's scientific contract and relative member. No arbitrary
    path prefix, processing field, or dependency is discarded.
    """
    by_key = {item.key: item for item in work_items}
    if len(by_key) != len(work_items):
        raise ValueError("Work-item graph contains duplicate keys")
    validate_output_ownership(work_items)
    result: dict[str, dict] = {}
    pending = set(by_key)
    while pending:
        ready = sorted(key for key in pending if set(by_key[key].dependencies) <= result.keys())
        if not ready:
            raise ValueError("Work-item graph is cyclic or lacks a required dependency")
        for key in ready:
            spec = by_key[key]
            parents = [by_key[parent] for parent in sorted(set(spec.dependencies))]
            inputs = []
            for path in spec.input_paths:
                resolved = path.expanduser().resolve()
                references = []
                for parent in parents:
                    root = parent.output_root.expanduser().resolve()
                    if not resolved.is_relative_to(root):
                        continue
                    prefix = parent.output_prefix
                    if prefix and not (
                        resolved.name == prefix or resolved.name.startswith(prefix + "_")
                    ):
                        continue
                    references.append(
                        {
                            "artifact": fingerprint(result[parent.key]),
                            "member": str(resolved.relative_to(root)),
                        }
                    )
                if len(references) > 1:
                    raise ValueError(f"Direct input has ambiguous producer ownership: {path}")
                inputs.append(references[0] if references else {"source": str(resolved)})
            root = spec.output_root.expanduser().resolve()
            members = []
            for path in spec.expected_outputs:
                resolved = path.expanduser().resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Expected output escapes its artifact root: {path}")
                members.append(str(resolved.relative_to(root)))
            result[key] = {
                "project": spec.project,
                "participant": spec.participant,
                "module": spec.module,
                "entities": dict(sorted(spec.entities.items())),
                "configuration": spec.config_fingerprint,
                "dependencies": sorted(fingerprint(result[parent.key]) for parent in parents),
                "inputs": sorted(inputs, key=fingerprint),
                "output": {
                    "prefix": spec.output_prefix,
                    "expected": sorted(set(members)),
                    "format": spec.contract.output.format,
                },
                "processing": dict(spec.contract.processing),
            }
            pending.remove(key)
    return result


@dataclass(frozen=True)
class ArtifactCandidate:
    """An owned artifact offered for validation against a requested scientific contract."""

    branch: str
    key: str
    contract: Mapping
    generation: int
    root: Path
    evidence: Mapping
    contract_fingerprint: str | None = None


def select_artifact(
    topology: BranchTopology,
    branch: str,
    contract: Mapping,
    candidates: Sequence[ArtifactCandidate],
    *,
    validate: Callable[[ArtifactCandidate], bool],
    inherit: bool = True,
) -> ArtifactCandidate | None:
    """Choose the nearest scientifically compatible, currently validated artifact.

    A stale local artifact does not hide a fresh ancestor. Siblings and descendants
    are never eligible. Validation must check current output and input evidence;
    a saved success flag alone is not sufficient. None means compute locally.
    """
    expected = fingerprint(dict(contract))
    for owner in topology.ancestors(branch, inherit=inherit):
        eligible = [
            candidate
            for candidate in candidates
            if candidate.branch == owner
            and (candidate.contract_fingerprint or fingerprint(dict(candidate.contract)))
            == expected
            and validate(candidate)
        ]
        if len(eligible) > 1:
            raise ValueError(f"Multiple compatible artifacts have the same owner: {owner}")
        if eligible:
            return eligible[0]
    return None
