"""Resolve requested scientific graphs into owned work and inherited reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from nro.orchestration.artifact_resolution import (
    ArtifactCandidate,
    scientific_contracts,
    select_artifact,
)
from nro.orchestration.branches import BranchPaths, BranchTopology
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.execution_context import ExecutionContext, InputBinding


@dataclass(frozen=True)
class ResolvedInstance:
    """One requested logical instance and its selected owner.

    A reused artifact carries its validated generation and receives no demand.
    Otherwise the consumer owns the computation. Its context includes direct
    input bindings; scheduler admission must pin generations of newly built
    parents before launching it.
    """

    spec: InstanceSpec
    contract: dict
    artifact: ArtifactCandidate | None
    context: ExecutionContext


@dataclass(frozen=True)
class BranchPlan:
    """An endpoint closure with no implicit demand on inherited producers.

    This is a detached planning result, not scheduler authorization. Revalidate
    candidate generations and branch authority when admitting work.
    """

    branch: str
    terminals: tuple[str, ...]
    instances: tuple[ResolvedInstance, ...]
    specifications: tuple[InstanceSpec, ...] = ()
    inherit: bool = True

    @property
    def work(self) -> tuple[ResolvedInstance, ...]:
        """Return only computations owned by the requesting branch."""
        return tuple(item for item in self.instances if item.artifact is None)


def resolve_branch_plan(
    topology: BranchTopology,
    paths: BranchPaths,
    instances: Sequence[InstanceSpec],
    terminals: Sequence[str],
    candidates: Sequence[ArtifactCandidate],
    *,
    validate: Callable[[ArtifactCandidate], bool],
    inherit: bool = True,
    contracts: Mapping[str, dict] | None = None,
) -> BranchPlan:
    """Select fresh ancestors and trim computations hidden behind reused outputs.

    Inputs must describe a complete logical graph rooted in shared BIDS paths.
    Only the requested endpoint closure is considered. Validation is supplied by
    the scientific catalog; this function neither opens a registry nor imports
    processing modules. Missing or stale ancestors cause local computation.
    """
    record = topology.records.get(paths.branch)
    if record is None or record.retired:
        raise ValueError("Cannot plan work for an unregistered or retired branch")
    by_key = {item.key: item for item in instances}
    if len(by_key) != len(instances):
        raise ValueError("Instance graph contains duplicate keys")
    terminals = tuple(dict.fromkeys(terminals))
    if not terminals or any(key not in by_key for key in terminals):
        raise ValueError("Expected registered graph endpoints")
    closure: set[str] = set()
    pending = list(terminals)
    while pending:
        key = pending.pop()
        if key in closure:
            continue
        if key not in by_key:
            raise ValueError("Requested graph lacks a required dependency")
        closure.add(key)
        pending.extend(by_key[key].dependencies)
    if contracts is None:
        contracts = scientific_contracts(tuple(by_key[key] for key in sorted(closure)))
    else:
        contracts = dict(contracts)
        if set(contracts) != closure or not all(
            isinstance(value, dict) for value in contracts.values()
        ):
            raise ValueError("Precompiled contracts must cover the requested graph")
    covered: set[str] = set()
    pending = [parent for key in terminals for parent in by_key[key].dependencies]
    while pending:
        key = pending.pop()
        if key not in covered:
            covered.add(key)
            pending.extend(by_key[key].dependencies)
    terminals = tuple(key for key in terminals if key not in covered)
    selected: dict[str, ArtifactCandidate | None] = {}
    required: set[str] = set()
    pending = list(terminals)
    while pending:
        key = pending.pop()
        if key in required:
            continue
        required.add(key)
        spec = by_key[key]

        def valid(candidate: ArtifactCandidate) -> bool:
            owner = BranchPaths(candidate.branch, paths.bids, paths.work, paths.development)
            owner.require_output(candidate.root, spec.project)
            if type(candidate.generation) is not int or candidate.generation < 0:
                raise ValueError("Inherited artifacts require a nonnegative generation")
            return validate(candidate)

        selected[key] = select_artifact(
            topology, paths.branch, contracts[key], candidates, validate=valid, inherit=inherit
        )
        if selected[key] is None:
            pending.extend(spec.dependencies)

    resolved: dict[str, ResolvedInstance] = {}
    pending_keys = set(required)
    while pending_keys:
        ready = sorted(
            key
            for key in pending_keys
            if selected[key] is not None or set(by_key[key].dependencies) <= resolved.keys()
        )
        if not ready:
            raise ValueError("Requested graph contains a dependency cycle")
        for key in ready:
            spec = by_key[key]
            bindings = []
            if selected[key] is None:
                for parent_key in dict.fromkeys(spec.dependencies):
                    parent = resolved[parent_key]
                    artifact = parent.artifact
                    # None remains unresolved until the producer is fresh.
                    # Generation zero is valid for adopted native derivatives.
                    bindings.append(
                        InputBinding(
                            artifact.branch if artifact else paths.branch,
                            artifact.key if artifact else parent_key,
                            artifact.generation if artifact else None,
                            parent.spec.output_root,
                            artifact.root
                            if artifact
                            else parent.context.output_path(parent.spec.output_root),
                            parent.spec.output_prefix,
                        )
                    )
            context = ExecutionContext(paths, spec.project, key, tuple(bindings))
            logical_root = paths.source_project(spec.project) / "derivatives"
            if not spec.output_root.is_relative_to(logical_root):
                raise ValueError("Logical graph outputs must be in shared derivative coordinates")
            context.output_path(spec.output_root)
            for output in spec.expected_outputs:
                context.output_path(output)
            if selected[key] is None:
                for path in spec.input_paths:
                    context.input_path(path)
            resolved[key] = ResolvedInstance(spec, contracts[key], selected[key], context)
            pending_keys.remove(key)
    return BranchPlan(
        paths.branch,
        terminals,
        tuple(resolved.values()),
        tuple(by_key[key] for key in sorted(closure)),
        inherit,
    )
