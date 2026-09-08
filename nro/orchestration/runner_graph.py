"""Declarative, filesystem-backed execution graphs for nro modules.

The orchestration registry schedules module instances. Each :class:`Runner`
owns one :class:`RunnerGraph`, which describes every step inside one such
instance before any step is considered for execution. The graph is
deterministic from resolved BIDS data and workflow configuration; filesystem
freshness affects execution records, never topology.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from nro.engine.io import atomic_write_json


class NodeState(str, Enum):
    """Freshness outcome of a declared step in the current execution."""
    FRESH = "fresh"
    DIRTY = "dirty"


class StepKind(str, Enum):
    """Execution mechanism for a Python action, command, or directory-producing tool."""
    PYTHON = "python"
    COMMAND = "command"
    DIRECTORY = "directory"


Validator = Callable[[], tuple[bool, str]]
Action = Callable[[], None]


def path_mtime(path: Path) -> float:
    """Return the latest mtime beneath an artifact without failing on absence."""
    if not path.exists():
        return float("-inf")
    try:
        if path.is_file():
            return float(path.stat().st_mtime)
        newest = float(path.stat().st_mtime)
        for root, dirs, files in os.walk(path):
            for name in (*dirs, *files):
                try:
                    newest = max(newest, float((Path(root) / name).stat().st_mtime))
                except OSError:
                    pass
        return newest
    except OSError:
        return float("-inf")


def artifact_decision(
    outputs: Iterable[Path],
    force: bool,
    *,
    inputs: Iterable[Optional[Path]] = (),
) -> tuple[bool, str]:
    """Apply the shared file-artifact freshness rule."""
    output_paths = [Path(path) for path in outputs]
    if not output_paths:
        raise ValueError(
            "A resumable step must declare at least one output file (a public artifact); "
            "use an end-of-step completion breadcrumb for directory-producing tools."
        )
    directories = [str(path) for path in output_paths if path.is_dir()]
    if directories:
        raise ValueError(
            "Resumable outputs must be files, not mutable directories; use a completion "
            "breadcrumb: " + ", ".join(directories)
        )
    if force:
        return True, "Forced re-run requested."
    missing = [
        str(path)
        for path in output_paths
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        return True, "Missing or empty outputs: " + ", ".join(missing)
    input_paths = [
        Path(path) for path in inputs if path is not None and Path(path).exists()
    ]
    if not input_paths:
        return False, "Output(s) exist and are up to date."
    oldest_output = min(path_mtime(path) for path in output_paths)
    stale_inputs = [
        str(path) for path in input_paths if path_mtime(path) > oldest_output
    ]
    if stale_inputs:
        return True, "Re-running because inputs are newer than outputs: " + ", ".join(
            stale_inputs
        )
    return False, "Output(s) exist and are up to date."


@dataclass(frozen=True)
class Step:
    """Immutable declaration of one transform in a module DAG."""

    name: str
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    kind: StepKind
    id: str = ""
    action: Optional[Action] = field(default=None, repr=False, compare=False)
    command: tuple[str, ...] = ()
    force: bool = False
    validate: Optional[Validator] = field(default=None, repr=False, compare=False)
    after: tuple[str, ...] = ()
    env: Optional[Mapping[str, str]] = field(default=None, repr=False, compare=False)
    cwd: Optional[Path] = None
    direct: bool = False
    prepare: Optional[Action] = field(default=None, repr=False, compare=False)
    finalize: Optional[Action] = field(default=None, repr=False, compare=False)
    directory: Optional[Path] = None
    breadcrumb: Optional[Path] = None
    breadcrumb_text: str = "complete\n"
    reset_directory: bool = True
    completion_boundary: bool = False

    @classmethod
    def python(
        cls,
        *,
        name: str,
        inputs: Sequence[Optional[Path]] = (),
        outputs: Sequence[Path],
        action: Action,
        id: str = "",
        force: bool = False,
        validate: Optional[Validator] = None,
        after: Sequence[str] = (),
        completion_boundary: bool = False,
    ) -> "Step":
        """Declare a Python action without executing it.

        Inputs determine producer edges; after adds explicit predecessor IDs.
        Outputs must be durable files. An optional validator returns validity and
        a diagnostic reason; completion_boundary marks a public recovery boundary.
        """
        return cls(
            id=id,
            name=name,
            inputs=tuple(Path(path) for path in inputs if path is not None),
            outputs=tuple(Path(path) for path in outputs),
            kind=StepKind.PYTHON,
            action=action,
            force=bool(force),
            validate=validate,
            after=tuple(after),
            completion_boundary=bool(completion_boundary),
        )

    @classmethod
    def command_step(
        cls,
        command: Sequence[str],
        *,
        name: str = "",
        inputs: Sequence[Optional[Path]] = (),
        outputs: Sequence[Path],
        id: str = "",
        force: bool = False,
        validate: Optional[Validator] = None,
        after: Sequence[str] = (),
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
        direct: bool = False,
        prepare: Optional[Action] = None,
        finalize: Optional[Action] = None,
    ) -> "Step":
        """Declare an external command and its file boundary.

        prepare and finalize run around execution. direct bypasses the container;
        env and cwd apply to the command. No command runs in this factory.
        """
        return cls(
            id=id,
            name=name,
            inputs=tuple(Path(path) for path in inputs if path is not None),
            outputs=tuple(Path(path) for path in outputs),
            kind=StepKind.COMMAND,
            command=tuple(str(value) for value in command),
            force=bool(force),
            validate=validate,
            after=tuple(after),
            env=None if env is None else dict(env),
            cwd=None if cwd is None else Path(cwd),
            direct=bool(direct),
            prepare=prepare,
            finalize=finalize,
        )

    @classmethod
    def directory_step(
        cls,
        *,
        name: str,
        directory: Path,
        breadcrumb: Path,
        action: Action,
        validate: Validator,
        inputs: Sequence[Optional[Path]] = (),
        outputs: Sequence[Path] = (),
        id: str = "",
        force: bool = False,
        after: Sequence[str] = (),
        breadcrumb_text: str = "complete\n",
        reset_directory: bool = True,
        completion_boundary: bool = False,
    ) -> "Step":
        """Declare a tool-owned directory with validated completion.

        The breadcrumb is published only after action and validation succeed.
        reset_directory permits removal of an incomplete directory on execution;
        additional outputs remain part of the declared file boundary.
        """
        additional = tuple(Path(path) for path in outputs)
        marker = Path(breadcrumb)
        return cls(
            id=id,
            name=name,
            inputs=tuple(Path(path) for path in inputs if path is not None),
            outputs=(*additional, marker),
            kind=StepKind.DIRECTORY,
            action=action,
            force=bool(force),
            validate=validate,
            after=tuple(after),
            directory=Path(directory),
            breadcrumb=marker,
            breadcrumb_text=breadcrumb_text,
            reset_directory=bool(reset_directory),
            completion_boundary=bool(completion_boundary),
        )


@dataclass
class StepResult:
    """Runtime state for one immutable :class:`Step` definition."""

    should_run: bool
    reason: str
    state: Optional[NodeState] = None
    number: Optional[int] = None
    execution: Optional[str] = None


@dataclass
class RunnerOperation:
    """A numbered diagnostic operation outside the artifact DAG."""

    step: int
    name: str
    outputs: tuple[Path, ...]
    execution: str
    reason: str = ""


class RunnerGraph:
    """A Runner-owned module DAG, populated and frozen before execution."""

    def __init__(self, module_name: str) -> None:
        """Create an empty mutable graph with no execution results."""
        self.module_name = str(module_name)
        self._steps: list[Step] = []
        self._by_id: dict[str, Step] = {}
        self._dependencies: dict[str, tuple[str, ...]] = {}
        self._order: tuple[str, ...] = ()
        self._frozen = False
        self._results: dict[str, StepResult] = {}
        self.operations: list[RunnerOperation] = []

    @property
    def steps(self) -> tuple[Step, ...]:
        """Return declarations in insertion order as an immutable tuple."""
        return tuple(self._steps)

    @property
    def frozen(self) -> bool:
        """Return whether topology has been finalized for traversal."""
        return self._frozen

    @property
    def results(self) -> Mapping[str, StepResult]:
        """Expose step execution records indexed by declared step ID."""
        return dict(self._results)

    @staticmethod
    def _node_id(outputs: Sequence[Path]) -> str:
        label = outputs[0].name if outputs else "operation"
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "operation"
        identity = "\0".join(
            str(Path(path).resolve(strict=False)) for path in outputs
        ).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:12]
        return f"{slug}-{digest}"

    def add(self, step: Step) -> Step:
        """Add a step definition; no freshness check or action occurs here."""
        if self._frozen:
            raise RuntimeError("Cannot add a step after the runner graph is frozen.")
        if not isinstance(step, Step):
            raise TypeError(f"RunnerGraph.add() requires a Step, got {type(step).__name__}")
        if not step.outputs:
            raise ValueError(f"Step {step.name!r} must declare at least one output file.")
        normalized = replace(
            step,
            id=step.id or self._node_id(step.outputs),
            inputs=tuple(Path(path) for path in step.inputs),
            outputs=tuple(Path(path) for path in step.outputs),
        )
        if normalized.id in self._by_id:
            raise ValueError(f"Duplicate step id in module DAG: {normalized.id}")
        self._steps.append(normalized)
        self._by_id[normalized.id] = normalized
        return normalized

    def freeze(self) -> "RunnerGraph":
        """Validate topology, infer artifact edges, and make the graph immutable."""
        if self._frozen:
            return self
        producers: dict[Path, str] = {}
        for step in self._steps:
            for output in step.outputs:
                path = output.resolve(strict=False)
                previous = producers.get(path)
                if previous is not None:
                    raise ValueError(
                        f"Output {output} is produced by both {previous!r} and {step.id!r}."
                    )
                producers[path] = step.id

        dependencies: dict[str, set[str]] = {step.id: set(step.after) for step in self._steps}
        for step in self._steps:
            unknown = dependencies[step.id] - self._by_id.keys()
            if unknown:
                raise ValueError(
                    f"Step {step.id!r} names unknown dependencies: {', '.join(sorted(unknown))}"
                )
            dependencies[step.id].update(
                producer
                for path in step.inputs
                if (producer := producers.get(path.resolve(strict=False))) is not None
                and producer != step.id
            )

        dependents: dict[str, list[str]] = defaultdict(list)
        indegree = {step.id: len(dependencies[step.id]) for step in self._steps}
        insertion = {step.id: index for index, step in enumerate(self._steps)}
        for child, parents in dependencies.items():
            for parent in parents:
                dependents[parent].append(child)
        ready = sorted((key for key, value in indegree.items() if value == 0), key=insertion.get)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(dependents[current], key=insertion.get):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort(key=insertion.get)
        if len(order) != len(self._steps):
            cyclic = sorted(key for key, value in indegree.items() if value)
            raise ValueError("Module DAG contains a cycle involving: " + ", ".join(cyclic))

        self._dependencies = {
            key: tuple(sorted(values, key=insertion.get))
            for key, values in dependencies.items()
        }
        self._order = tuple(order)
        self._frozen = True
        return self

    def ordered_steps(self) -> tuple[Step, ...]:
        """Return topologically ordered declarations; raise RuntimeError before freeze."""
        if not self._frozen:
            raise RuntimeError("Runner graph must be frozen before traversal.")
        return tuple(self._by_id[key] for key in self._order)

    def dependencies(self, step: Step | str) -> tuple[str, ...]:
        """Return predecessor IDs for a step in a frozen graph."""
        if not self._frozen:
            raise RuntimeError("Runner graph must be frozen before reading dependencies.")
        key = step if isinstance(step, str) else step.id
        return self._dependencies[key]

    def record_decision(self, step: Step, *, should_run: bool, reason: str) -> None:
        """Record the first freshness decision for a step.

        Raise RuntimeError if the graph is mutable or a decision already exists.
        """
        if not self._frozen:
            raise RuntimeError("Cannot record execution state for an unfrozen graph.")
        if step.id in self._results:
            raise RuntimeError(f"Step {step.id!r} received more than one freshness decision.")
        self._results[step.id] = StepResult(bool(should_run), str(reason))

    def revise_decision(self, step: Step, *, should_run: bool, reason: str) -> None:
        """Update an existing decision during recovery; reject steps with no decision."""
        result = self._results.get(step.id)
        if result is None:
            raise RuntimeError(f"Step {step.id!r} has no freshness decision to revise.")
        result.should_run = bool(should_run)
        result.reason = str(reason)

    def state(self, step: Step | str) -> Optional[NodeState]:
        """Return the recorded freshness state, or None before a result is available."""
        key = step if isinstance(step, str) else step.id
        result = self._results.get(key)
        return None if result is None else result.state

    def record_step(
        self,
        *,
        step: int,
        name: str,
        outputs: Iterable[Path | str] = (),
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        """Attach a displayed execution event to its declared step or logged operation.

        Reject ambiguous output matches and execution before a freshness decision.
        """
        displayed = tuple(Path(value) for value in outputs if str(value).strip())
        displayed_paths = {path.resolve(strict=False) for path in displayed}
        for operation in reversed(self.operations):
            if operation.step == int(step):
                operation.execution = status
                if reason:
                    operation.reason = reason
                return
        for definition in self._steps:
            result = self._results.get(definition.id)
            if result is not None and result.number == int(step):
                result.execution = status
                if reason:
                    result.reason = reason
                if status in {"success", "error"}:
                    result.state = NodeState.DIRTY
                elif status in {"skipping", "fresh"}:
                    result.state = NodeState.FRESH
                return
        matches = [
            definition
            for definition in self._steps
            if {path.resolve(strict=False) for path in definition.outputs} == displayed_paths
        ]
        if len(matches) != 1:
            if displayed:
                raise RuntimeError(
                    f"Runner step {step:03d} ({name}) does not match exactly one declared "
                    "DAG step: " + ", ".join(str(path) for path in displayed)
                )
            self.record_operation(
                step=step,
                name=name,
                outputs=displayed,
                status=status,
                reason=reason or "Non-resumable runner operation.",
            )
            return
        definition = matches[0]
        result = self._results.get(definition.id)
        if result is None:
            raise RuntimeError(
                f"Runner step {step:03d} ({name}) executed before its freshness decision."
            )
        result.number = int(step)
        result.execution = status
        if reason:
            result.reason = reason
        if status in {"success", "error"}:
            result.state = NodeState.DIRTY
        elif status in {"skipping", "fresh"}:
            result.state = NodeState.FRESH

    def record_operation(
        self,
        *,
        step: int,
        name: str,
        outputs: Iterable[Path | str] = (),
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        """Append an aggregate or non-resumable operation to execution history."""
        self.operations.append(
            RunnerOperation(
                step=int(step),
                name=name,
                outputs=tuple(Path(value) for value in outputs if str(value).strip()),
                execution=status,
                reason=reason or "Aggregate or non-resumable operation.",
            )
        )

    def canonical_outputs(self, outputs: Iterable[Path | str]) -> tuple[Path, ...]:
        """Return declared output ordering when the supplied paths identify a step."""
        displayed = tuple(Path(value) for value in outputs if str(value).strip())
        displayed_paths = {path.resolve(strict=False) for path in displayed}
        for step in self._steps:
            if {path.resolve(strict=False) for path in step.outputs} == displayed_paths:
                return step.outputs
        return displayed

    def _contract_step(self, step: Step) -> dict[str, object]:
        return {
            "id": step.id,
            "name": step.name,
            "kind": step.kind.value,
            "inputs": [str(path.resolve(strict=False)) for path in step.inputs],
            "outputs": [str(path.resolve(strict=False)) for path in step.outputs],
            "dependencies": list(self._dependencies[step.id]),
        }

    def contract_payload(self, *, signature: str) -> dict[str, object]:
        """Serialize frozen topology and the supplied substantive signature.

        Runtime decisions are excluded; calling before freeze raises RuntimeError.
        """
        if not self._frozen:
            raise RuntimeError("Runner graph must be frozen before serialization.")
        return {
            "version": 2,
            "module": self.module_name,
            "signature": str(signature),
            "nodes": [self._contract_step(step) for step in self.ordered_steps()],
        }

    def bind_contract(self, path: Path, *, signature: str) -> None:
        """Validate the complete graph against its prior substantive contract."""
        if not self._frozen:
            raise RuntimeError("Runner graph must be frozen before binding its contract.")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if not isinstance(existing, dict) or existing.get("signature") != signature:
            return
        current = self.contract_payload(signature=signature)
        if existing.get("module") != current["module"] or existing.get("nodes") != current["nodes"]:
            raise RuntimeError(
                "Module DAG topology changed under an immutable source/workflow "
                f"contract ({signature})."
            )

    def reconcile_contract(self, path: Path, *, signature: str) -> dict[str, object]:
        """Validate prior topology, atomically save the contract, and return its payload."""
        contract = self.contract_payload(signature=signature)
        self.bind_contract(path, signature=signature)
        atomic_write_json(path, contract, sort_keys=True)
        return contract

    def validate_execution_contract(self) -> None:
        """Require one complete execution decision for every declared step.

        Raise RuntimeError when execution records violate the frozen graph.
        """
        failures: list[str] = []
        for step in self.ordered_steps():
            result = self._results.get(step.id)
            if result is None:
                failures.append(f"{step.id} received no freshness decision")
                continue
            if result.number is None:
                failures.append(f"{step.id} was never executed or skipped")
                continue
            if result.should_run and result.execution not in {"success", "error"}:
                failures.append(
                    f"step {result.number:03d} ({step.name}) was planned to run but ended "
                    f"with execution state {result.execution!r}"
                )
            if not result.should_run and result.execution not in {"skipping", "fresh"}:
                failures.append(
                    f"step {result.number:03d} ({step.name}) was planned fresh but ended "
                    f"with execution state {result.execution!r}"
                )
        if failures:
            raise RuntimeError(
                "Runner graph execution contract was violated:\n- " + "\n- ".join(failures)
            )

    def payload(self) -> dict[str, object]:
        """Serialize topology and execution records for the runner report."""
        if not self._frozen:
            raise RuntimeError("Runner graph must be frozen before serialization.")
        nodes: list[dict[str, object]] = []
        for step in self.ordered_steps():
            result = self._results.get(step.id)
            nodes.append(
                {
                    **self._contract_step(step),
                    "decision": (
                        None if result is None else "run" if result.should_run else "skip"
                    ),
                    "execution": None if result is None else result.execution,
                    "reason": None if result is None else result.reason,
                    "step": None if result is None else result.number,
                }
            )
        return {
            "version": 2,
            "module": self.module_name,
            "generated_at": time.time(),
            "nodes": nodes,
            "operations": [
                {
                    "step": operation.step,
                    "name": operation.name,
                    "outputs": [str(path) for path in operation.outputs],
                    "execution": operation.execution,
                    "reason": operation.reason,
                }
                for operation in self.operations
            ],
        }

    def write(self, path: Path) -> None:
        """Write the graph report to the supplied path."""
        atomic_write_json(path, self.payload(), sort_keys=True)
