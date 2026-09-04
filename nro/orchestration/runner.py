"""Shared container-aware command runner for scientific modules."""

from __future__ import annotations

import logging
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from nro.orchestration.runner_graph import (
    NodeState,
    RunnerGraph,
    Step,
    StepKind,
    artifact_decision,
)
from nro.engine.io import atomic_write_json, atomic_write_text
from nro.engine.execution import collect_bind_directories, strip_ansi


def shlex_quote(s: str) -> str:
    if not s:
        return "''"
    if all(ch.isalnum() or ch in "._/+-=:" for ch in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _parse_bind_spec(bind_spec: str) -> tuple[Path, str, str | None] | None:
    """Parse a container bind into its host source, target, and options."""
    fields = bind_spec.split(":")
    if len(fields) == 1:
        source = destination = fields[0]
        options = None
    elif len(fields) in {2, 3}:
        source, destination = fields[:2]
        options = fields[2] if len(fields) == 3 else None
    else:
        return None
    if not source or not destination:
        return None
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        return None
    return source_path, str(destination_path), options


@dataclass(frozen=True)
class ContainerSpec:
    image: Path
    engine: str = "singularity"
    cleanenv: bool = True
    extra_binds: Tuple[str, ...] = ()
    home_dir: Optional[Path] = None
    inner_setup: str = ""


@dataclass
class _RunnerExecutionState:
    module_name: str
    started_at: float
    boundary_step: int
    boundary_name: str
    active_steps: list[tuple[int, str]]
    failed_step: Optional[tuple[int, str]] = None
    graph: Optional[RunnerGraph] = None


_RUNNER_EXECUTION: ContextVar[Optional[_RunnerExecutionState]] = ContextVar(
    "nro_runner_execution",
    default=None,
)


def write_completion_breadcrumb(path: Path, text: str = "complete\n") -> Path:
    """Atomically mark a directory-producing or compound step complete."""
    atomic_write_text(path, text)
    return path


class Runner:
    """The sole execution boundary for every nro module step.

    It owns numbered logging, command construction, container/host execution,
    captured failure reporting, and the terminal runner report. Scientific
    modules construct immutable :class:`Step` objects externally and pass them
    through :meth:`add_step`; only :meth:`execute` may decide freshness or
    invoke their actions.
    """

    def __init__(
        self,
        *,
        module_name: str,
        container: Optional[ContainerSpec],
        binds: Sequence[str],
        logger: logging.Logger,
        next_step: Callable[[], int],
        step_log_separator: str = "=" * 50,
    ) -> None:
        self._container = container
        self._binds = tuple(binds)
        self._declared_paths: list[Path] = []
        self._container_mount_cache: (
            tuple[list[str], tuple[tuple[str, str], ...]] | None
        ) = None
        self._logger = logger
        self._next_step = next_step
        self._graph = RunnerGraph(module_name)
        self._definition_inputs: tuple[Path, ...] = ()
        self._step_log_separator = step_log_separator
        if self._container is None:
            return
        if shutil.which(self._container.engine) is None:
            raise SystemExit(f"Container engine not found on PATH: {self._container.engine!r}")
        if not self._container.image.exists():
            raise SystemExit(f"Container image not found: {self._container.image}")

    def using_container(self) -> bool:
        return self._container is not None

    def container_engine(self) -> Optional[str]:
        return None if self._container is None else self._container.engine

    def add_step(self, step: Step) -> Step:
        """Add one externally constructed step to this runner's module DAG."""
        if not isinstance(step, Step):
            raise TypeError(f"Runner.add_step() requires a Step, got {type(step).__name__}")
        name = step.name
        if not name and step.kind is StepKind.COMMAND:
            name = self._guess_step_name(step.command)
        if not name:
            raise ValueError("A Python or directory step must have a name.")
        added = self._graph.add(
            replace(
                step,
                name=name,
                inputs=(*self._definition_inputs, *step.inputs),
            )
        )
        self._declared_paths.extend((*added.inputs, *added.outputs))
        if added.cwd is not None:
            self._declared_paths.append(added.cwd)
        self._container_mount_cache = None
        return added

    def set_definition_inputs(self, inputs: Sequence[Path]) -> None:
        """Set inputs inherited by subsequently added scientific steps."""
        if self._graph.frozen:
            raise RuntimeError("Cannot change definition inputs after the DAG is frozen.")
        self._definition_inputs = tuple(Path(path) for path in inputs)

    def execute(self) -> dict[str, NodeState]:
        """Traverse one complete, frozen module DAG.

        Graph construction is deliberately outside this method. Freshness,
        semantic validation, dirty propagation, logging, and execution all
        begin only after :meth:`RunnerGraph.freeze` has validated the complete
        topology.
        """
        state = _RUNNER_EXECUTION.get()
        if state is None:
            raise RuntimeError("Runner.execute() requires an active run_context().")
        graph = self._graph.freeze()
        state.graph = graph

        ledger = os.environ.get("NRO_STEP_LEDGER")
        signature = os.environ.get("NRO_RUNNER_GRAPH_SIGNATURE", "direct")
        if ledger:
            graph.bind_contract(
                Path(ledger).with_name("runner-contract.json"),
                signature=signature,
            )

        states: dict[str, NodeState] = {}
        completion = [step for step in graph.ordered_steps() if step.completion_boundary]
        if len(completion) > 1:
            raise RuntimeError("A module DAG may declare at most one completion boundary.")
        if completion:
            boundary = completion[0]
            boundary_run, boundary_reason = artifact_decision(
                boundary.outputs,
                boundary.force,
                inputs=boundary.inputs,
            )
            if not boundary_run and boundary.validate is not None:
                valid, validation_reason = boundary.validate()
                if not valid:
                    boundary_run = True
                    boundary_reason = validation_reason
            if not boundary_run:
                self._logger.info(
                    "Skipping %s: %s",
                    graph.module_name.lower(),
                    boundary_reason,
                )
                for step in graph.ordered_steps():
                    reason = (
                        boundary_reason
                        if step.id == boundary.id
                        else f"Module completion boundary is fresh: {boundary.name}."
                    )
                    graph.record_decision(step, should_run=False, reason=reason)
                    self._skip_declared_step(step, reason=reason)
                    states[step.id] = NodeState.FRESH
                return states

        for step in graph.ordered_steps():
            upstream_dirty = any(
                states[parent] is NodeState.DIRTY
                for parent in graph.dependencies(step)
            )
            should_run, reason = artifact_decision(
                step.outputs,
                step.force or upstream_dirty,
                inputs=step.inputs,
            )
            if upstream_dirty:
                reason = "Re-running because an upstream step produced new artifacts."
            graph.record_decision(step, should_run=should_run, reason=reason)

            if not should_run and step.validate is not None:
                valid, validation_reason = step.validate()
                if not valid:
                    should_run = True
                    reason = validation_reason
                    graph.revise_decision(
                        step,
                        should_run=True,
                        reason=validation_reason,
                    )

            if should_run:
                self._execute_declared_step(step, reason=reason)
                states[step.id] = NodeState.DIRTY
            else:
                self._skip_declared_step(step, reason=reason)
                states[step.id] = NodeState.FRESH
        return states

    def _skip_declared_step(self, step: Step, *, reason: str) -> None:
        if step.kind is StepKind.COMMAND:
            self.log_skip(
                step.command,
                cwd=step.cwd,
                step_name=step.name,
                outputs=step.outputs,
                reason=reason,
            )
            return
        self.log_python_step(
            step_name=step.name,
            outputs=step.outputs,
            running=False,
            reason=reason,
        )

    def _execute_declared_step(self, step: Step, *, reason: str) -> None:
        if step.kind is StepKind.PYTHON:
            if step.action is None:
                raise RuntimeError(f"Python step {step.id!r} has no action.")
            with self.python_step(
                step_name=step.name,
                outputs=step.outputs,
                reason=reason,
            ):
                step.action()
                self._validate_artifact_outputs(step.outputs)
                self._validate_step_result(step)
            return

        if step.kind is StepKind.COMMAND:
            if not step.command:
                raise RuntimeError(f"Command step {step.id!r} has no command.")
            method = self.run_direct if step.direct else self.run
            method(
                step.command,
                env=None if step.env is None else dict(step.env),
                cwd=step.cwd,
                step_name=step.name,
                outputs=step.outputs,
                reason=reason,
                prepare=step.prepare,
                finalize=step.finalize,
            )
            self._validate_artifact_outputs(step.outputs)
            self._validate_step_result(step)
            return

        if step.kind is not StepKind.DIRECTORY:
            raise RuntimeError(f"Unsupported step kind: {step.kind!r}")
        if step.action is None or step.validate is None:
            raise RuntimeError(
                f"Directory step {step.id!r} requires an action and validator."
            )
        assert step.directory is not None and step.breadcrumb is not None
        directory = step.directory
        breadcrumb = step.breadcrumb
        resolved = directory.resolve(strict=False)
        if resolved == Path(resolved.anchor) or resolved == Path.home().resolve(strict=False):
            raise ValueError(f"Refusing to manage an unsafe directory artifact: {directory}")
        with self.python_step(
            step_name=step.name,
            outputs=step.outputs,
            reason=reason,
        ):
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                directory.unlink()
            elif directory.exists() and step.reset_directory:
                shutil.rmtree(directory)
            try:
                breadcrumb.relative_to(directory)
                breadcrumb_is_internal = True
            except ValueError:
                breadcrumb_is_internal = False
            if not breadcrumb_is_internal or not step.reset_directory:
                breadcrumb.unlink(missing_ok=True)
            step.action()
            if not directory.is_dir():
                raise RuntimeError(
                    "Directory artifact producer did not create its output directory: "
                    f"{directory}"
                )
            valid, validation_reason = step.validate()
            if not valid:
                raise RuntimeError(validation_reason)
            additional_outputs = tuple(
                output for output in step.outputs if output != breadcrumb
            )
            self._validate_artifact_outputs(additional_outputs)
            write_completion_breadcrumb(breadcrumb, step.breadcrumb_text)

    @staticmethod
    def _validate_artifact_outputs(outputs: Sequence[Path]) -> None:
        missing = [
            str(path)
            for path in outputs
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(
                "Artifact step did not produce nonempty file output(s): "
                + ", ".join(missing)
            )

    @staticmethod
    def _validate_step_result(step: Step) -> None:
        if step.validate is None:
            return
        valid, validation_reason = step.validate()
        if not valid:
            raise RuntimeError(validation_reason)

    def _container_prefix(self) -> list[str]:
        return self._container_prefix_for_cwd(None)

    def _container_mounts(self) -> tuple[list[str], tuple[tuple[str, str], ...]]:
        """Return bind specifications and host-to-container path mappings.

        Module-discovered directories are mounted at compact root-level names.
        This keeps the paths seen by scientific software short even when the
        corresponding BIDS paths on the host are deeply nested. Explicit binds
        remain available as fallbacks and retain their configured destinations.
        """
        assert self._container is not None
        if self._container_mount_cache is not None:
            return self._container_mount_cache
        extra_binds = [
            str(value).strip()
            for value in self._container.extra_binds
            if str(value).strip()
        ]
        used_targets = {
            parsed[1]
            for bind_spec in extra_binds
            if (parsed := _parse_bind_spec(bind_spec)) is not None
        }
        generated_binds: list[str] = []
        mappings: list[tuple[str, str]] = []
        alias_index = 0
        generated_sources = collect_bind_directories(
            [*(Path(value) for value in self._binds), *self._declared_paths]
        )
        for value in generated_sources:
            source_text = str(value).strip()
            if not source_text:
                continue
            source = Path(source_text).expanduser().resolve()
            while f"/n{alias_index}" in used_targets:
                alias_index += 1
            target = f"/n{alias_index}"
            alias_index += 1
            used_targets.add(target)
            generated_binds.append(f"{source}:{target}")
            mappings.append((str(source), target))

        for bind_spec in extra_binds:
            parsed = _parse_bind_spec(bind_spec)
            if parsed is not None:
                source, target, _options = parsed
                mappings.append((str(source), target))

        seen_binds: set[str] = set()
        binds: list[str] = []
        for bind_spec in [*generated_binds, *extra_binds]:
            if bind_spec in seen_binds:
                continue
            seen_binds.add(bind_spec)
            binds.append(bind_spec)
        result = binds, tuple(
            sorted(
                (mapping for mapping in mappings if mapping[0] != mapping[1]),
                key=lambda mapping: len(mapping[0]),
                reverse=True,
            )
        )
        self._container_mount_cache = result
        return result

    def _container_path_mappings(self) -> tuple[tuple[str, str], ...]:
        assert self._container is not None
        _binds, mappings = self._container_mounts()
        if self._container.home_dir is None:
            return mappings
        home = str(self._container.home_dir.expanduser().resolve())
        return tuple(
            sorted(
                (*(mapping for mapping in mappings if mapping[0] != home), (home, "/nh")),
                key=lambda mapping: len(mapping[0]),
                reverse=True,
            )
        )

    def _translate_container_text(self, value: str) -> str:
        """Translate absolute host paths embedded anywhere in container text."""
        translated = str(value)
        for source, target in self._container_path_mappings():
            pattern = re.compile(
                rf"(?<![A-Za-z0-9._-]){re.escape(source)}(?=$|/|[\s,;:\]\[(){{}}'\"=])"
            )
            translated = pattern.sub(
                lambda _match, replacement=target: replacement,
                translated,
            )
        return translated

    def _restore_host_paths(self, value: str) -> str:
        """Translate reserved short container aliases in captured output."""
        restored = str(value)
        mappings = sorted(
            self._container_path_mappings(),
            key=lambda mapping: len(mapping[1]),
            reverse=True,
        )
        for source, target in mappings:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9._-]){re.escape(target)}(?=$|/|[\s,;:\]\[(){{}}'\"=])"
            )
            restored = pattern.sub(
                lambda _match, replacement=source: replacement,
                restored,
            )
        return restored

    def _container_prefix_for_cwd(self, cwd: Optional[Path]) -> list[str]:
        assert self._container is not None
        engine = self._container.engine
        prefix = [engine, "exec"]
        if self._container.cleanenv:
            prefix.append("--cleanenv")
        if self._container.home_dir is not None:
            self._container.home_dir.mkdir(parents=True, exist_ok=True)
            prefix += ["-H", f"{self._container.home_dir.resolve()}:/nh"]
        binds, _mappings = self._container_mounts()
        for bind_spec in binds:
            if engine == "singularity":
                prefix += ["-B", bind_spec]
            else:
                prefix += ["--bind", bind_spec]
        if cwd is not None:
            prefix += ["--pwd", self._translate_container_text(str(cwd.resolve()))]
        prefix.append(str(self._container.image))
        return prefix

    def _host_env_for_container(self, inner_env: Optional[dict[str, str]]) -> dict[str, str]:
        host_env = dict(os.environ)
        if inner_env:
            for k, v in inner_env.items():
                translated = self._translate_container_text(v)
                host_env[f"SINGULARITYENV_{k}"] = translated
                host_env[f"APPTAINERENV_{k}"] = translated
        return host_env

    def _inner_cmd(self, cmd: Sequence[str], inner_env: Optional[dict[str, str]]) -> str:
        translated_cmd = (
            [self._translate_container_text(str(value)) for value in cmd]
            if self._container is not None
            else [str(value) for value in cmd]
        )
        inner = " ".join(shlex_quote(x) for x in translated_cmd)
        if self._container is None:
            return inner
        exports = ""
        if inner_env:
            export_lines = [
                f"export {k}={shlex_quote(self._translate_container_text(v))}"
                for k, v in inner_env.items()
            ]
            exports = "; ".join(export_lines)
        setup = self._translate_container_text(
            (self._container.inner_setup or "").strip()
        )
        parts = ["set -e"]
        if setup:
            parts.append(setup)
        if exports:
            parts.append(exports)
        parts.append(inner)
        return "; ".join(parts)

    @staticmethod
    def _format_cmd(cmd: Sequence[str]) -> str:
        return " ".join(shlex_quote(x) for x in cmd)

    @staticmethod
    def _cwd_note(cwd: Optional[Path]) -> str:
        return f" [cwd={cwd}]" if cwd is not None else ""

    @staticmethod
    def _guess_step_name(cmd: Sequence[str]) -> str:
        if not cmd:
            return "Command"
        head = Path(str(cmd[0])).name
        shell_text = " ".join(str(x) for x in cmd)
        args = [str(x) for x in cmd]
        lookup_match = re.search(r"command -v\s+([A-Za-z0-9._+-]+)", shell_text)
        if lookup_match:
            return f"Checking for software: {lookup_match.group(1)}"
        resolve_match = re.search(r"__NRO_CMD__=.*?([A-Za-z0-9._+-]+)", shell_text)
        if resolve_match:
            return f"Resolving executable path: {resolve_match.group(1)}"
        find_bin_match = re.search(r"-path\s+.+\*/bin/([A-Za-z0-9._+-]+)", shell_text)
        if find_bin_match:
            return f"Searching for executable: {find_bin_match.group(1)}"
        exists_bin_match = re.search(r"\[\s+-e\s+([^ \t'\"]+/([A-Za-z0-9._+-]+))", shell_text)
        if exists_bin_match:
            return f"Checking path exists: {exists_bin_match.group(2)}"
        if "__NRO_EXISTS__" in shell_text or re.search(r"\[\s+-e\s+", shell_text):
            return "Path Existence Check"
        if "fsl_get_standard" in shell_text:
            return "FSL Template Lookup"
        if re.search(r"\benv\b", shell_text):
            return "Environment Query"
        if re.search(r"\bfind\s+/", shell_text):
            return "File Lookup"
        if head == "bash":
            return "Shell Command"
        if head == "mri_convert":
            if len(args) >= 3:
                src = args[-2].lower()
                dst = args[-1].lower()
                if src.endswith(".mgz") and (dst.endswith(".nii") or dst.endswith(".nii.gz")):
                    return "Converting from FreeSurfer to NIfTI"
            return "MRI Conversion"
        if head == "fslmaths" and "-Tmean" in args:
            return "Temporal Mean Image"
        mapping = {
            "N4BiasFieldCorrection": "Bias Field Correction",
            "antsApplyTransforms": "ANTs Transform Application",
            "antsRegistration": "SyN Registration",
            "applywarp": "Warp Application",
            "bbregister": "Boundary-Based Registration",
            "bet": "Brain Extraction",
            "convertwarp": "Warp Composition",
            "flirt": "Linear Registration",
            "fnirt": "Nonlinear Registration",
            "fsl_regfilt": "ICA Denoising",
            "fslinfo": "Image Header Query",
            "fslmaths": "Image Math",
            "fslmerge": "Image Merge",
            "fslroi": "Volume Extraction",
            "fslstats": "Image Statistics",
            "mcflirt": "Motion Correction",
            "melodic": "MELODIC ICA",
            "mri_vol2vol": "Mask Resampling",
            "singularity": "Container Command",
            "apptainer": "Container Command",
            "topup": "Fieldmap Estimation",
        }
        return mapping.get(head, head.replace("_", " ").title())

    @staticmethod
    def _combined_output(stdout: Optional[str], stderr: Optional[str]) -> str:
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(stderr)
        return "".join(parts)

    @staticmethod
    def _validate_declared_outputs(
        outputs: Optional[Sequence[Path | str]], *, cwd: Optional[Path] = None
    ) -> None:
        """Fail a completed command that did not materialize its artifacts."""
        if not outputs:
            return
        paths = [
            Path(value) if Path(value).is_absolute() or cwd is None else cwd / Path(value)
            for value in outputs
        ]
        invalid = [
            str(path)
            for path in paths
            if not path.is_file() or path.stat().st_size == 0
        ]
        if invalid:
            raise RuntimeError(
                "Command completed without producing nonempty file output(s): "
                + ", ".join(invalid)
            )

    @staticmethod
    def _format_outputs(outputs: Optional[Sequence[Path | str]]) -> str:
        if not outputs:
            return "None"
        items = [str(p) for p in outputs if str(p).strip()]
        if not items:
            return "None"
        return ", ".join(items)

    @staticmethod
    def _canonical_step_outputs(
        outputs: Optional[Sequence[Path | str]],
        *,
        cwd: Optional[Path] = None,
    ) -> list[Path | str]:
        displayed = list(outputs or ())
        state = _RUNNER_EXECUTION.get()
        if displayed and state is not None and state.graph is not None:
            canonical = state.graph.canonical_outputs(displayed)
            if canonical != tuple(Path(value) for value in displayed):
                return list(canonical)
        if cwd is None:
            return displayed
        return [
            value
            if Path(value).is_absolute()
            else cwd / Path(value)
            for value in displayed
        ]

    @staticmethod
    def _emit_step_event(
        *,
        step: int,
        name: str,
        status: str,
        outputs: Optional[Sequence[Path | str]] = None,
        command: Optional[str] = None,
        cwd: Optional[Path] = None,
        reason: Optional[str] = None,
        elapsed: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update the durable current-step ledger when running under a worker."""
        configured = os.environ.get("NRO_STEP_LEDGER")
        if not configured:
            return
        path = Path(configured)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
        key = f"{step:03d}:{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}"
        record = {
            "step_id": key,
            "step": step,
            "name": name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outputs": [str(value) for value in (outputs or ())],
            "command": command,
            "cwd": str(cwd) if cwd is not None else None,
            "reason": reason,
            "elapsed_seconds": elapsed,
            "error": error,
        }
        try:
            current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if not isinstance(current, dict):
                current = {}
        except (OSError, json.JSONDecodeError):
            current = {}
        previous = current.get(key, {})
        if status in {"success", "error"} and previous:
            for field in ("outputs", "command", "cwd", "reason"):
                if record.get(field) in (None, [], ""):
                    record[field] = previous.get(field)
        # A staleness check that skips an already-successful step should not
        # erase the last actual execution record.
        if not (status == "skipping" and current.get(key, {}).get("status") == "success"):
            current[key] = record
            atomic_write_json(path, current, sort_keys=True, mode=0o664)
        events = path.with_name("step-events.jsonl")
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        try:
            events.chmod(0o664)
        except PermissionError:
            pass

    @staticmethod
    def _status_from_reason(*, running: bool, reason: Optional[str]) -> str:
        if not running:
            return "Skipping"
        text = (reason or "").strip().lower()
        if ("newer than outputs" in text or text.startswith("forced re-run")
                or text.startswith("re-running") or text.startswith("instance completion certificate")):
            return "Rerunning"
        return "Running"

    @staticmethod
    def _guess_outputs(cmd: Sequence[str]) -> list[str]:
        args = [str(x) for x in cmd]
        if not args:
            return []
        head = Path(args[0]).name
        out: list[str] = []

        def add(value: str) -> None:
            v = str(value).strip()
            if not v:
                return
            if v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            if v and v not in out:
                out.append(v)

        for i, arg in enumerate(args):
            for prefix in (
                "--out=",
                "--iout=",
                "--fout=",
                "--dfout=",
                "--jacout=",
                "--rbmout=",
                "--reg=",
                "--fslmat=",
                "--warp1=",
                "--warp2=",
                "--premat=",
                "--postmat=",
                "--omat=",
                "--o=",
                "--jacobian=",
            ):
                if arg.startswith(prefix):
                    add(arg.split("=", 1)[1])
            if arg in {"-out", "-omat", "-o", "--reg", "--fslmat", "--iout", "--fout"} and i + 1 < len(args):
                add(args[i + 1])

        if head == "mcflirt":
            for i, arg in enumerate(args):
                if arg == "-out" and i + 1 < len(args):
                    base = args[i + 1]
                    add(base)
                    add(base + ".par")
                    add(base + ".mat")
        elif head == "fslmaths" and len(args) >= 2:
            add(args[-1])
        elif head == "fslroi" and len(args) >= 3:
            add(args[2])
        elif head == "fslmerge" and "-t" in args:
            idx = args.index("-t")
            if idx + 1 < len(args):
                add(args[idx + 1])
        elif head == "convert_xfm":
            for i, arg in enumerate(args):
                if arg == "-omat" and i + 1 < len(args):
                    add(args[i + 1])
        elif head == "bbregister":
            for i, arg in enumerate(args):
                if arg in {"--reg", "--fslmat"} and i + 1 < len(args):
                    add(args[i + 1])
        elif head == "antsApplyTransforms":
            for i, arg in enumerate(args):
                if arg == "-o" and i + 1 < len(args):
                    add(args[i + 1])
        elif head == "antsRegistration":
            for i, arg in enumerate(args):
                if arg == "--output" and i + 1 < len(args):
                    spec = args[i + 1]
                    if spec.startswith("[") and spec.endswith("]"):
                        spec = spec[1:-1]
                    prefix = spec.split(",", 1)[0]
                    if prefix:
                        add(prefix + "Warped.nii.gz")
                        add(prefix + "Composite.h5")
                        add(prefix + "InverseComposite.h5")
        elif head == "topup":
            for prefix in ("--out=", "--iout=", "--fout=", "--dfout=", "--jacout=", "--rbmout="):
                for arg in args:
                    if arg.startswith(prefix):
                        add(arg.split("=", 1)[1])
        elif head == "melodic":
            for arg in args:
                if arg.startswith("--outdir="):
                    directory = Path(arg.split("=", 1)[1])
                    add(str(directory / "melodic_IC.nii.gz"))
                    add(str(directory / "melodic_mix"))
                    add(str(directory / "melodic_FTmix"))
        elif head == "wb_command":
            if "-convert-warpfield" in args and "-to-fnirt" in args:
                idx = args.index("-to-fnirt")
                if idx + 1 < len(args):
                    add(args[idx + 1])

        return out

    def _log_command_start(
        self,
        label: str,
        cmd: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        step_name: Optional[str] = None,
        outputs: Optional[Sequence[Path | str]] = None,
        reason: Optional[str] = None,
    ) -> tuple[str, float, int]:
        cmd_str = self._format_cmd(cmd)
        step = self._next_step()
        human_name = str(step_name or self._guess_step_name(cmd)).strip()
        inferred_outputs = self._canonical_step_outputs(
            list(outputs) if outputs is not None else self._guess_outputs(cmd),
            cwd=cwd,
        )
        status = self._status_from_reason(running=True, reason=reason)
        self._logger.info(self._step_log_separator)
        self._logger.info("[Step %03d]", step)
        self._logger.info("Name: %s", human_name or "Command")
        self._logger.info("Output(s): %s", self._format_outputs(inferred_outputs))
        self._logger.info("Status: %s", status)
        if reason:
            self._logger.info("Status Reason: %s", reason)
        self._logger.info("Cmd: %s%s", cmd_str, self._cwd_note(cwd))
        self._activate_step(step, human_name or "Command")
        self._emit_step_event(
            step=step, name=human_name or "Command", status=status.lower(),
            outputs=inferred_outputs, command=cmd_str, cwd=cwd, reason=reason,
        )
        self._record_graph_step(
            step=step,
            name=human_name or "Command",
            outputs=inferred_outputs,
            status=status.lower(),
            reason=reason,
        )
        return cmd_str, time.perf_counter(), step

    def log_skip(
        self,
        cmd: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        step_name: Optional[str] = None,
        outputs: Optional[Sequence[Path | str]] = None,
        reason: Optional[str] = None,
    ) -> None:
        step = self._next_step()
        human_name = str(step_name or self._guess_step_name(cmd)).strip()
        inferred_outputs = self._canonical_step_outputs(
            list(outputs) if outputs is not None else self._guess_outputs(cmd),
            cwd=cwd,
        )
        self._logger.info(self._step_log_separator)
        self._logger.info("[Step %03d]", step)
        self._logger.info("Name: %s", human_name or "Command")
        self._logger.info("Output(s): %s", self._format_outputs(inferred_outputs))
        self._logger.info("Status: %s", self._status_from_reason(running=False, reason=reason))
        if reason:
            self._logger.info("Status Reason: %s", reason)
        self._emit_step_event(
            step=step, name=human_name or "Command", status="skipping",
            outputs=inferred_outputs, cwd=cwd, reason=reason,
        )
        self._record_graph_step(
            step=step,
            name=human_name or "Command",
            outputs=inferred_outputs,
            status="skipping",
            reason=reason,
        )

    def log_python_step(
        self,
        *,
        step_name: str,
        outputs: Optional[Sequence[Path | str]] = None,
        running: bool,
        reason: Optional[str] = None,
    ) -> int:
        step = self._next_step()
        canonical_outputs = self._canonical_step_outputs(outputs)
        self._logger.info(self._step_log_separator)
        self._logger.info("[Step %03d]", step)
        self._logger.info("Name: %s", step_name or "Python Step")
        self._logger.info("Output(s): %s", self._format_outputs(canonical_outputs))
        self._logger.info("Status: %s", self._status_from_reason(running=running, reason=reason))
        if reason:
            self._logger.info("Status Reason: %s", reason)
        if running:
            self._activate_step(step, step_name or "Python Step")
        self._emit_step_event(
            step=step, name=step_name or "Python Step",
            status=self._status_from_reason(running=running, reason=reason).lower(),
            outputs=canonical_outputs, reason=reason,
        )
        self._record_graph_step(
            step=step,
            name=step_name or "Python Step",
            outputs=canonical_outputs,
            status=self._status_from_reason(running=running, reason=reason).lower(),
            reason=reason,
        )
        return step

    def log_python_success(self, *, step: int, step_name: str, started_at: float) -> None:
        self._log_command_success("Step", started_at, step, step_name=step_name)

    def log_python_failure(self, *, step: int, step_name: str, error: BaseException) -> None:
        self._log_command_failure("Step", step, step_name=step_name, output=str(error))

    def log_runner_success(self, *, module_name: str, started_at: float) -> None:
        """Emit the standardized terminal report for a successful runner."""
        self._logger.info(self._step_log_separator)
        self._logger.info("Name: %s", module_name)
        self._logger.info("Status: Success")
        self._logger.info("Total Time Elapsed: %.3fs", time.perf_counter() - started_at)

    def log_runner_failure(
        self,
        *,
        module_name: str,
        started_at: float,
        step: int,
        step_name: str,
        error: BaseException,
    ) -> None:
        """Emit the standardized terminal report for a failed runner."""
        self._logger.error(self._step_log_separator)
        self._logger.error("Name: %s", module_name)
        self._logger.error("Status: Failure")
        self._logger.error("Failed Step: %03d — %s", step, step_name)
        self._logger.error("Error: %s: %s", type(error).__name__, error)
        self._logger.error("Total Time Elapsed: %.3fs", time.perf_counter() - started_at)

    def _activate_step(self, step: int, step_name: str) -> None:
        state = _RUNNER_EXECUTION.get()
        if state is not None:
            state.active_steps.append((step, step_name))

    @staticmethod
    def _record_graph_step(
        *,
        step: int,
        name: str,
        outputs: Optional[Sequence[Path | str]],
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        state = _RUNNER_EXECUTION.get()
        if state is not None and state.graph is not None:
            state.graph.record_step(
                step=step,
                name=name,
                outputs=outputs or (),
                status=status,
                reason=reason,
            )

    def _finish_step(self, step: int) -> None:
        state = _RUNNER_EXECUTION.get()
        if state is None:
            return
        for index in range(len(state.active_steps) - 1, -1, -1):
            if state.active_steps[index][0] == step:
                del state.active_steps[index:]
                return

    def _record_failed_step(self, step: int, step_name: str) -> None:
        state = _RUNNER_EXECUTION.get()
        if state is not None and state.failed_step is None:
            state.failed_step = (step, step_name)

    @contextmanager
    def python_step(
        self,
        *,
        step_name: str,
        outputs: Optional[Sequence[Path | str]] = None,
        reason: Optional[str] = None,
    ):
        """Run Python work as a numbered step with automatic terminal logging."""
        step = self.log_python_step(
            step_name=step_name,
            outputs=outputs,
            running=True,
            reason=reason,
        )
        started_at = time.perf_counter()
        try:
            yield step
            self._validate_declared_outputs(outputs)
        except BaseException as error:
            self.log_python_failure(step=step, step_name=step_name, error=error)
            raise
        else:
            self.log_python_success(step=step, step_name=step_name, started_at=started_at)

    @contextmanager
    def run_context(self, *, started_at: Optional[float] = None):
        """Own one complete runner invocation and report its terminal state."""
        if _RUNNER_EXECUTION.get() is not None:
            raise RuntimeError("Nested runner execution contexts are not supported")
        runner_started = time.perf_counter() if started_at is None else started_at
        boundary_step = self._next_step()
        module_name = self._graph.module_name
        boundary_name = f"{module_name} Execution"
        state = _RunnerExecutionState(
            module_name=module_name,
            started_at=runner_started,
            boundary_step=boundary_step,
            boundary_name=boundary_name,
            active_steps=[(boundary_step, boundary_name)],
            graph=self._graph,
        )
        ledger = os.environ.get("NRO_STEP_LEDGER")
        signature = os.environ.get("NRO_RUNNER_GRAPH_SIGNATURE", "direct")
        token = _RUNNER_EXECUTION.set(state)
        self._logger.info(self._step_log_separator)
        self._logger.info("[Step %03d]", boundary_step)
        self._logger.info("Name: %s", boundary_name)
        self._logger.info("Output(s): None")
        self._logger.info("Status: Running")
        self._record_graph_step(
            step=boundary_step,
            name=boundary_name,
            outputs=None,
            status="running",
        )
        boundary_started = time.perf_counter()
        try:
            yield self
        except BaseException as error:
            failed_step, failed_name = state.failed_step or state.active_steps[-1]
            if state.failed_step is None:
                self._log_command_failure(
                    "Step",
                    failed_step,
                    step_name=failed_name,
                    output=str(error),
                )
            self.log_runner_failure(
                module_name=module_name,
                started_at=runner_started,
                step=failed_step,
                step_name=failed_name,
                error=error,
            )
            raise
        else:
            if state.active_steps != [(boundary_step, boundary_name)]:
                failed_step, failed_name = state.active_steps[-1]
                error = RuntimeError(
                    f"Runner exited with unfinished step {failed_step:03d}: {failed_name}"
                )
                self._log_command_failure(
                    "Step",
                    failed_step,
                    step_name=failed_name,
                    output=str(error),
                )
                self.log_runner_failure(
                    module_name=module_name,
                    started_at=runner_started,
                    step=failed_step,
                    step_name=failed_name,
                    error=error,
                )
                raise error
            try:
                if state.graph is not None:
                    state.graph.validate_execution_contract()
                if ledger and state.graph is not None:
                    state.graph.reconcile_contract(
                        Path(ledger).with_name("runner-contract.json"),
                        signature=signature,
                    )
            except BaseException as error:
                # Contract violations use the same terminal failure report as
                # scientific step failures.
                self._log_command_failure(
                    "Step",
                    boundary_step,
                    step_name=boundary_name,
                    output=str(error),
                )
                self.log_runner_failure(
                    module_name=module_name,
                    started_at=runner_started,
                    step=boundary_step,
                    step_name=boundary_name,
                    error=error,
                )
                raise
            self._log_command_success(
                "Step",
                boundary_started,
                boundary_step,
                step_name=boundary_name,
            )
            self.log_runner_success(
                module_name=module_name,
                started_at=runner_started,
            )
        finally:
            ledger = os.environ.get("NRO_STEP_LEDGER")
            if ledger and state.graph is not None:
                try:
                    state.graph.write(Path(ledger).with_name("runner-graph.json"))
                except OSError as error:
                    # Never obscure the runner's real terminal state because
                    # an auxiliary observability artifact could not be written.
                    self._logger.warning("Could not write runner DAG ledger: %s", error)
            _RUNNER_EXECUTION.reset(token)

    def _log_command_success(
        self,
        label: str,
        started_at: float,
        step: int,
        *,
        cwd: Optional[Path] = None,
        step_name: Optional[str] = None,
    ) -> None:
        elapsed = time.perf_counter() - started_at
        human_name = str(step_name or "").strip()
        if human_name:
            self._logger.info("%s [step %03d] %s succeeded in %.3fs%s", label, step, human_name, elapsed, self._cwd_note(cwd))
        else:
            self._logger.info("%s [step %03d] succeeded in %.3fs%s", label, step, elapsed, self._cwd_note(cwd))
        self._emit_step_event(
            step=step, name=human_name or "Command", status="success", cwd=cwd, elapsed=elapsed,
        )
        self._record_graph_step(
            step=step,
            name=human_name or "Command",
            outputs=None,
            status="success",
        )
        self._finish_step(step)

    def _log_command_failure(
        self,
        label: str,
        step: int,
        *,
        cwd: Optional[Path] = None,
        step_name: Optional[str] = None,
        output: str = "",
    ) -> None:
        human_name = str(step_name or "").strip()
        if step > 0:
            self._record_failed_step(step, human_name or "Command")
        if human_name:
            self._logger.error("%s [step %03d] %s failed%s", label, step, human_name, self._cwd_note(cwd))
        else:
            self._logger.error("%s [step %03d] failed%s", label, step, self._cwd_note(cwd))
        self._emit_step_event(
            step=step, name=human_name or "Command", status="error", cwd=cwd,
            error=strip_ansi(output or ""),
        )
        self._record_graph_step(
            step=step,
            name=human_name or "Command",
            outputs=None,
            status="error",
        )
        cleaned = strip_ansi(output or "")
        lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return
        self._logger.error("Captured output:")
        for line in lines:
            self._logger.error("%s", line)

    def run(
        self,
        cmd: Sequence[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[Path] = None,
        step_name: Optional[str] = None,
        outputs: Optional[Sequence[Path | str]] = None,
        reason: Optional[str] = None,
        prepare: Optional[Callable[[], None]] = None,
        finalize: Optional[Callable[[], None]] = None,
    ) -> None:
        human_name = str(step_name or self._guess_step_name(cmd)).strip()
        if self._container is None:
            cmd_str, started_at, step = self._log_command_start("Step", cmd, cwd=cwd, step_name=human_name, outputs=outputs, reason=reason)
            try:
                if prepare is not None:
                    prepare()
                subprocess.run(list(cmd), env=env, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=True)
                if finalize is not None:
                    finalize()
                self._validate_declared_outputs(outputs, cwd=cwd)
            except subprocess.CalledProcessError as e:
                self._log_command_failure("Step", step, cwd=cwd, step_name=human_name, output=self._combined_output(e.stdout, e.stderr))
                raise SystemExit(f"Command failed ({e.returncode}): {cmd_str}") from e
            except BaseException as error:
                self._log_command_failure(
                    "Step", step, cwd=cwd, step_name=human_name, output=str(error)
                )
                raise
            self._log_command_success("Step", started_at, step, cwd=cwd, step_name=human_name)
            return

        full_cmd = self._container_prefix_for_cwd(cwd) + ["bash", "-lc", self._inner_cmd(cmd, env)]
        cmd_str, started_at, step = self._log_command_start("Step (container)", full_cmd, cwd=cwd, step_name=human_name, outputs=outputs, reason=reason)
        host_env = self._host_env_for_container(env)
        try:
            if prepare is not None:
                prepare()
            subprocess.run(full_cmd, env=host_env, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=True)
            if finalize is not None:
                finalize()
            self._validate_declared_outputs(outputs, cwd=cwd)
        except subprocess.CalledProcessError as e:
            self._log_command_failure("Step (container)", step, cwd=cwd, step_name=human_name, output=self._combined_output(e.stdout, e.stderr))
            raise SystemExit(f"Command failed ({e.returncode}): {cmd_str}") from e
        except BaseException as error:
            self._log_command_failure(
                "Step (container)", step, cwd=cwd, step_name=human_name, output=str(error)
            )
            raise
        self._log_command_success("Step (container)", started_at, step, cwd=cwd, step_name=human_name)

    def run_child(
        self,
        cmd: Sequence[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[Path] = None,
        capture_stdout: bool = False,
        stream_output: bool = False,
        timeout_seconds: Optional[float] = None,
        direct: bool = False,
    ) -> Optional[str]:
        """Execute one command within the active fixed-output artifact node.

        Directory-producing tools often require several commands but still
        represent one atomic DAG node. These commands are logged beneath that
        Step and may not create independent, output-less pseudo-nodes.
        """
        state = _RUNNER_EXECUTION.get()
        if state is None or len(state.active_steps) < 2:
            raise RuntimeError("Child commands require an active artifact step")
        if capture_stdout and stream_output:
            raise ValueError("A child command cannot both return and stream stdout")
        if self._container is None or direct:
            full_cmd = list(cmd)
            host_env = env
        else:
            full_cmd = self._container_prefix_for_cwd(cwd) + ["bash", "-lc", self._inner_cmd(cmd, env)]
            host_env = self._host_env_for_container(env)
        rendered = self._format_cmd(full_cmd)
        self._logger.info("Cmd: %s%s", rendered, self._cwd_note(cwd))
        try:
            if stream_output:
                started_at = time.monotonic()
                deadline = (
                    started_at + timeout_seconds
                    if timeout_seconds is not None
                    else None
                )
                proc = subprocess.Popen(
                    full_cmd,
                    env=host_env,
                    cwd=str(cwd) if cwd else None,
                    text=True,
                )
                while True:
                    remaining = (
                        None if deadline is None else max(0.0, deadline - time.monotonic())
                    )
                    wait_for = 60.0 if remaining is None else min(60.0, remaining)
                    try:
                        return_code = proc.wait(timeout=wait_for)
                        break
                    except subprocess.TimeoutExpired:
                        if deadline is not None and time.monotonic() >= deadline:
                            proc.terminate()
                            try:
                                proc.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            raise subprocess.TimeoutExpired(full_cmd, timeout_seconds)
                        self._logger.info(
                            "Command still running after %.0f seconds: %s",
                            time.monotonic() - started_at,
                            rendered,
                        )
                if return_code:
                    raise subprocess.CalledProcessError(return_code, full_cmd)
            else:
                proc = subprocess.run(
                    full_cmd,
                    env=host_env,
                    cwd=str(cwd) if cwd else None,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=timeout_seconds,
                )
        except subprocess.CalledProcessError as error:
            for line in strip_ansi(error.stdout or "").splitlines():
                if line.strip():
                    self._logger.error("Stdout: %s", line.rstrip())
            for line in strip_ansi(error.stderr or "").splitlines():
                if line.strip():
                    self._logger.error("Stderr: %s", line.rstrip())
            raise SystemExit(f"Command failed ({error.returncode}): {rendered}") from error
        except subprocess.TimeoutExpired as error:
            raise SystemExit(
                f"Command timed out after {timeout_seconds}s: {rendered}"
            ) from error
        if proc.stderr:
            for line in strip_ansi(proc.stderr).splitlines():
                if line.strip():
                    self._logger.info("Stderr: %s", line.rstrip())
        if not capture_stdout:
            return None
        output = (proc.stdout or "").strip()
        return self._restore_host_paths(output) if self._container is not None else output

    def run_out(
        self,
        cmd: Sequence[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[Path] = None,
        outputs: Optional[Sequence[Path | str]] = None,
        reason: Optional[str] = None,
        quiet: bool = False,
    ) -> str:
        # ``quiet`` is retained as a caller-facing logging hint, but unified
        # execution always records a numbered operation.  There is no hidden
        # subprocess path.
        _ = quiet
        if self._container is None:
            _, started_at, step = self._log_command_start(
                "Step", cmd, cwd=cwd, outputs=outputs, reason=reason
            )
            try:
                proc = subprocess.run(
                    list(cmd), check=True, env=env,
                    cwd=str(cwd) if cwd else None, text=True, capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                self._log_command_failure(
                    "Step", step, cwd=cwd,
                    output=self._combined_output(e.stdout, e.stderr),
                )
                raise SystemExit(
                    f"Command failed ({e.returncode}): {self._format_cmd(cmd)}"
                ) from e
            self._log_command_success("Step", started_at, step, cwd=cwd)
            return (proc.stdout or "").strip()
        full_cmd = self._container_prefix_for_cwd(cwd) + ["bash", "-lc", self._inner_cmd(cmd, env)]
        host_env = self._host_env_for_container(env)
        _, started_at, step = self._log_command_start(
            "Step (container)", full_cmd, cwd=cwd, outputs=outputs, reason=reason
        )
        try:
            proc = subprocess.run(
                full_cmd, check=True, env=host_env,
                cwd=str(cwd) if cwd else None, text=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            self._log_command_failure(
                "Step (container)", step, cwd=cwd,
                output=self._combined_output(e.stdout, e.stderr),
            )
            raise SystemExit(
                f"Command failed ({e.returncode}): {self._format_cmd(full_cmd)}"
            ) from e
        self._log_command_success("Step (container)", started_at, step, cwd=cwd)
        return self._restore_host_paths((proc.stdout or "").strip())

    def run_direct(
        self,
        args: Sequence[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[Path] = None,
        step_name: Optional[str] = None,
        outputs: Optional[Sequence[Path | str]] = None,
        reason: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        prepare: Optional[Callable[[], None]] = None,
        finalize: Optional[Callable[[], None]] = None,
    ) -> None:
        cmd_str, started_at, step = self._log_command_start("Step", args, cwd=cwd, step_name=step_name, outputs=outputs, reason=reason)
        try:
            if prepare is not None:
                prepare()
            subprocess.run(
                list(args), env=env, cwd=str(cwd) if cwd else None,
                text=True, capture_output=True, check=True, timeout=timeout_seconds,
            )
            if finalize is not None:
                finalize()
            self._validate_declared_outputs(outputs, cwd=cwd)
        except subprocess.CalledProcessError as e:
            self._log_command_failure("Step", step, cwd=cwd, step_name=step_name, output=self._combined_output(e.stdout, e.stderr))
            raise SystemExit(f"Command failed ({e.returncode}): {cmd_str}") from e
        except subprocess.TimeoutExpired as e:
            self._log_command_failure("Step", step, cwd=cwd, step_name=step_name, output=self._combined_output(e.stdout, e.stderr))
            raise SystemExit(f"Command timed out after {timeout_seconds}s: {cmd_str}") from e
        except BaseException as error:
            self._log_command_failure(
                "Step", step, cwd=cwd, step_name=step_name, output=str(error)
            )
            raise
        self._log_command_success("Step", started_at, step, cwd=cwd, step_name=step_name)

    def require_cmds(self, cmds: Sequence[str]) -> None:
        requested = tuple(dict.fromkeys(str(command) for command in cmds))
        with self.python_step(
            step_name="Dependency Preflight",
            outputs=None,
            reason=f"Checking {len(requested)} required command(s).",
        ):
            missing: list[str] = []
            for c in requested:
                if self._container is None:
                    full_cmd = ["bash", "-lc", f"command -v {shlex_quote(c)} >/dev/null 2>&1"]
                    proc = subprocess.run(full_cmd, env=os.environ.copy(), text=True, capture_output=True)
                    if proc.returncode != 0:
                        missing.append(c)
                else:
                    check_cmd = self._inner_cmd(["command", "-v", c], None)
                    full_cmd = self._container_prefix() + ["bash", "-lc", check_cmd]
                    proc = subprocess.run(full_cmd, env=self._host_env_for_container(None), text=True, capture_output=True)
                    if proc.returncode != 0:
                        missing.append(c)
            if missing:
                if self._container is None:
                    msg = "Missing required commands on PATH:\n" + "\n".join(f"- {c}" for c in missing)
                else:
                    msg = "Missing required commands inside container:\n" + "\n".join(f"- {c}" for c in missing)
                    msg += f"\n\nContainer image: {self._container.image}"
                raise SystemExit(msg)
