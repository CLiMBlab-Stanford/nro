"""Run planned analysis work items and publish their outputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nro.analysis.planning import AnalysisWorkItem, provenance_path
from nro.engine.io import atomic_write_text
from nro.engine.paths import study_work_root


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of one executed work item."""

    item: AnalysisWorkItem
    succeeded: bool
    log: Path
    message: str = ""


def _input_spec(value):
    if value is None:
        return None
    if hasattr(value, "artifacts"):
        return [artifact.as_spec() for artifact in value.artifacts]
    return value.as_spec()


def specification(item: AnalysisWorkItem, *, code_root: Path, staging: Path) -> dict:
    """Return the JSON specification the runner reads."""
    analysis = item.analysis
    return {
        "analysis": analysis.name,
        "commit": analysis.source.commit,
        "code_root": str(code_root),
        "entry": analysis.manifest.entry,
        "inputs": {name: _input_spec(value) for name, value in item.input_values().items()},
        "outputs": {slot: str(staging / path.name) for slot, path in item.outputs.items()},
        "parameters": item.parameters,
        "entities": {"project": item.project, "participant": item.participant, **item.entities},
        "work_dir": str(staging.parent),
    }


def runner_command(container: str | None, spec_path: Path) -> list[str]:
    """Return the command that runs `nro.analysis.runner` on one specification.

    Without a container the current interpreter runs it. With one, the site's
    container runtime runs the image with the site binds and this checkout on
    `PYTHONPATH`.
    """
    runner = ["-m", "nro.analysis.runner", str(spec_path)]
    if container is None:
        return [sys.executable, *runner]
    from nro.site.configuration import CHECKOUT, settings

    values = settings()[0]
    command = [str(values["runtime"]), "exec", "--cleanenv"]
    for bind in values.get("binds") or ():
        command.extend(["--bind", str(bind)])
    command.extend(["--env", f"PYTHONPATH={CHECKOUT}", container, "python", *runner])
    return command


def publish_output(source: Path, destination: Path) -> None:
    """Copy one staged output with its metadata and replace `destination` atomically.

    Empty files are allowed. Parent directories are created as needed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.publishing")
    shutil.copy2(source, staged)
    os.replace(staged, destination)


def execute(
    item: AnalysisWorkItem,
    *,
    publish: bool = True,
    work_dir: Path | None = None,
) -> ExecutionResult:
    """Run one ready work item.

    Outputs are written to a staging directory. After the entry function returns,
    every slot must exist and be nonempty. Published outputs are then copied into
    place with a provenance sidecar; files the code wrote outside its slots are
    left in the work directory and never published.
    """
    if item.status not in {"ready", "fresh"} or item.signature is None:
        raise ValueError(f"{item.key} is {item.status} and cannot run: {item.message}")
    analysis = item.analysis
    code_root = analysis.source.snapshot()
    if work_dir is None:
        work_dir = study_work_root(analysis.study.name) / "runs" / item.signature[:20]
    if work_dir.exists():
        shutil.rmtree(work_dir)
    staging = work_dir / "outputs"
    staging.mkdir(parents=True)
    spec = specification(item, code_root=code_root, staging=staging)
    spec_path = work_dir / "spec.json"
    atomic_write_text(spec_path, json.dumps(spec, indent=2, sort_keys=True, default=str) + "\n")
    log = work_dir / "log.txt"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            runner_command(item.analysis.container, spec_path),
            cwd=work_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        return ExecutionResult(item, False, log, f"analysis code exited with {result.returncode}")
    missing = [
        slot
        for slot, path in item.outputs.items()
        if not (staging / path.name).is_file() or (staging / path.name).stat().st_size == 0
    ]
    if missing:
        return ExecutionResult(item, False, log, "unwritten output slots: " + ", ".join(missing))
    if publish:
        provenance = {
            "Analysis": analysis.name,
            "Commit": analysis.source.commit,
            "Signature": item.signature,
            "Parameters": item.parameters,
            "Entities": spec["entities"],
            "Inputs": spec["inputs"],
            "Published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        for slot, path in item.outputs.items():
            publish_output(staging / path.name, path)
            atomic_write_text(
                provenance_path(path),
                json.dumps({**provenance, "Slot": slot}, indent=2, sort_keys=True, default=str)
                + "\n",
            )
        item.status = "fresh"
    return ExecutionResult(item, True, log)
