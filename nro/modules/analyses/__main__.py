"""Run one scheduled study-analysis work item.

The planner writes an immutable specification for each work item. This entry
point chooses the input files from the outputs of the bound upstream work
items, runs the analysis code through ``nro.analysis.runner``, publishes each
output slot with a provenance sidecar, and writes the completion manifest that
assessment reads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from nro.analysis.artifacts import Artifact, file_entities
from nro.analysis.manifest import InputSpec
from nro.analysis.planning import provenance_path
from nro.analysis.resolve import (
    Resolution,
    _drop_sidecars,
    _matches_run,
    _raw_candidates,
    _signature,
    _upstream_files,
    choose,
)
from nro.engine.bids import matches_filter
from nro.engine.io import atomic_write_text


def _spec(name: str, entry: dict) -> InputSpec:
    return InputSpec(
        name=name,
        source=entry["module"] or "raw",
        workflow=entry.get("workflow"),
        config=None,
        select=entry["select"],
        many=entry["many"],
        minimum=entry["min"],
        maximum=entry["max"],
        optional=entry["optional"],
        path=entry.get("path"),
    )


def _physical_root(context, upstream: dict) -> Path:
    """Return where the bound producer of one upstream work item wrote its outputs."""
    logical = Path(upstream["root"])
    if context is None:
        return logical
    for binding in context.inputs:
        if binding.logical_root == logical and binding.prefix == upstream["prefix"]:
            return binding.physical_root
    raise ValueError(f"No input binding for upstream work item {upstream['key']}")


def resolve_inputs(specification: dict, context=None) -> dict[str, Resolution]:
    """Choose each input's files from bound upstream outputs or source BIDS."""
    entities = specification["entities"]
    project, participant = entities["project"], entities["participant"]
    run_entities = specification["run_entities"] or None
    where = f"sub-{participant} in project {project}"
    result = {}
    for name, entry in specification["inputs"].items():
        spec = _spec(name, entry)
        if spec.module is None:
            bids = context.paths.bids if context is not None else Path(os.environ["NRO_BIDS"])
            candidates = _raw_candidates(
                spec,
                project=project,
                participant=participant,
                run_entities=run_entities,
                bids_root=bids,
            )
        else:
            candidates = []
            for upstream in entry["upstream"]:
                root = _physical_root(context, upstream)
                recorded = {str(key): str(value) for key, value in upstream["entities"].items()}
                item = {"output_root": str(root), "output_prefix": upstream["prefix"]}
                for path in _drop_sidecars(_upstream_files(item), spec.select):
                    found = {**recorded, **file_entities(path)}
                    if matches_filter(found, spec.select) and _matches_run(found, run_entities):
                        candidates.append(
                            Artifact(
                                path=path,
                                entities=found,
                                module=spec.module,
                                signature=_signature(
                                    {"work_item": upstream["key"], "file": path.name}
                                ),
                                project=project,
                                participant=participant,
                                lineage=entry.get("workflow"),
                            )
                        )
        result[name] = choose(spec, candidates, where=where)
    return result


def _runner_command(container: str | None, path: Path) -> list[str]:
    runner = ["-m", "nro.analysis.runner", str(path)]
    if container is None:
        return [sys.executable, *runner]
    from nro.site.configuration import CHECKOUT, settings

    values = settings()[0]
    command = [str(values["runtime"]), "exec", "--cleanenv"]
    for bind in values.get("binds") or ():
        command.extend(["--bind", str(bind)])
    command.extend(["--env", f"PYTHONPATH={CHECKOUT}", container, "python", *runner])
    return command


def _publish(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.publishing")
    shutil.copy2(source, staged)
    os.replace(staged, destination)


def run(specification: dict, *, context=None, work_dir: Path) -> None:
    """Execute one specification and publish its outputs.

    Raises ``RuntimeError`` when an input cannot be resolved, the analysis
    code fails, or a slot is left empty. Nothing is published in those cases.
    """
    resolutions = resolve_inputs(specification, context)
    problems = [value.message for value in resolutions.values() if value.status != "ok"]
    if problems:
        raise RuntimeError("; ".join(problems))

    def destination(path: str) -> Path:
        return context.output_path(Path(path)) if context is not None else Path(path)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    staging = work_dir / "outputs"
    staging.mkdir(parents=True)
    inputs = {}
    for name, resolution in resolutions.items():
        value = resolution.value
        if value is None:
            inputs[name] = None
        elif hasattr(value, "artifacts"):
            inputs[name] = [artifact.as_spec() for artifact in value.artifacts]
        else:
            inputs[name] = value.as_spec()
    runner_spec = {
        "analysis": specification["analysis"],
        "commit": specification["commit"],
        "code_root": specification["code_root"],
        "entry": specification["entry"],
        "inputs": inputs,
        "outputs": {
            slot: str(staging / Path(path).name) for slot, path in specification["outputs"].items()
        },
        "parameters": specification["parameters"],
        "entities": specification["entities"],
        "work_dir": str(work_dir),
    }
    runner_path = work_dir / "runner.json"
    atomic_write_text(runner_path, json.dumps(runner_spec, indent=2, sort_keys=True) + "\n")
    log = work_dir / "log.txt"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            _runner_command(specification["container"], runner_path),
            cwd=work_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    sys.stderr.write(log.read_text(encoding="utf-8", errors="replace"))
    if completed.returncode != 0:
        raise RuntimeError(f"Analysis code exited with status {completed.returncode}")
    missing = [
        slot
        for slot, path in runner_spec["outputs"].items()
        if not Path(path).is_file() or Path(path).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError("Analysis left output slots unwritten: " + ", ".join(missing))
    provenance = {
        "Analysis": specification["analysis"],
        "Commit": specification["commit"],
        "Parameters": specification["parameters"],
        "Entities": specification["entities"],
        "Inputs": inputs,
        "Published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    published = []
    for slot, logical in specification["outputs"].items():
        target = destination(logical)
        _publish(staging / Path(logical).name, target)
        sidecar = provenance_path(target)
        atomic_write_text(
            sidecar,
            json.dumps({**provenance, "Slot": slot}, indent=2, sort_keys=True, default=str) + "\n",
        )
        published.extend((target.name, sidecar.name))
    completion = destination(specification["completion"])
    atomic_write_text(
        completion,
        json.dumps(
            {
                "complete": True,
                "public_outputs": published,
                "analysis": specification["analysis"],
                "commit": specification["commit"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def main(argv: list[str] | None = None, *, execution_context=None) -> int:
    """Run the work item named by ``--specification``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    args = parser.parse_args(argv)
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    from nro.engine.paths import study_work_root

    study = specification["analysis"].split("/", 1)[0]
    branch = execution_context.paths.branch if execution_context is not None else "main"
    work_dir = study_work_root(study) / "attempts" / branch / args.specification.stem
    run(specification, context=execution_context, work_dir=work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
