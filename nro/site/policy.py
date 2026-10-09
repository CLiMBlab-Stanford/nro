"""Declare durable site settings and how running services consume them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SiteSetting:
    """Describe one public site key and its internal application lifecycle."""

    storage_key: str
    kind: str
    effect: str
    description: str
    minimum: int | None = None
    path: bool = False
    maintenance: bool = False
    scheduler_operation: str | None = None


def _setting(
    storage_key: str,
    kind: str,
    effect: str,
    description: str,
    **kwargs,
) -> SiteSetting:
    return SiteSetting(storage_key, kind, effect, description, **kwargs)


SITE_SETTINGS = {
    "storage.definitions": _setting(
        "definitions", "str", "installation", "Definitions store", path=True, maintenance=True
    ),
    "storage.bids": _setting(
        "bids",
        "str",
        "installation",
        "Directory containing BIDS projects",
        path=True,
        maintenance=True,
    ),
    "storage.work": _setting(
        "work", "str", "installation", "Intermediate work directory", path=True, maintenance=True
    ),
    "storage.development": _setting(
        "development", "str", "installation", "Development branch data", path=True, maintenance=True
    ),
    "storage.registry": _setting(
        "registry",
        "str",
        "installation",
        "Shared control state and logs",
        path=True,
        maintenance=True,
    ),
    "resources.images": _setting(
        "images", "str", "installation", "Container image directory", path=True, maintenance=True
    ),
    "resources.gradient_coefficients": _setting(
        "gradient_coefficients",
        "str",
        "installation",
        "Gradient coefficient directory",
        path=True,
        maintenance=True,
    ),
    "resources.templates": _setting(
        "templates", "str", "installation", "TemplateFlow directory", path=True, maintenance=True
    ),
    "resources.workbench": _setting(
        "workbench", "str", "installation", "wb_command launcher", path=True, maintenance=True
    ),
    "resources.oslom": _setting(
        "oslom", "str", "installation", "oslom_undir executable", path=True, maintenance=True
    ),
    "resources.license": _setting(
        "license", "str", "installation", "FreeSurfer license", path=True, maintenance=True
    ),
    "resources.qunex": _setting(
        "qunex", "str", "installation", "QuNex container", path=True, maintenance=True
    ),
    "resources.synthstrip": _setting(
        "synthstrip", "str", "installation", "SynthStrip container", path=True, maintenance=True
    ),
    "resources.synbold": _setting(
        "synbold", "str", "installation", "SynBOLD-DISCO container", path=True, maintenance=True
    ),
    "resources.gradient_unwarp": _setting(
        "gradient_unwarp",
        "str",
        "installation",
        "Gradient-unwarping container",
        path=True,
        maintenance=True,
    ),
    "resources.freesurfer": _setting(
        "freesurfer", "str", "installation", "FreeSurfer container", path=True, maintenance=True
    ),
    "resources.fastsurfer": _setting(
        "fastsurfer", "str", "installation", "FastSurfer container", path=True, maintenance=True
    ),
    "resources.workbench_image": _setting(
        "workbench_image",
        "str",
        "installation",
        "Container-backed Workbench image",
        path=True,
        maintenance=True,
    ),
    "resources.fastsurfer_data": _setting(
        "fastsurfer_data",
        "str",
        "installation",
        "FastSurfer model data",
        path=True,
        maintenance=True,
    ),
    "resources.synthstroke_data": _setting(
        "synthstroke_data",
        "str",
        "installation",
        "SynthStroke model data",
        path=True,
        maintenance=True,
    ),
    "resources.mni_template": _setting(
        "mni_template", "str", "installation", "MNI template image", path=True, maintenance=True
    ),
    "execution.runtime": _setting(
        "runtime", "str", "new process", "Singularity or Apptainer executable", maintenance=True
    ),
    "execution.binds": _setting(
        "binds", "list", "new process", "Container bind paths", maintenance=True
    ),
    "execution.concurrency": _setting(
        "concurrency",
        "int",
        "immediate",
        "Maximum general workers shared by derivative and ingestion work",
        minimum=1,
        scheduler_operation="concurrency",
    ),
    "execution.gpu_concurrency": _setting(
        "gpu_concurrency",
        "int",
        "immediate",
        "Maximum workers executing GPU steps",
        minimum=1,
        scheduler_operation="gpu_concurrency",
    ),
    "execution.worker_idle_timeout_seconds": _setting(
        "worker_idle_timeout", "int", "new request", "Idle time before a worker exits", minimum=1
    ),
    "execution.worker_drain_minutes": _setting(
        "worker_drain_minutes", "int", "new request", "Time reserved for worker shutdown", minimum=0
    ),
    "slurm.partition": _setting("partition", "str", "new allocation", "Default Slurm partition"),
    "slurm.viewer_partition": _setting(
        "viewing_partition", "str", "new viewer", "Slurm partition for interactive viewers"
    ),
    "slurm.account": _setting("account", "optional_str", "new allocation", "Slurm account"),
    "slurm.scheduler.time_hours": _setting(
        "scheduler_time", "int", "next scheduler", "Scheduler wall time", minimum=1
    ),
    "slurm.scheduler.memory_gb": _setting(
        "scheduler_memory", "int", "next scheduler", "Scheduler host memory", minimum=1
    ),
    "slurm.scheduler.cpus": _setting(
        "scheduler_cpus", "int", "next scheduler", "Scheduler CPUs", minimum=1
    ),
    "slurm.planner.time_hours": _setting(
        "planner_time", "int", "next planner", "Planner wall time", minimum=1
    ),
    "slurm.planner.memory_gb": _setting(
        "planner_memory", "int", "next planner", "Planner host memory", minimum=1
    ),
    "slurm.planner.cpus": _setting(
        "planner_cpus", "int", "next planner", "Planner CPUs", minimum=1
    ),
    "slurm.worker.time_hours": _setting(
        "worker_time", "int", "new request", "General and GPU worker wall time", minimum=1
    ),
    "slurm.worker.memory_gb": _setting(
        "worker_memory", "int", "new request", "Initial worker host memory", minimum=1
    ),
    "slurm.worker.max_memory_gb": _setting(
        "worker_max_memory", "int", "new request", "Maximum retried worker host memory", minimum=1
    ),
    "slurm.worker.cpus": _setting(
        "worker_cpus", "int", "new request", "General and GPU worker CPUs", minimum=1
    ),
    "slurm.long_worker.time_hours": _setting(
        "long_worker_time", "int", "new request", "Long CPU worker wall time", minimum=1
    ),
    "slurm.long_worker.cpus": _setting(
        "long_worker_cpus", "int", "new request", "Long CPU worker CPUs", minimum=1
    ),
    "slurm.viewer.time_hours": _setting(
        "viewer_time", "int", "new viewer", "Viewer server wall time", minimum=1
    ),
    "slurm.viewer.memory_gb": _setting(
        "viewer_memory", "int", "new viewer", "Viewer server host memory", minimum=1
    ),
    "slurm.viewer.cpus": _setting(
        "viewer_cpus", "int", "new viewer", "Viewer server CPUs", minimum=1
    ),
    "bidsify.default_server": _setting(
        "flywheel_server", "optional_str", "next request", "Default Flywheel server"
    ),
    "bidsify.default_project": _setting(
        "flywheel_project", "optional_str", "next request", "Default Flywheel group/project"
    ),
}

BY_STORAGE_KEY = {spec.storage_key: name for name, spec in SITE_SETTINGS.items()}


def canonical_key(value: str) -> str:
    """Return a public dotted key, accepting a legacy flat key when unambiguous."""
    if value in SITE_SETTINGS:
        return value
    try:
        return BY_STORAGE_KEY[value]
    except KeyError as error:
        raise ValueError(f"Unknown site setting {value!r}") from error


def parse_cli_value(spec: SiteSetting, value: str) -> object:
    """Parse one command-line value according to its declared site type."""
    if spec.kind == "int":
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError("value must be an integer") from error
        if spec.minimum is not None and parsed < spec.minimum:
            raise ValueError(f"value must be at least {spec.minimum}")
        return parsed
    if spec.kind == "list":
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("value must be a JSON list") from error
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("value must be a JSON list of strings")
        return parsed
    if spec.kind == "optional_str" and value in {"-", "null", "None"}:
        return ""
    parsed = str(value)
    if spec.path:
        parsed = str(Path(parsed).expanduser())
    return parsed


def format_cli_value(value: object) -> str:
    """Render a site value without losing list or null structure."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value == "":
        return "null"
    return str(value)
