"""Slurm worker scripts and submission shared by derivatives and ingestion."""

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

from nro.engine.io import atomic_write_text
from .registry import Registry

DEFAULT_WORKER_IDLE_TIMEOUT = 30


def _worker_source_root() -> Path:
    """Return the repository root that workers import."""
    return Path(__file__).resolve().parents[2]


def _write_worker_script(
    registry: Registry,
    *,
    bids_root: Path,
    partition: str,
    account: str | None,
    hours: int,
    memory_gb: int,
    cpus: int,
    idle_timeout: int = DEFAULT_WORKER_IDLE_TIMEOUT,
    drain_seconds: int = 15 * 60,
) -> Path:
    from nro.configuration.site import site_file
    site_path = site_file()
    selected_site = str(site_path) if site_path.is_file() else None
    profile_payload = json.dumps(
        {
            "bids_root": str(bids_root),
            "site_config": selected_site,
            "partition": partition,
            "account": account,
            "hours": hours,
            "cpus": cpus,
            "idle_timeout": idle_timeout,
            "drain_seconds": drain_seconds,
        },
        sort_keys=True,
    )
    profile = hashlib.sha256(profile_payload.encode("utf-8")).hexdigest()[:12]
    path = registry.paths.workers / f"worker-large-{memory_gb}gb-{profile}.sbatch"
    command = [
        sys.executable, "-m", "nro.orchestration.worker",
        "--bids-root", str(bids_root), "--resource-class", "large",
        "--memory-gb", str(memory_gb),
        "--idle-timeout", str(idle_timeout),
        "--walltime-seconds", str(hours * 60 * 60),
        "--drain-seconds", str(drain_seconds),
        "--profile", profile,
    ]
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=nro-worker",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={hours}:00:00",
        f"#SBATCH --mem={memory_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={registry.paths.workers}/slurm-%j.log",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines.extend((
        "set -euo pipefail",
        *(["export NRO_SITE_CONFIG=" + shlex.quote(selected_site)] if selected_site else []),
        f"cd {shlex.quote(str(_worker_source_root()))}",
        "exec " + shlex.join(command),
    ))
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _submit_workers(
    registry: Registry,
    request_id: str | None,
    script: Path,
    memory_gb: int,
) -> list[str]:
    submitted: list[str] = []
    registry.reconcile_scheduler_submissions()
    for submission_id, _token in registry.reserve_worker_submissions(
        request_id=request_id, resource_class="large", memory_gb=memory_gb
    ):
        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(script)],
                check=True,
                text=True,
                capture_output=True,
            )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id:
                raise RuntimeError(f"sbatch returned no job ID: {result.stdout!r}")
            registry.update_submission(submission_id, state="submitted", slurm_job_id=job_id)
            submitted.append(job_id)
        except BaseException:
            registry.update_submission(submission_id, state="error")
            raise
    return submitted
