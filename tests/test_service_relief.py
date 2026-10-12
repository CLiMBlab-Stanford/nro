"""Tests for conservative worker relief during service startup."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from nro.orchestration import scheduler_bus, service_relief
from nro.orchestration.service_relief import PendingService
from nro.orchestration.worker import Worker


def _scheduler_launch(control: Path, job_id: str = "123") -> None:
    claim = scheduler_bus.claim_launch(control)
    assert claim is not None
    scheduler_bus.update_launch_job(claim, job_id)


def _queue(monkeypatch, row: str) -> None:
    monkeypatch.setattr(
        service_relief.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            (
                "JobId=123 JobState=PENDING "
                f"Reason={row} Partition=sphinx Account=nlp NumCPUs=4 MinMemoryNode=4G\n"
            ),
            "",
        ),
    )


def test_job_limit_allows_one_idle_worker_to_relieve_scheduler(monkeypatch, tmp_path) -> None:
    control = tmp_path / ".nro"
    _scheduler_launch(control)
    _queue(monkeypatch, "QOSMaxJobsPerUserLimit")
    now = 1_000_000_000_000.0

    pending = service_relief.pending_service(
        control,
        worker_cpus=2,
        worker_memory_gb=32,
        worker_partition="sphinx",
        worker_account="nlp",
        now=now,
    )

    assert pending is not None
    assert pending.name == "scheduler"
    assert pending.job_id == "123"
    assert pending.reason == "QOSMaxJobsPerUserLimit"
    assert service_relief.claim_relief(control, pending, worker_id="worker-1", now=now)
    assert not service_relief.claim_relief(control, pending, worker_id="worker-2", now=now + 30)
    assert service_relief.claim_relief(control, pending, worker_id="worker-2", now=now + 61)


def test_priority_does_not_retire_workers(monkeypatch, tmp_path) -> None:
    control = tmp_path / ".nro"
    _scheduler_launch(control)
    _queue(monkeypatch, "Priority")

    assert (
        service_relief.pending_service(
            control,
            worker_cpus=8,
            worker_memory_gb=64,
            worker_partition="sphinx",
            worker_account="nlp",
            now=1_000_000_000_000.0,
        )
        is None
    )


def test_resource_relief_requires_one_compatible_allocation(monkeypatch, tmp_path) -> None:
    control = tmp_path / ".nro"
    _scheduler_launch(control)
    _queue(monkeypatch, "Resources")
    options = {
        "control": control,
        "worker_memory_gb": 32,
        "worker_partition": "sphinx",
        "worker_account": "nlp",
        "now": 1_000_000_000_000.0,
    }

    assert service_relief.pending_service(worker_cpus=2, **options) is None
    assert service_relief.pending_service(worker_cpus=4, **options) is not None
    assert (
        service_relief.pending_service(
            worker_cpus=4,
            **{**options, "worker_partition": "other"},
        )
        is None
    )


def test_elected_worker_exits_before_registration(monkeypatch, tmp_path) -> None:
    pending = PendingService("scheduler", "token", "123", "AssocMaxJobsLimit")
    monkeypatch.setattr(
        "nro.orchestration.worker.await_service_relief",
        lambda *_args, **_kwargs: pending,
    )

    class Registry:
        paths = SimpleNamespace(control=tmp_path / ".nro")

        def register_worker(self, *_args, **_kwargs):
            raise AssertionError("a retiring allocation must not register as a worker")

    worker = Worker(Registry(), resource_class="large", worker_id="worker-1")

    assert worker.run() == 0
    assert worker.service_relief == "scheduler"


def test_relief_cancels_only_pending_workers_from_same_store(monkeypatch, tmp_path) -> None:
    control = tmp_path / ".nro"
    workers = control / "shared/scheduler/workers"
    workers.mkdir(parents=True)
    own = workers / "worker-large-32gb-profile.sbatch"
    own.touch()
    other = tmp_path / "other/worker-large-32gb-profile.sbatch"
    other.parent.mkdir()
    other.touch()
    calls = []

    def run(command, **_options):
        calls.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command,
                0,
                "100|nro-worker\n101|nro-worker\n102|unrelated\n",
                "",
            )
        if command[-1] == "100":
            return subprocess.CompletedProcess(
                command, 0, f"JobId=100 JobState=PENDING Command={own}\n", ""
            )
        if command[-1] == "101":
            return subprocess.CompletedProcess(
                command, 0, f"JobId=101 JobState=PENDING Command={other}\n", ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(service_relief.subprocess, "run", run)

    assert service_relief.cancel_pending_workers(control) == 1
    assert calls[-1] == ["scancel", "--state=PENDING", "100"]
