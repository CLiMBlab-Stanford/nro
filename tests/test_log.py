from __future__ import annotations

import json
from pathlib import Path

import nro.bin.log as log_cli
from nro.bidsify.store import IngestionStore
from nro.configuration.store import ConfigStore
from nro.engine.cli import core_selection
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.planner import build_subject_instances
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.scheduler_service import logs as scheduler_logs


def _write(path: Path, text: str = "log\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _registry_with_logs(tmp_path: Path) -> tuple[Path, Registry, Path, Path, Path, Path]:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")

    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="func",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    registry.create_request(
        registered=registered,
        target_module="func",
        selectors={},
        instances=instances,
        terminal_instance_keys=[
            instance.key for instance in instances if instance.module == "func"
        ],
        concurrency=2,
        partition=None,
    )
    rows = {str(row["module"]): row for row in registry.instance_rows()}
    now = utcnow()
    anat_instance_log = _write(registry.paths.events / "anat-attempt.log", "anat instance\n")
    func_instance_log = _write(registry.paths.events / "func-attempt.log", "func instance\n")
    anat_worker_log = _write(registry.paths.workers / "slurm-101.log", "anat worker\n")
    func_worker_log = _write(registry.paths.workers / "slurm-202.log", "func worker\n")
    registry.register_worker("anat-worker", resource_class="large", slurm_job_id="101")
    registry.register_worker("func-worker", resource_class="large", slurm_job_id="202")
    with registry.connection(write=True) as db:
        for module, worker, path in (
            ("anat", "anat-worker", anat_instance_log),
            ("func", "func-worker", func_instance_log),
        ):
            row = rows[module]
            db.execute(
                """INSERT INTO attempts(
                       instance_id, worker_id, state, revision_fingerprint, memory_gb,
                       started_at, completed_at, log_path, created_at
                   ) VALUES (?, ?, 'success', ?, 32, ?, ?, ?, ?)""",
                (
                    row["id"],
                    worker,
                    row["revision_fingerprint"],
                    now,
                    now,
                    str(path),
                    now,
                ),
            )
    return bids, registry, anat_instance_log, func_instance_log, anat_worker_log, func_worker_log


def test_collects_only_workers_that_attempted_matching_instances(tmp_path: Path) -> None:
    _bids, registry, _anat_instance, _func_instance, anat_worker, func_worker = _registry_with_logs(
        tmp_path
    )
    instance_ids = log_cli._matching_instance_ids(
        registry,
        projects={"demo"},
        participants=["01"],
        modules={"func"},
        workflows={"main"},
        selectors={"run": ("1",)},
    )

    filtered = log_cli.collect_log_paths(
        registry, instance_ids=instance_ids, instance_level=False, instance_filtered=True
    )
    unfiltered = log_cli.collect_log_paths(
        registry, instance_ids=set(), instance_level=False, instance_filtered=False
    )

    assert filtered == [func_worker.resolve()]
    assert set(unfiltered) == {anat_worker.resolve(), func_worker.resolve()}


def test_instance_level_collects_attempt_logs_for_matching_instances(tmp_path: Path) -> None:
    _bids, registry, anat_instance, func_instance, _anat_worker, _func_worker = _registry_with_logs(
        tmp_path
    )
    instance_ids = log_cli._matching_instance_ids(
        registry,
        projects={"demo"},
        participants=[],
        modules={"func"},
        workflows=set(),
        selectors={},
    )

    filtered = log_cli.collect_log_paths(
        registry, instance_ids=instance_ids, instance_level=True, instance_filtered=True
    )
    unfiltered = log_cli.collect_log_paths(
        registry, instance_ids=set(), instance_level=True, instance_filtered=False
    )

    assert filtered == [func_instance.resolve()]
    assert set(unfiltered) == {anat_instance.resolve(), func_instance.resolve()}


def test_main_opens_all_matching_logs_in_one_less_session(tmp_path: Path, monkeypatch) -> None:
    bids, _registry, _anat_instance, func_instance, _anat_worker, _func_worker = (
        _registry_with_logs(tmp_path)
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(log_cli.shutil, "which", lambda command: "/usr/bin/less")

    def fake_run(command: list[str], *, check: bool):
        assert check is False
        commands.append(command)

    monkeypatch.setattr(log_cli.subprocess, "run", fake_run)
    log_cli.main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "func",
            "-w",
            "main",
            "-r",
            "run=1",
            "-i",
        ]
    )

    assert commands == [["/usr/bin/less", "-R", "--", str(func_instance.resolve())]]


def test_bidsify_module_selector_opens_matching_ingestion_logs(tmp_path: Path, monkeypatch) -> None:
    _bids, registry, *_ = _registry_with_logs(tmp_path)
    store = IngestionStore(registry)
    store.root.mkdir(parents=True, exist_ok=True)
    request_id = "request-01"
    (store.root / f"{request_id}.json").write_text(
        json.dumps(
            {
                "id": request_id,
                "project": "demo",
                "participant": "01",
                "session": "visit1",
            }
        )
    )
    request_log = _write(store.root / f"{request_id}.log", "ingestion\n")
    commands: list[list[str]] = []
    monkeypatch.setattr(log_cli.shutil, "which", lambda _command: "/usr/bin/less")
    monkeypatch.setattr(
        log_cli.subprocess,
        "run",
        lambda command, *, check: commands.append(command),
    )

    log_cli.main(["-m", "bidsify", "-P", "demo", "-p", "01", "-r", "ses=visit1"])

    assert commands == [["/usr/bin/less", "-R", "--", str(request_log.resolve())]]
    selection = core_selection(log_cli.build_parser().parse_args(["-m", "bidsify"]))
    assert log_cli.collect_bidsify_log_paths(registry, selection) == [request_log.resolve()]


def test_scheduler_resolves_branch_bidsification_logs(tmp_path: Path, monkeypatch) -> None:
    registry = Registry.for_project("", bids_root=tmp_path / "bids")
    branches = BranchStore(registry.paths.control)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(
        "nro.orchestration.branches.checkout_identity",
        lambda _checkout: (checkout, "main", "revision"),
    )
    branches.authorize_checkout("main", checkout, revision=branches.initialize().revision)
    store = IngestionStore(registry)
    store.root.mkdir(parents=True, exist_ok=True)
    request_id = "request-01"
    (store.root / f"{request_id}.json").write_text(
        json.dumps(
            {
                "id": request_id,
                "server": "cni",
                "project": "demo",
                "participant": "01",
                "session": "visit1",
                "state": "failed",
                "stage": "convert",
                "issues": [],
            }
        )
    )
    request_log = _write(store.root / f"{request_id}.log")

    result = scheduler_logs(
        registry,
        checkout=checkout,
        selection={
            "projects": [],
            "ingestion_projects": [],
            "participants": ["01"],
            "modules": ["bidsify"],
            "workflows": [],
            "selectors": {"ses": ("visit1",)},
        },
        instance_level=False,
    )

    assert result == {"paths": [str(request_log)]}
