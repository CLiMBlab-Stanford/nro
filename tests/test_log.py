from __future__ import annotations

import json
from pathlib import Path

import nro.bin.log as log_cli
from nro.bidsify.store import IngestionStore
from nro.definitions.store import ConfigStore
from nro.engine.cli import core_selection
from nro.orchestration import scheduler_client
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.planner import build_subject_work_items
from nro.orchestration.registry import Registry, utcnow


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
    work_items = build_subject_work_items(
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
        work_items=work_items,
        terminal_work_item_keys=[
            work_item.key for work_item in work_items if work_item.module == "func"
        ],
        concurrency=2,
        partition=None,
    )
    rows = {str(row["module"]): row for row in registry.work_item_rows()}
    now = utcnow()
    anat_work_item_log = _write(registry.paths.events / "anat-attempt.log", "anat work_item\n")
    func_work_item_log = _write(registry.paths.events / "func-attempt.log", "func work_item\n")
    anat_worker_log = _write(registry.paths.workers / "slurm-101.log", "anat worker\n")
    func_worker_log = _write(registry.paths.workers / "slurm-202.log", "func worker\n")
    registry.register_worker("anat-worker", resource_class="large", slurm_job_id="101")
    registry.register_worker("func-worker", resource_class="large", slurm_job_id="202")
    with registry.connection(write=True) as db:
        for module, worker, path in (
            ("anat", "anat-worker", anat_work_item_log),
            ("func", "func-worker", func_work_item_log),
        ):
            row = rows[module]
            db.execute(
                """INSERT INTO attempts(
                       work_item_id, worker_id, state, revision_fingerprint, memory_gb,
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
    return bids, registry, anat_work_item_log, func_work_item_log, anat_worker_log, func_worker_log


def test_collects_only_workers_that_attempted_matching_work_items(tmp_path: Path) -> None:
    _bids, registry, _anat_work_item, _func_work_item, anat_worker, func_worker = (
        _registry_with_logs(tmp_path)
    )
    work_item_ids = log_cli._matching_work_item_ids(
        registry,
        projects={"demo"},
        participants=["01"],
        modules={"func"},
        workflows={"main"},
        selectors={"run": ("1",)},
    )

    filtered = log_cli.collect_log_paths(
        registry, work_item_ids=work_item_ids, worker_level=True, work_item_filtered=True
    )
    unfiltered = log_cli.collect_log_paths(
        registry, work_item_ids=set(), worker_level=True, work_item_filtered=False
    )

    assert filtered == [func_worker.resolve()]
    assert set(unfiltered) == {anat_worker.resolve(), func_worker.resolve()}


def test_work_item_rows_expose_the_readable_configuration_route(tmp_path: Path) -> None:
    _bids, registry, *_logs = _registry_with_logs(tmp_path)
    func = next(row for row in registry.work_item_rows() if row["module"] == "func")

    assert json.loads(func["configuration_route_json"]) == [
        {"config": "main", "module": "anat"},
        {"config": "main", "module": "func"},
    ]


def test_default_level_collects_attempt_logs_for_matching_work_items(tmp_path: Path) -> None:
    _bids, registry, anat_work_item, func_work_item, _anat_worker, _func_worker = (
        _registry_with_logs(tmp_path)
    )
    work_item_ids = log_cli._matching_work_item_ids(
        registry,
        projects={"demo"},
        participants=[],
        modules={"func"},
        workflows=set(),
        selectors={},
    )

    filtered = log_cli.collect_log_paths(
        registry, work_item_ids=work_item_ids, worker_level=False, work_item_filtered=True
    )
    unfiltered = log_cli.collect_log_paths(
        registry, work_item_ids=set(), worker_level=False, work_item_filtered=False
    )

    assert filtered == [func_work_item.resolve()]
    assert set(unfiltered) == {anat_work_item.resolve(), func_work_item.resolve()}


def test_running_filter_keeps_only_work_items_with_active_attempts(tmp_path: Path) -> None:
    _bids, registry, _anat_work_item, _func_work_item, *_ = _registry_with_logs(tmp_path)
    func = next(row for row in registry.work_item_rows() if row["module"] == "func")
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE attempts SET state='running', completed_at=NULL WHERE work_item_id=?",
            (func["id"],),
        )

    assert log_cli._matching_work_item_ids(
        registry,
        projects={"demo"},
        participants=[],
        modules=set(),
        workflows=set(),
        selectors={},
        running_only=True,
    ) == {func["id"]}


def test_running_option_is_available() -> None:
    args = log_cli.build_parser(prog="nro log").parse_args(["--running"])

    assert args.running is True


def test_main_opens_all_matching_logs_in_one_less_session(tmp_path: Path, monkeypatch) -> None:
    bids, _registry, _anat_work_item, func_work_item, _anat_worker, _func_worker = (
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
        ]
    )

    assert commands == [["/usr/bin/less", "-R", "-S", "--", str(func_work_item.resolve())]]


def test_worker_option_opens_matching_worker_logs(tmp_path: Path, monkeypatch) -> None:
    bids, _registry, _anat_work_item, _func_work_item, _anat_worker, func_worker = (
        _registry_with_logs(tmp_path)
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(log_cli.shutil, "which", lambda _command: "/usr/bin/less")
    monkeypatch.setattr(
        log_cli.subprocess,
        "run",
        lambda command, *, check: commands.append(command),
    )

    directory = next(
        row["directory_label"] for row in _registry.work_item_rows() if row["module"] == "func"
    )
    log_cli.main(["-P", "demo", "-m", "func", "-i", f"func/{directory}", "--worker"])

    assert commands == [["/usr/bin/less", "-R", "-S", "--", str(func_worker.resolve())]]


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

    assert commands == [["/usr/bin/less", "-R", "-S", "--", str(request_log.resolve())]]
    selection = core_selection(log_cli.build_parser().parse_args(["-m", "bidsify"]))
    assert log_cli.collect_bidsify_log_paths(registry, selection) == [request_log.resolve()]
    assert log_cli.collect_bidsify_log_paths(registry, selection, running_only=True) == []


def test_cached_logs_resolve_ingestion_logs_for_each_branch(tmp_path: Path, monkeypatch) -> None:
    control = tmp_path / "control"
    rows = [
        {
            "id": "request-main",
            "branch": "main",
            "project": "demo",
            "participant": "01",
            "session": "visit1",
            "state": "failed",
        },
        {
            "id": "request-dev",
            "branch": "dev",
            "project": "demo",
            "participant": "01",
            "session": "visit1",
            "state": "running",
        },
    ]
    monkeypatch.setattr(
        scheduler_client,
        "_cached_status",
        lambda _control, _checkout: {"rows": [], "visible_ids": [], "ingestion": rows},
    )
    selection = {
        "projects": [],
        "ingestion_projects": [],
        "participants": ["01"],
        "modules": ["bidsify"],
        "workflows": [],
        "selectors": {"ses": ("visit1",)},
    }

    def resolve(**options) -> list[str]:
        return scheduler_client.logs(
            control, tmp_path / "bids", checkout=tmp_path, worker_level=False, **options
        )["paths"]

    paths = ControlPaths(control)
    assert resolve(selection=selection) == sorted(
        [
            str(paths.ingestion / "request-main.log"),
            str(paths.branch("dev") / "ingestion" / "request-dev.log"),
        ]
    )
    assert resolve(selection=selection, running_only=True) == [
        str(paths.branch("dev") / "ingestion" / "request-dev.log")
    ]
    assert resolve(selection={**selection, "selectors": {"run": ("1",)}}) == []
