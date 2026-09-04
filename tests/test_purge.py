from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.planner import build_subject_instances
from nro.bin.purge import main as purge_main
from nro.engine.cli import page_text
from nro.configuration.store import ConfigStore


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _registry_with_two_runs(tmp_path: Path):
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    for run in ("1", "10"):
        stem = f"sub-01_task-rest_run-{run}_bold"
        _write(subject / "func" / f"{stem}.nii.gz")
        _write(subject / "func" / f"{stem}.json", "{}")
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="clean",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    request = registry.create_request(
        registered=registered,
        target_module="clean",
        selectors={},
        instances=instances,
        terminal_instance_keys=[instance.key for instance in instances if instance.module == "clean"],
        concurrency=2,
        partition=None,
    )
    return bids, registry, instances, request


def _instance_row(registry: Registry, module: str, run: str | None = None) -> dict:
    for row in registry.instance_rows():
        entities = json.loads(row["entities_json"])
        if row["module"] == module and (run is None or entities.get("run") == run):
            return row
    raise AssertionError(f"Missing instance {module=} {run=}")


def test_targeted_purge_removes_only_directly_selected_instance(tmp_path: Path, capsys) -> None:
    bids, registry, _instances, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _instance_row(registry, "anat")
    func1 = _instance_row(registry, "func", "1")
    func10 = _instance_row(registry, "func", "10")
    clean1 = _instance_row(registry, "clean", "1")

    anat_file = _write(Path(anat["output_root"]) / "sub-01_desc-test_T1w.nii.gz")
    func1_file = _write(
        Path(func1["output_root"]) / "func" / f"{func1['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    func10_file = _write(
        Path(func10["output_root"]) / "func" / f"{func10['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    clean1_file = _write(
        Path(clean1["output_root"]) / "func" / f"{clean1['output_prefix']}_space-T1w_desc-clean_bold.nii.gz"
    )
    func1_work = _write(
        work
        / "demo" / "derivatives" / "preprocessing" / "main" / "sub-01" / "func"
        / f"{func1['output_prefix']}_bold" / "scratch.txt"
    ).parent
    func10_work = _write(
        work
        / "demo" / "derivatives" / "preprocessing" / "main" / "sub-01" / "func"
        / f"{func10['output_prefix']}_bold" / "scratch.txt"
    ).parent
    completion = _write(Path(func1["manifest_path"]), "{}")

    purge_main(
        [
            "-p", "01", "-P", "demo", "-m", "func", "-r", "run=1",
            "--bids-root", str(bids), "--work-root", str(work), "-f", "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "all"
    assert result["instances"] == 1
    assert not func1_file.exists()
    assert not func1_work.exists()
    assert not completion.exists()
    assert func10_file.exists()
    assert func10_work.exists()
    assert anat_file.exists()
    assert clean1_file.exists()  # downstream files are deliberately untouched
    assert _instance_row(registry, "func", "1")["artifact_state"] == "missing"


def test_func_purge_cannot_remove_anat_for_minimal_run_prefix(
    tmp_path: Path, capsys
) -> None:
    """A minimally named sub-01_bold run must not own every sub-01_* file."""
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_bold.nii.gz")
    _write(subject / "func" / "sub-01_bold.json", "{}")
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
    registry.register_instances(instances)
    anat = next(instance for instance in instances if instance.module == "anat")
    func = next(instance for instance in instances if instance.module == "func")
    anat_file = _write(Path(anat.output_root) / "sub-01_desc-preproc_T1w.nii.gz")
    freesurfer_file = _write(
        Path(anat.output_root).parent.parent / "code" / "freesurfer" / "sub-01"
        / "scripts" / "recon-all.done"
    )
    func_file = _write(
        Path(func.output_root) / "func" / "sub-01_space-T1w_desc-preproc_bold.nii.gz"
    )

    purge_main(
        [
            "-p", "01", "-P", "demo", "-m", "func", "--bids-root", str(bids),
            "--work-root", str(tmp_path / "work"), "-f", "--json",
        ]
    )
    json.loads(capsys.readouterr().out)

    assert not func_file.exists()
    assert anat_file.exists()
    assert freesurfer_file.exists()


def test_logs_only_purge_removes_matching_and_inactive_worker_logs(
    tmp_path: Path, capsys
) -> None:
    bids, registry, _instances, request = _registry_with_two_runs(tmp_path)
    terminal_instance = _instance_row(registry, "func", "1")
    active_instance = _instance_row(registry, "func", "10")
    derivative = _write(
        Path(terminal_instance["output_root"]) / "func"
        / f"{terminal_instance['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    terminal_log = _write(registry.paths.events / "terminal" / "attempt-1.log")
    active_log = _write(registry.paths.events / "active" / "attempt-2.log")
    terminal_worker_log = _write(registry.paths.workers / "slurm-101.log")
    active_worker_log = _write(registry.paths.workers / "slurm-202.log")
    now = utcnow()
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO attempts(instance_id, state, revision_fingerprint, memory_gb,
               started_at, completed_at, log_path, created_at)
               VALUES (?, 'success', ?, 32, ?, ?, ?, ?)""",
            (terminal_instance["id"], terminal_instance["revision_fingerprint"], now, now, str(terminal_log), now),
        )
        db.execute(
            """INSERT INTO attempts(instance_id, state, revision_fingerprint, memory_gb,
               started_at, log_path, created_at)
               VALUES (?, 'running', ?, 32, ?, ?, ?)""",
            (active_instance["id"], active_instance["revision_fingerprint"], now, str(active_log), now),
        )
        db.execute(
            """INSERT INTO scheduler_submissions(
               intent_token, request_id, resource_class, memory_gb, state, slurm_job_id, created_at)
               VALUES ('terminal', ?, 'large', 32, 'complete', '101', ?)""",
            (request, now),
        )
        db.execute(
            """INSERT INTO scheduler_submissions(
               intent_token, request_id, resource_class, memory_gb, state, slurm_job_id, created_at)
               VALUES ('active', ?, 'large', 32, 'running', '202', ?)""",
            (request, now),
        )

    purge_main(
        [
            "--logs", "--force", "--bids-root", str(bids),
            "--work-root", str(tmp_path / "work"), "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "logs"
    assert result["attempt_logs"] == 1
    assert result["worker_logs"] == 1
    assert not terminal_log.exists()
    assert active_log.exists()
    assert not terminal_worker_log.exists()
    assert active_worker_log.exists()
    assert derivative.exists()


def test_targeted_purge_refuses_active_attempt(tmp_path: Path) -> None:
    bids, registry, _instances, _request = _registry_with_two_runs(tmp_path)
    instance = _instance_row(registry, "func", "1")
    derivative = _write(
        Path(instance["output_root"]) / "func"
        / f"{instance['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    now = utcnow()
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO attempts(instance_id, state, revision_fingerprint, memory_gb,
               started_at, log_path, created_at)
               VALUES (?, 'running', ?, 32, ?, ?, ?)""",
            (instance["id"], instance["revision_fingerprint"], now, str(tmp_path / "active.log"), now),
        )

    with pytest.raises(SystemExit, match="Refusing to purge active derivative"):
        purge_main(
            [
                "-p", "01", "-P", "demo", "-m", "func", "-r", "run=1",
                "--bids-root", str(bids), "--work-root", str(tmp_path / "work"),
            ]
        )
    assert derivative.exists()


def test_purge_accepts_multiple_direct_job_types(tmp_path: Path, capsys) -> None:
    bids, registry, _instances, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _instance_row(registry, "anat")
    func = _instance_row(registry, "func", "1")
    clean = _instance_row(registry, "clean", "1")
    anat_file = _write(Path(anat["output_root"]) / "sub-01_desc-test_T1w.nii.gz")
    func_file = _write(
        Path(func["output_root"]) / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    clean_file = _write(
        Path(clean["output_root"]) / "func"
        / f"{clean['output_prefix']}_space-T1w_desc-clean_bold.nii.gz"
    )

    purge_main(
        [
            "-p", "01", "-P", "demo", "-m", "func", "clean", "-r", "run=1",
            "--bids-root", str(bids), "--work-root", str(work), "-f", "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["instances"] == 2
    assert not func_file.exists()
    assert not clean_file.exists()
    assert anat_file.exists()


def test_purge_removes_one_space_smoothing_target_from_shared_subject_directory(
    tmp_path: Path, capsys
) -> None:
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
        module="microparcellation",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
        spaces=("fsnative", "T1w"),
        smoothing_levels=(0, 2),
    )
    registry.register_instances(instances)
    micro = [row for row in registry.instance_rows() if row["module"] == "microparcellation"]
    selected = next(
        row for row in micro
        if json.loads(row["entities_json"]) == {"space": "fsnative", "smoothing": "0"}
    )
    preserved = next(
        row for row in micro
        if json.loads(row["entities_json"]) == {"space": "fsnative", "smoothing": "2"}
    )
    selected_file = _write(
        Path(selected["output_root"]) / f"{selected['output_prefix']}_manifest.yaml"
    )
    preserved_file = _write(
        Path(preserved["output_root"]) / f"{preserved['output_prefix']}_manifest.yaml"
    )
    work = tmp_path / "work"
    selected_work = _write(
        work / "demo/derivatives/microparcellation/main/sub-01"
        / "space-fsnative_smoothing-0mm/scratch.txt"
    ).parent
    preserved_work = _write(
        work / "demo/derivatives/microparcellation/main/sub-01"
        / "space-fsnative_smoothing-2mm/scratch.txt"
    ).parent

    purge_main(
        [
            "-p", "01", "-P", "demo", "-m", "microparcellation",
            "-s", "fsnative", "-S", "0", "--bids-root", str(bids),
            "--work-root", str(work), "-f", "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["instances"] == 1
    assert not selected_file.exists()
    assert not selected_work.exists()
    assert preserved_file.exists()
    assert preserved_work.exists()


def test_bare_purge_removes_all_registered_derivatives_but_not_foreign_ones(
    tmp_path: Path, capsys
) -> None:
    bids, registry, _instances, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _instance_row(registry, "anat")
    func = _instance_row(registry, "func", "1")
    anat_file = _write(Path(anat["output_root"]) / "sub-01_desc-test_T1w.nii.gz")
    func_file = _write(
        Path(func["output_root"]) / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    func_work = _write(
        work / "demo/derivatives/preprocessing/main/sub-01/func"
        / f"{func['output_prefix']}_bold/scratch.txt"
    )
    foreign = _write(
        bids / "demo/derivatives/other-pipeline/sub-01/foreign_result.nii.gz"
    )
    attempt_log = _write(registry.paths.events / "complete" / "instance.log")
    now = utcnow()
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO attempts(instance_id, state, revision_fingerprint, memory_gb,
               started_at, completed_at, log_path, created_at)
               VALUES (?, 'success', ?, 32, ?, ?, ?, ?)""",
            (
                func["id"], func["revision_fingerprint"], now, now,
                str(attempt_log), now,
            ),
        )

    purge_main(
        [
            "--force", "--bids-root", str(bids), "--work-root", str(work),
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "all"
    assert result["instances"] == 5
    assert not anat_file.exists()
    assert not func_file.exists()
    assert not func_work.exists()
    assert not attempt_log.exists()
    assert foreign.exists()
    assert result["attempt_logs"] == 1


def test_purge_reports_plan_and_requires_confirmation(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    bids, registry, _instances, _request = _registry_with_two_runs(tmp_path)
    func = _instance_row(registry, "func", "1")
    derivative = _write(
        Path(func["output_root"]) / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    purge_main(
        [
            "-P", "demo", "-m", "func", "-r", "run=1",
            "--bids-root", str(bids), "--work-root", str(tmp_path / "work"),
        ]
    )
    output = capsys.readouterr().out

    assert "Planned purge" in output
    assert "Public derivative paths" in output
    assert str(derivative) in output
    assert "Purge cancelled." in output
    assert derivative.exists()


def test_purge_plan_uses_pager_for_interactive_output(monkeypatch) -> None:
    calls = []

    class Interactive:
        def isatty(self):
            return True

        def write(self, _value):
            raise AssertionError("interactive purge plans should use the pager")

    monkeypatch.setattr("nro.engine.cli.sys.stdout", Interactive())
    monkeypatch.setattr("nro.engine.cli.shutil.which", lambda _: "/usr/bin/less")
    monkeypatch.setattr(
        "nro.engine.cli.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    page_text("planned paths\n")

    assert calls == [
        (["/usr/bin/less", "-R"], {"input": "planned paths\n", "text": True, "check": False})
    ]
