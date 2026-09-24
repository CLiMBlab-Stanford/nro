from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.bin.purge import main as purge_main
from nro.configuration.store import ConfigStore
from nro.engine.cli import page_text
from nro.orchestration.branch_purge import (
    _contains_protected_output,
    _protected_output_index,
)
from nro.orchestration.planner import build_subject_work_items
from nro.orchestration.purge_paths import _remove_path
from nro.orchestration.registry import Registry, utcnow


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
    work_items = build_subject_work_items(
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
        work_items=work_items,
        terminal_work_item_keys=[
            work_item.key for work_item in work_items if work_item.module == "clean"
        ],
        concurrency=2,
        partition=None,
    )
    return bids, registry, work_items, request


def _work_item_row(registry: Registry, module: str, run: str | None = None) -> dict:
    for row in registry.work_item_rows():
        entities = json.loads(row["entities_json"])
        if row["module"] == module and (run is None or entities.get("run") == run):
            return row
    raise AssertionError(f"Missing work_item {module=} {run=}")


def test_remove_path_prunes_empty_parents_but_preserves_boundary(tmp_path: Path) -> None:
    derivatives = tmp_path / "derivatives"
    artifact = _write(derivatives / "module" / "main" / "sub-01" / "artifact.nii.gz")

    assert _remove_path(artifact, dry_run=False, prune_root=derivatives)

    assert derivatives.is_dir()
    assert not (derivatives / "module").exists()


def test_remove_path_stops_pruning_at_nonempty_directory(tmp_path: Path) -> None:
    derivatives = tmp_path / "derivatives"
    artifact = _write(derivatives / "module" / "main" / "sub-01" / "artifact.nii.gz")
    retained = _write(artifact.parent / "retained.nii.gz")

    assert _remove_path(artifact, dry_run=False, prune_root=derivatives)

    assert artifact.parent.is_dir()
    assert retained.is_file()


def test_remove_path_unlinks_symlink_without_removing_external_target(tmp_path: Path) -> None:
    derivatives = tmp_path / "derivatives"
    target = _write(tmp_path / "external" / "target.txt")
    link = derivatives / "module" / "main" / "target-link"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    assert _remove_path(link, dry_run=False, prune_root=derivatives)

    assert not link.is_symlink()
    assert target.read_text() == "x"
    assert derivatives.is_dir()


def test_protected_output_index_detects_only_equal_or_descendant_paths(tmp_path: Path) -> None:
    protected = tmp_path / "derivatives" / "module" / "sub-01" / "result.nii.gz"
    sibling = tmp_path / "derivatives" / "module-other"
    index = _protected_output_index((protected,))

    assert _contains_protected_output(protected, index)
    assert _contains_protected_output(protected.parent, index)
    assert not _contains_protected_output(sibling, index)
    assert not _contains_protected_output(protected.parent / "other.nii.gz", index)


def test_targeted_purge_removes_only_directly_selected_work_item(tmp_path: Path, capsys) -> None:
    bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _work_item_row(registry, "anat")
    func1 = _work_item_row(registry, "func", "1")
    func10 = _work_item_row(registry, "func", "10")
    clean1 = _work_item_row(registry, "clean", "1")

    anat_file = _write(Path(anat["output_root"]) / "sub-01_desc-test_T1w.nii.gz")
    func1_file = _write(
        Path(func1["output_root"])
        / "func"
        / f"{func1['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    func10_file = _write(
        Path(func10["output_root"])
        / "func"
        / f"{func10['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    clean1_file = _write(
        Path(clean1["output_root"])
        / "func"
        / f"{clean1['output_prefix']}_space-T1w_desc-clean_bold.nii.gz"
    )
    func1_work = _write(
        work
        / "demo"
        / "derivatives"
        / "nro"
        / "func"
        / func1["directory_label"]
        / "sub-01"
        / "func"
        / f"{func1['output_prefix']}_bold"
        / "scratch.txt"
    ).parent
    func10_work = _write(
        work
        / "demo"
        / "derivatives"
        / "nro"
        / "func"
        / func10["directory_label"]
        / "sub-01"
        / "func"
        / f"{func10['output_prefix']}_bold"
        / "scratch.txt"
    ).parent

    purge_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "func",
            "-r",
            "run=1",
            "--work-root",
            str(work),
            "-f",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "all"
    assert result["work_items"] == 1
    assert not func1_file.exists()
    assert not func1_work.exists()
    assert func10_file.exists()
    assert func10_work.exists()
    assert anat_file.exists()
    assert clean1_file.exists()  # downstream files are deliberately untouched
    assert _work_item_row(registry, "func", "1")["artifact_state"] == "missing"


def test_purge_record_cleanup_preserves_only_required_ancestors(tmp_path: Path) -> None:
    _bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    func1 = _work_item_row(registry, "func", "1")
    func10 = _work_item_row(registry, "func", "10")
    clean1 = _work_item_row(registry, "clean", "1")
    clean10 = _work_item_row(registry, "clean", "10")

    cancelled = registry.cancel_purged_demand((clean1["id"],))
    removed, retained = registry.forget_purged_work_items((clean1["id"],))

    assert cancelled > 0
    assert removed == 2
    assert retained == ()
    assert clean1["id"] not in {row["id"] for row in registry.work_item_rows()}
    assert func1["id"] not in {row["id"] for row in registry.work_item_rows()}
    assert clean10["id"] in {row["id"] for row in registry.work_item_rows()}

    removed, retained = registry.forget_purged_work_items((func10["id"],))

    assert removed == 0
    assert retained == (func10["id"],)
    assert registry.retained_dependency_modules(retained) == {"clean": 1}


def test_dependency_tombstone_is_collected_with_its_final_dependent(tmp_path: Path) -> None:
    _bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    func1 = _work_item_row(registry, "func", "1")
    clean1 = _work_item_row(registry, "clean", "1")

    registry.cancel_purged_demand((func1["id"],))
    removed, retained = registry.forget_purged_work_items((func1["id"],))

    assert removed == 0
    assert retained == (func1["id"],)

    registry.cancel_purged_demand((clean1["id"],))
    removed, retained = registry.forget_purged_work_items((clean1["id"],))

    assert removed == 2
    assert retained == ()
    remaining = {row["id"] for row in registry.work_item_rows()}
    assert func1["id"] not in remaining
    assert clean1["id"] not in remaining


def test_func_purge_cannot_remove_anat_for_minimal_run_prefix(tmp_path: Path, capsys) -> None:
    """A minimally named sub-01_bold run must not own every sub-01_* file."""
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_bold.nii.gz")
    _write(subject / "func" / "sub-01_bold.json", "{}")
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
    registry.register_work_items(work_items)
    anat = next(work_item for work_item in work_items if work_item.module == "anat")
    func = next(work_item for work_item in work_items if work_item.module == "func")
    anat_file = _write(Path(anat.output_root) / "sub-01_desc-preproc_T1w.nii.gz")
    freesurfer_file = _write(
        Path(anat.output_root).parent.parent
        / "code"
        / "freesurfer"
        / "sub-01"
        / "scripts"
        / "recon-all.done"
    )
    func_file = _write(
        Path(func.output_root) / "func" / "sub-01_space-T1w_desc-preproc_bold.nii.gz"
    )

    purge_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "func",
            "--work-root",
            str(tmp_path / "work"),
            "-f",
            "--json",
        ]
    )
    json.loads(capsys.readouterr().out)

    assert not func_file.exists()
    assert anat_file.exists()
    assert freesurfer_file.exists()


def test_anat_purge_removes_partial_subject_outputs(tmp_path: Path, capsys) -> None:
    """Anat owns session outputs and WORK created before final publication."""
    bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _work_item_row(registry, "anat")
    subject_root = Path(anat["output_root"]).parent
    partial_public = _write(
        subject_root / "ses-01" / "anat" / "sub-01_ses-01_desc-reference_T1w.nii.gz"
    )
    partial_work = _write(
        work
        / "demo"
        / "derivatives"
        / "nro"
        / "anat"
        / anat["directory_label"]
        / "sub-01"
        / "subject_reference"
        / "sub-01_desc-selected_T1w.nii.gz"
    )

    purge_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "anat",
            "--work-root",
            str(work),
            "-f",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["work_items"] == 1
    assert not partial_public.exists()
    assert not partial_work.exists()
    assert not subject_root.exists()


def test_logs_only_purge_removes_matching_and_inactive_worker_logs(tmp_path: Path, capsys) -> None:
    bids, registry, _work_items, request = _registry_with_two_runs(tmp_path)
    terminal_work_item = _work_item_row(registry, "func", "1")
    active_work_item = _work_item_row(registry, "func", "10")
    derivative = _write(
        Path(terminal_work_item["output_root"])
        / "func"
        / f"{terminal_work_item['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    terminal_log = _write(registry.paths.events / "terminal" / "attempt-1.log")
    active_log = _write(registry.paths.events / "active" / "attempt-2.log")
    terminal_worker_log = _write(registry.paths.workers / "slurm-101.log")
    active_worker_log = _write(registry.paths.workers / "slurm-202.log")
    now = utcnow()
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO attempts(work_item_id, state, revision_fingerprint, memory_gb,
               started_at, completed_at, log_path, created_at)
               VALUES (?, 'success', ?, 32, ?, ?, ?, ?)""",
            (
                terminal_work_item["id"],
                terminal_work_item["revision_fingerprint"],
                now,
                now,
                str(terminal_log),
                now,
            ),
        )
        db.execute(
            """INSERT INTO attempts(work_item_id, state, revision_fingerprint, memory_gb,
               started_at, log_path, created_at)
               VALUES (?, 'running', ?, 32, ?, ?, ?)""",
            (
                active_work_item["id"],
                active_work_item["revision_fingerprint"],
                now,
                str(active_log),
                now,
            ),
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
            "--logs",
            "--force",
            "--work-root",
            str(tmp_path / "work"),
            "--json",
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
    bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    work_item = _work_item_row(registry, "func", "1")
    derivative = _write(
        Path(work_item["output_root"])
        / "func"
        / f"{work_item['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    now = utcnow()
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO attempts(work_item_id, state, revision_fingerprint, memory_gb,
               started_at, log_path, created_at)
               VALUES (?, 'running', ?, 32, ?, ?, ?)""",
            (
                work_item["id"],
                work_item["revision_fingerprint"],
                now,
                str(tmp_path / "active.log"),
                now,
            ),
        )

    with pytest.raises(SystemExit, match="Refusing to purge active derivative"):
        purge_main(
            [
                "-p",
                "01",
                "-P",
                "demo",
                "-m",
                "func",
                "-r",
                "run=1",
                "--work-root",
                str(tmp_path / "work"),
            ]
        )
    assert derivative.exists()


def test_purge_accepts_multiple_direct_job_types(tmp_path: Path, capsys) -> None:
    bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _work_item_row(registry, "anat")
    func = _work_item_row(registry, "func", "1")
    clean = _work_item_row(registry, "clean", "1")
    anat_file = _write(Path(anat["output_root"]) / "sub-01_desc-test_T1w.nii.gz")
    func_file = _write(
        Path(func["output_root"])
        / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    clean_file = _write(
        Path(clean["output_root"])
        / "func"
        / f"{clean['output_prefix']}_space-T1w_desc-clean_bold.nii.gz"
    )

    purge_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "func",
            "clean",
            "-r",
            "run=1",
            "--work-root",
            str(work),
            "-f",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["work_items"] == 2
    assert not func_file.exists()
    assert not clean_file.exists()
    assert anat_file.exists()


def test_purge_removes_one_space_smoothing_subject_artifact(tmp_path: Path, capsys) -> None:
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
        module="microparcellation",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
        spaces=("fsnative", "T1w"),
        smoothing_levels=(0, 2),
    )
    registry.register_work_items(work_items)
    micro = [row for row in registry.work_item_rows() if row["module"] == "microparcellation"]
    selected = next(
        row
        for row in micro
        if json.loads(row["entities_json"]) == {"space": "fsnative", "smoothing": "0"}
    )
    preserved = next(
        row
        for row in micro
        if json.loads(row["entities_json"]) == {"space": "fsnative", "smoothing": "2"}
    )
    assert Path(selected["output_root"]) == Path(preserved["output_root"])
    selected_file = _write(
        Path(selected["output_root"]) / f"{selected['output_prefix']}_manifest.yaml"
    )
    preserved_file = _write(
        Path(preserved["output_root"]) / f"{preserved['output_prefix']}_manifest.yaml"
    )
    work = tmp_path / "work"
    selected_work = _write(
        work
        / "demo/derivatives/nro/microparcellation"
        / selected["directory_label"]
        / "space-fsnative_smoothing-0mm/sub-01/scratch.txt"
    ).parent
    preserved_work = _write(
        work
        / "demo/derivatives/nro/microparcellation"
        / preserved["directory_label"]
        / "space-fsnative_smoothing-2mm/sub-01/scratch.txt"
    ).parent

    purge_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "microparcellation",
            "-s",
            "fsnative",
            "-S",
            "0",
            "--work-root",
            str(work),
            "-f",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["work_items"] == 1
    assert not selected_file.exists()
    assert not selected_work.exists()
    assert preserved_file.exists()
    assert preserved_work.exists()


def test_bare_purge_removes_all_registered_derivatives_but_not_foreign_ones(
    tmp_path: Path, capsys
) -> None:
    bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    work = tmp_path / "work"
    anat = _work_item_row(registry, "anat")
    func = _work_item_row(registry, "func", "1")
    anat_file = _write(Path(anat["output_root"]) / "sub-01_desc-test_T1w.nii.gz")
    func_file = _write(
        Path(func["output_root"])
        / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    func_work = _write(
        work
        / "demo/derivatives/nro/func"
        / func["directory_label"]
        / "sub-01/func"
        / f"{func['output_prefix']}_bold/scratch.txt"
    )
    foreign = _write(bids / "demo/derivatives/other-system/sub-01/foreign_result.nii.gz")
    attempt_log = _write(registry.paths.events / "complete" / "work-item.log")
    now = utcnow()
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO attempts(work_item_id, state, revision_fingerprint, memory_gb,
               started_at, completed_at, log_path, created_at)
               VALUES (?, 'success', ?, 32, ?, ?, ?, ?)""",
            (
                func["id"],
                func["revision_fingerprint"],
                now,
                now,
                str(attempt_log),
                now,
            ),
        )

    purge_main(
        [
            "--force",
            "--work-root",
            str(work),
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "all"
    assert result["work_items"] == 5
    assert not anat_file.exists()
    assert not func_file.exists()
    assert not func_work.exists()
    assert not attempt_log.exists()
    assert foreign.exists()
    assert result["attempt_logs"] == 1


def test_purge_reports_plan_and_requires_confirmation(tmp_path: Path, capsys, monkeypatch) -> None:
    bids, registry, _work_items, _request = _registry_with_two_runs(tmp_path)
    func = _work_item_row(registry, "func", "1")
    derivative = _write(
        Path(func["output_root"])
        / "func"
        / f"{func['output_prefix']}_space-T1w_desc-preproc_bold.nii.gz"
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    purge_main(
        [
            "-P",
            "demo",
            "-m",
            "func",
            "-r",
            "run=1",
            "--work-root",
            str(tmp_path / "work"),
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
