import logging
import json
import subprocess
from itertools import count
from pathlib import Path

import pytest

from nro.orchestration.manifests import file_record
from nro.orchestration.runner_graph import RunnerGraph, Step, artifact_decision
from nro.orchestration.runner import ContainerSpec, Runner
from nro.engine.execution import collect_bind_directories


def test_bind_collection_uses_existing_ancestors_for_future_outputs(
    tmp_path: Path,
) -> None:
    public_session = tmp_path / "public" / "sub-01" / "ses-01"
    work_session = tmp_path / "work" / "sub-01" / "ses-01"
    input_directory = tmp_path / "source" / "sub-01" / "func"
    public_session.mkdir(parents=True)
    work_session.mkdir(parents=True)
    input_directory.mkdir(parents=True)
    source = input_directory / "sub-01_task-rest_bold.nii.gz"
    source.write_text("bold")

    binds = collect_bind_directories(
        (
            source,
            public_session / "func",
            work_session / "func" / "sub-01_task-rest_bold",
        )
    )

    assert binds == sorted(
        (str(public_session), str(input_directory), str(work_session))
    )


def test_bind_collection_rejects_relative_paths_and_host_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        collect_bind_directories((Path("relative/output.nii.gz"),))

    with pytest.raises(ValueError, match="filesystem root"):
        collect_bind_directories((Path("/definitely-not-an-existing-nro-path/output"),))


def test_bind_collection_preserves_host_symlink_spelling(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects"
    subjects.mkdir()
    (subjects / "fsaverage").symlink_to("/container-only/fsaverage")

    binds = collect_bind_directories((subjects / "fsaverage",))

    assert binds == [str(subjects)]


def test_generated_bind_uses_short_alias_beneath_identity_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "container.sif"
    image.touch()
    shared_root = tmp_path / "shared"
    generated = shared_root / "project" / "func"
    generated.mkdir(parents=True)
    home = shared_root / "work" / "container-home"
    monkeypatch.setattr(
        "nro.orchestration.runner.shutil.which", lambda _command: "/usr/bin/singularity"
    )
    runner = Runner(
        module_name="Container Binds",
        container=ContainerSpec(
            image=image,
            extra_binds=(f"{shared_root}:{shared_root}",),
            home_dir=home,
        ),
        binds=(str(generated),),
        logger=logging.getLogger("test.runner.binds"),
        next_step=count(1).__next__,
    )

    prefix = runner._container_prefix()

    assert home.is_dir()
    assert prefix.count("-B") == 2
    assert f"{generated}:/n0" in prefix
    assert f"{shared_root}:{shared_root}" in prefix
    assert f"{home.resolve()}:/nh" in prefix

    source_image = generated / "sub-01_task-rest_bold.nii.gz"
    inner = runner._inner_cmd(
        ["tool", f"{source_image}[0]", f"--output={generated / 'result.nii.gz'}"],
        {"SUBJECTS_DIR": str(generated / "freesurfer")},
    )

    assert f"/n0/{source_image.name}[0]" in inner
    assert "--output=/n0/result.nii.gz" in inner
    assert "SUBJECTS_DIR=/n0/freesurfer" in inner
    assert str(generated) not in inner


def test_container_cwd_uses_short_mounted_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "container.sif"
    image.touch()
    work = tmp_path / "deep" / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(
        "nro.orchestration.runner.shutil.which", lambda _command: "/usr/bin/singularity"
    )
    runner = Runner(
        module_name="Container Working Directory",
        container=ContainerSpec(image=image),
        binds=(str(work),),
        logger=logging.getLogger("test.runner.cwd"),
        next_step=count(1).__next__,
    )

    prefix = runner._container_prefix_for_cwd(work)

    assert prefix[prefix.index("--pwd") + 1] == "/n0"


def test_container_aliases_are_restored_in_captured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "container.sif"
    image.touch()
    source = tmp_path / "subjects"
    source.mkdir()
    monkeypatch.setattr(
        "nro.orchestration.runner.shutil.which", lambda _command: "/usr/bin/singularity"
    )
    runner = Runner(
        module_name="Container Output Translation",
        container=ContainerSpec(image=image),
        binds=(str(source),),
        logger=logging.getLogger("test.runner.output-translation"),
        next_step=count(1).__next__,
    )

    assert runner._restore_host_paths("__PATH__=/n0/fsaverage") == (
        f"__PATH__={source}/fsaverage"
    )


def test_resumable_step_requires_output_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must declare at least one output file"):
        artifact_decision([], False)

    directory = tmp_path / "unpredictable-output"
    directory.mkdir()
    with pytest.raises(ValueError, match="use a completion breadcrumb"):
        artifact_decision([directory], False)


def test_orchestration_rebuild_signal_does_not_override_timestamp_resume(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("source")
    output.write_text("output")
    monkeypatch.setenv("NRO_TASK_REBUILD", "1")

    should_run, reason = artifact_decision([output], False, inputs=[source])

    assert not should_run
    assert reason == "Output(s) exist and are up to date."


def test_fresh_public_boundary_does_not_require_private_work(tmp_path: Path) -> None:
    private = tmp_path / "work" / "intermediate.txt"
    public = tmp_path / "derivatives" / "result.txt"
    manifest = tmp_path / "derivatives" / "manifest.json"
    actions: list[str] = []

    def construct() -> Runner:
        runner = Runner(
            module_name="Boundary Module",
            container=None,
            binds=(),
            logger=logging.getLogger("test.runner.public-boundary"),
            next_step=count(1).__next__,
        )

        def write_private() -> None:
            actions.append("private")
            private.parent.mkdir(parents=True, exist_ok=True)
            private.write_text("intermediate")

        def publish() -> None:
            actions.append("publish")
            public.parent.mkdir(parents=True, exist_ok=True)
            public.write_text(private.read_text())
            manifest.write_text("complete")

        runner.add_step(Step.python(
            name="Build Private Intermediate",
            outputs=(private,),
            action=write_private,
        ))
        runner.add_step(Step.python(
            name="Publish Derivative",
            inputs=(private,),
            outputs=(public, manifest),
            action=publish,
            validate=lambda: (
                public.is_file() and public.read_text() == "intermediate",
                "Public derivative is invalid.",
            ),
            completion_boundary=True,
        ))
        return runner

    runner = construct()
    with runner.run_context():
        runner.execute()
    assert actions == ["private", "publish"]

    private.unlink()
    actions.clear()
    runner = construct()
    with runner.run_context():
        states = runner.execute()

    assert actions == []
    assert all(state.value == "fresh" for state in states.values())


def test_melodic_command_inference_never_tracks_only_a_directory(tmp_path: Path) -> None:
    output = tmp_path / "melodic.ica"
    inferred = Runner._guess_outputs(["melodic", f"--outdir={output}"])
    assert inferred == [
        str(output / "melodic_IC.nii.gz"),
        str(output / "melodic_mix"),
        str(output / "melodic_FTmix"),
    ]


def test_completion_manifest_rejects_directory_artifacts(tmp_path: Path) -> None:
    directory = tmp_path / "compound-output"
    directory.mkdir()
    with pytest.raises(ValueError, match="must expose a final completion breadcrumb"):
        file_record(directory)


def test_directory_artifact_clears_stale_contents_and_writes_breadcrumb_last(
    tmp_path: Path,
) -> None:
    runner = Runner(
        module_name="Directory Artifact Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.directory-artifact"),
        next_step=count(1).__next__,
    )
    directory = tmp_path / "opaque"
    directory.mkdir()
    old = directory / "old-result.txt"
    old.write_text("old")
    breadcrumb = directory / ".nro_complete"
    breadcrumb.write_text("old completion")
    observed: list[bool] = []

    def produce() -> None:
        observed.append(not directory.exists())
        directory.mkdir(parents=True)
        (directory / "new-result.txt").write_text("new")
        observed.append(not breadcrumb.exists())

    def validate() -> tuple[bool, str]:
        valid = (directory / "new-result.txt").is_file()
        return valid, "new opaque result is present" if valid else "missing opaque result"

    runner.add_step(Step.directory_step(
        name="Generate Opaque Directory",
        directory=directory,
        breadcrumb=breadcrumb,
        force=True,
        action=produce,
        validate=validate,
    ))
    with runner.run_context():
        runner.execute()

    assert observed == [True, True]
    assert not old.exists()
    assert breadcrumb.read_text() == "complete\n"


def test_directory_artifact_failure_leaves_no_completion_breadcrumb(
    tmp_path: Path,
) -> None:
    runner = Runner(
        module_name="Directory Artifact Failure Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.directory-artifact-failure"),
        next_step=count(1).__next__,
    )
    directory = tmp_path / "opaque"
    breadcrumb = directory / ".nro_complete"

    def produce_incomplete_directory() -> None:
        directory.mkdir(parents=True)

    runner.add_step(Step.directory_step(
        name="Generate Incomplete Directory",
        directory=directory,
        breadcrumb=breadcrumb,
        action=produce_incomplete_directory,
        validate=lambda: (False, "incomplete opaque output"),
    ))
    with pytest.raises(RuntimeError, match="incomplete opaque output"):
        with runner.run_context():
            runner.execute()

    assert directory.is_dir()
    assert not breadcrumb.exists()


def test_directory_timeout_cannot_publish_or_reuse_partial_results(tmp_path: Path) -> None:
    directory = tmp_path / "oslom"
    result = directory / "tp"
    breadcrumb = directory / ".nro_complete"
    for timeout in (True, False):
        runner = Runner(
            module_name="Timeout Test", container=None, binds=(),
            logger=logging.getLogger("test.runner.timeout"),
            next_step=count(1).__next__,
        )

        def produce() -> None:
            assert not directory.exists()
            directory.mkdir()
            result.write_text("partial" if timeout else "finished")
            if timeout:
                raise subprocess.TimeoutExpired("oslom", 1)

        runner.add_step(Step.directory_step(
            name="Fit", directory=directory, breadcrumb=breadcrumb,
            outputs=(result,), action=produce,
            validate=lambda: (result.is_file(), "result exists"),
        ))
        if timeout:
            with pytest.raises(subprocess.TimeoutExpired), runner.run_context():
                runner.execute()
            assert result.exists()
            assert not breadcrumb.exists()
        else:
            with runner.run_context():
                runner.execute()
            assert breadcrumb.exists()
            assert result.read_text() == "finished"


def test_standard_runner_success_report_uses_runner_start(monkeypatch, caplog) -> None:
    runner = Runner(
        module_name="Test Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.orchestration.runner"),
        next_step=count(1).__next__,
        step_log_separator="=" * 12,
    )
    monkeypatch.setattr("nro.orchestration.runner.time.perf_counter", lambda: 105.25)

    with caplog.at_level(logging.INFO, logger="test.orchestration.runner"):
        runner.log_runner_success(
            module_name="Cleaning Module",
            started_at=5.0,
        )

    assert "============" in caplog.text
    assert "Name: Cleaning Module" in caplog.text
    assert "Status: Success" in caplog.text
    assert "Total Time Elapsed: 100.250s" in caplog.text


def test_runner_failure_reports_active_numbered_step(caplog) -> None:
    runner = Runner(
        module_name="Test Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.failure"),
        next_step=count(1).__next__,
    )

    with caplog.at_level(logging.INFO, logger="test.runner.failure"):
        with pytest.raises(ValueError, match="invalid metadata"):
            with runner.run_context():
                with runner.python_step(step_name="Read Cleaning Metadata"):
                    raise ValueError("invalid metadata")

    assert "Status: Failure" in caplog.text
    assert "Failed Step: 002 — Read Cleaning Metadata" in caplog.text
    assert "Error: ValueError: invalid metadata" in caplog.text
    assert "Total Time Elapsed:" in caplog.text


def test_runner_failure_between_child_steps_uses_execution_step(caplog) -> None:
    runner = Runner(
        module_name="Test Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.boundary"),
        next_step=count(1).__next__,
    )

    with caplog.at_level(logging.INFO, logger="test.runner.boundary"):
        with pytest.raises(RuntimeError, match="orchestration failure"):
            with runner.run_context():
                with runner.python_step(step_name="Completed Child"):
                    pass
                raise RuntimeError("orchestration failure")

    assert "Failed Step: 001 — Test Module Execution" in caplog.text
    assert "Error: RuntimeError: orchestration failure" in caplog.text


def test_runner_rejects_unfinished_numbered_step(caplog) -> None:
    runner = Runner(
        module_name="Test Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.unfinished"),
        next_step=count(1).__next__,
    )

    with caplog.at_level(logging.INFO, logger="test.runner.unfinished"):
        with pytest.raises(RuntimeError, match="unfinished step 002"):
            with runner.run_context():
                runner.log_python_step(
                    step_name="Forgotten Step",
                    running=True,
                )

    assert "Failed Step: 002 — Forgotten Step" in caplog.text
    assert "Status: Failure" in caplog.text


def test_container_command_preflight_runs_configured_inner_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "qunex.sif"
    image.touch()
    calls: list[list[str]] = []

    monkeypatch.setattr("nro.orchestration.runner.shutil.which", lambda _cmd: "/usr/bin/singularity")

    def fake_run(command, **_kwargs):
        calls.append([str(part) for part in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("nro.orchestration.runner.subprocess.run", fake_run)
    runner = Runner(
        module_name="Test Module",
        container=ContainerSpec(
            image=image,
            inner_setup="source /opt/qunex/env/qunex_environment.sh",
        ),
        binds=(),
        logger=logging.getLogger("test.runner.container-preflight"),
        next_step=count(1).__next__,
    )

    runner.require_cmds(["flirt"])

    assert len(calls) == 1
    assert calls[0][-2:] == [
        "-lc",
        "set -e; source /opt/qunex/env/qunex_environment.sh; command -v flirt",
    ]


def test_dependency_preflight_failure_is_attributed_to_numbered_step(
    monkeypatch: pytest.MonkeyPatch,
    caplog,
) -> None:
    runner = Runner(
        module_name="Test Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.dependency-preflight"),
        next_step=count(1).__next__,
    )

    monkeypatch.setattr(
        "nro.orchestration.runner.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", ""),
    )

    with caplog.at_level(logging.INFO, logger="test.runner.dependency-preflight"):
        with pytest.raises(SystemExit, match="Missing required commands on PATH"):
            with runner.run_context():
                runner.require_cmds(["definitely-not-installed"])

    assert "Failed Step: 002 — Dependency Preflight" in caplog.text
    assert "Error: SystemExit: Missing required commands on PATH" in caplog.text


def test_structured_step_ledger_records_outputs(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "current-steps.json"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    runner = Runner(
        module_name="Ledger Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.ledger"),
        next_step=count(1).__next__,
    )
    output = tmp_path / "output.txt"

    runner.add_step(Step.python(
        name="Generate Output",
        outputs=(output,),
        action=lambda: output.write_text("done"),
    ))
    with runner.run_context():
        runner.execute()

    current = json.loads(ledger.read_text())
    generated = next(value for value in current.values() if value["name"] == "Generate Output")
    assert generated["status"] == "success"
    assert generated["outputs"] == [str(output)]
    assert (tmp_path / "step-events.jsonl").is_file()
    graph = json.loads((tmp_path / "runner-graph.json").read_text())
    generated_node = next(
        node for node in graph["nodes"] if node["name"] == "Generate Output"
    )
    assert generated_node["outputs"] == [str(output)]
    assert generated_node["execution"] == "success"


def test_python_artifact_validator_can_reopen_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("invalid")
    runner = Runner(
        module_name="Validator Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.validator"),
        next_step=count(1).__next__,
    )

    step = runner.add_step(Step.python(
        name="Validated Output",
        outputs=(output,),
        action=lambda: output.write_text("valid"),
        validate=lambda: (output.read_text() == "valid", "Semantic validation failed."),
    ))
    with runner.run_context():
        states = runner.execute()

    assert states[step.id].value == "dirty"
    assert output.read_text() == "valid"


def test_step_ledger_records_exact_planned_artifact_path(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "current-steps.json"
    output = tmp_path / "private" / "checkpoint.txt"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    runner = Runner(
        module_name="Canonical Output Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.canonical-output"),
        next_step=count(1).__next__,
    )

    runner.add_step(Step.python(
        name="Write Checkpoint",
        outputs=(output,),
        action=lambda: (output.parent.mkdir(parents=True), output.write_text("done")),
    ))
    with runner.run_context():
        runner.execute()

    current = json.loads(ledger.read_text())
    checkpoint = next(
        value for value in current.values() if value["name"] == "Write Checkpoint"
    )
    assert checkpoint["outputs"] == [str(output)]


def test_runner_preserves_exact_declared_output_path(tmp_path: Path) -> None:
    output = tmp_path / "private" / "checkpoint.txt"
    runner = Runner(
        module_name="Exact Output Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.reject-basename"),
        next_step=count(1).__next__,
    )
    step = runner.add_step(Step.python(
        name="Checkpoint",
        outputs=(output,),
        action=lambda: (output.parent.mkdir(parents=True), output.write_text("done")),
    ))
    assert step.outputs == (output,)


def test_graph_execution_is_forbidden_outside_runner_context(
    tmp_path: Path,
) -> None:
    runner = Runner(
        module_name="Context required",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.context-required"),
        next_step=count(1).__next__,
    )
    runner.add_step(Step.python(
        name="Output",
        outputs=(tmp_path / "output",),
        action=lambda: None,
    ))
    with pytest.raises(RuntimeError, match=r"active run_context"):
        runner.execute()


def test_compound_step_commands_use_uniform_user_facing_log_labels(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.runner.command-labels")
    runner = Runner(
        module_name="Command Label Module",
        container=None,
        binds=(),
        logger=logger,
        next_step=count(1).__next__,
    )
    output = tmp_path / "complete.txt"

    def execute() -> None:
        captured = runner.run_child(
            ["bash", "-c", "printf result; printf warning >&2"],
            capture_stdout=True,
        )
        output.write_text(str(captured))

    runner.add_step(Step.python(
        name="Compound Step",
        outputs=(output,),
        action=execute,
    ))

    with caplog.at_level(logging.INFO, logger=logger.name):
        with runner.run_context():
            runner.execute()

    assert "Cmd: bash -c" in caplog.text
    assert "Stderr: warning" in caplog.text
    assert "Child Cmd:" not in caplog.text
    assert "Child output:" not in caplog.text
    assert output.read_text() == "result"


def test_module_dag_contract_rejects_topology_change_for_same_signature(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "runner-contract.json"
    first_output = tmp_path / "first.txt"
    first = RunnerGraph("Immutable")
    first.add(
        Step.python(
            name="First", inputs=(tmp_path / "source.txt",),
            outputs=(first_output,), action=lambda: None,
        )
    )
    first.freeze()
    first.reconcile_contract(contract, signature="source-and-workflow")

    changed = RunnerGraph("Immutable")
    changed_output = tmp_path / "changed.txt"
    changed.add(
        Step.python(
            name="Changed", inputs=(tmp_path / "source.txt",),
            outputs=(changed_output,), action=lambda: None,
        )
    )
    changed.freeze()
    with pytest.raises(RuntimeError, match="topology changed"):
        changed.reconcile_contract(contract, signature="source-and-workflow")


def test_bound_module_contract_rejects_mutation_before_execution(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "runner-contract.json"
    first_output = tmp_path / "first.txt"
    graph = RunnerGraph("Immutable")
    graph.add(Step.python(name="First", outputs=(first_output,), action=lambda: None))
    graph.freeze()
    graph.reconcile_contract(contract, signature="source-and-workflow")

    changed = RunnerGraph("Immutable")
    changed.add(
        Step.python(name="Changed", outputs=(tmp_path / "changed.txt",), action=lambda: None)
    )
    changed.freeze()
    with pytest.raises(RuntimeError, match="topology changed"):
        changed.bind_contract(contract, signature="source-and-workflow")


def test_outputless_numbered_operations_are_not_dag_nodes(tmp_path: Path) -> None:
    graph = RunnerGraph("Operations").freeze()
    graph.record_operation(step=1, name="Boundary", outputs=(), status="success")
    payload = graph.payload()
    assert payload["nodes"] == []
    assert payload["operations"][0]["name"] == "Boundary"


def test_command_artifact_validator_reopens_invalid_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("invalid")
    runner = Runner(
        module_name="Validated Command Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.command-validator"),
        next_step=count(1).__next__,
    )

    def fake_run(command, **kwargs):
        Path(command[-1]).write_text("valid")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("nro.orchestration.runner.subprocess.run", fake_run)
    step = runner.add_step(Step.command_step(
        ["write-output", str(output)],
        outputs=(output,),
        validate=lambda: (
            output.read_text() == "valid",
            "Semantic command validation failed.",
        ),
    ))
    with runner.run_context():
        states = runner.execute()

    assert states[step.id].value == "dirty"
    assert output.read_text() == "valid"


def test_successful_command_without_declared_output_fails_its_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    output = tmp_path / "missing.txt"
    runner = Runner(
        module_name="Missing Output Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.missing-command-output"),
        next_step=count(1).__next__,
    )
    monkeypatch.setattr(
        "nro.orchestration.runner.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    runner.add_step(Step.command_step(["successful-but-empty"], outputs=(output,)))
    with caplog.at_level(logging.INFO, logger="test.runner.missing-command-output"):
        with pytest.raises(RuntimeError, match="without producing nonempty file"):
            with runner.run_context():
                runner.execute()

    assert "Failed Step: 002 — Successful-But-Empty" in caplog.text
