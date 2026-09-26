import io
import json
import logging
import subprocess
from itertools import count
from pathlib import Path

import pytest

from nro.engine.execution import allocated_cpus, collect_bind_directories, thread_environment
from nro.orchestration.artifact_records import file_record
from nro.orchestration.resource_handoff import RESOURCE_HANDOFF_EXIT
from nro.orchestration.runner import ContainerSpec, Runner
from nro.orchestration.runner_graph import NodeState, RunnerGraph, Step, artifact_decision


def test_each_runner_owns_an_independent_step_counter() -> None:
    first = Runner(
        module_name="first",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.first-counter"),
    )
    second = Runner(
        module_name="second",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.second-counter"),
    )

    assert first.log_python_step(step_name="one", running=False) == 1
    assert first.log_python_step(step_name="two", running=False) == 2
    assert second.log_python_step(step_name="one", running=False) == 1


def test_runner_adds_stage_declarations_in_order(tmp_path: Path) -> None:
    runner = Runner(
        module_name="stage",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.stage"),
    )
    first = Step.python(
        name="First",
        outputs=(tmp_path / "first",),
        action=lambda: None,
    )
    second = Step.python(
        name="Second",
        inputs=first.outputs,
        outputs=(tmp_path / "second",),
        action=lambda: None,
    )

    added = runner.add_steps((first, second))

    assert added == runner._graph.steps


def test_runner_yields_dirty_step_to_required_worker_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "gpu.txt"
    request = tmp_path / "handoff.json"
    runner = Runner(
        module_name="Resource Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.resource-handoff"),
    )
    runner.add_step(
        Step.python(
            id="gpu-step",
            name="GPU Step",
            outputs=(output,),
            action=lambda: output.write_text("done"),
            resource_class="gpu",
        )
    )
    monkeypatch.setenv("NRO_WORKER_RESOURCE_CLASS", "large")
    monkeypatch.setenv("NRO_RESOURCE_HANDOFF", str(request))

    with pytest.raises(SystemExit) as error, runner.run_context():
        runner.execute()

    assert error.value.code == RESOURCE_HANDOFF_EXIT
    assert json.loads(request.read_text()) == {
        "kind": "execute_step",
        "reason": "Step GPU Step requires a gpu worker.",
        "resource_class": "gpu",
        "step_id": "gpu-step",
    }
    assert not output.exists()


def test_runner_rejects_unknown_step_resource_class(tmp_path: Path) -> None:
    runner = Runner(
        module_name="Resource Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.resource-class"),
    )

    with pytest.raises(ValueError, match="unsupported resource class"):
        runner.add_step(
            Step.python(
                name="Unknown Resource",
                outputs=(tmp_path / "output.txt",),
                action=lambda: None,
                resource_class="accelerator",
            )
        )


def test_step_resource_class_is_not_part_of_the_scientific_contract(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    cpu = RunnerGraph("Resource Test")
    gpu = RunnerGraph("Resource Test")
    cpu.add(Step.python(id="stage", name="Stage", outputs=(output,), action=lambda: None))
    gpu.add(
        Step.python(
            id="stage",
            name="Stage",
            outputs=(output,),
            action=lambda: None,
            resource_class="gpu",
        )
    )

    assert cpu.freeze().contract_payload(signature="same") == gpu.freeze().contract_payload(
        signature="same"
    )


def test_targeted_resource_worker_runs_only_the_requested_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "upstream.txt"
    target = tmp_path / "target.txt"
    downstream = tmp_path / "downstream.txt"
    upstream.write_text("ready")
    runner = Runner(
        module_name="Target Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.target-step"),
    )
    runner.add_step(
        Step.python(
            id="upstream",
            name="Upstream",
            outputs=(upstream,),
            action=lambda: upstream.write_text("rebuilt"),
        )
    )
    runner.add_step(
        Step.python(
            id="gpu-step",
            name="GPU Step",
            inputs=(upstream,),
            outputs=(target,),
            action=lambda: target.write_text("gpu"),
            force=True,
            resource_class="gpu",
        )
    )
    runner.add_step(
        Step.python(
            id="downstream",
            name="Downstream",
            inputs=(target,),
            outputs=(downstream,),
            action=lambda: downstream.write_text("cpu"),
        )
    )
    monkeypatch.setenv("NRO_WORKER_RESOURCE_CLASS", "gpu")
    monkeypatch.setenv("NRO_TARGET_STEP_ID", "gpu-step")
    monkeypatch.setenv("NRO_RESOURCE_HANDOFF", str(tmp_path / "handoff.json"))
    ledger = tmp_path / "current-steps.json"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    monkeypatch.setenv("NRO_RUNNER_GRAPH_SIGNATURE", "test-signature")
    contract = runner._graph.freeze().contract_payload(signature="test-signature")
    contract["nodes"] = [node for node in contract["nodes"] if node["id"] == "upstream"]
    (tmp_path / "runner-contract.json").write_text(json.dumps(contract), encoding="utf-8")

    with runner.run_context():
        states = runner.execute()

    assert states == {"upstream": NodeState.FRESH, "gpu-step": NodeState.DIRTY}
    assert target.read_text() == "gpu"
    assert not downstream.exists()
    contract = json.loads((tmp_path / "runner-contract.json").read_text())
    assert [node["id"] for node in contract["nodes"]] == ["upstream", "gpu-step"]

    resumed = Runner(
        module_name="Target Test",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.target-resume"),
    )
    resumed.add_step(
        Step.python(
            id="upstream",
            name="Upstream",
            outputs=(upstream,),
            action=lambda: upstream.write_text("rebuilt"),
        )
    )
    resumed.add_step(
        Step.python(
            id="gpu-step",
            name="GPU Step",
            inputs=(upstream,),
            outputs=(target,),
            action=lambda: target.write_text("unexpected rerun"),
            force=True,
            resource_class="gpu",
        )
    )
    resumed.add_step(
        Step.python(
            id="downstream",
            name="Downstream",
            inputs=(target,),
            outputs=(downstream,),
            action=lambda: downstream.write_text("cpu"),
        )
    )
    monkeypatch.delenv("NRO_TARGET_STEP_ID")
    monkeypatch.setenv("NRO_WORKER_RESOURCE_CLASS", "large")
    monkeypatch.setenv("NRO_COMPLETED_RESOURCE_STEPS", '["gpu-step"]')

    with resumed.run_context():
        resumed.execute()

    assert target.read_text() == "gpu"
    assert downstream.read_text() == "cpu"


def test_allocated_cpus_prefers_explicit_worker_allocation(monkeypatch) -> None:
    monkeypatch.setenv("NRO_ALLOCATED_CPUS", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")

    assert allocated_cpus() == 3
    assert set(thread_environment().values()) == {"3"}


def test_allocated_cpus_uses_slurm_allocation(monkeypatch) -> None:
    monkeypatch.delenv("NRO_ALLOCATED_CPUS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")

    assert allocated_cpus() == 5


def test_allocated_cpus_rejects_invalid_worker_allocation(monkeypatch) -> None:
    monkeypatch.setenv("NRO_ALLOCATED_CPUS", "0")

    with pytest.raises(ValueError, match="positive integer"):
        allocated_cpus()


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

    assert binds == sorted((str(public_session), str(input_directory), str(work_session)))


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

    assert runner._restore_host_paths("__PATH__=/n0/fsaverage") == (f"__PATH__={source}/fsaverage")


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

        runner.add_step(
            Step.python(
                name="Build Private Intermediate",
                outputs=(private,),
                action=write_private,
            )
        )
        runner.add_step(
            Step.python(
                name="Publish Derivative",
                inputs=(private,),
                outputs=(public, manifest),
                action=publish,
                validate=lambda: (
                    public.is_file() and public.read_text() == "intermediate",
                    "Public derivative is invalid.",
                ),
                completion_boundary=True,
            )
        )
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
    with pytest.raises(ValueError, match="not a regular file"):
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

    runner.add_step(
        Step.directory_step(
            name="Generate Opaque Directory",
            directory=directory,
            breadcrumb=breadcrumb,
            force=True,
            action=produce,
            validate=validate,
        )
    )
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

    runner.add_step(
        Step.directory_step(
            name="Generate Incomplete Directory",
            directory=directory,
            breadcrumb=breadcrumb,
            action=produce_incomplete_directory,
            validate=lambda: (False, "incomplete opaque output"),
        )
    )
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
            module_name="Timeout Test",
            container=None,
            binds=(),
            logger=logging.getLogger("test.runner.timeout"),
            next_step=count(1).__next__,
        )

        def produce() -> None:
            assert not directory.exists()
            directory.mkdir()
            result.write_text("partial" if timeout else "finished")
            if timeout:
                raise subprocess.TimeoutExpired("oslom", 1)

        runner.add_step(
            Step.directory_step(
                name="Fit",
                directory=directory,
                breadcrumb=breadcrumb,
                outputs=(result,),
                action=produce,
                validate=lambda: (result.is_file(), "result exists"),
            )
        )
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

    monkeypatch.setattr(
        "nro.orchestration.runner.shutil.which", lambda _cmd: "/usr/bin/singularity"
    )

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
    runner.add_step(
        Step.python(
            name="Check Dependencies",
            outputs=(Path("dependencies.complete"),),
            action=lambda: runner.require_cmds(["definitely-not-installed"]),
        )
    )

    with caplog.at_level(logging.INFO, logger="test.runner.dependency-preflight"):
        with pytest.raises(SystemExit, match="Missing required commands on PATH"):
            with runner.run_context():
                runner.execute()

    assert "Failed Step: 002 — Check Dependencies" in caplog.text
    assert "Error: SystemExit: Missing required commands on PATH" in caplog.text
    assert "Dependency Preflight" not in caplog.text


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

    runner.add_step(
        Step.python(
            name="Generate Output",
            outputs=(output,),
            action=lambda: output.write_text("done"),
        )
    )
    with runner.run_context():
        runner.execute()

    current = json.loads(ledger.read_text())
    generated = next(value for value in current.values() if value["name"] == "Generate Output")
    assert generated["status"] == "success"
    assert generated["outputs"] == [str(output)]
    assert (tmp_path / "step-events.jsonl").is_file()
    graph = json.loads((tmp_path / "runner-graph.json").read_text())
    generated_node = next(node for node in graph["nodes"] if node["name"] == "Generate Output")
    assert generated_node["outputs"] == [str(output)]
    assert generated_node["execution"] == "success"


def test_missing_descendant_is_not_reported_as_a_rerun(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "current-steps.json"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    runner = Runner(
        module_name="Fresh Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.fresh-descendant"),
        next_step=count(1).__next__,
    )
    parent = runner.add_step(
        Step.python(
            name="First Output",
            outputs=(first,),
            action=lambda: first.write_text("first"),
        )
    )
    runner.add_step(
        Step.python(
            name="Second Output",
            inputs=(first,),
            outputs=(second,),
            action=lambda: second.write_text("second"),
            after=(parent.id,),
        )
    )

    with runner.run_context():
        runner.execute()

    current = json.loads(ledger.read_text())
    descendant = next(value for value in current.values() if value["name"] == "Second Output")
    assert descendant["reason"] == f"Missing or empty outputs: {second}"


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

    step = runner.add_step(
        Step.python(
            name="Validated Output",
            outputs=(output,),
            action=lambda: output.write_text("valid"),
            validate=lambda: (output.read_text() == "valid", "Semantic validation failed."),
        )
    )
    with runner.run_context():
        states = runner.execute()

    assert states[step.id].value == "dirty"
    assert output.read_text() == "valid"


def test_dirty_step_replaces_output_symlink_without_writing_target(tmp_path: Path) -> None:
    from nro.orchestration.branches import BranchPaths
    from nro.orchestration.execution_context import ExecutionContext

    paths = BranchPaths("main", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "DEV")
    target = tmp_path / "raw" / "source.txt"
    target.parent.mkdir(parents=True)
    target.write_text("raw")
    output = paths.private_project("demo") / "derivatives/nro/func/alias.txt"
    output.parent.mkdir(parents=True)
    output.symlink_to(target)
    observed = []

    def replace_alias() -> None:
        observed.append(output.exists() or output.is_symlink())
        output.write_text("derived")

    runner = Runner(
        module_name="Alias Replacement Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.alias-replacement"),
        next_step=count(1).__next__,
        execution_context=ExecutionContext(paths, "demo", "func:test", ()),
    )
    runner.add_step(
        Step.python(
            name="Replace Alias",
            outputs=(output,),
            action=replace_alias,
            force=True,
        )
    )

    with runner.run_context():
        runner.execute()

    assert observed == [False]
    assert not output.is_symlink()
    assert output.read_text() == "derived"
    assert target.read_text() == "raw"


def test_step_ledger_records_exact_planned_artifact_path(tmp_path: Path, monkeypatch) -> None:
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

    runner.add_step(
        Step.python(
            name="Write Checkpoint",
            outputs=(output,),
            action=lambda: (output.parent.mkdir(parents=True), output.write_text("done")),
        )
    )
    with runner.run_context():
        runner.execute()

    current = json.loads(ledger.read_text())
    checkpoint = next(value for value in current.values() if value["name"] == "Write Checkpoint")
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
    step = runner.add_step(
        Step.python(
            name="Checkpoint",
            outputs=(output,),
            action=lambda: (output.parent.mkdir(parents=True), output.write_text("done")),
        )
    )
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
    runner.add_step(
        Step.python(
            name="Output",
            outputs=(tmp_path / "output",),
            action=lambda: None,
        )
    )
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

    runner.add_step(
        Step.python(
            name="Compound Step",
            outputs=(output,),
            action=execute,
        )
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        with runner.run_context():
            runner.execute()

    assert "Cmd: bash -c" in caplog.text
    assert "Stderr: warning" in caplog.text
    assert "Child Cmd:" not in caplog.text
    assert "Child output:" not in caplog.text
    assert output.read_text() == "result"


def test_streaming_child_heartbeat_names_step_without_repeating_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Process:
        stderr = None

        def __init__(self) -> None:
            self.waits = 0
            self.stdout = io.StringIO("\n\nFreeSurfer output\n   \n")

        def wait(self, *, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["expensive-tool", "--large-argument"], timeout)
            return 0

    logger = logging.getLogger("test.runner.heartbeat")
    runner = Runner(
        module_name="Heartbeat Module",
        container=None,
        binds=(),
        logger=logger,
        next_step=count(1).__next__,
    )
    output = tmp_path / "complete.txt"

    def execute() -> None:
        runner.run_child(["expensive-tool", "--large-argument"], stream_output=True)
        output.write_text("complete")

    runner.add_step(Step.python(name="Surface Reconstruction", outputs=(output,), action=execute))
    times = iter((100.0, 161.0))
    monkeypatch.setattr("nro.orchestration.runner.time.monotonic", lambda: next(times))
    monkeypatch.setattr("nro.orchestration.runner.subprocess.Popen", lambda *_a, **_k: Process())

    with caplog.at_level(logging.INFO, logger=logger.name):
        with runner.run_context():
            runner.execute()

    heartbeat = next(line for line in caplog.messages if "still running after" in line)
    assert heartbeat == "Surface Reconstruction still running after 61 seconds"
    assert "expensive-tool" not in heartbeat
    assert capsys.readouterr().out == "FreeSurfer output\n"


def test_child_command_can_discard_verbose_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Runner(
        module_name="Quiet child module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.runner.quiet-child"),
        next_step=count(1).__next__,
    )
    output = tmp_path / "complete.txt"

    def execute() -> None:
        runner.run_child(
            ["bash", "-c", "printf verbose-output"],
            discard_stdout=True,
        )
        output.write_text("complete")

    runner.add_step(Step.python(name="Quiet child", outputs=(output,), action=execute))
    with runner.run_context():
        runner.execute()

    assert output.read_text() == "complete"
    assert capsys.readouterr().out == ""


def test_module_dag_contract_rejects_topology_change_for_same_signature(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "runner-contract.json"
    first_output = tmp_path / "first.txt"
    first = RunnerGraph("Immutable")
    first.add(
        Step.python(
            name="First",
            inputs=(tmp_path / "source.txt",),
            outputs=(first_output,),
            action=lambda: None,
        )
    )
    first.freeze()
    first.reconcile_contract(contract, signature="source-and-workflow")

    changed = RunnerGraph("Immutable")
    changed_output = tmp_path / "changed.txt"
    changed.add(
        Step.python(
            name="Changed",
            inputs=(tmp_path / "source.txt",),
            outputs=(changed_output,),
            action=lambda: None,
        )
    )
    changed.freeze()
    with pytest.raises(RuntimeError, match="topology changed"):
        changed.reconcile_contract(contract, signature="source-and-workflow")


@pytest.mark.parametrize("contract_value", (None, "not JSON", "[]", "{}"))
def test_uncontracted_existing_outputs_are_not_trusted(
    tmp_path: Path, contract_value: str | None
) -> None:
    contract = tmp_path / "runner-contract.json"
    if contract_value is not None:
        contract.write_text(contract_value)
    existing = tmp_path / "existing.txt"
    missing = tmp_path / "missing.txt"
    existing.write_text("partial attempt")
    graph = RunnerGraph("Interrupted")
    existing_step = graph.add(
        Step.python(name="Existing", outputs=(existing,), action=lambda: None)
    )
    graph.add(Step.python(name="Missing", outputs=(missing,), action=lambda: None))
    graph.freeze()

    assert graph.changed_steps(contract, signature="current") == frozenset({existing_step.id})


def test_runner_contract_changes_remain_step_specific(tmp_path: Path) -> None:
    contract = tmp_path / "runner-contract.json"
    first = RunnerGraph("Independent siblings")
    first.add(
        Step.python(
            name="Changed",
            outputs=(tmp_path / "changed.txt",),
            parameters={"method": "old"},
            action=lambda: None,
        )
    )
    first.add(
        Step.python(
            name="Unchanged",
            outputs=(tmp_path / "unchanged.txt",),
            parameters={"method": "stable"},
            action=lambda: None,
        )
    )
    first.freeze()
    first.reconcile_contract(contract, signature="same-work-item")

    current = RunnerGraph("Independent siblings")
    changed = current.add(
        Step.python(
            name="Changed",
            outputs=(tmp_path / "changed.txt",),
            parameters={"method": "new"},
            action=lambda: None,
        )
    )
    current.add(
        Step.python(
            name="Unchanged",
            outputs=(tmp_path / "unchanged.txt",),
            parameters={"method": "stable"},
            action=lambda: None,
        )
    )
    current.freeze()

    assert current.changed_steps(contract, signature="same-work-item") == frozenset({changed.id})


def test_module_dag_contract_ignores_source_capture_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "runner-contract.json"
    captures = tmp_path / "implementations"
    old_source = captures / ("a" * 64)
    new_source = captures / ("b" * 64)
    relative = Path("nro/modules/networks/resources/reference.nii.gz")
    output = tmp_path / "output.txt"

    first = RunnerGraph("Relocatable resources")
    first.add(
        Step.python(
            name="Use reference",
            inputs=(old_source / relative,),
            outputs=(output,),
            action=lambda: None,
        )
    )
    first.freeze()
    # Emulate a contract written from an older source capture. The old capture
    # need not remain on disk for comparison.
    monkeypatch.delenv("NRO_EXECUTION_SOURCE_ROOT", raising=False)
    first.reconcile_contract(contract, signature="source-and-workflow")

    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(new_source))
    relocated = RunnerGraph("Relocatable resources")
    relocated.add(
        Step.python(
            name="Use reference",
            inputs=(new_source / relative,),
            outputs=(output,),
            action=lambda: None,
        )
    )
    relocated.freeze()

    relocated.bind_contract(contract, signature="source-and-workflow")
    assert relocated.changed_steps(contract, signature="source-and-workflow") == frozenset()


def test_module_dag_contract_rejects_different_captured_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "runner-contract.json"
    captures = tmp_path / "implementations"
    source = captures / ("a" * 64)
    output = tmp_path / "output.txt"
    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(source))

    first = RunnerGraph("Relocatable resources")
    first.add(
        Step.python(
            name="Use reference",
            inputs=(source / "nro/resources/first.nii.gz",),
            outputs=(output,),
            action=lambda: None,
        )
    )
    first.freeze()
    first.reconcile_contract(contract, signature="source-and-workflow")

    changed = RunnerGraph("Relocatable resources")
    changed.add(
        Step.python(
            name="Use reference",
            inputs=(source / "nro/resources/second.nii.gz",),
            outputs=(output,),
            action=lambda: None,
        )
    )
    changed.freeze()
    with pytest.raises(RuntimeError, match="topology changed"):
        changed.bind_contract(contract, signature="source-and-workflow")


def test_command_signature_ignores_source_capture_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captures = tmp_path / "implementations"
    old_source = captures / ("a" * 64)
    new_source = captures / ("b" * 64)
    relative = Path("nro/resources/reference.nii.gz")
    output = tmp_path / "output.txt"

    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(old_source))
    old = RunnerGraph("Relocatable command")
    old.add(
        Step.command_step(
            ("tool", "--reference", str(old_source / relative)),
            outputs=(output,),
        )
    )
    old.freeze()

    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(new_source))
    new = RunnerGraph("Relocatable command")
    new.add(
        Step.command_step(
            ("tool", "--reference", str(new_source / relative)),
            outputs=(output,),
        )
    )
    new.freeze()

    assert (
        old.contract_payload(signature="contract")["nodes"][0]["command_signature"]
        == new.contract_payload(signature="contract")["nodes"][0]["command_signature"]
    )


def test_scientific_change_reruns_only_step_and_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "current-steps.json"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    outputs = {name: tmp_path / f"{name}.txt" for name in ("first", "child", "independent")}
    actions: list[str] = []

    def construct(value: int) -> Runner:
        runner = Runner(
            module_name="Scoped invalidation",
            container=None,
            binds=(),
            logger=logging.getLogger("test.runner.scoped-invalidation"),
            next_step=count(1).__next__,
        )

        def write(name: str) -> None:
            actions.append(name)
            outputs[name].write_text(str(value))

        runner.add_step(
            Step.python(
                name="First",
                outputs=(outputs["first"],),
                action=lambda: write("first"),
                parameters={"value": value},
            )
        )
        runner.add_step(
            Step.python(
                name="Child",
                inputs=(outputs["first"],),
                outputs=(outputs["child"],),
                action=lambda: write("child"),
            )
        )
        runner.add_step(
            Step.python(
                name="Independent",
                outputs=(outputs["independent"],),
                action=lambda: write("independent"),
            )
        )
        return runner

    monkeypatch.setenv("NRO_RUNNER_GRAPH_SIGNATURE", "first")
    with construct(1).run_context() as runner:
        runner.execute()
    assert actions == ["first", "child", "independent"]

    actions.clear()
    monkeypatch.setenv("NRO_RUNNER_GRAPH_SIGNATURE", "second")
    with construct(2).run_context() as runner:
        states = runner.execute()

    assert actions == ["first", "child"]
    assert list(states.values())[-1].value == "fresh"


def test_interrupted_module_preserves_successful_step_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "current-steps.json"
    contract = tmp_path / "runner-contract.json"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    monkeypatch.setenv("NRO_RUNNER_GRAPH_SIGNATURE", "same-work-item")
    first_output = tmp_path / "first.txt"
    second_output = tmp_path / "second.txt"
    actions: list[str] = []

    def construct(*, fail_second: bool) -> Runner:
        runner = Runner(
            module_name="Interrupted module",
            container=None,
            binds=(),
            logger=logging.getLogger("test.runner.interrupted-module"),
            next_step=count(1).__next__,
        )

        def first() -> None:
            actions.append("first")
            first_output.write_text("complete")

        def second() -> None:
            actions.append("second")
            if fail_second:
                raise RuntimeError("interrupted")
            second_output.write_text("complete")

        runner.add_step(Step.python(name="First", outputs=(first_output,), action=first))
        runner.add_step(
            Step.python(
                name="Second",
                inputs=(first_output,),
                outputs=(second_output,),
                action=second,
            )
        )
        return runner

    with pytest.raises(RuntimeError, match="interrupted"):
        with construct(fail_second=True).run_context() as runner:
            runner.execute()

    partial = json.loads(contract.read_text())
    assert partial["version"] == 4
    assert [node["name"] for node in partial["nodes"]] == ["First"]

    actions.clear()
    with construct(fail_second=False).run_context() as runner:
        runner.execute()

    assert actions == ["second"]
    completed = json.loads(contract.read_text())
    assert [node["name"] for node in completed["nodes"]] == ["First", "Second"]


def test_missing_contract_reason_does_not_claim_scientific_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger = tmp_path / "current-steps.json"
    output = tmp_path / "existing.txt"
    output.write_text("unverified")
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    logger = logging.getLogger("test.runner.missing-step-contract")
    runner = Runner(
        module_name="Missing contract",
        container=None,
        binds=(),
        logger=logger,
        next_step=count(1).__next__,
    )
    runner.add_step(
        Step.python(
            name="Rewrite unverified output",
            outputs=(output,),
            action=lambda: output.write_text("verified"),
        )
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        with runner.run_context():
            runner.execute()

    assert "no successful step contract" in "\n".join(caplog.messages)
    assert "scientific declaration changed" not in "\n".join(caplog.messages)


def test_interruption_after_upstream_change_revokes_descendant_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "current-steps.json"
    contract = tmp_path / "runner-contract.json"
    monkeypatch.setenv("NRO_STEP_LEDGER", str(ledger))
    monkeypatch.setenv("NRO_RUNNER_GRAPH_SIGNATURE", "same-topology")
    upstream = tmp_path / "upstream.txt"
    downstream = tmp_path / "downstream.txt"
    actions: list[str] = []

    def construct(*, value: int, fail_downstream: bool = False) -> Runner:
        runner = Runner(
            module_name="Changed upstream",
            container=None,
            binds=(),
            logger=logging.getLogger("test.runner.changed-upstream"),
            next_step=count(1).__next__,
        )

        def write_upstream() -> None:
            actions.append("upstream")
            upstream.write_text(str(value))

        def write_downstream() -> None:
            actions.append("downstream")
            if fail_downstream:
                raise RuntimeError("downstream interrupted")
            downstream.write_text(str(value))

        runner.add_step(
            Step.python(
                name="Upstream",
                outputs=(upstream,),
                action=write_upstream,
                parameters={"value": value},
            )
        )
        runner.add_step(
            Step.python(
                name="Downstream",
                inputs=(upstream,),
                outputs=(downstream,),
                action=write_downstream,
            )
        )
        return runner

    with construct(value=1).run_context() as runner:
        runner.execute()

    actions.clear()
    with pytest.raises(RuntimeError, match="downstream interrupted"):
        with construct(value=2, fail_downstream=True).run_context() as runner:
            runner.execute()

    partial = json.loads(contract.read_text())
    assert [node["name"] for node in partial["nodes"]] == ["Upstream"]

    actions.clear()
    with construct(value=2).run_context() as runner:
        runner.execute()
    assert actions == ["downstream"]


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
    step = runner.add_step(
        Step.command_step(
            ["write-output", str(output)],
            outputs=(output,),
            validate=lambda: (
                output.read_text() == "valid",
                "Semantic command validation failed.",
            ),
        )
    )
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
