import os
from pathlib import Path

import pytest

from nro.orchestration.runner_graph import RunnerGraph, StagePlan, Step, artifact_decision


def test_shared_artifact_rule_detects_newer_input_without_deleting_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.write_text("one")
    output.write_text("result")
    assert artifact_decision((output,), False, inputs=(source,))[0] is False

    newer = output.stat().st_mtime + 2
    os.utime(source, (newer, newer))
    should_run, reason = artifact_decision((output,), False, inputs=(source,))
    assert should_run
    assert "newer" in reason
    assert output.exists()


def test_missing_inputs_are_ignored_but_existing_private_changes_are_not(
    tmp_path: Path,
) -> None:
    output = tmp_path / "derivative"
    missing_work = tmp_path / "work" / "checkpoint"
    output.write_text("published")
    assert artifact_decision((output,), False, inputs=(missing_work,))[0] is False

    missing_work.parent.mkdir()
    missing_work.write_text("changed")
    newer = output.stat().st_mtime + 2
    os.utime(missing_work, (newer, newer))
    assert artifact_decision((output,), False, inputs=(missing_work,))[0] is True


def test_completion_marker_governs_freshness_for_retained_outputs(tmp_path: Path) -> None:
    retained = tmp_path / "retained"
    generated = tmp_path / "generated"
    source = tmp_path / "source"
    marker = tmp_path / "stage.complete"
    for path in (retained, generated, source, marker):
        path.write_text(path.name)
    os.utime(retained, (1, 1))
    os.utime(generated, (2, 2))
    os.utime(source, (3, 3))
    os.utime(marker, (4, 4))

    should_run, reason = artifact_decision(
        (retained, generated, marker),
        False,
        inputs=(source,),
        freshness_marker=marker,
    )

    assert should_run is False
    assert reason == "Output(s) exist and are up to date."


def test_completion_marker_must_be_a_declared_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    marker = tmp_path / "stage.complete"
    output.write_text("output")
    marker.write_text("complete")

    with pytest.raises(ValueError, match="must also be a declared output"):
        artifact_decision((output,), False, freshness_marker=marker)


def test_completion_marker_does_not_change_the_scientific_contract(tmp_path: Path) -> None:
    output = tmp_path / "output"
    marker = tmp_path / "stage.complete"
    outputs = (output, marker)
    without_marker = RunnerGraph("test")
    without_marker.add(Step.command_step(("true",), outputs=outputs))
    without_marker.freeze()
    with_marker = RunnerGraph("test")
    with_marker.add(Step.command_step(("true",), outputs=outputs, freshness_marker=marker))
    with_marker.freeze()

    assert with_marker.contract_payload(signature="same") == without_marker.contract_payload(
        signature="same"
    )


def test_frozen_graph_infers_dependencies_from_all_declared_producers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    intermediate = tmp_path / "intermediate"
    result = tmp_path / "result"
    graph = RunnerGraph("test")
    producer = graph.add(
        Step.python(
            name="Produce intermediate",
            inputs=(source,),
            outputs=(intermediate,),
            action=lambda: None,
        )
    )
    consumer = graph.add(
        Step.python(
            name="Produce result",
            inputs=(intermediate,),
            outputs=(result,),
            action=lambda: None,
        )
    )

    graph.freeze()

    assert graph.dependencies(consumer) == (producer.id,)
    assert graph.ordered_steps() == (producer, consumer)


def test_graph_construction_does_not_consult_artifact_freshness(
    tmp_path: Path,
) -> None:
    output = tmp_path / "derivative"
    output.write_text("published")
    graph = RunnerGraph("test")
    step = graph.add(Step.python(name="Always declared", outputs=(output,), action=lambda: None))

    graph.freeze()

    assert graph.steps == (step,)
    assert graph.results == {}


def test_stage_plan_binds_steps_to_typed_products(tmp_path: Path) -> None:
    step = Step.python(
        name="Produce output",
        outputs=(tmp_path / "output",),
        action=lambda: None,
    )
    plan = StagePlan(steps=(step,), products={"output": step.outputs[0]})

    assert plan.steps == (step,)
    assert plan.products == {"output": tmp_path / "output"}

    with pytest.raises(TypeError, match="only Step"):
        StagePlan(steps=(object(),), products=None)  # type: ignore[arg-type]


def test_frozen_graph_rejects_mutation(tmp_path: Path) -> None:
    graph = RunnerGraph("test").freeze()
    with pytest.raises(RuntimeError, match="after the runner graph is frozen"):
        graph.add(
            Step.python(
                name="Too late",
                outputs=(tmp_path / "output",),
                action=lambda: None,
            )
        )


def test_graph_rejects_duplicate_output_producers(tmp_path: Path) -> None:
    output = tmp_path / "output"
    graph = RunnerGraph("test")
    graph.add(Step.python(name="First", outputs=(output,), action=lambda: None))
    with pytest.raises(ValueError, match="Duplicate step id"):
        graph.add(Step.python(name="Second", outputs=(output,), action=lambda: None))


def test_substantive_contract_ignores_command_rendering(tmp_path: Path) -> None:
    contract = tmp_path / "runner-contract.json"
    source = tmp_path / "source"
    output = tmp_path / "output"

    original = RunnerGraph("test")
    original.add(
        Step.command_step(
            ["tool", "--input", str(source), "--output", str(output)],
            name="Transform",
            inputs=(source,),
            outputs=(output,),
        )
    )
    original.freeze().reconcile_contract(contract, signature="same-data-and-workflow")

    reformatted = RunnerGraph("test")
    reformatted.add(
        Step.command_step(
            ["tool", f"--input={source}", f"--output={output}"],
            name="Transform",
            inputs=(source,),
            outputs=(output,),
        )
    )
    reformatted.freeze().bind_contract(contract, signature="same-data-and-workflow")
    assert reformatted.changed_steps(contract, signature="new-data-and-workflow") == frozenset()


def test_changed_steps_compares_scientific_declarations_independently(tmp_path: Path) -> None:
    contract = tmp_path / "runner-contract.json"
    first_output = tmp_path / "first.txt"
    second_output = tmp_path / "second.txt"
    original = RunnerGraph("test")
    first = original.add(
        Step.python(
            name="First label",
            outputs=(first_output,),
            action=lambda: None,
            parameters={"resolution": 1},
        )
    )
    original.add(
        Step.python(
            name="Second",
            outputs=(second_output,),
            action=lambda: None,
            parameters={"method": "same"},
        )
    )
    original.freeze().reconcile_contract(contract, signature="old-work_item")

    current = RunnerGraph("test")
    changed = current.add(
        Step.python(
            name="Renamed first label",
            outputs=(first_output,),
            action=lambda: None,
            parameters={"resolution": 2},
        )
    )
    current.add(
        Step.python(
            name="Second",
            outputs=(second_output,),
            action=lambda: None,
            parameters={"method": "same"},
        )
    )
    current.freeze()

    assert first.id == changed.id
    assert current.changed_steps(contract, signature="old-work_item") == frozenset({changed.id})
