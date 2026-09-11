"""Exercise branch output routing through the anatomical builder and runner."""

import json
import logging
from dataclasses import replace

import pytest

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.modules.anat import module as anat
from nro.modules.anat.common import AnatImage
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

pytestmark = pytest.mark.integration


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(
        BranchPaths("dev", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV"),
        "demo",
        "anat:test",
        (),
    )


def runner(context):
    return Runner(
        module_name="Test",
        container=None,
        binds=(),
        logger=logging.getLogger(__name__),
        next_step=iter(range(100)).__next__,
        execution_context=context,
    )


def test_runner_executes_owned_outputs(context):
    output = context.paths.output_project("demo") / "derivatives/test/result"
    output.parent.mkdir(parents=True)
    job = runner(context)
    job.add_step(
        Step.python(name="Write", outputs=(output,), action=lambda: output.write_text("ok"))
    )
    with job.run_context():
        job.execute()
    assert output.read_text() == "ok"
    assert not (context.paths.bids / "demo/derivatives").exists()


@pytest.mark.parametrize("field", ["outputs", "directory", "breadcrumb", "cwd"])
def test_runner_rejects_foreign_destinations_before_graph_mutation(context, field):
    own = context.paths.output_project("demo") / "derivatives/test/result"
    foreign = context.paths.bids / "demo/derivatives/result"
    step = Step.python(name="Write", outputs=(own,), action=lambda: None)
    step = replace(step, **{field: (foreign,) if field == "outputs" else foreign})
    job = runner(context)
    with pytest.raises(ValueError, match="outside"):
        job.add_step(step)
    assert not job._graph.steps


def test_runner_rechecks_destinations_before_execution(context, tmp_path):
    parent = context.paths.output_project("demo") / "derivatives/test"
    output = parent / "result"
    job = runner(context)
    job.add_step(
        Step.python(name="Write", outputs=(output,), action=lambda: output.write_text("bad"))
    )
    parent.parent.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    parent.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"), job.run_context():
        job.execute()
    assert not list(foreign.iterdir())


def test_runner_rechecks_after_preceding_step(context, tmp_path):
    base = context.paths.output_project("demo") / "derivatives/test"
    base.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    marker = base / "first.complete"
    output = base / "redirect/result"

    def redirect():
        (base / "redirect").symlink_to(foreign, target_is_directory=True)
        marker.write_text("done")

    job = runner(context)
    job.add_step(Step.python(name="Redirect", outputs=(marker,), action=redirect))
    job.add_step(
        Step.python(
            name="Write",
            inputs=(marker,),
            outputs=(output,),
            action=lambda: output.write_text("bad"),
        )
    )
    with pytest.raises(ValueError, match="outside"), job.run_context():
        job.execute()
    assert marker.exists()
    assert not list(foreign.iterdir())


@pytest.mark.parametrize("modalities", [("T1w",), ("T2w",), ("T1w", "T2w")])
def test_anatomical_graph_routes_all_outputs(context, tmp_path, monkeypatch, modalities):
    images = []
    for session in ("ses-1", "ses-2"):
        for modality in modalities:
            source = (
                context.paths.source_project("demo")
                / "sub-1"
                / session
                / "anat"
                / f"sub-1_{session}_{modality}.nii.gz"
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            images.append(AnatImage(source, None, modality, session, {}, "series", 1.0))
    template = tmp_path / "template.nii.gz"
    template.write_bytes(b"template")
    synthstrip = tmp_path / "synthstrip.sif"
    synthstrip.write_bytes(b"image")
    monkeypatch.setattr(
        anat,
        "find_fsaverage_template_surface",
        lambda **kwargs: tmp_path / f"{kwargs['hemi']}.sphere.surf.gii",
    )
    base = context.paths.source_project("demo") / "derivatives/preprocessing/main"
    options = anat.Options(
        "demo",
        "main",
        base / "sub-1/anat",
        context.paths.work / "demo/derivatives/preprocessing/main/sub-1/anat",
        base / "code/freesurfer",
        "sub-1",
        "fsaverage6",
        "average",
        template,
        None,
        synthstrip,
        False,
        2,
    )
    inputs = anat.Inputs(
        "sub-1",
        tuple(i for i in images if i.modality == "T1w"),
        tuple(i for i in images if i.modality == "T2w"),
    )
    job = anat.build_module(inputs, options, execution_context=context)
    graph = job._graph.freeze()
    assert any("to-fsaverage6" in path.name for step in graph.steps for path in step.outputs)
    for step in graph.steps:
        for output in step.outputs:
            context.require_output(output)
    assert not base.exists()
    assert not context.paths.development.exists()
    # Execute the real initialization action without invoking imaging software.
    graph.steps[0].action()
    configuration = next(
        step for step in graph.steps if step.name == "Write Anatomical Configuration"
    )
    configuration.outputs[0].write_text(
        json.dumps(
            {
                "selection_strategy": "average",
                "fs_subject": "sub-1",
                "mni_template": str(template),
                "synthstrip_image": str(synthstrip),
                "configuration_fingerprint": "previous-preprocessing-fingerprint",
            }
        )
    )
    assert configuration.validate is not None and configuration.validate()[0]
    owner = context.paths.output_project("demo") / "derivatives/preprocessing/main"
    assert (owner / "sub-1/anat").is_dir()
    assert (owner / "code/freesurfer").is_dir()
    manifest = next(s for s in graph.steps if s.completion_boundary)
    assert manifest.outputs == (owner / "sub-1/anat/sub-1_desc-preprocessAnat_manifest.json",)
    for session in ("ses-1", "ses-2"):
        assert any(
            p.is_relative_to(owner / "sub-1" / session / "anat")
            for s in graph.steps
            for p in s.outputs
        )
    assert not base.exists()
