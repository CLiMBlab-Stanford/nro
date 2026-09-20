"""Exercise branch output routing through the anatomical builder and runner."""

import json
import logging
from dataclasses import replace

import pytest

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.modules.anat import module as anat
from nro.modules.anat.inputs import AnatImage
from nro.modules.anat.lesion_policy import NEUROLIT_CHECKPOINTS
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import ContainerSpec, Runner
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
            if modalities == ("T1w", "T2w") and (
                (session == "ses-1" and modality == "T2w")
                or (session == "ses-2" and modality == "T1w")
            ):
                continue
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
    base = context.paths.source_project("demo") / "derivatives/nro/anat/main"
    options = anat.Options(
        "demo",
        "main",
        base / "sub-1/anat",
        context.paths.work / "demo/derivatives/nro/anat/main/sub-1",
        base / "code/freesurfer",
        "sub-1",
        "fsaverage6",
        "average",
        "off",
        tmp_path / "gradient.sif",
        "singularity",
        template,
        ContainerSpec(
            image=synthstrip,
            engine="true",
            home_dir=context.paths.work / "demo/derivatives/nro/anat/main/sub-1/_qunex_home",
        ),
        synthstrip,
        False,
        freesurfer_image=synthstrip,
    )
    inputs = anat.Inputs(
        "sub-1",
        tuple(i for i in images if i.modality == "T1w"),
        tuple(i for i in images if i.modality == "T2w"),
    )
    job = anat.build_module(inputs, options, execution_context=context)
    assert job._container is not None
    assert job._container.home_dir.is_relative_to(context.paths.development)
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
                "gradient_unwarping": "off",
                "fs_subject": "sub-1",
                "mni_template": str(template),
                "synthstrip_image": str(synthstrip),
                "configuration_fingerprint": "previous-preprocessing-fingerprint",
            }
        )
    )
    assert configuration.validate is not None and configuration.validate()[0]
    owner = context.paths.output_project("demo") / "derivatives/nro/anat/main"
    assert (owner / "sub-1/anat").is_dir()
    assert (owner / "code/freesurfer").is_dir()
    manifest = next(s for s in graph.steps if s.completion_boundary)
    assert manifest.outputs == (owner / "sub-1/anat/sub-1_desc-preprocessAnat_manifest.json",)
    registrations = [step for step in graph.steps if step.name == "Register T2w to ACPC T1w"]
    if modalities == ("T1w", "T2w"):
        assert len(registrations) == 1
        registration = registrations[0]
        assert registration.inputs[0].is_relative_to(context.paths.development / "dev" / "WORK")
        assert registration.inputs[1] == (
            owner / "sub-1/anat/sub-1_space-ACPC_desc-preproc_T1w.nii.gz"
        )
        assert registration.outputs == (
            owner / "sub-1/anat/sub-1_space-ACPC_desc-preproc_T2w.nii.gz",
            owner / "sub-1/anat/sub-1_from-T2w_to-ACPC_mode-image_xfm.mat",
        )
    else:
        assert not registrations
    for session in ("ses-1", "ses-2"):
        assert any(
            p.is_relative_to(owner / "sub-1" / session / "anat")
            for s in graph.steps
            for p in s.outputs
        )
    assert not base.exists()


def test_lesion_anatomical_graph_is_fixed_and_uses_cut_public_surfaces(
    context, tmp_path, monkeypatch
):
    source = context.paths.source_project("demo") / "sub-1/ses-1/anat/sub-1_ses-1_T1w.nii.gz"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    template = tmp_path / "template.nii.gz"
    template.write_bytes(b"template")
    image = tmp_path / "image.sif"
    image.write_bytes(b"image")
    masker = tmp_path / "synthstroke"
    masker.write_bytes(b"command")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    fastsurfer_data = tmp_path / "fastsurfer-data"
    for name in NEUROLIT_CHECKPOINTS:
        checkpoint = fastsurfer_data / "LIT" / "weights" / name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("model")
    monkeypatch.setattr(
        anat,
        "neuroimaging_environment",
        lambda **kwargs: {"FS_LICENSE": str(license_file)},
    )
    monkeypatch.setattr(
        anat,
        "find_fsaverage_template_surface",
        lambda **kwargs: tmp_path / f"{kwargs['hemi']}.sphere.surf.gii",
    )
    base = context.paths.source_project("demo") / "derivatives/nro/anat/main"
    options = anat.Options(
        "demo",
        "main",
        base / "sub-1/anat",
        context.paths.work / "demo/derivatives/nro/anat/main/sub-1",
        base / "code/freesurfer",
        "sub-1",
        "fsaverage6",
        "first",
        "off",
        tmp_path / "gradient.sif",
        "singularity",
        template,
        ContainerSpec(image=image, engine="true"),
        image,
        False,
        freesurfer_image=image,
        lesion=True,
        lesion_masker_command=masker,
        fastsurfer_image=image,
        fastsurfer_data=fastsurfer_data,
    )

    graph = anat.build_module(
        anat.Inputs("sub-1", (AnatImage(source, None, "T1w", "ses-1", {}, "series", 1.0),), ()),
        options,
        execution_context=context,
    )._graph.freeze()
    names = {step.name for step in graph.steps}

    assert "FastSurfer-LIT Reconstruction" in names
    assert "FreeSurfer Recon-All" not in names
    assert "Automatic Lesion Masking" in names
    assert "Render Lesion Mask QC" in names
    assert "Cut Lesion from Cortical Surfaces" in names
    mask_step = next(step for step in graph.steps if step.name == "Automatic Lesion Masking")
    fastsurfer_step = next(
        step for step in graph.steps if step.name == "FastSurfer-LIT Reconstruction"
    )
    assert "desc-preproc_T1w" in mask_step.inputs[0].name
    assert "desc-fastSurferInput_T1w" in fastsurfer_step.inputs[0].name
    assert fastsurfer_step.scientific_signature
    assert any("selectedFullHead" in path.name for step in graph.steps for path in step.outputs)
    manifest = next(step for step in graph.steps if step.completion_boundary)
    assert any("desc-inpainted_T1w" in str(path) for path in manifest.inputs)
    assert any("desc-lesionQC.png" in str(path) for path in manifest.inputs)
    assert any("surfaceVertexMapping" in str(path) for path in manifest.inputs)
    assert any("surfaceValidity" in str(path) for path in manifest.inputs)
    assert any("lesionReconstruction_summary" in str(path) for path in manifest.inputs)
