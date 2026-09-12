"""Construct functional work with selected ancestor anatomy and owned outputs."""

import json
from dataclasses import fields, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.modules.func import module as func
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext, InputBinding

pytestmark = pytest.mark.integration


@pytest.fixture
def functional_case(tmp_path, monkeypatch):
    paths = BranchPaths("feature/test", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV")
    logical = paths.source_project("demo") / "derivatives/preprocessing/main/sub-1/anat"
    ancestor = BranchPaths("dev", paths.bids, paths.work, paths.development)
    physical = ancestor.output_project("demo") / "derivatives/preprocessing/main/sub-1/anat"
    physical.mkdir(parents=True)
    image = physical / "sub-1_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), image)
    template = tmp_path / "template.nii.gz"
    template.write_bytes(image.read_bytes())
    subjects = physical.parents[1] / "code/freesurfer"
    subjects.mkdir(parents=True)
    manifest = physical / "sub-1_desc-preprocessAnat_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "fs_subject": "sub-1",
                "freesurfer_subjects_dir": str(subjects),
                "mni_template": str(template),
                "outputs": {
                    "subject_t1w": str(image),
                    "brain_mask": str(image),
                    "xfms": {"t1_to_mni": str(image), "mni_to_t1": str(image)},
                    "surfaces": {
                        f"{hemi}.{surface}": str(image)
                        for hemi in ("lh", "rh")
                        for surface in ("white", "pial", "midthickness", "sphere.reg")
                    },
                },
            }
        )
    )
    context = ExecutionContext(
        paths,
        "demo",
        "func:test",
        (InputBinding("dev", "anat:test", 1, logical, physical, "sub-1"),),
    )
    raw = paths.source_project("demo") / "sub-1/ses-1/func/sub-1_ses-1_task-rest_bold.nii.gz"
    raw.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3, 5), dtype=np.float32), np.eye(4)), raw)
    metadata = raw.with_name(raw.name.replace(".nii.gz", ".json"))
    metadata.write_text(
        json.dumps({"RepetitionTime": 2.0, "PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05})
    )
    config_path = (
        Path(func.__file__).parents[2]
        / "configuration/starters/configs/preprocessing/main_preprocessing.yml"
    )
    cfg = yaml.safe_load(config_path.read_text())
    configure({"common": {"qunex_container": "/tmp/qunex.sif"}, "get_confounds": cfg["confounds"]})
    values = {
        field.name: cfg["func"][field.name]
        for field in fields(func.Options)
        if field.name in cfg["func"]
    }
    values.update(
        fsaverage_template=cfg["fsaverage_template"],
        out_dir=logical.parent / "ses-1/func",
        work_dir=paths.work / "demo/derivatives/preprocessing/main/sub-1/ses-1/func/run",
        project="demo",
        preprocessing_id="main",
        sub_id="sub-1",
        ses_id="ses-1",
        container=None,
        synbold_disco_image=tmp_path / "synbold.sif",
        synbold_disco_license=tmp_path / "license",
        synbold_disco_engine="singularity",
        sdc_method="syn",
        output_spaces=("T1w", "fsnative", "fsaverage6", "MNI152NLin2009cAsym"),
    )
    values["synbold_disco_image"].write_bytes(b"test image")
    monkeypatch.setattr(
        func,
        "find_fsaverage_template_surface",
        lambda **kwargs: tmp_path / f"{kwargs['hemi']}.sphere.surf.gii",
    )
    return (
        func.Inputs(None, raw, None, None, epi_json=metadata),
        func.Options(**values),
        context,
        manifest,
        image,
    )


@pytest.mark.parametrize("aroma", [True, False])
@pytest.mark.parametrize("fieldmaps", [False, True])
def test_functional_graph_reads_selected_anatomy_and_owns_writes(functional_case, aroma, fieldmaps):
    inputs, options, context, manifest, image = functional_case
    if fieldmaps:
        fmap = inputs.epi.parent.parent / "fmap"
        fmap.mkdir()
        images, metadata = [], []
        for direction in ("j", "j-"):
            path = fmap / f"fieldmap-{len(images)}.nii.gz"
            path.write_bytes(image.read_bytes())
            sidecar = path.with_name(path.name.replace(".nii.gz", ".json"))
            sidecar.write_text(
                json.dumps({"PhaseEncodingDirection": direction, "TotalReadoutTime": 0.05})
            )
            images.append(path)
            metadata.append(sidecar)
        inputs = replace(
            inputs, se1=images[0], se2=images[1], se1_json=metadata[0], se2_json=metadata[1]
        )
    before = {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}
    job = func.build_module(
        inputs, replace(options, clean_ica_aroma=aroma), execution_context=context
    )
    graph = job._graph.freeze()
    assert any(image in step.inputs for step in graph.steps)
    assert any("space-fsaverage6" in path.name for step in graph.steps for path in step.outputs)
    for step in graph.steps:
        for path in step.outputs:
            context.require_output(path)
    public = context.paths.output_project("demo") / "derivatives/preprocessing/main"
    assert not public.exists()
    graph.steps[0].action()
    assert (public / "sub-1/ses-1/func").is_dir()
    final = next(step for step in graph.steps if step.completion_boundary)
    assert (
        public / "sub-1/ses-1/func/sub-1_ses-1_task-rest_desc-preprocessFunc_manifest.json"
        in final.outputs
    )
    assert {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()} == before
    assert not (context.paths.source_project("demo") / "derivatives").exists()


def test_functional_graph_rejects_unselected_anatomy(functional_case):
    inputs, options, context, _, _ = functional_case
    with pytest.raises(ValueError, match="not selected"):
        func.build_module(inputs, options, execution_context=replace(context, inputs=()))
    assert not context.paths.output_project("demo").exists()


def test_fieldmapless_synbold_graph_accepts_missing_sbref(functional_case):
    inputs, options, context, _, _ = functional_case
    options.synbold_disco_license.write_text("test license")

    job = func.build_module(
        inputs,
        replace(options, sdc_method="synbold_disco"),
        execution_context=context,
    )

    steps = job._graph.freeze().steps
    selection = next(
        step for step in steps if step.name == "Select Functional Registration Reference"
    )
    assert inputs.sbref is None
    assert any(step.name == "TOPUP Distortion Estimation Directory" for step in steps)
    assert (
        selection.outputs[0]
        in next(
            step for step in steps if step.name == "TOPUP Distortion Estimation Directory"
        ).inputs
    )


def test_marss_precedes_the_ordinary_reference_and_resampling_graph(functional_case):
    inputs, options, context, _, _ = functional_case
    job = func.build_module(
        inputs,
        replace(options, marss_mode="diagnose"),
        execution_context=context,
    )
    steps = job._graph.freeze().steps
    names = [step.name for step in steps]
    correction_index = names.index("Diagnose and Correct Simultaneous-Slice Artifact")
    reference_index = names.index("Robust BOLD Reference and Motion Correction")
    resampling_index = next(
        index for index, name in enumerate(names) if name.startswith("Resample BOLD (")
    )
    assert names.index("Estimate Native Motion for MARSS") < correction_index
    assert correction_index < reference_index < resampling_index
    selected_native = next(
        path
        for path in steps[correction_index].outputs
        if path.name.endswith("_desc-marss_bold.nii.gz")
    )
    assert selected_native in steps[reference_index].inputs


def test_debug_truncation_and_marss_construct_a_fixed_graph(functional_case):
    inputs, options, context, _, _ = functional_case
    job = func.build_module(
        inputs,
        replace(options, debug_first_nvols=3, marss_mode="auto"),
        execution_context=context,
    )
    steps = job._graph.freeze().steps
    reference = next(
        step for step in steps if step.name == "Robust BOLD Reference and Motion Correction"
    )

    assert any(path.name.endswith("_desc-marss_bold.nii.gz") for path in reference.inputs)
    assert not any(path.exists() for path in reference.inputs)


def test_functional_graph_rejects_project_mismatch(functional_case):
    inputs, options, context, _, _ = functional_case
    with pytest.raises(ValueError, match="project differs"):
        func.build_module(inputs, replace(options, project="other"), execution_context=context)


def test_functional_graph_does_not_replace_a_missing_selected_input(functional_case):
    inputs, options, context, manifest, _ = functional_case
    other = context.inputs[0].logical_root / manifest.name
    other.parent.mkdir(parents=True)
    other.write_bytes(manifest.read_bytes())
    manifest.unlink()
    with pytest.raises(SystemExit, match="Missing anatomical manifest"):
        func.build_module(inputs, options, execution_context=context)
    assert not context.paths.output_project("demo").exists()
