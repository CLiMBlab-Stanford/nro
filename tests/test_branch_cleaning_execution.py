"""Clean selected ancestor data without writing into either producer's tree."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from nro.configuration.runtime import configure
from nro.engine.images import load_gifti_timeseries, sidecar_json_path
from nro.modules.clean import module as clean
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext, InputBinding

pytestmark = pytest.mark.integration


@pytest.fixture(params=[None, "ses-1"])
def cleaning_case(tmp_path, request):
    session = request.param
    paths = BranchPaths("feature/clean", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV")
    ancestor = BranchPaths("dev", paths.bids, paths.work, paths.development)
    relative = Path("sub-1") / session if session else Path("sub-1")
    stem = "sub-1" + ("_ses-1" if session else "") + "_task-test_run-01"
    raw = paths.source_project("demo") / relative / "func"
    raw.mkdir(parents=True)
    events = raw / f"{stem}_events.tsv"
    events.write_text("onset\tduration\ttrial_type\n10\t2\ta\n40\t2\ta\n")
    logical = paths.source_project("demo") / "derivatives/preprocessing/main"
    anat = logical / "sub-1/anat"
    anat.mkdir(parents=True)
    mask = anat / "sub-1_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), mask)
    anatomy = anat / "sub-1_desc-preprocessAnat_manifest.json"
    anatomy.write_text(
        json.dumps(
            {
                "mni_template": str(mask),
                "outputs": {
                    "gray_matter_mask": str(mask),
                    "surfaces": {"lh.midthickness": str(mask), "rh.midthickness": str(mask)},
                },
            }
        )
    )
    func = ancestor.output_project("demo") / "derivatives/preprocessing/main" / relative / "func"
    func.mkdir(parents=True)
    rng = np.random.default_rng(9)
    metrics = []
    for space in ("T1w", "fsnative"):
        names = (
            [f"{stem}_space-T1w_desc-preproc_bold.nii.gz"]
            if space == "T1w"
            else [
                f"{stem}_space-fsnative_hemi-{hemi}_desc-preproc_bold.func.gii"
                for hemi in ("L", "R")
            ]
        )
        for name in names:
            path = func / name
            if space == "T1w":
                nib.save(
                    nib.Nifti1Image(rng.normal(size=(3, 3, 3, 60)).astype("float32"), np.eye(4)),
                    path,
                )
            else:
                nib.save(
                    nib.gifti.GiftiImage(
                        darrays=[
                            nib.gifti.GiftiDataArray(rng.normal(size=7).astype("float32"))
                            for _ in range(60)
                        ]
                    ),
                    path,
                )
            sidecar_json_path(path).write_text(json.dumps({"RepetitionTime": 2.0}))
            metrics.append(str(path))
    manifest = func / f"{stem}_desc-preprocessFunc_manifest.json"
    manifest.write_text(json.dumps({"run_stem": stem, "public_outputs": {"clean_inputs": metrics}}))
    pd.DataFrame({"trans_x": rng.normal(size=60), "motion_outlier00": [1] + [0] * 59}).to_csv(
        func / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False
    )
    (func / f"{stem}_desc-confounds_timeseries.json").write_text("{}")
    context = ExecutionContext(
        paths,
        "demo",
        "clean:test",
        (
            InputBinding("dev", "func:test", 1, logical / relative / "func", func, stem),
            InputBinding("main", "anat:test", 1, anat, anat, "sub-1"),
        ),
    )
    cfg = yaml.safe_load(
        (
            Path(clean.__file__).parents[2] / "configuration/starters/configs/clean/main_clean.yml"
        ).read_text()
    )
    cfg.update(
        container="/tmp/qunex.sif",
        container_engine="singularity",
        container_bind=[],
        no_container=True,
    )
    configure(
        {
            "clean": cfg,
            "common": {
                "project": "demo",
                "preprocessing_id": "main",
                "clean_id": "main",
                "qunex_home_dirname": "_qunex_home",
                "wb_command": "wb_command",
            },
        }
    )
    args = ["--sub-id", "sub-1", "--run-stem", stem]
    if session:
        args.extend(["--ses-id", session])
    return args, context, (anat, func), events


@pytest.mark.parametrize("space", ["T1w", "fsnative"])
@pytest.mark.parametrize("smoothing", [0, 2])
def test_clean_graph_keeps_selected_inputs_and_owned_outputs(cleaning_case, space, smoothing):
    args, context, sources, events = cleaning_case
    before = {p: p.read_bytes() for root in sources for p in root.rglob("*") if p.is_file()}
    job, _, count = clean.build_module(
        args + ["--space", space, "--smoothing", str(smoothing)], execution_context=context
    )
    graph = job._graph.freeze()
    for step in graph.steps:
        for path in step.outputs:
            context.require_output(path)
    assert any(events in step.inputs for step in graph.steps)
    assert count == (2 if space == "fsnative" else 1)
    assert not context.paths.output_project("demo").exists()
    if space == "fsnative" and smoothing == 0:
        with job.run_context():
            job.execute()
        final = next(step for step in graph.steps if step.completion_boundary)
        assert final.outputs[0].is_file()
        for path in context.paths.output_project("demo").rglob("*.func.gii"):
            data = load_gifti_timeseries(path)
            assert data.shape == (60, 7)
            assert np.isfinite(data).all()
            sidecar = json.loads(sidecar_json_path(path).read_text())
            assert sidecar["Cleaning"]["RetainedFrames"] == 59
            assert sidecar["Cleaning"]["CleaningDefined"]
    else:
        graph.steps[0].action()
    assert {p: p.read_bytes() for root in sources for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("missing", ["main", "dev"])
def test_clean_rejects_missing_selection(cleaning_case, missing):
    args, context, _, _ = cleaning_case
    context = replace(
        context, inputs=tuple(item for item in context.inputs if item.branch != missing)
    )
    with pytest.raises(ValueError, match="not selected"):
        clean.build_module(
            args + ["--space", "fsnative", "--smoothing", "0"], execution_context=context
        )
    assert not context.paths.output_project("demo").exists()


def test_clean_rejects_foreign_container_home(cleaning_case):
    args, context, _, _ = cleaning_case
    with pytest.raises(ValueError, match="outside"):
        clean.build_module(
            args
            + [
                "--space",
                "fsnative",
                "--smoothing",
                "0",
                "--container-home",
                str(context.paths.work / "foreign"),
            ],
            execution_context=context,
        )
