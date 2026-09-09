"""Exercise the microparcellation entry point with per-run producer selections."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from nro.modules.microparcellation import __main__ as entry
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext, InputBinding
from nro.orchestration.runner import Runner

pytestmark = pytest.mark.integration


@pytest.fixture
def case(tmp_path, monkeypatch):
    paths = BranchPaths("feature/micro", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV")
    dev = BranchPaths("dev", paths.bids, paths.work, paths.development)
    logical = paths.source_project("demo") / "derivatives/clean/main/sub-1"
    bindings, roots = [], []
    for index in (1, 2):
        # The first producer is main; raw data always remain in the shared tree.
        root = (
            logical if index == 1 else dev.output_project("demo") / "derivatives/clean/main/sub-1"
        )
        root.mkdir(parents=True, exist_ok=True)
        roots.append(root)
        stem = f"sub-1_task-test_run-{index:02d}"
        raw = paths.source_project("demo") / "sub-1/func" / f"{stem}_bold.nii.gz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ones((3, 3, 3, 4), dtype=np.float32), np.eye(4)), raw)
        for space in ("T1w", "fsnative"):
            prefix = f"{stem}_space-{space}_smoothing-2mm"
            bindings.append(
                InputBinding(
                    "main" if index == 1 else "dev",
                    f"clean:{index}:{space}",
                    1,
                    logical,
                    root,
                    prefix,
                )
            )
            if space == "T1w":
                (root / f"{prefix}_desc-clean_bold.nii.gz").write_bytes(raw.read_bytes())
            else:
                for hemi in ("L", "R"):
                    nib.save(
                        nib.gifti.GiftiImage(
                            darrays=[
                                nib.gifti.GiftiDataArray(np.ones(3, dtype=np.float32))
                                for _ in range(4)
                            ]
                        ),
                        root / f"{prefix}_hemi-{hemi}_desc-clean_bold.func.gii",
                    )
            (root / f"{prefix}_desc-confounds_timeseries.tsv").write_text(
                "motion_outlier00\n0\n0\n0\n0\n"
            )
    logical_anat = paths.source_project("demo") / "derivatives/preprocessing/main/sub-1/anat"
    anatomy = dev.output_project("demo") / "derivatives/preprocessing/main/sub-1/anat"
    anatomy.mkdir(parents=True)
    mask = anatomy / "sub-1_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), mask)
    surfaces = {}
    for hemi in ("L", "R"):
        for kind in ("pial", "white", "midthickness", "inflated"):
            path = anatomy / f"sub-1_hemi-{hemi}_{kind}.surf.gii"
            nib.save(
                nib.gifti.GiftiImage(
                    darrays=[
                        nib.gifti.GiftiDataArray(
                            np.eye(3, dtype=np.float32), intent="NIFTI_INTENT_POINTSET"
                        ),
                        nib.gifti.GiftiDataArray(
                            np.array([[0, 1, 2]], dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE"
                        ),
                    ]
                ),
                path,
            )
            surfaces[f"{'lh' if hemi == 'L' else 'rh'}.{kind}"] = str(path)
    (anatomy / "sub-1_desc-preprocessAnat_manifest.json").write_text(
        json.dumps({"outputs": {"gray_matter_mask": str(mask), "surfaces": surfaces}})
    )
    roots.append(anatomy)
    bindings.append(InputBinding("dev", "anat:1", 1, logical_anat, anatomy, "sub-1"))
    context = ExecutionContext(paths, "demo", "micro:1", tuple(bindings))
    cfg = yaml.safe_load(
        (
            Path(entry.__file__).parents[2]
            / "configuration/starters/configs/microparcellation/main_microparcellation.yml"
        ).read_text()
    )
    cfg["preprocessing_directory"] = "main"
    monkeypatch.setattr(entry, "select_runtime_config", lambda **kwargs: tmp_path / "runtime.yml")
    monkeypatch.setattr(entry, "load_runtime_configuration", lambda *args: ("main", cfg))
    monkeypatch.setattr(
        entry,
        "load_runtime_workflow_snapshot",
        lambda *args: {
            "configurations": {
                "preprocessing": {"resolved": {"func": {"output_spaces": ["T1w", "fsnative"]}}},
                "clean": {"directory": "main"},
            }
        },
    )
    return context, roots, cfg


@pytest.mark.parametrize("space", ["T1w", "fsnative"])
def test_entry_routes_mixed_run_owners_and_outputs(case, monkeypatch, space):
    context, roots, _ = case
    before = {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}
    graphs = []

    def initialize_only(self):
        graph = self._graph.freeze()
        graphs.append(graph)
        for step in graph.steps:
            for path in step.outputs:
                context.require_output(path)
        graph.steps[0].action()
        raise RuntimeError("Test stopped after initialization")

    monkeypatch.setattr(Runner, "execute", initialize_only)
    with pytest.raises(RuntimeError, match="Test stopped after initialization"):
        entry.main(["-P", "demo", "-p", "1", "-s", space], execution_context=context)
    assert len(graphs) == 1
    inputs = {p for step in graphs[0].steps for p in step.inputs}
    for root in roots[:2]:
        assert any(p.parent == root and f"_space-{space}_" in p.name for p in inputs)
    assert any(p.parent == roots[2] for p in inputs)
    target = (
        context.paths.output_project("demo")
        / f"derivatives/microparcellation/main/space-{space}_smoothing-2mm/sub-1"
    )
    assert target.is_dir()
    assert any(
        p.parent == target and "Index_manifest.json" in p.name
        for step in graphs[0].steps
        for p in step.outputs
    )
    assert {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()} == before


def test_entry_rejects_missing_run_binding(case):
    context, _, _ = case
    context = replace(
        context, inputs=tuple(b for b in context.inputs if b.key != "clean:2:fsnative")
    )
    with pytest.raises(ValueError, match="not selected"):
        entry.main(["-P", "demo", "-p", "1"], execution_context=context)
    assert not context.paths.output_project("demo").exists()


def test_entry_rejects_foreign_output_override(case):
    context, _, cfg = case
    cfg["output_dir"] = str(context.paths.source_project("demo") / "sub-1")
    with pytest.raises(ValueError, match="outside"):
        entry.main(["-P", "demo", "-p", "1"], execution_context=context)
