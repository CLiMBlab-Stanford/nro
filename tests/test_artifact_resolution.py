"""Branch storage identity does not substitute for scientific equivalence."""

from pathlib import Path

import pytest

from nro.configuration.store import fingerprint
from nro.orchestration.artifact_resolution import (
    ArtifactCandidate,
    scientific_contracts,
    select_artifact,
)
from nro.orchestration.branches import BranchRecord, BranchTopology
from nro.orchestration.contracts import InstanceSpec


def graph(root, *, key_prefix="", config="same", format="example-v1"):
    common = dict(
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=1,
        config_fingerprint=config,
        directory_label="main",
        runtime_config=root / "runtime.yml",
        command=("example",),
        resource_class="small",
        output_format=format,
    )
    parent = InstanceSpec.create(
        key=key_prefix + "parent",
        module="parent",
        dependencies=(),
        input_paths=(Path("/shared/raw.nii"),),
        output_root=root / "parent",
        output_prefix="sub-01",
        expected_outputs=(root / "parent/sub-01_result.nii",),
        **common,
    )
    child = InstanceSpec.create(
        key=key_prefix + "child",
        module="child",
        dependencies=(parent.key,),
        input_paths=parent.expected_outputs,
        output_root=root / "child",
        output_prefix="sub-01",
        expected_outputs=(root / "child/sub-01_result.nii",),
        **common,
    )
    return parent, child


def test_contracts_ignore_owned_locations_and_registry_keys(tmp_path):
    original = graph(tmp_path / "main")
    relocated = graph(tmp_path / "feature", key_prefix="feature:")
    first, second = scientific_contracts(original), scientific_contracts(relocated)
    assert first["parent"] == second["feature:parent"]
    assert first["child"] == second["feature:child"]
    assert first["child"]["inputs"] == [
        {
            "artifact": fingerprint(first["parent"]),
            "member": "sub-01_result.nii",
        }
    ]


@pytest.mark.parametrize(
    "change", ["config", "format", "source", "topology", "output", "processing"]
)
def test_substantive_changes_still_invalidate_downstream_contracts(tmp_path, change):
    parent, child = graph(tmp_path)
    before = scientific_contracts((parent, child))["child"]
    if change == "config":
        parent = parent.evolve(config_fingerprint="changed")
    elif change == "format":
        parent = parent.evolve(output_format="example-v2")
    elif change == "source":
        parent = parent.evolve(input_paths=(Path("/shared/different.nii"),))
    elif change == "topology":
        child = child.evolve(dependencies=())
    elif change == "output":
        parent = parent.evolve(
            expected_outputs=(*parent.expected_outputs, tmp_path / "parent/sub-01_extra.json")
        )
    else:
        parent = parent.evolve(processing={"algorithm": "different"})
    assert scientific_contracts((parent, child))["child"] != before


def test_runtime_changes_do_not_affect_scientific_contracts(tmp_path):
    parent, child = graph(tmp_path)
    changed = child.evolve(
        command=("other", "--formatting"),
        runtime_config=tmp_path / "new.yml",
        memory_gb=128,
        configuration_lineage_id=42,
    )
    assert scientific_contracts((parent, child)) == scientific_contracts((parent, changed))


def test_inheritance_requires_current_validation_and_excludes_siblings(tmp_path):
    topology = BranchTopology.reserved().register(BranchRecord("feature", "dev"))
    topology = topology.register(BranchRecord("sibling", "dev"))
    candidates = [
        ArtifactCandidate(
            branch, branch, {"science": 1}, 1, tmp_path / branch, {"fresh": branch != "feature"}
        )
        for branch in ("main", "dev", "feature", "sibling")
    ]

    def validate(candidate):
        return candidate.evidence["fresh"]

    assert (
        select_artifact(topology, "feature", {"science": 1}, candidates, validate=validate).branch
        == "dev"
    )
    assert (
        select_artifact(
            topology, "feature", {"science": 1}, candidates, validate=validate, inherit=False
        )
        is None
    )
    assert (
        select_artifact(topology, "main", {"science": 1}, candidates[1:], validate=validate) is None
    )
    assert (
        select_artifact(topology, "feature", {"science": 2}, candidates, validate=validate) is None
    )


def test_invalid_or_ambiguous_graphs_are_rejected(tmp_path):
    parent, child = graph(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        scientific_contracts((parent, parent))
    with pytest.raises(ValueError, match="lacks"):
        scientific_contracts((child,))
    with pytest.raises(ValueError, match="escapes"):
        scientific_contracts((parent.evolve(expected_outputs=(tmp_path / "elsewhere",)), child))


def test_branch_graph_registration_is_atomic_and_location_independent(tmp_path):
    from nro.orchestration.branch_store import BranchStore

    store = BranchStore(tmp_path / "control")
    store.initialize()
    registry = store.registry("dev")
    initial = graph(tmp_path / "first")
    records = registry.record_graph(initial, expected_revisions={"parent": None, "child": None})
    registry.record_observation("child", {"fresh": True}, expected_revision=1)
    relocated = graph(tmp_path / "second")
    registry.record_graph(relocated, expected_revisions={"parent": 1, "child": 1})
    assert next(item for item in registry.instances() if item.key == "child").observation == {
        "fresh": True
    }
    with pytest.raises(ValueError, match="graph changed"):
        registry.record_graph(
            graph(tmp_path / "second", config="changed"),
            expected_revisions={"parent": 1, "child": None},
        )
    assert next(item for item in registry.instances() if item.key == "parent") == records[0]


def test_branch_workflows_do_not_change_other_branches(tmp_path):
    from nro.configuration.store import ConfigStore
    from nro.orchestration.branch_store import BranchStore

    store = BranchStore(tmp_path / "control")
    store.initialize()
    dev = store.registry("dev")
    workflow = ConfigStore().resolve("main")
    registered = dev.register_workflow(workflow)
    assert dev.runtime_config_path(registered, "clean").is_file()
    with store.registry("main").connection() as db:
        assert db.execute("SELECT COUNT(*) FROM workflow_revisions").fetchone()[0] == 0
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='workers'").fetchone()


def test_execution_paths_keep_raw_data_shared_and_outputs_local(tmp_path):
    from nro.orchestration.branches import BranchPaths
    from nro.orchestration.execution_context import ExecutionContext, InputBinding

    paths = BranchPaths("feature", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "DEV")
    logical = paths.bids / "demo/derivatives/preprocessing/main/sub-01/anat"
    selected = paths.bids / "demo/derivatives/preprocessing/other/sub-01/anat"
    context = ExecutionContext(
        paths, "demo", "child", (InputBinding("main", "parent", 3, logical, selected, "sub-01"),)
    )
    assert context.input_path(logical / "sub-01_result.nii") == selected / "sub-01_result.nii"
    raw = paths.bids / "demo/sub-01/anat/sub-01_T1w.nii"
    assert context.input_path(raw) == raw
    assert context.output_path(logical / "sub-01_result.nii") == (
        paths.development
        / "feature/BIDS/demo/derivatives/preprocessing/main/sub-01/anat/sub-01_result.nii"
    )
    with pytest.raises(ValueError, match="not selected"):
        context.input_path(logical / "sub-02_result.nii")
    with pytest.raises(ValueError, match="not selected"):
        context.input_path(paths.development / "sibling/BIDS/demo/derivatives/result.nii")
    with pytest.raises(ValueError, match="normalized"):
        context.input_path(logical / "../sub-01_result.nii")
    with pytest.raises(ValueError, match="outside"):
        context.output_path(raw)
    assert ExecutionContext.from_dict(context.as_dict()) == context


def test_debug_bids_cannot_supply_scientific_inputs(tmp_path):
    from nro.orchestration.branches import BranchPaths
    from nro.orchestration.execution_context import ExecutionContext, InputBinding

    paths = BranchPaths("dev", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "DEV")
    raw = paths.source_project("demo") / "sub-01/anat"
    debug = paths.output_project("demo") / "sub-01/anat"
    artifact = paths.source_project("demo") / "derivatives/anat/main/sub-01/anat"
    context = ExecutionContext(paths, "demo", "child", ())
    with pytest.raises(ValueError, match="not selected"):
        context.input_path(debug / "sub-01_T1w.nii.gz")
    for logical in (raw, artifact):
        binding = InputBinding("dev", "debug", 1, logical, debug, None)
        selected = ExecutionContext(paths, "demo", "child", (binding,))
        with pytest.raises(ValueError, match="raw scientific|outside"):
            selected.input_path(logical / "sub-01_T1w.nii.gz")
    debug.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    raw.symlink_to(debug, target_is_directory=True)
    with pytest.raises(ValueError, match="not selected"):
        context.input_path(raw / "sub-01_T1w.nii.gz")
