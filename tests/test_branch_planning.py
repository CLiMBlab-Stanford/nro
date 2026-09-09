"""Demand stays local even when a complete graph resolves across branch owners."""

from dataclasses import replace

import pytest

from nro.orchestration.artifact_resolution import ArtifactCandidate, scientific_contracts
from nro.orchestration.branch_planning import resolve_branch_plan
from nro.orchestration.branches import BranchPaths, BranchRecord, BranchTopology
from nro.orchestration.contracts import InstanceSpec


def fixture(tmp_path):
    tree = BranchTopology.reserved().register(BranchRecord("feature", "dev"))
    tree = tree.register(BranchRecord("sibling", "dev"))
    paths = BranchPaths("feature", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "DEV")
    root = paths.source_project("demo") / "derivatives"
    common = dict(
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=1,
        config_fingerprint="same",
        directory_label="main",
        runtime_config=tmp_path / "runtime.yml",
        command=("example",),
        resource_class="small",
        output_format="example-v1",
        output_prefix="sub-01",
    )
    parent = InstanceSpec.create(
        key="parent",
        module="extension_parent",
        dependencies=(),
        input_paths=(paths.source_project("demo") / "sub-01/raw.nii",),
        output_root=root / "parent",
        expected_outputs=(root / "parent/sub-01_result.nii",),
        **common,
    )
    child = InstanceSpec.create(
        key="child",
        module="extension_child",
        dependencies=("parent",),
        input_paths=parent.expected_outputs,
        output_root=root / "child",
        expected_outputs=(root / "child/sub-01_result.nii",),
        **common,
    )
    return tree, paths, (parent, child)


def candidate(paths, graph, key="parent", branch="main", *, fresh=True):
    owner = replace(paths, branch=branch)
    root = owner.output_project("demo") / "derivatives" / key
    root.mkdir(parents=True, exist_ok=True)
    (root / "sub-01_result.nii").write_text("result")
    return ArtifactCandidate(
        branch, "producer:" + key, scientific_contracts(graph)[key], 7, root, {"fresh": fresh}
    )


def validated(item):
    return item.evidence["fresh"] and (item.root / "sub-01_result.nii").is_file()


def test_mixed_owner_graph_only_demands_consumer(tmp_path):
    tree, paths, graph = fixture(tmp_path)
    inherited = candidate(paths, graph)
    plan = resolve_branch_plan(tree, paths, graph, ("child",), (inherited,), validate=validated)
    assert [item.spec.key for item in plan.work] == ["child"]
    child = plan.work[0]
    assert (
        child.context.input_path(graph[0].expected_outputs[0])
        == inherited.root / "sub-01_result.nii"
    )
    assert child.context.inputs[0].generation == 7
    assert child.context.inputs[0].key == "producer:parent"
    assert child.context.output_path(graph[1].output_root).is_relative_to(
        paths.output_project("demo")
    )


def test_reused_endpoint_prunes_upstream_work_and_redundant_targets(tmp_path):
    tree, paths, graph = fixture(tmp_path)
    inherited = candidate(paths, graph, "child")
    plan = resolve_branch_plan(
        tree, paths, graph, ("parent", "child"), (inherited,), validate=validated
    )
    assert plan.terminals == ("child",)
    assert plan.work == ()
    assert len(plan.instances) == 1


def test_lost_inherited_artifact_replans_locally(tmp_path):
    tree, paths, graph = fixture(tmp_path)
    inherited = candidate(paths, graph)
    first = resolve_branch_plan(tree, paths, graph, ("child",), (inherited,), validate=validated)
    (inherited.root / "sub-01_result.nii").unlink()
    second = resolve_branch_plan(tree, paths, graph, ("child",), (inherited,), validate=validated)
    assert len(first.work) == 1
    assert [item.spec.key for item in second.work] == ["parent", "child"]
    assert second.work[1].context.inputs[0].branch == "feature"
    assert (
        second.work[1]
        .context.input_path(graph[0].expected_outputs[0])
        .is_relative_to(paths.output_project("demo"))
    )
    assert first.work[0].context.inputs[0].branch == "main"


def test_nearest_fresh_owner_and_inheritance_opt_out(tmp_path):
    tree, paths, graph = fixture(tmp_path)
    candidates = tuple(
        candidate(paths, graph, branch=branch, fresh=branch != "feature")
        for branch in ("main", "dev", "feature", "sibling")
    )
    plan = resolve_branch_plan(tree, paths, graph, ("child",), candidates, validate=validated)
    assert plan.work[0].context.inputs[0].branch == "dev"
    local = resolve_branch_plan(
        tree, paths, graph, ("child",), candidates, validate=validated, inherit=False
    )
    assert len(local.work) == 2


@pytest.mark.parametrize("bad", ["root", "generation", "raw_binding"])
def test_invalid_selection_cannot_escape_input_boundaries(tmp_path, bad):
    tree, paths, graph = fixture(tmp_path)
    inherited = candidate(paths, graph)
    if bad == "root":
        inherited = replace(inherited, root=paths.output_project("demo") / "sub-01")
    elif bad == "generation":
        inherited = replace(inherited, generation=-1)
    else:
        graph = (
            graph[0],
            graph[1].evolve(input_paths=(paths.output_project("demo") / "sub-01/raw.nii",)),
        )
    with pytest.raises(ValueError):
        resolve_branch_plan(tree, paths, graph, ("child",), (inherited,), validate=validated)


def test_selection_needs_no_processing_catalog(tmp_path, monkeypatch):
    import nro.orchestration.catalog as catalog

    tree, paths, graph = fixture(tmp_path)
    monkeypatch.setattr(catalog, "module_descriptor", lambda *a: pytest.fail("catalog accessed"))
    plan = resolve_branch_plan(tree, paths, graph, ("child",), (), validate=validated)
    assert len(plan.work) == 2


def registered_store(tmp_path, monkeypatch):
    import nro.orchestration.branches as branches
    from nro.orchestration.branch_store import BranchStore

    tree, paths, graph = fixture(tmp_path)
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (checkout, "feature", "abc"))
    store = BranchStore(tmp_path / "control")
    snapshot = store.initialize()
    store.register("feature", "dev", revision=snapshot.revision, checkout=checkout)
    return store, checkout, paths, graph


def test_authorized_planning_records_only_consumer_science(tmp_path, monkeypatch):
    store, checkout, paths, graph = registered_store(tmp_path, monkeypatch)
    inherited = candidate(paths, graph)
    plan = store.resolve_plan(checkout, paths, graph, ("child",), (inherited,), validate=validated)
    assert len(plan.work) == 1
    assert len(store.registry("feature").instances()) == 2
    assert store.registry("main").instances() == ()
    with store.registry("feature").connection() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='workers'").fetchone()


@pytest.mark.parametrize("change", ["topology", "science", "checkout"])
def test_plan_rejects_concurrent_changes_without_overwriting_them(tmp_path, monkeypatch, change):
    import nro.orchestration.branches as branches

    store, checkout, paths, graph = registered_store(tmp_path, monkeypatch)
    inherited = candidate(paths, graph)

    def validate(item):
        if change == "topology":
            store.register("other", "dev", revision=store.read().revision)
        elif change == "science":
            store.registry("feature").record_instance(
                "child", {"new": "science"}, expected_revision=None
            )
        else:
            monkeypatch.setattr(branches, "checkout_identity", lambda _: (checkout, "dev", "def"))
        return validated(item)

    with pytest.raises(ValueError):
        store.resolve_plan(checkout, paths, graph, ("child",), (inherited,), validate=validate)
    records = store.registry("feature").instances()
    assert len(records) == (1 if change == "science" else 0)
    if records:
        assert records[0].contract == {"new": "science"}


def test_checkout_cannot_plan_as_main(tmp_path, monkeypatch):
    store, checkout, paths, graph = registered_store(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="authorized checkout"):
        store.resolve_plan(
            checkout, replace(paths, branch="main"), graph, ("child",), (), validate=validated
        )
    assert store.registry("main").instances() == ()
