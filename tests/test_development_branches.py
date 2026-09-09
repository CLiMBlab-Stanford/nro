"""Branch ancestry and output authority without changing live Git or registry state."""

import pytest

from nro.orchestration.branches import BranchPaths, BranchRecord, BranchTopology, branch_id


def topology():
    return (
        BranchTopology.reserved()
        .register(BranchRecord("networks", "dev"))
        .register(BranchRecord("networks/faster", "networks"))
    )


def test_branch_ids_preserve_distinctions():
    names = ["dev", "feature/a", "feature-a", "feature%2Fa", "feature_a"]
    ids = [branch_id(name) for name in names]
    assert len(set(ids)) == len(names)
    assert all("/" not in value for value in ids)
    assert ids[0] == "dev"


@pytest.mark.parametrize("name", ["", "../bad", "a..b", "a b", "-dev", "HEAD", "a@{b", "a" * 201])
def test_invalid_branch_ids(name):
    with pytest.raises(ValueError):
        branch_id(name)


def test_feature_inheritance_follows_dev_and_never_siblings():
    tree = topology().register(BranchRecord("other", "dev"))
    assert tree.ancestors("networks/faster") == ("networks/faster", "networks", "dev", "main")
    assert tree.ancestors("networks/faster", inherit=False) == ("networks/faster",)
    assert tree.ancestors("main") == ("main",)
    with pytest.raises(ValueError, match="descend from dev"):
        tree.register(BranchRecord("bypass", "main"))


def test_reparent_preserves_prior_topology_and_rejects_cycles():
    tree = topology()
    changed = tree.reparent("networks/faster", "dev")
    assert changed.ancestors("networks/faster") == ("networks/faster", "dev", "main")
    assert tree.ancestors("networks/faster")[1] == "networks"
    with pytest.raises(ValueError, match="cycle"):
        tree.reparent("networks", "networks/faster")
    with pytest.raises(ValueError, match="Unregistered"):
        tree.reparent("networks", "missing")
    with pytest.raises(ValueError, match="spine"):
        tree.reparent("dev", "networks")


def test_retirement_preserves_names_and_requires_reparenting():
    tree = topology()
    with pytest.raises(ValueError, match="Reparent"):
        tree.retire("networks")
    changed = tree.reparent("networks/faster", "dev").retire("networks")
    assert changed.records["networks"].retired
    with pytest.raises(ValueError, match="already reserved"):
        changed.register(BranchRecord("networks", "dev"))
    with pytest.raises(ValueError, match="cannot be retired"):
        tree.retire("main")


def test_main_branch_name_alone_grants_no_authority(tmp_path, monkeypatch):
    import nro.orchestration.branches as branches

    root = tmp_path / "production"
    current = [root, "main", "a" * 40]
    monkeypatch.setattr(branches, "checkout_identity", lambda _: tuple(current))
    tree = BranchTopology.reserved()
    with pytest.raises(ValueError, match="not authorized"):
        tree.require_checkout(root)
    tree = tree.authorize_checkout("main", root)
    assert tree.require_checkout(root) == "main"
    current[0] = tmp_path / "personal"
    with pytest.raises(ValueError, match="not authorized"):
        tree.require_checkout(current[0])
    current[:] = [root, "dev", "b" * 40]
    with pytest.raises(ValueError, match="not authorized"):
        tree.require_checkout(root)
    with pytest.raises(ValueError, match="two branches"):
        tree.authorize_checkout("dev", root)


@pytest.mark.parametrize("branch", ["dev", "feature/test"])
def test_development_paths_separate_raw_and_derivatives(tmp_path, branch):
    paths = BranchPaths(branch, tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV")
    base = tmp_path / "NRO_DEV" / branch_id(branch)
    assert paths.source_project("demo") == tmp_path / "BIDS/demo"
    assert paths.output_project("demo") == base / "BIDS/demo"
    assert paths.private_project("demo") == base / "WORK/demo"
    artifact = base / "BIDS/demo/derivatives/clean/main/sub-01/result.nii.gz"
    assert paths.require_output(artifact, "demo") == artifact
    with pytest.raises(ValueError, match="outside"):
        paths.require_output(tmp_path / "BIDS/demo/derivatives/result", "demo")
    with pytest.raises(ValueError, match="outside"):
        paths.require_output(tmp_path / "NRO_DEV/other/BIDS/demo/derivatives/result", "demo")
    with pytest.raises(ValueError, match="outside"):
        paths.require_output(base / "BIDS/demo/sub-01/source.nii.gz", "demo")
    assert not list(tmp_path.iterdir())


def test_main_paths_and_symlink_escape(tmp_path):
    paths = BranchPaths("main", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "NRO_DEV")
    assert paths.output_project("demo") == tmp_path / "BIDS/demo"
    assert paths.private_project("demo") == tmp_path / "WORK/demo"
    derivative = paths.output_project("demo") / "derivatives"
    derivative.mkdir(parents=True)
    (derivative / "escape").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        paths.require_output(derivative / "escape/file", "demo")
    with pytest.raises(ValueError, match="Invalid project"):
        paths.output_project("../other")


def test_overlapping_roots_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not overlap"):
        BranchPaths("dev", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "BIDS/development")
