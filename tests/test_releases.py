"""Release approval records human attestation without changing scientific identity."""

import json
import subprocess

import pytest

from nro.orchestration.branch_store import BranchStore
from nro.orchestration.releases import ReleaseStore


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test Maintainer")
    git(root, "config", "user.email", "maintainer@example.invalid")
    (root / "pyproject.toml").write_text('[project]\nname="example"\nversion="0.0.1"\n')
    git(root, "add", ".")
    git(root, "commit", "-m", "Synthetic initial source")
    branches = BranchStore(tmp_path / "control")
    initial = branches.initialize()
    branches.authorize_checkout("main", root, revision=initial.revision)
    return root, ReleaseStore(branches)


def commit_version(root, version):
    (root / "pyproject.toml").write_text(f'[project]\nname="example"\nversion="{version}"\n')
    git(root, "add", ".")
    git(root, "commit", "-m", "Synthetic version change")


def test_attestation_does_not_tag_deploy_or_change_science(release):
    root, store = release
    before = git(root, "rev-parse", "HEAD")
    row = store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    assert row["commit"] == before
    assert row["attested_by"] == "Test Maintainer <maintainer@example.invalid>"
    assert store.require_approved(root) == row
    assert store.history() == (row,)
    assert git(root, "tag") == ""
    assert git(root, "rev-parse", "HEAD") == before
    assert store.branches.registry("main").instances() == ()


def test_approval_requires_explicit_attestation(release):
    root, store = release
    with pytest.raises(ValueError, match="attestation"):
        store.approve(root, "0.0.1", pr="example#1", attest_merged=False)
    assert not store.path.exists()


@pytest.mark.parametrize("dirty", ["tracked", "untracked"])
def test_dirty_source_cannot_be_approved_or_used(release, dirty):
    root, store = release
    store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    path = root / ("pyproject.toml" if dirty == "tracked" else "new.py")
    path.write_text("changed")
    with pytest.raises(ValueError, match="clean"):
        store.require_approved(root)
    with pytest.raises(ValueError, match="clean"):
        store.approve(root, "0.0.1", pr="example#1", attest_merged=True)


def test_minor_increment_and_package_version_are_required(release):
    root, store = release
    with pytest.raises(ValueError, match="match"):
        store.approve(root, "0.1.0", pr="example#1", attest_merged=True)
    store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    commit_version(root, "0.0.2")
    with pytest.raises(ValueError, match="minor"):
        store.approve(root, "0.0.2", pr="example#2", attest_merged=True)
    commit_version(root, "0.1.0")
    store.approve(root, "0.1.0", pr="example#2", attest_merged=True)
    assert len(store.history()) == 2


def test_unapproved_commit_and_switched_branch_are_rejected(release):
    root, store = release
    with pytest.raises(ValueError, match="attestation"):
        store.require_approved(root)
    store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    commit_version(root, "0.1.0")
    with pytest.raises(ValueError, match="attestation"):
        store.require_approved(root)
    git(root, "switch", "-c", "dev")
    with pytest.raises(ValueError, match="authorized"):
        store.approve(root, "0.1.0", pr="example#2", attest_merged=True)


def test_duplicate_approval_cannot_replace_earlier_attestation(release):
    root, store = release
    first = store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    with pytest.raises(ValueError, match="minor"):
        store.approve(root, "0.0.1", pr="different#2", attest_merged=True)
    assert store.history() == (first,)


def test_symlinked_ledger_is_rejected(release, tmp_path):
    root, store = release
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"schema": 1, "releases": []}))
    store.path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        store.approve(root, "0.0.1", pr="example#1", attest_merged=True)


def test_release_cli_requires_flags_before_writes(tmp_path):
    from nro.bin.release import main

    with pytest.raises(SystemExit) as error:
        main(["0.0.1", "--bids-root", str(tmp_path / "BIDS")])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_new_release_must_descend_from_previous_source(release):
    root, store = release
    store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    git(root, "switch", "--orphan", "replacement")
    commit_version(root, "0.1.0")
    git(root, "branch", "-M", "main")
    with pytest.raises(ValueError, match="Git state"):
        store.approve(root, "0.1.0", pr="example#2", attest_merged=True)
    assert len(store.history()) == 1


def test_missing_human_identity_cannot_record_attestation(release):
    root, store = release
    git(root, "config", "user.email", "")
    with pytest.raises(ValueError, match="human maintainer"):
        store.approve(root, "0.0.1", pr="example#1", attest_merged=True)
    assert store.history() == ()


def test_shared_scheduler_repair_preserves_branch_runtime_and_keeps_backup(
    release, tmp_path, monkeypatch
):
    import sqlite3

    from nro.orchestration import scheduler_repair
    from nro.orchestration.registry import SCHEMA_VERSION, Registry

    root, store = release
    store.approve(root, "0.0.1", pr="test#1", attest_merged=True)
    registry = Registry.for_project(
        "", bids_root=tmp_path / "BIDS", registry_path=store.branches.control
    )
    registry.initialize()
    scientific = store.branches.registry("dev")
    runtime = scientific.root / "workflows" / "example.yml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("retained runtime")
    certificate = scientific.root / "manifests" / "example.json"
    certificate.parent.mkdir(parents=True, exist_ok=True)
    certificate.write_text("{}")
    with sqlite3.connect(registry.paths.database) as db:
        db.execute("PRAGMA user_version=999")
    monkeypatch.setattr(scheduler_repair, "CHECKOUT", root)
    result = scheduler_repair.repair(registry, checkout=root, confirm=lambda activity: True)
    assert runtime.read_text() == "retained runtime"
    assert scientific.database.is_file()
    assert not certificate.exists()
    from pathlib import Path

    backup = Path(result["backup"])
    archived = json.loads((backup / "index.json").read_text())
    assert str(certificate.parent) in archived.values()
    with registry.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert not db.execute("SELECT * FROM requests").fetchall()
