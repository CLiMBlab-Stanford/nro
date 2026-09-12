"""Worker selection follows the central installation, not the requesting checkout."""

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nro.configuration.site import settings
from nro.orchestration import scheduler_implementation as implementation
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.execution_pins import capture_site
from nro.orchestration.registry import Registry
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.submission import _write_worker_script


@pytest.fixture
def central(tmp_path):
    root = tmp_path / "main"
    support = root / "nro/orchestration"
    support.mkdir(parents=True)
    (root / "nro/__init__.py").write_text("")
    (support / "__init__.py").write_text("")
    shutil.copyfile(
        Path(__file__).parents[1] / "nro/orchestration/source_launcher.py",
        support / "source_launcher.py",
    )
    (support / "worker.py").write_text(
        'import json,sys\nprint(json.dumps({"worker": "central", "args": sys.argv[1:]}))\n'
    )
    config = root / "nro/configuration"
    config.mkdir()
    (config / "__init__.py").write_text("")
    (config / "site.py").write_text("ENVIRONMENT_KEYS = {}\n")
    (root / "pyproject.toml").write_text('[project]\nname="nro"\nversion="0.0.1"\n')
    (root / ".gitignore").write_text(".nro-installation.json\n")
    for args in [
        ("init", "-b", "main"),
        ("config", "user.name", "Test Maintainer"),
        ("config", "user.email", "test@example.invalid"),
        ("add", "."),
        ("commit", "-m", "Test release"),
    ]:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    branches = BranchStore(registry.paths.control)
    branches.authorize_checkout("main", root, revision=branches.initialize().revision)
    releases = ReleaseStore(branches)
    release = releases.approve(root, "0.0.1", pr="test#1", attest_merged=True)
    values = {
        **settings()[0],
        "registry": str(registry.paths.control),
        "bids": str(registry.paths.bids_root),
    }
    site = capture_site(tmp_path / "site", values)
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").symlink_to(sys.executable)
    (root / ".nro-installation.json").write_text(
        json.dumps(
            dict(
                mode="shared",
                ready=True,
                checkout=str(root),
                environment=str(environment),
                site=str(site),
                release=release,
            )
        )
    )
    return registry, root, environment / "bin/python"


def test_different_checkout_submits_central_worker(central, monkeypatch, tmp_path):
    registry, root, python = central
    implementation.activate(registry, root)
    original = implementation.installation_record
    monkeypatch.setattr(
        implementation,
        "installation_record",
        lambda *args: original(*args) if args else {"mode": "branch"},
    )
    development = tmp_path / "development"
    development.mkdir()
    monkeypatch.chdir(development)
    script = _write_worker_script(
        registry,
        bids_root=registry.paths.bids_root,
        partition="test",
        account=None,
        hours=1,
        memory_gb=2,
        cpus=1,
    )
    command = shlex.split(
        next(line for line in script.read_text().splitlines() if line.startswith("exec "))
    )[1:]
    assert command[0] == str(python)
    assert not any(str(development) in item for item in command)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["worker"] == "central"
    implementation.run_local_worker(registry, memory_gb=2, drain_seconds=0)


def test_bound_scheduler_reuses_snapshot_published_at_activation(central, monkeypatch):
    registry, root, _ = central
    record = implementation.activate(registry, root)
    snapshot = ControlPaths(registry.paths.control).implementations / record["source_digest"]
    assert snapshot.is_dir()

    def unexpected_capture(*_args, **_kwargs):
        raise AssertionError("routine scheduler selection must not recapture source")

    monkeypatch.setattr(
        "nro.orchestration.source_snapshots.SourceStore.capture", unexpected_capture
    )
    source, _, _ = implementation.capture_worker_implementation(
        registry.paths.control, registry.paths.bids_root
    )

    assert source.root == snapshot
    assert source.digest == record["source_digest"]


def test_activation_requires_quiescence_and_changed_commit_never_falls_back(central):
    registry, root, _ = central
    registry.register_worker("active", resource_class="large")
    with pytest.raises(ValueError, match="active workers"):
        implementation.activate(registry, root)
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET state='stopped'")
    implementation.activate(registry, root)
    (root / "nro/orchestration/worker.py").write_text('raise RuntimeError("changed")\n')
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "Changed source"],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="checkout changed"):
        implementation.capture_worker_implementation(
            registry.paths.control, registry.paths.bids_root
        )


def test_installation_activation_preserves_demand_and_clears_barrier(central):
    registry, root, _ = central
    registry.initialize()
    with sqlite3.connect(registry.paths.database) as db:
        db.execute(
            "INSERT INTO requests VALUES "
            "('request','user','demo',1,'anat','{}',2,NULL,'active','now','now')"
        )
        db.execute("INSERT INTO metadata VALUES ('maintenance_mode','installation')")
        db.execute("INSERT INTO metadata VALUES ('installation_checkout',?)", (str(root),))

    implementation.activate(registry, root, installation_maintenance=True)

    with registry.connection() as db:
        assert db.execute("SELECT state FROM requests WHERE id='request'").fetchone()[0] == "active"
        assert (
            db.execute("SELECT value FROM metadata WHERE key='maintenance_mode'").fetchone() is None
        )
        assert (
            db.execute("SELECT value FROM metadata WHERE key='installation_checkout'").fetchone()
            is None
        )


def test_worker_import_does_not_load_scientific_modules():
    code = (
        "import sys; import nro.orchestration.worker; "
        'prefixes=tuple("nro.modules."+n+"." for n in '
        '("anat","func","clean","dynconn","microparcellation","networks"))+("nro.modules.firstlevels.",); '
        "assert not any(m.startswith(prefixes) for m in sys.modules)"
    )
    env = {key: value for key, value in os.environ.items() if key != "NRO_SITE_CONFIG"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_branch_cannot_fall_back_to_its_own_worker_without_activation(tmp_path, monkeypatch):
    monkeypatch.setattr(implementation, "installation_record", lambda: {"mode": "branch"})
    with pytest.raises(ValueError, match="No central scheduler"):
        implementation.capture_worker_implementation(tmp_path / "control", tmp_path / "BIDS")


def test_changed_script_and_direct_wrong_worker_are_rejected(central):
    registry, root, _ = central
    implementation.activate(registry, root)
    script = _write_worker_script(
        registry,
        bids_root=registry.paths.bids_root,
        partition="test",
        account=None,
        hours=1,
        memory_gb=2,
        cpus=1,
    )
    implementation.validate_worker_script(registry.paths.control, script)
    script.write_text(script.read_text().replace("nro.orchestration.worker", "nro.modules.func"))
    with pytest.raises(ValueError, match="central implementation"):
        implementation.validate_worker_script(registry.paths.control, script)
    with pytest.raises(ValueError, match="active central implementation"):
        implementation.require_worker_source(registry.paths.control)
