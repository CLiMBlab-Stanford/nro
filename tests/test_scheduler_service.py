"""Exercise the JSON handoff through the actual central interpreter and source."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nro.configuration.site import settings
from nro.configuration.store import ConfigStore
from nro.orchestration import scheduler_client
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.compiled_request import encode_spec, export_workflow
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.execution_cache import cache_lock
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.execution_pins import capture_site
from nro.orchestration.registry import Registry
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.scheduler_implementation import activate
from nro.orchestration.source_snapshots import SourceStore

pytestmark = pytest.mark.integration


def git(checkout, *args):
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
    )


def test_real_service_admits_runs_and_reports_foreign_catalog(tmp_path):
    root = tmp_path / "main"
    shutil.copytree(
        Path(__file__).parents[1] / "nro",
        root / "nro",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (root / "pyproject.toml").write_text('[project]\nname="nro"\nversion="0.0.1"\n')
    (root / ".gitignore").write_text(".nro-installation.json\n__pycache__/\n")
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test Maintainer")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "Test release")
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    branches = BranchStore(registry.paths.control)
    branches.authorize_checkout("main", root, revision=branches.initialize().revision)
    ReleaseStore(branches).approve(root, "0.0.1", pr="test#1", attest_merged=True)
    values = {
        **settings()[0],
        "registry": str(registry.paths.control),
        "bids": str(registry.paths.bids_root),
        "work": str(tmp_path / "WORK"),
        "development": str(tmp_path / "DEV"),
    }
    site = capture_site(tmp_path / "site", values)
    environment = tmp_path / "env"
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
            )
        )
    )
    activate(registry, root)
    feature = tmp_path / "feature"
    git(root, "worktree", "add", "-b", "feature", str(feature))
    branches.register("feature", "dev", revision=branches.read().revision, checkout=feature)
    module = feature / "nro/probe_extension"
    module.mkdir()
    (module / "__init__.py").write_text("")
    (module / "__main__.py").write_text(
        "from pathlib import Path\n"
        "def main(argv, *, execution_context):\n"
        "    out = execution_context.output_path(Path(argv[0]))\n"
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        '    out.write_text("feature result")\n'
    )
    science = branches.registry("feature")
    registered = science.register_workflow(ConfigStore().resolve("main"))
    paths = BranchPaths(
        "feature", Path(values["bids"]), Path(values["work"]), Path(values["development"])
    )
    output = (
        paths.source_project("demo") / "derivatives/preprocessing/main/sub-01/sub-01_result.txt"
    )
    spec = InstanceSpec.create(
        key="extension",
        module="probe_extension",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=registered.lineages["preprocessing"],
        config_fingerprint="science",
        directory_label="main",
        runtime_config=science.runtime_config_path(registered, "preprocessing"),
        command=(sys.executable, "-m", "nro.probe_extension", str(output)),
        dependencies=(),
        input_paths=(),
        output_root=output.parent,
        output_prefix="sub-01",
        expected_outputs=(output,),
        resource_class="small",
        output_format="probe-v1",
        processing={},
    )
    science.record_graph((spec,), expected_revisions={spec.key: None})
    source = SourceStore(tmp_path / "source").capture(feature)
    payload = dict(
        protocol=1,
        branch="feature",
        registry_id=science.record.registry_id,
        project="demo",
        context=ExecutionContext(paths, "demo", spec.key, ()).as_dict(),
        specifications=[encode_spec(spec)],
        revisions={spec.key: 1},
        terminals=[spec.key],
        inherit=True,
        workflow=export_workflow(science, registered),
        source=dict(root=str(source.root), digest=source.digest),
        site=str(site),
        python=sys.executable,
        selectors={},
        concurrency=1,
        partition="test",
        release=None,
    )
    with cache_lock(registry.paths.control):
        result = scheduler_client.exchange(
            scheduler_client.command(registry.paths.control, paths.bids),
            dict(operation="admit", project="demo", checkout=str(feature), payload=payload),
        )
    scheduler_client.supply(
        registry.paths.control,
        paths.bids,
        checkout=feature,
        request_ids=[result["request_id"]],
        options={"local": True, "memory": 32, "drain_minutes": 0, "worker_poll_interval": 0.01},
    )
    report = scheduler_client.status(
        registry.paths.control, paths.bids, checkout=feature, mode="verify"
    )
    assert len(report["visible_ids"]) == 1
    assert report["rows"][0]["status"] == "Success"
    assert not output.exists()
    assert (
        paths.output_project("demo") / output.relative_to(paths.source_project("demo"))
    ).read_text() == "feature result"
    assert not (root / "nro/probe_extension").exists()
    (feature / ".nro-installation.json").write_text(
        json.dumps(
            dict(
                mode="branch",
                ready=True,
                checkout=str(feature),
                environment=str(Path(sys.executable).parent.parent),
                site=str(site),
                branch="feature",
            )
        )
    )
    command = [sys.executable, "-m", "nro.bin.status", "--json"]
    result = subprocess.run(
        command,
        cwd=feature,
        env={**os.environ, "NRO_SITE_CONFIG": str(site), "PYTHONPATH": str(feature)},
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["instances"][0]["status"] == "Success"
    output.parent.mkdir(parents=True)
    output.write_text("main result")
    command = [sys.executable, "-m", "nro.bin.purge", "--force", "--json"]
    result = subprocess.run(
        command,
        cwd=feature,
        env={**os.environ, "NRO_SITE_CONFIG": str(site), "PYTHONPATH": str(feature)},
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["instances"] == 1
    assert output.read_text() == "main result"
    assert not (
        paths.output_project("demo") / output.relative_to(paths.source_project("demo"))
    ).exists()
    with sqlite3.connect(science.database) as db:
        db.execute("PRAGMA user_version=999")
    command = [sys.executable, "-m", "nro.bin.run", "--repair", "--json"]
    result = subprocess.run(
        command,
        cwd=feature,
        env={**os.environ, "NRO_SITE_CONFIG": str(site), "PYTHONPATH": str(feature)},
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["repaired"]
    assert len(science.instances()) == 1
    assert (science.root / "registry-before-repair.sqlite3").is_file()
    from nro.bidsify.config import load_config
    from nro.bidsify.store import IngestionStore

    (feature / "nro/bidsify/stages.py").write_text(
        "def run_stage(record, registry, *, source=None, branch_paths=None):\n"
        '    return dict(state="awaiting_approval", seen_branch=branch_paths.branch,\n'
        '                seen_root=str(branch_paths.output_project(record["project"])))\n'
    )
    (feature / "nro/orchestration/registry.py").write_text(
        'raise RuntimeError("development ingestion imported its Registry")\n'
    )
    ingestion_source = SourceStore(tmp_path / "ingestion-source").capture(feature)
    pin = dict(
        branch="feature",
        registry_id=science.record.registry_id,
        checkout=str(feature),
        source_root=str(ingestion_source.root),
        source_digest=ingestion_source.digest,
        site=str(site),
        python=sys.executable,
        release=None,
        command_prefix=list(
            ingestion_source.command((sys.executable, "-m", "nro.bidsify"), site=site)
        ),
    )
    debug = IngestionStore(registry, branch_paths=paths, execution=pin)
    config = load_config()
    config.update(concurrency=1, staging=str(tmp_path / "ingestion-staging"))
    record = debug.create(
        server="cni", remote_session="test-session", project="demo", config=config
    )
    scheduler_client.supply(
        registry.paths.control,
        paths.bids,
        checkout=feature,
        request_ids=[],
        options={"local": True, "memory": 32, "drain_minutes": 0, "worker_poll_interval": 0.01},
    )
    completed = debug.get(record["id"])
    assert completed["state"] == "awaiting_approval"
    assert completed["seen_branch"] == "feature"
    assert completed["seen_root"] == str(paths.output_project("demo"))
    assert not IngestionStore(registry).rows()
