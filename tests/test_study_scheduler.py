"""Run a study analysis through the real scheduler, admission, and a local worker."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nro.analysis.layout import StudySource, load_analysis
from nro.analysis.scaffold import add_analysis, init_study
from nro.definitions.store import ConfigStore
from nro.modules.analyses.planning import UpstreamRequest, plan_work_items
from nro.orchestration import scheduler_client
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.compiled_request import encode_spec, export_workflow
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.execution_cache import cache_lock
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.execution_pins import capture_site
from nro.orchestration.planning_context import work_item_key
from nro.orchestration.registry import Registry
from nro.orchestration.releases import ReleaseStore
from nro.orchestration.scheduler_implementation import activate
from nro.orchestration.source_snapshots import SourceStore
from nro.site.configuration import protected_site_fingerprint, settings

pytestmark = pytest.mark.integration

MANIFEST = """\
version: 1
scope: subject
inputs:
  runs:
    from: contrasts
    workflow: main
    select: {stat: t, suffix: statmap, extension: .tsv}
    many: true
    min: 2
outputs:
  summary: "sub-{participant}_contrast-{contrast}_desc-runs_summary.tsv"
parameters:
  contrast: {expand: [S-N]}
"""

CODE = """\
def run(inputs, params, outputs):
    rows = [{"run": item.entities["run"], "contrast": params["contrast"]} for item in inputs["runs"]]
    outputs["summary"].write_tsv(rows)
"""

PROBE = """\
from pathlib import Path


def main(argv, *, execution_context):
    logical = (
        execution_context.paths.source_project("demo")
        / "derivatives/nro/contrasts" / argv[0] / "sub-01"
    )
    root = execution_context.output_path(logical)
    root.mkdir(parents=True, exist_ok=True)
    for run in ("1", "2"):
        (root / f"sub-01_model-main_run-{run}_stat-t_statmap.tsv").write_text("value\\n1\\n")
    (root / "sub-01_model-main_desc-probe_manifest.json").write_text(
        '{"complete": true, "public_outputs": ['
        '"sub-01_model-main_run-1_stat-t_statmap.tsv", '
        '"sub-01_model-main_run-2_stat-t_statmap.tsv"]}'
    )
"""


def git(checkout, *args):
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
    )


def test_feature_branch_runs_study_analysis_after_its_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("NRO_SCHEDULER_LOCAL", "1")
    for name, value in {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }.items():
        monkeypatch.setenv(name, value)
    # Scheduler and worker subprocesses read the test's site file, so the study
    # roots must be there rather than in this process's defaults.
    studies = tmp_path / "STUDIES"
    site_file = Path(os.environ["NRO_SITE_CONFIG"])
    site_file.write_text(
        site_file.read_text()
        + f'study_code = "{studies / "CODE"}"\nstudy_results = "{studies / "RESULTS"}"\n'
    )

    root = tmp_path / "main"
    shutil.copytree(
        Path(__file__).parents[1] / "nro",
        root / "nro",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (root / "pyproject.toml").write_text('[project]\nname="nro"\nversion="0.0.1"\n')
    (root / ".gitignore").write_text(".nro-installation.json\n__pycache__/\n")
    git(root, "init", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-m", "Test release")
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    (registry.paths.bids_root / "demo" / "sub-01").mkdir(parents=True)
    branches = BranchStore(registry.paths.control)
    branches.authorize_checkout("main", root, revision=branches.initialize().revision)
    release = ReleaseStore(branches).approve(root, "0.0.1", pr="test#1", attest_merged=True)
    values = {
        **settings()[0],
        "registry": str(registry.paths.control),
        "bids": str(registry.paths.bids_root),
        "development": str(tmp_path / "DEV"),
        "study_code": str(studies / "CODE"),
        "study_results": str(studies / "RESULTS"),
    }
    site = capture_site(tmp_path / "site", values)
    (root / ".nro-installation.json").write_text(
        json.dumps(
            dict(
                mode="shared",
                ready=True,
                checkout=str(root),
                environment=str(Path(sys.executable).parent.parent),
                site=str(site),
                release=release,
            )
        )
    )
    activate(registry, root)
    feature = tmp_path / "feature"
    git(root, "worktree", "add", "-b", "feature", str(feature))
    branches.register("feature", "dev", revision=branches.read().revision, checkout=feature)
    probe = feature / "nro/probe_upstream"
    probe.mkdir()
    (probe / "__init__.py").write_text("")
    (probe / "__main__.py").write_text(PROBE)

    repository = init_study("s1", projects=("demo",))
    analysis_root = add_analysis("s1", "runs")
    (analysis_root / "analysis.yml").write_text(MANIFEST)
    (analysis_root / "analysis.py").write_text(CODE)
    git(repository, "add", "--all")
    git(repository, "commit", "-m", "Add runs")

    science = branches.registry("feature")
    registered = science.register_workflow(ConfigStore().resolve("main"))
    paths = BranchPaths(
        "feature",
        Path(values["bids"]),
        Path(values["work"]),
        Path(values["development"]),
        Path(values["study_results"]),
    )
    label = registered.directory_for("contrasts")
    upstream_root = paths.source_project("demo") / "derivatives/nro/contrasts" / label / "sub-01"
    upstream = WorkItemSpec.create(
        key=work_item_key(
            "demo",
            "contrasts",
            registered.lineage_fingerprints["contrasts"],
            "01",
            {"model": "main"},
        ),
        module="contrasts",
        project="demo",
        participant="01",
        entities={"model": "main"},
        scope="subject",
        module_lineage_id=registered.lineages["contrasts"],
        config_fingerprint="probe",
        directory_label=label,
        runtime_config=science.runtime_config_path(registered, "contrasts"),
        command=(sys.executable, "-m", "nro.probe_upstream", label),
        dependencies=(),
        input_paths=(),
        output_root=upstream_root,
        output_prefix="sub-01_model-main",
        expected_outputs=(upstream_root / "sub-01_model-main_desc-probe_manifest.json",),
        resource_class="small",
        output_format="probe-v1",
        processing={},
    )
    analysis = load_analysis(StudySource.at("s1"), "runs")
    lineage = science.register_unbound_lineage("analyses", analysis.name, {"study": "s1"})
    (terminal,), unavailable = plan_work_items(
        analysis,
        lineage=lineage,
        upstream=(UpstreamRequest("contrasts", "main", {upstream.key: upstream}),),
        bids_root=paths.bids,
    )
    assert unavailable == () and terminal.dependencies == (upstream.key,)
    graph = (upstream, terminal)
    science.record_work_item_graph(graph, expected_revisions={spec.key: None for spec in graph})
    contracts = {row.key: row.contract for row in science.work_items()}
    source = SourceStore(tmp_path / "source").capture(feature)
    payload = dict(
        protocol=1,
        branch="feature",
        registry_id=science.record.registry_id,
        project="demo",
        context=ExecutionContext(paths, "demo", terminal.key, ()).as_dict(),
        specifications=[encode_spec(spec) for spec in graph],
        revisions={spec.key: 1 for spec in graph},
        contracts=contracts,
        terminals=[terminal.key],
        inherit=True,
        workflow=export_workflow(science, None),
        source=dict(root=str(source.root), digest=source.digest),
        site=str(site),
        site_fingerprint=protected_site_fingerprint(Path(values["definitions"])),
        python=sys.executable,
        selectors={"study_analysis": analysis.name},
        concurrency=1,
        partition="test",
        release=None,
    )
    with cache_lock(registry.paths.control):
        result = scheduler_client.exchange(
            scheduler_client.command(registry.paths.control, paths.bids),
            dict(
                operation="admit_many",
                checkout=str(feature),
                entries=[dict(project="demo", payload=payload)],
            ),
        )
    scheduler_client.supply(
        registry.paths.control,
        paths.bids,
        checkout=feature,
        request_ids=result["request_ids"],
        options={
            "local": True,
            "memory": 32,
            "max_memory": 128,
            "cpus": 2,
            "drain_minutes": 0,
            "worker_poll_interval": 0.01,
        },
    )
    report = scheduler_client.status(
        registry.paths.control, paths.bids, checkout=feature, mode="verify"
    )
    statuses = {row["module"]: row["status"] for row in report["rows"]}
    assert statuses == {"contrasts": "Success", "analyses": "Success"}, report["rows"]
    logical = terminal.output_root / "sub-01_contrast-SN_desc-runs_summary.tsv"
    published = paths.output_studies / logical.relative_to(paths.studies)
    assert not logical.exists()
    assert published.read_text().splitlines() == ["run\tcontrast", "1\tS-N", "2\tS-N"]
    assert published.with_name(published.name + ".provenance.json").is_file()
    with registry.connection() as db:
        (workflow_revision,) = db.execute(
            "SELECT workflow_revision_id FROM requests WHERE id=?", (result["request_ids"][0],)
        ).fetchone()
    assert workflow_revision is None
    receipts = list((published.parents[1] / ".nro" / "work_items" / "analyses").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["module"] == "analyses"

    (feature / ".nro-installation.json").write_text(
        json.dumps(
            dict(
                mode="branch",
                ready=True,
                checkout=str(feature),
                environment=str(Path(sys.executable).parent.parent),
                site=str(site),
                branch="feature",
                registry_id=science.record.registry_id,
            )
        )
    )
    environment = {**os.environ, "NRO_SITE_CONFIG": str(site), "PYTHONPATH": str(feature)}
    status = subprocess.run(
        [sys.executable, "-m", "nro.bin.status", "--json"],
        cwd=feature,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(status.stdout)["work_items"]
    assert {row["module"]: row["status"] for row in rows}["analyses"] == "Success"
    study_status = subprocess.run(
        [sys.executable, "-m", "nro.bin.status", "--json", "--study", "s1/runs"],
        cwd=feature,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert [row["module"] for row in json.loads(study_status.stdout)["work_items"]] == ["analyses"]
    stray = published.with_name("sub-01_desc-stray_summary.tsv")
    stray.write_text("old\n")
    orphan = paths.output_studies / "s1/analyses/deleted/project-demo/sub-01/x.tsv"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("old\n")
    gc = subprocess.run(
        [sys.executable, "-m", "nro.bin.gc", "--force", "--json"],
        cwd=feature,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert gc.returncode == 0, gc.stderr
    assert set(json.loads(gc.stdout)["public_paths"]) == {str(stray), str(orphan)}
    assert not stray.exists() and not orphan.exists()
    assert published.is_file() and receipts[0].is_file()
    purge = subprocess.run(
        [sys.executable, "-m", "nro.bin.purge", "--force", "--json", "--module", "contrasts"],
        cwd=feature,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert purge.returncode == 0, purge.stderr
    assert published.is_file(), "purging upstream outputs must not delete study results"
    status = subprocess.run(
        [sys.executable, "-m", "nro.bin.status", "--json"],
        cwd=feature,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(status.stdout)["work_items"]
    assert {row["module"]: row["status"] for row in rows}.get("analyses") != "Success"
    purge = subprocess.run(
        [sys.executable, "-m", "nro.bin.purge", "--force", "--json"],
        cwd=feature,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert purge.returncode == 0, purge.stderr
    assert not published.exists() and not receipts[0].exists()
