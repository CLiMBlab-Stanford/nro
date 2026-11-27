"""Study analyses: manifests, input resolution, planning, execution, and commands."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import nro.site.configuration as site
from nro.analysis import InputSet, ResolutionError, query
from nro.analysis.artifacts import file_entities
from nro.analysis.execution import execute
from nro.analysis.layout import StudySource, load_analysis
from nro.analysis.manifest import ManifestError, parse_manifest
from nro.analysis.planning import plan_analysis
from nro.analysis.scaffold import add_analysis, init_study
from nro.bin import study as study_command
from nro.engine.cifti import write_indexed_cifti_sidecar

MANIFEST = """\
version: 1
scope: subject
inputs:
  maps:
    from: contrasts
    workflow: main
    select: {model: main, task: lang, node: run, stat: t, suffix: statmap}
    many: true
    min: 2
outputs:
  similarity: "sub-{participant}_contrast-{contrast}_desc-betweenrun_similarity.tsv"
parameters:
  contrast: {expand: [S-N, W-N]}
  exclude_medial_wall: true
"""

CODE = """\
import numpy as np
from nro.analysis import Inputs, Outputs


def run(inputs: Inputs, params: dict, outputs: Outputs) -> None:
    maps = {a.entities["run"]: a.load().map(Contrast=params["contrast"]) for a in inputs["maps"]}
    runs = sorted(maps)
    rows = []
    for i, first in enumerate(runs):
        for second in runs[i + 1 :]:
            r = np.corrcoef(maps[first], maps[second])[0, 1]
            rows.append({"run_a": first, "run_b": second, "z": float(np.arctanh(r))})
    outputs["similarity"].write_tsv(rows)
"""


class FakeIndex:
    """Upstream index with one contrasts lineage per workflow."""

    def __init__(self, items, lineages=None):
        self.items = items
        self.rows = lineages or [
            {"directory_label": "main-abc", "config_id": "main", "workflow_id": "main"}
        ]

    def lineages(self, project, module, *, workflow, config):
        return [
            row
            for row in self.rows
            if (workflow is None or row["workflow_id"] == workflow)
            and (config is None or row["config_id"] == config)
        ]

    def work_items(self, project, module, directory_label, participant):
        return [item for item in self.items if item["participant"] == participant]


def _cifti(path: Path, values: np.ndarray) -> None:
    axis = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(values.shape[1]), values.shape[1], "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    names = ["S-N", "W-N"]
    header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(names), axis))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Cifti2Image(values.astype(np.float32), header=header), str(path))
    write_indexed_cifti_sidecar(
        path, [{"Name": name, "Contrast": name} for name in names], lookup_fields=("Contrast",)
    )


@pytest.fixture
def study_roots(tmp_path, monkeypatch):
    monkeypatch.setitem(site.DEFAULTS, "study_code", str(tmp_path / "code"))
    monkeypatch.setitem(site.DEFAULTS, "study_results", str(tmp_path / "results"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.org")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.org")
    return tmp_path


@pytest.fixture
def upstream(tmp_path):
    bids = tmp_path / "bids"
    (bids / "p1" / "sub-01").mkdir(parents=True)
    (bids / "p1" / "sub-02").mkdir(parents=True)
    root = bids / "p1" / "derivatives" / "nro" / "contrasts" / "main-abc" / "sub-01"
    prefix = "sub-01_model-main_task-lang_space-fsLR32k_smoothing-0mm"
    rng = np.random.default_rng(0)
    for run in (1, 2, 3):
        _cifti(
            root
            / "task-lang"
            / "node-run"
            / f"{prefix}_node-run_run-{run}_stat-t_statmap.dscalar.nii",
            rng.normal(size=(2, 50)),
        )
    items = [
        {
            "participant": "01",
            "entities": {"model": "lang/main", "space": "fsLR32k"},
            "artifact_state": "fresh",
            "artifact_fingerprint": "f1",
            "generation": 1,
            "output_root": str(root),
            "output_prefix": prefix,
            "expected_outputs": [],
            "key": "k1",
        },
        {
            "participant": "02",
            "entities": {"model": "lang/main", "space": "fsLR32k"},
            "artifact_state": "missing",
            "artifact_fingerprint": "f2",
            "generation": 0,
            "output_root": str(bids / "nowhere"),
            "output_prefix": None,
            "expected_outputs": [],
            "key": "k2",
        },
    ]
    return bids, FakeIndex(items)


def _commit(repository: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(repository), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", message], check=True)


def _study(study_roots) -> Path:
    repository = init_study("s1", projects=("p1",))
    directory = add_analysis("s1", "similarity")
    (directory / "analysis.yml").write_text(MANIFEST)
    (directory / "analysis.py").write_text(CODE)
    _commit(repository, "Add similarity")
    return repository


def test_expansion_labels_and_literal_lists():
    manifest = parse_manifest(MANIFEST, analysis="similarity")
    combinations = manifest.combinations()
    assert [labels for _, labels in combinations] == [{"contrast": "SN"}, {"contrast": "WN"}]
    assert combinations[0][0] == {"contrast": "S-N", "exclude_medial_wall": True}
    listed = parse_manifest(
        MANIFEST.replace(
            "exclude_medial_wall: true", "pairs: {expand: {SN: [S, N]}}\n  rois: [A, B]"
        ).replace("{contrast}_desc", "{contrast}_pairs-{pairs}_desc"),
        analysis="similarity",
    )
    parameters, labels = listed.combinations()[0]
    assert parameters["pairs"] == ["S", "N"] and parameters["rois"] == ["A", "B"]
    assert labels["pairs"] == "SN"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("{contrast}_desc", "desc"), "every expanded parameter"),
        (("[S-N, W-N]", "[S-N, SN]"), "share the BIDS labels"),
        (("workflow: main", "workflow: main\n    config: main"), "both workflow and config"),
        (("from: contrasts", "from: lineage"), "unknown module"),
        (("scope: subject", "scope: study"), "not supported yet"),
        (("from: contrasts", "from: analysis:s1/other"), "same study, as analysis:NAME"),
        (("from: contrasts", "from: analysis:other"), "takes no workflow"),
        (("from: contrasts\n    workflow: main", "from: analysis:similarity"), "own outputs"),
        (("scope: subject", "scope: subject\nparticipants: {match: '*'}"), "only to scope"),
        (("[S-N, W-N]", "[[S, N]]"), "non-scalar"),
    ],
)
def test_manifest_rejections(change, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(MANIFEST.replace(*change), analysis="similarity")


def test_file_entities_parse_compound_extensions():
    entities = file_entities("sub-01_task-lang_run-1_stat-t_statmap.dscalar.nii")
    assert entities["suffix"] == "statmap" and entities["extension"] == ".dscalar.nii"
    assert entities["run"] == "1"


def test_query_resolves_fresh_contract_outputs_without_sidecars(upstream):
    bids, index = upstream
    maps = query(
        project="p1",
        participant="sub-01",
        module="contrasts",
        select={"stat": "t", "node": "run"},
        many=True,
        index=index,
        bids_root=bids,
    )
    assert isinstance(maps, InputSet) and len(maps) == 3
    assert {artifact.entities["run"] for artifact in maps} == {"1", "2", "3"}
    assert maps[0].load().map(Contrast="W-N").shape == (50,)
    with pytest.raises(ResolutionError, match="3 files match"):
        query(
            project="p1",
            participant="01",
            module="contrasts",
            index=index,
            bids_root=bids,
            select={"stat": "t"},
        )
    with pytest.raises(ResolutionError, match="not fresh"):
        query(
            project="p1",
            participant="02",
            module="contrasts",
            index=index,
            bids_root=bids,
            many=True,
        )
    with pytest.raises(ResolutionError, match="no registered contrasts lineage"):
        query(
            project="p1",
            participant="01",
            module="contrasts",
            workflow="other",
            index=index,
            bids_root=bids,
            many=True,
        )


def test_config_input_rejects_ambiguous_lineages(upstream):
    bids, index = upstream
    index.rows = [
        {"directory_label": "main-abc", "config_id": "main", "workflow_id": "main"},
        {"directory_label": "main-def", "config_id": "main", "workflow_id": "lesion"},
    ]
    with pytest.raises(ResolutionError, match="workflows lesion, main"):
        query(
            project="p1",
            participant="01",
            module="contrasts",
            config="main",
            index=index,
            bids_root=bids,
            many=True,
        )


def test_plan_run_and_freshness(study_roots, upstream):
    bids, index = upstream
    _study(study_roots)
    analysis = load_analysis(StudySource.at("s1"), "similarity")
    items = plan_analysis(analysis, index=index, bids_root=bids)
    assert [(item.participant, item.status) for item in items] == [
        ("01", "ready"),
        ("01", "ready"),
        ("02", "waiting"),
        ("02", "waiting"),
    ]
    first = items[0]
    assert first.outputs["similarity"] == (
        study_roots
        / "results/s1/analyses/similarity/project-p1/sub-01"
        / "sub-01_contrast-SN_desc-betweenrun_similarity.tsv"
    )
    result = execute(first)
    assert result.succeeded, result.log.read_text()
    table = first.outputs["similarity"].read_text().splitlines()
    assert table[0] == "run_a\trun_b\tz" and len(table) == 4
    provenance = json.loads(
        first.outputs["similarity"]
        .with_name(first.outputs["similarity"].name + ".provenance.json")
        .read_text()
    )
    assert provenance["Commit"] == analysis.source.commit

    replanned = plan_analysis(analysis, index=index, bids_root=bids)
    assert [item.status for item in replanned[:2]] == ["fresh", "ready"]

    index.items[0]["generation"] = 2
    assert plan_analysis(analysis, index=index, bids_root=bids)[0].status == "ready"


def test_commit_to_another_analysis_keeps_results_fresh(study_roots, upstream):
    bids, index = upstream
    repository = _study(study_roots)
    analysis = load_analysis(StudySource.at("s1"), "similarity")
    assert execute(plan_analysis(analysis, index=index, bids_root=bids)[0]).succeeded
    add_analysis("s1", "other")
    _commit(repository, "Add other")
    later = load_analysis(StudySource.at("s1"), "similarity")
    assert later.source.commit != analysis.source.commit
    assert plan_analysis(later, index=index, bids_root=bids)[0].status == "fresh"
    (repository / "analyses/similarity/analysis.py").write_text(CODE + "\n# edit\n")
    _commit(repository, "Edit similarity")
    edited = load_analysis(StudySource.at("s1"), "similarity")
    assert plan_analysis(edited, index=index, bids_root=bids)[0].status == "ready"


def test_unwritten_slot_fails_and_publishes_nothing(study_roots, upstream):
    bids, index = upstream
    repository = _study(study_roots)
    (repository / "analyses/similarity/analysis.py").write_text(
        "def run(inputs, params, outputs):\n    pass\n"
    )
    _commit(repository, "Break")
    analysis = load_analysis(StudySource.at("s1"), "similarity")
    item = plan_analysis(analysis, index=index, bids_root=bids)[0]
    result = execute(item)
    assert not result.succeeded and "similarity" in result.message
    assert not item.outputs["similarity"].exists()


def test_uncommitted_changes_are_ignored_by_planning(study_roots):
    repository = _study(study_roots)
    (repository / "analyses/similarity/analysis.yml").write_text("version: [")
    assert load_analysis(StudySource.at("s1"), "similarity").manifest.version == 1
    with pytest.raises(ManifestError):
        load_analysis(StudySource.at("s1", None), "similarity")


def test_study_commands(study_roots, upstream, capsys, monkeypatch):
    bids, index = upstream
    monkeypatch.setattr(site, "bids_root", lambda: bids)
    monkeypatch.setattr("nro.analysis.resolve.default_index", lambda: index)
    _study(study_roots)
    study_command.main(["validate", "s1"])
    study_command.main(["list"])
    study_command.main(["check", "s1/similarity", "contrast=S-N"])
    output = capsys.readouterr().out
    assert "s1/similarity: valid" in output
    assert "ready" in output and "contrast-WN" not in output
    study_command.main(["try", "s1/similarity", "contrast=SN", "--participant", "01"])
    assert "succeeded" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        study_command.main(["run", "s1/similarity"])
    output = capsys.readouterr().out
    assert "2 published" in output and "2 blocked" in output
    study_command.main(["run", "s1/similarity", "--participant", "01"])
    assert "0 published, 0 failed, 2 fresh" in capsys.readouterr().out


def test_analysis_entry_is_checked_without_import(study_roots):
    repository = _study(study_roots)
    (repository / "analyses/similarity/analysis.py").write_text(
        "raise SystemExit\ndef other(): pass\n"
    )
    _commit(repository, "Rename")
    with pytest.raises(ManifestError, match="does not define a top-level function 'run'"):
        load_analysis(StudySource.at("s1"), "similarity")


def test_registry_index_reads_registered_lineages_and_work_items(tmp_path):
    from nro.definitions.store import ConfigStore
    from nro.orchestration.contracts import WorkItemSpec
    from nro.orchestration.registry import Registry

    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "output"
    registry.register_work_items(
        (
            WorkItemSpec.create(
                key="contrasts:" + "c" * 64,
                module="contrasts",
                project="demo",
                participant="01",
                entities={"model": "lang/main"},
                scope="subject",
                module_lineage_id=registered.lineages["contrasts"],
                config_fingerprint=workflow.configuration("contrasts").fingerprint,
                directory_label=registered.directory_for("contrasts"),
                runtime_config=registry.runtime_config_path(registered, "contrasts"),
                command=("python", "-m", "nro.modules.contrasts"),
                dependencies=(),
                input_paths=(),
                output_root=output,
                output_prefix="sub-01_model-main",
                resource_class="medium",
                expected_outputs=(output / "manifest.json",),
            ),
        )
    )
    label = registered.directory_for("contrasts")
    config_id = workflow.configuration("contrasts").config_id
    assert registry.module_lineage_directories("contrasts", workflow_id="main") == [
        {"directory_label": label, "config_id": config_id, "workflow_id": "main"}
    ]
    assert [
        row["workflow_id"]
        for row in registry.module_lineage_directories("contrasts", config_id=config_id)
    ] == ["main"]
    (row,) = registry.lineage_work_items("demo", "contrasts", label, "01")
    assert row["output_prefix"] == "sub-01_model-main"
    assert row["output_root"] == str(output.resolve())
    assert row["entities"] == {"model": "lang/main"}
    assert registry.lineage_work_items("demo", "contrasts", label, "02") == []


def test_snapshot_index_matches_workflow_and_config(upstream):
    from nro.analysis.resolve import SnapshotIndex

    bids, fake = upstream
    item = fake.items[0]
    rows = [
        {
            "project": "p1",
            "module": "contrasts",
            "participant": "01",
            "directory_label": "main-abc",
            "config_id": "main",
            "workflow_ids": "main,lesion",
            "entities_json": json.dumps(item["entities"]),
            "artifact_state": "fresh",
            "artifact_fingerprint": "f1",
            "current_generation": 1,
            "output_root": item["output_root"],
            "output_prefix": item["output_prefix"],
            "work_item_key": "k1",
        }
    ]
    index = SnapshotIndex(rows)
    maps = query(
        project="p1",
        participant="01",
        module="contrasts",
        workflow="lesion",
        select={"stat": "t"},
        many=True,
        index=index,
        bids_root=bids,
    )
    assert len(maps) == 3
    assert [
        row["workflow_id"]
        for row in index.lineages("p1", "contrasts", workflow=None, config="main")
    ] == ["lesion", "main"]
    with pytest.raises(ResolutionError, match="no registered"):
        query(
            project="p1",
            participant="01",
            module="contrasts",
            workflow="other",
            many=True,
            index=index,
            bids_root=bids,
        )


def _upstream_spec(root: Path, prefix: str, *, key: str = "contrasts:" + "a" * 64):
    from nro.orchestration.contracts import WorkItemSpec

    return WorkItemSpec.create(
        key=key,
        module="contrasts",
        project="p1",
        participant="01",
        entities={"model": "main", "task": "lang", "space": "fsLR32k", "smoothing": "0"},
        scope="subject",
        module_lineage_id=1,
        config_fingerprint="c",
        directory_label="main-abc",
        runtime_config=root / "runtime.yml",
        command=("python", "-m", "nro.modules.contrasts"),
        dependencies=(),
        input_paths=(),
        output_root=root,
        output_prefix=prefix,
        resource_class="medium",
        output_format="maps",
        expected_outputs=(root / "manifest.json",),
    )


class _StubPlanner:
    def __init__(self, bids, spec):
        self.bids_root = bids
        self._source_digest = "d" * 64
        self.spec = spec
        self.calls = []

    def plan(self, **kwargs):
        from nro.orchestration.planner import PlanningResult

        self.calls.append(kwargs)
        return PlanningResult((), {self.spec.key: self.spec}, {}, frozenset(), (), "d", {})


class _StubRegistry:
    def register_unbound_lineage(self, configuration_class, config_id, resolved):
        import hashlib

        from nro.orchestration.workflow_registry import UnboundLineage

        digest = hashlib.sha256(config_id.encode()).hexdigest()
        label = config_id.replace("/", ".") + "-x"
        return UnboundLineage(7, configuration_class, config_id, digest, label, True)

    def register_workflow(self, workflow):
        return object()


class _Store:
    def resolve(self, workflow_id):
        return workflow_id


def test_scheduled_plan_depends_on_upstream_and_runs_from_bindings(study_roots, upstream):
    from nro.modules.analyses import __main__ as scheduled
    from nro.orchestration.study_planning import plan_request

    bids, index = upstream
    _study(study_roots)
    analysis = load_analysis(StudySource.at("s1"), "similarity")
    item = index.items[0]
    spec = _upstream_spec(Path(item["output_root"]), item["output_prefix"])
    planner = _StubPlanner(bids, spec)
    plan = plan_request(planner, _StubRegistry(), analysis, _Store(), participants=("01",))

    (request,) = plan.requests
    assert request.workflow_id is None and request.registered is None
    assert planner.calls[0]["modules"] == ("contrasts",)
    terminals = [plan.work_items[key] for key in request.terminal_keys]
    assert [item.entities for item in terminals] == [{"contrast": "SN"}, {"contrast": "WN"}]
    first = terminals[0]
    assert first.dependencies == (spec.key,) and spec.key in plan.work_items
    assert first.contract.processing["analysis"]["commit"] == analysis.source.commit
    assert first.expected_outputs[0].name == (
        "sub-01_analysis-similarity_contrast-SN_manifest.json"
    )

    specification = json.loads(Path(first.runtime_config).read_text())
    # A recovered work item can outlive its extracted code; the commit is extracted again.
    shutil.rmtree(specification["code_root"])
    scheduled.run(specification, work_dir=study_roots / "attempt")
    output = Path(specification["outputs"]["similarity"])
    assert len(output.read_text().splitlines()) == 4
    completion = json.loads(first.expected_outputs[0].read_text())
    assert completion["complete"] is True and output.name in completion["public_outputs"]

    broken = {
        **specification,
        "inputs": {"maps": {**specification["inputs"]["maps"], "upstream": []}},
    }
    with pytest.raises(RuntimeError, match="fewer than min 2"):
        scheduled.run(broken, work_dir=study_roots / "attempt2")


def test_analyses_module_is_not_a_workflow_module():
    from nro.orchestration.catalog import BUILTIN_MODULES, module_descriptor, normalize_module

    assert module_descriptor("analyses").dynamic_processing_keys == ("analysis",)
    assert "analyses" not in {item.name for item in BUILTIN_MODULES}
    with pytest.raises(ValueError, match="--study-analysis"):
        normalize_module("analyses")


def _analysis_lineage(config_id: str) -> dict:
    from nro.orchestration.workflow_registry import (
        lineage_directory_label,
        unbound_lineage_fingerprint,
    )

    lineage = unbound_lineage_fingerprint("analyses", config_id)
    return {
        "configuration_class": "analyses",
        "configuration": {"id": config_id, "fingerprint": "x", "resolved": {}},
        "lineage_fingerprint": lineage,
        "directory_label": lineage_directory_label(config_id, lineage),
        "upstream": [],
    }


def test_analysis_lineage_record_must_match_its_directory(tmp_path):
    from nro.orchestration.ownership import _validate_analysis_lineage

    marker = tmp_path / "s1/analyses/runs/project-p1/.nro/lineage.json"
    record = _analysis_lineage("s1/runs")
    assert record["directory_label"].startswith("s1.runs-")
    assert _validate_analysis_lineage(record, marker, record, None) is record
    moved = tmp_path / "s1/analyses/other/project-p1/.nro/lineage.json"
    with pytest.raises(ValueError, match="expected s1/other"):
        _validate_analysis_lineage(record, moved, record, None)
    chained = {**record, "upstream": [{"lineage_fingerprint": "parent"}]}
    with pytest.raises(ValueError, match="no upstream lineage"):
        _validate_analysis_lineage(chained, marker, chained, None)


GROUP_MANIFEST = """\
version: 1
scope: project
participants: {match: "*", pending: exclude}
inputs:
  similarity:
    from: analysis:similarity
    select: {contrast: SN}
outputs:
  summary: "project-{project}_desc-group_summary.json"
"""

GROUP_CODE = """\
def run(inputs, params, outputs):
    rows = {a.participant: len(a.load()) for a in inputs["similarity"]}
    outputs["summary"].write_json(
        {"rows": rows, "participants": params["participants"],
         "excluded": params["excluded_participants"]}
    )
"""


def _group_study(study_roots, manifest=GROUP_MANIFEST) -> Path:
    repository = _study(study_roots)
    directory = add_analysis("s1", "group", scope="project")
    (directory / "analysis.yml").write_text(manifest)
    (directory / "analysis.py").write_text(GROUP_CODE)
    _commit(repository, "Add group")
    return repository


def test_project_scope_reads_analysis_outputs_and_enumerates_participants(study_roots, upstream):
    bids, index = upstream
    _group_study(study_roots)
    group = load_analysis(StudySource.at("s1"), "group")
    similarity = load_analysis(StudySource.at("s1"), "similarity")

    (item,) = plan_analysis(group, index=index, bids_root=bids)
    assert item.key == "s1/group/project-p1" and item.participant is None
    assert item.status == "not_applicable" and item.excluded == ("01", "02")

    upstream_item = plan_analysis(similarity, index=index, bids_root=bids)[0]
    assert execute(upstream_item).succeeded
    (item,) = plan_analysis(group, index=index, bids_root=bids)
    assert item.status == "ready", item.message
    assert item.participants == ("01",) and item.excluded == ("02",)
    assert item.outputs["summary"] == (
        study_roots / "results/s1/analyses/group/project-p1/project-p1_desc-group_summary.json"
    )
    result = execute(item)
    assert result.succeeded, result.log.read_text()
    summary = json.loads(item.outputs["summary"].read_text())
    assert summary == {"rows": {"01": 3}, "participants": ["01"], "excluded": ["02"]}
    assert plan_analysis(group, index=index, bids_root=bids)[0].status == "fresh"

    # A new matching participant changes the enumerated set, even without inputs.
    (bids / "p1" / "sub-03").mkdir()
    (item,) = plan_analysis(group, index=index, bids_root=bids)
    assert item.status == "ready" and item.excluded == ("02", "03")

    # Rerunning the upstream analysis makes the group result stale too.
    shutil.rmtree(bids / "p1" / "sub-03")
    assert plan_analysis(group, index=index, bids_root=bids)[0].status == "fresh"
    index.items[0]["generation"] = 2
    # With pending: exclude, the stale upstream leaves no participant.
    assert plan_analysis(group, index=index, bids_root=bids)[0].status == "not_applicable"

    with pytest.raises(ValueError, match="project scope"):
        plan_analysis(group, index=index, bids_root=bids, participants=["01"])


def test_project_scope_waits_for_pending_participants_by_default(study_roots, upstream):
    bids, index = upstream
    manifest = GROUP_MANIFEST.replace(", pending: exclude", ', exclude: ["01"]')
    _group_study(study_roots, manifest)
    group = load_analysis(StudySource.at("s1"), "group")
    (item,) = plan_analysis(group, index=index, bids_root=bids)
    assert item.status == "waiting" and "sub-02" in item.message
    assert "01" not in item.participants + item.excluded


def test_project_scope_manifest_rules():
    with pytest.raises(ManifestError, match="covers many participants"):
        parse_manifest(
            GROUP_MANIFEST.replace("project-{project}_", "sub-{participant}_"), analysis="group"
        )
    with pytest.raises(ManifestError, match="reserved in project scope"):
        parse_manifest(GROUP_MANIFEST + "parameters: {participants: 1}\n", analysis="group")
    with pytest.raises(ManifestError, match="pending must be"):
        parse_manifest(GROUP_MANIFEST.replace("exclude}", "later}"), analysis="group")
    manifest = parse_manifest(GROUP_MANIFEST.replace('"*"', '"re:0[13]"'), analysis="group")
    assert manifest.participants.select(["01", "02", "03"]) == ("01", "03")


def test_scheduled_project_scope_reads_a_planned_analysis(study_roots, upstream):
    from nro.modules.analyses import __main__ as scheduled
    from nro.orchestration.study_planning import plan_request

    bids, index = upstream
    _group_study(study_roots)
    group = load_analysis(StudySource.at("s1"), "group")
    item = index.items[0]
    spec = _upstream_spec(Path(item["output_root"]), item["output_prefix"])
    plan = plan_request(_StubPlanner(bids, spec), _StubRegistry(), group, _Store())

    (request,) = plan.requests
    (terminal,) = [plan.work_items[key] for key in request.terminal_keys]
    assert terminal.is_project_scope and terminal.participant == ""
    assert request.participants == ("01",)
    contract = terminal.contract.processing["analysis"]
    assert contract["participants"] == ["01"] and contract["excluded_participants"] == ["02"]
    (dependency,) = terminal.dependencies
    upstream_item = plan.work_items[dependency]
    assert upstream_item.module == "analyses" and upstream_item.participant == "01"
    assert upstream_item.entities == {"contrast": "SN"}
    assert upstream_item.dependencies == (spec.key,)
    assert terminal.output_root.name == "project-p1"
    assert terminal.expected_outputs[0].name == "project-p1_analysis-group_manifest.json"

    # Run the upstream analysis, then the group item, from their specifications.
    upstream_specification = json.loads(Path(upstream_item.runtime_config).read_text())
    scheduled.run(upstream_specification, work_dir=study_roots / "attempt-upstream")
    specification = json.loads(Path(terminal.runtime_config).read_text())
    scheduled.run(specification, work_dir=study_roots / "attempt-group")
    summary = json.loads(Path(specification["outputs"]["summary"]).read_text())
    assert summary == {"rows": {"01": 3}, "participants": ["01"], "excluded": ["02"]}

    # A new matching participant changes the contract, so the item is stale.
    (bids / "p1" / "sub-03").mkdir()
    replanned = plan_request(_StubPlanner(bids, spec), _StubRegistry(), group, _Store())
    (again,) = [replanned.work_items[key] for key in replanned.requests[0].terminal_keys]
    assert again.key == terminal.key
    assert again.contract.configuration_fingerprint != terminal.contract.configuration_fingerprint

    with pytest.raises(ValueError, match="project scope"):
        plan_request(
            _StubPlanner(bids, spec), _StubRegistry(), group, _Store(), participants=("01",)
        )


def test_project_scope_work_item_identity():
    from nro.orchestration.contracts import subject_label
    from nro.orchestration.registry_work_items import work_item_relative_directory

    common = dict(
        key="k",
        module="analyses",
        project="p1",
        entities={"network": "aud"},
        module_lineage_id=1,
        config_fingerprint="c",
        directory_label="d",
        runtime_config=Path("r.json"),
        command=("true",),
        dependencies=(),
        input_paths=(),
        output_root=Path("/results/project-p1"),
        output_prefix=None,
        resource_class="small",
        output_format="slots",
    )
    from nro.orchestration.contracts import WorkItemSpec

    spec = WorkItemSpec.create(participant="", scope="project", **common)
    assert spec.is_project_scope and spec.participant == ""
    with pytest.raises(ValueError, match="requires a participant"):
        WorkItemSpec.create(participant="", scope="subject", **common)
    with pytest.raises(ValueError, match="takes no participant"):
        WorkItemSpec.create(participant="01", scope="project", **common)
    assert subject_label("") == "project" and subject_label("01") == "sub-01"
    row = {
        "project": "p1",
        "participant": "",
        "module": "analyses",
        "entities_json": '{"network": "aud"}',
        "work_item_key": "analyses:" + "a" * 64,
    }
    assert work_item_relative_directory(row) == Path(
        "p1/analyses/project/project_network-aud/" + "a" * 16
    )


def test_viz_serves_analysis_viewers_and_only_their_outputs(study_roots, upstream, monkeypatch):
    import threading
    import urllib.error
    import urllib.request
    from http.cookiejar import CookieJar

    from nro.viz.catalog import Catalogs
    from nro.viz.server import make_server

    bids, index = upstream
    repository = _group_study(study_roots, GROUP_MANIFEST + "viz: viz/index.html\n")
    with pytest.raises(ManifestError, match="no viewer page"):
        load_analysis(StudySource.at("s1"), "group")
    viewer = repository / "analyses/group/viz"
    viewer.mkdir()
    (viewer / "index.html").write_text("<!doctype html><title>group</title>")
    (viewer / "app.js").write_text("export const ok = true;\n")
    _commit(repository, "Add viewer")
    group = load_analysis(StudySource.at("s1"), "group")
    similarity = load_analysis(StudySource.at("s1"), "similarity")
    assert execute(plan_analysis(similarity, index=index, bids_root=bids)[0]).succeeded
    (item,) = plan_analysis(group, index=index, bids_root=bids)
    assert execute(item).succeeded

    monkeypatch.setattr("nro.viz.analyses.studies", lambda: ("s1",))
    server = make_server(Catalogs(lambda branch: []), host="127.0.0.1", port=0, token="secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def status(url):
        try:
            with opener.open(base + url) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    try:
        assert status("/api/analyses") == 403
        opener.open(f"{base}/?token=secret").read()
        with opener.open(f"{base}/api/analyses") as response:
            (listed,) = json.loads(response.read())
        assert listed["name"] == "s1/group" and listed["url"] == "/analyses/s1/group/index.html"
        with opener.open(base + listed["url"]) as response:
            assert b"<title>group</title>" in response.read()
            assert response.headers["Content-Type"].startswith("text/html")
        with opener.open(f"{base}/analyses/s1/group/app.js") as response:
            assert response.headers["Content-Type"].startswith("text/javascript")
        assert status("/analyses/s1/group/%2e%2e/analysis.py") == 403
        assert status("/analyses/s1/similarity/index.html") == 404
        with opener.open(f"{base}/api/analysis/outputs?analysis=s1/group") as response:
            (output,) = json.loads(response.read())
        assert output["participant"] is None and output["slot"] == "summary"
        with opener.open(
            f"{base}/api/analysis/file?analysis=s1/group&project=p1&file={output['file']}"
        ) as response:
            assert json.loads(response.read())["participants"] == ["01"]
        assert status("/api/analysis/file?analysis=s1/group&project=p1&file=analysis.yml") == 404
        assert status("/api/template-mesh?space=fsLR&hemi=L&kind=pial") == 400
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("branch_registries", "expected"), [(True, "SnapshotIndex"), (False, "RegistryIndex")]
)
def test_default_index_reads_the_snapshot_when_branch_registries_hold_lineages(
    monkeypatch, branch_registries, expected
):
    from nro.analysis import resolve

    monkeypatch.setattr(site, "require_execution_support", lambda *args, **kwargs: None)
    monkeypatch.setattr(resolve, "_uses_branch_registries", lambda: branch_registries)
    monkeypatch.setattr(resolve, "_checkout_branch", lambda: "feature")
    index = resolve.default_index()
    assert type(index).__name__ == expected
    if branch_registries:
        assert index.branch == "feature"
