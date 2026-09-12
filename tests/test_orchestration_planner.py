from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import nro.modules.func.planning as func_planning
from nro.configuration.store import ConfigStore
from nro.modules.anat.contract import anatomical_output_contract
from nro.modules.clean.contract import clean_output_contract
from nro.modules.func.contracts import final_resampling_contract, functional_output_contract
from nro.modules.microparcellation.contract import microparcellation_output_contract
from nro.modules.networks.contract import networks_output_contract
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.planner import Planner, build_subject_instances
from nro.orchestration.registry import Registry
from nro.orchestration.worker import _runner_graph_signature


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def branch_registry(tmp_path: Path) -> Registry:
    bids = tmp_path / "bids"
    for participant, tasks in (
        ("01", ("langlocSN", "rest")),
        ("02", ("rest",)),
        ("03", ("langlocSN",)),
    ):
        subject = bids / "demo" / f"sub-{participant}"
        _write(subject / "anat" / f"sub-{participant}_T1w.nii.gz")
        for task in tasks:
            stem = f"sub-{participant}_task-{task}_run-1"
            _write(subject / "func" / f"{stem}_bold.nii.gz")
            _write(subject / "func" / f"{stem}_bold.json", "{}")
            _write(
                subject / "func" / f"{stem}_events.tsv", "onset\tduration\ttrial_type\n0\t1\tS\n"
            )
    return Registry.for_project("demo", bids_root=bids)


def _plan_modules(registry, tmp_path, modules, workflow_ids=("main",)):
    workflows = {name: ConfigStore().resolve(name) for name in workflow_ids}
    return Planner(registry, bids_root=tmp_path / "bids").plan(
        projects=("demo",),
        requested_participants=(),
        modules=modules,
        workflows=workflows,
        registered_workflows={
            name: registry.register_workflow(workflow) for name, workflow in workflows.items()
        },
        selectors={},
        spaces=("fsnative", "T1w"),
        smoothing_levels=(0, 2),
        memory_gb=32,
        max_memory_gb=256,
    )


def test_request_pins_execution_without_changing_scientific_contract(branch_registry, tmp_path):
    plan = _plan_modules(branch_registry, tmp_path, ("anat",))
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    planner.register_requests(plan, selectors={}, concurrency=50, partition=None)
    for row in branch_registry.instance_rows():
        spec = plan.instances[row["instance_key"]]
        assert row["artifact_fingerprint"] == spec.contract_fingerprint
        assert json.loads(row["artifact_contract_json"]) == spec.instance_contract
        command = json.loads(row["command_json"])
        assert Path(command[1]).name == "source_launcher.py"
        assert command[2] == plan.source_digest
        assert Path(command[3]).is_file()
        assert command[5:] == list(spec.command[2:])


def test_source_change_after_planning_creates_no_demand(branch_registry, tmp_path):
    from dataclasses import replace

    plan = _plan_modules(branch_registry, tmp_path, ("anat",))
    plan = replace(plan, source_digest="0" * 64)
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    with pytest.raises(ValueError, match="Source changed during planning"):
        planner.register_requests(plan, selectors={}, concurrency=50, partition=None)
    assert branch_registry.request_rows() == []


def test_branch_planning_does_not_open_or_register_with_the_scheduler(
    branch_registry, tmp_path, monkeypatch
):
    from nro.orchestration.branch_store import BranchStore
    from nro.orchestration.control_paths import ControlPaths

    store = BranchStore(tmp_path / "branch-control")
    store.initialize()
    science = store.registry("dev")
    workflow = ConfigStore().resolve("main")
    registered = science.register_workflow(workflow)
    monkeypatch.setattr(
        Registry,
        "for_project",
        lambda *a, **k: pytest.fail("Scientific planning opened the scheduler"),
    )
    planner = Planner(science, bids_root=tmp_path / "bids")
    plan = planner.plan(
        projects=("demo",),
        requested_participants=("01",),
        modules=("anat",),
        workflows={"main": workflow},
        registered_workflows={"main": registered},
        selectors={},
        spaces=("fsnative",),
        smoothing_levels=(2,),
        memory_gb=32,
        max_memory_gb=256,
    )
    assert len(plan.requests) == 1
    instances = tuple(plan.instances.values())
    science.record_graph(instances, expected_revisions={spec.key: None for spec in instances})
    assert len(science.instances()) == 1
    assert instances[0].runtime_config.is_relative_to(science.root)
    with pytest.raises(ValueError, match="submission is not enabled"):
        planner.register_requests(plan, selectors={}, concurrency=10, partition=None)
    assert not ControlPaths(store.control).database.exists()


def test_all_planned_manifests_use_the_selected_bids_root(branch_registry, tmp_path):
    from nro.orchestration.artifact_resolution import scientific_contracts

    plan = _plan_modules(branch_registry, tmp_path, ("networks", "firstlevels"))
    instances = tuple(plan.instances.values())
    for spec in instances:
        assert all(path.is_relative_to(spec.output_root) for path in spec.expected_outputs)
    assert len(scientific_contracts(instances)) == len(instances)


@pytest.mark.parametrize(
    "modules",
    [
        ("microparcellation", "networks"),
        ("networks", "microparcellation"),
        ("anat", "func", "clean", "microparcellation", "networks", "networks"),
    ],
)
def test_upstream_targets_collapse_to_endpoint(branch_registry, tmp_path, modules):
    expected = _plan_modules(branch_registry, tmp_path, ("networks",))
    actual = _plan_modules(branch_registry, tmp_path, modules)
    assert actual.instances == expected.instances
    assert actual.matched_participants == expected.matched_participants
    assert len(actual.requests) == 1
    request = actual.requests[0]
    assert request.module == "networks"
    assert request.terminal_keys == expected.requests[0].terminal_keys
    assert request.instances == expected.requests[0].instances
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    planner.register_requests(actual, selectors={}, concurrency=50, partition=None)
    assert len(branch_registry.request_rows()) == 1
    branch_registry.request_cancellation(modules=("networks",))
    assert not any(row["demanded"] for row in branch_registry.instance_rows())


@pytest.mark.parametrize("modules", [("func", "firstlevels"), ("firstlevels", "func")])
def test_partial_upstream_coverage_keeps_only_uncovered_runs(
    branch_registry,
    tmp_path,
    modules,
):
    plan = _plan_modules(branch_registry, tmp_path, modules)
    requests = {request.module: request for request in plan.requests}
    assert set(requests) == {"func", "firstlevels"}
    func = requests["func"]
    assert func.participants == ("01", "02")
    assert len(func.terminal_keys) == 2
    assert {plan.instances[key].entities["task"] for key in func.terminal_keys} == {"rest"}
    assert {item.participant for item in func.instances} == {"01", "02"}
    assert all(item.entities.get("task") != "langlocSN" for item in func.instances)
    fits = requests["firstlevels"]
    assert fits.participants == ("01", "03")
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    planner.register_requests(plan, selectors={}, concurrency=50, partition=None)
    branch_registry.request_cancellation(modules=("firstlevels",))
    demanded = [row for row in branch_registry.instance_rows() if row["demanded"]]
    assert {row["participant"] for row in demanded} == {"01", "02"}
    assert all(json.loads(row["entities_json"]).get("task") != "langlocSN" for row in demanded)


def test_independent_endpoints_cover_upstream_without_merging_workflows(
    branch_registry,
    tmp_path,
):
    plan = _plan_modules(
        branch_registry, tmp_path, ("func", "networks", "firstlevels"), ("main", "oslom")
    )
    assert {(request.workflow_id, request.module) for request in plan.requests} == {
        (workflow, module)
        for workflow in ("main", "oslom")
        for module in ("networks", "firstlevels")
    }
    assert len(plan.requests) == 4


def test_func_instance_contract_tracks_final_resampling_policy(tmp_path: Path) -> None:
    workflow = ConfigStore().resolve("main")
    instance = InstanceSpec.create(
        key="func:" + "a" * 64,
        module="func",
        project="demo",
        participant="01",
        entities={},
        scope="run",
        configuration_lineage_id=1,
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        directory_label="main",
        runtime_config=tmp_path / "main_preprocess.yml",
        command=("python", "-m", "nro.modules.func"),
        dependencies=(),
        input_paths=(tmp_path / "bold.nii.gz",),
        output_root=tmp_path / "output",
        output_prefix="sub-01_task-rest_run-01",
        resource_class="large",
        processing={"final_resampling": final_resampling_contract()},
    )

    assert instance.instance_contract["processing"] == {
        "final_resampling": final_resampling_contract(),
    }


def test_runner_graph_signature_tracks_bids_state_but_not_command_spelling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bids" / "demo" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    _write(source, "first")
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    instance = InstanceSpec.create(
        key="anat:" + "b" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        directory_label="main",
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=tmp_path / "output",
        output_prefix=None,
        resource_class="large",
        expected_outputs=(tmp_path / "output" / "manifest.json",),
    )
    registry.register_instances((instance,))
    row = registry.instance_rows()[0]
    before = _runner_graph_signature(registry, row["id"])
    registry.register_instances(
        (instance.evolve(command=("python", "-m", "nro.modules.anat", "--verbose")),)
    )
    row = registry.instance_rows()[0]
    assert _runner_graph_signature(registry, row["id"]) == before
    source.write_text("second-state")
    after = _runner_graph_signature(registry, row["id"])
    assert after != before


def test_existing_instance_adopts_execution_recipe_and_output_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    original_output = tmp_path / "derivatives" / "manifest.json"
    instance = InstanceSpec.create(
        key="anat:" + "c" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        directory_label="main",
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=original_output.parent,
        output_prefix="sub-01",
        resource_class="large",
        expected_outputs=(original_output,),
    )
    registry.register_instances((instance,))
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh', artifact_reason='Current' "
            "WHERE instance_key=?",
            (instance.key,),
        )

    changed_command = instance.evolve(command=("python", "-m", "nro.modules.anat", "--verbose"))
    registry.register_instances((changed_command,))
    row = registry.instance_rows()[0]
    assert tuple(json.loads(row["command_json"])) == changed_command.command
    assert row["artifact_state"] == "fresh"

    changed_output = tmp_path / "changed" / "different.json"
    registry.register_instances(
        (
            changed_command.evolve(
                output_root=changed_output.parent,
                expected_outputs=(changed_output,),
            ),
        )
    )
    row = registry.instance_rows()[0]
    assert tuple(json.loads(row["expected_outputs_json"])) == (str(changed_output.resolve()),)
    assert row["output_root"] == str(changed_output.parent.resolve())
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Instance contract changed"


def test_active_demand_uses_the_latest_execution_recipe(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    instance = InstanceSpec.create(
        key="anat:" + "f" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        directory_label="main",
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=tmp_path / "output",
        output_prefix="sub-01",
        resource_class="large",
        expected_outputs=(tmp_path / "output" / "manifest.json",),
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )

    registry.register_instances((instance.evolve(command=(*instance.command, "--verbose")),))

    row = registry.instance_rows()[0]
    assert tuple(json.loads(row["command_json"])) == (*instance.command, "--verbose")


def test_existing_instance_adopts_changed_dependency_topology(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    base = InstanceSpec.create(
        key="anat:" + "d" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        directory_label="main",
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=tmp_path / "base",
        output_prefix="sub-01",
        resource_class="large",
        expected_outputs=(tmp_path / "base" / "manifest.json",),
    )
    upstream = base.evolve(
        key="anat:" + "e" * 64,
        output_root=tmp_path / "upstream",
        expected_outputs=(tmp_path / "upstream" / "manifest.json",),
    )
    registry.register_instances((base, upstream))
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh', artifact_reason='Current' "
            "WHERE instance_key=?",
            (base.key,),
        )

    registry.register_instances((base.evolve(dependencies=(upstream.key,)), upstream))

    row = next(item for item in registry.instance_rows() if item["instance_key"] == base.key)
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Instance contract changed"
    with registry.connection() as db:
        dependencies = db.execute(
            """SELECT upstream.instance_key
               FROM instance_dependencies dependency
               JOIN instances upstream ON upstream.id=dependency.upstream_instance_id
               WHERE dependency.instance_id=?""",
            (row["id"],),
        ).fetchall()
    assert [item["instance_key"] for item in dependencies] == [upstream.key]


def test_subject_planner_builds_filtered_complete_dag(tmp_path: Path, monkeypatch) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_dir-LR_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_dir-LR_run-1_bold.json", "{}")
    _write(subject / "func" / "sub-01_task-language_dir-RL_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-language_dir-RL_run-1_bold.json", "{}")

    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copytree(ConfigStore().root, configs, dirs_exist_ok=True)
    (configs / "workflows" / "rest_workflow.yml").write_text(
        yaml.safe_dump({"microparcellation": "rest"})
    )
    (configs / "configs" / "microparcellation" / "rest_microparcellation.yml").write_text(
        yaml.safe_dump({"input_filter": {"task": "rest"}})
    )
    store = ConfigStore()
    store.root = configs
    workflow = store.resolve("rest")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)

    inventory_calls = 0
    original_inventory = func_planning.load_session_inventory

    def counted_inventory(*args, **kwargs):
        nonlocal inventory_calls
        inventory_calls += 1
        return original_inventory(*args, **kwargs)

    monkeypatch.setattr(func_planning, "load_session_inventory", counted_inventory)

    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="networks",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    by_module = {}
    for instance in instances:
        by_module.setdefault(instance.module, []).append(instance)

    assert len(by_module["anat"]) == 1
    assert len(by_module["func"]) == 1
    assert len(by_module["clean"]) == 1
    assert len(by_module["microparcellation"]) == 1
    assert len(by_module["networks"]) == 1
    assert by_module["anat"][0].instance_contract["processing"] == {
        "output_metadata": anatomical_output_contract()
    }
    assert by_module["func"][0].instance_contract["processing"] == {
        "final_resampling": final_resampling_contract(),
        "output_metadata": functional_output_contract(),
    }
    assert by_module["func"][0].entities == {"task": "rest", "dir": "LR", "run": "1"}
    assert by_module["clean"][0].entities == {
        "task": "rest",
        "dir": "LR",
        "run": "1",
        "space": "fsnative",
        "smoothing": "2",
    }
    assert by_module["clean"][0].instance_contract["processing"] == {
        "output_metadata": clean_output_contract()
    }
    assert by_module["clean"][0].dependencies == (
        by_module["func"][0].key,
        by_module["anat"][0].key,
    )
    assert by_module["microparcellation"][0].entities == {"space": "fsnative", "smoothing": "2"}
    assert by_module["microparcellation"][0].dependencies == (by_module["clean"][0].key,)
    assert by_module["microparcellation"][0].instance_contract["processing"] == {
        "output_metadata": microparcellation_output_contract()
    }
    micro_outputs = by_module["microparcellation"][0].expected_outputs
    assert micro_outputs[0].name.endswith("_desc-microparcellation_manifest.yaml")
    assert micro_outputs[1].name.endswith("_desc-microparcellationQuality_metrics.json")
    assert micro_outputs[2].name.endswith("_desc-microparcellationIndex_manifest.json")
    assert by_module["networks"][0].dependencies == (
        by_module["microparcellation"][0].key,
        by_module["anat"][0].key,
    )
    assert by_module["networks"][0].instance_contract["processing"] == {
        "output_metadata": networks_output_contract()
    }
    assert registered.directories["preprocessing"] == "main"
    assert registered.directories["microparcellation"] == "rest"
    assert registered.directories["networks"] == "rest"
    assert inventory_calls == 1


def test_run_discovery_uses_source_bids_not_derivatives(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")

    derivative = bids / "demo" / "derivatives" / "preprocessing" / "main" / "sub-01" / "func"
    _write(derivative / "sub-01_task-rest_run-2_desc-preproc_bold.nii.gz")
    _write(derivative / "sub-01_task-rest_run-2_desc-preproc_bold.json", "{}")
    nested_derivative = subject / "derivatives" / "copied" / "func"
    _write(nested_derivative / "sub-01_task-rest_run-3_bold.nii.gz")
    _write(nested_derivative / "sub-01_task-rest_run-3_bold.json", "{}")

    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="clean",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )

    assert [instance.entities for instance in instances if instance.module == "func"] == [
        {"task": "rest", "run": "1"}
    ]
    assert [instance.entities for instance in instances if instance.module == "clean"] == [
        {"task": "rest", "run": "1", "space": "fsnative", "smoothing": "2"}
    ]


def test_functional_contract_tracks_inherited_bids_metadata(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    project = bids / "demo"
    subject = project / "sub-01"
    _write(project / "dataset_description.json", "{}")
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-story_bold.nii.gz")
    inherited = project / "task-story_bold.json"
    _write(
        inherited,
        json.dumps(
            {
                "RepetitionTime": 2.0,
                "PhaseEncodingDirection": "j-",
                "TotalReadoutTime": 0.05,
            }
        ),
    )
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)

    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="func",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    functional = next(instance for instance in instances if instance.module == "func")

    assert inherited in functional.input_paths
    assert subject / "func" / "sub-01_task-story_bold.json" not in functional.input_paths


def test_subject_planner_creates_only_requested_space_smoothing_cross_product(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)

    instances = build_subject_instances(
        project="demo",
        participant="01",
        module="networks",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
        spaces=("fsnative", "T1w"),
        smoothing_levels=(0, 2),
    )
    by_module = {
        name: [instance for instance in instances if instance.module == name]
        for name in ("anat", "func", "clean", "microparcellation", "networks")
    }
    expected_pairs = {
        ("fsnative", "0"),
        ("fsnative", "2"),
        ("T1w", "0"),
        ("T1w", "2"),
    }

    assert len(by_module["anat"]) == 1
    assert len(by_module["func"]) == 1
    assert {
        (item.entities["space"], item.entities["smoothing"]) for item in by_module["clean"]
    } == expected_pairs
    assert {
        (item.entities["space"], item.entities["smoothing"])
        for item in by_module["microparcellation"]
    } == expected_pairs
    assert {
        (item.entities["space"], item.entities["smoothing"]) for item in by_module["networks"]
    } == expected_pairs
    clean_keys = {
        (item.entities["space"], item.entities["smoothing"]): item.key
        for item in by_module["clean"]
    }
    for item in by_module["microparcellation"]:
        pair = (item.entities["space"], item.entities["smoothing"])
        assert item.dependencies == (clean_keys[pair],)
