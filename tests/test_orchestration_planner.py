from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import nro.modules.func.planning as func_planning
from nro.configuration.definition_migrations import refresh_manifest
from nro.configuration.hardware import resolve_gradient_unwarping
from nro.configuration.markup import SubjectMarkup
from nro.configuration.store import ConfigStore, configuration_fingerprint, fingerprint
from nro.modules.anat import planning as anat_planning
from nro.modules.anat.contract import (
    anatomical_output_contract,
    bias_correction_contract,
    pose_normalization_contract,
    surface_reconstruction_contract,
)
from nro.modules.clean.contract import clean_output_contract
from nro.modules.func.contract import final_resampling_contract, functional_output_contract
from nro.modules.microparcellation.contract import microparcellation_output_contract
from nro.modules.networks.contract import networks_output_contract
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planner import Planner, RegisteredTarget, build_subject_work_items
from nro.orchestration.planning_context import SubjectPlanningContext
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


def test_lesion_anatomy_keeps_its_parent_work_item_on_cpu(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    context = SubjectPlanningContext(
        project="demo",
        participant="01",
        sub_id="sub-01",
        bids_root=bids,
        project_root=bids / "demo",
        subject_dir=subject,
        workflow=workflow,
        registered=registered,
        registry=registry,
        runs=(),
        aggregate_source_inputs=(),
        target_pairs=(),
        memory_gb=32,
        max_memory_gb=256,
        definitions_roots=(),
        gradient_coefficients_root=tmp_path / "gradients",
        source_markup=SubjectMarkup("main", "demo", subject, lesion=True),
    )

    work_item = anat_planning.plan_work_items(context, {}, module_descriptor("anat"))[0]

    assert work_item.resource_class == "large"


def test_fastsurfer_anatomy_keeps_parent_on_cpu_and_records_backend(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo/sub-01"
    _write(subject / "anat/sub-01_T1w.nii.gz")
    definitions = tmp_path / "definitions"
    shutil.copytree(ConfigStore().root, definitions)
    _write(
        definitions / "configs/anat/fast_anat.yml",
        "surface_reconstruction_engine: fastsurfer\n",
    )
    _write(definitions / "workflows/fast_workflow.yml", "anat: fast\n")
    refresh_manifest(definitions)
    workflow = ConfigStore(definitions).resolve("fast")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    context = SubjectPlanningContext(
        project="demo",
        participant="01",
        sub_id="sub-01",
        bids_root=bids,
        project_root=bids / "demo",
        subject_dir=subject,
        workflow=workflow,
        registered=registered,
        registry=registry,
        runs=(),
        aggregate_source_inputs=(),
        target_pairs=(),
        memory_gb=32,
        max_memory_gb=256,
        definitions_roots=(),
        gradient_coefficients_root=tmp_path / "gradients",
        source_markup=SubjectMarkup("main", "demo", subject),
    )

    work_item = anat_planning.plan_work_items(context, {}, module_descriptor("anat"))[0]

    assert work_item.resource_class == "large"
    assert work_item.work_item_contract["processing"]["surface_reconstruction"] == (
        surface_reconstruction_contract("fastsurfer")
    )


def test_request_pins_execution_without_changing_scientific_contract(branch_registry, tmp_path):
    plan = _plan_modules(branch_registry, tmp_path, ("anat",))
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    planner.register_requests(plan, selectors={}, concurrency=50, partition=None)
    for row in branch_registry.work_item_rows():
        spec = plan.work_items[row["work_item_key"]]
        assert row["artifact_fingerprint"] == spec.contract_fingerprint
        assert json.loads(row["artifact_contract_json"]) == spec.work_item_contract
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
    work_items = tuple(plan.work_items.values())
    science.record_work_item_graph(
        work_items, expected_revisions={spec.key: None for spec in work_items}
    )
    assert len(science.work_items()) == 1
    assert work_items[0].runtime_config.is_relative_to(science.root)
    with pytest.raises(ValueError, match="submission is not enabled"):
        planner.register_requests(plan, selectors={}, concurrency=10, partition=None)
    assert not ControlPaths(store.control).database.exists()


def test_planner_filters_terminal_routes_by_work_item_id(branch_registry, tmp_path) -> None:
    workflow = ConfigStore().resolve("main")
    registered = branch_registry.register_workflow(workflow)
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    arguments = dict(
        projects=("demo",),
        requested_participants=("01",),
        modules=("networks",),
        workflows={"main": workflow},
        registered_workflows={"main": registered},
        selectors={},
        spaces=("fsnative",),
        smoothing_levels=(2,),
        memory_gb=32,
        max_memory_gb=256,
    )

    absent = planner.plan(**arguments, lineage_ids=("networks/absent",))
    selected = planner.plan(
        **arguments,
        lineage_ids=(f"networks/{registered.directory_for('networks')}",),
    )

    assert absent.requests == ()
    assert selected.requests


def test_all_planned_manifests_use_the_selected_bids_root(branch_registry, tmp_path):
    from nro.orchestration.artifact_resolution import scientific_contracts

    plan = _plan_modules(branch_registry, tmp_path, ("networks", "firstlevels"))
    work_items = tuple(plan.work_items.values())
    for spec in work_items:
        assert all(path.is_relative_to(spec.output_root) for path in spec.expected_outputs)
    assert len(scientific_contracts(work_items)) == len(work_items)


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
    assert actual.work_items == expected.work_items
    assert actual.matched_participants == expected.matched_participants
    assert len(actual.requests) == 1
    request = actual.requests[0]
    assert request.module == "networks"
    assert request.terminal_keys == expected.requests[0].terminal_keys
    assert request.work_items == expected.requests[0].work_items
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    planner.register_requests(actual, selectors={}, concurrency=50, partition=None)
    assert len(branch_registry.request_rows()) == 1
    branch_registry.request_cancellation(modules=("networks",))
    assert not any(row["demanded"] for row in branch_registry.work_item_rows())


def test_dynconn_backed_networks_select_only_the_configured_source(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    definitions = tmp_path / "definitions"
    shutil.copytree(ConfigStore().root, definitions)
    _write(
        definitions / "configs/networks/dynconn_networks.yml",
        "connectivity_source: dynconn\n",
    )
    _write(definitions / "workflows/dynconn_workflow.yml", "networks: dynconn\n")
    refresh_manifest(definitions)
    store = ConfigStore(definitions)
    workflow = store.resolve("dynconn")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    main_registered = registry.register_workflow(store.resolve("main"))

    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="networks",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
        spaces=("fsnative",),
        smoothing_levels=(2,),
    )
    by_module = {item.module: item for item in work_items}

    assert set(by_module) == {"anat", "func", "clean", "dynconn", "networks"}
    assert (
        registered.lineage_fingerprints["networks"]
        != main_registered.lineage_fingerprints["networks"]
    )
    assert by_module["networks"].dependencies == (
        by_module["dynconn"].key,
        by_module["anat"].key,
    )
    assert any(
        "dynamicConnectivity_manifest" in path.name for path in by_module["networks"].input_paths
    )
    assert not any("microparcellation" in str(path) for path in by_module["networks"].input_paths)


def test_planner_can_reproduce_one_exact_registered_target(branch_registry, tmp_path):
    complete = _plan_modules(branch_registry, tmp_path, ("clean",))
    target = next(
        key
        for key in complete.requests[0].terminal_keys
        if complete.work_items[key].participant == "01"
        and complete.work_items[key].entities["space"] == "T1w"
        and complete.work_items[key].entities["smoothing"] == "2"
        and complete.work_items[key].entities["task"] == "rest"
    )
    workflow = ConfigStore().resolve("main")
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    exact = planner.plan_registered_targets(
        (
            RegisteredTarget(
                project="demo",
                participant="01",
                module="clean",
                workflow_id="main",
                work_item_key=target,
                entities=complete.work_items[target].entities,
            ),
        ),
        workflows={"main": workflow},
        registered_workflows={"main": branch_registry.register_workflow(workflow)},
        memory_gb=32,
        max_memory_gb=256,
    )

    assert len(exact.requests) == 1
    assert exact.requests[0].terminal_keys == (target,)
    assert exact.requests[0].participants == ("01",)
    assert target in exact.work_items
    assert all(work_item.participant == "01" for work_item in exact.work_items.values())
    assert {
        (work_item.entities.get("space"), work_item.entities.get("smoothing"))
        for work_item in exact.work_items.values()
        if work_item.module == "clean"
    } == {("T1w", "2")}


def test_registered_run_targets_share_one_subject_plan(branch_registry, tmp_path, monkeypatch):
    complete = _plan_modules(branch_registry, tmp_path, ("func",))
    target_keys = tuple(
        key
        for key, work_item in complete.work_items.items()
        if work_item.module == "func" and work_item.participant == "01"
    )
    assert len(target_keys) == 2
    workflow = ConfigStore().resolve("main")
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    original = planner.plan_subject
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, "plan_subject", counted)
    exact = planner.plan_registered_targets(
        tuple(
            RegisteredTarget(
                project="demo",
                participant="01",
                module="func",
                workflow_id="main",
                work_item_key=key,
                entities=complete.work_items[key].entities,
            )
            for key in target_keys
        ),
        workflows={"main": workflow},
        registered_workflows={"main": branch_registry.register_workflow(workflow)},
        memory_gb=32,
        max_memory_gb=256,
    )

    assert calls == 1
    assert {key for request in exact.requests for key in request.terminal_keys} == set(target_keys)


def test_registered_targets_preserve_workflow_associations(branch_registry, tmp_path):
    workflows = {name: ConfigStore().resolve(name) for name in ("main", "oslom")}
    registered = {
        name: branch_registry.register_workflow(workflow) for name, workflow in workflows.items()
    }
    complete = Planner(branch_registry, bids_root=tmp_path / "bids").plan(
        projects=("demo",),
        requested_participants=("01",),
        modules=("anat",),
        workflows=workflows,
        registered_workflows=registered,
        selectors={},
        spaces=("fsnative",),
        smoothing_levels=(2,),
        memory_gb=32,
        max_memory_gb=256,
    )
    targets = tuple(
        RegisteredTarget(
            project=request.project,
            participant="01",
            module="anat",
            workflow_id=request.workflow_id,
            work_item_key=request.terminal_keys[0],
            entities=complete.work_items[request.terminal_keys[0]].entities,
        )
        for request in complete.requests
    )

    exact = Planner(branch_registry, bids_root=tmp_path / "bids").plan_registered_targets(
        targets,
        workflows=workflows,
        registered_workflows=registered,
        memory_gb=32,
        max_memory_gb=256,
    )

    assert {request.workflow_id for request in exact.requests} == {"main", "oslom"}


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
    assert {plan.work_items[key].entities["task"] for key in func.terminal_keys} == {"rest"}
    assert {item.participant for item in func.work_items} == {"01", "02"}
    assert all(item.entities.get("task") != "langlocSN" for item in func.work_items)
    fits = requests["firstlevels"]
    assert fits.participants == ("01", "03")
    planner = Planner(branch_registry, bids_root=tmp_path / "bids")
    planner.register_requests(plan, selectors={}, concurrency=50, partition=None)
    branch_registry.request_cancellation(modules=("firstlevels",))
    demanded = [row for row in branch_registry.work_item_rows() if row["demanded"]]
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


def test_func_work_item_contract_tracks_final_resampling_policy(tmp_path: Path) -> None:
    workflow = ConfigStore().resolve("main")
    work_item = WorkItemSpec.create(
        key="func:" + "a" * 64,
        module="func",
        project="demo",
        participant="01",
        entities={},
        scope="run",
        module_lineage_id=1,
        config_fingerprint=workflow.configuration("func").fingerprint,
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

    assert work_item.work_item_contract["processing"] == {
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
    work_item = WorkItemSpec.create(
        key="anat:" + "b" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=tmp_path / "output",
        output_prefix=None,
        resource_class="large",
        expected_outputs=(tmp_path / "output" / "manifest.json",),
    )
    registry.register_work_items((work_item,))
    row = registry.work_item_rows()[0]
    before = _runner_graph_signature(registry, row["id"])
    registry.register_work_items(
        (work_item.evolve(command=("python", "-m", "nro.modules.anat", "--verbose")),)
    )
    row = registry.work_item_rows()[0]
    assert _runner_graph_signature(registry, row["id"]) == before
    source.write_text("second-state")
    after = _runner_graph_signature(registry, row["id"])
    assert after != before


def test_existing_work_item_adopts_execution_recipe_and_output_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    original_output = tmp_path / "derivatives" / "manifest.json"
    work_item = WorkItemSpec.create(
        key="anat:" + "c" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=original_output.parent,
        output_prefix="sub-01",
        resource_class="large",
        expected_outputs=(original_output,),
    )
    registry.register_work_items((work_item,))
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh', artifact_reason='Current' "
            "WHERE work_item_key=?",
            (work_item.key,),
        )

    changed_command = work_item.evolve(command=("python", "-m", "nro.modules.anat", "--verbose"))
    registry.register_work_items((changed_command,))
    row = registry.work_item_rows()[0]
    assert tuple(json.loads(row["command_json"])) == changed_command.command
    assert row["artifact_state"] == "fresh"

    changed_output = tmp_path / "changed" / "different.json"
    registry.register_work_items(
        (
            changed_command.evolve(
                output_root=changed_output.parent,
                expected_outputs=(changed_output,),
            ),
        )
    )
    row = registry.work_item_rows()[0]
    assert tuple(json.loads(row["expected_outputs_json"])) == (str(changed_output.resolve()),)
    assert row["output_root"] == str(changed_output.parent.resolve())
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Work-item contract changed"


def test_active_demand_uses_the_latest_execution_recipe(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    work_item = WorkItemSpec.create(
        key="anat:" + "f" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
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
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )

    registry.register_work_items((work_item.evolve(command=(*work_item.command, "--verbose")),))

    row = registry.work_item_rows()[0]
    assert tuple(json.loads(row["command_json"])) == (*work_item.command, "--verbose")


def test_registration_migrates_completed_historical_module_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    configuration = workflow.configuration("anat")
    historical = {
        key: value
        for key, value in configuration.values.items()
        if key not in {"fastsurfer_container", "surface_reconstruction_engine"}
    }
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    work_item = WorkItemSpec.create(
        key="anat:" + "9" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=configuration.module_fingerprint("anat"),
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=tmp_path / "output",
        output_prefix="sub-01",
        resource_class="large",
        expected_outputs=(tmp_path / "output" / "manifest.json",),
    )
    registry.register_work_items((work_item,))
    old_contract = json.loads(json.dumps(work_item.work_item_contract))
    old_contract["contract_schema"] = 4
    old_contract["configuration"] = "historical-anat-scientific-fingerprint"
    with registry.connection(write=True) as database:
        row = database.execute(
            "SELECT id,revision_fingerprint,command_json FROM work_items WHERE work_item_key=?",
            (work_item.key,),
        ).fetchone()
        lineage = database.execute(
            "SELECT lineage_fingerprint FROM module_lineages WHERE id=?",
            (registered.lineages["anat"],),
        ).fetchone()
        serialized = json.dumps(old_contract, sort_keys=True, separators=(",", ":"))
        database.execute(
            """UPDATE work_items
               SET artifact_contract_json=?,artifact_fingerprint=?,
                   artifact_state='fresh',artifact_reason='Completed successfully'
               WHERE id=?""",
            (serialized, fingerprint(old_contract), row["id"]),
        )
        database.execute(
            """INSERT INTO completions(
                   work_item_id,attempt_id,generation,completed_at,revision_fingerprint,
                   artifact_contract_json,artifact_fingerprint,config_id,config_fingerprint,
                   lineage_fingerprint,resolved_yaml,provenance_json,command_json
               ) VALUES (?,NULL,1,'2026-01-01T00:00:00+00:00',?,?,?,?,?,?,?,?,?)""",
            (
                row["id"],
                row["revision_fingerprint"],
                serialized,
                fingerprint(old_contract),
                "main",
                configuration_fingerprint("anat", "main", historical),
                lineage["lineage_fingerprint"],
                yaml.safe_dump(historical),
                "{}",
                row["command_json"],
            ),
        )

    with registry.connection(write=True) as database:
        registry._upsert_work_item_graph_locked(
            database,
            ((work_item, work_item.as_record()),),
            now="2026-01-02T00:00:00+00:00",
            owner_branch="main",
        )

    row = registry.work_item_rows()[0]
    assert row["artifact_state"] == "fresh"
    assert json.loads(row["artifact_contract_json"]) == work_item.work_item_contract


def test_existing_work_item_adopts_changed_dependency_topology(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    base = WorkItemSpec.create(
        key="anat:" + "d" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
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
    registry.register_work_items((base, upstream))
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh', artifact_reason='Current' "
            "WHERE work_item_key=?",
            (base.key,),
        )

    registry.register_work_items((base.evolve(dependencies=(upstream.key,)), upstream))

    row = next(item for item in registry.work_item_rows() if item["work_item_key"] == base.key)
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Work-item contract changed"
    with registry.connection() as db:
        dependencies = db.execute(
            """SELECT upstream.work_item_key
               FROM work_item_dependencies dependency
               JOIN work_items upstream ON upstream.id=dependency.upstream_work_item_id
               WHERE dependency.work_item_id=?""",
            (row["id"],),
        ).fetchall()
    assert [item["work_item_key"] for item in dependencies] == [upstream.key]


def test_equivalent_work_item_admission_does_not_rewrite_row(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    _write(source)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "derivatives" / "manifest.json"
    work_item = WorkItemSpec.create(
        key="anat:" + "9" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
        command=("python", "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(source,),
        output_root=output.parent,
        output_prefix="sub-01",
        resource_class="large",
        expected_outputs=(output,),
    )
    registry.register_work_items((work_item,))
    with registry.connection(write=True) as database:
        original = database.execute(
            "SELECT updated_at FROM work_items WHERE work_item_key=?", (work_item.key,)
        ).fetchone()[0]
        registry._upsert_work_item_graph_locked(
            database,
            ((work_item, work_item.as_record()),),
            now="2099-01-01T00:00:00+00:00",
        )
        current = database.execute(
            "SELECT updated_at FROM work_items WHERE work_item_key=?", (work_item.key,)
        ).fetchone()[0]
    assert current == original


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
    refresh_manifest(configs)
    store = ConfigStore(configs)
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

    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="networks",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    by_module = {}
    for work_item in work_items:
        by_module.setdefault(work_item.module, []).append(work_item)

    assert len(by_module["anat"]) == 1
    assert len(by_module["func"]) == 1
    assert len(by_module["clean"]) == 1
    assert len(by_module["microparcellation"]) == 1
    assert len(by_module["networks"]) == 1
    assert by_module["anat"][0].resource_class == "large"
    source_markup = {
        "id": "main",
        "project": "demo",
        "subject_dir": str(subject),
        "T1w": [],
        "T2w": [],
        "exclude": [],
        "lesion": False,
    }
    no_gradient_match = resolve_gradient_unwarping({}, mode="auto").scientific_record()
    assert by_module["anat"][0].work_item_contract["processing"] == {
        "gradient_unwarping": [
            {
                "source": str(subject / "anat" / "sub-01_T1w.nii.gz"),
                **no_gradient_match,
            }
        ],
        "bias_correction": bias_correction_contract(),
        "output_metadata": anatomical_output_contract(),
        "pose_normalization": pose_normalization_contract(),
        "surface_reconstruction": surface_reconstruction_contract(),
        "source_markup": source_markup,
    }
    assert by_module["func"][0].work_item_contract["processing"] == {
        "final_resampling": final_resampling_contract(gradient_unwarping=False),
        "gradient_unwarping": [
            {
                "source": str(subject / "func" / "sub-01_task-rest_dir-LR_run-1_bold.nii.gz"),
                **no_gradient_match,
            }
        ],
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
    assert by_module["clean"][0].work_item_contract["processing"] == {
        "output_metadata": clean_output_contract(),
    }
    assert by_module["clean"][0].dependencies == (
        by_module["func"][0].key,
        by_module["anat"][0].key,
    )
    assert by_module["microparcellation"][0].entities == {"space": "fsnative", "smoothing": "2"}
    assert by_module["microparcellation"][0].dependencies == (
        by_module["clean"][0].key,
        by_module["anat"][0].key,
    )
    assert any(
        path.name == "sub-01_desc-preprocessAnat_manifest.json"
        for path in by_module["microparcellation"][0].input_paths
    )
    assert by_module["microparcellation"][0].work_item_contract["processing"] == {
        "output_metadata": microparcellation_output_contract(),
    }
    micro_outputs = by_module["microparcellation"][0].expected_outputs
    assert micro_outputs[0].name.endswith("_desc-microparcellation_manifest.yaml")
    assert micro_outputs[1].name.endswith("_desc-microparcellationQuality_metrics.json")
    assert micro_outputs[2].name.endswith("_desc-microparcellationIndex_manifest.json")
    assert by_module["networks"][0].dependencies == (
        by_module["microparcellation"][0].key,
        by_module["anat"][0].key,
    )
    assert by_module["networks"][0].work_item_contract["processing"] == {
        "output_metadata": networks_output_contract(),
    }
    assert registered.directories["anat"].startswith("main-")
    assert registered.directories["func"].startswith("main-")
    assert registered.directories["microparcellation"].startswith("rest-")
    assert registered.directories["networks"].startswith("main-")
    assert inventory_calls == 1


def test_run_discovery_uses_source_bids_not_derivatives(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")

    derivative = bids / "demo" / "derivatives" / "nro" / "func" / "main" / "sub-01" / "func"
    _write(derivative / "sub-01_task-rest_run-2_desc-preproc_bold.nii.gz")
    _write(derivative / "sub-01_task-rest_run-2_desc-preproc_bold.json", "{}")
    nested_derivative = subject / "derivatives" / "copied" / "func"
    _write(nested_derivative / "sub-01_task-rest_run-3_bold.nii.gz")
    _write(nested_derivative / "sub-01_task-rest_run-3_bold.json", "{}")

    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="clean",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )

    assert [work_item.entities for work_item in work_items if work_item.module == "func"] == [
        {"task": "rest", "run": "1"}
    ]
    assert [work_item.entities for work_item in work_items if work_item.module == "clean"] == [
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

    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="func",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
    )
    functional = next(work_item for work_item in work_items if work_item.module == "func")

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

    work_items = build_subject_work_items(
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
        name: [work_item for work_item in work_items if work_item.module == name]
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
    anat_key = by_module["anat"][0].key
    for item in by_module["microparcellation"]:
        pair = (item.entities["space"], item.entities["smoothing"])
        assert item.dependencies == (clean_keys[pair], anat_key)


@pytest.mark.parametrize("space", ["MNI152NLin2009cAsym", "fsaverage6"])
def test_template_microparcellation_does_not_add_an_unused_anatomy_edge(
    tmp_path: Path, space: str
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)

    work_items = build_subject_work_items(
        project="demo",
        participant="01",
        module="microparcellation",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=bids,
        spaces=(space,),
        smoothing_levels=(2,),
    )
    anatomy = next(item for item in work_items if item.module == "anat")
    clean = next(item for item in work_items if item.module == "clean")
    micro = next(item for item in work_items if item.module == "microparcellation")

    assert micro.dependencies == (clean.key,)
    assert anatomy.key not in micro.dependencies
    assert not any("preprocessAnat_manifest" in path.name for path in micro.input_paths)
