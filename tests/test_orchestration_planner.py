from __future__ import annotations

import json
from pathlib import Path

import yaml
import shutil

import nro.func.planning as func_planning
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.registry import Registry
from nro.orchestration.planner import build_subject_instances
from nro.configuration.store import ConfigStore
from nro.orchestration.worker import _runner_graph_signature
from nro.func.contracts import final_resampling_contract


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


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
        command=("python", "-m", "nro.func"),
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
        command=("python", "-m", "nro.anat"),
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
        (instance.evolve(command=("python", "-m", "nro.anat", "--verbose")),)
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
        command=("python", "-m", "nro.anat"),
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

    changed_command = instance.evolve(
        command=("python", "-m", "nro.anat", "--verbose")
    )
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
    assert tuple(json.loads(row["expected_outputs_json"])) == (
        str(changed_output.resolve()),
    )
    assert row["output_root"] == str(changed_output.parent.resolve())
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Instance contract changed"


def test_active_demand_pins_its_execution_recipe(tmp_path: Path) -> None:
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
        command=("python", "-m", "nro.anat"),
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

    registry.register_instances(
        (instance.evolve(command=(*instance.command, "--verbose")),)
    )

    row = registry.instance_rows()[0]
    assert tuple(json.loads(row["command_json"])) == instance.command


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
        command=("python", "-m", "nro.anat"),
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

    row = next(
        item for item in registry.instance_rows() if item["instance_key"] == base.key
    )
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
    (configs / "microparcellation" / "rest_microparcellation.yml").write_text(
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
    assert by_module["func"][0].entities == {"task": "rest", "dir": "LR", "run": "1"}
    assert by_module["clean"][0].entities == {
        "task": "rest", "dir": "LR", "run": "1",
        "space": "fsnative", "smoothing": "2",
    }
    assert by_module["microparcellation"][0].entities == {
        "space": "fsnative", "smoothing": "2"
    }
    assert by_module["microparcellation"][0].dependencies == (by_module["clean"][0].key,)
    micro_outputs = by_module["microparcellation"][0].expected_outputs
    assert micro_outputs[0].name == "sub-01_space-fsnative_scale-2mm_manifest.yaml"
    assert micro_outputs[1].name.endswith("_desc-quality_metrics.json")
    assert micro_outputs[2].name.endswith("_desc-microparcellation_manifest.json")
    assert by_module["networks"][0].dependencies == (
        by_module["microparcellation"][0].key,
        by_module["anat"][0].key,
    )
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
        ("fsnative", "0"), ("fsnative", "2"),
        ("T1w", "0"), ("T1w", "2"),
    }

    assert len(by_module["anat"]) == 1
    assert len(by_module["func"]) == 1
    assert {
        (item.entities["space"], item.entities["smoothing"])
        for item in by_module["clean"]
    } == expected_pairs
    assert {
        (item.entities["space"], item.entities["smoothing"])
        for item in by_module["microparcellation"]
    } == expected_pairs
    assert {
        (item.entities["space"], item.entities["smoothing"])
        for item in by_module["networks"]
    } == expected_pairs
    clean_keys = {
        (item.entities["space"], item.entities["smoothing"]): item.key
        for item in by_module["clean"]
    }
    for item in by_module["microparcellation"]:
        pair = (item.entities["space"], item.entities["smoothing"])
        assert item.dependencies == (clean_keys[pair],)
