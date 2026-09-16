from __future__ import annotations

import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import yaml

from nro.configuration.runtime import load_runtime_configuration
from nro.configuration.store import (
    CONFIGURATION_CLASSES,
    ConfigStore,
    WorkflowError,
)
from nro.modules.anat.__main__ import build_parser as anat_parser
from nro.modules.clean.__main__ import build_parser as clean_parser
from nro.modules.func.__main__ import build_parser as func_parser
from nro.modules.microparcellation.__main__ import build_parser as microparcellation_parser
from nro.modules.networks.__main__ import build_parser as networks_parser
from nro.orchestration.branch_admission import _workflow as import_workflow
from nro.orchestration.compiled_request import export_workflow
from nro.orchestration.planner import Planner
from nro.orchestration.registry import APPLICATION_ID, SCHEMA_VERSION, Registry
from nro.orchestration.runtime import select_runtime_config
from nro.qc.registration import build_parser as registration_parser


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _test_store(path: Path) -> ConfigStore:
    repository_store = ConfigStore().root
    shutil.copytree(repository_store, path, dirs_exist_ok=True)
    store = ConfigStore()
    store.root = path
    return store


def _register_main_concurrently(bids_root: str) -> tuple[int, int]:
    registry = Registry.for_project("demo", bids_root=bids_root)
    result = registry.register_workflow(ConfigStore().resolve("main"))
    return result.revision_id, result.lineages["networks"]


def test_repository_main_workflow_resolves() -> None:
    resolved = ConfigStore().resolve("main")

    assert resolved.selections == {
        configuration_class: "main" for configuration_class in CONFIGURATION_CLASSES
    }
    assert resolved.configurations["microparcellation"].values["input_filter"] == {}
    assert resolved.path.parent.name == "workflows"
    assert resolved.configurations["anat"].path.parent.name == "anat"
    assert resolved.configurations["func"].path.parent.name == "func"


def test_registry_requires_the_current_schema_without_implicit_migration(
    tmp_path: Path,
) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()
    with sqlite3.connect(registry.paths.database) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION - 1}")
        connection.commit()

    with pytest.raises(
        RuntimeError,
        match="This development build does not migrate registries",
    ):
        registry.initialize()


def test_registry_reinitialize_restores_original_state_if_rebuild_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()
    marker = registry.paths.database.parent / "original-state"
    marker.write_text("preserve on failure")
    original_database = registry.paths.database.read_bytes()

    def fail() -> None:
        raise RuntimeError("injected rebuild failure")

    monkeypatch.setattr(registry, "_initialize_locked", fail)
    with pytest.raises(RuntimeError, match="injected rebuild failure"):
        registry.reinitialize()

    assert marker.read_text() == "preserve on failure"
    assert registry.paths.database.read_bytes() == original_database
    assert not tuple(registry.paths.control.glob(".repair-*"))


def test_workflow_paths_are_rejected_even_when_file_exists(tmp_path: Path) -> None:
    external = tmp_path / "external_workflow.yml"
    _write_yaml(external, {})

    with pytest.raises(WorkflowError, match="Invalid workflow ID"):
        ConfigStore().resolve(str(external))


@pytest.mark.parametrize(
    "parser_factory",
    (
        anat_parser,
        func_parser,
        clean_parser,
        microparcellation_parser,
        networks_parser,
    ),
)
def test_module_clis_select_configs_only_through_workflows(parser_factory) -> None:
    parser = parser_factory()
    args = parser.parse_args(["-p", "01", "-P", "demo", "-w", "experiment"])
    assert args.workflow == "experiment"
    with pytest.raises(SystemExit):
        parser.parse_args(["-p", "01", "-P", "demo", "--config", "external.yml"])


def test_registration_qc_retains_its_qctype_specific_interface() -> None:
    parser = registration_parser()
    args = parser.parse_args(["01", "-p", "demo", "-w", "experiment"])
    assert args.workflow == "experiment"
    with pytest.raises(SystemExit):
        parser.parse_args(["01", "-p", "demo", "--config", "external.yml"])


def test_private_runtime_environment_rejects_external_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external_func.yml"
    _write_yaml(external, {})
    monkeypatch.setenv("NRO_RUNTIME_CONFIG", str(external))

    with pytest.raises(ValueError, match="outside the central registry"):
        select_runtime_config(
            project="demo",
            workflow_id="main",
            configuration_class="func",
            bids_root=tmp_path / "bids",
        )


def test_store_reads_only_organized_configuration_files(tmp_path: Path) -> None:
    (tmp_path / "workflows").mkdir()
    (tmp_path / "configs" / "anat").mkdir(parents=True)
    _write_yaml(
        tmp_path / "workflows" / "experiment_workflow.yml",
        {"anat": "experiment"},
    )
    _write_yaml(
        tmp_path / "configs" / "anat" / "experiment_anat.yml",
        {"verbose": True},
    )
    resolved = _test_store(tmp_path).resolve("experiment")

    assert resolved.configurations["anat"].values["verbose"] is True
    assert resolved.path.parent.name == "workflows"


def test_workflow_defaults_omitted_classes_and_rejects_upstream_keys(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "workflows" / "experiment_workflow.yml", {"func": "experiment"})
    _write_yaml(tmp_path / "configs" / "func" / "experiment_func.yml", {})
    store = _test_store(tmp_path)
    resolved = store.resolve("experiment")

    assert resolved.selections["func"] == "experiment"
    assert all(
        resolved.selections[configuration_class] == "main"
        for configuration_class in set(CONFIGURATION_CLASSES) - {"func"}
    )

    _write_yaml(tmp_path / "workflows" / "bad_workflow.yml", {"clean": "bad"})
    _write_yaml(
        tmp_path / "configs" / "clean" / "bad_clean.yml",
        {"func_directory": "not-allowed"},
    )
    with pytest.raises(WorkflowError, match="managed by orchestration"):
        store.resolve("bad")


def test_func_space_selection_is_not_configuration(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "workflows" / "mismatch_workflow.yml",
        {"func": "mismatch"},
    )
    _write_yaml(
        tmp_path / "configs" / "func" / "mismatch_func.yml",
        {"fsaverage_template": "fsaverage", "output_spaces": ["fsaverage"]},
    )

    with pytest.raises(WorkflowError, match="fsaverage_template"):
        _test_store(tmp_path).resolve("mismatch")


def test_missing_referenced_config_is_an_error(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "workflows" / "missing_workflow.yml", {"networks": "does-not-exist"})

    with pytest.raises(WorkflowError, match="does-not-exist_networks.yml"):
        _test_store(tmp_path).resolve("missing")


def test_registry_reuses_config_lineage_across_workflows(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    _write_yaml(configs / "workflows" / "experiment_workflow.yml", {"func": "experiment"})
    _write_yaml(configs / "configs" / "func" / "experiment_func.yml", {})
    _write_yaml(
        configs / "workflows" / "experiment_nogsr_workflow.yml",
        {"func": "experiment", "clean": "nogsr"},
    )
    _write_yaml(configs / "configs" / "clean" / "nogsr_clean.yml", {"standardize": False})

    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)
    first = registry.register_workflow(store.resolve("experiment"))
    second = registry.register_workflow(store.resolve("experiment_nogsr"))

    assert first.revision == 1
    assert first.directories["anat"] == "main"
    assert first.directories["func"] == "experiment"
    assert first.directories["clean"] == "main"
    assert second.lineages["anat"] == first.lineages["anat"]
    assert second.lineages["func"] == first.lineages["func"]
    assert second.directories["clean"] == "nogsr"
    assert second.directories["networks"] == "main-2"


def test_func_variants_share_one_anatomical_work_item(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    (subject / "anat").mkdir(parents=True)
    (subject / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"anat")
    (subject / "func").mkdir()
    bold = subject / "func" / "sub-01_task-rest_bold.nii.gz"
    bold.write_bytes(b"bold")
    bold.with_name("sub-01_task-rest_bold.json").write_text(
        '{"RepetitionTime": 2.0}', encoding="utf-8"
    )
    _write_yaml(configs / "workflows" / "variant_workflow.yml", {"func": "variant"})
    _write_yaml(configs / "configs" / "func" / "variant_func.yml", {"clean_ica_aroma": False})

    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)
    main_workflow = store.resolve("main")
    variant_workflow = store.resolve("variant")
    main = registry.register_workflow(main_workflow)
    variant = registry.register_workflow(variant_workflow)

    assert main.lineages["func"] != variant.lineages["func"]
    assert main.directories["func"] == "main"
    assert variant.directories["func"] == "variant"
    assert variant.lineages["anat"] == main.lineages["anat"]
    assert variant.directories["anat"] == main.directories["anat"] == "main"

    planner = Planner(registry, bids_root=bids)
    main_specs = planner.plan_subject(
        project="demo",
        participant="01",
        module="func",
        workflow=main_workflow,
        registered=main,
    )
    variant_specs = planner.plan_subject(
        project="demo",
        participant="01",
        module="func",
        workflow=variant_workflow,
        registered=variant,
    )
    main_anat = next(spec for spec in main_specs if spec.module == "anat")
    variant_anat = next(spec for spec in variant_specs if spec.module == "anat")
    assert variant_anat.key == main_anat.key
    assert variant_anat.output_root == main_anat.output_root

    central = Registry.for_project("demo", bids_root=tmp_path / "central" / "bids")
    exported = export_workflow(registry, variant)
    assert any(
        binding["configuration_class"] == "anat"
        and binding["module_lineage_id"] == variant.lineages["anat"]
        for binding in exported["bindings"]
    )
    with central.connection(write=True) as db:
        _revision, lineage_mapping = import_workflow(
            db,
            exported,
            "dev-owner",
        )
    assert variant.lineages["anat"] in lineage_mapping


def test_workflow_mutation_allocates_numeric_revision_and_reuses_prefix(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    workflow_path = configs / "workflows" / "experiment_workflow.yml"
    _write_yaml(workflow_path, {"func": "experiment"})
    _write_yaml(configs / "configs" / "func" / "experiment_func.yml", {})
    _write_yaml(configs / "configs" / "clean" / "nogsr_clean.yml", {"standardize": False})

    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)
    registry.register_workflow(store.resolve("experiment"))
    _write_yaml(
        workflow_path,
        {"func": "experiment", "clean": "nogsr"},
    )
    second = registry.register_workflow(store.resolve("experiment"))
    repeated = registry.register_workflow(store.resolve("experiment"))

    assert second.revision == 2
    assert second.directories["func"] == "experiment"
    assert second.directories["clean"] == "nogsr"
    assert second.directories["networks"] == "main-2"
    assert repeated.revision_id == second.revision_id
    assert not repeated.created
    assert len(registry.workflow_history("experiment")) == 2
    snapshot = registry.paths.workflows / "experiment" / "2_workflow.yml"
    assert snapshot.is_file()


def test_all_main_lineage_reserves_main_directory(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(ConfigStore().resolve("main"))

    assert registered.directories == {
        configuration_class: "main" for configuration_class in CONFIGURATION_CLASSES
    }

    anat_id, _ = load_runtime_configuration(
        registry.runtime_config_path(registered, "anat"), "anat"
    )
    func_id, func = load_runtime_configuration(
        registry.runtime_config_path(registered, "func"), "func"
    )
    clean_id, clean = load_runtime_configuration(
        registry.runtime_config_path(registered, "clean"), "clean"
    )
    micro_id, micro = load_runtime_configuration(
        registry.runtime_config_path(registered, "microparcellation"),
        "microparcellation",
    )
    networks_id, networks = load_runtime_configuration(
        registry.runtime_config_path(registered, "networks"), "networks"
    )
    assert (anat_id, func_id, clean_id, micro_id, networks_id) == ("main",) * 5
    assert func["anat_directory"] == "main"
    assert clean["func_directory"] == "main"
    assert clean["anat_directory"] == "main"
    assert micro["clean_directory"] == "main"
    assert networks["microparcellation_directory"] == "main"


def test_evolved_main_configuration_reuses_its_named_directory(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    first = registry.register_workflow(store.resolve("main"))

    main_path = configs / "configs" / "microparcellation" / "main_microparcellation.yml"
    changed_main = {"coarsening": {"target_vertices": 30000}}
    _write_yaml(main_path, changed_main)
    second = registry.register_workflow(store.resolve("main"))

    assert second.revision == 2
    assert second.directories == {
        configuration_class: "main" for configuration_class in CONFIGURATION_CLASSES
    }
    assert second.lineages == first.lineages


def test_changed_named_config_gets_new_lineage_reused_by_contents(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    _write_yaml(configs / "workflows" / "experiment_workflow.yml", {"func": "alternate"})
    _write_yaml(configs / "workflows" / "same_content_workflow.yml", {"func": "alternate"})
    _write_yaml(configs / "configs" / "func" / "alternate_func.yml", {})
    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    first = registry.register_workflow(store.resolve("experiment"))

    _write_yaml(
        configs / "configs" / "func" / "alternate_func.yml",
        {"output_grid": "t1_native"},
    )
    changed = registry.register_workflow(store.resolve("experiment"))
    matching = registry.register_workflow(store.resolve("same_content"))

    assert first.directories["func"] == "alternate"
    assert changed.directories == first.directories
    assert matching.directories == changed.directories
    assert matching.lineages == changed.lineages


def test_registry_bootstrap_and_registration_are_process_safe(tmp_path: Path) -> None:
    bids = str(tmp_path / "bids")
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_register_main_concurrently, [bids] * 8))

    assert len(set(results)) == 1
    registry = Registry.for_project("demo", bids_root=bids)
    assert len(registry.workflow_history("main")) == 1


def test_module_directories_use_independent_configuration_namespaces(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    _write_yaml(configs / "workflows" / "experiment-2_workflow.yml", {"func": "alternate"})
    _write_yaml(configs / "configs" / "func" / "alternate_func.yml", {})
    workflow_path = configs / "workflows" / "experiment_workflow.yml"
    _write_yaml(workflow_path, {"func": "first"})
    _write_yaml(configs / "configs" / "func" / "first_func.yml", {})
    _write_yaml(
        configs / "configs" / "func" / "second_func.yml",
        {"output_grid": "t1_native"},
    )
    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)

    occupied = registry.register_workflow(store.resolve("experiment-2"))
    first = registry.register_workflow(store.resolve("experiment"))
    _write_yaml(workflow_path, {"func": "second"})
    second = registry.register_workflow(store.resolve("experiment"))

    assert occupied.directories["func"] == "alternate"
    assert first.directories["func"] == "first"
    assert second.revision == 2
    assert second.directories["func"] == "second"
