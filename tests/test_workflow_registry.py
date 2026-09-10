from __future__ import annotations

import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import yaml

from nro.configuration.runtime import load_runtime_configuration
from nro.configuration.store import DERIVATIVE_CLASSES, ConfigStore, WorkflowError
from nro.modules.anat.__main__ import build_parser as anat_parser
from nro.modules.clean.__main__ import build_parser as clean_parser
from nro.modules.func.__main__ import build_parser as func_parser
from nro.modules.microparcellation.__main__ import build_parser as microparcellation_parser
from nro.modules.networks.__main__ import build_parser as networks_parser
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
        derivative_class: "main" for derivative_class in DERIVATIVE_CLASSES
    }
    assert resolved.configurations["microparcellation"].values["input_filter"] == {}
    assert resolved.path.parent.name == "workflows"
    assert resolved.configurations["preprocessing"].path.parent.name == "preprocessing"


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
    external = tmp_path / "external_preprocess.yml"
    _write_yaml(external, {})
    monkeypatch.setenv("NRO_RUNTIME_CONFIG", str(external))

    with pytest.raises(ValueError, match="outside the central registry"):
        select_runtime_config(
            project="demo",
            workflow_id="main",
            derivative_class="preprocessing",
            bids_root=tmp_path / "bids",
        )


def test_store_reads_only_organized_configuration_files(tmp_path: Path) -> None:
    (tmp_path / "workflows").mkdir()
    (tmp_path / "configs" / "preprocessing").mkdir(parents=True)
    _write_yaml(
        tmp_path / "workflows" / "experiment_workflow.yml",
        {"preprocessing": "experiment"},
    )
    _write_yaml(
        tmp_path / "configs" / "preprocessing" / "experiment_preprocessing.yml",
        {"anat": {"nthreads": 7}},
    )
    resolved = _test_store(tmp_path).resolve("experiment")

    assert resolved.configurations["preprocessing"].values["anat"]["nthreads"] == 7
    assert resolved.path.parent.name == "workflows"


def test_workflow_defaults_omitted_classes_and_rejects_upstream_keys(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "workflows" / "experiment_workflow.yml", {"preprocessing": "experiment"})
    _write_yaml(tmp_path / "configs" / "preprocessing" / "experiment_preprocessing.yml", {})
    store = _test_store(tmp_path)
    resolved = store.resolve("experiment")

    assert resolved.selections["preprocessing"] == "experiment"
    assert all(
        resolved.selections[derivative_class] == "main"
        for derivative_class in DERIVATIVE_CLASSES[1:]
    )

    _write_yaml(tmp_path / "workflows" / "bad_workflow.yml", {"clean": "bad"})
    _write_yaml(
        tmp_path / "configs" / "clean" / "bad_clean.yml",
        {"preprocessing_directory": "not-allowed"},
    )
    with pytest.raises(WorkflowError, match="belong in a workflow"):
        store.resolve("bad")


def test_missing_referenced_config_is_an_error(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "workflows" / "missing_workflow.yml", {"networks": "does-not-exist"})

    with pytest.raises(WorkflowError, match="does-not-exist_networks.yml"):
        _test_store(tmp_path).resolve("missing")


def test_registry_reuses_config_lineage_across_workflows(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    _write_yaml(configs / "workflows" / "experiment_workflow.yml", {"preprocessing": "experiment"})
    _write_yaml(configs / "configs" / "preprocessing" / "experiment_preprocessing.yml", {})
    _write_yaml(
        configs / "workflows" / "experiment_nogsr_workflow.yml",
        {"preprocessing": "experiment", "clean": "nogsr"},
    )
    _write_yaml(configs / "configs" / "clean" / "nogsr_clean.yml", {"standardize": False})

    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)
    first = registry.register_workflow(store.resolve("experiment"))
    second = registry.register_workflow(store.resolve("experiment_nogsr"))

    assert first.revision == 1
    assert first.directories == {
        derivative_class: "experiment" for derivative_class in DERIVATIVE_CLASSES
    }
    assert second.lineages["preprocessing"] == first.lineages["preprocessing"]
    assert second.directories["preprocessing"] == "experiment"
    assert second.directories["clean"] == "experiment_nogsr"
    assert second.directories["networks"] == "experiment_nogsr"


def test_workflow_mutation_allocates_numeric_revision_and_reuses_prefix(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    workflow_path = configs / "workflows" / "experiment_workflow.yml"
    _write_yaml(workflow_path, {"preprocessing": "experiment"})
    _write_yaml(configs / "configs" / "preprocessing" / "experiment_preprocessing.yml", {})
    _write_yaml(configs / "configs" / "clean" / "nogsr_clean.yml", {"standardize": False})

    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)
    registry.register_workflow(store.resolve("experiment"))
    _write_yaml(
        workflow_path,
        {"preprocessing": "experiment", "clean": "nogsr"},
    )
    second = registry.register_workflow(store.resolve("experiment"))
    repeated = registry.register_workflow(store.resolve("experiment"))

    assert second.revision == 2
    assert second.directories["preprocessing"] == "experiment"
    assert second.directories["clean"] == "experiment-2"
    assert second.directories["networks"] == "experiment-2"
    assert repeated.revision_id == second.revision_id
    assert not repeated.created
    assert len(registry.workflow_history("experiment")) == 2
    snapshot = registry.paths.workflows / "experiment" / "2_workflow.yml"
    assert snapshot.is_file()


def test_all_main_lineage_reserves_main_directory(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(ConfigStore().resolve("main"))

    assert registered.directories == {
        derivative_class: "main" for derivative_class in DERIVATIVE_CLASSES
    }

    preprocess_id, _ = load_runtime_configuration(
        registry.runtime_config_path(registered, "preprocessing"), "preprocessing"
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
    assert (preprocess_id, clean_id, micro_id, networks_id) == ("main",) * 4
    assert clean["preprocessing_directory"] == "main"
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
        derivative_class: "main" for derivative_class in DERIVATIVE_CLASSES
    }
    assert second.lineages == first.lineages


def test_changed_named_config_gets_new_lineage_reused_by_contents(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    _write_yaml(configs / "workflows" / "experiment_workflow.yml", {"preprocessing": "alternate"})
    _write_yaml(configs / "workflows" / "same_content_workflow.yml", {"preprocessing": "alternate"})
    _write_yaml(configs / "configs" / "preprocessing" / "alternate_preprocessing.yml", {})
    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    first = registry.register_workflow(store.resolve("experiment"))

    _write_yaml(
        configs / "configs" / "preprocessing" / "alternate_preprocessing.yml",
        {"anat": {"nthreads": 7}},
    )
    changed = registry.register_workflow(store.resolve("experiment"))
    matching = registry.register_workflow(store.resolve("same_content"))

    assert first.directories == {
        derivative_class: "experiment" for derivative_class in DERIVATIVE_CLASSES
    }
    assert changed.directories == {
        derivative_class: "experiment" for derivative_class in DERIVATIVE_CLASSES
    }
    assert matching.directories == changed.directories
    assert matching.lineages == changed.lineages


def test_registry_bootstrap_and_registration_are_process_safe(tmp_path: Path) -> None:
    bids = str(tmp_path / "bids")
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_register_main_concurrently, [bids] * 8))

    assert len(set(results)) == 1
    registry = Registry.for_project("demo", bids_root=bids)
    assert len(registry.workflow_history("main")) == 1


def test_numeric_directory_names_skip_existing_workflow_name(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    bids = tmp_path / "bids"
    _write_yaml(configs / "workflows" / "experiment-2_workflow.yml", {"preprocessing": "alternate"})
    _write_yaml(configs / "configs" / "preprocessing" / "alternate_preprocessing.yml", {})
    workflow_path = configs / "workflows" / "experiment_workflow.yml"
    _write_yaml(workflow_path, {"preprocessing": "first"})
    _write_yaml(configs / "configs" / "preprocessing" / "first_preprocessing.yml", {})
    _write_yaml(
        configs / "configs" / "preprocessing" / "second_preprocessing.yml",
        {"anat": {"nthreads": 7}},
    )
    store = _test_store(configs)
    registry = Registry.for_project("demo", bids_root=bids)

    occupied = registry.register_workflow(store.resolve("experiment-2"))
    first = registry.register_workflow(store.resolve("experiment"))
    _write_yaml(workflow_path, {"preprocessing": "second"})
    second = registry.register_workflow(store.resolve("experiment"))

    assert occupied.directories["preprocessing"] == "experiment-2"
    assert first.directories["preprocessing"] == "experiment"
    assert second.revision == 2
    assert second.directories["preprocessing"] == "experiment-3"
