from __future__ import annotations

import sqlite3
from pathlib import Path

from nro.bin.log import build_parser as log_parser
from nro.orchestration.catalog import BUILTIN_MODULES, MODULES
from nro.orchestration.contracts import (
    ExecutionEnvelope,
    ExecutionRecipe,
    InstanceContract,
    InstanceIdentity,
    InstanceSpec,
    ResourceRequest,
)
from nro.orchestration.registry import Registry
from nro.bin.purge import build_parser as purge_parser
from nro.bin.run import build_parser as run_parser
from nro.bin.status import build_parser as status_parser
from nro.bin.stop import build_parser as stop_parser
from nro.configuration.store import ConfigStore, DERIVATIVE_CLASSES
from nro.configuration.paths import REGISTRY_PATH


ROOT = Path(__file__).parents[1]


def test_code_defines_the_canonical_system_vocabulary() -> None:
    assert DERIVATIVE_CLASSES == (
        "preprocessing",
        "clean",
        "microparcellation",
        "networks",
    )
    assert MODULES == ("anat", "func", "clean", "microparcellation", "networks")
    assert tuple(descriptor.name for descriptor in BUILTIN_MODULES) == MODULES


def test_instance_spec_has_the_documented_contract_hierarchy() -> None:
    annotations = InstanceSpec.__annotations__
    assert annotations["identity"] == "InstanceIdentity"
    assert annotations["contract"] == "InstanceContract"
    assert annotations["execution"] == "ExecutionRecipe"
    assert annotations["resources"] == "ResourceRequest"
    assert ExecutionEnvelope.from_registry_row
    assert all(
        value.__doc__
        for value in (
            InstanceIdentity,
            InstanceContract,
            ExecutionRecipe,
            ResourceRequest,
            InstanceSpec,
            ExecutionEnvelope,
        )
    )


def test_module_specific_planning_lives_with_each_scientific_module() -> None:
    assert not (ROOT / "nro" / "orchestration" / "instances.py").exists()
    for module in MODULES:
        assert (ROOT / "nro" / module / "planning.py").is_file()
    planner_source = (ROOT / "nro" / "orchestration" / "planner.py").read_text()
    for module in MODULES:
        assert f"from nro.{module}" not in planner_source


def test_control_commands_select_modules_consistently() -> None:
    for parser_factory in (
        run_parser,
        status_parser,
        stop_parser,
        log_parser,
        purge_parser,
    ):
        parser = parser_factory()
        actions = {action.dest: action for action in parser._actions}
        assert actions["module"].option_strings == ["-m", "--module"]
        assert "job" not in actions


def test_control_commands_share_selection_vocabulary_but_keep_local_options() -> None:
    expected = {
        "participant": ["-p", "--participant"],
        "project": ["-P", "--project"],
        "module": ["-m", "--module"],
        "workflow": ["-w", "--workflow"],
        "run": ["-r", "--run"],
        "space": ["-s", "--space"],
        "smoothing": ["-S", "--smoothing"],
    }
    for parser_factory in (run_parser, status_parser, stop_parser, log_parser, purge_parser):
        actions = {action.dest: action.option_strings for action in parser_factory()._actions}
        assert {name: actions[name] for name in expected} == expected

    assert "instance_level" in {
        action.dest for action in log_parser()._actions
    }
    assert "instance_level" not in {
        action.dest for action in status_parser()._actions
    }
    purge_actions = {action.dest: action for action in purge_parser()._actions}
    assert purge_actions["logs"].option_strings == ["-l", "--logs"]
    assert purge_actions["force"].option_strings == ["-f", "--force"]
    assert "-p" not in {
        option
        for action in run_parser()._actions
        if action.dest == "partition"
        for option in action.option_strings
    }


def test_primary_user_commands_live_in_bin() -> None:
    for name in ("run", "status", "stop", "log", "set", "purge", "publish", "qc"):
        assert (ROOT / "nro" / "bin" / f"{name}.py").is_file()
        assert not (ROOT / "nro" / f"{name}.py").exists()
        if name != "publish":
            assert not (ROOT / "nro" / "orchestration" / f"{name}.py").exists()
    publication_api = (ROOT / "nro" / "orchestration" / "publish.py").read_text()
    assert "def main(" not in publication_api
    assert (ROOT / "nro" / "qc" / "__main__.py").is_file()
    assert (ROOT / "nro" / "qc" / "engine.py").is_file()
    assert not (ROOT / "nro" / "qc.py").exists()
    assert not (ROOT / "nro" / "registration_qc").exists()


def test_default_registry_is_lab_wide() -> None:
    registry = Registry.for_project("nptl")
    assert registry.paths.control == REGISTRY_PATH
    assert registry.paths.control == Path("/juice6/u/nlp/climblab/.nro")


def test_workflow_api_exposes_configurations_by_derivative_class() -> None:
    workflow = ConfigStore().resolve("main")

    assert set(workflow.configurations) == set(DERIVATIVE_CLASSES)
    assert workflow.configuration("preprocessing").derivative_class == "preprocessing"
    assert not hasattr(workflow, "stages")


def test_fresh_registry_uses_class_and_module_columns(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()

    with sqlite3.connect(registry.paths.database) as connection:
        columns = {
            table: {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for table in (
                "configuration_lineages", "workflow_bindings", "requests", "instances"
            )
        }

    assert "derivative_class" in columns["configuration_lineages"]
    assert "derivative_class" in columns["workflow_bindings"]
    assert "target_module" in columns["requests"]
    assert "module" in columns["instances"]
    assert all("stage" not in table_columns for table_columns in columns.values())
