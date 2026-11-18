"""Queued work runs with its installation's current interpreter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nro.definitions.store import ConfigStore
from nro.orchestration import interpreters
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.registry import Registry


@pytest.fixture(autouse=True)
def _fresh_cache():
    interpreters._cache.clear()
    yield
    interpreters._cache.clear()


def _installation(root: Path, environment: str, *, checkout: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    python = root / ".nro-environments" / environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    (root / ".nro-installation.json").write_text(
        json.dumps({"checkout": str(checkout or root), "environment": str(python.parents[1])})
    )
    return python


def test_replaced_environment_resolves_to_the_current_interpreter(tmp_path: Path) -> None:
    checkout = tmp_path / "nro"
    current = _installation(checkout, "dependencies-new")
    removed = checkout / ".nro-environments/dependencies-old/bin/python"

    assert interpreters.current_interpreter(str(removed)) == str(current)
    command = (str(removed), str(checkout / "source_launcher.py"), "digest", "-", "-", "nro.x")
    assert interpreters.with_current_interpreter(command)[0] == str(current)
    module_command = (str(removed), "-m", "nro.modules.anat", "--flag")
    assert interpreters.with_current_interpreter(module_command) == (
        str(current),
        "-m",
        "nro.modules.anat",
        "--flag",
    )


@pytest.mark.parametrize("case", ["no record", "foreign record", "missing environment"])
def test_unowned_interpreters_are_left_unchanged(tmp_path: Path, case: str) -> None:
    checkout = tmp_path / "nro"
    interpreter = str(checkout / ".nro-environments/dependencies-old/bin/python")
    if case == "foreign record":
        _installation(checkout, "dependencies-new", checkout=tmp_path / "elsewhere")
    elif case == "missing environment":
        python = _installation(checkout, "dependencies-new")
        python.unlink()
    else:
        checkout.mkdir()

    assert interpreters.current_interpreter(interpreter) == interpreter


def test_other_commands_are_left_unchanged(tmp_path: Path) -> None:
    checkout = tmp_path / "nro"
    _installation(checkout, "dependencies-new")
    removed = str(checkout / ".nro-environments/dependencies-old/bin/python")
    for command in ((removed, "-c", "pass"), ("/bin/echo", "x"), (removed,)):
        assert interpreters.with_current_interpreter(command) == command


def test_claim_runs_queued_work_with_the_replaced_environment(tmp_path: Path) -> None:
    checkout = tmp_path / "nro"
    current = _installation(checkout, "dependencies-new")
    removed = checkout / ".nro-environments/dependencies-old/bin/python"
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "anat.txt"
    spec = WorkItemSpec.create(
        key="anat:" + "0" * 64,
        module="anat",
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        directory_label=registered.directory_for("anat"),
        runtime_config=registry.runtime_config_path(registered, "anat"),
        command=(str(removed), "-m", "nro.modules.anat"),
        dependencies=(),
        input_paths=(),
        output_root=output.parent,
        output_prefix=None,
        resource_class="large",
        expected_outputs=(output,),
        processing=module_descriptor("anat").processing_contract(),
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(spec,),
        terminal_work_item_keys=(spec.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("worker", resource_class="large")

    claimed = registry.claim_ready_work_item("worker", ("large",))
    recovered = registry.current_worker_assignment("worker")

    assert claimed.execution.command == (str(current), "-m", "nro.modules.anat")
    assert recovered.execution.command == claimed.execution.command
    assert sys.executable not in claimed.execution.command
