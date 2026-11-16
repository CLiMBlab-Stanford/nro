"""Cached claim candidates never bypass the claim predicate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nro.definitions.store import ConfigStore
from nro.orchestration import registry as registry_module
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.registry import Registry


def _registry_with_items(tmp_path: Path, count: int) -> tuple[Registry, list[str]]:
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    specs = []
    for index in range(count):
        output = tmp_path / f"anat-{index}.txt"
        specs.append(
            WorkItemSpec.create(
                key=f"anat:{index:064d}",
                module="anat",
                project="demo",
                participant=f"{index:02d}",
                entities={},
                scope="subject",
                module_lineage_id=registered.lineages["anat"],
                config_fingerprint=workflow.configuration("anat").fingerprint,
                directory_label=registered.directory_for("anat"),
                runtime_config=registry.runtime_config_path(registered, "anat"),
                command=(sys.executable, "-c", "pass"),
                dependencies=(),
                input_paths=(),
                output_root=output.parent,
                output_prefix=None,
                resource_class="large",
                expected_outputs=(output,),
                processing=module_descriptor("anat").processing_contract(),
            )
        )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=tuple(specs),
        terminal_work_item_keys=tuple(spec.key for spec in specs),
        concurrency=count,
        partition=None,
    )
    for index in range(count):
        registry.register_worker(f"worker-{index}", resource_class="large")
    return registry, [spec.key for spec in specs]


def _set_state(registry: Registry, key: str, state: str) -> None:
    with registry.connection(write=True) as database:
        database.execute(
            "UPDATE work_items SET artifact_state=? WHERE work_item_key=?", (state, key)
        )


@pytest.fixture(autouse=True)
def _empty_candidate_cache():
    registry_module._CLAIM_CANDIDATES.clear()
    yield
    registry_module._CLAIM_CANDIDATES.clear()


def test_cached_candidate_that_became_ineligible_is_skipped(tmp_path: Path) -> None:
    registry, keys = _registry_with_items(tmp_path, 3)
    with registry.scheduler_session():
        first = registry.claim_ready_work_item("worker-0", ("large",))
        assert first is not None and first.work_item_key == keys[0]
        _set_state(registry, keys[1], "fresh")
        second = registry.claim_ready_work_item("worker-1", ("large",))
        assert second is not None and second.work_item_key == keys[2]
        assert registry.claim_ready_work_item("worker-2", ("large",)) is None


def test_exhausted_cache_rechecks_for_newly_ready_work(tmp_path: Path) -> None:
    registry, keys = _registry_with_items(tmp_path, 3)
    _set_state(registry, keys[2], "fresh")
    with registry.scheduler_session():
        assert registry.claim_ready_work_item("worker-0", ("large",)).work_item_key == keys[0]
        assert registry.claim_ready_work_item("worker-1", ("large",)).work_item_key == keys[1]
        _set_state(registry, keys[2], "stale")
        third = registry.claim_ready_work_item("worker-2", ("large",))
        assert third is not None and third.work_item_key == keys[2]


def test_claims_outside_a_scheduler_session_do_not_cache(tmp_path: Path) -> None:
    registry, keys = _registry_with_items(tmp_path, 2)
    assert registry.claim_ready_work_item("worker-0", ("large",)).work_item_key == keys[0]
    assert registry_module._CLAIM_CANDIDATES == {}
