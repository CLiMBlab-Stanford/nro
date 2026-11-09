"""Registry queries must not bind one SQLite parameter per work item.

Large sites register tens of thousands of work items. SQLite rejects any
statement with more bound parameters than its variable limit, so these tests
lower that limit far below the number of work items and run the scheduler
paths that select, claim, invalidate, and purge many items at once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_orchestration_worker import _spec

from nro.definitions.store import ConfigStore
from nro.orchestration import dependency_state
from nro.orchestration.artifact_validation import assess_cached_artifacts
from nro.orchestration.completion_records import completion_records
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.registry_work_items import (
    forget_purged_work_items_detailed,
    retained_dependency_modules,
)
from nro.orchestration.sql_values import JSON_VALUES, json_values

_VARIABLE_LIMIT = 64
_PAIRS = 150


@pytest.fixture
def low_variable_limit(monkeypatch):
    original = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _VARIABLE_LIMIT)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)


def _large_request(tmp_path: Path):
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    specs = []
    for index in range(_PAIRS):
        anat = _spec(
            key=f"anat:{index:064d}",
            module="anat",
            lineage=registered.lineages["anat"],
            config_fingerprint=workflow.configuration("anat").fingerprint,
            runtime_config=registry.runtime_config_path(registered, "anat"),
            output=tmp_path / "out" / f"anat-{index}" / "anat.txt",
        )
        networks = _spec(
            key=f"networks:{index:064d}",
            module="networks",
            lineage=registered.lineages["networks"],
            config_fingerprint=workflow.configuration("networks").fingerprint,
            runtime_config=registry.runtime_config_path(registered, "networks"),
            output=tmp_path / "out" / f"networks-{index}" / "networks.txt",
            dependencies=(anat.key,),
        )
        specs.extend((anat, networks))
    terminals = tuple(spec.key for spec in specs if spec.module == "networks")
    request = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=tuple(specs),
        terminal_work_item_keys=terminals,
        concurrency=4,
        partition=None,
    )
    return registry, specs, request


def test_json_values_round_trip_integer_and_text_columns() -> None:
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    database.executemany("INSERT INTO items VALUES (?,?)", [(1, "1"), (2, "b"), (3, "c")])

    def ids(column: str, values) -> list[int]:
        return [
            row[0]
            for row in database.execute(
                f"SELECT id FROM items WHERE {column} IN ({JSON_VALUES}) ORDER BY id",
                (json_values(values),),
            )
        ]

    assert ids("id", [1, 3]) == [1, 3]
    assert ids("name", ["1", "c"]) == [1, 3]
    assert ids("id", []) == []
    assert [
        row[0]
        for row in database.execute(
            f"SELECT id FROM items WHERE id NOT IN ({JSON_VALUES}) ORDER BY id",
            (json_values([]),),
        )
    ] == [1, 2, 3]
    with pytest.raises(TypeError):
        json_values([1.5])


def test_large_request_scheduling_respects_sqlite_variable_limit(
    tmp_path: Path, low_variable_limit
) -> None:
    registry, specs, request = _large_request(tmp_path)

    assert registry.worker_capacity_needed(request_id=request, resource_class="large", memory_gb=32)
    registry.register_worker("cpu", resource_class="large")
    claimed = registry.claim_ready_work_item("cpu", ("large",))
    assert claimed is not None

    ids = registry.work_item_ids(tuple(spec.key for spec in specs))
    assert len(ids) == len(specs)
    anat_ids = [ids[spec.key] for spec in specs if spec.module == "anat"]
    with registry.connection(write=True) as db:
        dependency_state.synchronize(db, now=utcnow())
        dependency_state.invalidate(db, anat_ids, now=utcnow(), reason="test")
        assert completion_records(db, tuple(ids.values())) == {}
    assert assess_cached_artifacts(registry, ids.values()) == set()
    assert len(registry.work_item_status_snapshot()) == len(specs)

    registry.finish_attempt(claimed.attempt_id, state="cancelled")
    cancellation = registry.request_cancellation(modules=("networks", "anat"))
    assert cancellation["requests"] == 1


def test_large_purge_respects_sqlite_variable_limit(tmp_path: Path, low_variable_limit) -> None:
    registry, specs, _request = _large_request(tmp_path)
    ids = registry.work_item_ids(tuple(spec.key for spec in specs))
    registry.request_cancellation(modules=("networks", "anat"))

    with registry.connection(write=True) as db:
        assert retained_dependency_modules(db, tuple(ids.values())) == {}
        deleted, retained = forget_purged_work_items_detailed(db, tuple(ids.values()))

    assert sorted(deleted) == sorted(ids.values())
    assert retained == ()
