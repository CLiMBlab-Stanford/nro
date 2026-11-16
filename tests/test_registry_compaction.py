"""Current-state retention for the central scheduler registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nro.definitions.store import ConfigStore
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.registry import Registry, utcnow
from nro.orchestration.registry_compaction import compact_registry
from nro.orchestration.request_plans import decode_plan, encode_plan


def _spec(
    registry: Registry,
    registered,
    workflow,
    *,
    key: str,
    module: str,
    dependencies: tuple[str, ...] = (),
) -> WorkItemSpec:
    output = registry.paths.bids_root / "demo" / "derivatives" / f"{key}.txt"
    return WorkItemSpec.create(
        key=key,
        module=module,
        project="demo",
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=registered.lineages[module],
        config_fingerprint=workflow.configuration(module).fingerprint,
        directory_label=registered.directory_for(module),
        runtime_config=registry.runtime_config_path(registered, module),
        command=(sys.executable, "-c", "pass"),
        dependencies=dependencies,
        input_paths=(),
        output_root=output.parent,
        output_prefix=None,
        resource_class="large",
        expected_outputs=(output,),
        processing=module_descriptor(module).processing_contract(),
    )


def test_compaction_retains_current_state_and_removes_execution_history(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    parent = _spec(
        registry,
        registered,
        workflow,
        key="anat:" + "a" * 64,
        module="anat",
    )
    child = _spec(
        registry,
        registered,
        workflow,
        key="func:" + "b" * 64,
        module="func",
        dependencies=(parent.key,),
    )
    orphan = _spec(
        registry,
        registered,
        workflow,
        key="anat:" + "c" * 64,
        module="anat",
    )
    failed = _spec(
        registry,
        registered,
        workflow,
        key="anat:" + "d" * 64,
        module="anat",
    )
    ids = registry.register_work_items((parent, child, orphan, failed))
    first = registry.create_request(
        registered=registered,
        target_module="func",
        selectors={"spaces": ["fsnative"]},
        work_items=(parent, child),
        terminal_work_item_keys=(child.key,),
        concurrency=2,
        partition=None,
    )
    second = registry.create_request(
        registered=registered,
        target_module="func",
        selectors={"spaces": ["fsnative"]},
        work_items=(parent, child),
        terminal_work_item_keys=(child.key,),
        concurrency=2,
        partition=None,
    )
    full_plan = json.dumps(
        {
            "terminals": [child.key],
            "specifications": [{"key": parent.key}, {"key": child.key}],
            "inherited": [],
        }
    )
    now = utcnow()
    registry.register_worker("old-worker", resource_class="large")
    registry.register_worker("current-worker", resource_class="large")
    registry.register_worker("failed-worker", resource_class="large")
    registry.register_worker("unused-worker", resource_class="large")
    with registry.connection(write=True) as database:
        for request in (first, second):
            database.execute("INSERT INTO request_owners VALUES (?, 'main', 'owner')", (request,))
            database.execute("INSERT INTO request_plans VALUES (?, ?)", (request, full_plan))
            database.execute(
                "INSERT INTO request_artifacts VALUES (?, ?)", (request, ids[child.key])
            )
        database.execute(
            "UPDATE work_items SET artifact_state='fresh' WHERE id=?", (ids[parent.key],)
        )
        database.execute(
            """INSERT INTO attempts(
                   work_item_id,worker_id,state,revision_fingerprint,memory_gb,
                   completed_at,error_type,error_message,log_path,created_at
               ) VALUES (?,?, 'cancelled',?,32,?,'UserCancelled','old','/old.log',?)""",
            (ids[child.key], "old-worker", child.revision_fingerprint, now, now),
        )
        database.execute(
            """INSERT INTO attempts(
                   work_item_id,worker_id,state,revision_fingerprint,memory_gb,
                   started_at,log_path,created_at
               ) VALUES (?,?, 'running',?,32,?,'/current.log',?)""",
            (ids[child.key], "current-worker", child.revision_fingerprint, now, now),
        )
        database.execute(
            """INSERT INTO attempts(
                   work_item_id,worker_id,state,revision_fingerprint,memory_gb,
                   completed_at,error_type,error_message,log_path,created_at
               ) VALUES (?,?, 'error',?,32,?,'ScientificFailure','failed','/failed.log',?)""",
            (ids[failed.key], "failed-worker", failed.revision_fingerprint, now, now),
        )
        database.execute("UPDATE workers SET state='exited'")
        database.execute("UPDATE workers SET state='running' WHERE id='current-worker'")
        database.execute(
            """INSERT INTO scheduler_submissions(
                   intent_token,request_id,resource_class,memory_gb,state,slurm_job_id,created_at
               ) VALUES ('finished',?,'large',32,'complete','1',?)""",
            (first, now),
        )
        database.execute(
            """INSERT INTO scheduler_submissions(
                   intent_token,request_id,resource_class,memory_gb,state,slurm_job_id,created_at
               ) VALUES ('active',?,'large',32,'running','2',?)""",
            (second, now),
        )
        database.execute(
            """INSERT INTO scheduler_requests VALUES (
                   'expired','command','digest','{}','completed','{}',
                   '2000-01-01T00:00:00+00:00','2000-01-01T00:00:00+00:00')"""
        )
        database.execute(
            """INSERT INTO scheduler_requests VALUES (
                   'recent','command','digest','{}','completed','{}',?,?)""",
            (now, now),
        )

    report = compact_registry(registry, minimum_interval_seconds=0)

    assert report.coalesced_requests == 1
    assert report.compacted_plans == 1
    assert report.removed_requests == 1
    assert report.removed_attempts == 1
    assert report.removed_submissions == 1
    assert report.removed_workers == 2
    assert report.removed_work_items == 1
    assert report.expired_scheduler_requests == 1
    with registry.connection() as database:
        assert [row[0] for row in database.execute("SELECT id FROM requests")] == [second]
        assert decode_plan(
            database.execute(
                "SELECT payload_json FROM request_plans WHERE request_id=?", (second,)
            ).fetchone()[0]
        ) == {"terminals": [child.key]}
        assert {row[0] for row in database.execute("SELECT work_item_key FROM work_items")} == {
            parent.key,
            child.key,
            failed.key,
        }
        assert [row[0] for row in database.execute("SELECT state FROM attempts ORDER BY id")] == [
            "running",
            "error",
        ]
        assert {row[0] for row in database.execute("SELECT id FROM workers")} == {
            "current-worker",
            "failed-worker",
        }
        assert [row[0] for row in database.execute("SELECT id FROM scheduler_requests")] == [
            "recent"
        ]


def test_compaction_interval_skips_repeated_pass(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()

    first = compact_registry(registry, minimum_interval_seconds=60, now=100.0)
    second = compact_registry(registry, minimum_interval_seconds=60, now=120.0)

    assert not first.skipped
    assert second.skipped


def test_compaction_removes_large_attempt_history_with_set_based_sql(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    item = _spec(
        registry,
        registered,
        workflow,
        key="anat:" + "e" * 64,
        module="anat",
    )
    work_item_id = registry.register_work_items((item,))[item.key]
    now = utcnow()
    with registry.connection(write=True) as database:
        database.executemany(
            """INSERT INTO attempts(
                   work_item_id,state,revision_fingerprint,memory_gb,
                   completed_at,log_path,created_at
               ) VALUES (?,'success',?,32,?,'/old.log',?)""",
            ((work_item_id, item.revision_fingerprint, now, now) for _ in range(2_000)),
        )
        database.execute(
            """INSERT INTO attempts(
                   work_item_id,state,revision_fingerprint,memory_gb,
                   completed_at,error_type,error_message,log_path,created_at
               ) VALUES (?,'error',?,32,?,'ScientificFailure','latest',
                         '/latest.log',?)""",
            (work_item_id, item.revision_fingerprint, now, now),
        )

    report = compact_registry(registry, minimum_interval_seconds=0)

    assert report.removed_attempts == 2_000
    with registry.connection() as database:
        assert [
            tuple(row) for row in database.execute("SELECT state,error_message FROM attempts")
        ] == [("error", "latest")]


def test_existing_derivative_retains_missing_dependency_and_publication_handle(
    tmp_path: Path,
) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    parent = _spec(
        registry,
        registered,
        workflow,
        key="anat:" + "e" * 64,
        module="anat",
    )
    child = _spec(
        registry,
        registered,
        workflow,
        key="func:" + "f" * 64,
        module="func",
        dependencies=(parent.key,),
    )
    request = registry.create_request(
        registered=registered,
        target_module="func",
        selectors={},
        work_items=(parent, child),
        terminal_work_item_keys=(child.key,),
        concurrency=1,
        partition=None,
    )
    ids = registry.work_item_ids((parent.key, child.key))
    with registry.connection(write=True) as database:
        database.execute(
            "UPDATE work_items SET artifact_state='fresh' WHERE id=?", (ids[child.key],)
        )
        database.execute("UPDATE requests SET state='satisfied' WHERE id=?", (request,))
        database.execute("INSERT INTO request_owners VALUES (?, 'main', 'owner')", (request,))
        database.execute(
            "INSERT INTO request_plans VALUES (?, ?)",
            (request, encode_plan({"terminals": [child.key]})),
        )
        database.executemany(
            "INSERT INTO request_artifacts VALUES (?, ?)",
            ((request, ids[parent.key]), (request, ids[child.key])),
        )

    report = compact_registry(registry, minimum_interval_seconds=0)

    assert report.removed_work_items == 0
    with registry.connection() as database:
        assert {row[0] for row in database.execute("SELECT work_item_key FROM work_items")} == {
            parent.key,
            child.key,
        }
        assert (
            database.execute(
                "SELECT COUNT(*) FROM request_work_items WHERE request_id=?", (request,)
            ).fetchone()[0]
            == 0
        )
        assert [
            row[0]
            for row in database.execute(
                "SELECT work_item_id FROM request_artifacts WHERE request_id=?", (request,)
            )
        ] == [ids[child.key]]
    published_request, terminals = registry.publication_work_items(request)
    assert published_request["state"] == "satisfied"
    assert [item["id"] for item in terminals] == [ids[child.key]]


def test_large_request_plan_round_trip_is_compressed() -> None:
    payload = {
        "terminals": ["func:" + "f" * 64],
        "specifications": [
            {"key": f"func:{index:064x}", "value": "x" * 1000} for index in range(100)
        ],
        "inherited": [{"id": index, "branch": "main"} for index in range(20)],
    }

    encoded = encode_plan(payload)

    assert isinstance(encoded, bytes)
    assert len(encoded) < len(json.dumps(payload)) // 4
    assert decode_plan(encoded) == payload


def test_compaction_retains_started_attempts_of_the_current_revision(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    item = _spec(
        registry,
        registered,
        workflow,
        key="anat:" + "f" * 64,
        module="anat",
    )
    work_item_id = registry.register_work_items((item,))[item.key]
    with registry.connection(write=True) as database:
        for revision, started, finished, state, message in (
            (
                "old-revision",
                "2026-10-01T00:00:00+00:00",
                "2026-10-01T01:00:00+00:00",
                "success",
                "old",
            ),
            (
                item.revision_fingerprint,
                "2026-10-02T00:00:00+00:00",
                "2026-10-03T00:00:00+00:00",
                "cancelled",
                "expired",
            ),
            (item.revision_fingerprint, "2026-10-03T00:10:00+00:00", None, "running", None),
        ):
            database.execute(
                """INSERT INTO attempts(
                       work_item_id,state,revision_fingerprint,memory_gb,started_at,
                       completed_at,error_type,error_message,log_path,created_at
                   ) VALUES (?,?,?,32,?,?,?,?,'/attempt.log',?)""",
                (
                    work_item_id,
                    state,
                    revision,
                    started,
                    finished,
                    "WorkerAllocationExpired" if state == "cancelled" else None,
                    message,
                    started,
                ),
            )

    report = compact_registry(registry, minimum_interval_seconds=0)

    assert report.removed_attempts == 1
    with registry.connection() as database:
        assert [row[0] for row in database.execute("SELECT state FROM attempts ORDER BY id")] == [
            "cancelled",
            "running",
        ]


def test_attempt_history_pruning_uses_work_item_indexes(tmp_path: Path) -> None:
    """Each candidate attempt must not scan the artifact or resource-step tables."""
    import inspect
    import re

    from nro.orchestration import registry_compaction

    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()
    source = inspect.getsource(registry_compaction._remove_attempt_history)
    statement = re.search(r'"""(DELETE FROM attempts AS candidate.*?)"""', source, re.S).group(1)
    with registry.connection() as database:
        plan = [row[-1] for row in database.execute("EXPLAIN QUERY PLAN " + statement)]
    scans = [
        step for step in plan if re.match(r"SCAN (TABLE )?(artifacts|resource_step_tasks)\b", step)
    ]
    assert scans == [], "\n".join(plan)


def test_plan_compaction_skips_plans_it_already_verified(monkeypatch) -> None:
    import sqlite3

    from nro.orchestration import request_plans

    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.execute("CREATE TABLE requests(id TEXT PRIMARY KEY, state TEXT)")
    database.execute("CREATE TABLE request_plans(request_id TEXT PRIMARY KEY, payload_json)")
    payload = {"inherited": [{"id": 1}], "terminals": ["a"], "specifications": ["x" * 5000]}
    database.execute("INSERT INTO requests VALUES ('r', 'active')")
    database.execute("INSERT INTO request_plans VALUES ('r', ?)", (json.dumps(payload, indent=2),))

    assert request_plans.compact_terminal_plans(database) == 1
    assert (
        decode_plan(database.execute("SELECT payload_json FROM request_plans").fetchone()[0])
        == payload
    )

    def unexpected_decode(_value):
        raise AssertionError("a verified plan must not be decoded again")

    monkeypatch.setattr(request_plans, "decode_plan", unexpected_decode)
    assert request_plans.compact_terminal_plans(database) == 0

    database.execute("UPDATE requests SET state='satisfied'")
    monkeypatch.undo()
    assert request_plans.compact_terminal_plans(database) == 1
    assert decode_plan(
        database.execute("SELECT payload_json FROM request_plans").fetchone()[0]
    ) == {"terminals": ["a"]}


def test_plan_reconciliation_summary_reads_replay_fields() -> None:
    from nro.orchestration.request_plans import plan_reconciliation_summary

    stored = encode_plan(
        {
            "branch": "main",
            "registry_id": "registry",
            "inherited": [{"id": 3}],
            "specifications": ["x" * 5000],
        }
    )
    summary = plan_reconciliation_summary(stored)
    assert summary == {
        "has_specifications": True,
        "branch": "main",
        "registry_id": "registry",
        "inherited": ({"id": 3},),
    }
    assert plan_reconciliation_summary(stored) is summary
    assert (
        plan_reconciliation_summary(encode_plan({"terminals": []}))["has_specifications"] is False
    )
