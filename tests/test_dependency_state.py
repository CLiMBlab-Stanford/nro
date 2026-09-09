"""Cancellation barriers follow resolved inputs and survive graph edits."""

import sys
from pathlib import Path

import pytest

from nro.configuration.store import ConfigStore
from nro.orchestration import dependency_state, manifests
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.registry import Registry


@pytest.fixture
def graph(tmp_path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    specs = []
    for name, parents in [
        ("main-root", ()),
        ("dev-middle", ("main-root",)),
        ("feature-leaf", ("dev-middle",)),
        ("unrelated", ()),
    ]:
        output = tmp_path / name / "result.txt"
        output.parent.mkdir()
        output.write_text("original result")
        specs.append(
            InstanceSpec.create(
                key=name,
                module="anat",
                project="demo",
                participant="01",
                entities={},
                scope="subject",
                configuration_lineage_id=registered.lineages["preprocessing"],
                directory_label="main",
                config_fingerprint=workflow.configuration("preprocessing").fingerprint,
                runtime_config=registry.runtime_config_path(registered, "preprocessing"),
                command=(sys.executable, "-c", "pass"),
                input_paths=(),
                expected_outputs=(output,),
                output_root=output.parent,
                output_prefix=None,
                dependencies=parents,
                resource_class="large",
            )
        )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        instances=specs,
        terminal_instance_keys=("feature-leaf", "unrelated"),
        concurrency=5,
        partition=None,
    )
    rows = {row["instance_key"]: row for row in registry.instance_rows()}
    with registry.connection(write=True) as db:
        db.execute("UPDATE instances SET artifact_state='fresh', current_generation=1")
        db.execute(
            "UPDATE instances SET artifact_state='stale' WHERE instance_key IN ('feature-leaf','unrelated')"
        )
    for worker in ("reader", "other", "writer"):
        registry.register_worker(worker, resource_class="large")
    leaf = registry.claim_ready_instance("reader", ("large",))
    assert leaf.instance_key == "feature-leaf"
    other = registry.claim_ready_instance("other", ("large",))
    assert other.instance_key == "unrelated"
    return registry, specs, rows, leaf, other


def dirty_root(registry, rows):
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='stale' WHERE id=?", (rows["main-root"]["id"],)
        )


def test_rebuild_waits_for_transitive_reader_shutdown_without_cancelling_unrelated_work(graph):
    registry, specs, rows, leaf, other = graph
    dirty_root(registry, rows)
    assert registry.claim_ready_instance("writer", ("large",)) is None
    assert registry.attempt_cancel_requested(leaf.attempt_id)
    assert not registry.attempt_cancel_requested(other.attempt_id)
    assert (
        next(row for row in registry.instance_rows() if row["instance_key"] == "dev-middle")[
            "artifact_state"
        ]
        == "stale"
    )
    assert registry.reserve_worker_submissions(request_id=None, resource_class="large") == []
    registry.finish_attempt(leaf.attempt_id, state="cancelled")
    writer = registry.claim_ready_instance("writer", ("large",))
    assert writer.instance_key == "main-root"


def test_removed_graph_edge_does_not_release_captured_input(graph):
    registry, specs, rows, leaf, _ = graph
    updated = [
        spec.evolve(dependencies=()) if spec.key == "feature-leaf" else spec for spec in specs
    ]
    registry.register_instances(updated)
    with registry.connection() as db:
        assert not db.execute(
            "SELECT 1 FROM instance_dependencies WHERE instance_id=?", (leaf.instance_id,)
        ).fetchone()
        assert db.execute(
            "SELECT 1 FROM attempt_dependencies WHERE attempt_id=? AND upstream_instance_id=?",
            (leaf.attempt_id, rows["main-root"]["id"]),
        ).fetchone()
    dirty_root(registry, rows)
    assert registry.claim_ready_instance("writer", ("large",)) is None
    registry.finish_attempt(leaf.attempt_id, state="cancelled")
    assert registry.claim_ready_instance("writer", ("large",)).instance_key in {
        "main-root",
        "feature-leaf",
    }


def test_changed_generation_rejects_late_completion(graph):
    registry, specs, rows, leaf, _ = graph
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET current_generation=2 WHERE id=?", (rows["main-root"]["id"],)
        )
    with pytest.raises(dependency_state.AttemptInvalidated):
        manifests.record_completion(
            registry,
            instance_id=leaf.instance_id,
            attempt_id=leaf.attempt_id,
            outputs=specs[2].expected_outputs,
        )
    assert not Path(leaf.manifest_path).exists()


def test_cancellation_during_inventory_prevents_completion_publication(graph, monkeypatch):
    registry, specs, rows, leaf, _ = graph
    original = manifests.inventory

    def changed(paths):
        result = original(paths)
        dirty_root(registry, rows)
        registry.cancel_attempts_with_stale_upstreams()
        return result

    monkeypatch.setattr(manifests, "inventory", changed)
    with pytest.raises(dependency_state.AttemptInvalidated):
        manifests.record_completion(
            registry,
            instance_id=leaf.instance_id,
            attempt_id=leaf.attempt_id,
            outputs=specs[2].expected_outputs,
        )
    assert not Path(leaf.manifest_path).exists()
    registry.finish_attempt(leaf.attempt_id, state="success")
    with registry.connection() as db:
        assert (
            db.execute("SELECT state FROM attempts WHERE id=?", (leaf.attempt_id,)).fetchone()[0]
            == "cancelled"
        )


def test_mutation_timeout_preserves_files_and_cancels_readers(graph):
    registry, specs, rows, leaf, other = graph
    with pytest.raises(RuntimeError, match="confirmed shutdown"):
        with registry.artifact_mutation([rows["main-root"]["id"]], timeout=0):
            pytest.fail("Output writes started before the reader stopped")
    assert Path(specs[0].expected_outputs[0]).read_text() == "original result"
    assert registry.attempt_cancel_requested(leaf.attempt_id)
    assert not registry.attempt_cancel_requested(other.attempt_id)
    with registry.connection() as db:
        assert not db.execute("SELECT 1 FROM artifact_mutations").fetchone()


def test_mutation_reservation_blocks_new_writers_until_exit(graph):
    registry, specs, rows, leaf, _ = graph
    registry.finish_attempt(leaf.attempt_id, state="cancelled", error_type="UpstreamStale")
    with registry.artifact_mutation([rows["main-root"]["id"]]):
        with registry.connection(write=True) as db:
            db.execute(
                "UPDATE instances SET artifact_state='fresh' WHERE instance_key IN ('main-root','dev-middle')"
            )
        assert registry.claim_ready_instance("writer", ("large",)) is None
    assert registry.claim_ready_instance("writer", ("large",)).instance_key == "main-root"


def test_orphaned_worker_does_not_release_a_live_process_group(graph, monkeypatch):
    from nro.orchestration import execution

    registry, specs, rows, leaf, _ = graph
    dirty_root(registry, rows)
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET pid=999999999, lease_expires_at=0 WHERE id='reader'")
    registry.record_attempt_process(leaf.attempt_id, 12345)
    monkeypatch.setattr(execution, "process_group_alive", lambda _: True)
    assert registry.recover_orphaned_attempts() == 0
    assert registry.claim_ready_instance("writer", ("large",)) is None
    monkeypatch.setattr(execution, "process_group_alive", lambda _: False)
    assert registry.recover_orphaned_attempts() == 1
    assert registry.claim_ready_instance("writer", ("large",)).instance_key == "main-root"


def test_cancelled_consumer_requeues_under_its_original_demand(graph):
    registry, specs, rows, leaf, _ = graph
    requests = registry.request_rows()
    dirty_root(registry, rows)
    registry.cancel_attempts_with_stale_upstreams()
    registry.finish_attempt(leaf.attempt_id, state="cancelled")
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh', current_generation=2 WHERE instance_key IN ('main-root','dev-middle')"
        )
    retried = registry.claim_ready_instance("writer", ("large",))
    assert retried.instance_key == leaf.instance_key
    assert retried.execution == leaf.execution
    assert [row["id"] for row in registry.request_rows()] == [row["id"] for row in requests]


def test_completed_consumers_remain_invalid_after_generation_replacement(graph):
    registry, specs, rows, leaf, _ = graph
    manifests.record_completion(
        registry,
        instance_id=leaf.instance_id,
        attempt_id=leaf.attempt_id,
        outputs=specs[2].expected_outputs,
    )
    registry.finish_attempt(leaf.attempt_id, state="success")
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET current_generation=2 WHERE id=?", (rows["dev-middle"]["id"],)
        )
    registry.cancel_attempts_with_stale_upstreams()
    state = next(row for row in registry.instance_rows() if row["id"] == leaf.instance_id)
    assert state["artifact_state"] == "stale"
    retried = registry.claim_ready_instance("writer", ("large",))
    assert retried.instance_id == leaf.instance_id
    registry.cancel_attempts_with_stale_upstreams()
    assert not registry.attempt_cancel_requested(retried.attempt_id)
    manifests.record_completion(
        registry,
        instance_id=retried.instance_id,
        attempt_id=retried.attempt_id,
        outputs=specs[2].expected_outputs,
    )


def test_outdated_completed_consumer_does_not_invalidate_current_sibling(graph):
    registry, specs, rows, leaf, other = graph
    registry.finish_attempt(leaf.attempt_id, state="cancelled")
    with registry.connection(write=True) as db:
        db.execute("UPDATE instances SET artifact_state='fresh' WHERE id=?", (leaf.instance_id,))
        db.execute(
            "UPDATE instance_dependencies SET required_generation=0 WHERE instance_id=?",
            (leaf.instance_id,),
        )
        db.execute(
            "INSERT INTO instance_dependencies(instance_id, upstream_instance_id, role, required_generation) VALUES (?, ?, ?, ?)",
            (other.instance_id, rows["dev-middle"]["id"], "input", 1),
        )
        dependency_state.capture_inputs(db, other.attempt_id, other.instance_id)
    registry.cancel_attempts_with_stale_upstreams()
    assert not registry.attempt_cancel_requested(other.attempt_id)
