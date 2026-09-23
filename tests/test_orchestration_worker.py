from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nro.bin.status import main as status_main
from nro.configuration.store import ConfigStore
from nro.orchestration import completion, dependency_state
from nro.orchestration.artifact_records import file_record
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.execution import ExecutionResult
from nro.orchestration.manifests import assess_registry
from nro.orchestration.publish import publish
from nro.orchestration.registry import (
    Registry,
    RegistryLock,
    RegistryLockTimeout,
    discover_registry_projects,
)
from nro.orchestration.registry_work_items import work_item_relative_directory
from nro.orchestration.scheduler_maintenance import refresh_scheduler_state
from nro.orchestration.worker import Worker, _looks_like_oom


def test_cuda_memory_failure_does_not_request_more_host_memory(tmp_path: Path) -> None:
    log = tmp_path / "attempt.log"
    log.write_text(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.69 GiB.\n",
        encoding="utf-8",
    )

    assert not _looks_like_oom(log, 1)


def test_host_memory_failure_requests_more_host_memory(tmp_path: Path) -> None:
    log = tmp_path / "attempt.log"
    log.write_text("worker terminated: out of memory\n", encoding="utf-8")

    assert _looks_like_oom(log, 1)


def test_dag_construction_failure_does_not_request_more_memory(tmp_path: Path) -> None:
    log = tmp_path / "attempt.log"
    log.write_text(
        "ValueError: Duplicate step id in module DAG: stage:abc123\n",
        encoding="utf-8",
    )

    assert not _looks_like_oom(log, 1)


def test_oversized_worker_expands_pool_from_lowest_memory_tier(tmp_path: Path, monkeypatch) -> None:
    class CapacityClient:
        def __init__(self) -> None:
            self.paths = type("Paths", (), {"control": tmp_path})()
            self.calls: list[tuple[str, dict[str, object]]] = []

        def request_capacity(self, kind: str, **values: object) -> None:
            self.calls.append((kind, values))

    client = CapacityClient()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    worker = Worker(
        client,  # type: ignore[arg-type]
        resource_class="large",
        memory_gb=256,
        profile="profile",
    )

    worker._expand_ready_pool()

    assert client.calls == [
        (
            "expand",
            {"resource_class": "large", "memory_gb": 1, "profile": "profile"},
        )
    ]


def _spec(
    *,
    key: str,
    module: str,
    lineage: int,
    config_fingerprint: str,
    runtime_config: Path,
    output: Path,
    dependencies: tuple[str, ...] = (),
    inputs: tuple[Path, ...] = (),
    project: str = "demo",
    resource_class: str = "large",
) -> WorkItemSpec:
    directory_label = runtime_config.stem.rsplit("_", 1)[0]
    command = (
        sys.executable,
        "-c",
        f"from pathlib import Path; p=Path({str(output)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok')",
    )
    return WorkItemSpec.create(
        key=key,
        module=module,
        project=project,
        participant="01",
        entities={},
        scope="subject",
        module_lineage_id=lineage,
        config_fingerprint=config_fingerprint,
        directory_label=directory_label,
        runtime_config=runtime_config,
        command=command,
        dependencies=dependencies,
        input_paths=inputs,
        output_root=output.parent,
        output_prefix=None,
        resource_class=resource_class,
        expected_outputs=(output,),
        processing=module_descriptor(module).processing_contract(),
    )


def test_worker_resource_class_does_not_change_scientific_freshness(tmp_path: Path) -> None:
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    general = _spec(
        key="anat:" + "0" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "anat.txt",
    )

    gpu = general.evolve(resource_class="gpu")

    assert gpu.contract_fingerprint == general.contract_fingerprint
    assert gpu.work_item_contract == general.work_item_contract


def test_resource_step_handoff_releases_parent_and_resumes_after_gpu(
    tmp_path: Path,
) -> None:
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    spec = _spec(
        key="anat:" + "a" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "anat.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(spec,),
        terminal_work_item_keys=(spec.key,),
        concurrency=2,
        partition=None,
    )
    registry.register_worker("cpu", resource_class="large")
    registry.register_worker("gpu", resource_class="gpu")

    parent = registry.claim_ready_work_item("cpu", ("large",))
    assert parent is not None
    task_id = registry.defer_resource_step(
        parent.attempt_id,
        step_id="neurolit-inpainting",
        resource_class="gpu",
        memory_gb=32,
    )
    assert registry.claim_ready_work_item("cpu", ("large",)) is None
    pending = registry.work_item_status_snapshot()[0]
    assert pending["status"] == "Queued"
    assert pending["waiting_resource_class"] == "gpu"
    assert pending["unfinished_dependency_ids"] == ()

    task = registry.claim_resource_step("gpu", resource_class="gpu", memory_gb=32)
    assert task is not None
    assert task.resource_task_id == task_id
    assert task.target_step_id == "neurolit-inpainting"
    assert registry.work_item_status_snapshot()[0]["status"] == "Running"
    registry.finish_resource_step(task_id, task.attempt_id, state="success")
    assert registry.work_item_status_snapshot()[0]["status"] == "Queued"

    resumed = registry.claim_ready_work_item("cpu", ("large",))
    assert resumed is not None
    assert resumed.work_item_id == parent.work_item_id
    assert resumed.completed_resource_steps == ("neurolit-inpainting",)


def test_user_cancelled_resource_step_reports_stopped(tmp_path: Path) -> None:
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    spec = _spec(
        key="anat:" + "b" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "anat.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(spec,),
        terminal_work_item_keys=(spec.key,),
        concurrency=2,
        partition=None,
    )
    registry.register_worker("cpu", resource_class="large")
    parent = registry.claim_ready_work_item("cpu", ("large",))
    assert parent is not None
    registry.defer_resource_step(
        parent.attempt_id,
        step_id="neurolit-inpainting",
        resource_class="gpu",
        memory_gb=32,
    )

    registry.request_cancellation(modules=("anat",))
    stopped = registry.work_item_status_snapshot()[0]
    assert stopped["status"] == "Stopped"
    assert stopped["error_type"] == "UserCancelled"
    assert stopped["error_message"] == "Demand was cancelled before the resource step ran"


def test_resource_step_completion_rejects_changed_upstream_generation(tmp_path: Path) -> None:
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")
    upstream = _spec(
        key="anat:" + "b" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "upstream.txt",
    )
    child = _spec(
        key="anat:" + "c" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "child.txt",
        dependencies=(upstream.key,),
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(upstream, child),
        terminal_work_item_keys=(child.key,),
        concurrency=2,
        partition=None,
    )
    with registry.connection(write=True) as database:
        database.execute(
            """UPDATE work_items SET artifact_state='fresh',artifact_reason='test',
                      current_generation=1 WHERE work_item_key=?""",
            (upstream.key,),
        )
    registry.register_worker("cpu", resource_class="large")
    registry.register_worker("gpu", resource_class="gpu")
    parent = registry.claim_ready_work_item("cpu", ("large",))
    assert parent is not None and parent.work_item_key == child.key
    task_id = registry.defer_resource_step(
        parent.attempt_id,
        step_id="gpu-step",
        resource_class="gpu",
    )
    task = registry.claim_resource_step("gpu", resource_class="gpu", memory_gb=32)
    assert task is not None
    with registry.connection(write=True) as database:
        database.execute(
            "UPDATE work_items SET current_generation=2 WHERE work_item_key=?",
            (upstream.key,),
        )

    with pytest.raises(dependency_state.AttemptInvalidated, match="upstream artifact changed"):
        registry.finish_resource_step(task_id, task.attempt_id, state="success")


def test_gpu_worker_does_not_claim_whole_work_items(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")
    gpu = _spec(
        key="anat:" + "2" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "gpu.txt",
        resource_class="gpu",
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(gpu,),
        terminal_work_item_keys=(gpu.key,),
        concurrency=1,
        partition=None,
    )

    assert (
        Worker(
            registry,
            resource_class="gpu",
            idle_timeout=0,
            poll_interval=0.01,
        ).run()
        == 0
    )
    with registry.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_gpu_worker_exits_without_claiming_general_work(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    general = _spec(
        key="anat:" + "3" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "general.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(general,),
        terminal_work_item_keys=(general.key,),
        concurrency=1,
        partition=None,
    )

    assert (
        Worker(
            registry,
            resource_class="gpu",
            idle_timeout=0,
            poll_interval=0.01,
        ).run()
        == 0
    )
    with registry.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_general_and_gpu_capacity_are_reserved_independently(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")
    general = _spec(
        key="anat:" + "4" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "general.txt",
    )
    resource_parent = _spec(
        key="anat:" + "5" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "gpu.txt",
    )
    request = registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(general, resource_parent),
        terminal_work_item_keys=(general.key, resource_parent.key),
        concurrency=2,
        partition=None,
    )
    registry.register_worker("cpu", resource_class="large")
    parent = registry.claim_ready_work_item("cpu", ("large",))
    assert parent is not None
    registry.defer_resource_step(
        parent.attempt_id,
        step_id="neurolit-inpainting",
        resource_class="gpu",
        memory_gb=32,
    )
    registry.close_worker("cpu")

    gpu_reservations = registry.reserve_worker_submissions(
        request_id=request, resource_class="gpu", memory_gb=32
    )
    general_reservations = registry.reserve_worker_submissions(
        request_id=request, resource_class="large", memory_gb=32
    )

    assert len(gpu_reservations) == 1
    assert len(general_reservations) == 1


def test_gpu_concurrency_is_independent_of_request_concurrency(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")
    specs = tuple(
        _spec(
            key="anat:" + digit * 64,
            module="anat",
            lineage=registered.lineages["anat"],
            config_fingerprint=workflow.configuration("anat").fingerprint,
            runtime_config=runtime,
            output=tmp_path / f"parent-{digit}.txt",
        )
        for digit in ("6", "7")
    )
    request = registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=specs,
        terminal_work_item_keys=tuple(spec.key for spec in specs),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("cpu", resource_class="large")
    for index in range(2):
        parent = registry.claim_ready_work_item("cpu", ("large",))
        assert parent is not None
        registry.defer_resource_step(
            parent.attempt_id,
            step_id=f"gpu-step-{index}",
            resource_class="gpu",
            memory_gb=32,
        )
    registry.close_worker("cpu")
    registry.set_gpu_concurrency(2)

    assert (
        len(
            registry.reserve_worker_submissions(
                request_id=request,
                resource_class="gpu",
                memory_gb=32,
            )
        )
        == 2
    )


def test_higher_memory_gpu_work_is_visible_to_initial_supply(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    parent_spec = _spec(
        key="anat:" + "6" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "gpu.txt",
    )
    request = registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(parent_spec,),
        terminal_work_item_keys=(parent_spec.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("cpu", resource_class="large")
    parent = registry.claim_ready_work_item("cpu", ("large",))
    assert parent is not None
    registry.defer_resource_step(
        parent.attempt_id,
        step_id="neurolit-inpainting",
        resource_class="gpu",
        memory_gb=64,
    )

    assert not registry.worker_capacity_needed(
        request_id=request, resource_class="gpu", memory_gb=32
    )
    assert registry.worker_capacity_needed(request_id=request, resource_class="gpu", memory_gb=64)


def test_pending_lower_memory_gpu_does_not_create_higher_tier_demand(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    parent_spec = _spec(
        key="anat:" + "9" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "gpu.txt",
    )
    request = registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(parent_spec,),
        terminal_work_item_keys=(parent_spec.key,),
        concurrency=8,
        partition=None,
    )
    registry.register_worker("cpu", resource_class="large")
    parent = registry.claim_ready_work_item("cpu", ("large",))
    assert parent is not None
    registry.defer_resource_step(
        parent.attempt_id,
        step_id="neurolit-inpainting",
        resource_class="gpu",
        memory_gb=32,
    )

    assert (
        len(
            registry.reserve_worker_submissions(
                request_id=request,
                resource_class="gpu",
                memory_gb=32,
                minimum_memory_gb=0,
            )
        )
        == 1
    )
    for lower, upper in ((32, 64), (64, 128), (128, 256)):
        assert (
            registry.reserve_worker_submissions(
                request_id=request,
                resource_class="gpu",
                memory_gb=upper,
                minimum_memory_gb=lower,
            )
            == []
        )


def test_work_item_private_paths_use_the_logical_digest() -> None:
    base = {
        "project": "demo",
        "participant": "01",
        "entities_json": "{}",
        "module": "microparcellation",
    }
    first = work_item_relative_directory(
        {**base, "work_item_key": "owner:microparcellation:" + "a" * 64}
    )
    second = work_item_relative_directory(
        {**base, "work_item_key": "owner:microparcellation:" + "b" * 64}
    )

    assert first != second
    assert first.name == "a" * 16
    assert second.name == "b" * 16


def test_registry_rejects_work_item_bound_to_the_wrong_lineage_directory(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="anat:" + "1" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "outputs" / "anat.txt",
    ).evolve(directory_label="another-lineage")

    with pytest.raises(ValueError, match="does not match its registered module lineage"):
        registry.register_work_items((work_item,))


def test_registry_rejects_output_collision_across_registration_calls(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")
    first = _spec(
        key="anat:" + "2" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "outputs" / "first.txt",
    )
    second = _spec(
        key="anat:" + "3" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=first.expected_outputs[0],
    )
    registry.register_work_items((first,))

    with pytest.raises(ValueError, match="output claim conflicts"):
        registry.register_work_items((second,))


def test_existing_work_item_adopts_current_configuration_snapshot(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    first = _spec(
        key="networks:" + "9" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint="old-configuration",
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "network.txt",
    )
    registry.register_work_items((first,))
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh', artifact_reason='Current' "
            "WHERE work_item_key=?",
            (first.key,),
        )
    current_runtime = tmp_path / "current_networks.yml"
    current_runtime.write_text("labeling:\n  enabled: true\n", encoding="utf-8")
    current = first.evolve(
        config_fingerprint="current-configuration",
        runtime_config=current_runtime,
    )

    registry.register_work_items((current,))

    row = next(item for item in registry.work_item_rows() if item["work_item_key"] == first.key)
    assert row["runtime_config_path"] == str(current_runtime)
    assert row["revision_fingerprint"] == current.revision_fingerprint
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Work-item contract changed"


def test_central_registry_shares_concurrency_without_cross_project_cancellation(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    alpha_registry = Registry.for_project("alpha", bids_root=bids)
    beta_registry = Registry.for_project("beta", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = alpha_registry.register_workflow(workflow)
    runtime_config = alpha_registry.runtime_config_path(registered, "networks")
    alpha = _spec(
        key="networks:" + "a" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=runtime_config,
        output=tmp_path / "alpha-output" / "network.txt",
        project="alpha",
    )
    beta = _spec(
        key="networks:" + "b" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=runtime_config,
        output=tmp_path / "beta-output" / "network.txt",
        project="beta",
    )
    alpha_request = alpha_registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(alpha,),
        terminal_work_item_keys=(alpha.key,),
        concurrency=1,
        partition=None,
    )
    beta_request = beta_registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(beta,),
        terminal_work_item_keys=(beta.key,),
        concurrency=1,
        partition=None,
    )

    assert alpha_registry.paths.database == beta_registry.paths.database
    assert alpha_registry.paths.control == tmp_path / ".nro"
    assert discover_registry_projects(bids) == ["alpha", "beta"]
    rows = {row["project"]: row for row in alpha_registry.work_item_rows()}
    assert set(rows) == {"alpha", "beta"}
    with pytest.raises(ValueError, match="only its selected project"):
        beta_registry.register_work_items((alpha,))
    assert (
        len(
            alpha_registry.reserve_worker_submissions(
                request_id=alpha_request,
                resource_class="large",
            )
        )
        == 1
    )
    assert (
        beta_registry.reserve_worker_submissions(
            request_id=beta_request,
            resource_class="large",
        )
        == []
    )

    alpha_registry.request_cancellation(modules=("networks",), force=True)
    request_states = {row["project"]: row["state"] for row in alpha_registry.request_rows()}
    assert request_states == {"alpha": "cancelled", "beta": "active"}


def test_worker_runs_dependency_graph_and_manifests_detect_staleness(
    tmp_path: Path, capsys
) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    raw = tmp_path / "raw.nii.gz"
    raw.write_text("raw")
    anat_output = tmp_path / "outputs" / "anat.txt"
    network_output = (
        bids
        / "demo"
        / "derivatives"
        / "nro"
        / "networks"
        / registered.directories["networks"]
        / "sub-01"
        / "network.txt"
    )
    anat = _spec(
        key="anat:" + "a" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=anat_output,
        inputs=(raw,),
    )
    anat = anat.evolve(config_fingerprint=workflow.configuration("anat").scientific_fingerprint)
    network = _spec(
        key="networks:" + "b" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=network_output,
        dependencies=(anat.key,),
    )
    network = network.evolve(
        config_fingerprint=workflow.configuration("networks").scientific_fingerprint
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(anat, network),
        terminal_work_item_keys=(network.key,),
        concurrency=1,
        partition=None,
    )

    initial = {row["module"]: row for row in registry.work_item_status_snapshot()}
    assert initial["anat"]["status"] == "Queued"
    assert initial["anat"]["unfinished_dependency_ids"] == ()
    assert initial["networks"]["status"] == "Waiting"
    assert initial["networks"]["unfinished_dependency_ids"] == (initial["anat"]["id"],)

    assert Worker(registry, resource_class="large", idle_timeout=0.2, poll_interval=0.02).run() == 0
    worker_log = capsys.readouterr().out
    assert "started (Slurm job " in worker_log
    assert "; class=large; memory=32 GB)" in worker_log
    assert "claimed work item" in worker_log
    assert "work-item log:" in worker_log
    assert "attempt state=success" in worker_log
    assert "idle timeout reached" in worker_log
    assert "stopped (state=exited)" in worker_log
    rows = {row["module"]: row for row in registry.work_item_rows()}
    assert rows["anat"]["artifact_state"] == "fresh"
    assert rows["networks"]["artifact_state"] == "fresh"
    assert rows["anat"]["current_generation"] == 1
    assert registry.request_rows()[0]["state"] == "satisfied"
    from nro.orchestration.completion_records import completion_record

    with registry.connection() as db:
        manifest = completion_record(db, rows["anat"]["id"])
    assert manifest is not None
    assert manifest["configuration"]["id"] == "main"
    assert manifest["software"]["name"] == "nro"
    assert manifest["public_outputs"]
    assert manifest["artifact_contract"] == json.loads(rows["anat"]["artifact_contract_json"])
    assert manifest["artifact_fingerprint"] == rows["anat"]["artifact_fingerprint"]
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_fingerprint='changed-contract' WHERE id=?",
            (rows["anat"]["id"],),
        )
    state = assess_registry(registry)[rows["anat"]["id"]]
    assert state == ("stale", "Work-item contract changed")
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_fingerprint=? WHERE id=?",
            (manifest["artifact_fingerprint"], rows["anat"]["id"]),
        )
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    steps = json.loads((Path(rows["anat"]["log_path"]).parent / "current-steps.json").read_text())
    assert steps["orchestration:completion"]["status"] == "success"
    assert stat.S_IMODE(Path(rows["anat"]["log_path"]).parent.stat().st_mode) == 0o2775

    publication = publish(
        registry,
        request_id=registry.request_rows()[0]["id"],
        destination=tmp_path / "published",
        validate=False,
    )
    assert (publication / "sub-01" / "network.txt").read_text() == "ok"
    assert (publication / "dataset_description.json").is_file()
    provenance = json.loads((publication / ".nro-publication.json").read_text())
    assert "workflow_snapshot" not in provenance
    assert "request_id" not in provenance
    assert "configuration" in provenance["work_items"][0]
    assert provenance["work_items"][0]["upstream"][0]["module"] == "anat"

    # Existing private artifacts are part of the completion record, but WORK cleanup
    # is allowed once public derivatives are complete.
    private = tmp_path / "work" / "intermediate.txt"
    private.parent.mkdir(parents=True)
    private.write_text("original")
    private_record = file_record(private)
    with registry.connection(write=True) as db:
        db.execute(
            """INSERT INTO artifacts(
                   work_item_id,attempt_id,direction,path,size,mtime_ns,
                   digest_algorithm,digest,metadata_json
               ) VALUES (?,?,?,?,?,?,?,?, '{}')""",
            (
                rows["anat"]["id"],
                manifest["attempt_id"],
                "private",
                private_record["path"],
                private_record["size"],
                private_record["mtime_ns"],
                "sha256",
                private_record["sha256"],
            ),
        )
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    private.unlink()
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    private.write_text("changed")
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "stale"
    private.unlink()
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"

    raw.write_text("changed")
    states = assess_registry(registry)
    rows = {row["module"]: row for row in registry.work_item_rows()}
    assert states[rows["anat"]["id"]][0] == "stale"
    assert states[rows["networks"]["id"]][0] == "stale"


def test_registry_lock_timeout_cancels_and_retries_scientific_work(
    tmp_path: Path, monkeypatch
) -> None:
    import nro.orchestration.worker as worker_module

    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "outputs" / "network.txt"
    work_item = _spec(
        key="networks:" + "c" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    original = worker_module.record_completion
    calls = 0

    def intermittent_completion(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RegistryLockTimeout("busy")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_module, "record_completion", intermittent_completion)

    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()

    row = registry.work_item_rows()[0]
    assert row["artifact_state"] == "fresh"
    with registry.connection() as db:
        attempts = list(
            db.execute(
                "SELECT state,error_type FROM attempts WHERE work_item_id=? ORDER BY id",
                (row["id"],),
            )
        )
    assert [tuple(attempt) for attempt in attempts] == [
        ("cancelled", "RegistryUnavailable"),
        ("success", None),
    ]


def test_completion_record_and_generation_commit_atomically(tmp_path: Path) -> None:
    """A late database failure must not leave a partial successful generation."""
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="anat:" + "d" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").scientific_fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "outputs" / "anat.txt",
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
    with registry.connection(write=True) as db:
        db.execute(
            """CREATE TRIGGER reject_test_artifact BEFORE INSERT ON artifacts
               BEGIN SELECT RAISE(ABORT, 'injected artifact failure'); END"""
        )

    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()

    with registry.connection() as db:
        row = db.execute(
            "SELECT id,current_generation,artifact_state FROM work_items WHERE work_item_key=?",
            (work_item.key,),
        ).fetchone()
        assert row["current_generation"] == 0
        assert row["artifact_state"] != "fresh"
        assert (
            db.execute("SELECT 1 FROM completions WHERE work_item_id=?", (row["id"],)).fetchone()
            is None
        )
        assert (
            db.execute("SELECT 1 FROM artifacts WHERE work_item_id=?", (row["id"],)).fetchone()
            is None
        )


def test_worker_waits_for_scheduler_output_visibility(tmp_path: Path, monkeypatch) -> None:
    import nro.orchestration.worker as worker_module

    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "outputs" / "network.txt"
    work_item = _spec(
        key="networks:" + "d" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    visibility = iter((False, True))
    monkeypatch.setattr(
        registry, "outputs_visible", lambda _outputs: next(visibility), raising=False
    )
    monkeypatch.setattr(worker_module, "OUTPUT_VISIBILITY_POLL_INTERVAL", 0.0)

    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()

    row = registry.work_item_rows()[0]
    assert row["artifact_state"] == "fresh"
    assert "Waiting for published outputs" in Path(row["log_path"]).read_text()


def test_completion_inventory_retries_a_transiently_missing_output(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "published.json"

    def publish_after_first_probe(_seconds: float) -> None:
        output.write_text("{}")

    monkeypatch.setattr(completion.time, "sleep", publish_after_first_probe)

    records = completion._completion_output_inventory((output,))

    assert records[0]["path"] == str(output)


def test_completion_inventory_rejects_a_directory_without_retry(
    tmp_path: Path, monkeypatch
) -> None:
    directory = tmp_path / "published"
    directory.mkdir()
    slept = False

    def sleep(_seconds: float) -> None:
        nonlocal slept
        slept = True

    monkeypatch.setattr(completion.time, "sleep", sleep)

    with pytest.raises(ValueError, match="not a regular file"):
        completion._completion_output_inventory((directory,))
    assert not slept


def test_resumed_work_item_reuses_one_fixed_work_item_log(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "f" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "network.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("first", resource_class="large")
    first = registry.claim_ready_work_item("first", ("large",))
    assert first is not None
    registry.finish_attempt(first.attempt_id, state="cancelled", error_type="UpstreamStale")

    registry.register_worker("second", resource_class="large")
    second = registry.claim_ready_work_item("second", ("large",))
    assert second is not None
    assert first.log_path == second.log_path
    assert first.log_path.name == "work-item.log"


def test_user_cancelled_attempt_requires_new_run_request(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "1" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "network.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("first", resource_class="large")
    first = registry.claim_ready_work_item("first", ("large",))
    assert first is not None
    cancellation = registry.request_cancellation(modules=("networks",))
    assert cancellation["attempts"] == 1
    assert registry.work_item_status_snapshot()[0]["status"] == "Stopping"
    registry.finish_attempt(first.attempt_id, state="cancelled")
    stopped = registry.work_item_status_snapshot()[0]
    assert stopped["status"] == "Stopped"
    assert stopped["error_type"] == "UserCancelled"

    registry.register_worker("before-new-run", resource_class="large")
    assert registry.claim_ready_work_item("before-new-run", ("large",)) is None

    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    assert registry.work_item_status_snapshot()[0]["status"] == "Queued"
    registry.register_worker("after-new-run", resource_class="large")
    assert registry.claim_ready_work_item("after-new-run", ("large",)) is not None


def test_missing_private_manifest_uses_native_filesystem_evidence(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    raw = tmp_path / "raw_T1w.nii.gz"
    raw.write_text("raw")
    output_root = bids / "demo" / "derivatives" / "nro" / "anat" / "main" / "sub-01" / "anat"
    derivative = output_root / "sub-01_desc-preproc_T1w.nii.gz"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    derivative.write_text("derivative")
    native_manifest = output_root / "sub-01_desc-preprocessAnat_manifest.json"
    native_manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "public_outputs": [str(derivative)],
                "output_metadata_contract": module_descriptor("anat").processing_contract()[
                    "output_metadata"
                ],
            }
        )
    )
    work_item = _spec(
        key="anat:" + "0" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=derivative,
        inputs=(raw,),
    ).evolve(
        output_root=output_root,
        output_prefix="sub-01",
        expected_outputs=(native_manifest,),
    )
    work_item_ids = registry.register_work_items((work_item,))
    row = registry.work_item_rows()[0]
    states = assess_registry(registry, work_item_ids=work_item_ids.values())

    assert states[row["id"]][0] == "fresh", states[row["id"]]
    assert "private orchestration provenance is unavailable" in states[row["id"]][1]

    native_payload = json.loads(native_manifest.read_text())
    native_payload["output_metadata_contract"] = {"layout": "superseded"}
    native_manifest.write_text(json.dumps(native_payload))
    states = assess_registry(registry, work_item_ids=work_item_ids.values())
    assert states[row["id"]][0] == "stale"
    assert "current contract" in states[row["id"]][1]

    native_payload["output_metadata_contract"] = module_descriptor("anat").processing_contract()[
        "output_metadata"
    ]
    native_manifest.write_text(json.dumps(native_payload))
    future_ns = native_manifest.stat().st_mtime_ns + 10_000_000_000
    os.utime(raw, ns=(future_ns, future_ns))
    states = assess_registry(registry, work_item_ids=work_item_ids.values())
    assert states[row["id"]][0] == "stale"
    assert "direct input is newer" in states[row["id"]][1]


def test_failed_work_item_requires_new_demand_before_retry(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "nro" / "networks" / "main" / "sub-01" / "result.txt"
    base = _spec(
        key="networks:" + "c" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    implementation = tmp_path / "task_implementation.py"
    implementation.write_text("raise SystemExit(2)\n")
    failing = base.evolve(command=(sys.executable, str(implementation)))
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(failing,),
        terminal_work_item_keys=(failing.key,),
        concurrency=1,
        partition=None,
    )
    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()
    assert registry.request_rows()[0]["state"] == "active"
    assert registry.work_item_rows()[0]["attempt_state"] == "error"

    implementation.write_text(
        "from pathlib import Path\n"
        f"p = Path({str(output)!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text('ok')\n"
    )
    second_request = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(failing,),
        terminal_work_item_keys=(failing.key,),
        concurrency=1,
        partition=None,
    )
    assert registry.work_item_rows()[0]["retry_requested"] == 1
    assert registry.work_item_status_snapshot()[0]["status"] == "Queued"
    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()
    assert output.read_text() == "ok"
    requests = {row["id"]: row for row in registry.request_rows()}
    assert requests[second_request]["state"] == "satisfied"


def test_expired_dead_worker_lease_releases_work_item_for_successor(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "nro" / "networks" / "main" / "sub-01" / "result.txt"
    work_item = _spec(
        key="networks:" + "d" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("dead", resource_class="large")
    assert registry.claim_ready_work_item("dead", ("large",)) is not None
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET pid=999999999, lease_expires_at=0 WHERE id='dead'")

    assert registry.recover_orphaned_attempts() == 1
    registry.register_worker("successor", resource_class="large")
    claimed = registry.claim_ready_work_item("successor", ("large",))
    assert claimed is not None
    assert claimed.work_item_key == work_item.key


def test_fresh_artifact_does_not_propagate_historical_attempt_error(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    upstream = _spec(
        key="anat:" + "h" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "outputs" / "anat.txt",
    )
    downstream = _spec(
        key="networks:" + "i" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "networks.txt",
        dependencies=(upstream.key,),
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(upstream, downstream),
        terminal_work_item_keys=(downstream.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_work_item("failed-worker", ("large",))
    assert claimed is not None and claimed.work_item_key == upstream.key
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="historical failure",
    )

    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh' WHERE work_item_key=?",
            (upstream.key,),
        )

    snapshot = {row["work_item_key"]: row for row in registry.work_item_status_snapshot()}
    assert snapshot[upstream.key]["attempt_state"] == "error"
    assert snapshot[upstream.key]["status"] == "Success"
    assert snapshot[upstream.key]["root_failure_ids"] == ()
    assert snapshot[downstream.key]["status"] == "Queued"
    assert snapshot[downstream.key]["root_failure_ids"] == ()


def test_missing_undemanded_artifact_does_not_report_historical_error(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="anat:" + "j" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "outputs" / "anat.txt",
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
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_work_item("failed-worker", ("large",))
    assert claimed is not None
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="historical failure",
    )
    with registry.connection(write=True) as db:
        db.execute("UPDATE requests SET state='cancelled'")
        db.execute("UPDATE request_work_items SET demand_state='cancelled'")
        db.execute(
            "UPDATE work_items SET artifact_state='missing',artifact_reason='Purged by user'"
        )

    row = registry.work_item_status_snapshot()[0]
    assert row["attempt_state"] == "error"
    assert row["status"] == "Missing"
    assert row["artifact_reason"] == "Purged by user"

    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='corrupt',artifact_reason='Invalid output'"
        )
    row = registry.work_item_status_snapshot()[0]
    assert row["status"] == "Corrupt"
    assert row["artifact_reason"] == "Invalid output"


def test_failed_rebuild_after_purge_blocks_demanded_descendants(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    upstream = _spec(
        key="anat:" + "k" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "outputs" / "anat.txt",
    )
    downstream = _spec(
        key="networks:" + "l" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "networks.txt",
        dependencies=(upstream.key,),
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(upstream, downstream),
        terminal_work_item_keys=(downstream.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_work_item("failed-worker", ("large",))
    assert claimed is not None and claimed.work_item_key == upstream.key
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="rebuild failure",
    )
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='missing',artifact_reason='Purged by user' "
            "WHERE work_item_key=?",
            (upstream.key,),
        )

    snapshot = {row["work_item_key"]: row for row in registry.work_item_status_snapshot()}
    root_id = snapshot[upstream.key]["id"]
    assert snapshot[upstream.key]["status"] == "Error"
    assert snapshot[upstream.key]["root_failure_ids"] == (root_id,)
    assert snapshot[downstream.key]["status"] == "Blocked"
    assert snapshot[downstream.key]["root_failure_ids"] == (root_id,)


def test_oom_escalates_memory_and_larger_worker_retries(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "nro" / "networks" / "main" / "sub-01" / "result.txt"
    base = _spec(
        key="networks:" + "e" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    command = (
        sys.executable,
        "-c",
        (
            "import os; from pathlib import Path; "
            "m=int(os.environ['NRO_WORKER_MEMORY_GB']); "
            f"p=Path({str(output)!r}); "
            "(os._exit(137) if m < 64 else (p.parent.mkdir(parents=True, exist_ok=True), p.write_text('ok')))"
        ),
    )
    work_item = base.evolve(command=command, max_memory_gb=256)
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )

    Worker(
        registry, resource_class="large", memory_gb=32, idle_timeout=0.1, poll_interval=0.01
    ).run()
    row = registry.work_item_rows()[0]
    assert row["memory_gb"] == 64
    assert row["oom_count"] == 1
    assert registry.request_rows()[0]["state"] == "active"

    Worker(
        registry, resource_class="large", memory_gb=64, idle_timeout=0.1, poll_interval=0.01
    ).run()
    row = registry.work_item_rows()[0]
    assert row["artifact_state"] == "fresh"
    assert row["oom_count"] == 1
    assert output.read_text() == "ok"


def test_oom_at_memory_ceiling_is_terminal_error(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "nro" / "networks" / "main" / "sub-01" / "result.txt"
    base = _spec(
        key="networks:" + "f" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    work_item = base.evolve(
        command=(sys.executable, "-c", "raise SystemExit(137)"),
        max_memory_gb=32,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )

    Worker(
        registry, resource_class="large", memory_gb=32, idle_timeout=0.1, poll_interval=0.01
    ).run()
    row = registry.work_item_rows()[0]
    assert row["oom_count"] == 1
    assert row["memory_gb"] == 32
    assert registry.request_rows()[0]["state"] == "active"


def test_worker_drains_before_walltime_without_claiming(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "nro" / "networks" / "main" / "sub-01" / "result.txt"
    work_item = _spec(
        key="networks:" + "1" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )

    Worker(
        registry,
        resource_class="large",
        walltime_seconds=1,
        drain_seconds=2,
        idle_timeout=0.1,
        poll_interval=0.01,
    ).run()

    assert not output.exists()
    assert registry.work_item_rows()[0]["attempt_state"] is None


def test_idle_worker_exits_while_another_worker_runs_long_work_item(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="anat:" + "9" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "anat.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=50,
        partition=None,
    )
    registry.register_worker("busy", resource_class="large")
    assert registry.claim_ready_work_item("busy", ("large",)) is not None

    started = time.monotonic()
    Worker(registry, resource_class="large", idle_timeout=0.05, poll_interval=0.01).run()

    # Registry setup can slow under a full parallel test run. This bound still
    # distinguishes an idle exit from waiting for the other worker's attempt.
    assert time.monotonic() - started < 5


def test_targeted_cancellation_prunes_orphaned_dependencies(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    specs = []
    terminals = []
    for participant, marker in (("01", "2"), ("02", "3")):
        anat = _spec(
            key="anat:" + marker * 64,
            module="anat",
            lineage=registered.lineages["anat"],
            config_fingerprint=workflow.configuration("anat").fingerprint,
            runtime_config=registry.runtime_config_path(registered, "anat"),
            output=tmp_path / participant / "anat.txt",
        ).evolve(participant=participant)
        network = _spec(
            key="networks:" + marker * 64,
            module="networks",
            lineage=registered.lineages["networks"],
            config_fingerprint=workflow.configuration("networks").fingerprint,
            runtime_config=registry.runtime_config_path(registered, "networks"),
            output=tmp_path / participant / "network.txt",
            dependencies=(anat.key,),
        ).evolve(participant=participant)
        specs.extend((anat, network))
        terminals.append(network.key)
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=tuple(specs),
        terminal_work_item_keys=tuple(terminals),
        concurrency=1,
        partition=None,
    )

    assert registry.request_cancellation(
        participants=("01",), modules=("networks",), workflows=("missing",)
    ) == {"work_items": 0, "requests": 0, "attempts": 0}
    result = registry.request_cancellation(
        participants=("01",), modules=("networks",), workflows=("main",)
    )

    assert result["work_items"] == 2
    assert registry.request_rows()[0]["state"] == "active"
    with registry.connection() as db:
        demands = {
            (row["participant"], row["module"]): row["demand_state"]
            for row in db.execute(
                """SELECT t.participant, t.module, rt.demand_state
                   FROM request_work_items rt JOIN work_items t ON t.id=rt.work_item_id"""
            )
        }
    assert demands[("01", "anat")] == "cancelled"
    assert demands[("01", "networks")] == "cancelled"
    assert demands[("02", "anat")] == "active"
    assert demands[("02", "networks")] == "active"


def test_future_successor_does_not_suppress_immediate_pool_growth(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "result.txt"
    work_item = _spec(
        key="networks:" + "4" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("live", resource_class="large")
    assert (
        registry.reserve_worker_successor(worker_id="live", resource_class="large", memory_gb=32)
        is not None
    )
    second = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(
            work_item,
            _spec(
                key="networks:" + "6" * 64,
                module="networks",
                lineage=registered.lineages["networks"],
                config_fingerprint=workflow.configuration("networks").fingerprint,
                runtime_config=registry.runtime_config_path(registered, "networks"),
                output=tmp_path / "result-2.txt",
            ),
        ),
        terminal_work_item_keys=(work_item.key, "networks:" + "6" * 64),
        concurrency=2,
        partition=None,
    )

    reservations = registry.reserve_worker_submissions(
        request_id=second, resource_class="large", memory_gb=32
    )
    assert len(reservations) == 1


@pytest.mark.parametrize(
    "returncode,stdout,stderr,expected",
    [
        (0, "RUNNING\n", "", False),
        (0, "PENDING\n", "", False),
        (0, "", "", True),
        (1, "", "slurm_load_jobs error: Invalid job id specified\n", True),
        (1, "", "slurm_load_jobs error: Unable to contact slurm controller", None),
        (1, "", "slurm_load_jobs error: Access denied", None),
        (1, "", "", None),
        (1, "RUNNING\n", "slurm_load_jobs error: Invalid job id specified", None),
        (1, "", "slurm_load_jobs error: Invalid job id specified\nConnection failure", None),
    ],
)
def test_slurm_terminal_distinguishes_absence_from_query_failure(
    monkeypatch,
    returncode,
    stdout,
    stderr,
    expected,
):
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: "/bin/squeue")
    monkeypatch.setattr(RegistryLock, "_slurm_accounting_terminal", lambda _job_id: None)

    def query(command, **kwargs):
        assert command == ["squeue", "--noheader", "--jobs", "123", "--format", "%T"]
        assert kwargs["env"]["LC_ALL"] == "C"
        assert kwargs["timeout"] == 5
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr("nro.orchestration.registry.subprocess.run", query)
    assert RegistryLock._slurm_terminal("123") is expected


@pytest.mark.parametrize("error", [OSError("unavailable"), subprocess.TimeoutExpired("squeue", 15)])
def test_slurm_terminal_execution_errors_remain_unknown(monkeypatch, error):
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: "/bin/squeue")
    monkeypatch.setattr(RegistryLock, "_slurm_accounting_terminal", lambda _job_id: None)

    def query(*args, **kwargs):
        raise error

    monkeypatch.setattr("nro.orchestration.registry.subprocess.run", query)
    assert RegistryLock._slurm_terminal("123") is None
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: None)
    assert RegistryLock._slurm_terminal("123") is None


@pytest.mark.parametrize(
    "state,expected",
    [
        ("RUNNING", False),
        ("PENDING", False),
        ("COMPLETING", False),
        ("COMPLETED", True),
        ("FAILED", True),
        ("CANCELLED by 123", True),
        ("OUT_OF_MEMORY", True),
        ("UNKNOWN", None),
    ],
)
def test_slurm_accounting_provides_conclusive_fallback(monkeypatch, state, expected):
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: "/bin/sacct")

    def query(command, **kwargs):
        assert command == [
            "sacct",
            "--jobs",
            "123",
            "--noheader",
            "--parsable2",
            "--format",
            "JobIDRaw,State",
        ]
        assert kwargs["timeout"] == 10
        assert kwargs["env"]["LC_ALL"] == "C"
        return subprocess.CompletedProcess(command, 0, f"123|{state}\n123.batch|FAILED\n", "")

    monkeypatch.setattr("nro.orchestration.registry.subprocess.run", query)
    assert RegistryLock._slurm_accounting_terminal("123") is expected


def test_observational_connections_share_the_cross_host_lock(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids", lock_timeout=0.05)
    registry.initialize()

    with registry._lock():
        with pytest.raises(RegistryLockTimeout):
            with registry.read_connection():
                pass

    with registry.read_connection() as db:
        assert db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] > 0
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM metadata")


def test_registry_lock_reports_interactive_wait_owner(tmp_path: Path, monkeypatch) -> None:
    class Terminal:
        def __init__(self) -> None:
            self.output = ""

        @staticmethod
        def isatty() -> bool:
            return True

        def write(self, value: str) -> None:
            self.output += value

        @staticmethod
        def flush() -> None:
            pass

    registry = Registry.for_project("demo", bids_root=tmp_path / "bids", lock_timeout=0.05)
    registry.initialize()
    terminal = Terminal()
    monkeypatch.setattr("nro.orchestration.registry.sys.stderr", terminal)
    monkeypatch.setattr("nro.orchestration.registry._WAIT_NOTICE_SECONDS", 0.0)

    with registry._lock():
        with pytest.raises(RegistryLockTimeout):
            with registry.read_connection():
                pass

    assert "Waiting for registry access" in terminal.output
    assert "held by" in terminal.output
    assert socket.gethostname().split(".", 1)[0] in terminal.output
    assert terminal.output.endswith("\r\x1b[2K")


def test_expired_registry_lock_lease_is_recoverable_from_another_host(tmp_path: Path) -> None:
    lock_path = tmp_path / "artifact-mutation.lock"
    recovery_path = tmp_path / "artifact-mutation.recovery-lock"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps(
            {
                "token": "abandoned",
                "hostname": "other-node.example",
                "pid": 123,
                "uid": os.getuid(),
                "slurm_job_id": None,
                "slurm_array_task_id": None,
                "acquired_at": "2026-01-01T00:00:00+00:00",
                "lease_expires_at": time.time() - 1,
            }
        )
    )

    with RegistryLock(lock_path, recovery_path, timeout=0.1, lease_seconds=300):
        owner = json.loads((lock_path / "owner.json").read_text())
        assert owner["token"] != "abandoned"
        assert owner["lease_expires_at"] > time.time()

    assert not lock_path.exists()


def test_registry_lock_renews_its_cross_host_lease(tmp_path: Path, monkeypatch) -> None:
    lock_path = tmp_path / "artifact-mutation.lock"
    lock = RegistryLock(
        lock_path,
        tmp_path / "artifact-mutation.recovery-lock",
        lease_seconds=3600,
    )
    with lock:
        initial = json.loads((lock_path / "owner.json").read_text())["lease_expires_at"]
        monkeypatch.setattr("nro.orchestration.registry.time.time", lambda: initial + 10)
        assert lock._renew_lease()
        renewed = json.loads((lock_path / "owner.json").read_text())["lease_expires_at"]
        assert renewed > initial


@pytest.mark.parametrize(
    "diagnostic,replacements",
    [
        ("slurm_load_jobs error: Invalid job id specified\n", 3),
        ("slurm_load_jobs error: Unable to contact slurm controller\n", 0),
    ],
)
def test_reconciliation_releases_only_confirmed_expired_worker_slots(
    tmp_path,
    monkeypatch,
    diagnostic,
    replacements,
):
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_items = tuple(
        _spec(
            key=f"networks:{index}" + "8" * 63,
            module="networks",
            lineage=registered.lineages["networks"],
            config_fingerprint=workflow.configuration("networks").fingerprint,
            runtime_config=registry.runtime_config_path(registered, "networks"),
            output=tmp_path / f"result-{index}.txt",
        )
        for index in range(4)
    )
    request = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=work_items,
        terminal_work_item_keys=tuple(item.key for item in work_items),
        concurrency=50,
        partition=None,
    )
    submissions = registry.reserve_worker_submissions(request_id=request, resource_class="large")
    assert len(submissions) == 4
    for index, (submission_id, _) in enumerate(submissions):
        registry.update_submission(submission_id, state="submitted", slurm_job_id=str(index))
    registry.register_worker("live", resource_class="large", slurm_job_id="3")
    registry.mark_submission_running("3")
    assert registry.reserve_worker_submissions(request_id=request, resource_class="large") == []
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: "/bin/squeue")

    def query(command, **kwargs):
        if command[command.index("--jobs") + 1] == "3":
            return subprocess.CompletedProcess(command, 0, "RUNNING\n", "")
        return subprocess.CompletedProcess(command, 1, "", diagnostic)

    monkeypatch.setattr("nro.orchestration.registry.subprocess.run", query)
    assert registry.reconcile_scheduler_submissions() == replacements
    assert registry.reconcile_scheduler_submissions() == 0
    reserved = registry.reserve_worker_submissions(request_id=request, resource_class="large")
    assert len(reserved) == replacements
    with registry.connection() as db:
        assert db.execute("SELECT state FROM workers WHERE id='live'").fetchone()[0] == "idle"
        assert (
            db.execute("SELECT state FROM scheduler_submissions WHERE slurm_job_id='3'").fetchone()[
                0
            ]
            == "running"
        )


def test_worker_reservations_follow_current_dag_width(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    runtime = registry.runtime_config_path(registered, "anat")
    func_runtime = registry.runtime_config_path(registered, "func")
    anat = _spec(
        key="anat:" + "7" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "anat.txt",
    )
    funcs = tuple(
        _spec(
            key=f"func:{index}" + "8" * 63,
            module="func",
            lineage=registered.lineages["func"],
            config_fingerprint=workflow.configuration("func").fingerprint,
            runtime_config=func_runtime,
            output=tmp_path / f"func-{index}.txt",
            dependencies=(anat.key,),
        )
        for index in range(50)
    )
    request = registry.create_request(
        registered=registered,
        target_module="func",
        selectors={},
        work_items=(anat, *funcs),
        terminal_work_item_keys=tuple(work_item.key for work_item in funcs),
        concurrency=50,
        partition=None,
    )

    initial = registry.reserve_worker_submissions(
        request_id=request, resource_class="large", memory_gb=32
    )
    assert len(initial) == 1

    registry.update_submission(initial[0][0], state="running", slurm_job_id="101")
    registry.register_worker("worker-101", resource_class="large", slurm_job_id="101")
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh' WHERE work_item_key=?", (anat.key,)
        )

    expanded = registry.reserve_worker_submissions(
        request_id=request, resource_class="large", memory_gb=32
    )
    # One anatomical worker is already live; when it completes and exposes
    # fifty independent runs, the shared concurrency cap of 50 requires 49
    # additional worker submissions.
    assert len(expanded) == 49


def test_cancellation_preserves_another_users_shared_demand(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "5" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "result.txt",
    )
    for owner in ("alice", "bob"):
        registry.create_request(
            registered=registered,
            target_module="networks",
            selectors={},
            work_items=(work_item,),
            terminal_work_item_keys=(work_item.key,),
            concurrency=1,
            partition=None,
            user_name=owner,
        )

    result = registry.request_cancellation(
        participants=("01",), modules=("networks",), user_name="alice"
    )

    assert result["requests"] == 1
    requests = {row["user_name"]: row["state"] for row in registry.request_rows()}
    assert requests == {"alice": "cancelled", "bob": "active"}
    assert registry.work_item_rows()[0]["demanded"] == 1


@pytest.mark.parametrize("launcher_cancelled", [False, True])
def test_worker_walltime_sigterm_is_reported_as_timeout(
    tmp_path: Path, launcher_cancelled: bool
) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "outputs" / "network.txt"
    work_item = _spec(
        key="networks:" + "0" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("worker", resource_class="large")
    claimed = registry.claim_ready_work_item("worker", ("large",))
    assert claimed is not None

    class TimedOutLauncher:
        def terminate(self) -> None:
            pass

        def run(self, *args, **kwargs) -> ExecutionResult:
            return ExecutionResult(-signal.SIGTERM, launcher_cancelled, False)

    worker = Worker(
        registry,
        resource_class="large",
        launcher=TimedOutLauncher(),
        worker_id="worker",
        walltime_seconds=1,
    )
    worker.deadline = time.monotonic()
    worker._execute(claimed)

    row = registry.work_item_status_snapshot()[0]
    assert row["status"] == "Timeout"
    assert row["error_type"] == "Timeout"
    assert "longer --time allocation" in row["error_message"]


def test_status_update_can_reconcile_an_existing_slurm_timeout(tmp_path: Path, monkeypatch) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "9" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "network.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("worker", resource_class="large", slurm_job_id="123")
    claimed = registry.claim_ready_work_item("worker", ("large",))
    assert claimed is not None
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="CalledProcessError",
        error_message="Derivative command exited with status -15: see work-item.log",
    )
    monkeypatch.setattr(RegistryLock, "_slurm_timed_out", lambda job_id: job_id == "123")

    assert registry.reconcile_attempt_timeouts() == 1
    assert registry.reconcile_attempt_timeouts() == 0
    row = registry.work_item_status_snapshot()[0]
    assert row["status"] == "Timeout"
    assert row["error_type"] == "Timeout"


def test_forced_cancellation_removes_every_users_shared_demand(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "f" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "result.txt",
    )
    for owner in ("alice", "bob"):
        registry.create_request(
            registered=registered,
            target_module="networks",
            selectors={},
            work_items=(work_item,),
            terminal_work_item_keys=(work_item.key,),
            concurrency=1,
            partition=None,
            user_name=owner,
        )
    registry.register_worker("shared", resource_class="large")
    assert registry.claim_ready_work_item("shared", ("large",)) is not None

    result = registry.request_cancellation(
        participants=("01",), modules=("networks",), user_name="alice", force=True
    )

    assert result == {"work_items": 2, "requests": 2, "attempts": 1}
    assert {row["state"] for row in registry.request_rows()} == {"cancelled"}
    row = registry.work_item_rows()[0]
    assert row["demanded"] == 0
    assert row["attempt_state"] == "cancel_requested"


def test_worker_shutdown_is_owner_scoped_and_preserves_work_item_demand(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "2" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "result.txt",
    )
    request = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    submission_id, _token = registry.reserve_worker_submissions(
        request_id=request, resource_class="large", memory_gb=32
    )[0]
    registry.update_submission(submission_id, state="submitted", slurm_job_id="101")
    registry.register_worker("alice-worker", resource_class="large", slurm_job_id="101")
    registry.register_worker("bob-worker", resource_class="large", slurm_job_id="202")
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET user_name='alice' WHERE id='alice-worker'")
        db.execute("UPDATE workers SET user_name='bob' WHERE id='bob-worker'")
    claimed = registry.claim_ready_work_item("alice-worker", ("large",))
    assert claimed is not None

    shutdown = registry.request_worker_shutdown(user_name="alice")

    assert shutdown["workers"] == 1
    assert shutdown["attempts"] == 1
    assert shutdown["submission_count"] == 1
    assert shutdown["job_ids"] == ("101",)
    assert shutdown["submissions"] == [(submission_id, "101")]
    with registry.connection() as db:
        workers = {
            str(row["id"]): str(row["state"]) for row in db.execute("SELECT id, state FROM workers")
        }
        attempt = db.execute(
            "SELECT state, error_type FROM attempts WHERE id=?", (claimed.attempt_id,)
        ).fetchone()
    assert workers == {"alice-worker": "shutdown_requested", "bob-worker": "idle"}
    assert tuple(attempt) == ("cancel_requested", "WorkerTerminated")
    assert registry.work_item_rows()[0]["demanded"] == 1

    finalized = registry.confirm_worker_shutdown(("alice-worker",))

    assert finalized == {"workers": 1, "attempts": 1, "ingestion": 0}
    with registry.connection() as db:
        assert db.execute("SELECT state FROM workers WHERE id='alice-worker'").fetchone()[0] == (
            "terminated"
        )
        assert (
            db.execute("SELECT state FROM attempts WHERE id=?", (claimed.attempt_id,)).fetchone()[0]
            == "cancelled"
        )
    assert registry.work_item_rows()[0]["demanded"] == 1


def test_repair_shutdown_covers_all_users_and_blocks_late_workers(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    registry.initialize()
    registry.register_worker("alice-worker", resource_class="large")
    registry.register_worker("bob-worker", resource_class="large")
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET user_name='alice' WHERE id='alice-worker'")
        db.execute("UPDATE workers SET user_name='bob' WHERE id='bob-worker'")

    shutdown = registry.request_worker_shutdown(all_users=True, for_repair=True)
    registry.register_worker("late-worker", resource_class="large")

    assert shutdown["workers"] == 2
    with registry.connection() as db:
        maintenance = db.execute(
            "SELECT value FROM metadata WHERE key='maintenance_mode'"
        ).fetchone()
        workers = {
            str(row["id"]): str(row["state"]) for row in db.execute("SELECT id, state FROM workers")
        }
    assert maintenance["value"] == "repair"
    assert workers == {
        "alice-worker": "shutdown_requested",
        "bob-worker": "shutdown_requested",
        "late-worker": "shutdown_requested",
    }
    assert registry.claim_ready_work_item("late-worker", ("large",)) is None


def test_existing_request_tracks_evolving_shared_multirun_dependencies(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    raw_a = tmp_path / "sub-01_task-rest_run-01_bold.nii.gz"
    raw_b = tmp_path / "sub-01_task-rest_run-02_bold.nii.gz"
    raw_a.write_text("a")
    raw_b.write_text("b")
    parent_a = _spec(
        key="clean:" + "6" * 64,
        module="clean",
        lineage=registered.lineages["clean"],
        config_fingerprint=workflow.configuration("clean").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "clean"),
        output=tmp_path / "a.txt",
    )
    parent_b = _spec(
        key="clean:" + "7" * 64,
        module="clean",
        lineage=registered.lineages["clean"],
        config_fingerprint=workflow.configuration("clean").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "clean"),
        output=tmp_path / "b.txt",
    )
    aggregate = _spec(
        key="microparcellation:" + "8" * 64,
        module="microparcellation",
        lineage=registered.lineages["microparcellation"],
        config_fingerprint=workflow.configuration("microparcellation").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "microparcellation"),
        output=tmp_path / "aggregate.txt",
        dependencies=(parent_a.key,),
        inputs=(raw_a,),
    )
    first = registry.create_request(
        registered=registered,
        target_module="microparcellation",
        selectors={},
        work_items=(parent_a, aggregate),
        terminal_work_item_keys=(aggregate.key,),
        concurrency=1,
        partition=None,
        user_name="first-owner",
    )
    expanded = aggregate.evolve(
        dependencies=(parent_a.key, parent_b.key),
        input_paths=(raw_a, raw_b),
    )
    registry.create_request(
        registered=registered,
        target_module="microparcellation",
        selectors={},
        work_items=(parent_a, parent_b, expanded),
        terminal_work_item_keys=(expanded.key,),
        concurrency=1,
        partition=None,
        user_name="second-owner",
    )

    with registry.connection() as db:
        first_b = db.execute(
            """SELECT rt.demand_state FROM request_work_items rt JOIN work_items t ON t.id=rt.work_item_id
               WHERE rt.request_id=? AND t.work_item_key=?""",
            (first, parent_b.key),
        ).fetchone()
    assert first_b is not None and first_b["demand_state"] == "active"

    registry.request_cancellation(modules=("microparcellation",), user_name="second-owner")
    with registry.connection() as db:
        first_b_after = db.execute(
            """SELECT rt.demand_state FROM request_work_items rt JOIN work_items t ON t.id=rt.work_item_id
               WHERE rt.request_id=? AND t.work_item_key=?""",
            (first, parent_b.key),
        ).fetchone()
    assert first_b_after["demand_state"] == "active"


def test_freshness_detects_newly_matching_multirun_input_before_replanning(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    raw_dir = bids / "demo" / "sub-01" / "func"

    def _write_raw(run):
        return (raw_dir / f"sub-01_task-rest_run-{run}_bold.nii.gz").write_text("raw")

    raw_dir.mkdir(parents=True)
    _write_raw("1")
    (raw_dir / "sub-01_task-rest_run-1_bold.json").write_text("{}")
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    clean_base = _spec(
        key="clean:" + "9" * 64,
        module="clean",
        lineage=registered.lineages["clean"],
        config_fingerprint=workflow.configuration("clean").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "clean"),
        output=tmp_path / "clean-output" / "clean.txt",
    )
    clean = clean_base.evolve(
        entities={"task": "rest", "run": "1"},
        command=(*clean_base.command, "nro.modules.clean"),
    )
    micro = _spec(
        key="microparcellation:" + "a" * 64,
        module="microparcellation",
        lineage=registered.lineages["microparcellation"],
        config_fingerprint=workflow.configuration("microparcellation").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "microparcellation"),
        output=tmp_path / "micro-output" / "micro.txt",
        dependencies=(clean.key,),
    )
    registry.create_request(
        registered=registered,
        target_module="microparcellation",
        selectors={},
        work_items=(clean, micro),
        terminal_work_item_keys=(micro.key,),
        concurrency=1,
        partition=None,
    )
    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()
    assert {row["module"]: row["artifact_state"] for row in registry.work_item_rows()} == {
        "clean": "fresh",
        "microparcellation": "fresh",
    }

    events = raw_dir / "sub-01_task-rest_run-1_events.tsv"
    events.write_text("onset\tduration\n0\t1\n")
    event_states = assess_registry(registry)
    clean_row = next(row for row in registry.work_item_rows() if row["module"] == "clean")
    assert event_states[clean_row["id"]][0] == "stale"
    assert "direct input set changed" in event_states[clean_row["id"]][1]
    events.unlink()
    assert assess_registry(registry)[clean_row["id"]][0] == "stale"
    assert assess_registry(registry)[clean_row["id"]][0] == "fresh"

    _write_raw("2")
    (raw_dir / "sub-01_task-rest_run-2_bold.json").write_text("{}")
    states = assess_registry(registry)
    micro_row = next(
        row for row in registry.work_item_rows() if row["module"] == "microparcellation"
    )

    assert states[micro_row["id"]][0] == "stale"
    assert "raw run universe changed" in states[micro_row["id"]][1]
    with registry.connection(write=True) as db:
        db.execute("UPDATE requests SET state='active'")
    registry.register_worker("waiting-for-replan", resource_class="large")
    assert registry.claim_ready_work_item("waiting-for-replan", ("large",)) is None


def test_successor_reconciles_recovered_oom_at_memory_ceiling(tmp_path: Path, monkeypatch) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    base = _spec(
        key="networks:" + "b" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "output" / "result.txt",
    )
    work_item = base.evolve(max_memory_gb=32)
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("oom-worker", resource_class="large", memory_gb=32, slurm_job_id="123")
    assert registry.claim_ready_work_item("oom-worker", ("large",), memory_gb=32)
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET pid=999999999, lease_expires_at=0 WHERE id='oom-worker'")
    monkeypatch.setattr(
        "nro.orchestration.registry.RegistryLock._slurm_out_of_memory",
        staticmethod(lambda _job: True),
    )
    monkeypatch.setattr(
        "nro.orchestration.registry.RegistryLock._slurm_terminal",
        staticmethod(lambda _job: True),
    )
    refresh_scheduler_state(registry)

    Worker(
        registry,
        resource_class="large",
        memory_gb=32,
        idle_timeout=0.05,
        poll_interval=0.01,
    ).run()

    assert registry.request_rows()[0]["state"] == "active"
    row = registry.work_item_rows()[0]
    assert row["oom_count"] == 1
    assert "OUT_OF_MEMORY" in row["error_message"]


def test_recovered_gpu_step_oom_does_not_inflate_parent_memory(tmp_path: Path, monkeypatch) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="anat:" + "g" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "output" / "result.txt",
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
    registry.register_worker("cpu", resource_class="large", memory_gb=32)
    parent = registry.claim_ready_work_item("cpu", ("large",), memory_gb=32)
    assert parent is not None
    registry.defer_resource_step(
        parent.attempt_id,
        step_id="neurolit-inpainting",
        resource_class="gpu",
        memory_gb=32,
    )
    registry.close_worker("cpu")
    registry.register_worker("gpu-oom", resource_class="gpu", memory_gb=32, slurm_job_id="456")
    task = registry.claim_resource_step("gpu-oom", resource_class="gpu", memory_gb=32)
    assert task is not None
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET pid=999999999, lease_expires_at=0 WHERE id='gpu-oom'")
    monkeypatch.setattr(
        "nro.orchestration.registry.RegistryLock._slurm_out_of_memory",
        staticmethod(lambda _job: True),
    )
    monkeypatch.setattr(
        "nro.orchestration.registry.RegistryLock._slurm_terminal",
        staticmethod(lambda _job: True),
    )

    assert registry.recover_orphaned_attempts() == 1

    row = registry.work_item_rows()[0]
    assert row["memory_gb"] == 32
    assert row["oom_count"] == 0
    with registry.connection() as db:
        attempt = db.execute(
            "SELECT state,error_type,error_message FROM attempts WHERE id=?",
            (task.attempt_id,),
        ).fetchone()
        resource_task = db.execute(
            "SELECT state,error_type,error_message FROM resource_step_tasks WHERE id=?",
            (task.resource_task_id,),
        ).fetchone()
    assert dict(attempt) == {
        "state": "error",
        "error_type": "OutOfMemory",
        "error_message": "Slurm reported OUT_OF_MEMORY for worker job 456",
    }
    assert dict(resource_task) == dict(attempt)


def test_new_demand_survives_an_inflight_cancellation(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    work_item = _spec(
        key="networks:" + "d" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "output" / "result.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("first", resource_class="large")
    claimed = registry.claim_ready_work_item("first", ("large",))
    assert claimed is not None
    registry.request_cancellation(participants=("01",), modules=("networks",))
    assert registry.work_item_rows()[0]["attempt_state"] == "cancel_requested"

    second_request = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(work_item,),
        terminal_work_item_keys=(work_item.key,),
        concurrency=1,
        partition=None,
    )
    registry.finish_attempt(
        claimed.attempt_id,
        state="cancelled",
        error_message="Cancellation requested",
    )
    registry.reconcile_requests()
    requests = {row["id"]: row["state"] for row in registry.request_rows()}
    assert requests[second_request] == "active"

    registry.register_worker("second", resource_class="large")
    retried = registry.claim_ready_work_item("second", ("large",))
    assert retried is not None
    assert retried.work_item_key == work_item.key


def test_status_is_read_only_and_worker_cancels_stale_downstream(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    upstream = _spec(
        key="anat:" + "c" * 64,
        module="anat",
        lineage=registered.lineages["anat"],
        config_fingerprint=workflow.configuration("anat").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "anat"),
        output=tmp_path / "outputs" / "anat.txt",
    )
    downstream = _spec(
        key="networks:" + "d" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=tmp_path / "outputs" / "networks.txt",
        dependencies=(upstream.key,),
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        work_items=(upstream, downstream),
        terminal_work_item_keys=(downstream.key,),
        concurrency=2,
        partition=None,
    )
    rows = {row["work_item_key"]: row for row in registry.work_item_rows()}
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh' WHERE id=?", (rows[upstream.key]["id"],)
        )
        db.execute(
            "UPDATE work_items SET artifact_state='stale' WHERE id=?", (rows[downstream.key]["id"],)
        )
    registry.register_worker("downstream-worker", resource_class="large")
    claimed = registry.claim_ready_work_item("downstream-worker", ("large",))
    assert claimed is not None and claimed.work_item_key == downstream.key
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='stale' WHERE id=?", (rows[upstream.key]["id"],)
        )
    status_main(["-p", "demo", "--json"])
    assert not registry.attempt_cancel_requested(claimed.attempt_id)

    refresh_scheduler_state(registry)
    assert registry.attempt_cancel_requested(claimed.attempt_id)


def test_fatal_work_item_failure_cancels_active_transitive_descendants(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    specs = []
    previous = None
    for index, module in enumerate(("anat", "func", "clean")):
        configuration_class = module
        lineage = (
            registered.lineages["anat"]
            if module == "anat"
            else registered.lineages["func"]
            if module == "func"
            else registered.lineages["clean"]
        )
        spec = _spec(
            key=f"{module}:" + chr(ord("e") + index) * 64,
            module=module,
            lineage=lineage,
            config_fingerprint=workflow.configuration(configuration_class).fingerprint,
            runtime_config=registry.runtime_config_path(registered, configuration_class),
            output=tmp_path / "outputs" / f"{module}.txt",
            dependencies=((previous.key,) if previous else ()),
        )
        specs.append(spec)
        previous = spec
    registry.create_request(
        registered=registered,
        target_module="clean",
        selectors={},
        work_items=tuple(specs),
        terminal_work_item_keys=(specs[-1].key,),
        concurrency=3,
        partition=None,
    )
    rows = {row["work_item_key"]: row for row in registry.work_item_rows()}
    with registry.connection(write=True) as db:
        for spec in specs:
            db.execute(
                "UPDATE work_items SET artifact_state='fresh' WHERE id=?", (rows[spec.key]["id"],)
            )
        db.execute(
            "UPDATE work_items SET artifact_state='stale' WHERE id=?", (rows[specs[-1].key]["id"],)
        )
    registry.register_worker("clean-worker", resource_class="large")
    claimed = registry.claim_ready_work_item("clean-worker", ("large",))
    assert claimed is not None and claimed.work_item_key == specs[-1].key
    cancelled = registry.cancel_attempts_downstream_of_failure(int(rows[specs[0].key]["id"]))
    assert [row["work_item_key"] for row in cancelled] == [specs[-1].key]
    assert registry.attempt_cancel_requested(claimed.attempt_id)
