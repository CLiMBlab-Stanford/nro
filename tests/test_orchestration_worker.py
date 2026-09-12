from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nro.bin.status import main as status_main
from nro.configuration.store import ConfigStore
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.manifests import assess_registry, file_record
from nro.orchestration.publish import publish
from nro.orchestration.registry import (
    Registry,
    RegistryLock,
    RegistryLockTimeout,
    discover_registry_projects,
)
from nro.orchestration.worker import Worker


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
) -> InstanceSpec:
    command = (
        sys.executable,
        "-c",
        f"from pathlib import Path; p=Path({str(output)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok')",
    )
    return InstanceSpec.create(
        key=key,
        module=module,
        project=project,
        participant="01",
        entities={},
        scope="subject",
        configuration_lineage_id=lineage,
        config_fingerprint=config_fingerprint,
        directory_label="main",
        runtime_config=runtime_config,
        command=command,
        dependencies=dependencies,
        input_paths=inputs,
        output_root=output.parent,
        output_prefix=None,
        resource_class="large",
        expected_outputs=(output,),
        processing=module_descriptor(module).processing_contract(),
    )


def test_existing_instance_adopts_current_configuration_snapshot(tmp_path: Path) -> None:
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
    registry.register_instances((first,))
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh', artifact_reason='Current' "
            "WHERE instance_key=?",
            (first.key,),
        )
    current_runtime = tmp_path / "current_networks.yml"
    current_runtime.write_text("labeling:\n  enabled: true\n", encoding="utf-8")
    current = first.evolve(
        config_fingerprint="current-configuration",
        runtime_config=current_runtime,
    )

    registry.register_instances((current,))

    row = next(item for item in registry.instance_rows() if item["instance_key"] == first.key)
    assert row["runtime_config_path"] == str(current_runtime)
    assert row["revision_fingerprint"] == current.revision_fingerprint
    assert row["artifact_state"] == "stale"
    assert row["artifact_reason"] == "Instance contract changed"


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
        instances=(alpha,),
        terminal_instance_keys=(alpha.key,),
        concurrency=1,
        partition=None,
    )
    beta_request = beta_registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        instances=(beta,),
        terminal_instance_keys=(beta.key,),
        concurrency=1,
        partition=None,
    )

    assert alpha_registry.paths.database == beta_registry.paths.database
    assert alpha_registry.paths.control == tmp_path / ".nro"
    assert discover_registry_projects(bids) == ["alpha", "beta"]
    rows = {row["project"]: row for row in alpha_registry.instance_rows()}
    assert (
        Path(rows["alpha"]["manifest_path"]).relative_to(alpha_registry.paths.manifests).parts[0]
        == "alpha"
    )
    assert (
        Path(rows["beta"]["manifest_path"]).relative_to(alpha_registry.paths.manifests).parts[0]
        == "beta"
    )
    with pytest.raises(ValueError, match="only its selected project"):
        beta_registry.register_instances((alpha,))
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
    network_output = bids / "demo" / "derivatives" / "networks" / "main" / "sub-01" / "network.txt"
    anat = _spec(
        key="anat:" + "a" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        output=anat_output,
        inputs=(raw,),
    )
    anat = anat.evolve(
        config_fingerprint=workflow.configuration("preprocessing").scientific_fingerprint
    )
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
        instances=(anat, network),
        terminal_instance_keys=(network.key,),
        concurrency=1,
        partition=None,
    )

    assert Worker(registry, resource_class="large", idle_timeout=0.2, poll_interval=0.02).run() == 0
    worker_log = capsys.readouterr().out
    assert "started (Slurm job " in worker_log
    assert "; class=large; memory=32 GB)" in worker_log
    assert "claimed instance" in worker_log
    assert "instance log:" in worker_log
    assert "attempt state=success" in worker_log
    assert "idle timeout reached" in worker_log
    assert "stopped (state=exited)" in worker_log
    rows = {row["module"]: row for row in registry.instance_rows()}
    assert rows["anat"]["artifact_state"] == "fresh"
    assert rows["networks"]["artifact_state"] == "fresh"
    assert rows["anat"]["current_generation"] == 1
    assert registry.request_rows()[0]["state"] == "satisfied"
    manifest = json.loads(Path(rows["anat"]["manifest_path"]).read_text())
    assert manifest["configuration"]["id"] == "main"
    assert manifest["runtime_config"]["sha256"]
    assert manifest["software"]["name"] == "nro"
    assert manifest["public_outputs"]
    assert manifest["artifact_contract"] == json.loads(rows["anat"]["artifact_contract_json"])
    assert manifest["artifact_fingerprint"] == rows["anat"]["artifact_fingerprint"]
    manifest_path = Path(rows["anat"]["manifest_path"])
    manifest["private_artifacts"].append(file_record(manifest_path))
    manifest_path.write_text(json.dumps(manifest))
    # A ledger may record its own control-plane certificate. This is execution
    # metadata, not a private scientific artifact.
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_fingerprint='changed-contract' WHERE id=?",
            (rows["anat"]["id"],),
        )
    state = assess_registry(registry)[rows["anat"]["id"]]
    assert state == ("stale", "Instance contract changed")
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_fingerprint=? WHERE id=?",
            (manifest["artifact_fingerprint"], rows["anat"]["id"]),
        )
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    steps = json.loads((Path(rows["anat"]["log_path"]).parent / "current-steps.json").read_text())
    assert steps["orchestration:completion-manifest"]["status"] == "success"
    assert stat.S_IMODE(Path(rows["anat"]["log_path"]).parent.stat().st_mode) == 0o2775
    assert stat.S_IMODE(Path(rows["anat"]["manifest_path"]).parent.stat().st_mode) == 0o2775

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
    assert "configuration" in provenance["instances"][0]
    assert provenance["instances"][0]["upstream"][0]["module"] == "anat"

    # Existing private artifacts are part of the certificate, but WORK cleanup
    # is allowed once public derivatives are complete.
    private = tmp_path / "work" / "intermediate.txt"
    private.parent.mkdir(parents=True)
    private.write_text("original")
    manifest["private_artifacts"] = [file_record(private)]
    Path(rows["anat"]["manifest_path"]).write_text(json.dumps(manifest))
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    private.unlink()
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"
    private.write_text("changed")
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "stale"
    private.unlink()
    assert assess_registry(registry)[rows["anat"]["id"]][0] == "fresh"

    raw.write_text("changed")
    states = assess_registry(registry)
    rows = {row["module"]: row for row in registry.instance_rows()}
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
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
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

    row = registry.instance_rows()[0]
    assert row["artifact_state"] == "fresh"
    with registry.connection() as db:
        attempts = list(
            db.execute(
                "SELECT state,error_type FROM attempts WHERE instance_id=? ORDER BY id",
                (row["id"],),
            )
        )
    assert [tuple(attempt) for attempt in attempts] == [
        ("cancelled", "RegistryUnavailable"),
        ("success", None),
    ]


def test_resumed_instance_reuses_one_fixed_instance_log(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("first", resource_class="large")
    first = registry.claim_ready_instance("first", ("large",))
    assert first is not None
    registry.finish_attempt(first.attempt_id, state="cancelled", error_type="UpstreamStale")

    registry.register_worker("second", resource_class="large")
    second = registry.claim_ready_instance("second", ("large",))
    assert second is not None
    assert first.log_path == second.log_path
    assert first.log_path.name == "instance.log"


def test_user_cancelled_attempt_requires_new_run_request(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("first", resource_class="large")
    first = registry.claim_ready_instance("first", ("large",))
    assert first is not None
    cancellation = registry.request_cancellation(modules=("networks",))
    assert cancellation["attempts"] == 1
    assert registry.instance_status_snapshot()[0]["status"] == "Stopping"
    registry.finish_attempt(first.attempt_id, state="cancelled")
    stopped = registry.instance_status_snapshot()[0]
    assert stopped["status"] == "Stopped"
    assert stopped["error_type"] == "UserCancelled"

    registry.register_worker("before-new-run", resource_class="large")
    assert registry.claim_ready_instance("before-new-run", ("large",)) is None

    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )
    assert registry.instance_status_snapshot()[0]["status"] == "Queued"
    registry.register_worker("after-new-run", resource_class="large")
    assert registry.claim_ready_instance("after-new-run", ("large",)) is not None


def test_missing_private_manifest_uses_native_filesystem_evidence(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    raw = tmp_path / "raw_T1w.nii.gz"
    raw.write_text("raw")
    output_root = bids / "demo" / "derivatives" / "preprocessing" / "main" / "sub-01" / "anat"
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
    instance = _spec(
        key="anat:" + "0" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        output=derivative,
        inputs=(raw,),
    ).evolve(
        output_root=output_root,
        output_prefix="sub-01",
        expected_outputs=(native_manifest,),
    )
    instance_ids = registry.register_instances((instance,))
    row = registry.instance_rows()[0]
    private_manifest = Path(row["manifest_path"])
    private_manifest.parent.mkdir(parents=True, exist_ok=True)
    private_manifest.write_text("not valid JSON")

    states = assess_registry(registry, instance_ids=instance_ids.values())

    assert states[row["id"]][0] == "fresh", states[row["id"]]
    assert "private orchestration provenance is unavailable" in states[row["id"]][1]
    assert private_manifest.read_text() == "not valid JSON"

    native_payload = json.loads(native_manifest.read_text())
    native_payload["output_metadata_contract"] = {"layout": "superseded"}
    native_manifest.write_text(json.dumps(native_payload))
    states = assess_registry(registry, instance_ids=instance_ids.values())
    assert states[row["id"]][0] == "stale"
    assert "current contract" in states[row["id"]][1]

    native_payload["output_metadata_contract"] = module_descriptor("anat").processing_contract()[
        "output_metadata"
    ]
    native_manifest.write_text(json.dumps(native_payload))
    future_ns = native_manifest.stat().st_mtime_ns + 10_000_000_000
    os.utime(raw, ns=(future_ns, future_ns))
    states = assess_registry(registry, instance_ids=instance_ids.values())
    assert states[row["id"]][0] == "stale"
    assert "direct input is newer" in states[row["id"]][1]


def test_failed_instance_requires_new_demand_before_retry(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "networks" / "main" / "sub-01" / "result.txt"
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
        instances=(failing,),
        terminal_instance_keys=(failing.key,),
        concurrency=1,
        partition=None,
    )
    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()
    assert registry.request_rows()[0]["state"] == "active"
    assert registry.instance_rows()[0]["attempt_state"] == "error"

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
        instances=(failing,),
        terminal_instance_keys=(failing.key,),
        concurrency=1,
        partition=None,
    )
    assert registry.instance_rows()[0]["retry_requested"] == 1
    assert registry.instance_status_snapshot()[0]["status"] == "Queued"
    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()
    assert output.read_text() == "ok"
    requests = {row["id"]: row for row in registry.request_rows()}
    assert requests[second_request]["state"] == "satisfied"


def test_expired_dead_worker_lease_releases_instance_for_successor(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "networks" / "main" / "sub-01" / "result.txt"
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("dead", resource_class="large")
    assert registry.claim_ready_instance("dead", ("large",)) is not None
    with registry.connection(write=True) as db:
        db.execute("UPDATE workers SET pid=999999999, lease_expires_at=0 WHERE id='dead'")

    assert registry.recover_orphaned_attempts() == 1
    registry.register_worker("successor", resource_class="large")
    claimed = registry.claim_ready_instance("successor", ("large",))
    assert claimed is not None
    assert claimed.instance_key == instance.key


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
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
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
        instances=(upstream, downstream),
        terminal_instance_keys=(downstream.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_instance("failed-worker", ("large",))
    assert claimed is not None and claimed.instance_key == upstream.key
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="historical failure",
    )

    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh' WHERE instance_key=?",
            (upstream.key,),
        )

    snapshot = {row["instance_key"]: row for row in registry.instance_status_snapshot()}
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
    instance = _spec(
        key="anat:" + "j" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        output=tmp_path / "outputs" / "anat.txt",
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
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_instance("failed-worker", ("large",))
    assert claimed is not None
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="historical failure",
    )
    with registry.connection(write=True) as db:
        db.execute("UPDATE requests SET state='cancelled'")
        db.execute("UPDATE request_instances SET demand_state='cancelled'")
        db.execute("UPDATE instances SET artifact_state='missing',artifact_reason='Purged by user'")

    row = registry.instance_status_snapshot()[0]
    assert row["attempt_state"] == "error"
    assert row["status"] == "Missing"
    assert row["artifact_reason"] == "Purged by user"


def test_failed_rebuild_after_purge_blocks_demanded_descendants(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    upstream = _spec(
        key="anat:" + "k" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
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
        instances=(upstream, downstream),
        terminal_instance_keys=(downstream.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_instance("failed-worker", ("large",))
    assert claimed is not None and claimed.instance_key == upstream.key
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="rebuild failure",
    )
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='missing',artifact_reason='Purged by user' "
            "WHERE instance_key=?",
            (upstream.key,),
        )

    snapshot = {row["instance_key"]: row for row in registry.instance_status_snapshot()}
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
    output = bids / "demo" / "derivatives" / "networks" / "main" / "sub-01" / "result.txt"
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
    instance = base.evolve(command=command, max_memory_gb=256)
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )

    Worker(
        registry, resource_class="large", memory_gb=32, idle_timeout=0.1, poll_interval=0.01
    ).run()
    row = registry.instance_rows()[0]
    assert row["memory_gb"] == 64
    assert row["oom_count"] == 1
    assert registry.request_rows()[0]["state"] == "active"

    Worker(
        registry, resource_class="large", memory_gb=64, idle_timeout=0.1, poll_interval=0.01
    ).run()
    row = registry.instance_rows()[0]
    assert row["artifact_state"] == "fresh"
    assert row["oom_count"] == 1
    assert output.read_text() == "ok"


def test_oom_at_memory_ceiling_is_terminal_error(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "networks" / "main" / "sub-01" / "result.txt"
    base = _spec(
        key="networks:" + "f" * 64,
        module="networks",
        lineage=registered.lineages["networks"],
        config_fingerprint=workflow.configuration("networks").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "networks"),
        output=output,
    )
    instance = base.evolve(
        command=(sys.executable, "-c", "raise SystemExit(137)"),
        max_memory_gb=32,
    )
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )

    Worker(
        registry, resource_class="large", memory_gb=32, idle_timeout=0.1, poll_interval=0.01
    ).run()
    row = registry.instance_rows()[0]
    assert row["oom_count"] == 1
    assert row["memory_gb"] == 32
    assert registry.request_rows()[0]["state"] == "active"


def test_worker_drains_before_walltime_without_claiming(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = bids / "demo" / "derivatives" / "networks" / "main" / "sub-01" / "result.txt"
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
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
    assert registry.instance_rows()[0]["attempt_state"] is None


def test_idle_worker_exits_while_another_worker_runs_long_instance(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    instance = _spec(
        key="anat:" + "9" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
        output=tmp_path / "anat.txt",
    )
    registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=50,
        partition=None,
    )
    registry.register_worker("busy", resource_class="large")
    assert registry.claim_ready_instance("busy", ("large",)) is not None

    started = time.monotonic()
    Worker(registry, resource_class="large", idle_timeout=0.05, poll_interval=0.01).run()

    assert time.monotonic() - started < 0.5


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
            lineage=registered.lineages["preprocessing"],
            config_fingerprint=workflow.configuration("preprocessing").fingerprint,
            runtime_config=registry.runtime_config_path(registered, "preprocessing"),
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
        instances=tuple(specs),
        terminal_instance_keys=tuple(terminals),
        concurrency=1,
        partition=None,
    )

    assert registry.request_cancellation(
        participants=("01",), modules=("networks",), workflows=("missing",)
    ) == {"instances": 0, "requests": 0, "attempts": 0}
    result = registry.request_cancellation(
        participants=("01",), modules=("networks",), workflows=("main",)
    )

    assert result["instances"] == 2
    assert registry.request_rows()[0]["state"] == "active"
    with registry.connection() as db:
        demands = {
            (row["participant"], row["module"]): row["demand_state"]
            for row in db.execute(
                """SELECT t.participant, t.module, rt.demand_state
                   FROM request_instances rt JOIN instances t ON t.id=rt.instance_id"""
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
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
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
        instances=(
            instance,
            _spec(
                key="networks:" + "6" * 64,
                module="networks",
                lineage=registered.lineages["networks"],
                config_fingerprint=workflow.configuration("networks").fingerprint,
                runtime_config=registry.runtime_config_path(registered, "networks"),
                output=tmp_path / "result-2.txt",
            ),
        ),
        terminal_instance_keys=(instance.key, "networks:" + "6" * 64),
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

    def query(command, **kwargs):
        assert command == ["squeue", "--noheader", "--jobs", "123", "--format", "%T"]
        assert kwargs["env"]["LC_ALL"] == "C"
        assert kwargs["timeout"] == 15
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr("nro.orchestration.registry.subprocess.run", query)
    assert RegistryLock._slurm_terminal("123") is expected


@pytest.mark.parametrize("error", [OSError("unavailable"), subprocess.TimeoutExpired("squeue", 15)])
def test_slurm_terminal_execution_errors_remain_unknown(monkeypatch, error):
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: "/bin/squeue")

    def query(*args, **kwargs):
        raise error

    monkeypatch.setattr("nro.orchestration.registry.subprocess.run", query)
    assert RegistryLock._slurm_terminal("123") is None
    monkeypatch.setattr("nro.orchestration.registry.shutil.which", lambda name: None)
    assert RegistryLock._slurm_terminal("123") is None


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
    instances = tuple(
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
        instances=instances,
        terminal_instance_keys=tuple(item.key for item in instances),
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
    runtime = registry.runtime_config_path(registered, "preprocessing")
    anat = _spec(
        key="anat:" + "7" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=runtime,
        output=tmp_path / "anat.txt",
    )
    funcs = tuple(
        _spec(
            key=f"func:{index}" + "8" * 63,
            module="func",
            lineage=registered.lineages["preprocessing"],
            config_fingerprint=workflow.configuration("preprocessing").fingerprint,
            runtime_config=runtime,
            output=tmp_path / f"func-{index}.txt",
            dependencies=(anat.key,),
        )
        for index in range(50)
    )
    request = registry.create_request(
        registered=registered,
        target_module="func",
        selectors={},
        instances=(anat, *funcs),
        terminal_instance_keys=tuple(instance.key for instance in funcs),
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
        db.execute("UPDATE instances SET artifact_state='fresh' WHERE instance_key=?", (anat.key,))

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
    instance = _spec(
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
            instances=(instance,),
            terminal_instance_keys=(instance.key,),
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
    assert registry.instance_rows()[0]["demanded"] == 1


def test_forced_cancellation_removes_every_users_shared_demand(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    instance = _spec(
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
            instances=(instance,),
            terminal_instance_keys=(instance.key,),
            concurrency=1,
            partition=None,
            user_name=owner,
        )
    registry.register_worker("shared", resource_class="large")
    assert registry.claim_ready_instance("shared", ("large",)) is not None

    result = registry.request_cancellation(
        participants=("01",), modules=("networks",), user_name="alice", force=True
    )

    assert result == {"instances": 2, "requests": 2, "attempts": 1}
    assert {row["state"] for row in registry.request_rows()} == {"cancelled"}
    row = registry.instance_rows()[0]
    assert row["demanded"] == 0
    assert row["attempt_state"] == "cancel_requested"


def test_worker_shutdown_is_owner_scoped_and_preserves_instance_demand(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
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
    claimed = registry.claim_ready_instance("alice-worker", ("large",))
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
    assert registry.instance_rows()[0]["demanded"] == 1


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
    assert registry.claim_ready_instance("late-worker", ("large",)) is None


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
        instances=(parent_a, aggregate),
        terminal_instance_keys=(aggregate.key,),
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
        instances=(parent_a, parent_b, expanded),
        terminal_instance_keys=(expanded.key,),
        concurrency=1,
        partition=None,
        user_name="second-owner",
    )

    with registry.connection() as db:
        first_b = db.execute(
            """SELECT rt.demand_state FROM request_instances rt JOIN instances t ON t.id=rt.instance_id
               WHERE rt.request_id=? AND t.instance_key=?""",
            (first, parent_b.key),
        ).fetchone()
    assert first_b is not None and first_b["demand_state"] == "active"

    registry.request_cancellation(modules=("microparcellation",), user_name="second-owner")
    with registry.connection() as db:
        first_b_after = db.execute(
            """SELECT rt.demand_state FROM request_instances rt JOIN instances t ON t.id=rt.instance_id
               WHERE rt.request_id=? AND t.instance_key=?""",
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
        instances=(clean, micro),
        terminal_instance_keys=(micro.key,),
        concurrency=1,
        partition=None,
    )
    Worker(registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01).run()
    assert {row["module"]: row["artifact_state"] for row in registry.instance_rows()} == {
        "clean": "fresh",
        "microparcellation": "fresh",
    }

    events = raw_dir / "sub-01_task-rest_run-1_events.tsv"
    events.write_text("onset\tduration\n0\t1\n")
    event_states = assess_registry(registry)
    clean_row = next(row for row in registry.instance_rows() if row["module"] == "clean")
    assert event_states[clean_row["id"]][0] == "stale"
    assert "direct input set changed" in event_states[clean_row["id"]][1]
    events.unlink()
    assert assess_registry(registry)[clean_row["id"]][0] == "stale"
    assert assess_registry(registry)[clean_row["id"]][0] == "fresh"

    _write_raw("2")
    (raw_dir / "sub-01_task-rest_run-2_bold.json").write_text("{}")
    states = assess_registry(registry)
    micro_row = next(
        row for row in registry.instance_rows() if row["module"] == "microparcellation"
    )

    assert states[micro_row["id"]][0] == "stale"
    assert "raw run universe changed" in states[micro_row["id"]][1]
    with registry.connection(write=True) as db:
        db.execute("UPDATE requests SET state='active'")
    registry.register_worker("waiting-for-replan", resource_class="large")
    assert registry.claim_ready_instance("waiting-for-replan", ("large",)) is None


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
    instance = base.evolve(max_memory_gb=32)
    registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("oom-worker", resource_class="large", memory_gb=32, slurm_job_id="123")
    assert registry.claim_ready_instance("oom-worker", ("large",), memory_gb=32)
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

    Worker(
        registry,
        resource_class="large",
        memory_gb=32,
        idle_timeout=0.05,
        poll_interval=0.01,
    ).run()

    assert registry.request_rows()[0]["state"] == "active"
    row = registry.instance_rows()[0]
    assert row["oom_count"] == 1
    assert "OUT_OF_MEMORY" in row["error_message"]


def test_new_demand_survives_an_inflight_cancellation(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    instance = _spec(
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
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
        concurrency=1,
        partition=None,
    )
    registry.register_worker("first", resource_class="large")
    claimed = registry.claim_ready_instance("first", ("large",))
    assert claimed is not None
    registry.request_cancellation(participants=("01",), modules=("networks",))
    assert registry.instance_rows()[0]["attempt_state"] == "cancel_requested"

    second_request = registry.create_request(
        registered=registered,
        target_module="networks",
        selectors={},
        instances=(instance,),
        terminal_instance_keys=(instance.key,),
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
    retried = registry.claim_ready_instance("second", ("large",))
    assert retried is not None
    assert retried.instance_key == instance.key


def test_status_is_read_only_and_worker_cancels_stale_downstream(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    upstream = _spec(
        key="anat:" + "c" * 64,
        module="anat",
        lineage=registered.lineages["preprocessing"],
        config_fingerprint=workflow.configuration("preprocessing").fingerprint,
        runtime_config=registry.runtime_config_path(registered, "preprocessing"),
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
        instances=(upstream, downstream),
        terminal_instance_keys=(downstream.key,),
        concurrency=2,
        partition=None,
    )
    rows = {row["instance_key"]: row for row in registry.instance_rows()}
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh' WHERE id=?", (rows[upstream.key]["id"],)
        )
        db.execute(
            "UPDATE instances SET artifact_state='stale' WHERE id=?", (rows[downstream.key]["id"],)
        )
    registry.register_worker("downstream-worker", resource_class="large")
    claimed = registry.claim_ready_instance("downstream-worker", ("large",))
    assert claimed is not None and claimed.instance_key == downstream.key
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='stale' WHERE id=?", (rows[upstream.key]["id"],)
        )
    status_main(["-p", "demo", "--json"])
    assert not registry.attempt_cancel_requested(claimed.attempt_id)

    Worker(
        registry, resource_class="large", idle_timeout=0.1, poll_interval=0.01
    )._refresh_scheduler_state()
    assert registry.attempt_cancel_requested(claimed.attempt_id)


def test_fatal_instance_failure_cancels_active_transitive_descendants(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    specs = []
    previous = None
    for index, module in enumerate(("anat", "func", "clean")):
        spec = _spec(
            key=f"{module}:" + chr(ord("e") + index) * 64,
            module=module,
            lineage=registered.lineages["preprocessing" if module in {"anat", "func"} else "clean"],
            config_fingerprint=workflow.configuration(
                "preprocessing" if module in {"anat", "func"} else "clean"
            ).fingerprint,
            runtime_config=registry.runtime_config_path(
                registered, "preprocessing" if module in {"anat", "func"} else "clean"
            ),
            output=tmp_path / "outputs" / f"{module}.txt",
            dependencies=((previous.key,) if previous else ()),
        )
        specs.append(spec)
        previous = spec
    registry.create_request(
        registered=registered,
        target_module="clean",
        selectors={},
        instances=tuple(specs),
        terminal_instance_keys=(specs[-1].key,),
        concurrency=3,
        partition=None,
    )
    rows = {row["instance_key"]: row for row in registry.instance_rows()}
    with registry.connection(write=True) as db:
        for spec in specs:
            db.execute(
                "UPDATE instances SET artifact_state='fresh' WHERE id=?", (rows[spec.key]["id"],)
            )
        db.execute(
            "UPDATE instances SET artifact_state='stale' WHERE id=?", (rows[specs[-1].key]["id"],)
        )
    registry.register_worker("clean-worker", resource_class="large")
    claimed = registry.claim_ready_instance("clean-worker", ("large",))
    assert claimed is not None and claimed.instance_key == specs[-1].key
    cancelled = registry.cancel_attempts_downstream_of_failure(int(rows[specs[0].key]["id"]))
    assert [row["instance_key"] for row in cancelled] == [specs[-1].key]
    assert registry.attempt_cancel_requested(claimed.attempt_id)
