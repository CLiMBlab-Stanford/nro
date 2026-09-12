"""Branch recipes share real scheduler claims, execution, and cancellation."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from nro.configuration.store import ConfigStore
from nro.orchestration.artifact_resolution import ArtifactCandidate
from nro.orchestration.branch_admission import admit_plan
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.registry import Registry
from nro.orchestration.source_snapshots import SourceStore
from nro.orchestration.worker import Worker

pytestmark = pytest.mark.integration


def source_tree(tmp_path, name):
    checkout = tmp_path / name
    package = checkout / "nro"
    support = package / "orchestration"
    support.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (support / "__init__.py").write_text("")
    actual = Path(__file__).resolve().parents[1] / "nro/orchestration"
    for filename in (
        "source_launcher.py",
        "attempt_entry.py",
        "execution_context.py",
        "branches.py",
    ):
        shutil.copyfile(actual / filename, support / filename)
    (support / "runtime.py").write_text(
        'CONFIGURATION_FINGERPRINT_ENV="NRO_CONFIGURATION_FINGERPRINT"\n'
    )
    configuration = package / "configuration"
    configuration.mkdir()
    (configuration / "__init__.py").write_text("")
    (configuration / "site.py").write_text("ENVIRONMENT_KEYS={}\n")
    module = package / ("probe_" + name)
    module.mkdir()
    (module / "__init__.py").write_text("")
    (module / "__main__.py").write_text(
        "from pathlib import Path\n"
        "def main(argv, *, execution_context):\n"
        "    out = execution_context.output_path(Path(argv[0]))\n"
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        f"    out.write_text({name!r})\n"
    )
    (checkout / "pyproject.toml").write_text('[project]\nname="nro"\nversion="0.0.dev0"\n')
    return checkout


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import nro.orchestration.branches as branches_module

    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    branches = BranchStore(registry.paths.control)
    branches.initialize()
    identities = {}
    monkeypatch.setattr(
        branches_module,
        "checkout_identity",
        lambda path: (Path(path), identities[Path(path)], "a" * 40),
    )
    site = tmp_path / "site.toml"
    site.write_text("")

    def prepare(name, *, parent="dev", demand=True):
        root = source_tree(tmp_path, name)
        for module in (tmp_path / parent / "nro").glob("probe_*"):
            shutil.copytree(module, root / "nro" / module.name)
        identities[root] = name
        branches.register(name, parent, revision=branches.read().revision, checkout=root)
        science = branches.registry(name)
        registered = science.register_workflow(ConfigStore().resolve("main"))
        paths = BranchPaths(name, registry.paths.bids_root, tmp_path / "WORK", tmp_path / "DEV")
        out = (
            paths.source_project("demo") / "derivatives/preprocessing/main/sub-01/sub-01_result.txt"
        )
        spec = InstanceSpec.create(
            key="same-logical-key",
            module="probe_" + name,
            project="demo",
            participant="01",
            entities={},
            scope="subject",
            configuration_lineage_id=registered.lineages["preprocessing"],
            config_fingerprint="test-science",
            directory_label="main",
            runtime_config=science.runtime_config_path(registered, "preprocessing"),
            command=(sys.executable, "-m", "nro.probe_" + name, str(out)),
            dependencies=(),
            input_paths=(),
            output_root=out.parent,
            output_prefix="sub-01",
            expected_outputs=(out,),
            resource_class="small",
            output_format="probe-v1",
        )
        plan = branches.resolve_plan(root, paths, (spec,), (spec.key,), (), validate=lambda _: True)
        source = SourceStore(tmp_path / "cache").capture(root)
        request = (
            admit_plan(
                registry,
                branches,
                root,
                plan,
                registered,
                source=source,
                site=site,
                python=Path(sys.executable),
                concurrency=1,
                partition=None,
                selectors={},
            )
            if demand
            else None
        )
        return root, paths, spec, plan, registered, source, request

    return registry, branches, site, prepare


def test_two_catalogs_share_capacity_and_complete_through_worker(setup):
    registry, branches, site, prepare = setup
    one = prepare("one")
    two = prepare("two")
    workers = [Worker(registry, resource_class="large", poll_interval=0.01) for _ in range(2)]
    for worker in workers:
        registry.register_worker(worker.worker_id, resource_class="large")
    first = registry.claim_ready_instance(workers[0].worker_id, ("small",))
    assert first is not None
    assert registry.claim_ready_instance(workers[1].worker_id, ("small",)) is None
    workers[0]._execute(first)
    with registry.connection() as db:
        attempt = dict(
            db.execute("SELECT * FROM attempts WHERE id=?", (first.attempt_id,)).fetchone()
        )
    assert attempt["state"] == "success", attempt["error_message"]
    second = registry.claim_ready_instance(workers[0].worker_id, ("small",))
    assert second is not None and second.instance_id != first.instance_id
    workers[0]._execute(second)
    with registry.connection() as db:
        assert all(row["state"] == "success" for row in db.execute("SELECT state FROM attempts"))
    for name, prepared in (("one", one), ("two", two)):
        _, paths, spec, *_ = prepared
        output = (
            paths.output_project("demo") / "derivatives/preprocessing/main/sub-01/sub-01_result.txt"
        )
        assert output.read_text() == name
        assert not spec.expected_outputs[0].exists()
        row = next(row for row in registry.instance_rows() if row["module"] == "probe_" + name)
        completion = json.loads(Path(row["manifest_path"]).read_text())
        assert completion["implementation"]["branch"] == name
        assert completion["implementation"]["source_digest"] == prepared[5].digest
        assert "nro.orchestration.attempt_entry" in completion["software"]["command"]
    assert first.log_path != second.log_path
    assert {row["instance_key"] for row in registry.instance_rows()} == {
        branches.registry(name).record.registry_id + ":same-logical-key" for name in ("one", "two")
    }
    for prepared in (one, two):
        shutil.rmtree(prepared[5].root)
    from nro.orchestration.manifests import assess_registry

    assert all(state == "fresh" for state, _ in assess_registry(registry).values())
    assert all(state == "fresh" for state, _ in assess_registry(registry, compiled=True).values())
    assert "nro.probe_one" not in sys.modules and "nro.probe_two" not in sys.modules


@pytest.mark.parametrize("conflict", (False, True))
def test_promotion_checks_current_target_contract_and_retains_producer(
    setup, conflict, monkeypatch
):
    from nro.orchestration.manifests import assess_registry
    from nro.orchestration.promotion import preview, publish

    registry, branches, site, prepare = setup
    parent = prepare("parent")
    child = prepare("child", parent="parent", demand=False)
    root, paths, _, _, registered, source, _ = child
    spec = parent[2].evolve(
        configuration_lineage_id=registered.lineages["preprocessing"],
        runtime_config=branches.registry("child").runtime_config_path(registered, "preprocessing"),
    )
    plan = branches.resolve_plan(root, paths, (spec,), (spec.key,), (), validate=lambda _: True)
    admit_plan(
        registry,
        branches,
        root,
        plan,
        registered,
        source=source,
        site=site,
        python=Path(sys.executable),
        concurrency=1,
        partition=None,
        selectors={},
    )
    with registry.connection(write=True) as db:
        db.execute("UPDATE requests SET state='registered' WHERE id=?", (parent[-1],))
    worker = Worker(registry, resource_class="large", poll_interval=0.01)
    registry.register_worker(worker.worker_id, resource_class="large")
    claimed = registry.claim_ready_instance(worker.worker_id, ("small",))
    worker._execute(claimed)
    output = (
        parent[1].output_project("demo") / "derivatives/preprocessing/main/sub-01/sub-01_result.txt"
    )
    if conflict:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("incompatible target")
        with registry.connection(write=True) as db:
            target_id = db.execute(
                "SELECT instance_id FROM request_artifacts WHERE request_id=?", (parent[-1],)
            ).fetchone()[0]
            manifest = db.execute(
                "SELECT manifest_path FROM instances WHERE id=?", (target_id,)
            ).fetchone()[0]
        Path(manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest).write_text(json.dumps({"manifest_version": -1}))
    report = preview(
        registry,
        checkout=parent[0],
        source="child",
        requests=[parent[-1]],
        pr="test#1",
        attest=True,
    )
    assert report["items"][0]["action"] == ("replace" if conflict else "copy")
    if conflict:
        with pytest.raises(ValueError, match="--replace"):
            publish(registry, checkout=parent[0], report=report, replace=False, attest=True)
        assert output.read_text() == "incompatible target"
        import nro.orchestration.promotion as promotion

        replace = promotion.os.replace
        failed = False

        def interrupt(source, destination):
            nonlocal failed
            if not failed and ".nro-promotion-" in str(source) and ".rollback" not in str(source):
                failed = True
                raise OSError("interrupted publication")
            return replace(source, destination)

        monkeypatch.setattr(promotion.os, "replace", interrupt)
        with pytest.raises(OSError, match="interrupted publication"):
            publish(registry, checkout=parent[0], report=report, replace=True, attest=True)
        assert output.read_text() == "incompatible target"
        monkeypatch.setattr(promotion.os, "replace", replace)
    else:
        import nro.orchestration.ownership as ownership

        write = ownership.write_instance_ownership
        failed = False

        def interrupt_ownership(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("interrupted ownership publication")
            return write(*args, **kwargs)

        monkeypatch.setattr(ownership, "write_instance_ownership", interrupt_ownership)
        with pytest.raises(OSError, match="interrupted ownership publication"):
            publish(registry, checkout=parent[0], report=report, replace=False, attest=True)
        monkeypatch.setattr(ownership, "write_instance_ownership", write)
    result = publish(registry, checkout=parent[0], report=report, replace=conflict, attest=True)
    assert result["promoted"] == 1
    assert (
        paths.output_project("demo") / "derivatives/preprocessing/main/sub-01/sub-01_result.txt"
    ).read_text() == "parent"
    output = (
        parent[1].output_project("demo") / "derivatives/preprocessing/main/sub-01/sub-01_result.txt"
    )
    assert output.read_text() == "parent"
    assess_registry(registry, compiled=True)
    report = preview(
        registry,
        checkout=parent[0],
        source="child",
        requests=[parent[-1]],
        pr="test#1",
        attest=True,
    )
    assert report["items"][0]["action"] == "keep"
    assert (
        publish(registry, checkout=parent[0], report=report, replace=False, attest=True)["promoted"]
        == 0
    )
    row = next(row for row in registry.instance_rows() if row["output_root"] == str(output.parent))
    provenance = json.loads(Path(row["manifest_path"]).read_text())["implementation"]
    assert provenance["branch"] == "child"
    assert provenance["promotion"]["to_branch"] == "parent"
    with pytest.raises(ValueError, match="attestation"):
        preview(
            registry,
            checkout=parent[0],
            source="child",
            requests=[parent[-1]],
            pr="test#1",
            attest=False,
        )


def test_promotion_rollback_recovers_renames_at_journal_boundaries(tmp_path):
    from nro.orchestration.promotion import _rollback_publication

    introduced = tmp_path / "introduced"
    introduced.write_text("new")
    moved_source = tmp_path / "stage" / "introduced"
    _rollback_publication(
        {
            "publication": [
                {
                    "destination": str(introduced),
                    "backup": str(tmp_path / "unused"),
                    "source": str(moved_source),
                    "had_destination": False,
                    "state": "prepared",
                }
            ]
        }
    )
    assert not introduced.exists()

    destination = tmp_path / "replaced"
    destination.write_text("new")
    backup = tmp_path / "stage" / ".rollback" / "replaced"
    backup.parent.mkdir(parents=True)
    backup.write_text("old")
    _rollback_publication(
        {
            "publication": [
                {
                    "destination": str(destination),
                    "backup": str(backup),
                    "source": str(tmp_path / "stage" / "replaced"),
                    "had_destination": True,
                    "state": "prepared",
                }
            ]
        }
    )
    assert destination.read_text() == "old"
    assert not backup.exists()


def test_retirement_immediately_cancels_owned_attempts(setup):
    from nro.orchestration.branch_operations import update

    registry, branches, _, prepare = setup
    prepared = prepare("retiring")
    checkout, request = prepared[0], prepared[-1]
    worker = Worker(registry, resource_class="large", poll_interval=0.01)
    registry.register_worker(worker.worker_id, resource_class="large")
    claim = registry.claim_ready_instance(worker.worker_id, ("small",))
    assert claim is not None
    result = update(
        registry,
        checkout=checkout,
        branch="retiring",
        action="retire",
        revision=branches.read().revision,
    )
    assert result["retired"]
    assert result["cancelled_attempts"] == 1
    assert registry.attempt_cancel_requested(claim.attempt_id)
    with registry.connection() as db:
        assert (
            db.execute("SELECT state FROM requests WHERE id=?", (request,)).fetchone()[0]
            == "cancelled"
        )


def test_inherited_read_has_no_parent_demand_and_cancels_on_parent_change(setup):
    from nro.orchestration import dependency_state
    from nro.orchestration.manifests import record_completion

    registry, branches, site, prepare = setup
    _, _, producer, producer_plan, *_ = prepare("parent")
    worker = Worker(registry, resource_class="large", poll_interval=0.01)
    registry.register_worker(worker.worker_id, resource_class="large")
    built = registry.claim_ready_instance(worker.worker_id, ("small",))
    worker._execute(built)
    registry.reconcile_requests()
    original = next(row for row in registry.instance_rows() if row["id"] == built.instance_id)
    checkout, paths, base, _, registered, source, _ = prepare(
        "child", parent="parent", demand=False
    )
    output = base.output_root / "sub-01_child.txt"
    consumer = base.evolve(
        key="consumer",
        dependencies=(producer.key,),
        input_paths=producer.expected_outputs,
        expected_outputs=(output,),
        command=(sys.executable, "-m", "nro.probe_child", str(output)),
    )
    candidate = ArtifactCandidate(
        "parent",
        producer.key,
        producer_plan.instances[0].contract,
        original["current_generation"],
        Path(original["output_root"]),
        {},
    )
    requested_parent = producer.evolve(
        runtime_config=base.runtime_config, configuration_lineage_id=base.configuration_lineage_id
    )
    plan = branches.resolve_plan(
        checkout,
        paths,
        (requested_parent, consumer),
        (consumer.key,),
        (candidate,),
        validate=lambda _: True,
    )
    assert tuple(item.spec.key for item in plan.work) == ("consumer",)
    request = admit_plan(
        registry,
        branches,
        checkout,
        plan,
        registered,
        source=source,
        site=site,
        python=Path(sys.executable),
        concurrency=1,
        partition=None,
        selectors={},
    )
    with registry.connection() as db:
        assert [
            row[0]
            for row in db.execute(
                "SELECT instance_id FROM request_instances WHERE request_id=?", (request,)
            )
        ] != [built.instance_id]
        assert not db.execute(
            "SELECT 1 FROM request_instances WHERE request_id=? AND instance_id=?",
            (request, built.instance_id),
        ).fetchone()
    assert (
        next(row for row in registry.instance_rows() if row["id"] == built.instance_id) == original
    )
    claim = registry.claim_ready_instance(worker.worker_id, ("small",))
    assert claim is not None and claim.instance_id != built.instance_id
    with registry.connection() as db:
        assert (
            db.execute(
                "SELECT generation FROM attempt_dependencies WHERE attempt_id=? AND upstream_instance_id=?",
                (claim.attempt_id, built.instance_id),
            ).fetchone()[0]
            == original["current_generation"]
        )
        context = json.loads(
            db.execute(
                "SELECT context_json FROM attempt_execution WHERE attempt_id=?", (claim.attempt_id,)
            ).fetchone()[0]
        )
        assert context["inputs"][0]["generation"] == original["current_generation"]
    with registry.connection(write=True) as db:
        db.execute("UPDATE instances SET artifact_state='stale' WHERE id=?", (built.instance_id,))
    registry.cancel_attempts_with_stale_upstreams()
    assert registry.attempt_cancel_requested(claim.attempt_id)
    with pytest.raises(dependency_state.AttemptInvalidated):
        record_completion(
            registry,
            instance_id=claim.instance_id,
            attempt_id=claim.attempt_id,
            outputs=(
                paths.output_project("demo") / output.relative_to(paths.source_project("demo")),
            ),
        )
    from nro.orchestration.branch_reconciliation import reconcile_branch_requests

    assert reconcile_branch_requests(registry) == 1
    registry.finish_attempt(claim.attempt_id, state="cancelled")
    local_parent = registry.claim_ready_instance(worker.worker_id, ("small",))
    assert local_parent is not None and local_parent.instance_id != built.instance_id
    assert str(paths.output_project("demo")) in str(local_parent.output_root)
    worker._execute(local_parent)
    local_consumer = registry.claim_ready_instance(worker.worker_id, ("small",))
    assert local_consumer.instance_id == claim.instance_id
    worker._execute(local_consumer)
    with registry.connection() as db:
        assert (
            db.execute(
                "SELECT state FROM attempts WHERE id=?", (local_consumer.attempt_id,)
            ).fetchone()[0]
            == "success"
        )
        assert (
            db.execute(
                "SELECT COUNT(*) FROM attempts WHERE instance_id=?", (built.instance_id,)
            ).fetchone()[0]
            == 1
        )
    assert reconcile_branch_requests(registry) == 0


def test_equivalent_demand_updates_next_recipe_without_rewriting_attempt(setup, tmp_path):
    registry, branches, site, prepare = setup
    checkout, _, _, plan, registered, source, _ = prepare("one")
    worker = Worker(registry, resource_class="large", poll_interval=0.01)
    registry.register_worker(worker.worker_id, resource_class="large")
    attempt = registry.claim_ready_instance(worker.worker_id, ("small",))
    assert attempt is not None
    with registry.connection() as db:
        previous = dict(db.execute("SELECT * FROM instance_execution").fetchone())
        command = db.execute("SELECT command_json FROM instances").fetchone()[0]
        captured_attempt = dict(
            db.execute(
                "SELECT * FROM attempt_execution WHERE attempt_id=?", (attempt.attempt_id,)
            ).fetchone()
        )
    (checkout / "nro/__init__.py").write_text("RUNTIME_CHANGE = True\n")
    changed = SourceStore(tmp_path / "cache").capture(checkout)
    assert changed.digest != source.digest
    admit_plan(
        registry,
        branches,
        checkout,
        plan,
        registered,
        source=changed,
        site=site,
        python=Path(sys.executable),
        concurrency=1,
        partition=None,
        selectors={},
    )
    with registry.connection() as db:
        current = dict(db.execute("SELECT * FROM instance_execution").fetchone())
        current_command = db.execute("SELECT command_json FROM instances").fetchone()[0]
        assert current != previous
        assert json.loads(current["provenance_json"])["source_digest"] == changed.digest
        assert current_command != command
        assert json.loads(current_command)[2] == changed.digest
        assert (
            dict(
                db.execute(
                    "SELECT * FROM attempt_execution WHERE attempt_id=?", (attempt.attempt_id,)
                ).fetchone()
            )
            == captured_attempt
        )


def test_detached_service_rejects_late_science_without_opening_branch_code(setup, monkeypatch):
    from nro.orchestration import branch_planning, branch_reconciliation, scheduler_service
    from nro.orchestration.branch_registry import BranchRegistry
    from nro.orchestration.scheduler_service import admit

    registry, branches, site, prepare = setup
    checkout, paths, spec, plan, registered, source, request_id = prepare("one")
    with registry.connection() as db:
        payload = json.loads(
            db.execute(
                "SELECT payload_json FROM request_plans WHERE request_id=?", (request_id,)
            ).fetchone()[0]
        )
    payload["revisions"] = {spec.key: 2}
    values = {key: str(getattr(paths, key)) for key in ("bids", "work", "development")}
    monkeypatch.setattr(
        BranchRegistry,
        "instances",
        lambda *args: pytest.fail("Central service read branch science"),
    )
    monkeypatch.setattr(
        scheduler_service,
        "scientific_contracts",
        lambda *_args: pytest.fail("Central service recompiled submitted contracts"),
    )
    monkeypatch.setattr(
        branch_planning,
        "scientific_contracts",
        lambda *_args: pytest.fail("Central plan recompiled submitted contracts"),
    )
    monkeypatch.setattr(
        branch_reconciliation,
        "scientific_contracts",
        lambda *_args: pytest.fail("Central candidate lookup recompiled stored contracts"),
    )
    request_id = admit(registry, payload, checkout=checkout, site_values=values)
    assert request_id
    payload["revisions"][spec.key] = 1
    with pytest.raises(ValueError, match="newer scientific request"):
        admit(registry, payload, checkout=checkout, site_values=values)
    payload["revisions"][spec.key] = 2
    payload["contracts"][spec.key]["processing"] = {"changed": True}
    with pytest.raises(ValueError, match="newer scientific request"):
        admit(registry, payload, checkout=checkout, site_values=values)
    with registry.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM request_plans").fetchone()[0] == 2


def test_central_status_and_stop_are_branch_scoped(setup):
    from nro.orchestration.scheduler_service import status, stop

    registry, branches, site, prepare = setup
    one = prepare("one")
    two = prepare("two")
    first = status(registry, checkout=one[0], mode="cached")
    second = status(registry, checkout=two[0], mode="preview")
    assert len(first["visible_ids"]) == len(second["visible_ids"]) == 1
    assert set(first["visible_ids"]).isdisjoint(second["visible_ids"])
    result = stop(registry, checkout=one[0], selection={"force": True})
    assert result["requests"] == 1
    with registry.connection() as db:
        assert (
            db.execute("SELECT state FROM requests WHERE id=?", (one[-1],)).fetchone()[0]
            == "cancelled"
        )
        assert (
            db.execute("SELECT state FROM requests WHERE id=?", (two[-1],)).fetchone()[0]
            == "active"
        )
