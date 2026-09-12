"""Cache collection preserves executable work and never follows artifact selectors."""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nro.bin.purge import main as purge_main
from nro.configuration.store import ConfigStore
from nro.orchestration import execution_cache
from nro.orchestration.execution_cache import cache_lock, collect_cache
from nro.orchestration.registry import Registry


@pytest.fixture
def cache(tmp_path):
    registry = Registry.for_project("demo", bids_root=tmp_path / "BIDS")
    registry.initialize()
    source = registry.paths.control / "shared/cache/implementations" / ("a" * 64)
    source.mkdir(parents=True)
    (source / "source.json").write_text("{}")
    site = registry.paths.control / "shared/cache/execution-sites" / ("b" * 64 + ".toml")
    site.parent.mkdir()
    site.write_text("")
    return registry, source, site


def demand(registry):
    registered = registry.register_workflow(ConfigStore().resolve("main"))
    return registry.create_request(
        registered=registered,
        target_module="anat",
        selectors={},
        instances=(),
        terminal_instance_keys=(),
        concurrency=1,
        partition=None,
    )


def test_idle_cleanup_is_limited_to_known_cache_entries(cache):
    from nro.orchestration.branch_store import BranchStore

    registry, source, site = cache
    catalog = BranchStore(registry.paths.control)
    catalog.initialize()
    registration = catalog.path.read_bytes()
    untouched = [
        registry.paths.control / "shared/notes.txt",
        registry.paths.events / "test.log",
        source.parent / "unknown/file",
    ]
    for path in untouched:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retain")
    link = source.parent / ("c" * 64)
    link.symlink_to(untouched[0].parent, target_is_directory=True)
    preview = collect_cache(registry, dry_run=True)
    assert set(preview.paths) == {source, site}
    assert source.exists() and site.exists()
    actual = collect_cache(registry)
    assert actual == preview
    assert not source.exists() and not site.exists()
    assert all(path.read_text() == "retain" for path in untouched)
    assert catalog.path.read_bytes() == registration
    assert link.is_symlink()


@pytest.mark.parametrize("state", ["idle", "running", "draining", "shutdown_requested"])
def test_workers_protect_cache_even_with_expired_leases(cache, state):
    registry, source, site = cache
    registry.register_worker("worker", resource_class="large", lease_seconds=-10)
    registry.heartbeat_worker("worker", state=state, lease_seconds=-10)
    result = collect_cache(registry)
    assert result.reason == "active workers"
    assert not result.paths and set(result.retained) == {source, site}


def test_demand_protects_queued_work_and_retries(cache):
    registry, source, site = cache
    demand(registry)
    assert collect_cache(registry).reason == "outstanding demand"
    assert source.exists() and site.exists()


@pytest.mark.parametrize("state", ["prepared", "submitted", "running", "cancel_requested"])
def test_submissions_protect_workers_before_registration(cache, state):
    registry, source, site = cache
    with registry.connection(write=True) as db:
        db.execute(
            "INSERT INTO scheduler_submissions(intent_token,resource_class,state,created_at) "
            "VALUES (?,?,?,?)",
            ("token", "large", state, "2026-09-08"),
        )
    assert collect_cache(registry).reason == "worker submissions"
    assert source.exists() and site.exists()


@pytest.mark.parametrize("state", ["queued", "running"])
def test_ingestion_protects_cache_without_derivative_demand(cache, state):
    registry, source, site = cache
    path = registry.paths.control / "shared/ingestion/session.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"state": state, "branch": "main"}))
    assert collect_cache(registry).reason == "queued or running ingestion"
    assert source.exists() and site.exists()


def test_running_cleanup_keeps_its_own_source_and_site(cache, monkeypatch):
    registry, source, site = cache
    monkeypatch.setattr(
        execution_cache, "__file__", str(source / "nro/orchestration/execution_cache.py")
    )
    monkeypatch.setenv("NRO_SITE_CONFIG", str(site))
    assert set(collect_cache(registry).retained) == {source, site}
    assert source.exists() and site.exists()


def test_worker_exit_automatically_collects_unused_cache(cache, monkeypatch):
    import nro.orchestration.worker as worker

    registry, source, site = cache
    monkeypatch.setattr(worker.signal, "signal", lambda *_args: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert worker.Worker(registry, resource_class="large", idle_timeout=0).run() == 0
    assert not source.exists() and not site.exists()


def test_submission_rejects_cache_removed_after_script_preparation(cache, monkeypatch):
    from nro.orchestration import submission

    registry, _, _ = cache
    script = submission._write_worker_script(
        registry,
        bids_root=registry.paths.bids_root,
        partition="test",
        account=None,
        hours=1,
        memory_gb=4,
        cpus=1,
    )
    collect_cache(registry)
    monkeypatch.setattr(
        submission.subprocess, "run", lambda *_a, **_kw: pytest.fail("No sbatch expected")
    )
    with pytest.raises(FileNotFoundError):
        submission._submit_workers(registry, None, script, 4)


def test_collection_waits_for_reference_publication(cache):
    registry, source, site = cache
    started = threading.Event()

    def collect():
        started.set()
        return collect_cache(registry)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with cache_lock(registry.paths.control):
            pending = executor.submit(collect)
            assert started.wait(5)
            demand(registry)
        assert pending.result(timeout=5).reason == "outstanding demand"
    assert source.exists() and site.exists()


def test_cache_purge_ignores_all_artifact_selectors(cache, capsys):
    registry, source, site = cache
    args = [
        "--cache",
        "-P",
        "not-a-project",
        "-p",
        "not-a-participant",
        "-m",
        "anat",
        "-w",
        "not-a-workflow",
        "-f",
        "--json",
    ]
    purge_main(args + ["--dry-run"])
    assert len(json.loads(capsys.readouterr().out)["paths"]) == 2
    assert source.exists() and site.exists()
    purge_main(args)
    assert json.loads(capsys.readouterr().out)["mode"] == "cache"
    assert not source.exists() and not site.exists()


def test_confirmed_purge_rechecks_scheduler_state(cache, monkeypatch, capsys):
    from nro.bin import purge

    registry, source, site = cache
    monkeypatch.setattr(purge, "page_text", lambda _: None)

    def confirm():
        demand(registry)
        return True

    monkeypatch.setattr(purge, "_confirm", confirm)
    purge_main(
        [
            "--cache",
        ]
    )
    assert "outstanding demand" in capsys.readouterr().out
    assert source.exists() and site.exists()


def test_preview_does_not_authorize_later_cache_entries(cache):
    registry, source, site = cache
    preview = collect_cache(registry, dry_run=True)
    later = site.parent / ("c" * 64 + ".toml")
    later.write_text("")
    result = collect_cache(registry, approved=preview.paths)
    assert set(result.paths) == {source, site}
    assert result.retained == (later,)


def test_unavailable_registry_and_symlink_roots_fail_closed(cache):
    registry, source, site = cache
    registry.paths.database.rename(registry.paths.database.with_suffix(".saved"))
    assert collect_cache(registry).reason == "registry is unavailable"
    assert source.exists() and site.exists()
    old = site.parent.with_name("moved-sites")
    site.parent.rename(old)
    site.parent.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        collect_cache(registry)


def test_automatic_cleanup_failure_is_nonfatal(cache, monkeypatch, capsys):
    registry, _, _ = cache

    def fail(*_args):
        raise RuntimeError("Test cleanup failure")

    monkeypatch.setattr(execution_cache, "collect_cache", fail)
    execution_cache.cleanup_cache(registry)
    assert "cleanup deferred" in capsys.readouterr().err


def test_service_lease_blocks_collection_and_releases_after_exit(cache):
    registry, source, site = cache
    with execution_cache.service_lease(registry.paths.control):
        assert collect_cache(registry).reason == "active scheduler service calls"
        assert source.exists() and site.exists()
    assert set(collect_cache(registry).paths) == {source, site}


def test_service_child_retains_lease_after_client_releases_it(cache):
    registry, source, site = cache
    with execution_cache.service_lease(registry.paths.control) as lease:
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            pass_fds=(lease[1],),
        )
    try:
        assert collect_cache(registry).reason == "active scheduler service calls"
    finally:
        process.communicate(timeout=10)
    assert set(collect_cache(registry).paths) == {source, site}
