"""Global capacity and recovery include isolated development ingestion records."""

from nro.bidsify.config import load_config
from nro.bidsify.index import IngestionIndex
from nro.bidsify.store import IngestionStore
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.registry import Registry


def records(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "BIDS")
    catalog = BranchStore(registry.paths.control)
    catalog.initialize()
    paths = BranchPaths("dev", registry.paths.bids_root, tmp_path / "WORK", tmp_path / "DEV")
    main = IngestionStore(registry)
    debug = IngestionStore(registry, branch_paths=paths)
    config = load_config()
    config.update(staging=str(tmp_path / "staging"), concurrency=1)
    first = main.create(server="cni", remote_session="example", project="demo", config=config)
    second = debug.create(server="cni", remote_session="example", project="demo", config=config)
    return registry, main, debug, first, second


def test_debug_stages_consume_global_capacity(tmp_path):
    registry, main, debug, first, second = records(tmp_path)
    second.update(state="running", worker="debug-worker")
    with registry.connection(write=True):
        debug.write_locked(second)
    registry.register_worker("main-worker", resource_class="large", memory_gb=32, slurm_job_id=None)
    assert main.claim("main-worker", 32) is None
    with registry.connection():
        active, ready, limit = IngestionIndex(registry).summary(32)
    assert (active, ready, limit) == (1, 1, 1)
    assert main.get(first["id"])["state"] == "queued"
    with registry.connection(write=True):
        assert IngestionIndex(registry).recover_locked({"debug-worker"}) == 1
    assert debug.get(second["id"])["state"] == "interrupted"
    assert main.claim("main-worker", 32)["id"] == first["id"]


def test_settings_update_all_namespaces_without_merging_sessions(tmp_path):
    registry, main, debug, first, second = records(tmp_path)
    assert registry.set_active_concurrency(3) == 2
    assert main.get(first["id"])["config"]["concurrency"] == 3
    assert debug.get(second["id"])["config"]["concurrency"] == 3
    assert len(main.rows()) == len(debug.rows()) == 1
    assert first["id"] != second["id"]


def test_debug_queues_do_not_request_workers_before_admission_is_enabled(tmp_path):
    registry, main, debug, first, second = records(tmp_path)
    first["state"] = "cancelled"
    with registry.connection(write=True):
        main.write_locked(first)
        assert IngestionIndex(registry).summary(32) == (0, 0, 0)
    assert debug.get(second["id"])["state"] == "queued"


def test_admitted_debug_queue_uses_shared_worker_capacity(tmp_path):
    registry, main, debug, first, second = records(tmp_path)
    first["state"] = "cancelled"
    debug.execution = {"test_pin": True}
    with registry.connection(write=True):
        main.write_locked(first)
    debug.admit_pending(second["id"])
    registry.register_worker(
        "debug-worker", resource_class="large", memory_gb=32, slurm_job_id=None
    )
    with registry.connection():
        assert IngestionIndex(registry).summary(32) == (0, 1, 1)
    claimed = IngestionIndex(registry).claim("debug-worker", 32)
    assert claimed["branch"] == "dev"
    assert claimed["id"] == second["id"]
    assert main.get(first["id"])["state"] == "cancelled"


def test_retired_queues_release_cache_but_running_stages_remain_pinned(tmp_path, monkeypatch):
    from nro.orchestration import branches

    registry, main, debug, first, second = records(tmp_path)
    catalog = BranchStore(registry.paths.control)
    checkout = tmp_path / "retiring-checkout"
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (checkout, "retiring", "a" * 40))
    catalog.register("retiring", "dev", revision=catalog.read().revision, checkout=checkout)
    paths = BranchPaths("retiring", registry.paths.bids_root, tmp_path / "WORK", tmp_path / "DEV")
    store = IngestionStore(registry, branch_paths=paths, execution={"pin": True})
    queued = store.create(
        server="cni", remote_session="retiring", project="demo", config=load_config()
    )
    catalog.retire("retiring", revision=catalog.read().revision, checkout=checkout)
    assert queued["id"] not in {row["id"] for row in IngestionIndex(registry).execution_records()}
    queued.update(state="running", worker="retired-worker")
    with registry.connection(write=True):
        store.write_locked(queued)
    assert queued["id"] in {row["id"] for row in IngestionIndex(registry).execution_records()}
