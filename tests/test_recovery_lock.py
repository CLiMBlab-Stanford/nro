"""Test owner-identified recovery locks and their scheduler and registry callers."""

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from nro.engine.io import atomic_write_json
from nro.orchestration import recovery_lock, scheduler_bus, scheduler_client
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.recovery_lock import RecoveryLock
from nro.orchestration.registry import RegistryLock


def write_owner(path, **fields):
    path.mkdir()
    record = {
        "protocol": recovery_lock.PROTOCOL,
        "token": "other",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_start": recovery_lock.process_start(os.getpid()),
        "created_at": time.time(),
    }
    atomic_write_json(path / "owner.json", {**record, **fields})


def age(path, seconds):
    then = time.time() - seconds
    os.utime(path, (then, then))


def dead_pid():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_lock_excludes_and_releases(tmp_path):
    path = tmp_path / "launch.recovery"
    first, second = RecoveryLock(path), RecoveryLock(path)

    assert first.acquire()
    assert json.loads((path / "owner.json").read_text())["token"] == first.token
    assert not second.acquire()
    first.release()
    assert not path.exists()
    assert second.acquire()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["launch.recovery"]


def test_simultaneous_contenders_elect_one_owner(tmp_path):
    path = tmp_path / "launch.recovery"
    locks = [RecoveryLock(path) for _index in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda lock: lock.acquire(), locks))

    assert results.count(True) == 1
    assert [lock.owned() for lock in locks] == results


def test_lock_held_by_live_local_process_is_kept(tmp_path):
    path = tmp_path / "launch.recovery"
    write_owner(path, created_at=0.0)
    age(path, 3600)

    assert not RecoveryLock(path).acquire()


def test_lock_from_dead_local_process_is_replaced(tmp_path):
    path = tmp_path / "launch.recovery"
    write_owner(path, pid=dead_pid())

    lock = RecoveryLock(path)
    assert lock.acquire()
    assert lock.owned()


def test_lock_from_reused_pid_is_replaced(tmp_path):
    path = tmp_path / "launch.recovery"
    write_owner(path, process_start="another-boot:1")

    assert RecoveryLock(path).acquire()


def test_remote_lock_is_replaced_only_after_its_lease(tmp_path):
    path = tmp_path / "launch.recovery"
    write_owner(path, host="elsewhere", created_at=time.time() - 10)

    assert not RecoveryLock(path, lease_seconds=60).acquire()
    assert RecoveryLock(path, lease_seconds=5).acquire()


def test_legacy_empty_lock_is_replaced_after_grace(tmp_path):
    path = tmp_path / "launch.recovery"
    path.mkdir()

    assert not RecoveryLock(path).acquire()
    age(path, recovery_lock.UNOWNED_GRACE_SECONDS + 1)
    lock = RecoveryLock(path)
    assert lock.acquire()
    assert lock.owned()


def test_takeover_restores_a_lock_that_replaced_the_stale_one(tmp_path, monkeypatch):
    path = tmp_path / "launch.recovery"
    write_owner(path, token="stale", pid=dead_pid())
    winner = RecoveryLock(path)
    late = RecoveryLock(path)
    read_owner = recovery_lock._read_owner
    reads = []

    def replaced_after_check(target):
        record = read_owner(target)
        if target == path:
            reads.append(record)
            # The late contender's third read judges the stale owner. A faster
            # contender then replaces it before the late one renames it away.
            if len(reads) == 3:
                assert winner.acquire()
        return record

    monkeypatch.setattr(recovery_lock, "_read_owner", replaced_after_check)
    assert not late.acquire()
    assert reads[2]["token"] == "stale"
    assert winner.owned()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["launch.recovery"]


def test_release_leaves_a_replacement_owner_in_place(tmp_path):
    path = tmp_path / "launch.recovery"
    first = RecoveryLock(path)
    assert first.acquire()
    os.rename(path, tmp_path / "taken")
    write_owner(path, token="replacement")

    first.release()
    assert json.loads((path / "owner.json").read_text())["token"] == "replacement"


def test_launch_claim_recovers_ended_job_despite_orphaned_recovery_dir(tmp_path, monkeypatch):
    control = tmp_path / ".nro"
    service = scheduler_bus.prepare(control).service
    launch = ControlPaths(control).service_launch
    launch.mkdir()
    atomic_write_json(
        launch / "owner.json",
        {
            "created_at": "2026-10-22T15:53:38.882468+00:00",
            "host": "login007",
            "job_id": "26594578",
            "pid": 1,
            "protocol": scheduler_bus.PROTOCOL,
            "token": "dead",
            "user": "user",
        },
    )
    legacy = service / "launch.recovery"
    legacy.mkdir()
    age(legacy, 86400)
    monkeypatch.setattr(scheduler_bus, "_slurm_terminal", lambda job_id: job_id == "26594578")

    claim = scheduler_bus.claim_launch(control)

    assert claim is not None
    assert scheduler_bus.read_launch(control)["token"] == claim.token
    assert not legacy.exists()


def test_registry_lock_recovers_despite_orphaned_recovery_dir(tmp_path):
    path = tmp_path / "registry.lock"
    recovery = tmp_path / "registry.lock.recovery"
    path.mkdir()
    (path / "owner.json").write_text(
        json.dumps({"token": "dead", "hostname": socket.gethostname(), "pid": dead_pid()})
    )
    age(path, 3600)
    recovery.mkdir()
    age(recovery, 3600)

    with RegistryLock(path, recovery, timeout=5, stale_after=1):
        assert json.loads((path / "owner.json").read_text())["token"] != "dead"


def test_registry_waiter_skips_recovery_lock_for_a_young_owner(tmp_path, monkeypatch):
    path = tmp_path / "registry.lock"
    path.mkdir()
    lock = RegistryLock(path, tmp_path / "registry.lock.recovery", stale_after=300)
    monkeypatch.setattr(
        recovery_lock.RecoveryLock,
        "acquire",
        lambda _self: pytest.fail("a young owner needs no recovery lock"),
    )

    assert lock._recover_if_safe() is False


def test_exchange_fails_when_ended_allocation_is_not_replaced(monkeypatch, tmp_path):
    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_client, "LAUNCH_CHECK_SECONDS", 0.0)
    monkeypatch.setattr(scheduler_bus, "read_active", lambda _control: None)
    monkeypatch.setattr(
        scheduler_bus, "read_launch", lambda _control: {"token": "dead", "job_id": "26594578"}
    )
    monkeypatch.setattr(scheduler_bus, "_launch_abandoned", lambda _record: True)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="allocation 26594578 ended before the scheduler"):
        scheduler_client.exchange(
            endpoint, {"operation": "status"}, timeout=None, require_service=True
        )


def test_exchange_keeps_waiting_while_allocation_is_pending(monkeypatch, tmp_path):
    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_client, "LAUNCH_CHECK_SECONDS", 0.0)
    monkeypatch.setattr(scheduler_bus, "read_active", lambda _control: None)
    monkeypatch.setattr(
        scheduler_bus, "read_launch", lambda _control: {"token": "live", "job_id": "101"}
    )
    checks = []

    def pending(_record):
        checks.append(None)
        if len(checks) >= 3:
            raise KeyboardInterrupt
        return False

    monkeypatch.setattr(scheduler_bus, "_launch_abandoned", pending)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: True)

    with pytest.raises(KeyboardInterrupt):
        scheduler_client.exchange(
            endpoint, {"operation": "status"}, timeout=None, require_service=True
        )
    assert len(checks) == 3
