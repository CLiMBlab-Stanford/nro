"""Reclaim execution snapshots when the shared scheduler has no work using them."""

from __future__ import annotations

import fcntl
import os
import re
import shlex
import shutil
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from nro.bidsify.index import IngestionIndex
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import Registry, RegistryLock, ensure_shared_directory


@dataclass(frozen=True)
class CacheCollection:
    """Cache paths removed or proposed, retained paths, and any scheduler blocker."""

    paths: tuple[Path, ...] = ()
    retained: tuple[Path, ...] = ()
    reason: str | None = None


def cache_lock(control: Path) -> RegistryLock:
    """Serialize cache collection with publication of execution references.

    Acquire this lock before the registry lock when both are needed.
    """
    paths = ControlPaths(control)
    paths.require_current_layout()
    ensure_shared_directory(paths.cache)
    return RegistryLock(
        paths.cache / "execution-cache.lock", paths.cache / "execution-cache.recovery-lock"
    )


def cache_publication(function):
    """Keep cache files alive until a registry operation publishes their references."""

    @wraps(function)
    def protected(owner, *args, **kwargs):
        registry = getattr(owner, "registry", owner)
        with cache_lock(registry.paths.control):
            return function(owner, *args, **kwargs)

    return protected


@contextmanager
def service_lease(control: Path):
    """Protect short-lived service processes without holding the publication lock.

    The kernel releases the lease after a crash. Collection removes unlocked
    lease files, including those left by an interrupted client on another node.
    """
    with cache_lock(control):
        root = ControlPaths(control).cache / "services"
        ensure_shared_directory(root)
        path = root / (uuid.uuid4().hex + ".lock")
        stream = path.open("xb")
        path.chmod(0o664)
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield path.name, stream.fileno()
    finally:
        stream.close()
        with cache_lock(control):
            try:
                remaining = path.open("r+b")
            except FileNotFoundError:
                remaining = None
            if remaining is not None:
                with remaining:
                    try:
                        fcntl.flock(remaining, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        path.unlink(missing_ok=True)


def _active_service(control: Path, *, ignore: str | None = None) -> bool:
    root = ControlPaths(control).cache / "services"
    if root.is_symlink():
        raise ValueError("Service lease directory cannot be a symlink")
    for path in root.glob("*.lock"):
        if path.name == ignore:
            continue
        if path.is_symlink() or not re.fullmatch(r"[0-9a-f]{32}\.lock", path.name):
            raise ValueError("Invalid service lease path")
        try:
            stream = path.open("r+b")
        except FileNotFoundError:
            continue
        with stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            path.unlink(missing_ok=True)
    return False


def _candidates(control: Path) -> tuple[Path, ...]:
    paths = []
    for name, pattern, directory in (
        ("implementations", r"[0-9a-f]{64}", True),
        ("execution-sites", r"[0-9a-f]{64}\.toml", False),
    ):
        root = ControlPaths(control).cache / name
        if root.is_symlink():
            raise ValueError(f"Execution cache directory cannot be a symlink: {root}")
        if not root.exists():
            continue
        for path in root.iterdir():
            if (
                re.fullmatch(pattern, path.name)
                and not path.is_symlink()
                and (path.is_dir() if directory else path.is_file())
            ):
                paths.append(path)
    return tuple(sorted(paths))


def _busy(registry: Registry, db, *, ignore_service: str | None = None) -> str | None:
    if _active_service(registry.paths.control, ignore=ignore_service):
        return "active scheduler service calls"
    checks = (
        ("SELECT 1 FROM requests WHERE state='active' LIMIT 1", "outstanding demand"),
        (
            "SELECT 1 FROM attempts WHERE state IN ('queued','running','cancel_requested') LIMIT 1",
            "active attempts",
        ),
        (
            "SELECT 1 FROM workers WHERE state IN ('idle','running','draining','shutdown_requested') LIMIT 1",
            "active workers",
        ),
        (
            "SELECT 1 FROM scheduler_submissions WHERE state IN "
            "('prepared','submitted','running','cancel_requested') LIMIT 1",
            "worker submissions",
        ),
        (
            "SELECT 1 FROM metadata WHERE key='maintenance_mode' OR key LIKE 'branch_maintenance:%' LIMIT 1",
            "registry maintenance",
        ),
    )
    for query, reason in checks:
        if db.execute(query).fetchone():
            return reason
    if IngestionIndex(registry).execution_records():
        return "queued or running ingestion"
    return None


def collect_cache(
    registry: Registry,
    *,
    dry_run: bool = False,
    approved: tuple[Path, ...] | None = None,
    service: str | None = None,
) -> CacheCollection:
    """Remove unused source/site snapshots only when the shared pool is quiescent.

    Selectors never limit this operation. Outstanding demand protects retries;
    scheduler and worker records protect pending allocations and replacements.
    Uncertain or unreadable registry state fails closed. Keep the executing
    process's source and site file until a later invocation can reclaim them.
    Do not remove logs, definitions, images, artifacts, or unknown cache entries.
    If approved is supplied, delete only paths from that confirmed preview.
    """
    control = registry.paths.control
    if not any(
        path.exists()
        for path in (ControlPaths(control).implementations, ControlPaths(control).execution_sites)
    ):
        return CacheCollection()
    with cache_lock(control):
        candidates = _candidates(control)
        if not candidates:
            return CacheCollection()
        if not registry.paths.database.is_file():
            return CacheCollection(retained=candidates, reason="registry is unavailable")
        with registry.connection() as db:
            reason = _busy(registry, db, ignore_service=service)
            if reason:
                return CacheCollection(retained=candidates, reason=reason)
            current = {Path(__file__).resolve().parents[2]}
            if os.environ.get("NRO_SITE_CONFIG"):
                current.add(Path(os.environ["NRO_SITE_CONFIG"]).resolve())
            allowed = frozenset(approved) if approved is not None else None
            paths = tuple(
                path
                for path in candidates
                if path not in current and (allowed is None or path in allowed)
            )
            removed = frozenset(paths)
            retained = tuple(path for path in candidates if path not in removed)
            if not dry_run:
                for path in paths:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            return CacheCollection(paths, retained)


def cleanup_cache(registry: Registry) -> None:
    """Attempt automatic collection without turning a cleanup failure into a job failure."""
    try:
        result = collect_cache(registry)
        if result.paths:
            print(f"Removed {len(result.paths)} unused execution cache entries.", file=sys.stderr)
    except Exception as error:
        print(f"WARNING: execution cache cleanup deferred: {error}", file=sys.stderr)


def validate_script_cache(script: Path) -> None:
    """Reject a prepared worker script whose snapshots were removed before submission."""
    from nro.orchestration.source_snapshots import SourceSnapshot

    for line in script.read_text().splitlines():
        if not line.startswith("exec "):
            continue
        command = shlex.split(line[5:])
        if len(command) >= 6 and Path(command[1]).name == "source_launcher.py":
            SourceSnapshot(Path(command[1]).parents[2], command[2]).verify()
            import hashlib

            if hashlib.sha256(Path(command[3]).read_bytes()).hexdigest() != command[4]:
                raise ValueError("Prepared worker site settings changed; submit a new request")
