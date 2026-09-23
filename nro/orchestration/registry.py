"""NFS-conscious, externally serialized SQLite registry.

Every SQLite connection is opened while owning the same atomic directory lock.
The database intentionally uses rollback journaling rather than WAL so clients
on different compute nodes never depend on a shared-memory WAL index.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import random
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence

import yaml

from nro.configuration.paths import BIDS_PATH, REGISTRY_PATH
from nro.configuration.store import (
    CONFIGURATION_CLASSES,
    fingerprint,
)
from nro.engine.cli import matches_work_item_selectors as matches_selectors
from nro.engine.io import atomic_write_text
from nro.orchestration import dependency_state
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.migrations import migrate_database
from nro.orchestration.registry_schema import (
    APPLICATION_ID,
    SCHEMA_VERSION,
    current_schema_sql,
)
from nro.orchestration.registry_schema import (
    SCHEMA as SCHEMA_DEFINITION,
)
from nro.orchestration.registry_work_items import work_item_relative_directory
from nro.orchestration.resources import WORK_ITEM_RESOURCE_CLASSES, compatible_work_item_classes
from nro.orchestration.workflow_registry import RegisteredWorkflow, WorkflowRegistry

if TYPE_CHECKING:
    from nro.orchestration.contracts import ExecutionEnvelope, WorkItemSpec


_WAIT_NOTICE_SECONDS = 0.75
_WAIT_FRAMES = ("·", "•", "●", "•")
_WAIT_COLORS = ("\x1b[95m", "\x1b[94m", "\x1b[96m", "\x1b[92m", "\x1b[93m")
_CLEAR_LINE = "\r\x1b[2K"
_RESET = "\x1b[0m"


def utcnow() -> str:
    """Return the current UTC time in ISO 8601 form."""
    return datetime.now(timezone.utc).isoformat()


def discover_registry_projects(bids_root: str | Path) -> list[str]:
    """Return projects represented in the central registry without changing it."""
    registry = Registry.for_project("", bids_root=bids_root)
    if not registry.paths.database.is_file():
        return []
    with registry.read_connection() as connection:
        rows = connection.execute(
            """SELECT project FROM bids_projects
               UNION SELECT project FROM work_items
               UNION SELECT project FROM requests"""
        ).fetchall()
    return sorted(str(row["project"]) for row in rows if str(row["project"]))


def ensure_shared_directory(path: str | Path) -> Path:
    """Set group/setgid access on this directory and parents created for it.

    Existing ancestors belong to their owners and are not changed. The control
    directory need not have a particular name or reside under a .nro directory.
    """
    directory = Path(path)
    if directory.resolve().parent == directory.resolve():
        raise ValueError("The filesystem root cannot be a private-control directory")
    candidates = [directory]
    current = directory.parent
    while not current.exists():
        candidates.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o2775)
    for candidate in candidates:
        try:
            candidate.chmod(0o2775)
        except PermissionError:
            pass
    return directory


def _atomic_text(path: Path, text: str) -> None:
    ensure_shared_directory(path.parent)
    atomic_write_text(path, text, mode=0o664, durable=True)


def _remove_tree(path: Path) -> None:
    """Remove a tree despite transient NFS directory-entry visibility."""
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 4:
                # Open files become temporary .nfs entries. The completed
                # repair must not fail merely because NFS defers cleanup of
                # its now-detached quarantine.
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(0.05 * (attempt + 1))


@dataclass(frozen=True)
class RegistryPaths:
    """Resolved project context and paths into the shared orchestration store."""

    project: str
    bids_root: Path
    project_root: Path
    control: Path
    database: Path
    lock: Path
    recovery_lock: Path
    events: Path
    workers: Path
    snapshots: Path
    workflows: Path

    @classmethod
    def for_project(
        cls,
        project: str,
        *,
        bids_root: str | Path = BIDS_PATH,
        registry_path: str | Path | None = None,
    ) -> "RegistryPaths":
        """Resolve project and control paths without creating directories.

        An explicit registry_path overrides the site default. A nondefault BIDS
        root otherwise uses a sibling .nro directory.
        """
        resolved_bids_root = Path(bids_root).expanduser().resolve()
        project_root = resolved_bids_root / project
        if registry_path is None:
            control = (
                REGISTRY_PATH
                if resolved_bids_root == Path(BIDS_PATH).expanduser().resolve()
                else resolved_bids_root.parent / ".nro"
            )
        else:
            control = Path(registry_path).expanduser().resolve()
        paths = ControlPaths(control)
        paths.require_current_layout()
        science = paths.branch("main")
        return cls(
            project=project,
            bids_root=resolved_bids_root,
            project_root=project_root,
            control=control,
            database=paths.database,
            lock=paths.scheduler / "registry.lock",
            recovery_lock=paths.scheduler / "registry.lock.recovery",
            events=science / "events",
            workers=paths.scheduler / "workers",
            snapshots=science / "snapshots",
            workflows=science / "workflows",
        )


@dataclass(frozen=True)
class LockOwner:
    """Process and scheduler identity recorded by a registry lock holder."""

    token: str
    hostname: str
    pid: int
    uid: int
    slurm_job_id: str | None
    slurm_array_task_id: str | None
    acquired_at: str
    lease_expires_at: float | None


class RegistryLockTimeout(TimeoutError):
    """Raised when the registry lock cannot be acquired within its timeout."""

    pass


class RegistryLock:
    """Filesystem lock with owner metadata and conservative abandoned-lock recovery."""

    def __init__(
        self,
        path: Path,
        recovery_path: Path,
        *,
        timeout: float = 120.0,
        stale_after: float = 300.0,
        lease_seconds: float | None = None,
    ) -> None:
        """Configure lock paths and waiting/recovery thresholds in seconds."""
        if lease_seconds is not None and lease_seconds <= 0:
            raise ValueError("Lock lease duration must be positive")
        self.path = path
        self.recovery_path = recovery_path
        self.timeout = timeout
        self.stale_after = stale_after
        self.lease_seconds = lease_seconds
        self.owner = LockOwner(
            token=uuid.uuid4().hex,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            uid=os.getuid(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            acquired_at=utcnow(),
            lease_expires_at=None,
        )
        self._held = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def owner_path(self) -> Path:
        """Return the owner-metadata file within the lock directory."""
        return self.path / "owner.json"

    def _read_owner(self) -> dict | None:
        try:
            return json.loads(self.owner_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _slurm_terminal(job_id: str) -> bool | None:
        """Check whether a job has left Slurm; query failures remain unknown.

        Slurm may report an expired job ID as an error instead of an empty
        result. Only that specific diagnostic establishes absence on failure.
        Accounting provides a bounded fallback when the live queue is
        temporarily unavailable.
        """
        if shutil.which("squeue"):
            try:
                result = subprocess.run(
                    ["squeue", "--noheader", "--jobs", str(job_id), "--format", "%T"],
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=5,
                    env={**os.environ, "LC_ALL": "C"},
                )
            except (OSError, subprocess.SubprocessError):
                result = None
            if result is not None:
                if not result.returncode:
                    return not bool(result.stdout.strip())
                if (
                    not result.stdout.strip()
                    and result.stderr.strip() == "slurm_load_jobs error: Invalid job id specified"
                ):
                    return True
        return RegistryLock._slurm_accounting_terminal(job_id)

    @staticmethod
    def _slurm_accounting_terminal(job_id: str) -> bool | None:
        """Return terminal state from Slurm accounting when it is conclusive."""
        if not shutil.which("sacct"):
            return None
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "--jobs",
                    str(job_id),
                    "--noheader",
                    "--parsable2",
                    "--format",
                    "JobIDRaw,State",
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode:
            return None
        states = {
            fields[1].split(maxsplit=1)[0].split("+", 1)[0].upper()
            for line in result.stdout.splitlines()
            if len(fields := line.split("|")) >= 2 and fields[0].strip() == str(job_id)
        }
        if not states:
            return None
        active = {
            "CONFIGURING",
            "COMPLETING",
            "PENDING",
            "REQUEUED",
            "REQUEUE_FED",
            "REQUEUE_HOLD",
            "RESIZING",
            "RUNNING",
            "SIGNALING",
            "STAGE_OUT",
            "STOPPED",
            "SUSPENDED",
        }
        terminal = {
            "BOOT_FAIL",
            "CANCELLED",
            "COMPLETED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "REVOKED",
            "SPECIAL_EXIT",
            "TIMEOUT",
        }
        if states <= terminal:
            return True
        if states & active:
            return False
        return None

    @staticmethod
    def _slurm_out_of_memory(job_id: str) -> bool | None:
        if not shutil.which("sacct"):
            return None
        try:
            result = subprocess.run(
                ["sacct", "--jobs", str(job_id), "--noheader", "--parsable2", "--format", "State"],
                check=True,
                text=True,
                capture_output=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        states = [line.split("|", 1)[0].strip().upper() for line in result.stdout.splitlines()]
        return any(state.startswith("OUT_OF_MEMORY") for state in states)

    @staticmethod
    def _slurm_timed_out(job_id: str) -> bool | None:
        """Return whether Slurm accounting identifies a wall-time expiry."""
        if not shutil.which("sacct"):
            return None
        try:
            result = subprocess.run(
                ["sacct", "--jobs", str(job_id), "--noheader", "--parsable2", "--format", "State"],
                check=True,
                text=True,
                capture_output=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        states = [line.split("|", 1)[0].strip().upper() for line in result.stdout.splitlines()]
        return any(state.startswith("TIMEOUT") for state in states)

    def _owner_definitively_dead(self, owner: dict | None) -> bool:
        if owner is None:
            return False
        lease_expires_at = owner.get("lease_expires_at")
        if lease_expires_at is not None:
            try:
                return time.time() > float(lease_expires_at)
            except (TypeError, ValueError):
                return False
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return False
        if age < self.stale_after:
            return False
        if owner.get("hostname") == socket.gethostname():
            try:
                os.kill(int(owner["pid"]), 0)
            except ProcessLookupError:
                return True
            except (KeyError, ValueError, PermissionError, OSError):
                pass
        job_id = owner.get("slurm_job_id")
        if job_id:
            return self._slurm_terminal(str(job_id)) is True
        if owner.get("hostname") != socket.gethostname():
            return False
        return False

    def _renew_lease(self) -> bool:
        """Renew this owner's cross-host lease while fencing recovery."""
        if self.lease_seconds is None or not self._held:
            return False
        try:
            self.recovery_path.mkdir(mode=0o2775)
            self.recovery_path.chmod(0o2775)
        except FileExistsError:
            return False
        try:
            current = self._read_owner()
            if not current or current.get("token") != self.owner.token:
                return False
            self.owner = replace(
                self.owner,
                lease_expires_at=time.time() + self.lease_seconds,
            )
            _atomic_text(self.owner_path, json.dumps(asdict(self.owner), indent=2) + "\n")
            return True
        finally:
            try:
                self.recovery_path.rmdir()
            except OSError:
                pass

    def _heartbeat(self) -> None:
        """Renew a long-lived lease until release begins."""
        assert self.lease_seconds is not None
        interval = min(15.0, self.lease_seconds / 3.0)
        while not self._heartbeat_stop.wait(interval):
            try:
                self._renew_lease()
            except OSError:
                # A transient shared-filesystem failure is retried. If renewal
                # remains impossible, expiry makes recovery possible elsewhere.
                pass

    def _start_heartbeat(self) -> None:
        if self.lease_seconds is None:
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            name="nro-lock-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
            self._heartbeat_thread = None

    def _recover_if_safe(self) -> bool:
        try:
            self.recovery_path.mkdir(mode=0o2775)
            self.recovery_path.chmod(0o2775)
        except FileExistsError:
            return False
        try:
            if not self.path.exists():
                return True
            owner = self._read_owner()
            if not self._owner_definitively_dead(owner):
                return False
            token = str((owner or {}).get("token") or "unknown")
            stale = self.path.with_name(f"registry.lock.stale-{token}-{uuid.uuid4().hex}")
            try:
                self.path.rename(stale)
            except FileNotFoundError:
                return True
            shutil.rmtree(stale, ignore_errors=True)
            return True
        finally:
            try:
                self.recovery_path.rmdir()
            except OSError:
                pass

    def _show_wait(self, frame: int) -> bool:
        """Refresh the interactive registry-wait indicator."""
        if not sys.stderr.isatty():
            return False
        owner = self._read_owner() or {}
        job = owner.get("slurm_job_id")
        host = str(owner.get("hostname") or "").split(".", 1)[0]
        if job:
            detail = f" (held by Slurm job {job}" + (f" on {host})" if host else ")")
        elif owner.get("pid") and host:
            detail = f" (held by process {owner['pid']} on {host})"
        else:
            detail = ""
        marker = _WAIT_FRAMES[frame % len(_WAIT_FRAMES)]
        if "NO_COLOR" not in os.environ:
            color = _WAIT_COLORS[frame % len(_WAIT_COLORS)]
            marker = f"{color}{marker}{_RESET}"
        sys.stderr.write(f"{_CLEAR_LINE}{marker} Waiting for registry access{detail}...")
        sys.stderr.flush()
        return True

    def _owner_is_old(self) -> bool:
        """Return whether lock recovery may need a scheduler query."""
        try:
            return time.time() - self.path.stat().st_mtime >= self.stale_after
        except OSError:
            return False

    @staticmethod
    def _clear_wait(visible: bool) -> None:
        """Remove an interactive registry-wait indicator."""
        if visible:
            sys.stderr.write(_CLEAR_LINE)
            sys.stderr.flush()

    def acquire(self) -> "RegistryLock":
        """Acquire the lock, recovering only demonstrably abandoned ownership.

        Raise RegistryLockTimeout when the configured waiting period expires.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
        started = time.monotonic()
        deadline = started + self.timeout
        delay = 0.05
        frame = 0
        wait_visible = False
        while True:
            try:
                self.path.mkdir(mode=0o2775)
                self.path.chmod(0o2775)
            except FileExistsError:
                now = time.monotonic()
                if now - started >= _WAIT_NOTICE_SECONDS or self._owner_is_old():
                    wait_visible = self._show_wait(frame) or wait_visible
                    frame += 1
                try:
                    self._recover_if_safe()
                except BaseException:
                    self._clear_wait(wait_visible)
                    raise
                if time.monotonic() >= deadline:
                    self._clear_wait(wait_visible)
                    raise RegistryLockTimeout(
                        f"Timed out waiting for registry lock {self.path}; owner={self._read_owner()}"
                    )
                try:
                    time.sleep(random.uniform(delay, min(2.0, delay * 2.0)))
                except BaseException:
                    self._clear_wait(wait_visible)
                    raise
                delay = min(2.0, delay * 1.6)
                continue
            self._clear_wait(wait_visible)
            try:
                if self.lease_seconds is not None:
                    self.owner = replace(
                        self.owner,
                        lease_expires_at=time.time() + self.lease_seconds,
                    )
                _atomic_text(self.owner_path, json.dumps(asdict(self.owner), indent=2) + "\n")
            except BaseException:
                shutil.rmtree(self.path, ignore_errors=True)
                raise
            self._held = True
            self._start_heartbeat()
            return self

    def release(self) -> None:
        """Release the lock owned by this object and remove its owner record."""
        if not self._held:
            return
        self._stop_heartbeat()
        current = self._read_owner()
        if not current or current.get("token") != self.owner.token:
            self._held = False
            raise RuntimeError(f"Registry lock ownership changed while held: {self.path}")
        released = self.path.with_name(f"registry.lock.released-{self.owner.token}")
        self.path.rename(released)
        # The rename has already released the actual lock.  On the shared NFS
        # filesystem, immediate recursive removal can occasionally observe a
        # transient non-empty directory after ``owner.json`` was removed.  A
        # Cleanup failure must not turn an otherwise completed work item into an
        # error, nor can it affect another lock (the token is unique).
        for attempt in range(3):
            try:
                shutil.rmtree(released)
                break
            except OSError:
                if attempt == 2:
                    shutil.rmtree(released, ignore_errors=True)
                else:
                    time.sleep(0.05 * (attempt + 1))
        self._held = False

    def __enter__(self) -> "RegistryLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class Registry(WorkflowRegistry):
    """Transactional authority for work items, demand, attempts, and workers.

    Construction does not initialize storage. Mutating methods acquire the
    registry lock; callers should not alter the database directly.
    """

    def __init__(
        self,
        paths: RegistryPaths,
        *,
        lock_timeout: float = 120.0,
        stale_lock_after: float = 300.0,
        installation_maintenance: bool = False,
    ) -> None:
        """Bind resolved paths and lock timings without opening the database.

        ``installation_maintenance`` lets the installer resume an incomplete
        shared setup. Normal callers must leave it disabled.
        """
        from nro.configuration.site import require_execution_support

        require_execution_support(installation_maintenance=installation_maintenance)
        self.paths = paths
        self.lock_timeout = lock_timeout
        self.stale_lock_after = stale_lock_after

    @classmethod
    def for_project(
        cls,
        project: str,
        *,
        bids_root: str | Path = BIDS_PATH,
        registry_path: str | Path | None = None,
        **kwargs,
    ) -> "Registry":
        """Create a registry handle for a project using the shared control store."""
        return cls(
            RegistryPaths.for_project(
                project,
                bids_root=bids_root,
                registry_path=registry_path,
            ),
            **kwargs,
        )

    def _lock(self) -> RegistryLock:
        return RegistryLock(
            self.paths.lock,
            self.paths.recovery_lock,
            timeout=self.lock_timeout,
            stale_after=self.stale_lock_after,
        )

    def existing_database_path(self) -> Path:
        """Return the registry path without mutating shared state."""
        return self.paths.database

    def _prepare_directories(self) -> None:
        ensure_shared_directory(self.paths.control)
        try:
            self.paths.control.chmod(0o2775)
        except PermissionError:
            pass
        for path in (
            self.paths.events,
            self.paths.workers,
            self.paths.snapshots,
            self.paths.workflows,
        ):
            ensure_shared_directory(path)
            try:
                path.chmod(0o2775)
            except PermissionError:
                pass
        if self.paths.database.exists():
            try:
                self.paths.database.chmod(0o664)
            except PermissionError:
                pass

    def _initialize_locked(self) -> None:
        if self.paths.database.exists():
            return
        temporary = self.paths.database.with_name(
            f"registry.sqlite3.initializing-{uuid.uuid4().hex}"
        )
        try:
            connection = sqlite3.connect(temporary)
            try:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                connection.executescript(current_schema_sql())
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (
                        ("registry_uuid", uuid.uuid4().hex),
                        ("schema_version", str(SCHEMA_VERSION)),
                        ("created_at", utcnow()),
                        ("scope", "lab"),
                    ),
                )
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                connection.commit()
                result = connection.execute("PRAGMA integrity_check").fetchone()
                if result != ("ok",):
                    raise RuntimeError(f"New registry failed integrity check: {result}")
            finally:
                connection.close()
            os.chmod(temporary, 0o664)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            if self.paths.database.exists():
                raise RuntimeError(
                    f"Registry appeared unexpectedly during initialization: {self.paths.database}"
                )
            os.replace(temporary, self.paths.database)
        finally:
            temporary.unlink(missing_ok=True)

    def _connect_locked(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.database, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        app_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if app_id != APPLICATION_ID:
            connection.close()
            raise RuntimeError(f"Not an nro registry: {self.paths.database}")
        if version != SCHEMA_VERSION:
            connection.close()
            raise RuntimeError(
                f"Unsupported registry schema {version}; expected {SCHEMA_VERSION}: "
                f"{self.paths.database}. Runtime commands do not modify registry schemas. "
                "Run shared installation maintenance to migrate a supported schema or "
                "reconstruct older private control state. Public derivatives are stored "
                "outside the registry and are not removed."
            )
        return connection

    @contextlib.contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Open a locked database context, optionally for a write transaction.

        Schema incompatibility raises RuntimeError. Writes commit on success and
        roll back on failure; the context releases its connection and lock.
        """
        if os.environ.get("NRO_PROCESS_ROLE") == "worker":
            raise RuntimeError(
                "Workers and scientific subprocesses cannot open the scheduler registry"
            )
        self._prepare_directories()
        with self._lock():
            self._initialize_locked()
            connection = self._connect_locked()
            try:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
                if write:
                    connection.commit()
            except BaseException:
                if write:
                    connection.rollback()
                raise
            finally:
                connection.close()

    @contextlib.contextmanager
    def _repair_connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Open the registry without requiring the current schema version.

        This narrow escape hatch exists only so repair can stop workers before
        replacing an obsolete registry. It deliberately provides no migration
        behavior: the worker-control tables must still have the fields used by
        the current repair procedure.
        """
        if os.environ.get("NRO_PROCESS_ROLE") == "worker":
            raise RuntimeError(
                "Workers and scientific subprocesses cannot open the scheduler registry"
            )
        self._prepare_directories()
        with self._lock():
            self._initialize_locked()
            connection = sqlite3.connect(self.paths.database, timeout=60.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            app_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            if app_id != APPLICATION_ID:
                connection.close()
                raise RuntimeError(f"Not an nro registry: {self.paths.database}")
            try:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
                if write:
                    connection.commit()
            except BaseException:
                if write:
                    connection.rollback()
                raise
            finally:
                connection.close()

    @contextlib.contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Open a query-only connection while holding the registry lock.

        SQLite file locks alone are not reliable enough to coordinate registry
        access across the site's compute nodes. Observational commands use the
        same cross-host lock as mutations, while ``query_only`` prevents their
        connections from changing registry state.
        """
        with self.connection() as connection:
            connection.execute("PRAGMA query_only=ON")
            yield connection

    def initialize(self) -> None:
        """Create and validate the private registry structure without requesting work."""
        with self.connection():
            pass

    def stored_schema_version(self) -> int:
        """Read the scheduler schema without requiring it to match this source."""
        with self._repair_connection() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def migrate_schema(self) -> Path | None:
        """Atomically migrate supported scheduler state while its maintenance lock is held."""
        self._prepare_directories()
        with self._lock():
            if not self.paths.database.is_file():
                self._initialize_locked()
                return None
            return migrate_database(self.paths.database, SCHEMA_DEFINITION)

    def reinitialize(
        self, *, preserve_branch_runtime: bool = False, retain_backup: bool = False
    ) -> Path | None:
        """Rebuild work state, preserving ingestion, registrations, and execution sources.

        Existing state is treated as opaque and is never migrated. The active
        registry lock remains in place throughout replacement so another
        registry client cannot observe a partially rebuilt control directory.
        If initialization fails, the original control state is restored.
        Running ingestion leases must be resolved before replacement.

        preserve_branch_runtime retains scientific databases, configurations,
        and workflow snapshots. retain_backup
        keeps the replaced state and returns its directory with an original-path
        index; otherwise the replaced state is removed and None is returned.
        """
        self._prepare_directories()
        scheduler = ControlPaths(self.paths.control).scheduler
        quarantine = scheduler / f".repair-{uuid.uuid4().hex}"
        retained = {
            self.paths.lock,
            self.paths.recovery_lock,
            quarantine,
            scheduler / "artifact-mutation.lock",
            scheduler / "artifact-mutation.recovery-lock",
        }
        science = (
            self.paths.events,
            self.paths.snapshots,
            self.paths.workflows,
        )
        if preserve_branch_runtime:
            science = ()
            retained.add(scheduler / "implementation.json")
        retained.update(scheduler.glob(".repair-*"))

        def replaceable():
            return tuple(
                path
                for path in (*scheduler.iterdir(), *science)
                if path not in retained and path.exists()
            )

        from nro.orchestration.execution_cache import cache_lock

        with cache_lock(self.paths.control), self._lock():
            if self.paths.database.exists():
                connection = sqlite3.connect(self.paths.database)
                try:
                    if (
                        connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_mutations'"
                        ).fetchone()
                        and connection.execute(
                            "SELECT 1 FROM artifact_mutations LIMIT 1"
                        ).fetchone()
                    ):
                        raise ValueError(
                            "Resolve the active or interrupted artifact mutation before registry repair"
                        )
                finally:
                    connection.close()
            from nro.bidsify.index import IngestionIndex

            if any(r["state"] == "running" for r in IngestionIndex(self).rows()):
                raise ValueError("Resolve active ingestion worker leases before registry repair")
            quarantine.mkdir(mode=0o2775)
            moved: list[tuple[Path, Path]] = []
            try:
                for index, source in enumerate(replaceable()):
                    destination = quarantine / str(index)
                    source.rename(destination)
                    moved.append((source, destination))
                if retain_backup:
                    from nro.engine.io import atomic_write_json

                    atomic_write_json(
                        quarantine / "index.json",
                        {str(destination.name): str(source) for source, destination in moved},
                        durable=True,
                    )
                self._prepare_directories()
                self._initialize_locked()
                connection = self._connect_locked()
                connection.close()
            except BaseException:
                for path in replaceable():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                for source, destination in moved:
                    destination.rename(source)
                (quarantine / "index.json").unlink(missing_ok=True)
                quarantine.rmdir()
                raise
        if retain_backup:
            return quarantine
        _remove_tree(quarantine)
        return None

    def restore_reinitialization(self, quarantine: Path) -> None:
        """Restore the exact scheduler state retained by :meth:`reinitialize`.

        This operation is reserved for a failed shared-maintenance transaction.
        The same cross-host locks used for replacement prevent clients from
        observing the rollback halfway through.
        """
        quarantine = Path(quarantine).resolve()
        scheduler = ControlPaths(self.paths.control).scheduler
        if quarantine.parent != scheduler or not (quarantine / "index.json").is_file():
            raise ValueError("Scheduler recovery directory is invalid")
        index = json.loads((quarantine / "index.json").read_text())
        if not isinstance(index, dict):
            raise ValueError("Scheduler recovery index is invalid")
        restored = [(Path(str(source)), quarantine / str(name)) for name, source in index.items()]
        from nro.orchestration.execution_cache import cache_lock

        with cache_lock(self.paths.control), self._lock():
            for destination, _source in restored:
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink(missing_ok=True)
            for destination, source in restored:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
            (quarantine / "index.json").unlink()
            quarantine.rmdir()

    def workflow_history(self, workflow_id: str) -> list[dict]:
        """Return recorded revisions for a workflow in history order."""
        with self.connection() as db:
            rows = db.execute(
                """
                SELECT id, workflow_id, revision, definition_fingerprint,
                       source_path, created_at
                FROM workflow_revisions
                WHERE workflow_id=? ORDER BY revision
                """,
                (workflow_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def _upsert_work_item_graph_locked(
        self,
        db: sqlite3.Connection,
        work_item_records: Sequence[tuple["WorkItemSpec", dict]],
        *,
        now: str,
        external_ids: Mapping[str, int] | None = None,
        owner_branch: str | None = None,
    ) -> dict[str, int]:
        """Merge work items and dependency edges inside the current transaction."""
        from nro.orchestration.registry_work_items import upsert_work_item_graph

        return upsert_work_item_graph(
            db,
            work_item_records,
            now=now,
            external_ids=external_ids,
            owner_branch=owner_branch,
        )

    @staticmethod
    def _normalize_active_request_graph_locked(db: sqlite3.Connection) -> None:
        """Reconcile active demand inside the current transaction."""
        from nro.orchestration.registry_work_items import normalize_active_request_graph

        normalize_active_request_graph(db)

    def _validate_work_item_projects(self, work_items: Sequence["WorkItemSpec"]) -> None:
        foreign_projects = {
            spec.project for spec in work_items if spec.project != self.paths.project
        }
        if foreign_projects:
            raise ValueError(
                "A registry operation may contain work items from only its selected project: "
                + ", ".join(sorted(foreign_projects))
            )

    def register_work_items(self, work_items: Sequence["WorkItemSpec"]) -> dict[str, int]:
        """Discover work-item contracts and edges without creating a request."""
        self._validate_work_item_projects(work_items)
        work_item_records = tuple((spec, spec.as_record()) for spec in work_items)
        with self.connection(write=True) as db:
            work_item_ids = self._upsert_work_item_graph_locked(db, work_item_records, now=utcnow())
            self._normalize_active_request_graph_locked(db)
            return work_item_ids

    def register_owned_lineages(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        branch_registry_id: str | None = None,
    ) -> dict[str, int]:
        """Restore module lineages from derivative ownership records.

        Development lineages receive the same central namespace used during
        normal branch admission. Returned IDs remain keyed by the public,
        branch-local lineage fingerprints.
        """
        ordered = sorted(
            records,
            key=lambda item: (
                CONFIGURATION_CLASSES.index(str(item["configuration_class"])),
                str(item["lineage_fingerprint"]),
            ),
        )
        lineage_ids: dict[str, int] = {}
        now = utcnow()
        with self.connection(write=True) as db:
            for record in ordered:
                configuration_class = str(record["configuration_class"])
                public_fingerprint = str(record["lineage_fingerprint"])
                lineage_fingerprint = (
                    fingerprint({"owner": branch_registry_id, "lineage": public_fingerprint})
                    if branch_registry_id is not None
                    else public_fingerprint
                )
                directory_label = str(record["directory_label"])
                configuration = record["configuration"]
                if not isinstance(configuration, Mapping):
                    raise ValueError("Ownership record configuration must be a mapping")
                existing = db.execute(
                    """SELECT id, config_id, directory_label
                       FROM module_lineages
                       WHERE configuration_class=? AND lineage_fingerprint=?""",
                    (configuration_class, lineage_fingerprint),
                ).fetchone()
                if branch_registry_id is None:
                    collision = db.execute(
                        """SELECT lineage_fingerprint FROM module_lineages
                           WHERE configuration_class=? AND directory_label=?""",
                        (configuration_class, directory_label),
                    ).fetchone()
                    if collision and str(collision["lineage_fingerprint"]) != lineage_fingerprint:
                        raise ValueError(
                            f"Derivative directory {configuration_class}/{directory_label} "
                            "declares conflicting module lineages"
                        )
                if existing:
                    if (
                        str(existing["config_id"]) != str(configuration["id"])
                        or str(existing["directory_label"]) != directory_label
                    ):
                        raise ValueError(
                            f"Ownership record conflicts with registered lineage {lineage_fingerprint}"
                        )
                    lineage_id = int(existing["id"])
                    db.execute(
                        """UPDATE module_lineages
                           SET config_fingerprint=?, resolved_yaml=? WHERE id=?""",
                        (
                            str(configuration["fingerprint"]),
                            yaml.safe_dump(configuration["resolved"], sort_keys=False),
                            lineage_id,
                        ),
                    )
                else:
                    cursor = db.execute(
                        """
                        INSERT INTO module_lineages(
                            configuration_class, config_id, config_fingerprint,
                            lineage_fingerprint, resolved_yaml, directory_label, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            configuration_class,
                            str(configuration["id"]),
                            str(configuration["fingerprint"]),
                            lineage_fingerprint,
                            yaml.safe_dump(configuration["resolved"], sort_keys=False),
                            directory_label,
                            now,
                        ),
                    )
                    lineage_id = int(cursor.lastrowid)
                lineage_ids[public_fingerprint] = lineage_id

            for record in ordered:
                lineage_id = lineage_ids[str(record["lineage_fingerprint"])]
                db.execute(
                    "DELETE FROM module_lineage_dependencies WHERE module_lineage_id=?",
                    (lineage_id,),
                )
                for upstream in record["upstream"]:
                    parent_fingerprint = str(upstream["lineage_fingerprint"])
                    try:
                        parent_id = lineage_ids[parent_fingerprint]
                    except KeyError as error:
                        raise ValueError(
                            f"Ownership record lacks upstream lineage {parent_fingerprint}"
                        ) from error
                    db.execute(
                        """INSERT INTO module_lineage_dependencies(
                               module_lineage_id,
                               upstream_module_lineage_id, role
                           ) VALUES (?, ?, ?)""",
                        (lineage_id, parent_id, str(upstream["role"])),
                    )
        return lineage_ids

    def replace_bids_inventory(self, inventory: Mapping[str, Sequence[str]]) -> None:
        """Replace the discovered source project and participant catalog."""
        now = utcnow()
        with self.connection(write=True) as db:
            db.execute("DELETE FROM bids_participants")
            db.execute("DELETE FROM bids_projects")
            for project, participants in sorted(inventory.items()):
                project_path = self.paths.bids_root / project
                db.execute(
                    "INSERT INTO bids_projects(project, path, discovered_at) VALUES (?, ?, ?)",
                    (project, str(project_path), now),
                )
                db.executemany(
                    """INSERT INTO bids_participants(
                           project, participant, path, discovered_at
                       ) VALUES (?, ?, ?, ?)""",
                    (
                        (
                            project,
                            participant.removeprefix("sub-"),
                            str(project_path / f"sub-{participant.removeprefix('sub-')}"),
                            now,
                        )
                        for participant in participants
                    ),
                )

    def work_item_ids(self, work_item_keys: Sequence[str]) -> dict[str, int]:
        """Resolve registered work-item keys without changing demand."""
        if not work_item_keys:
            return {}
        unique = tuple(dict.fromkeys(work_item_keys))
        placeholders = ",".join("?" for _ in unique)
        with self.connection() as db:
            return {
                str(row["work_item_key"]): int(row["id"])
                for row in db.execute(
                    f"SELECT id, work_item_key FROM work_items WHERE work_item_key IN ({placeholders})",
                    unique,
                )
            }

    def create_request(
        self,
        *,
        registered: RegisteredWorkflow,
        target_module: str,
        selectors: dict,
        work_items: Sequence["WorkItemSpec"],
        terminal_work_item_keys: Sequence[str],
        concurrency: int,
        partition: str | None,
        user_name: str | None = None,
    ) -> str:
        """Merge a planned graph and create a new active demand request."""
        if concurrency < 1:
            raise ValueError("Concurrency must be at least one")
        self._validate_work_item_projects(work_items)
        request_id = uuid.uuid4().hex
        now = utcnow()
        terminal = set(terminal_work_item_keys)
        work_item_records = tuple((spec, spec.as_record()) for spec in work_items)
        with self.connection(write=True) as db:
            db.execute(
                """
                INSERT INTO requests(
                    id, user_name, project, workflow_revision_id, target_module,
                    selectors_json, concurrency, partition_name, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    request_id,
                    user_name or getpass.getuser(),
                    self.paths.project,
                    registered.revision_id,
                    target_module,
                    json.dumps(selectors, sort_keys=True),
                    concurrency,
                    partition,
                    now,
                    now,
                ),
            )
            work_item_ids = self._upsert_work_item_graph_locked(db, work_item_records, now=now)
            for spec in work_items:
                work_item_id = work_item_ids[spec.key]
                db.execute(
                    """
                    INSERT INTO request_work_items(request_id, work_item_id, role, demand_state)
                    VALUES (?, ?, ?, 'active')
                    """,
                    (request_id, work_item_id, "target" if spec.key in terminal else "dependency"),
                )
            # A shared multirun work item can gain or lose upstream runs while an
            # older request is still active. Keep every active request aligned
            # with the current global graph.
            self._normalize_active_request_graph_locked(db)
        return request_id

    def request_rows(self, *, include_terminal: bool = True) -> list[dict]:
        """Read request records matching the supplied selection filters."""
        where = "" if include_terminal else "WHERE state='active'"
        with self.connection() as db:
            rows = db.execute(
                f"""
                SELECT r.*, wr.workflow_id, wr.revision AS workflow_revision
                FROM requests r
                JOIN workflow_revisions wr ON wr.id=r.workflow_revision_id
                {where}
                ORDER BY r.created_at
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def set_active_concurrency(self, concurrency: int) -> int:
        """Set the lab-wide concurrency limit on every active request."""
        if concurrency < 1:
            raise ValueError("Concurrency must be at least one")
        with self.connection(write=True) as db:
            cursor = db.execute(
                "UPDATE requests SET concurrency=? WHERE state='active'",
                (concurrency,),
            )
            from nro.bidsify.index import IngestionIndex

            return int(cursor.rowcount) + IngestionIndex(self).set_concurrency_locked(concurrency)

    def set_gpu_concurrency(self, concurrency: int) -> int:
        """Set the independent limit for resource-specific GPU runner steps."""
        if concurrency < 1:
            raise ValueError("GPU concurrency must be at least one")
        with self.connection(write=True) as db:
            db.execute(
                """INSERT INTO metadata(key,value) VALUES ('gpu_concurrency',?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (str(concurrency),),
            )
        return 1

    def work_item_rows(self, *, read_only: bool = False) -> list[dict]:
        """Read work-item records with their current orchestration and artifact state."""
        from nro.orchestration.registry_status import work_item_rows

        manager = self.read_connection() if read_only else self.connection()
        with manager as db:
            return work_item_rows(db)

    def work_item_dependencies(self, *, read_only: bool = False) -> list[tuple[int, int]]:
        """Return dependency relationships for the selected work-item records."""
        from nro.orchestration.registry_status import work_item_dependencies

        manager = self.read_connection() if read_only else self.connection()
        with manager as db:
            return work_item_dependencies(db)

    def work_item_status_snapshot(
        self,
        *,
        read_only: bool = False,
        artifact_states: Mapping[int, tuple[str, str]] | None = None,
    ) -> list[dict]:
        """Return the registry's canonical current state for every work item.

        ``work_items`` retain artifact freshness and ``attempts`` retain immutable
        execution history.  This method is the sole projection of those facts
        into one current work-item state for user-facing tools. A read-only
        preview may supply temporary artifact states without persisting them.
        """
        from nro.orchestration.registry_status import (
            project_work_item_status,
            work_item_dependencies,
            work_item_rows,
        )

        manager = self.read_connection() if read_only else self.connection()
        with manager as db:
            rows = work_item_rows(db)
            dependencies = work_item_dependencies(db)
        return project_work_item_status(rows, dependencies, artifact_states=artifact_states)

    def register_worker(
        self,
        worker_id: str,
        *,
        resource_class: str,
        memory_gb: int = 32,
        slurm_job_id: str | None = None,
        user_name: str | None = None,
        hostname: str | None = None,
        pid: int | None = None,
        lease_seconds: float = 120.0,
    ) -> None:
        """Register or refresh a worker lease and its scheduler allocation.

        Workers registering during maintenance are marked for shutdown.
        """
        now = utcnow()
        worker_user = user_name if user_name is not None else getpass.getuser()
        worker_host = hostname if hostname is not None else socket.gethostname()
        worker_pid = pid if pid is not None else os.getpid()
        with self.connection(write=True) as db:
            repair = db.execute(
                "SELECT value FROM metadata WHERE key='maintenance_mode'"
            ).fetchone()
            state = "shutdown_requested" if repair is not None else "idle"
            db.execute(
                """
                INSERT INTO workers(id, user_name, resource_class, memory_gb, slurm_job_id, state,
                                    hostname, pid, lease_expires_at, started_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET state=excluded.state, lease_expires_at=excluded.lease_expires_at,
                    user_name=excluded.user_name, resource_class=excluded.resource_class,
                    memory_gb=excluded.memory_gb, slurm_job_id=excluded.slurm_job_id,
                    hostname=excluded.hostname, pid=excluded.pid, updated_at=excluded.updated_at
                """,
                (
                    worker_id,
                    worker_user,
                    resource_class,
                    memory_gb,
                    slurm_job_id,
                    state,
                    worker_host,
                    worker_pid,
                    time.time() + lease_seconds,
                    now,
                    now,
                ),
            )

    def heartbeat_worker(self, worker_id: str, *, state: str, lease_seconds: float = 120.0) -> None:
        """Renew a worker lease without clearing an existing shutdown request."""
        with self.connection(write=True) as db:
            db.execute(
                """UPDATE workers
                   SET state=CASE WHEN state='shutdown_requested' THEN state ELSE ? END,
                       lease_expires_at=?, updated_at=? WHERE id=?""",
                (state, time.time() + lease_seconds, utcnow(), worker_id),
            )

    def worker_shutdown_requested(self, worker_id: str) -> bool:
        """Return whether the worker has a persisted shutdown request."""
        with self.connection() as db:
            row = db.execute("SELECT state FROM workers WHERE id=?", (worker_id,)).fetchone()
            return bool(row and row["state"] == "shutdown_requested")

    def worker_pool_activity(self, *, for_repair: bool = False) -> dict:
        """Return active worker processes and scheduler submissions."""
        connection = self._repair_connection if for_repair else self.connection
        with connection() as db:
            workers = [
                dict(row)
                for row in db.execute(
                    """SELECT id, user_name, hostname, pid, slurm_job_id, state,
                                      lease_expires_at
                       FROM workers WHERE state IN
                           ('idle', 'running', 'draining', 'shutdown_requested')"""
                )
            ]
            submissions = [
                dict(row)
                for row in db.execute(
                    """SELECT id, slurm_job_id, state
                       FROM scheduler_submissions
                       WHERE state IN
                           ('prepared', 'submitted', 'running', 'cancel_requested')"""
                )
            ]
        return {"workers": workers, "submissions": submissions}

    def request_worker_shutdown(
        self,
        *,
        user_name: str | None = None,
        all_users: bool = False,
        for_repair: bool = False,
    ) -> dict:
        """Stop worker allocations without withdrawing work-item demand."""
        if all_users and user_name is not None:
            raise ValueError("user_name and all_users are mutually exclusive")
        owner = user_name or getpass.getuser()
        now = utcnow()
        connection = self._repair_connection if for_repair else self.connection
        with connection(write=True) as db:
            if for_repair:
                db.execute(
                    """INSERT INTO metadata(key, value) VALUES ('maintenance_mode', 'repair')
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value"""
                )
            worker_where = "" if all_users else "AND user_name=?"
            worker_parameters: tuple = () if all_users else (owner,)
            workers = [
                dict(row)
                for row in db.execute(
                    f"""SELECT id, user_name, hostname, pid, slurm_job_id, state,
                               lease_expires_at
                        FROM workers
                        WHERE state IN ('idle', 'running', 'draining', 'shutdown_requested')
                        {worker_where}""",
                    worker_parameters,
                )
            ]
            worker_ids = [str(row["id"]) for row in workers]
            if worker_ids:
                placeholders = ",".join("?" for _ in worker_ids)
                db.execute(
                    f"""UPDATE workers SET state='shutdown_requested', updated_at=?
                        WHERE id IN ({placeholders})""",
                    (now, *worker_ids),
                )
                attempt_count = db.execute(
                    f"""UPDATE attempts SET state='cancel_requested',
                               error_type='WorkerTerminated',
                               error_message=?
                        WHERE worker_id IN ({placeholders})
                          AND state IN ('queued', 'running')""",
                    (
                        "Registry repair requested worker shutdown"
                        if for_repair
                        else "Owning user requested worker shutdown",
                        *worker_ids,
                    ),
                ).rowcount
            else:
                attempt_count = 0

            if all_users:
                submission_query = """
                    SELECT DISTINCT ss.id, ss.slurm_job_id
                    FROM scheduler_submissions ss
                    WHERE ss.state IN ('prepared', 'submitted', 'running', 'cancel_requested')
                """
                submission_parameters = ()
            else:
                submission_query = """
                    SELECT DISTINCT ss.id, ss.slurm_job_id
                    FROM scheduler_submissions ss
                    LEFT JOIN requests r ON r.id=ss.request_id
                    LEFT JOIN workers predecessor ON predecessor.id=ss.predecessor_worker_id
                    LEFT JOIN workers allocation ON allocation.slurm_job_id=ss.slurm_job_id
                    LEFT JOIN metadata ingestion_owner ON ingestion_owner.key='ingestion_submission_owner:' || ss.id
                    WHERE ss.state IN ('prepared', 'submitted', 'running', 'cancel_requested')
                      AND (ingestion_owner.value=? OR (ingestion_owner.value IS NULL
                           AND (r.user_name=? OR predecessor.user_name=? OR allocation.user_name=?)))
                """
                submission_parameters = (owner, owner, owner, owner)
            submissions = [dict(row) for row in db.execute(submission_query, submission_parameters)]
            submission_ids = [int(row["id"]) for row in submissions]
            if submission_ids:
                placeholders = ",".join("?" for _ in submission_ids)
                db.execute(
                    f"""UPDATE scheduler_submissions
                        SET state=CASE WHEN slurm_job_id IS NULL THEN 'cancelled'
                                       ELSE 'cancel_requested' END
                        WHERE id IN ({placeholders})""",
                    tuple(submission_ids),
                )

        jobs = {
            str(row["slurm_job_id"]) for row in (*workers, *submissions) if row.get("slurm_job_id")
        }
        return {
            "workers": len(workers),
            "attempts": int(attempt_count),
            "submission_count": len(submissions),
            "submissions": [
                (int(row["id"]), str(row["slurm_job_id"]))
                for row in submissions
                if row.get("slurm_job_id")
            ],
            "job_ids": tuple(sorted(jobs)),
            "worker_rows": workers,
        }

    def close_worker(self, worker_id: str, *, state: str = "exited") -> None:
        """Record a terminal worker state and end its active lease."""
        with self.connection(write=True) as db:
            db.execute(
                "UPDATE workers SET state=?, lease_expires_at=NULL, updated_at=? WHERE id=?",
                (state, utcnow(), worker_id),
            )

    def confirm_worker_shutdown(self, worker_ids: Iterable[str]) -> dict[str, int]:
        """Finalize worker records after their processes are confirmed inactive.

        The caller must first verify process or allocation termination. This
        method releases interrupted execution records while preserving demand.
        """
        selected = tuple(sorted(set(map(str, worker_ids))))
        if not selected:
            return {"workers": 0, "attempts": 0, "ingestion": 0}
        placeholders = ",".join("?" for _ in selected)
        now = utcnow()
        with self.connection(write=True) as db:
            rows = db.execute(
                f"SELECT id,state FROM workers WHERE id IN ({placeholders})", selected
            ).fetchall()
            active = [
                row["id"]
                for row in rows
                if row["state"] not in {"shutdown_requested", "exited", "terminated", "lost"}
            ]
            if active:
                raise RuntimeError(
                    "Cannot confirm workers that did not request shutdown: " + ", ".join(active)
                )
            from nro.bidsify.index import IngestionIndex

            ingestion = IngestionIndex(self).recover_locked(set(selected))
            IngestionIndex(self).clear_publication_barriers_locked(db)
            attempts = db.execute(
                f"""UPDATE attempts SET state='cancelled', completed_at=?
                    WHERE worker_id IN ({placeholders})
                      AND state IN ('queued','running','cancel_requested')""",
                (now, *selected),
            ).rowcount
            db.execute(
                f"""UPDATE resource_step_tasks SET state='cancelled',completed_at=?,updated_at=?,
                           error_type='WorkerTerminated',
                           error_message='Worker terminated during resource-specific runner step'
                    WHERE worker_id IN ({placeholders}) AND state='running'""",
                (now, now, *selected),
            )
            workers = db.execute(
                f"""UPDATE workers SET state='terminated', lease_expires_at=NULL, updated_at=?
                    WHERE id IN ({placeholders}) AND state='shutdown_requested'""",
                (now, *selected),
            ).rowcount
            dependency_state.synchronize(db, now=now)
        return {"workers": workers, "attempts": attempts, "ingestion": ingestion}

    def claim_ready_work_item(
        self,
        worker_id: str,
        resource_classes: Sequence[str],
        *,
        memory_gb: int = 32,
    ) -> "ExecutionEnvelope | None":
        """Claim one demanded, nonfresh work item whose upstream work items are fresh."""
        resource_classes = tuple(resource_classes) or WORK_ITEM_RESOURCE_CLASSES
        placeholders = ",".join("?" for _ in resource_classes)
        now = utcnow()
        with self.connection(write=True) as db:
            worker = db.execute("SELECT state FROM workers WHERE id=?", (worker_id,)).fetchone()
            if worker is None or worker["state"] == "shutdown_requested":
                return None
            maintenance = db.execute(
                "SELECT 1 FROM metadata WHERE key='maintenance_mode'"
            ).fetchone()
            if maintenance is not None:
                return None
            assessment_lease = db.execute(
                "SELECT value FROM metadata WHERE key='artifact_assessment_lease_until'"
            ).fetchone()
            if assessment_lease is not None:
                try:
                    if float(assessment_lease["value"]) > time.time():
                        return None
                except (TypeError, ValueError):
                    pass
            dependency_state.synchronize(db, now=now)
            concurrency = int(
                db.execute(
                    "SELECT COALESCE(MAX(concurrency), 0) FROM requests WHERE state='active'"
                ).fetchone()[0]
            )
            active = int(
                db.execute(
                    """SELECT COUNT(*) FROM attempts a
                       WHERE a.state IN ('queued','running','cancel_requested')
                         AND NOT EXISTS (
                           SELECT 1 FROM resource_step_tasks task WHERE task.attempt_id=a.id
                         )"""
                ).fetchone()[0]
            )
            from nro.bidsify.index import IngestionIndex

            ingestion_active, _, ingestion_limit = IngestionIndex(self).summary(memory_gb)
            concurrency = max(concurrency, ingestion_limit)
            active += ingestion_active
            if concurrency < 1 or active >= concurrency:
                return None
            row = db.execute(
                f"""
                SELECT t.*, ci.config_fingerprint
                FROM work_items t
                JOIN module_lineages ci ON ci.id=t.module_lineage_id
                WHERE t.resource_class IN ({placeholders})
                  AND t.memory_gb <= ?
                  AND {dependency_state.WRITE_READY}
                  AND t.artifact_state != 'fresh'
                  AND NOT EXISTS (
                      SELECT 1 FROM metadata publication
                      WHERE publication.key='bids_publication:' || t.project
                  )
                  AND NOT (
                      t.module IN ('dynconn', 'microparcellation')
                      AND t.artifact_reason LIKE 'Selected raw run universe changed:%'
                  )
                  AND EXISTS (
                      SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.work_item_id=t.id AND rt.demand_state='active' AND r.state='active'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM work_item_dependencies td JOIN work_items up ON up.id=td.upstream_work_item_id
                      WHERE td.work_item_id=t.id AND up.artifact_state != 'fresh'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM attempts active WHERE active.work_item_id=t.id
                      AND active.state IN ('queued', 'running', 'cancel_requested')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM resource_step_tasks task WHERE task.work_item_id=t.id
                      AND task.generation=t.current_generation
                      AND task.revision_fingerprint=t.revision_fingerprint
                      AND task.state IN ('pending','running','error')
                  )
                  AND (
                      NOT EXISTS (SELECT 1 FROM attempts old WHERE old.work_item_id=t.id)
                      OR COALESCE((SELECT state FROM attempts old WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1), '') = 'success'
                      OR COALESCE((SELECT error_type FROM attempts old
                                   WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1), '')
                         IN ('UpstreamStale', 'UpstreamFailed', 'WorkerTerminated',
                             'WorkItemGraphChanged', 'RegistryUnavailable')
                             OR COALESCE((SELECT error_type FROM attempts old
                                  WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1), '')
                                = 'ResourcePreconditionChanged'
                      OR EXISTS (
                          SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                          WHERE rt.work_item_id=t.id AND rt.demand_state='active' AND r.state='active'
                          AND r.updated_at > COALESCE(
                              (SELECT CASE WHEN old.state='cancelled'
                                       THEN old.started_at ELSE old.completed_at END
                               FROM attempts old WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1),
                              '')
                      )
                  )
                ORDER BY CASE WHEN EXISTS (
                             SELECT 1 FROM resource_step_tasks resumed
                             WHERE resumed.work_item_id=t.id
                               AND resumed.generation=t.current_generation
                               AND resumed.revision_fingerprint=t.revision_fingerprint
                               AND resumed.state='success'
                         ) THEN 0 ELSE 1 END,
                         CASE t.module WHEN 'anat' THEN 1 WHEN 'func' THEN 2 WHEN 'clean' THEN 3
                                      WHEN 'microparcellation' THEN 4 WHEN 'dynconn' THEN 4 ELSE 5 END,
                         t.participant, t.work_item_key
                LIMIT 1
                """,
                (*resource_classes, memory_gb),
            ).fetchone()
            if row is None:
                return None
            work_item = dict(row)
            work_item["completed_resource_steps_json"] = self._completed_resource_steps_json(
                db, work_item
            )
            log_dir = self.paths.events / work_item_relative_directory(work_item)
            execution = db.execute(
                "SELECT * FROM work_item_execution WHERE work_item_id=?", (work_item["id"],)
            ).fetchone()
            if execution is not None:
                from nro.orchestration.branch_admission import prepare_attempt

                log_dir = (
                    ControlPaths(self.paths.control).branch(execution["branch"])
                    / "events"
                    / work_item_relative_directory(work_item)
                )
                ensure_shared_directory(log_dir)
                command = prepare_attempt(self, db, work_item, dict(execution), log_dir)
                if command is None:
                    return None
                work_item["command_json"] = json.dumps(command)
            ensure_shared_directory(log_dir)
            cursor = db.execute(
                """
                INSERT INTO attempts(work_item_id, worker_id, state, revision_fingerprint,
                                     memory_gb, started_at, log_path, created_at)
                VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    work_item["id"],
                    worker_id,
                    work_item["revision_fingerprint"],
                    memory_gb,
                    now,
                    str(log_dir / "work-item.log"),
                    now,
                ),
            )
            attempt_id = int(cursor.lastrowid)
            if execution is not None:
                payload = json.loads(Path(command[-3]).read_text())
                db.execute(
                    "INSERT INTO attempt_execution VALUES (?,?,?,?)",
                    (
                        attempt_id,
                        json.dumps(payload["context"]),
                        execution["provenance_json"],
                        work_item["command_json"],
                    ),
                )
            dependency_state.capture_inputs(db, attempt_id, int(work_item["id"]))
            # A work item has one current log, deliberately replaced by the next
            # attempt. Attempt history remains in the registry/events tables.
            log_path = log_dir / "work-item.log"
            db.execute(
                "UPDATE workers SET state='running', lease_expires_at=?, updated_at=? WHERE id=?",
                (time.time() + 120.0, now, worker_id),
            )
            work_item["attempt_id"] = attempt_id
            work_item["log_path"] = str(log_path)
            from nro.orchestration.contracts import ExecutionEnvelope

            return ExecutionEnvelope.from_registry_row(work_item)

    def current_worker_assignment(self, worker_id: str) -> "ExecutionEnvelope | None":
        """Recover the active assignment after an interrupted scheduler response."""
        with self.connection() as db:
            row = db.execute(
                """SELECT i.*,a.id AS attempt_id,a.log_path,
                          task.id AS resource_task_id,task.step_id AS target_step_id,
                          COALESCE(e.command_json,i.command_json) AS command_json
                   FROM attempts a JOIN work_items i ON i.id=a.work_item_id
                   LEFT JOIN attempt_execution e ON e.attempt_id=a.id
                   LEFT JOIN resource_step_tasks task ON task.attempt_id=a.id
                   WHERE a.worker_id=? AND a.state IN ('queued','running','cancel_requested')
                   ORDER BY a.id DESC LIMIT 1""",
                (worker_id,),
            ).fetchone()
            work_item = None if row is None else dict(row)
            if work_item is not None:
                work_item["completed_resource_steps_json"] = self._completed_resource_steps_json(
                    db, work_item
                )
        if row is None:
            return None
        from nro.orchestration.contracts import ExecutionEnvelope

        return ExecutionEnvelope.from_registry_row(work_item)

    @staticmethod
    def _completed_resource_steps_json(db: sqlite3.Connection, work_item: dict) -> str:
        """Encode resource steps completed in the work item's current generation."""
        return json.dumps(
            [
                str(row[0])
                for row in db.execute(
                    """SELECT step_id FROM resource_step_tasks
                       WHERE work_item_id=? AND generation=? AND revision_fingerprint=?
                         AND state='success'
                       ORDER BY id""",
                    (
                        int(work_item["id"]),
                        int(work_item["current_generation"]),
                        str(work_item["revision_fingerprint"]),
                    ),
                )
            ]
        )

    def defer_resource_step(
        self,
        attempt_id: int,
        *,
        step_id: str,
        resource_class: str,
        memory_gb: int = 32,
    ) -> int:
        """Yield a valid work-item attempt and queue one runner step for another worker."""
        if not step_id or resource_class != "gpu":
            raise ValueError("Resource handoffs require a step ID and the GPU resource class")
        if memory_gb < 1:
            raise ValueError("Resource handoff memory must be positive")
        now = utcnow()
        with self.connection(write=True) as db:
            work_item = db.execute(
                """SELECT i.* FROM attempts a JOIN work_items i ON i.id=a.work_item_id
                   WHERE a.id=?""",
                (attempt_id,),
            ).fetchone()
            if work_item is None:
                raise KeyError(f"Unknown attempt: {attempt_id}")
            dependency_state.check_completion(db, dict(work_item), attempt_id)
            cursor = db.execute(
                """INSERT INTO resource_step_tasks(
                       work_item_id,step_id,resource_class,memory_gb,generation,
                       revision_fingerprint,state,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,'pending',?,?)
                   ON CONFLICT(work_item_id,step_id,generation,revision_fingerprint)
                   DO UPDATE SET resource_class=excluded.resource_class,
                       memory_gb=excluded.memory_gb,state='pending',worker_id=NULL,
                       attempt_id=NULL,error_type=NULL,error_message=NULL,
                       completed_at=NULL,updated_at=excluded.updated_at
                   RETURNING id""",
                (
                    int(work_item["id"]),
                    step_id,
                    resource_class,
                    memory_gb,
                    int(work_item["current_generation"]),
                    str(work_item["revision_fingerprint"]),
                    now,
                    now,
                ),
            )
            task_id = int(cursor.fetchone()[0])
            db.execute(
                """UPDATE attempts SET state='success',completed_at=?,error_type='ResourceHandoff',
                       error_message=? WHERE id=?""",
                (now, f"Waiting for {resource_class} runner step {step_id}", attempt_id),
            )
            worker_id = db.execute(
                "SELECT worker_id FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()[0]
            if worker_id:
                db.execute(
                    "UPDATE workers SET state='idle',updated_at=? WHERE id=?", (now, worker_id)
                )
            return task_id

    def claim_resource_step(
        self,
        worker_id: str,
        *,
        resource_class: str,
        memory_gb: int,
    ) -> "ExecutionEnvelope | None":
        """Claim one ready runner step for the worker's exact resource class."""
        now = utcnow()
        with self.connection(write=True) as db:
            worker = db.execute("SELECT state FROM workers WHERE id=?", (worker_id,)).fetchone()
            if worker is None or worker["state"] == "shutdown_requested":
                return None
            if db.execute("SELECT 1 FROM metadata WHERE key='maintenance_mode'").fetchone():
                return None
            dependency_state.synchronize(db, now=now)
            concurrency_row = db.execute(
                "SELECT value FROM metadata WHERE key='gpu_concurrency'"
            ).fetchone()
            concurrency = int(concurrency_row[0]) if concurrency_row is not None else 1
            active = int(
                db.execute(
                    """SELECT COUNT(*) FROM resource_step_tasks
                       WHERE state='running' AND resource_class=?""",
                    (resource_class,),
                ).fetchone()[0]
            )
            if concurrency < 1 or active >= concurrency:
                return None
            row = db.execute(
                """SELECT i.*,ci.config_fingerprint,
                          task.id AS resource_task_id,task.step_id AS target_step_id
                   FROM resource_step_tasks task
                   JOIN work_items i ON i.id=task.work_item_id
                   JOIN module_lineages ci ON ci.id=i.module_lineage_id
                   WHERE task.resource_class=? AND task.memory_gb<=?
                     AND task.state IN ('pending','error')
                     AND task.generation=i.current_generation
                     AND task.revision_fingerprint=i.revision_fingerprint
                     AND i.artifact_state!='fresh'
                     AND NOT EXISTS (
                         SELECT 1 FROM attempts active WHERE active.work_item_id=i.id
                         AND active.state IN ('queued','running','cancel_requested')
                     )
                     AND EXISTS (
                         SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                         WHERE rt.work_item_id=i.id AND rt.demand_state='active' AND r.state='active'
                           AND (task.state='pending' OR r.updated_at>task.updated_at)
                     )
                   ORDER BY task.id LIMIT 1""",
                (resource_class, memory_gb),
            ).fetchone()
            if row is None:
                return None
            work_item = dict(row)
            work_item["completed_resource_steps_json"] = self._completed_resource_steps_json(
                db, work_item
            )
            execution = db.execute(
                "SELECT * FROM work_item_execution WHERE work_item_id=?", (work_item["id"],)
            ).fetchone()
            log_dir = self.paths.events / work_item_relative_directory(work_item)
            if execution is not None:
                from nro.orchestration.branch_admission import prepare_attempt

                log_dir = (
                    ControlPaths(self.paths.control).branch(execution["branch"])
                    / "events"
                    / work_item_relative_directory(work_item)
                )
                ensure_shared_directory(log_dir)
                command = prepare_attempt(self, db, work_item, dict(execution), log_dir)
                if command is None:
                    db.execute(
                        """UPDATE resource_step_tasks SET state='cancelled',completed_at=?,
                                  updated_at=?,error_type='ResourcePreconditionChanged',
                                  error_message='Branch inputs require local replanning'
                           WHERE id=?""",
                        (now, now, int(work_item["resource_task_id"])),
                    )
                    return None
                work_item["command_json"] = json.dumps(command)
            ensure_shared_directory(log_dir)
            cursor = db.execute(
                """INSERT INTO attempts(work_item_id,worker_id,state,revision_fingerprint,
                                         memory_gb,started_at,log_path,created_at)
                   VALUES (?,?,'running',?,?,?,?,?)""",
                (
                    int(work_item["id"]),
                    worker_id,
                    str(work_item["revision_fingerprint"]),
                    memory_gb,
                    now,
                    str(log_dir / f"resource-step-{int(work_item['resource_task_id'])}.log"),
                    now,
                ),
            )
            attempt_id = int(cursor.lastrowid)
            if execution is not None:
                payload = json.loads(Path(command[-3]).read_text())
                db.execute(
                    "INSERT INTO attempt_execution VALUES (?,?,?,?)",
                    (
                        attempt_id,
                        json.dumps(payload["context"]),
                        execution["provenance_json"],
                        work_item["command_json"],
                    ),
                )
            dependency_state.capture_inputs(db, attempt_id, int(work_item["id"]))
            db.execute(
                """UPDATE resource_step_tasks SET state='running',worker_id=?,attempt_id=?,
                       updated_at=? WHERE id=?""",
                (worker_id, attempt_id, now, int(work_item["resource_task_id"])),
            )
            db.execute(
                "UPDATE workers SET state='running',lease_expires_at=?,updated_at=? WHERE id=?",
                (time.time() + 120.0, now, worker_id),
            )
            work_item["attempt_id"] = attempt_id
            work_item["log_path"] = str(
                log_dir / f"resource-step-{int(work_item['resource_task_id'])}.log"
            )
            from nro.orchestration.contracts import ExecutionEnvelope

            return ExecutionEnvelope.from_registry_row(work_item)

    def finish_resource_step(
        self,
        task_id: int,
        attempt_id: int,
        *,
        state: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Publish a resource-step result under the ordinary attempt validation rules."""
        if state not in {"success", "error", "cancelled"}:
            raise ValueError(f"Invalid resource-step state: {state}")
        now = utcnow()
        with self.connection(write=True) as db:
            row = db.execute(
                """SELECT i.*,task.id AS resource_task_id,task.state AS resource_task_state
                   FROM resource_step_tasks task
                   JOIN work_items i ON i.id=task.work_item_id
                   WHERE task.id=? AND task.attempt_id=?""",
                (task_id, attempt_id),
            ).fetchone()
            if row is None:
                raise dependency_state.AttemptInvalidated(
                    "Resource-step assignment changed before publication"
                )
            if state == "success":
                dependency_state.check_completion(db, dict(row), attempt_id)
            attempt = db.execute(
                "SELECT worker_id,state FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if state == "success" and attempt and attempt["state"] == "cancel_requested":
                state = "cancelled"
            db.execute(
                """UPDATE attempts SET state=?,completed_at=?,error_type=COALESCE(?,error_type),
                       error_message=COALESCE(?,error_message) WHERE id=?""",
                (state, now, error_type, error_message, attempt_id),
            )
            db.execute(
                """UPDATE resource_step_tasks SET state=?,completed_at=?,updated_at=?,
                       error_type=?,error_message=? WHERE id=?""",
                (state, now, now, error_type, error_message, task_id),
            )
            if attempt and attempt["worker_id"]:
                db.execute(
                    "UPDATE workers SET state='idle',updated_at=? WHERE id=?",
                    (now, attempt["worker_id"]),
                )

    def attempt_cancel_requested(self, attempt_id: int) -> bool:
        """Return whether the current attempt has been marked for cancellation."""
        with self.read_connection() as db:
            row = db.execute("SELECT state FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            return bool(row and row["state"] == "cancel_requested")

    def record_attempt_process(self, attempt_id: int, process_group_id: int) -> None:
        """Record launch-in-progress (-1) or the supervised process group before polling."""
        with self.connection(write=True) as db:
            db.execute(
                "UPDATE attempts SET process_group_id=? WHERE id=?", (process_group_id, attempt_id)
            )

    def demanded_work_item_ids(self) -> tuple[int, ...]:
        """Return work items currently required by at least one active request."""
        with self.connection() as db:
            return tuple(
                int(row["work_item_id"])
                for row in db.execute(
                    """
                    SELECT DISTINCT rt.work_item_id
                    FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                    WHERE rt.demand_state='active' AND r.state='active'
                    ORDER BY rt.work_item_id
                    """
                )
            )

    def cancel_purged_demand(self, work_item_ids: Iterable[int]) -> int:
        """Withdraw requests that require artifacts selected for deletion."""
        from nro.orchestration.registry_work_items import cancel_purged_demand

        ids = tuple(sorted(set(work_item_ids)))
        with self.connection(write=True) as db:
            return cancel_purged_demand(db, ids, now=utcnow())

    def forget_purged_work_items(self, work_item_ids: Iterable[int]) -> tuple[int, tuple[int, ...]]:
        """Remove purged scheduler records that no surviving DAG still references."""
        from nro.orchestration.registry_work_items import forget_purged_work_items

        ids = tuple(sorted(set(work_item_ids)))
        with self.connection(write=True) as db:
            return forget_purged_work_items(db, ids)

    def reserve_artifact_assessment(
        self,
        *,
        minimum_interval: float = 30.0,
        lease_seconds: float = 300.0,
    ) -> bool:
        """Reserve one project-wide filesystem audit without holding the DB lock."""
        now = time.time()
        with self.connection(write=True) as db:
            values = {
                str(row["key"]): str(row["value"])
                for row in db.execute(
                    "SELECT key, value FROM metadata WHERE key IN "
                    "('artifact_assessment_lease_until', 'artifact_assessment_completed_at')"
                )
            }
            try:
                lease_until = float(values.get("artifact_assessment_lease_until", "0"))
            except ValueError:
                lease_until = 0.0
            try:
                completed_at = float(values.get("artifact_assessment_completed_at", "0"))
            except ValueError:
                completed_at = 0.0
            if lease_until > now or completed_at + minimum_interval > now:
                return False
            db.execute(
                """INSERT INTO metadata(key, value)
                   VALUES ('artifact_assessment_lease_until', ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (str(now + lease_seconds),),
            )
            return True

    def finish_artifact_assessment(self) -> None:
        """Release the audit lease and record when work-item states became authoritative."""
        now = time.time()
        with self.connection(write=True) as db:
            db.execute(
                """INSERT INTO metadata(key, value)
                   VALUES ('artifact_assessment_lease_until', '0')
                   ON CONFLICT(key) DO UPDATE SET value='0'"""
            )
            db.execute(
                """INSERT INTO metadata(key, value)
                   VALUES ('artifact_assessment_completed_at', ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (str(now),),
            )

    def cancel_attempts_with_stale_upstreams(self) -> list[dict]:
        """Request cancellation of active work that is now downstream of stale work.

        A worker allocation can execute only one work item at a time, so cancellation
        is deliberately attempt-scoped rather than a Slurm ``scancel`` of the
        entire worker.  The worker observes ``cancel_requested`` promptly and
        terminates just its child process before returning to the shared pool.
        """
        with self.connection(write=True) as db:
            return dependency_state.synchronize(db, now=utcnow())

    def cancel_attempts_downstream_of_failure(self, work_item_id: int) -> list[dict]:
        """Stop active descendant attempts after ``work_item_id`` has fatally failed.

        This is transitive rather than relying on each intermediate work item having
        already been reassessed.  It closes the short race in which an external
        change or a concurrent request allowed work from several DAG levels to
        be active when an ancestor fails.
        """
        with self.connection(write=True) as db:
            return dependency_state.invalidate(
                db,
                [work_item_id],
                now=utcnow(),
                reason=f"Resolved upstream work item failed: {work_item_id}",
                error_type="UpstreamFailed",
            )

    @contextlib.contextmanager
    def artifact_mutation(
        self, work_item_ids: Iterable[int], *, timeout: float = 30.0
    ) -> Iterator[None]:
        """Reserve outputs for deletion or replacement after cancelling their consumers.

        Target attempts must already be stopped. Other claims remain available
        while consumers shut down. A timeout performs no caller writes, leaves
        invalidation in place, and releases the reservation. After a hard crash,
        a later mutation recovers the filesystem lock before clearing old tokens.
        """
        ids = tuple(sorted(set(work_item_ids)))
        if not ids:
            yield
            return
        self._prepare_directories()
        scheduler = ControlPaths(self.paths.control).scheduler
        placeholders = ",".join("?" for _ in ids)
        token = uuid.uuid4().hex
        with RegistryLock(
            scheduler / "artifact-mutation.lock",
            scheduler / "artifact-mutation.recovery-lock",
            lease_seconds=300.0,
        ):
            with self.connection(write=True) as db:
                if db.execute(
                    f"SELECT 1 FROM attempts WHERE work_item_id IN ({placeholders}) AND state IN {dependency_state.ACTIVE}",
                    ids,
                ).fetchone():
                    raise RuntimeError(
                        "Stop target attempts before replacing or purging their outputs"
                    )
                db.execute("DELETE FROM artifact_mutations")
                db.executemany(
                    "INSERT INTO artifact_mutations VALUES (?,?)", [(item, token) for item in ids]
                )
                db.execute(
                    f"UPDATE work_items SET artifact_state='stale', artifact_reason='Outputs reserved for mutation', updated_at=? WHERE id IN ({placeholders})",
                    (utcnow(), *ids),
                )
                dependency_state.invalidate(
                    db,
                    ids,
                    now=utcnow(),
                    reason="A resolved upstream output is being replaced or purged",
                )
            try:
                deadline = time.monotonic() + timeout
                while True:
                    with self.connection() as db:
                        active = db.execute(
                            f"""SELECT 1 FROM attempt_dependencies pinned
                            JOIN attempts reader ON reader.id=pinned.attempt_id
                            WHERE pinned.upstream_work_item_id IN ({placeholders})
                              AND reader.state IN {dependency_state.ACTIVE} LIMIT 1""",
                            ids,
                        ).fetchone()
                    if not active:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Dependent attempts have not confirmed shutdown; no outputs were removed. Retry after they stop."
                        )
                    self.recover_orphaned_attempts()
                    time.sleep(0.1)
                yield
            finally:
                with self.connection(write=True) as db:
                    db.execute("DELETE FROM artifact_mutations WHERE token=?", (token,))

    def finish_attempt(
        self,
        attempt_id: int,
        *,
        state: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Persist attempt completion and update the associated work-item state."""
        if state not in {"success", "error", "cancelled"}:
            raise ValueError(f"Invalid terminal attempt state: {state}")
        with self.connection(write=True) as db:
            row = db.execute(
                "SELECT worker_id, state FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if state == "success" and row and row["state"] == "cancel_requested":
                state = "cancelled"
            db.execute(
                """
                UPDATE attempts SET state=?, completed_at=?,
                    error_type=COALESCE(?, error_type),
                    error_message=COALESCE(?, error_message)
                WHERE id=?
                """,
                (state, utcnow(), error_type, error_message, attempt_id),
            )
            if row and row["worker_id"]:
                db.execute(
                    "UPDATE workers SET state='idle', updated_at=? WHERE id=?",
                    (utcnow(), row["worker_id"]),
                )
            if state == "success":
                db.execute(
                    """
                    UPDATE requests SET state='satisfied', updated_at=?
                    WHERE state='active' AND NOT EXISTS (
                        SELECT 1 FROM request_work_items rt JOIN work_items t ON t.id=rt.work_item_id
                        WHERE rt.request_id=requests.id AND rt.role='target' AND t.artifact_state!='fresh'
                    )
                    """,
                    (utcnow(),),
                )

    @staticmethod
    def _record_oom_locked(
        db: sqlite3.Connection,
        attempt_id: int,
        *,
        message: str,
    ) -> int | None:
        row = db.execute(
            """
            SELECT a.work_item_id, a.worker_id, a.memory_gb AS attempt_memory,a.oom_detected,
                   t.memory_gb, t.max_memory_gb
            FROM attempts a JOIN work_items t ON t.id=a.work_item_id WHERE a.id=?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown attempt: {attempt_id}")
        if row["oom_detected"]:
            return (
                int(row["memory_gb"])
                if int(row["memory_gb"]) > int(row["attempt_memory"])
                else None
            )
        completed = utcnow()
        db.execute(
            """
            UPDATE attempts SET state='error', oom_detected=1, completed_at=?,
                error_type='OutOfMemory', error_message=? WHERE id=?
            """,
            (completed, message, attempt_id),
        )
        current = max(int(row["memory_gb"]), int(row["attempt_memory"]))
        limit = int(row["max_memory_gb"])
        next_memory = min(limit, current * 2) if current < limit else None
        if next_memory is not None:
            db.execute(
                """
                UPDATE work_items SET memory_gb=?, artifact_state='stale', artifact_reason=?, updated_at=?
                WHERE id=?
                """,
                (
                    next_memory,
                    f"OOM at {current} GB; retrying at {next_memory} GB",
                    utcnow(),
                    row["work_item_id"],
                ),
            )
            db.execute(
                """
                UPDATE requests SET updated_at=? WHERE state='active' AND id IN (
                    SELECT request_id FROM request_work_items
                    WHERE work_item_id=? AND demand_state='active'
                )
                """,
                (utcnow(), row["work_item_id"]),
            )
        else:
            db.execute(
                "UPDATE work_items SET artifact_reason=?, updated_at=? WHERE id=?",
                (f"OOM at configured ceiling of {limit} GB", utcnow(), row["work_item_id"]),
            )
        if row["worker_id"]:
            db.execute(
                "UPDATE workers SET state='idle', updated_at=? WHERE id=?",
                (utcnow(), row["worker_id"]),
            )
        return next_memory

    def record_oom(self, attempt_id: int, *, message: str) -> int | None:
        """Record an out-of-memory failure and update retry memory requirements."""
        with self.connection(write=True) as db:
            return self._record_oom_locked(db, attempt_id, message=message)

    def request_cancellation(
        self,
        *,
        participants: Sequence[str] = (),
        modules: Sequence[str] = (),
        workflows: Sequence[str] = (),
        lineages: Sequence[str] = (),
        selectors: dict[str, Sequence[str] | str | None] | None = None,
        include_dependents: bool = True,
        user_name: str | None = None,
        force: bool = False,
        branch_registry_id: str | None = None,
    ) -> dict[str, int]:
        """Cancel matching demand and signal attempts no longer needed by any request.

        Normally only requests owned by ``user_name`` are affected. ``force``
        deliberately removes matching demand from every user's active request.
        """
        from nro.engine.cli import matches_module_lineage

        participant_set = {value.removeprefix("sub-") for value in participants}
        module_set = set(modules)
        workflow_set = set(workflows)
        lineage_set = set(lineages)
        selectors = selectors or {}
        owner = user_name or getpass.getuser()
        with self.connection(write=True) as db:
            branch_requests = (
                None
                if branch_registry_id is None
                else {
                    row[0]
                    for row in db.execute(
                        "SELECT request_id FROM request_owners WHERE registry_id=?",
                        (branch_registry_id,),
                    )
                }
            )
            branch_work_items = (
                None
                if branch_registry_id is None
                else {
                    row[0]
                    for row in db.execute(
                        "SELECT work_item_id FROM branch_work_items WHERE registry_id=?",
                        (branch_registry_id,),
                    )
                }
            )
            eligible_requests = {
                str(row["id"])
                for row in db.execute(
                    """
                    SELECT r.id, r.user_name, wr.workflow_id FROM requests r
                    JOIN workflow_revisions wr ON wr.id=r.workflow_revision_id
                    WHERE r.state='active' AND r.project=?
                    """,
                    (self.paths.project,),
                )
                if (force or str(row["user_name"]) == owner)
                and (branch_requests is None or row["id"] in branch_requests)
                and (
                    not workflow_set
                    or str(row["workflow_id"]).removeprefix((branch_registry_id or "") + ":")
                    in workflow_set
                )
            }
            if not eligible_requests:
                return {"work_items": 0, "requests": 0, "attempts": 0}
            selected = {
                int(row["id"])
                for row in db.execute(
                    """SELECT t.id, t.participant, t.module, t.entities_json,
                              ci.directory_label
                       FROM work_items t JOIN module_lineages ci
                         ON ci.id=t.module_lineage_id
                       WHERE t.project=?""",
                    (self.paths.project,),
                )
                if (not participant_set or row["participant"] in participant_set)
                and (not module_set or row["module"] in module_set)
                and matches_module_lineage(row["module"], row["directory_label"], lineage_set)
                and matches_selectors(json.loads(row["entities_json"]), selectors)
                and (branch_work_items is None or row["id"] in branch_work_items)
            }
            if include_dependents:
                changed = True
                while changed:
                    before = len(selected)
                    for edge in db.execute(
                        "SELECT work_item_id, upstream_work_item_id FROM work_item_dependencies"
                    ):
                        if int(edge["upstream_work_item_id"]) in selected:
                            selected.add(int(edge["work_item_id"]))
                    changed = len(selected) != before
            if not selected:
                return {"work_items": 0, "requests": 0, "attempts": 0}
            work_item_placeholders = ",".join("?" for _ in selected)
            request_placeholders = ",".join("?" for _ in eligible_requests)
            values = (*tuple(selected), *tuple(eligible_requests))
            affected_requests = {
                str(row["request_id"])
                for row in db.execute(
                    f"""SELECT DISTINCT request_id FROM request_work_items
                        WHERE work_item_id IN ({work_item_placeholders})
                          AND request_id IN ({request_placeholders})
                          AND demand_state='active'""",
                    values,
                )
            }
            if not affected_requests:
                return {"work_items": 0, "requests": 0, "attempts": 0}
            cursor = db.execute(
                f"""UPDATE request_work_items SET demand_state='cancelled'
                    WHERE work_item_id IN ({work_item_placeholders})
                      AND request_id IN ({request_placeholders})
                      AND demand_state='active'""",
                values,
            )
            demand_count = cursor.rowcount
            cancelled_requests: list[str] = []
            edges = [
                (int(row["work_item_id"]), int(row["upstream_work_item_id"]))
                for row in db.execute(
                    "SELECT work_item_id, upstream_work_item_id FROM work_item_dependencies"
                )
            ]
            for request_id in affected_requests:
                active_targets = {
                    int(row["work_item_id"])
                    for row in db.execute(
                        """SELECT work_item_id FROM request_work_items
                           WHERE request_id=? AND role='target' AND demand_state='active'""",
                        (request_id,),
                    )
                }
                if not active_targets:
                    pruned = db.execute(
                        "UPDATE request_work_items SET demand_state='cancelled' WHERE request_id=? AND demand_state='active'",
                        (request_id,),
                    ).rowcount
                    demand_count += pruned
                    db.execute(
                        "UPDATE requests SET state='cancelled', updated_at=? WHERE id=?",
                        (utcnow(), request_id),
                    )
                    cancelled_requests.append(request_id)
                    continue
                required = set(active_targets)
                changed = True
                while changed:
                    before = len(required)
                    for work_item_id, upstream_id in edges:
                        if work_item_id in required:
                            required.add(upstream_id)
                    changed = len(required) != before
                active_work_items = {
                    int(row["work_item_id"])
                    for row in db.execute(
                        "SELECT work_item_id FROM request_work_items WHERE request_id=? AND demand_state='active'",
                        (request_id,),
                    )
                }
                orphaned = active_work_items - required
                if orphaned:
                    orphan_placeholders = ",".join("?" for _ in orphaned)
                    demand_count += db.execute(
                        f"""UPDATE request_work_items SET demand_state='cancelled'
                            WHERE request_id=? AND work_item_id IN ({orphan_placeholders})
                              AND demand_state='active'""",
                        (request_id, *tuple(orphaned)),
                    ).rowcount
            cursor = db.execute(
                """
                UPDATE attempts SET state='cancel_requested',
                    error_type='UserCancelled',
                    error_message='Cancellation requested directly by a user'
                WHERE state IN ('queued', 'running')
                  AND NOT EXISTS (
                      SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.work_item_id=attempts.work_item_id AND rt.demand_state='active' AND r.state='active'
                  )
                """
            )
            db.execute(
                """UPDATE resource_step_tasks SET state='cancelled',completed_at=?,updated_at=?,
                           error_type='UserCancelled',
                           error_message='Demand was cancelled before the resource step ran'
                    WHERE state='pending' AND NOT EXISTS (
                        SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                        WHERE rt.work_item_id=resource_step_tasks.work_item_id
                          AND rt.demand_state='active' AND r.state='active'
                    )""",
                (utcnow(), utcnow()),
            )
            return {
                "work_items": demand_count,
                "requests": len(cancelled_requests),
                "attempts": cursor.rowcount,
            }

    def reconcile_requests(self) -> None:
        """Update request states from their demanded work items' current outcomes."""
        with self.connection(write=True) as db:
            db.execute(
                """
                UPDATE requests SET state='satisfied', updated_at=?
                WHERE state='active' AND NOT EXISTS (
                    SELECT 1 FROM request_work_items rt JOIN work_items t ON t.id=rt.work_item_id
                    WHERE rt.request_id=requests.id AND rt.role='target'
                      AND rt.demand_state='active' AND t.artifact_state!='fresh'
                )
                """,
                (utcnow(),),
            )
            # A request may contain many independent branches (for example,
            # one functional run per acquisition).  A failed branch is
            # represented by its work item and attempt and blocks only its descendants;
            # it must not deactivate demand for unrelated ready branches.
            # Failed work items remain non-retryable until a later orchestration run
            # refreshes demand, per the claim predicate above.

    def reserve_worker_submissions(
        self,
        *,
        request_id: str | None,
        resource_class: str,
        memory_gb: int = 32,
        minimum_memory_gb: int = 0,
    ) -> list[tuple[int, str]]:
        """Reserve workers for one memory tier of ready derivative work."""
        return self._worker_submission_plan(
            request_id=request_id,
            resource_class=resource_class,
            memory_gb=memory_gb,
            minimum_memory_gb=minimum_memory_gb,
            reserve=True,
        )

    def worker_capacity_needed(
        self,
        *,
        request_id: str | None,
        resource_class: str,
        memory_gb: int = 32,
        minimum_memory_gb: int = 0,
    ) -> bool:
        """Return whether ready work needs another compatible worker."""
        return bool(
            self._worker_submission_plan(
                request_id=request_id,
                resource_class=resource_class,
                memory_gb=memory_gb,
                minimum_memory_gb=minimum_memory_gb,
                reserve=False,
            )
        )

    def _worker_submission_plan(
        self,
        *,
        request_id: str | None,
        resource_class: str,
        memory_gb: int,
        minimum_memory_gb: int,
        reserve: bool,
    ) -> list[tuple[int, str]]:
        """Compute needed capacity and optionally create its submission records."""
        if minimum_memory_gb < 0 or minimum_memory_gb >= memory_gb:
            raise ValueError("Worker memory tier bounds must satisfy 0 <= minimum < maximum")
        with self.connection(write=True) as db:
            if db.execute("SELECT 1 FROM metadata WHERE key='maintenance_mode'").fetchone():
                return []
            dependency_state.synchronize(db, now=utcnow())
            from nro.bidsify.index import IngestionIndex

            supports_ingestion = resource_class == "large"
            ingestion_active, ingestion_ready, ingestion_limit = (
                IngestionIndex(self).summary(memory_gb) if supports_ingestion else (0, 0, 0)
            )
            if supports_ingestion and minimum_memory_gb:
                _, lower_ingestion_ready, _ = IngestionIndex(self).summary(minimum_memory_gb)
                ingestion_active = 0
                ingestion_ready = max(0, ingestion_ready - lower_ingestion_ready)
            if request_id is None:
                request = db.execute(
                    "SELECT id FROM requests WHERE state='active' ORDER BY created_at LIMIT 1"
                ).fetchone()
                if request is None and not ingestion_ready:
                    return []
                request_id = str(request["id"]) if request else None
            desired = int(
                db.execute(
                    "SELECT COALESCE(MAX(concurrency), 0) FROM requests WHERE state='active'"
                ).fetchone()[0]
            )
            if resource_class == "gpu":
                gpu_limit = db.execute(
                    "SELECT value FROM metadata WHERE key='gpu_concurrency'"
                ).fetchone()
                desired = int(gpu_limit[0]) if gpu_limit is not None else 1
            compatible = compatible_work_item_classes(resource_class)
            placeholders = ",".join("?" for _ in compatible)
            active_work_items = int(
                db.execute(
                    f"""SELECT COUNT(*) FROM attempts a
                        JOIN work_items t ON t.id=a.work_item_id
                        WHERE a.state IN ('queued', 'running', 'cancel_requested')
                          AND t.resource_class IN ({placeholders})
                          AND t.memory_gb>? AND t.memory_gb<=?
                          AND NOT EXISTS (
                              SELECT 1 FROM resource_step_tasks task WHERE task.attempt_id=a.id
                          )""",
                    (*compatible, minimum_memory_gb, memory_gb),
                ).fetchone()[0]
            )
            ready_work_items = int(
                db.execute(
                    f"""
                    SELECT COUNT(*) FROM work_items t
                    WHERE t.resource_class IN ({placeholders})
                      AND t.memory_gb>? AND t.memory_gb<=?
                      AND {dependency_state.WRITE_READY}
                      AND t.artifact_state!='fresh'
                      AND NOT (
                          t.module IN ('dynconn', 'microparcellation')
                          AND t.artifact_reason LIKE 'Selected raw run universe changed:%'
                      )
                      AND EXISTS (
                          SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                          WHERE rt.work_item_id=t.id AND rt.demand_state='active' AND r.state='active'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM work_item_dependencies td JOIN work_items up ON up.id=td.upstream_work_item_id
                          WHERE td.work_item_id=t.id AND up.artifact_state!='fresh'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM attempts a WHERE a.work_item_id=t.id
                          AND a.state IN ('queued', 'running', 'cancel_requested')
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM resource_step_tasks task WHERE task.work_item_id=t.id
                          AND task.generation=t.current_generation
                          AND task.revision_fingerprint=t.revision_fingerprint
                          AND task.state IN ('pending','running','error')
                      )
                      AND (
                          NOT EXISTS (SELECT 1 FROM attempts old WHERE old.work_item_id=t.id)
                          OR COALESCE((SELECT state FROM attempts old WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1), '') = 'success'
                          OR COALESCE((SELECT error_type FROM attempts old
                                      WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1), '')
                             IN ('UpstreamStale', 'UpstreamFailed', 'WorkerTerminated',
                                 'WorkItemGraphChanged', 'RegistryUnavailable')
                          OR EXISTS (
                              SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                              WHERE rt.work_item_id=t.id AND rt.demand_state='active' AND r.state='active'
                              AND r.updated_at > COALESCE(
                                  (SELECT CASE WHEN old.state='cancelled'
                                           THEN old.started_at ELSE old.completed_at END
                                   FROM attempts old WHERE old.work_item_id=t.id ORDER BY id DESC LIMIT 1),
                                  '')
                          )
                      )
                    """,
                    (*compatible, minimum_memory_gb, memory_gb),
                ).fetchone()[0]
            )
            if resource_class == "gpu":
                active_work_items = int(
                    db.execute(
                        """SELECT COUNT(*) FROM resource_step_tasks
                           WHERE state='running' AND resource_class=? AND memory_gb>? AND memory_gb<=?""",
                        (resource_class, minimum_memory_gb, memory_gb),
                    ).fetchone()[0]
                )
                ready_work_items = int(
                    db.execute(
                        """SELECT COUNT(*) FROM resource_step_tasks task
                           JOIN work_items i ON i.id=task.work_item_id
                           WHERE task.resource_class=? AND task.memory_gb>? AND task.memory_gb<=?
                             AND task.state IN ('pending','error')
                             AND task.generation=i.current_generation
                             AND task.revision_fingerprint=i.revision_fingerprint
                             AND i.artifact_state!='fresh'
                             AND EXISTS (
                                 SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                                 WHERE rt.work_item_id=i.id AND rt.demand_state='active'
                                   AND r.state='active'
                                   AND (task.state='pending' OR r.updated_at>task.updated_at)
                             )""",
                        (resource_class, minimum_memory_gb, memory_gb),
                    ).fetchone()[0]
                )
            global_limit = max(desired, ingestion_limit)
            desired = min(
                global_limit,
                active_work_items + ready_work_items + ingestion_active + ingestion_ready,
            )
            live_workers = int(
                db.execute(
                    """SELECT COUNT(*) FROM workers
                       WHERE state IN ('idle', 'running') AND lease_expires_at>?
                         AND resource_class=? AND memory_gb>=?""",
                    (time.time(), resource_class, memory_gb),
                ).fetchone()[0]
            )
            pending = int(
                db.execute(
                    """SELECT COUNT(*) FROM scheduler_submissions
                       WHERE state IN ('prepared', 'submitted')
                         AND predecessor_worker_id IS NULL
                         AND resource_class=? AND memory_gb>=?""",
                    (resource_class, memory_gb),
                ).fetchone()[0]
            )
            pool_workers = int(
                db.execute(
                    """SELECT COUNT(*) FROM workers
                       WHERE state IN ('idle', 'running') AND lease_expires_at>?
                         AND resource_class=?""",
                    (time.time(), resource_class),
                ).fetchone()[0]
            )
            pool_pending = int(
                db.execute(
                    """SELECT COUNT(*) FROM scheduler_submissions
                       WHERE state IN ('prepared', 'submitted')
                         AND predecessor_worker_id IS NULL AND resource_class=?""",
                    (resource_class,),
                ).fetchone()[0]
            )
            available = max(0, global_limit - pool_workers - pool_pending)
            count = min(max(0, desired - live_workers - pending), available)
            if ingestion_ready:
                # Cloud credentials belong to the submitting user. Idle
                # workers belonging to someone else cannot fulfill this demand.
                own_idle = db.execute(
                    "SELECT COUNT(*) FROM workers WHERE state='idle' AND user_name=? AND lease_expires_at>? AND memory_gb>=?",
                    (getpass.getuser(), time.time(), memory_gb),
                ).fetchone()[0]
                count = max(
                    count,
                    min(
                        ingestion_ready,
                        max(
                            0,
                            max(desired, ingestion_limit)
                            - active_work_items
                            - ingestion_active
                            - pending,
                        ),
                    )
                    - own_idle,
                )
            reservations: list[tuple[int, str]] = []
            if not reserve:
                return [(0, "")] if count else []
            for _ in range(count):
                token = uuid.uuid4().hex
                cursor = db.execute(
                    """
                    INSERT INTO scheduler_submissions(
                        intent_token, request_id, resource_class, memory_gb, state, created_at
                    ) VALUES (?, ?, ?, ?, 'prepared', ?)
                    """,
                    (token, request_id, resource_class, memory_gb, utcnow()),
                )
                reservations.append((int(cursor.lastrowid), token))
                if ingestion_ready:
                    db.execute(
                        "INSERT INTO metadata(key,value) VALUES (?,?)",
                        (f"ingestion_submission_owner:{cursor.lastrowid}", getpass.getuser()),
                    )
            return reservations

    def reconcile_scheduler_submissions(self, *, prepared_timeout: float = 300.0) -> int:
        """Clear abandoned submission intents and terminal Slurm allocations."""
        with self.connection() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT id, state, slurm_job_id, created_at FROM scheduler_submissions WHERE state IN ('prepared', 'submitted', 'running')"
                )
            ]
        changes: list[tuple[str, int]] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            if row["state"] == "prepared":
                try:
                    age = (now - datetime.fromisoformat(row["created_at"])).total_seconds()
                except (TypeError, ValueError):
                    age = prepared_timeout + 1
                if age > prepared_timeout:
                    changes.append(("error", int(row["id"])))
            elif (
                row.get("slurm_job_id")
                and RegistryLock._slurm_terminal(str(row["slurm_job_id"])) is True
            ):
                changes.append(("complete", int(row["id"])))
        if changes:
            with self.connection(write=True) as db:
                db.executemany(
                    "UPDATE scheduler_submissions SET state=? WHERE id=?",
                    changes,
                )
        return len(changes)

    def reconcile_attempt_timeouts(self) -> int:
        """Classify SIGTERM failures whose Slurm allocations reached wall time."""
        with self.connection() as db:
            candidates = [
                dict(row)
                for row in db.execute(
                    """
                    SELECT attempt.id, worker.slurm_job_id
                    FROM attempts attempt
                    JOIN workers worker ON worker.id=attempt.worker_id
                    WHERE attempt.state='error'
                      AND attempt.error_type='CalledProcessError'
                      AND attempt.error_message LIKE 'Derivative command exited with status -15:%'
                      AND worker.slurm_job_id IS NOT NULL
                    """
                )
            ]
        timed_out = [
            row
            for row in candidates
            if RegistryLock._slurm_timed_out(str(row["slurm_job_id"])) is True
        ]
        if not timed_out:
            return 0
        with self.connection(write=True) as db:
            db.executemany(
                """
                UPDATE attempts SET error_type='Timeout', error_message=?
                WHERE id=? AND error_type='CalledProcessError'
                """,
                (
                    (
                        f"Worker allocation {row['slurm_job_id']} reached its Slurm wall-time "
                        "limit; resume this work with a longer --time allocation",
                        row["id"],
                    )
                    for row in timed_out
                ),
            )
        return len(timed_out)

    def reserve_worker_successor(
        self,
        *,
        worker_id: str,
        resource_class: str,
        memory_gb: int = 32,
    ) -> tuple[int, str] | None:
        """Persist one successor intent for a live worker before calling sbatch."""
        with self.connection(write=True) as db:
            worker = db.execute(
                "SELECT successor_submission_id FROM workers WHERE id=?",
                (worker_id,),
            ).fetchone()
            if worker is None or worker["successor_submission_id"] is not None:
                return None
            request = db.execute(
                "SELECT id FROM requests WHERE state='active' ORDER BY created_at LIMIT 1"
            ).fetchone()
            from nro.bidsify.index import IngestionIndex

            ingestion_active, ingestion_ready, _ = IngestionIndex(self).summary(memory_gb)
            if request is None and not (ingestion_active or ingestion_ready):
                return None
            token = uuid.uuid4().hex
            cursor = db.execute(
                """
                INSERT INTO scheduler_submissions(
                    intent_token, request_id, predecessor_worker_id, resource_class,
                    memory_gb, state, created_at
                ) VALUES (?, ?, ?, ?, ?, 'prepared', ?)
                """,
                (
                    token,
                    request["id"] if request else None,
                    worker_id,
                    resource_class,
                    memory_gb,
                    utcnow(),
                ),
            )
            submission_id = int(cursor.lastrowid)
            db.execute(
                "UPDATE workers SET successor_submission_id=?, updated_at=? WHERE id=?",
                (submission_id, utcnow(), worker_id),
            )
            return submission_id, token

    def required_memory_above(
        self, memory_gb: int, *, resource_classes: Sequence[str] = ()
    ) -> int | None:
        """Return the smallest ready work-item tier this worker cannot satisfy."""
        if not resource_classes:
            return None
        placeholders = ",".join("?" for _ in resource_classes)
        with self.connection() as db:
            row = db.execute(
                f"""
                SELECT MIN(t.memory_gb) AS memory_gb FROM work_items t
                WHERE t.memory_gb> ? AND t.artifact_state!='fresh'
                  AND t.resource_class IN ({placeholders})
                  AND {dependency_state.WRITE_READY}
                  AND EXISTS (
                      SELECT 1 FROM request_work_items rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.work_item_id=t.id AND rt.demand_state='active' AND r.state='active'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM work_item_dependencies td JOIN work_items up ON up.id=td.upstream_work_item_id
                      WHERE td.work_item_id=t.id AND up.artifact_state!='fresh'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM attempts a WHERE a.work_item_id=t.id
                      AND a.state IN ('queued', 'running', 'cancel_requested')
                  )
                """,
                (memory_gb, *resource_classes),
            ).fetchone()
            return int(row["memory_gb"]) if row and row["memory_gb"] is not None else None

    def reserve_adaptive_worker(
        self,
        *,
        resource_class: str,
        memory_gb: int,
    ) -> tuple[int, str] | None:
        """Reserve one higher-memory worker if no capable allocation exists."""
        with self.connection(write=True) as db:
            request = db.execute(
                "SELECT id, concurrency FROM requests WHERE state='active' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if request is None:
                return None
            capable_worker = db.execute(
                """
                SELECT 1 FROM workers WHERE state IN ('idle', 'running')
                  AND lease_expires_at>? AND resource_class=? AND memory_gb>=? LIMIT 1
                """,
                (time.time(), resource_class, memory_gb),
            ).fetchone()
            capable_submission = db.execute(
                """
                SELECT 1 FROM scheduler_submissions
                WHERE state IN ('prepared', 'submitted')
                  AND resource_class=? AND memory_gb>=? LIMIT 1
                """,
                (resource_class, memory_gb),
            ).fetchone()
            if capable_worker or capable_submission:
                return None
            token = uuid.uuid4().hex
            cursor = db.execute(
                """
                INSERT INTO scheduler_submissions(
                    intent_token, request_id, resource_class, memory_gb, state, created_at
                ) VALUES (?, ?, ?, ?, 'prepared', ?)
                """,
                (token, request["id"], resource_class, memory_gb, utcnow()),
            )
            return int(cursor.lastrowid), token

    def recover_orphaned_attempts(self) -> int:
        """Release leases only when a worker process/allocation is definitively gone."""
        with self.connection() as db:
            expired = [
                dict(row)
                for row in db.execute(
                    """
                    SELECT * FROM workers
                    WHERE (state IN ('idle', 'running', 'draining')
                      AND lease_expires_at IS NOT NULL AND lease_expires_at<?)
                      OR (state IN ('exited','terminated','lost') AND EXISTS (
                          SELECT 1 FROM attempts WHERE worker_id=workers.id
                          AND state IN ('queued','running','cancel_requested')))
                    """,
                    (time.time(),),
                )
            ]
        dead: list[str] = []
        oom_workers: set[str] = set()
        timed_out_workers: set[str] = set()
        local_host = socket.gethostname()
        for worker in expired:
            is_dead = False
            if worker["hostname"] == local_host:
                try:
                    os.kill(int(worker["pid"]), 0)
                except ProcessLookupError:
                    is_dead = True
                except (PermissionError, OSError, ValueError):
                    pass
            if not is_dead and worker.get("slurm_job_id"):
                is_dead = RegistryLock._slurm_terminal(str(worker["slurm_job_id"])) is True
            if is_dead:
                with self.connection() as db:
                    groups = [
                        int(row[0])
                        for row in db.execute(
                            f"SELECT process_group_id FROM attempts WHERE worker_id=? AND state IN {dependency_state.ACTIVE}",
                            (worker["id"],),
                        )
                    ]
                from nro.orchestration.execution import ShutdownUnconfirmed, process_group_alive

                allocation_ended = (
                    bool(worker.get("slurm_job_id"))
                    and RegistryLock._slurm_terminal(str(worker["slurm_job_id"])) is True
                )
                try:
                    if not allocation_ended and any(
                        group == -1
                        or (
                            group > 0
                            and (worker["hostname"] != local_host or process_group_alive(group))
                        )
                        for group in groups
                    ):
                        continue
                except ShutdownUnconfirmed:
                    continue
            if is_dead:
                worker_id = str(worker["id"])
                dead.append(worker_id)
                if (
                    worker.get("slurm_job_id")
                    and RegistryLock._slurm_out_of_memory(str(worker["slurm_job_id"])) is True
                ):
                    oom_workers.add(worker_id)
                elif (
                    worker.get("slurm_job_id")
                    and RegistryLock._slurm_timed_out(str(worker["slurm_job_id"])) is True
                ):
                    timed_out_workers.add(worker_id)
        if not dead:
            return 0
        recovered = 0
        with self.connection(write=True) as db:
            for worker_id in dead:
                from nro.bidsify.index import IngestionIndex

                recovered += IngestionIndex(self).recover_locked({worker_id})
                IngestionIndex(self).clear_publication_barriers_locked(db)
                attempts = db.execute(
                    "SELECT id, work_item_id FROM attempts WHERE worker_id=? AND state IN ('queued', 'running', 'cancel_requested')",
                    (worker_id,),
                ).fetchall()
                for attempt in attempts:
                    resource_task = db.execute(
                        "SELECT id FROM resource_step_tasks WHERE attempt_id=? AND state='running'",
                        (attempt["id"],),
                    ).fetchone()
                    if worker_id in oom_workers:
                        message = (
                            f"Slurm reported OUT_OF_MEMORY for worker job "
                            f"{next(item['slurm_job_id'] for item in expired if item['id'] == worker_id)}"
                        )
                        if resource_task is None:
                            self._record_oom_locked(
                                db,
                                int(attempt["id"]),
                                message=message,
                            )
                        else:
                            db.execute(
                                """UPDATE attempts SET state='error',completed_at=?,
                                          error_type='OutOfMemory',error_message=? WHERE id=?""",
                                (utcnow(), message, attempt["id"]),
                            )
                        db.execute(
                            """UPDATE resource_step_tasks SET state='error',completed_at=?,updated_at=?,
                                      error_type='OutOfMemory',error_message=?
                               WHERE attempt_id=? AND state='running'""",
                            (utcnow(), utcnow(), message, attempt["id"]),
                        )
                        recovered += 1
                        continue
                    if worker_id in timed_out_workers:
                        job_id = next(
                            item["slurm_job_id"] for item in expired if item["id"] == worker_id
                        )
                        db.execute(
                            """
                            UPDATE attempts SET state='error', completed_at=?, error_type='Timeout',
                                error_message=? WHERE id=?
                            """,
                            (
                                utcnow(),
                                f"Worker allocation {job_id} reached its Slurm wall-time limit; "
                                "resume this work with a longer --time allocation",
                                attempt["id"],
                            ),
                        )
                        db.execute(
                            """UPDATE resource_step_tasks SET state='error',completed_at=?,updated_at=?,
                                      error_type='Timeout',error_message=?
                               WHERE attempt_id=? AND state='running'""",
                            (
                                utcnow(),
                                utcnow(),
                                f"Worker allocation {job_id} reached its Slurm wall-time limit",
                                attempt["id"],
                            ),
                        )
                        recovered += 1
                        continue
                    completed = utcnow()
                    db.execute(
                        """
                        UPDATE attempts SET state='error', completed_at=?, error_type='WorkerLost',
                            error_message='Worker lease expired and its process/allocation is terminal'
                        WHERE id=?
                        """,
                        (completed, attempt["id"]),
                    )
                    db.execute(
                        """UPDATE resource_step_tasks SET state='error',completed_at=?,updated_at=?,
                                  error_type='WorkerLost',error_message=?
                           WHERE attempt_id=? AND state='running'""",
                        (
                            completed,
                            completed,
                            "Worker lease expired during resource-specific runner step",
                            attempt["id"],
                        ),
                    )
                    # This is an interrupted attempt, not a scientific failure.
                    # Renew existing demand so a successor may resume it.
                    db.execute(
                        """
                        UPDATE requests SET updated_at=? WHERE state='active' AND id IN (
                            SELECT request_id FROM request_work_items
                            WHERE work_item_id=? AND demand_state='active'
                        )
                        """,
                        (utcnow(), attempt["work_item_id"]),
                    )
                    recovered += 1
                db.execute(
                    "UPDATE workers SET state='lost', lease_expires_at=NULL, updated_at=? WHERE id=?",
                    (utcnow(), worker_id),
                )
        return recovered

    def update_submission(
        self,
        submission_id: int,
        *,
        state: str,
        slurm_job_id: str | None = None,
    ) -> None:
        """Persist a scheduler submission state and optional Slurm job ID."""
        with self.connection(write=True) as db:
            db.execute(
                """
                UPDATE scheduler_submissions SET state=?, slurm_job_id=COALESCE(?, slurm_job_id),
                    submitted_at=CASE WHEN ?='submitted' THEN ? ELSE submitted_at END
                WHERE id=?
                """,
                (state, slurm_job_id, state, utcnow(), submission_id),
            )

    def mark_submission_running(self, slurm_job_id: str) -> None:
        """Mark the submitted allocation matching a Slurm job ID as running."""
        with self.connection(write=True) as db:
            db.execute(
                "UPDATE scheduler_submissions SET state='running' WHERE slurm_job_id=? AND state='submitted'",
                (slurm_job_id,),
            )

    def mark_submission_complete(self, slurm_job_id: str | None) -> None:
        """Record completion for the matching scheduler allocation."""
        if not slurm_job_id:
            return
        with self.connection(write=True) as db:
            db.execute(
                "UPDATE scheduler_submissions SET state='complete' WHERE slurm_job_id=?",
                (slurm_job_id,),
            )

    def publication_work_items(self, request_id: str) -> tuple[dict, list[dict]]:
        """Return a request and its terminal work items for publication validation."""
        with self.connection() as db:
            request = db.execute(
                """
                SELECT r.*, wr.workflow_id, wr.revision AS workflow_revision
                FROM requests r JOIN workflow_revisions wr ON wr.id=r.workflow_revision_id
                WHERE r.id=? AND r.project=?
                """,
                (request_id, self.paths.project),
            ).fetchone()
            if request is None:
                raise KeyError(f"Unknown nro request: {request_id}")
            work_items = db.execute(
                """
                SELECT t.*,ci.configuration_class,ci.directory_label
                FROM request_work_items rt JOIN work_items t ON t.id=rt.work_item_id
                JOIN module_lineages ci ON ci.id=t.module_lineage_id
                WHERE rt.request_id=? AND rt.role='target' ORDER BY t.participant, t.work_item_key
                """,
                (request_id,),
            ).fetchall()
            plan = db.execute(
                "SELECT payload_json FROM request_plans WHERE request_id=?", (request_id,)
            ).fetchone()
            if plan is not None:
                terminals = set(json.loads(plan[0])["terminals"])
                work_items = [
                    row
                    for row in db.execute(
                        """SELECT t.*,ci.configuration_class,ci.directory_label,
                    COALESCE(e.logical_key,t.work_item_key) AS logical_key FROM request_artifacts rt
                    JOIN work_items t ON t.id=rt.work_item_id JOIN module_lineages ci ON ci.id=t.module_lineage_id
                    LEFT JOIN work_item_execution e ON e.work_item_id=t.id WHERE rt.request_id=?""",
                        (request_id,),
                    )
                    if row["logical_key"] in terminals
                ]
            return dict(request), [dict(row) for row in work_items]

    def unused_queued_worker_jobs(self) -> list[tuple[int, str]]:
        """Return queued pool jobs only when the project has no active demand."""
        with self.connection() as db:
            if db.execute("SELECT 1 FROM requests WHERE state='active' LIMIT 1").fetchone():
                return []
            return [
                (int(row["id"]), str(row["slurm_job_id"]))
                for row in db.execute(
                    """
                    SELECT id, slurm_job_id FROM scheduler_submissions
                    WHERE state='submitted' AND slurm_job_id IS NOT NULL
                    """
                )
            ]
