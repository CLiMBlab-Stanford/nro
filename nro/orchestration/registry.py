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
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence

import yaml

from nro.configuration.paths import BIDS_PATH, REGISTRY_PATH
from nro.configuration.store import (
    DERIVATIVE_CLASSES,
    fingerprint,
)
from nro.engine.cli import matches_instance_selectors as matches_selectors
from nro.engine.io import atomic_write_text
from nro.orchestration import dependency_state
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.workflow_registry import (
    WORKFLOW_SCHEMA,
    RegisteredWorkflow,
    WorkflowRegistry,
)

if TYPE_CHECKING:
    from nro.orchestration.contracts import ExecutionEnvelope, InstanceSpec


APPLICATION_ID = 0x4E524F31  # ASCII "NRO1"
SCHEMA_VERSION = 17


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
               UNION SELECT project FROM instances
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


def _instance_relative_directory(instance: dict) -> Path:
    project = str(instance["project"])
    participant = str(instance["participant"]).removeprefix("sub-")
    entities = (
        json.loads(instance["entities_json"])
        if isinstance(instance["entities_json"], str)
        else instance["entities_json"]
    )
    preferred = ("ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part", "chunk")
    ordered = [key for key in preferred if key in entities]
    ordered.extend(sorted(set(entities) - set(ordered)))
    name = "_".join([f"sub-{participant}", *(f"{key}-{entities[key]}" for key in ordered)])
    digest = str(instance["instance_key"]).split(":", 1)[-1][:16]
    return Path(project) / str(instance["module"]) / f"sub-{participant}" / name / digest


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
    manifests: Path
    requests: Path
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
            manifests=science / "manifests",
            requests=paths.scheduler / "requests",
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
    ) -> None:
        """Configure lock paths and waiting/recovery thresholds in seconds."""
        self.path = path
        self.recovery_path = recovery_path
        self.timeout = timeout
        self.stale_after = stale_after
        self.owner = LockOwner(
            token=uuid.uuid4().hex,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            uid=os.getuid(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
            acquired_at=utcnow(),
        )
        self._held = False

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
        """
        if not shutil.which("squeue"):
            return None
        try:
            result = subprocess.run(
                ["squeue", "--noheader", "--jobs", str(job_id), "--format", "%T"],
                check=False,
                text=True,
                capture_output=True,
                timeout=15,
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode:
            if (
                not result.stdout.strip()
                and result.stderr.strip() == "slurm_load_jobs error: Invalid job id specified"
            ):
                return True
            return None
        return not bool(result.stdout.strip())

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

    def _owner_definitively_dead(self, owner: dict | None) -> bool:
        if owner is None:
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

    def acquire(self) -> "RegistryLock":
        """Acquire the lock, recovering only demonstrably abandoned ownership.

        Raise RegistryLockTimeout when the configured waiting period expires.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
        deadline = time.monotonic() + self.timeout
        delay = 0.05
        while True:
            try:
                self.path.mkdir(mode=0o2775)
                self.path.chmod(0o2775)
            except FileExistsError:
                self._recover_if_safe()
                if time.monotonic() >= deadline:
                    raise RegistryLockTimeout(
                        f"Timed out waiting for registry lock {self.path}; owner={self._read_owner()}"
                    )
                time.sleep(random.uniform(delay, min(2.0, delay * 2.0)))
                delay = min(2.0, delay * 1.6)
                continue
            try:
                _atomic_text(self.owner_path, json.dumps(asdict(self.owner), indent=2) + "\n")
            except BaseException:
                shutil.rmtree(self.path, ignore_errors=True)
                raise
            self._held = True
            return self

    def release(self) -> None:
        """Release the lock owned by this object and remove its owner record."""
        if not self._held:
            return
        current = self._read_owner()
        if not current or current.get("token") != self.owner.token:
            self._held = False
            raise RuntimeError(f"Registry lock ownership changed while held: {self.path}")
        released = self.path.with_name(f"registry.lock.released-{self.owner.token}")
        self.path.rename(released)
        # The rename has already released the actual lock.  On the shared NFS
        # filesystem, immediate recursive removal can occasionally observe a
        # transient non-empty directory after ``owner.json`` was removed.  A
        # cleanup failure must not turn an otherwise completed instance into an
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


SCHEMA_SQL = (
    """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE bids_projects (
    project TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    discovered_at TEXT NOT NULL
);

CREATE TABLE bids_participants (
    project TEXT NOT NULL REFERENCES bids_projects(project) ON DELETE CASCADE,
    participant TEXT NOT NULL,
    path TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    PRIMARY KEY(project, participant)
);

"""
    + WORKFLOW_SCHEMA
    + """

CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    project TEXT NOT NULL,
    workflow_revision_id INTEGER NOT NULL REFERENCES workflow_revisions(id),
    target_module TEXT NOT NULL,
    selectors_json TEXT NOT NULL,
    concurrency INTEGER NOT NULL,
    partition_name TEXT,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE instances (
    id INTEGER PRIMARY KEY,
    instance_key TEXT NOT NULL UNIQUE,
    module TEXT NOT NULL,
    configuration_lineage_id INTEGER NOT NULL REFERENCES configuration_lineages(id),
    project TEXT NOT NULL,
    participant TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    scope TEXT NOT NULL,
    artifact_state TEXT NOT NULL,
    artifact_reason TEXT,
    current_generation INTEGER NOT NULL DEFAULT 0,
    manifest_path TEXT NOT NULL,
    resource_class TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    max_memory_gb INTEGER NOT NULL DEFAULT 256,
    revision_fingerprint TEXT NOT NULL,
    artifact_contract_json TEXT NOT NULL,
    artifact_fingerprint TEXT NOT NULL,
    command_json TEXT NOT NULL,
    runtime_config_path TEXT NOT NULL,
    input_paths_json TEXT NOT NULL,
    output_root TEXT NOT NULL,
    output_prefix TEXT,
    expected_outputs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE instance_dependencies (
    instance_id INTEGER NOT NULL REFERENCES instances(id),
    upstream_instance_id INTEGER NOT NULL REFERENCES instances(id),
    role TEXT NOT NULL,
    required_generation INTEGER,
    PRIMARY KEY(instance_id, upstream_instance_id, role)
);

CREATE TABLE request_instances (
    request_id TEXT NOT NULL REFERENCES requests(id),
    instance_id INTEGER NOT NULL REFERENCES instances(id),
    role TEXT NOT NULL,
    demand_state TEXT NOT NULL,
    PRIMARY KEY(request_id, instance_id)
);

CREATE TABLE workers (
    id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    hostname TEXT NOT NULL,
    pid INTEGER NOT NULL,
    resource_class TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    slurm_job_id TEXT,
    state TEXT NOT NULL,
    lease_expires_at REAL,
    successor_submission_id INTEGER,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE attempts (
    id INTEGER PRIMARY KEY,
    instance_id INTEGER NOT NULL REFERENCES instances(id),
    worker_id TEXT REFERENCES workers(id),
    state TEXT NOT NULL,
    revision_fingerprint TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    oom_detected INTEGER NOT NULL DEFAULT 0,
    process_group_id INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    error_type TEXT,
    error_message TEXT,
    log_path TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX one_active_attempt_per_instance
ON attempts(instance_id)
WHERE state IN ('queued', 'running', 'cancel_requested');

CREATE TABLE artifacts (
    id INTEGER PRIMARY KEY,
    instance_id INTEGER NOT NULL REFERENCES instances(id),
    attempt_id INTEGER REFERENCES attempts(id),
    direction TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER,
    mtime_ns INTEGER,
    digest_algorithm TEXT,
    digest TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE scheduler_submissions (
    id INTEGER PRIMARY KEY,
    intent_token TEXT NOT NULL UNIQUE,
    request_id TEXT REFERENCES requests(id),
    predecessor_worker_id TEXT REFERENCES workers(id),
    resource_class TEXT NOT NULL,
    memory_gb INTEGER NOT NULL DEFAULT 32,
    state TEXT NOT NULL,
    slurm_job_id TEXT,
    submitted_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX instance_module_participant ON instances(module, participant);
CREATE INDEX attempt_state ON attempts(state);
CREATE INDEX request_state ON requests(state);

CREATE TABLE attempt_dependencies (
    attempt_id INTEGER NOT NULL REFERENCES attempts(id),
    upstream_instance_id INTEGER NOT NULL REFERENCES instances(id),
    generation INTEGER NOT NULL,
    PRIMARY KEY(attempt_id, upstream_instance_id)
);
CREATE INDEX dependency_readers ON attempt_dependencies(upstream_instance_id);
CREATE TABLE artifact_mutations (
    instance_id INTEGER PRIMARY KEY REFERENCES instances(id),
    token TEXT NOT NULL
);

CREATE TABLE instance_execution (
    instance_id INTEGER PRIMARY KEY REFERENCES instances(id),
    branch TEXT NOT NULL,
    registry_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    context_json TEXT NOT NULL,
    binding_sources_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    scientific_contract_json TEXT NOT NULL,
    UNIQUE(registry_id, logical_key)
);
CREATE TABLE request_owners (
    request_id TEXT PRIMARY KEY REFERENCES requests(id),
    branch TEXT NOT NULL,
    registry_id TEXT NOT NULL
);
CREATE TABLE request_plans (
    request_id TEXT PRIMARY KEY REFERENCES requests(id),
    payload_json TEXT NOT NULL
);
CREATE TABLE compiled_revisions (
    registry_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    PRIMARY KEY(registry_id,logical_key)
);
CREATE TABLE branch_instances (
    registry_id TEXT NOT NULL,
    logical_key TEXT NOT NULL,
    instance_id INTEGER NOT NULL REFERENCES instances(id),
    scientific_contract_json TEXT NOT NULL,
    PRIMARY KEY(registry_id, logical_key)
);
CREATE TABLE request_artifacts (
    request_id TEXT NOT NULL REFERENCES requests(id),
    instance_id INTEGER NOT NULL REFERENCES instances(id),
    PRIMARY KEY(request_id,instance_id)
);
CREATE TABLE attempt_execution (
    attempt_id INTEGER PRIMARY KEY REFERENCES attempts(id),
    context_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    command_json TEXT NOT NULL
);
"""
)


class Registry(WorkflowRegistry):
    """Transactional authority for instances, demand, attempts, and worker state.

    Construction does not initialize storage. Mutating methods acquire the
    registry lock; callers should not alter the database directly.
    """

    def __init__(
        self,
        paths: RegistryPaths,
        *,
        lock_timeout: float = 120.0,
        stale_lock_after: float = 300.0,
    ) -> None:
        """Bind resolved paths and lock timings without opening the database."""
        from nro.configuration.site import require_execution_support

        require_execution_support()
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
            self.paths.manifests,
            self.paths.requests,
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
                connection.executescript(SCHEMA_SQL)
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
                f"{self.paths.database}. This development build does not migrate "
                "registries. Rebuild the lab-wide private control state with "
                "`python -m nro.bin.run --repair`. Public "
                "derivatives are stored outside the registry and are not removed."
            )
        return connection

    @contextlib.contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Open a locked database context, optionally for a write transaction.

        Schema incompatibility raises RuntimeError. Writes commit on success and
        roll back on failure; the context releases its connection and lock.
        """
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
        """Open a strictly read-only SQLite connection without the registry lock.

        This is for observational tools such as ``nro.bin.log``. Mutating
        operations and freshness assessment must continue through
        :meth:`connection`, which owns the cross-host lock.
        """
        database = self.existing_database_path()
        uri = database.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=60.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create and validate the private registry structure without requesting work."""
        with self.connection():
            pass

    def reinitialize(
        self, *, preserve_branch_runtime: bool = False, retain_backup: bool = False
    ) -> Path | None:
        """Rebuild work state, preserving ingestion, registrations, and execution sources.

        Existing state is treated as opaque and is never migrated. The active
        registry lock remains in place throughout replacement so another
        registry client cannot observe a partially rebuilt control directory.
        If initialization fails, the original control state is restored.
        Running ingestion leases must be resolved before replacement.

        preserve_branch_runtime retains scientific databases and configurations,
        archiving only their registry-bound completion certificates. retain_backup
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
            self.paths.manifests,
            self.paths.events,
            self.paths.snapshots,
            self.paths.workflows,
        )
        if preserve_branch_runtime:
            from nro.orchestration.branch_store import BranchStore

            catalog = BranchStore(self.paths.control)
            names = catalog.read().topology.records if catalog.path.exists() else {"main": None}
            science = tuple(
                ControlPaths(self.paths.control).branch(name) / "manifests" for name in names
            )
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
        shutil.rmtree(quarantine)
        return None

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

    def _upsert_instance_graph_locked(
        self,
        db: sqlite3.Connection,
        instance_records: Sequence[tuple["InstanceSpec", dict]],
        *,
        now: str,
        external_ids: Mapping[str, int] | None = None,
        owner_branch: str | None = None,
    ) -> dict[str, int]:
        """Merge logical instances and dependency edges without creating demand."""
        from nro.orchestration.manifests import _read_manifest

        instance_ids: dict[str, int] = dict(external_ids or {})
        replace_dependencies: dict[int, bool] = {}
        for spec, record in instance_records:
            manifest = (
                self.paths.manifests / _instance_relative_directory(record) / "completion.json"
            )
            if owner_branch is not None:
                manifest = (
                    ControlPaths(self.paths.control).branch(owner_branch)
                    / "manifests"
                    / _instance_relative_directory(record)
                    / "completion.json"
                )
            existing = db.execute(
                """SELECT id, scope, resource_class, revision_fingerprint,
                          artifact_contract_json, artifact_fingerprint,
                          command_json, runtime_config_path, input_paths_json,
                          output_root, output_prefix, expected_outputs_json
                   FROM instances WHERE instance_key=?""",
                (spec.key,),
            ).fetchone()
            if existing:
                instance_id = int(existing["id"])
                certificate = _read_manifest(manifest)
                # The planner is authoritative for the current execution
                # recipe. Only a change to the semantic instance contract,
                # rather than command spelling or resource settings, makes an
                # existing derivative stale.
                try:
                    recorded_contract = json.loads(existing["artifact_contract_json"])
                    if owner_branch is None:
                        from nro.orchestration.catalog import canonical_contract

                        recorded_contract = canonical_contract(
                            recorded_contract,
                            certificate.get("configuration") if certificate else None,
                        )
                    artifact_changed = (
                        fingerprint(recorded_contract) != record["artifact_fingerprint"]
                    )
                except (ValueError, TypeError, KeyError):
                    artifact_changed = True
                replace_dependencies[instance_id] = artifact_changed
                db.execute(
                    """
                    UPDATE instances SET scope=?, resource_class=?,
                        revision_fingerprint=?, artifact_contract_json=?,
                        artifact_fingerprint=?, command_json=?, runtime_config_path=?,
                        memory_gb=MAX(memory_gb, ?), max_memory_gb=MAX(max_memory_gb, ?),
                        input_paths_json=?, output_root=?, output_prefix=?,
                        expected_outputs_json=?,
                        artifact_state=CASE WHEN ? THEN 'stale' ELSE artifact_state END,
                        artifact_reason=CASE WHEN ? THEN 'Instance contract changed' ELSE artifact_reason END,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        record["scope"],
                        record["resource_class"],
                        record["revision_fingerprint"],
                        record["artifact_contract_json"],
                        record["artifact_fingerprint"],
                        record["command_json"],
                        record["runtime_config_path"],
                        record["memory_gb"],
                        record["max_memory_gb"],
                        record["input_paths_json"],
                        record["output_root"],
                        record["output_prefix"],
                        record["expected_outputs_json"],
                        artifact_changed,
                        artifact_changed,
                        now,
                        instance_id,
                    ),
                )
                if artifact_changed:
                    dependency_state.invalidate(
                        db,
                        [instance_id],
                        now=now,
                        reason="Resolved upstream instance contract changed",
                    )
                    db.execute(
                        """UPDATE attempts SET state='cancel_requested',
                                  error_type='InstanceGraphChanged',
                                  error_message='Instance contract changed while work was active'
                           WHERE instance_id=? AND state IN ('queued', 'running')""",
                        (instance_id,),
                    )
            else:
                cursor = db.execute(
                    """
                    INSERT INTO instances(
                        instance_key, module, configuration_lineage_id, project, participant,
                        entities_json, scope, artifact_state, artifact_reason,
                        manifest_path, resource_class, revision_fingerprint,
                        artifact_contract_json, artifact_fingerprint,
                        memory_gb, max_memory_gb, command_json, runtime_config_path, input_paths_json,
                        output_root, output_prefix, expected_outputs_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'missing', 'Not yet assessed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.key,
                        spec.module,
                        spec.configuration_lineage_id,
                        spec.project,
                        spec.participant,
                        record["entities_json"],
                        spec.scope,
                        str(manifest),
                        spec.resource_class,
                        record["revision_fingerprint"],
                        record["artifact_contract_json"],
                        record["artifact_fingerprint"],
                        record["memory_gb"],
                        record["max_memory_gb"],
                        record["command_json"],
                        record["runtime_config_path"],
                        record["input_paths_json"],
                        record["output_root"],
                        record["output_prefix"],
                        record["expected_outputs_json"],
                        now,
                        now,
                    ),
                )
                instance_id = int(cursor.lastrowid)
                replace_dependencies[instance_id] = True
            instance_ids[spec.key] = instance_id

        for spec, _record in instance_records:
            instance_id = instance_ids[spec.key]
            proposed = tuple(spec.dependencies)
            if not replace_dependencies[instance_id]:
                continue
            db.execute("DELETE FROM instance_dependencies WHERE instance_id=?", (instance_id,))
            for dependency in proposed:
                db.execute(
                    """
                    INSERT INTO instance_dependencies(instance_id, upstream_instance_id, role, required_generation)
                    VALUES (?, ?, ?, NULL)
                    """,
                    (
                        instance_id,
                        instance_ids[dependency],
                        "inherited" if dependency in (external_ids or {}) else "input",
                    ),
                )
        dependency_state.synchronize(db, now=now)
        return instance_ids

    @staticmethod
    def _normalize_active_request_graph_locked(db: sqlite3.Connection) -> None:
        """Reconcile existing active demand with the current global DAG."""
        parents: dict[int, list[int]] = {}
        for row in db.execute(
            "SELECT instance_id, upstream_instance_id FROM instance_dependencies WHERE role != 'inherited'"
        ):
            parents.setdefault(int(row["instance_id"]), []).append(int(row["upstream_instance_id"]))
        for active_request in db.execute("SELECT id FROM requests WHERE state='active'").fetchall():
            active_request_id = str(active_request["id"])
            targets = {
                int(row["instance_id"])
                for row in db.execute(
                    """SELECT instance_id FROM request_instances
                       WHERE request_id=? AND role='target' AND demand_state='active'""",
                    (active_request_id,),
                )
            }
            required = set(targets)
            pending = list(targets)
            while pending:
                instance_id = pending.pop()
                for upstream_id in parents.get(instance_id, ()):
                    if upstream_id not in required:
                        required.add(upstream_id)
                        pending.append(upstream_id)
            existing = {
                int(row["instance_id"]): str(row["demand_state"])
                for row in db.execute(
                    "SELECT instance_id, demand_state FROM request_instances WHERE request_id=?",
                    (active_request_id,),
                )
            }
            for required_id in required - set(existing):
                db.execute(
                    """INSERT INTO request_instances(request_id, instance_id, role, demand_state)
                       VALUES (?, ?, 'dependency', 'active')""",
                    (active_request_id, required_id),
                )
            orphaned = {
                instance_id
                for instance_id, demand_state in existing.items()
                if demand_state == "active" and instance_id not in required
            }
            if orphaned:
                placeholders = ",".join("?" for _ in orphaned)
                db.execute(
                    f"""UPDATE request_instances SET demand_state='cancelled'
                        WHERE request_id=? AND instance_id IN ({placeholders})
                          AND demand_state='active'""",
                    (active_request_id, *tuple(orphaned)),
                )

    def _validate_instance_projects(self, instances: Sequence["InstanceSpec"]) -> None:
        foreign_projects = {
            spec.project for spec in instances if spec.project != self.paths.project
        }
        if foreign_projects:
            raise ValueError(
                "A registry operation may contain instances from only its selected project: "
                + ", ".join(sorted(foreign_projects))
            )

    def register_instances(self, instances: Sequence["InstanceSpec"]) -> dict[str, int]:
        """Discover instance contracts and edges without creating a request."""
        self._validate_instance_projects(instances)
        instance_records = tuple((spec, spec.as_record()) for spec in instances)
        with self.connection(write=True) as db:
            instance_ids = self._upsert_instance_graph_locked(db, instance_records, now=utcnow())
            self._normalize_active_request_graph_locked(db)
            return instance_ids

    def register_owned_lineages(self, records: Sequence[Mapping[str, object]]) -> dict[str, int]:
        """Restore configuration lineages from derivative ownership records."""
        ordered = sorted(
            records,
            key=lambda item: (
                DERIVATIVE_CLASSES.index(str(item["derivative_class"])),
                str(item["lineage_fingerprint"]),
            ),
        )
        lineage_ids: dict[str, int] = {}
        now = utcnow()
        with self.connection(write=True) as db:
            for record in ordered:
                derivative_class = str(record["derivative_class"])
                lineage_fingerprint = str(record["lineage_fingerprint"])
                directory_label = str(record["directory_label"])
                configuration = record["configuration"]
                if not isinstance(configuration, Mapping):
                    raise ValueError("Ownership record configuration must be a mapping")
                existing = db.execute(
                    """SELECT id, config_id, directory_label
                       FROM configuration_lineages
                       WHERE derivative_class=? AND lineage_fingerprint=?""",
                    (derivative_class, lineage_fingerprint),
                ).fetchone()
                collision = db.execute(
                    """SELECT lineage_fingerprint FROM configuration_lineages
                       WHERE derivative_class=? AND directory_label=?""",
                    (derivative_class, directory_label),
                ).fetchone()
                if collision and str(collision["lineage_fingerprint"]) != lineage_fingerprint:
                    raise ValueError(
                        f"Derivative directory {derivative_class}/{directory_label} "
                        "declares conflicting configuration lineages"
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
                        """UPDATE configuration_lineages
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
                        INSERT INTO configuration_lineages(
                            derivative_class, config_id, config_fingerprint,
                            lineage_fingerprint, resolved_yaml, directory_label, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            derivative_class,
                            str(configuration["id"]),
                            str(configuration["fingerprint"]),
                            lineage_fingerprint,
                            yaml.safe_dump(configuration["resolved"], sort_keys=False),
                            directory_label,
                            now,
                        ),
                    )
                    lineage_id = int(cursor.lastrowid)
                lineage_ids[lineage_fingerprint] = lineage_id

            for record in ordered:
                lineage_id = lineage_ids[str(record["lineage_fingerprint"])]
                db.execute(
                    "DELETE FROM configuration_lineage_dependencies "
                    "WHERE configuration_lineage_id=?",
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
                        """INSERT INTO configuration_lineage_dependencies(
                               configuration_lineage_id,
                               upstream_configuration_lineage_id, role
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

    def instance_ids(self, instance_keys: Sequence[str]) -> dict[str, int]:
        """Resolve registered instance keys without changing demand."""
        if not instance_keys:
            return {}
        unique = tuple(dict.fromkeys(instance_keys))
        placeholders = ",".join("?" for _ in unique)
        with self.connection() as db:
            return {
                str(row["instance_key"]): int(row["id"])
                for row in db.execute(
                    f"SELECT id, instance_key FROM instances WHERE instance_key IN ({placeholders})",
                    unique,
                )
            }

    def create_request(
        self,
        *,
        registered: RegisteredWorkflow,
        target_module: str,
        selectors: dict,
        instances: Sequence["InstanceSpec"],
        terminal_instance_keys: Sequence[str],
        concurrency: int,
        partition: str | None,
        user_name: str | None = None,
    ) -> str:
        """Merge a planned graph and create a new active demand request."""
        if concurrency < 1:
            raise ValueError("Concurrency must be at least one")
        self._validate_instance_projects(instances)
        request_id = uuid.uuid4().hex
        now = utcnow()
        terminal = set(terminal_instance_keys)
        instance_records = tuple((spec, spec.as_record()) for spec in instances)
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
            instance_ids = self._upsert_instance_graph_locked(db, instance_records, now=now)
            for spec in instances:
                instance_id = instance_ids[spec.key]
                db.execute(
                    """
                    INSERT INTO request_instances(request_id, instance_id, role, demand_state)
                    VALUES (?, ?, ?, 'active')
                    """,
                    (request_id, instance_id, "target" if spec.key in terminal else "dependency"),
                )
            # A shared multirun instance can gain or lose upstream runs while an
            # older request is still active. Keep every active request aligned
            # with the current global graph.
            self._normalize_active_request_graph_locked(db)
        request_record = {
            "id": request_id,
            "project": self.paths.project,
            "workflow": registered.selector,
            "module": target_module,
            "selectors": selectors,
            "concurrency": concurrency,
            "partition": partition,
            "created_at": now,
        }
        _atomic_text(
            self.paths.requests / f"{request_id}.json",
            json.dumps(request_record, indent=2, sort_keys=True) + "\n",
        )
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

    def instance_rows(self, *, read_only: bool = False) -> list[dict]:
        """Read instance records with their current orchestration and artifact state."""
        manager = self.read_connection() if read_only else self.connection()
        with manager as db:
            rows = db.execute(
                """
                SELECT t.*, ci.directory_label, ci.lineage_fingerprint, ci.derivative_class,
                       EXISTS(
                         SELECT 1 FROM workflow_bindings binding
                         WHERE binding.configuration_lineage_id=t.configuration_lineage_id
                       ) AS recomputable,
                       EXISTS(SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                              WHERE rt.instance_id=t.id AND rt.demand_state='active' AND r.state='active') AS demanded,
                       (SELECT a.state FROM attempts a WHERE a.instance_id=t.id ORDER BY a.id DESC LIMIT 1) AS attempt_state,
                       (SELECT a.error_type FROM attempts a WHERE a.instance_id=t.id ORDER BY a.id DESC LIMIT 1) AS error_type,
                       (SELECT a.error_message FROM attempts a WHERE a.instance_id=t.id ORDER BY a.id DESC LIMIT 1) AS error_message,
                       (SELECT a.log_path FROM attempts a WHERE a.instance_id=t.id ORDER BY a.id DESC LIMIT 1) AS log_path,
                       (SELECT COUNT(*) FROM attempts a WHERE a.instance_id=t.id AND a.oom_detected=1) AS oom_count,
                       (SELECT a.memory_gb FROM attempts a WHERE a.instance_id=t.id ORDER BY a.id DESC LIMIT 1) AS attempt_memory_gb
                       ,EXISTS(
                         SELECT 1 FROM request_instances retry_rt JOIN requests retry ON retry.id=retry_rt.request_id
                         WHERE retry_rt.instance_id=t.id AND retry_rt.demand_state='active' AND retry.state='active'
                           AND retry.updated_at > COALESCE(
                             (SELECT CASE WHEN latest.state='cancelled'
                                      THEN latest.started_at ELSE latest.completed_at END
                              FROM attempts latest WHERE latest.instance_id=t.id
                              ORDER BY latest.id DESC LIMIT 1),
                             ''
                           )
                       ) AS retry_requested
                       ,(SELECT GROUP_CONCAT(DISTINCT wr.workflow_id)
                         FROM request_instances rt
                         JOIN requests r ON r.id=rt.request_id
                         JOIN workflow_revisions wr ON wr.id=r.workflow_revision_id
                         WHERE rt.instance_id=t.id) AS workflow_ids
                FROM instances t
                JOIN configuration_lineages ci ON ci.id=t.configuration_lineage_id
                ORDER BY t.participant, t.module, t.instance_key
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def instance_dependencies(self, *, read_only: bool = False) -> list[tuple[int, int]]:
        """Return dependency relationships for the selected instance records."""
        manager = self.read_connection() if read_only else self.connection()
        with manager as db:
            return [
                (int(row["instance_id"]), int(row["upstream_instance_id"]))
                for row in db.execute(
                    "SELECT instance_id, upstream_instance_id FROM instance_dependencies"
                )
            ]

    def instance_status_snapshot(
        self,
        *,
        read_only: bool = False,
        artifact_states: Mapping[int, tuple[str, str]] | None = None,
    ) -> list[dict]:
        """Return the registry's canonical current state for every instance.

        ``instances`` retain artifact freshness and ``attempts`` retain immutable
        execution history.  This method is the sole projection of those facts
        into one current instance state for user-facing tools.  A read-only
        preview may supply temporary artifact states without persisting them.
        """
        rows = self.instance_rows(read_only=read_only)
        if artifact_states:
            for row in rows:
                projected = artifact_states.get(int(row["id"]))
                if projected is not None:
                    row["artifact_state"], row["artifact_reason"] = projected
        by_id = {int(row["id"]): row for row in rows}
        parents: dict[int, list[int]] = {}
        for instance_id, upstream_id in self.instance_dependencies(read_only=read_only):
            parents.setdefault(instance_id, []).append(upstream_id)
        root_cache: dict[int, tuple[int, ...]] = {}

        def failure_roots(instance_id: int) -> tuple[int, ...]:
            if instance_id in root_cache:
                return root_cache[instance_id]
            row = by_id[instance_id]
            roots: set[int] = set()
            # Attempts are immutable execution history, while artifact_state is
            # the authoritative current result.  A derivative can validate as
            # fresh after an older failed attempt (for example through native
            # completion evidence), so that attempt must not remain a current
            # failure root or block downstream instances.
            if (
                row["artifact_state"] != "fresh"
                and row.get("attempt_state") == "error"
                and not row.get("retry_requested")
                and not (
                    row["artifact_state"] == "missing"
                    and row.get("artifact_reason") == "Purged by user"
                    and not row.get("demanded")
                )
            ):
                roots.add(instance_id)
            for parent in parents.get(instance_id, ()):
                if parent in by_id:
                    roots.update(failure_roots(parent))
            root_cache[instance_id] = tuple(sorted(roots))
            return root_cache[instance_id]

        snapshot: list[dict] = []
        for row in rows:
            item = dict(row)
            instance_id = int(item["id"])
            roots = failure_roots(instance_id)
            attempt = item.get("attempt_state")
            if item["artifact_state"] == "fresh":
                state = "Success"
            elif not item.get("recomputable") and not item.get("demanded"):
                state = "Unavailable"
            elif attempt == "error" and instance_id in roots:
                state = "Error"
            elif roots and item.get("demanded"):
                state = "Blocked"
            elif attempt == "cancel_requested":
                state = "Stopping"
            elif attempt == "running":
                state = "Running"
            elif attempt == "queued" or item.get("retry_requested"):
                state = "Queued"
            elif (
                item["artifact_state"] == "missing"
                and item.get("artifact_reason") == "Purged by user"
                and not item.get("demanded")
            ):
                state = "Missing"
            elif attempt == "error":
                state = "Error"
            elif item.get("demanded"):
                state = "Queued"
            elif attempt == "cancelled" and item.get("error_type") == "UserCancelled":
                state = "Stopped"
            elif item["artifact_state"] == "missing":
                state = "Missing"
            else:
                state = "Stale"
            item["status"] = state
            item["root_failure_ids"] = roots
            snapshot.append(item)
        return snapshot

    def register_worker(
        self,
        worker_id: str,
        *,
        resource_class: str,
        memory_gb: int = 32,
        slurm_job_id: str | None = None,
        lease_seconds: float = 120.0,
    ) -> None:
        """Register or refresh a worker lease and its scheduler allocation.

        Workers registering during maintenance are marked for shutdown.
        """
        now = utcnow()
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
                    slurm_job_id=excluded.slurm_job_id, updated_at=excluded.updated_at
                """,
                (
                    worker_id,
                    getpass.getuser(),
                    resource_class,
                    memory_gb,
                    slurm_job_id,
                    state,
                    socket.gethostname(),
                    os.getpid(),
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
        """Stop worker allocations without withdrawing instance demand."""
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

    def claim_ready_instance(
        self,
        worker_id: str,
        resource_classes: Sequence[str],
        *,
        memory_gb: int = 32,
    ) -> "ExecutionEnvelope | None":
        """Claim one demanded, nonfresh instance whose upstream instances are fresh."""
        if not resource_classes:
            return None
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
                    "SELECT COUNT(*) FROM attempts WHERE state IN ('queued', 'running', 'cancel_requested')"
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
                FROM instances t
                JOIN configuration_lineages ci ON ci.id=t.configuration_lineage_id
                WHERE t.resource_class IN ({placeholders})
                  AND t.memory_gb <= ?
                  AND {dependency_state.WRITE_READY}
                  AND t.artifact_state != 'fresh'
                  AND NOT (
                      t.module IN ('dynconn', 'microparcellation')
                      AND t.artifact_reason LIKE 'Selected raw run universe changed:%'
                  )
                  AND EXISTS (
                      SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.instance_id=t.id AND rt.demand_state='active' AND r.state='active'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM instance_dependencies td JOIN instances up ON up.id=td.upstream_instance_id
                      WHERE td.instance_id=t.id AND up.artifact_state != 'fresh'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM attempts active WHERE active.instance_id=t.id
                      AND active.state IN ('queued', 'running', 'cancel_requested')
                  )
                  AND (
                      NOT EXISTS (SELECT 1 FROM attempts old WHERE old.instance_id=t.id)
                      OR COALESCE((SELECT state FROM attempts old WHERE old.instance_id=t.id ORDER BY id DESC LIMIT 1), '') = 'success'
                      OR COALESCE((SELECT error_type FROM attempts old
                                   WHERE old.instance_id=t.id ORDER BY id DESC LIMIT 1), '')
                         IN ('UpstreamStale', 'UpstreamFailed', 'WorkerTerminated',
                             'InstanceGraphChanged', 'RegistryUnavailable')
                      OR EXISTS (
                          SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                          WHERE rt.instance_id=t.id AND rt.demand_state='active' AND r.state='active'
                          AND r.updated_at > COALESCE(
                              (SELECT CASE WHEN old.state='cancelled'
                                       THEN old.started_at ELSE old.completed_at END
                               FROM attempts old WHERE old.instance_id=t.id ORDER BY id DESC LIMIT 1),
                              '')
                      )
                  )
                ORDER BY CASE t.module WHEN 'anat' THEN 1 WHEN 'func' THEN 2 WHEN 'clean' THEN 3
                                      WHEN 'microparcellation' THEN 4 WHEN 'dynconn' THEN 4 ELSE 5 END,
                         t.participant, t.instance_key
                LIMIT 1
                """,
                (*resource_classes, memory_gb),
            ).fetchone()
            if row is None:
                return None
            instance = dict(row)
            log_dir = self.paths.events / _instance_relative_directory(instance)
            execution = db.execute(
                "SELECT * FROM instance_execution WHERE instance_id=?", (instance["id"],)
            ).fetchone()
            if execution is not None:
                from nro.orchestration.branch_admission import prepare_attempt

                log_dir = (
                    ControlPaths(self.paths.control).branch(execution["branch"])
                    / "events"
                    / _instance_relative_directory(instance)
                )
                ensure_shared_directory(log_dir)
                command = prepare_attempt(self, db, instance, dict(execution), log_dir)
                if command is None:
                    return None
                instance["command_json"] = json.dumps(command)
            ensure_shared_directory(log_dir)
            cursor = db.execute(
                """
                INSERT INTO attempts(instance_id, worker_id, state, revision_fingerprint,
                                     memory_gb, started_at, log_path, created_at)
                VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    instance["id"],
                    worker_id,
                    instance["revision_fingerprint"],
                    memory_gb,
                    now,
                    str(log_dir / "instance.log"),
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
                        instance["command_json"],
                    ),
                )
            dependency_state.capture_inputs(db, attempt_id, int(instance["id"]))
            # An instance has one current log, deliberately replaced by the next
            # attempt. Attempt history remains in the registry/events tables.
            log_path = log_dir / "instance.log"
            db.execute(
                "UPDATE workers SET state='running', lease_expires_at=?, updated_at=? WHERE id=?",
                (time.time() + 120.0, now, worker_id),
            )
            instance["attempt_id"] = attempt_id
            instance["log_path"] = str(log_path)
            from nro.orchestration.contracts import ExecutionEnvelope

            return ExecutionEnvelope.from_registry_row(instance)

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

    def demanded_instance_ids(self) -> tuple[int, ...]:
        """Return instances currently required by at least one active request."""
        with self.connection() as db:
            return tuple(
                int(row["instance_id"])
                for row in db.execute(
                    """
                    SELECT DISTINCT rt.instance_id
                    FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                    WHERE rt.demand_state='active' AND r.state='active'
                    ORDER BY rt.instance_id
                    """
                )
            )

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
        """Release the audit lease and record when instance states became authoritative."""
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

        A worker allocation can execute only one instance at a time, so cancellation
        is deliberately attempt-scoped rather than a Slurm ``scancel`` of the
        entire worker.  The worker observes ``cancel_requested`` promptly and
        terminates just its child process before returning to the shared pool.
        """
        with self.connection(write=True) as db:
            return dependency_state.synchronize(db, now=utcnow())

    def cancel_attempts_downstream_of_failure(self, instance_id: int) -> list[dict]:
        """Stop active descendant attempts after ``instance_id`` has fatally failed.

        This is transitive rather than relying on each intermediate instance having
        already been reassessed.  It closes the short race in which an external
        change or a concurrent request allowed work from several DAG levels to
        be active when an ancestor fails.
        """
        with self.connection(write=True) as db:
            return dependency_state.invalidate(
                db,
                [instance_id],
                now=utcnow(),
                reason=f"Resolved upstream instance failed: {instance_id}",
                error_type="UpstreamFailed",
            )

    @contextlib.contextmanager
    def artifact_mutation(
        self, instance_ids: Iterable[int], *, timeout: float = 30.0
    ) -> Iterator[None]:
        """Reserve outputs for deletion or replacement after cancelling their consumers.

        Target attempts must already be stopped. Other claims remain available
        while consumers shut down. A timeout performs no caller writes, leaves
        invalidation in place, and releases the reservation. After a hard crash,
        a later mutation recovers the filesystem lock before clearing old tokens.
        """
        ids = tuple(sorted(set(instance_ids)))
        if not ids:
            yield
            return
        self._prepare_directories()
        scheduler = ControlPaths(self.paths.control).scheduler
        placeholders = ",".join("?" for _ in ids)
        token = uuid.uuid4().hex
        with RegistryLock(
            scheduler / "artifact-mutation.lock", scheduler / "artifact-mutation.recovery-lock"
        ):
            with self.connection(write=True) as db:
                if db.execute(
                    f"SELECT 1 FROM attempts WHERE instance_id IN ({placeholders}) AND state IN {dependency_state.ACTIVE}",
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
                    f"UPDATE instances SET artifact_state='stale', artifact_reason='Outputs reserved for mutation', updated_at=? WHERE id IN ({placeholders})",
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
                            WHERE pinned.upstream_instance_id IN ({placeholders})
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
        """Persist attempt completion and update the associated instance state."""
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
                        SELECT 1 FROM request_instances rt JOIN instances t ON t.id=rt.instance_id
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
            SELECT a.instance_id, a.worker_id, a.memory_gb AS attempt_memory,
                   t.memory_gb, t.max_memory_gb
            FROM attempts a JOIN instances t ON t.id=a.instance_id WHERE a.id=?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown attempt: {attempt_id}")
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
                UPDATE instances SET memory_gb=?, artifact_state='stale', artifact_reason=?, updated_at=?
                WHERE id=?
                """,
                (
                    next_memory,
                    f"OOM at {current} GB; retrying at {next_memory} GB",
                    utcnow(),
                    row["instance_id"],
                ),
            )
            db.execute(
                """
                UPDATE requests SET updated_at=? WHERE state='active' AND id IN (
                    SELECT request_id FROM request_instances
                    WHERE instance_id=? AND demand_state='active'
                )
                """,
                (utcnow(), row["instance_id"]),
            )
        else:
            db.execute(
                "UPDATE instances SET artifact_reason=?, updated_at=? WHERE id=?",
                (f"OOM at configured ceiling of {limit} GB", utcnow(), row["instance_id"]),
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
        participant_set = {value.removeprefix("sub-") for value in participants}
        module_set = set(modules)
        workflow_set = set(workflows)
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
            branch_instances = (
                None
                if branch_registry_id is None
                else {
                    row[0]
                    for row in db.execute(
                        "SELECT instance_id FROM branch_instances WHERE registry_id=?",
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
                return {"instances": 0, "requests": 0, "attempts": 0}
            selected = {
                int(row["id"])
                for row in db.execute(
                    "SELECT id, participant, module, entities_json FROM instances WHERE project=?",
                    (self.paths.project,),
                )
                if (not participant_set or row["participant"] in participant_set)
                and (not module_set or row["module"] in module_set)
                and matches_selectors(json.loads(row["entities_json"]), selectors)
                and (branch_instances is None or row["id"] in branch_instances)
            }
            if include_dependents:
                changed = True
                while changed:
                    before = len(selected)
                    for edge in db.execute(
                        "SELECT instance_id, upstream_instance_id FROM instance_dependencies"
                    ):
                        if int(edge["upstream_instance_id"]) in selected:
                            selected.add(int(edge["instance_id"]))
                    changed = len(selected) != before
            if not selected:
                return {"instances": 0, "requests": 0, "attempts": 0}
            instance_placeholders = ",".join("?" for _ in selected)
            request_placeholders = ",".join("?" for _ in eligible_requests)
            values = (*tuple(selected), *tuple(eligible_requests))
            affected_requests = {
                str(row["request_id"])
                for row in db.execute(
                    f"""SELECT DISTINCT request_id FROM request_instances
                        WHERE instance_id IN ({instance_placeholders})
                          AND request_id IN ({request_placeholders})
                          AND demand_state='active'""",
                    values,
                )
            }
            if not affected_requests:
                return {"instances": 0, "requests": 0, "attempts": 0}
            cursor = db.execute(
                f"""UPDATE request_instances SET demand_state='cancelled'
                    WHERE instance_id IN ({instance_placeholders})
                      AND request_id IN ({request_placeholders})
                      AND demand_state='active'""",
                values,
            )
            demand_count = cursor.rowcount
            cancelled_requests: list[str] = []
            edges = [
                (int(row["instance_id"]), int(row["upstream_instance_id"]))
                for row in db.execute(
                    "SELECT instance_id, upstream_instance_id FROM instance_dependencies"
                )
            ]
            for request_id in affected_requests:
                active_targets = {
                    int(row["instance_id"])
                    for row in db.execute(
                        """SELECT instance_id FROM request_instances
                           WHERE request_id=? AND role='target' AND demand_state='active'""",
                        (request_id,),
                    )
                }
                if not active_targets:
                    pruned = db.execute(
                        "UPDATE request_instances SET demand_state='cancelled' WHERE request_id=? AND demand_state='active'",
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
                    for instance_id, upstream_id in edges:
                        if instance_id in required:
                            required.add(upstream_id)
                    changed = len(required) != before
                active_instances = {
                    int(row["instance_id"])
                    for row in db.execute(
                        "SELECT instance_id FROM request_instances WHERE request_id=? AND demand_state='active'",
                        (request_id,),
                    )
                }
                orphaned = active_instances - required
                if orphaned:
                    orphan_placeholders = ",".join("?" for _ in orphaned)
                    demand_count += db.execute(
                        f"""UPDATE request_instances SET demand_state='cancelled'
                            WHERE request_id=? AND instance_id IN ({orphan_placeholders})
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
                      SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.instance_id=attempts.instance_id AND rt.demand_state='active' AND r.state='active'
                  )
                """
            )
            return {
                "instances": demand_count,
                "requests": len(cancelled_requests),
                "attempts": cursor.rowcount,
            }

    def reconcile_requests(self) -> None:
        """Update request states from their demanded instances' current outcomes."""
        with self.connection(write=True) as db:
            db.execute(
                """
                UPDATE requests SET state='satisfied', updated_at=?
                WHERE state='active' AND NOT EXISTS (
                    SELECT 1 FROM request_instances rt JOIN instances t ON t.id=rt.instance_id
                    WHERE rt.request_id=requests.id AND rt.role='target'
                      AND rt.demand_state='active' AND t.artifact_state!='fresh'
                )
                """,
                (utcnow(),),
            )
            # A request may contain many independent branches (for example,
            # one functional run per acquisition).  A failed branch is
            # represented by its instance/attempt and blocks only its descendants;
            # it must not deactivate demand for unrelated ready branches.
            # Failed instances remain non-retryable until a later orchestration run
            # refreshes demand, per the claim predicate above.

    def reserve_worker_submissions(
        self,
        *,
        request_id: str | None,
        resource_class: str,
        memory_gb: int = 32,
    ) -> list[tuple[int, str]]:
        """Reserve workers for ready derivative and ingestion work under one limit."""
        with self.connection(write=True) as db:
            if db.execute("SELECT 1 FROM metadata WHERE key='maintenance_mode'").fetchone():
                return []
            dependency_state.synchronize(db, now=utcnow())
            from nro.bidsify.index import IngestionIndex

            ingestion_active, ingestion_ready, ingestion_limit = IngestionIndex(self).summary(
                memory_gb
            )
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
            compatible = {
                "large": ("large", "medium", "small"),
                "medium": ("medium", "small"),
                "small": ("small",),
            }.get(resource_class, (resource_class,))
            placeholders = ",".join("?" for _ in compatible)
            active_instances = int(
                db.execute(
                    "SELECT COUNT(*) FROM attempts WHERE state IN ('queued', 'running', 'cancel_requested')"
                ).fetchone()[0]
            )
            ready_instances = int(
                db.execute(
                    f"""
                    SELECT COUNT(*) FROM instances t
                    WHERE t.resource_class IN ({placeholders})
                      AND t.memory_gb<=?
                      AND {dependency_state.WRITE_READY}
                      AND t.artifact_state!='fresh'
                      AND NOT (
                          t.module IN ('dynconn', 'microparcellation')
                          AND t.artifact_reason LIKE 'Selected raw run universe changed:%'
                      )
                      AND EXISTS (
                          SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                          WHERE rt.instance_id=t.id AND rt.demand_state='active' AND r.state='active'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM instance_dependencies td JOIN instances up ON up.id=td.upstream_instance_id
                          WHERE td.instance_id=t.id AND up.artifact_state!='fresh'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM attempts a WHERE a.instance_id=t.id
                          AND a.state IN ('queued', 'running', 'cancel_requested')
                      )
                      AND (
                          NOT EXISTS (SELECT 1 FROM attempts old WHERE old.instance_id=t.id)
                          OR COALESCE((SELECT state FROM attempts old WHERE old.instance_id=t.id ORDER BY id DESC LIMIT 1), '') = 'success'
                          OR COALESCE((SELECT error_type FROM attempts old
                                      WHERE old.instance_id=t.id ORDER BY id DESC LIMIT 1), '')
                             IN ('UpstreamStale', 'UpstreamFailed', 'WorkerTerminated',
                                 'InstanceGraphChanged', 'RegistryUnavailable')
                          OR EXISTS (
                              SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                              WHERE rt.instance_id=t.id AND rt.demand_state='active' AND r.state='active'
                              AND r.updated_at > COALESCE(
                                  (SELECT CASE WHEN old.state='cancelled'
                                           THEN old.started_at ELSE old.completed_at END
                                   FROM attempts old WHERE old.instance_id=t.id ORDER BY id DESC LIMIT 1),
                                  '')
                          )
                      )
                    """,
                    (*compatible, memory_gb),
                ).fetchone()[0]
            )
            desired = min(
                max(desired, ingestion_limit),
                active_instances + ready_instances + ingestion_active + ingestion_ready,
            )
            live_workers = int(
                db.execute(
                    """SELECT COUNT(*) FROM workers
                       WHERE state IN ('idle', 'running') AND lease_expires_at>?
                         AND memory_gb>=?""",
                    (time.time(), memory_gb),
                ).fetchone()[0]
            )
            pending = int(
                db.execute(
                    """SELECT COUNT(*) FROM scheduler_submissions
                       WHERE state IN ('prepared', 'submitted')
                         AND predecessor_worker_id IS NULL AND memory_gb>=?""",
                    (memory_gb,),
                ).fetchone()[0]
            )
            count = max(0, desired - live_workers - pending)
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
                            - active_instances
                            - ingestion_active
                            - pending,
                        ),
                    )
                    - own_idle,
                )
            reservations: list[tuple[int, str]] = []
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

    def required_memory_above(self, memory_gb: int) -> int | None:
        """Return the smallest ready instance tier that this worker cannot satisfy."""
        with self.connection() as db:
            row = db.execute(
                f"""
                SELECT MIN(t.memory_gb) AS memory_gb FROM instances t
                WHERE t.memory_gb> ? AND t.artifact_state!='fresh'
                  AND {dependency_state.WRITE_READY}
                  AND EXISTS (
                      SELECT 1 FROM request_instances rt JOIN requests r ON r.id=rt.request_id
                      WHERE rt.instance_id=t.id AND rt.demand_state='active' AND r.state='active'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM instance_dependencies td JOIN instances up ON up.id=td.upstream_instance_id
                      WHERE td.instance_id=t.id AND up.artifact_state!='fresh'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM attempts a WHERE a.instance_id=t.id
                      AND a.state IN ('queued', 'running', 'cancel_requested')
                  )
                """,
                (memory_gb,),
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
                  AND lease_expires_at>? AND memory_gb>=? LIMIT 1
                """,
                (time.time(), memory_gb),
            ).fetchone()
            capable_submission = db.execute(
                """
                SELECT 1 FROM scheduler_submissions
                WHERE state IN ('prepared', 'submitted') AND memory_gb>=? LIMIT 1
                """,
                (memory_gb,),
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
        if not dead:
            return 0
        recovered = 0
        with self.connection(write=True) as db:
            for worker_id in dead:
                from nro.bidsify.index import IngestionIndex

                recovered += IngestionIndex(self).recover_locked({worker_id})
                attempts = db.execute(
                    "SELECT id, instance_id FROM attempts WHERE worker_id=? AND state IN ('queued', 'running', 'cancel_requested')",
                    (worker_id,),
                ).fetchall()
                for attempt in attempts:
                    if worker_id in oom_workers:
                        self._record_oom_locked(
                            db,
                            int(attempt["id"]),
                            message=(
                                f"Slurm reported OUT_OF_MEMORY for worker job "
                                f"{next(item['slurm_job_id'] for item in expired if item['id'] == worker_id)}"
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
                    # This is an interrupted attempt, not a scientific failure.
                    # Renew existing demand so a successor may resume it.
                    db.execute(
                        """
                        UPDATE requests SET updated_at=? WHERE state='active' AND id IN (
                            SELECT request_id FROM request_instances
                            WHERE instance_id=? AND demand_state='active'
                        )
                        """,
                        (utcnow(), attempt["instance_id"]),
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

    def publication_instances(self, request_id: str) -> tuple[dict, list[dict]]:
        """Return a request and its terminal instances for publication validation."""
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
            instances = db.execute(
                """
                SELECT t.*,ci.derivative_class FROM request_instances rt JOIN instances t ON t.id=rt.instance_id
                JOIN configuration_lineages ci ON ci.id=t.configuration_lineage_id
                WHERE rt.request_id=? AND rt.role='target' ORDER BY t.participant, t.instance_key
                """,
                (request_id,),
            ).fetchall()
            plan = db.execute(
                "SELECT payload_json FROM request_plans WHERE request_id=?", (request_id,)
            ).fetchone()
            if plan is not None:
                terminals = set(json.loads(plan[0])["terminals"])
                instances = [
                    row
                    for row in db.execute(
                        """SELECT t.*,ci.derivative_class,
                    COALESCE(e.logical_key,t.instance_key) AS logical_key FROM request_artifacts rt
                    JOIN instances t ON t.id=rt.instance_id JOIN configuration_lineages ci ON ci.id=t.configuration_lineage_id
                    LEFT JOIN instance_execution e ON e.instance_id=t.id WHERE rt.request_id=?""",
                        (request_id,),
                    )
                    if row["logical_key"] in terminals
                ]
            return dict(request), [dict(row) for row in instances]

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
