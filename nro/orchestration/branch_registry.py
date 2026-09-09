"""Branch-owned scientific records stored beside the shared scheduler registry."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Mapping

from nro.configuration.store import fingerprint
from nro.orchestration.branches import BranchRecord
from nro.orchestration.contracts import InstanceSpec
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryLock, ensure_shared_directory
from nro.orchestration.workflow_registry import WORKFLOW_SCHEMA, WorkflowRegistry

APPLICATION_ID = 0x4E524F42  # NROB: branch state, distinct from the scheduler database.
SCHEMA_VERSION = 2
SCHEMA = (
    """
CREATE TABLE identity (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE instances (
    instance_key TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    contract_json TEXT NOT NULL,
    contract_fingerprint TEXT NOT NULL,
    observation_json TEXT
);
"""
    + WORKFLOW_SCHEMA
)


@dataclass(frozen=True)
class BranchInstance:
    """A scientific contract revision and optional evidence about its artifact.

    Observations are supplied by the scientific validator. They are not worker
    state, scheduling authority, or proof that outputs still exist.
    """

    key: str
    revision: int
    contract: dict
    contract_fingerprint: str
    observation: dict | None


class BranchRegistry(WorkflowRegistry):
    """Store scientific state for one globally registered branch.

    All checkouts use the same centrally derived database path. This database
    contains no worker pool, execution claims, or concurrency settings. Registry
    identity, schema versions, and checkout locations are not scientific inputs.
    Obtain handles through BranchStore when resolving a checkout's authority.
    """

    def __init__(self, control: Path, record: BranchRecord, *, lock_timeout: float = 120) -> None:
        """Bind the canonical location and expected registration without creating files."""
        self.control = Path(control).expanduser().resolve()
        self.record = record
        self.root = ControlPaths(self.control).branch(record.name)
        self.paths = SimpleNamespace(workflows=self.root / "workflows")
        self.database = self.root / "registry.sqlite3"
        self.lock_timeout = lock_timeout

    def _location(self) -> None:
        if self.root.resolve() != self.root or self.database.resolve() != self.database:
            raise ValueError("Branch registry location cannot be redirected through a symlink")

    def _identity(self) -> dict[str, str]:
        return {
            "branch": self.record.name,
            "registry_id": self.record.registry_id,
            "scheduler_control": str(self.control),
        }

    def _lock(self) -> RegistryLock:
        self._location()
        return RegistryLock(
            self.root / "registry.lock",
            self.root / "registry.recovery-lock",
            timeout=self.lock_timeout,
        )

    def _connect(self, *, write: bool = False, check_schema: bool = True) -> sqlite3.Connection:
        self._location()
        if not self.database.is_file():
            raise FileNotFoundError(f"Missing registered branch database: {self.database}")
        uri = self.database.as_uri() + ("?mode=rw" if write else "?mode=ro")
        db = sqlite3.connect(uri, uri=True, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            if db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                raise ValueError(f"Not an nro branch registry: {self.database}")
            identity = dict(db.execute("SELECT key,value FROM identity"))
            if identity != self._identity():
                raise ValueError(
                    f"Branch registry identity does not match its central registration: {self.database}"
                )
            if check_schema and db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported scientific registry schema for branch {self.record.name}"
                )
            if write:
                db.execute("PRAGMA journal_mode=DELETE")
                db.execute("PRAGMA synchronous=FULL")
            return db
        except BaseException:
            db.close()
            raise

    def validate_identity(self) -> None:
        """Verify the registration binding without interpreting branch-specific tables."""
        with self._lock():
            self._connect(check_schema=False).close()

    def initialize(self) -> None:
        """Create an empty scientific registry atomically, or validate an existing binding.

        Never replace or merge an existing database. The central registration
        publisher calls this before making a new branch visible.
        """
        self._location()
        ensure_shared_directory(self.root)
        with self._lock():
            if self.database.exists():
                self._connect(check_schema=False).close()
                return
            temporary = self.root / f".initializing-{uuid.uuid4().hex}.sqlite3"
            try:
                db = sqlite3.connect(temporary)
                try:
                    db.execute("PRAGMA journal_mode=DELETE")
                    db.execute("PRAGMA synchronous=FULL")
                    db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    db.executescript(SCHEMA)
                    db.executemany("INSERT INTO identity VALUES (?,?)", self._identity().items())
                    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    db.commit()
                finally:
                    db.close()
                temporary.chmod(0o664)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                temporary.rename(self.database)
                directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock():
            db = self._connect(write=write)
            try:
                if write:
                    db.execute("BEGIN IMMEDIATE")
                yield db
                if write:
                    db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    @staticmethod
    def _decode(row: sqlite3.Row) -> BranchInstance:
        return BranchInstance(
            row["instance_key"],
            row["revision"],
            json.loads(row["contract_json"]),
            row["contract_fingerprint"],
            json.loads(row["observation_json"]) if row["observation_json"] else None,
        )

    def instances(self) -> tuple[BranchInstance, ...]:
        """Read this branch's scientific records without consulting or changing the pool."""
        with self._connection() as db:
            return tuple(
                self._decode(row)
                for row in db.execute("SELECT * FROM instances ORDER BY instance_key")
            )

    def record_instance(
        self,
        key: str,
        contract: Mapping,
        *,
        expected_revision: int | None,
    ) -> BranchInstance:
        """Create or update a contract, rejecting edits based on an outdated revision.

        The caller supplies the compiled scientific contract, not raw user
        configuration or an execution recipe. None means the caller expects
        no existing record. Equivalent contracts
        retain their revision and observations. A substantive change increments
        the revision and clears observations; this method does not schedule work.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("Expected a nonempty instance key")
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision < 1
        ):
            raise ValueError("Expected a positive instance revision or None")
        value = dict(contract)
        serialized = json.dumps(value, sort_keys=True, allow_nan=False)
        digest = fingerprint(value)
        with self._connection(write=True) as db:
            row = db.execute("SELECT * FROM instances WHERE instance_key=?", (key,)).fetchone()
            current = row["revision"] if row else None
            if current != expected_revision:
                raise ValueError(
                    "Scientific instance changed; read the current revision before retrying"
                )
            if row and row["contract_fingerprint"] == digest:
                return self._decode(row)
            revision = (current or 0) + 1
            db.execute(
                "INSERT INTO instances VALUES (?,?,?,?,NULL) ON CONFLICT(instance_key) "
                "DO UPDATE SET revision=excluded.revision, contract_json=excluded.contract_json, "
                "contract_fingerprint=excluded.contract_fingerprint, observation_json=NULL",
                (key, revision, serialized, digest),
            )
            return self._decode(
                db.execute("SELECT * FROM instances WHERE instance_key=?", (key,)).fetchone()
            )

    def record_observation(self, key: str, observation: Mapping, *, expected_revision: int) -> None:
        """Attach validator evidence only if the observed contract is still current."""
        serialized = json.dumps(dict(observation), sort_keys=True, allow_nan=False)
        with self._connection(write=True) as db:
            result = db.execute(
                "UPDATE instances SET observation_json=? WHERE instance_key=? AND revision=?",
                (serialized, key, expected_revision),
            )
            if result.rowcount != 1:
                raise ValueError("Scientific instance changed before its observation was recorded")

    def record_observations(self, observations: Mapping[str, tuple[int, Mapping]]) -> None:
        """Attach assessment results to specified contract revisions atomically."""
        if not observations:
            return
        with self._connection(write=True) as db:
            for key, (revision, observation) in observations.items():
                result = db.execute(
                    "UPDATE instances SET observation_json=? WHERE instance_key=? AND revision=?",
                    (
                        json.dumps(dict(observation), sort_keys=True, allow_nan=False),
                        key,
                        revision,
                    ),
                )
                if result.rowcount != 1:
                    raise ValueError(
                        "Scientific instance changed before its observation was recorded"
                    )

    def record_graph(
        self,
        instances: tuple[InstanceSpec, ...],
        *,
        expected_revisions: Mapping[str, int | None],
    ) -> tuple[BranchInstance, ...]:
        """Atomically record a compiled graph without making storage location scientific.

        Supply the revision read for every submitted instance. A concurrent change
        rejects the entire batch. Equivalent contracts preserve their observations;
        execution recipes and resource requirements belong to the scheduler handoff.
        """
        from nro.orchestration.artifact_resolution import scientific_contracts

        contracts = scientific_contracts(instances)
        if set(contracts) != set(expected_revisions):
            raise ValueError("Expected revisions must cover exactly the submitted graph")
        with self._connection(write=True) as db:
            current = {}
            for key in contracts:
                row = db.execute("SELECT * FROM instances WHERE instance_key=?", (key,)).fetchone()
                if (row["revision"] if row else None) != expected_revisions[key]:
                    raise ValueError(
                        "Scientific graph changed; read current revisions before retrying"
                    )
                current[key] = row
            for key, value in contracts.items():
                digest = fingerprint(value)
                row = current[key]
                if row and row["contract_fingerprint"] == digest:
                    continue
                db.execute(
                    """INSERT INTO instances VALUES (?,?,?,?,NULL)
                    ON CONFLICT(instance_key) DO UPDATE SET revision=excluded.revision,
                    contract_json=excluded.contract_json, contract_fingerprint=excluded.contract_fingerprint,
                    observation_json=NULL""",
                    (
                        key,
                        (row["revision"] if row else 0) + 1,
                        json.dumps(value, sort_keys=True, allow_nan=False),
                        digest,
                    ),
                )
            return tuple(
                self._decode(
                    db.execute("SELECT * FROM instances WHERE instance_key=?", (key,)).fetchone()
                )
                for key in contracts
            )

    def connection(self, *, write: bool = False):
        """Open a serialized scientific transaction; this store has no scheduler tables."""
        return self._connection(write=write)

    def rebuild(self, workflows: list[dict], instances: list[dict]) -> None:
        """Replace scientific state from admitted records, keeping a recovery copy.

        The caller must reserve branch maintenance and stop its attempts first.
        This does not remove artifacts, execution history, configuration snapshots,
        or another branch's records. Unadmitted observations are discarded.
        """
        self._location()
        ensure_shared_directory(self.root)
        with self._lock():
            temporary = self.root / f".rebuilding-{uuid.uuid4().hex}.sqlite3"
            try:
                db = sqlite3.connect(temporary)
                try:
                    db.executescript(SCHEMA)
                    db.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    db.executemany("INSERT INTO identity VALUES (?,?)", self._identity().items())
                    for payload in workflows:
                        for table, rows in (
                            ("configuration_lineages", payload["lineages"]),
                            ("configuration_lineage_dependencies", payload["dependencies"]),
                            ("workflow_revisions", [payload["revision"]]),
                            ("workflow_bindings", payload["bindings"]),
                        ):
                            columns = [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
                            for row in rows:
                                db.execute(
                                    f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) "
                                    f"VALUES ({','.join('?' for _ in columns)})",
                                    [row[key] for key in columns],
                                )
                    for row in instances:
                        db.execute(
                            "INSERT INTO instances VALUES (?,?,?,?,NULL)",
                            (
                                row["key"],
                                row["revision"],
                                json.dumps(row["contract"], sort_keys=True),
                                fingerprint(row["contract"]),
                            ),
                        )
                    import yaml

                    from nro.orchestration.registry import utcnow

                    for path in sorted(self.paths.workflows.glob("*/*_workflow.yml")):
                        snapshot = yaml.safe_load(path.read_text())
                        existing = db.execute(
                            "SELECT definition_fingerprint FROM workflow_revisions WHERE workflow_id=? AND revision=?",
                            (snapshot["workflow_id"], snapshot["revision"]),
                        ).fetchone()
                        if existing:
                            if existing[0] != snapshot["definition_fingerprint"]:
                                raise ValueError(
                                    "Stored workflow snapshot conflicts with admitted scientific records"
                                )
                            continue
                        resolved = dict(
                            selections=snapshot["selections"],
                            configurations={
                                key: value["resolved"]
                                for key, value in snapshot["configurations"].items()
                            },
                        )
                        db.execute(
                            """INSERT INTO workflow_revisions(workflow_id,revision,definition_fingerprint,
                            source_path,resolved_yaml,created_at) VALUES (?,?,?,?,?,?)""",
                            (
                                snapshot["workflow_id"],
                                snapshot["revision"],
                                snapshot["definition_fingerprint"],
                                str(path),
                                yaml.safe_dump(resolved),
                                utcnow(),
                            ),
                        )
                    db.commit()
                finally:
                    db.close()
                temporary.chmod(0o664)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                backup = self.root / "registry-before-repair.sqlite3"
                if backup.is_symlink():
                    raise ValueError("Repair backup cannot be a symlink")
                if self.database.exists():
                    shutil.copy2(self.database, backup)
                os.replace(temporary, self.database)
                descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
