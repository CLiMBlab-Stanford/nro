"""Durable ingestion records, serialized by the central registry lock.

The ingestion directory survives derivative registry repair. Records are
atomic JSON documents; worker claims share the lock used by derivative claims.
"""

import getpass
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from nro.engine.io import atomic_write_text
from nro.orchestration.branches import BranchPaths

from .config import bids_label, identifier
from .identity import identity_issues

ACTIVE = {"queued", "running"}
REVIEW_LEASE_SECONDS = 120.0


class ReviewBusyError(ValueError):
    """Report that another terminal currently owns this session's review."""


class IngestionStore:
    """Read shared ingestion state and make locked, revision-checked changes."""

    def __init__(
        self,
        registry,
        *,
        branch_paths: BranchPaths | None = None,
        branch: str | None = None,
        execution: dict | None = None,
    ):
        """Bind production or isolated debugging records under the central lock.

        The caller authorizes branch_paths. Development data never become raw
        inputs for scientific planning, which retains the shared BIDS root.
        """
        self.registry = registry
        self.execution = execution
        self.branch_paths = branch_paths
        if branch_paths is not None and branch is not None and branch_paths.branch != branch:
            raise ValueError("Ingestion namespace differs from publication paths")
        self.branch = branch_paths.branch if branch_paths is not None else (branch or "main")
        from nro.orchestration.control_paths import ControlPaths

        control = ControlPaths(registry.paths.control)
        if branch_paths is not None and branch_paths.bids != registry.paths.bids_root.resolve():
            raise ValueError("Ingestion context differs from the shared raw BIDS root")
        self.root = (
            control.ingestion
            if self.branch == "main"
            else control.branch(self.branch) / "ingestion"
        )

    def project_root(self, project: str) -> Path:
        """Resolve a publication destination without changing scientific source paths."""
        identifier(project)
        if self.branch != "main" and self.branch_paths is None:
            raise ValueError("Debug publication requires authorized branch paths")
        root = (
            self.registry.paths.bids_root / project
            if self.branch_paths is None
            else self.branch_paths.output_project(project)
        )
        if root.resolve() != root:
            raise ValueError("BIDS publication root cannot be redirected through a symlink")
        return root

    def require_record(self, record: dict) -> None:
        """Reject records submitted to a different publication namespace."""
        if record.get("branch") != self.branch:
            raise ValueError("Ingestion record belongs to a different branch")

    def rows(self) -> list[dict]:
        """Read atomically published records without creating state or contacting servers."""
        return [json.loads(path.read_text()) for path in sorted(self.root.glob("*.json"))]

    def get(self, request_id: str) -> dict:
        """Read one persistent request by its stable ID."""
        return json.loads((self.root / f"{identifier(request_id)}.json").read_text())

    def write_locked(self, record: dict) -> None:
        """Publish a record; callers must hold the registry lock."""
        from .paths import secure_directory

        self.require_record(record)
        secure_directory(self.root)
        record["updated"] = time.time()
        path = self.root / f"{identifier(record['id'])}.json"
        atomic_write_text(
            path, json.dumps(record, indent=2, sort_keys=True) + "\n", mode=0o660, durable=True
        )

    def create(
        self,
        *,
        server: str,
        remote_session: str,
        project: str,
        participant: str | None = None,
        session: str | None = None,
        config: dict,
        replace: bool = False,
    ) -> dict:
        """Register a remote session even when its BIDS identity is not yet known.

        Return an identical unfinished request without duplicating work. Missing
        labels must be filled through leased review before BIDS organization.
        A source session retains its first destination across projects,
        cancellation, and re-bidsification. Known labels carry into new attempts.
        """
        for value in (server, remote_session, project):
            identifier(value)
        if self.branch != "main" and self.branch_paths is None:
            raise ValueError("Debug creation requires authorized branch paths")
        for value in (participant, session):
            if value is not None:
                bids_label(value)
        config = deepcopy(config)
        if self.branch != "main":
            config["staging"] = str(self.root / "staging")
        if (
            Path(config["staging"])
            .resolve()
            .is_relative_to(self.registry.paths.bids_root.resolve())
        ):
            raise ValueError("Shared ingestion staging must be outside the BIDS tree")
        with self.registry.connection(write=True):
            matches = [
                r
                for r in self.rows()
                if (r["server"], r["remote_session"]) == (server, remote_session)
            ]
            for previous in matches:
                if previous["project"] != project:
                    raise ValueError(
                        f"This source session belongs to BIDS project {previous['project']}; "
                        "re-bidsification cannot reassign it"
                    )
                for field, value in (("participant", participant), ("session", session)):
                    if (
                        previous[field] is not None
                        and value is not None
                        and previous[field] != value
                    ):
                        raise ValueError(
                            "This source session already has a different BIDS destination"
                        )
            for field in ("participant", "session"):
                known = {r[field] for r in matches if r[field] is not None}
                if len(known) > 1:
                    raise ValueError(
                        "This source session has conflicting registered BIDS destinations"
                    )
                if known:
                    if field == "participant":
                        participant = next(iter(known))
                    else:
                        session = next(iter(known))
            unfinished = [r for r in matches if r["state"] not in {"published", "cancelled"}]
            if unfinished:
                if any(
                    value is not None and unfinished[-1][key] != value
                    for key, value in (("participant", participant), ("session", session))
                ):
                    raise ValueError(
                        "Remote session already has a different unfinished BIDS mapping"
                    )
                return unfinished[-1]
            if any(r["state"] == "published" for r in matches) and not replace:
                raise ValueError("This remote session is already published; select --rebidsify")
            request_id = uuid.uuid4().hex
            record = dict(
                id=request_id,
                server=server,
                remote_session=remote_session,
                branch=self.branch,
                project=project,
                participant=participant,
                session=session,
                config=deepcopy(config),
                state="queued",
                stage="inspect",
                revision=0,
                executor_uid=os.getuid(),
                worker=None,
                issues=[],
                acquisitions=[],
                replace=replace,
                approval=None,
                created=time.time(),
            )
            if self.execution is not None:
                record["execution"] = deepcopy(self.execution)
            self._validate_destination_locked(record)
            self.write_locked(record)
            return record

    def _validate_destination_locked(self, record: dict) -> None:
        if identity_issues(record):
            return
        if any(
            r["id"] != record["id"]
            and all(r[key] == record[key] for key in ("project", "participant", "session"))
            and r["state"] not in {"published", "cancelled"}
            for r in self.rows()
        ):
            raise ValueError("Another unfinished ingestion request owns this destination")
        target = (
            self.project_root(record["project"])
            / f"sub-{record['participant']}"
            / f"ses-{record['session']}"
        )
        if target.exists() and not record["replace"]:
            raise ValueError("This destination already exists; reconcile it using --rebidsify")

    def _review_path(self, request_id: str) -> Path:
        return self.root / "reviews" / f"{identifier(request_id)}.json"

    def _review(self, request_id: str) -> dict | None:
        path = self._review_path(request_id)
        return json.loads(path.read_text()) if path.is_file() else None

    def _write_review(self, request_id: str, lease: dict) -> None:
        from .paths import secure_directory

        path = self._review_path(request_id)
        secure_directory(path.parent)
        atomic_write_text(path, json.dumps(lease), mode=0o660, durable=True)

    def _require_review(self, request_id: str, token: str) -> dict:
        lease = self._review(request_id)
        if not lease or lease["token"] != token or lease["expires"] <= time.time():
            raise ValueError("Review lease expired or ownership changed; reopen this session")
        return lease

    def acquire_review(self, request_id: str, *, seconds: float = REVIEW_LEASE_SECONDS) -> str:
        """Lease one registered session, rejecting active reviewers and executing workers.

        Expired leases can be replaced. The returned token is required for
        decisions and event-file writes; selecting sessions does not lease them.
        """
        if seconds <= 0:
            raise ValueError("Review lease duration must be positive")
        with self.registry.connection(write=True):
            record = self.get(request_id)
            lease = self._review(request_id)
            if lease and lease["expires"] > time.time():
                raise ReviewBusyError(
                    f"Session is being reviewed by {lease['user']} on {lease['hostname']} (PID {lease['pid']}); skipped"
                )
            if record["state"] in {"running", "published", "cancelled"}:
                raise ReviewBusyError(f"Session is {record['state']}; review skipped")
            token = uuid.uuid4().hex
            self._write_review(
                request_id,
                dict(
                    token=token,
                    user=getpass.getuser(),
                    hostname=socket.gethostname(),
                    pid=os.getpid(),
                    expires=time.time() + seconds,
                ),
            )
            return token

    def renew_review(
        self, request_id: str, token: str, *, seconds: float = REVIEW_LEASE_SECONDS
    ) -> None:
        """Extend an unexpired lease without reviving lost ownership or changing decisions."""
        if seconds <= 0:
            raise ValueError("Review lease duration must be positive")
        with self.registry.connection(write=True):
            lease = self._require_review(request_id, token)
            lease["expires"] = time.time() + seconds
            self._write_review(request_id, lease)

    def release_review(self, request_id: str, token: str) -> None:
        """Release this terminal's lease, leaving any replacement owner's lease intact."""
        with self.registry.connection(write=True):
            lease = self._review(request_id)
            if lease and lease["token"] == token:
                self._review_path(request_id).unlink()

    @contextmanager
    def review_session(self, request_id: str, *, seconds: float = REVIEW_LEASE_SECONDS):
        """Renew one session's lease while prompting and release it on return or exception.

        No registry lock is held while waiting for input. An abandoned terminal
        loses ownership after two minutes without a successful renewal.
        """
        token = self.acquire_review(request_id, seconds=seconds)
        stopped, lost = threading.Event(), threading.Event()

        def heartbeat():
            while not stopped.wait(seconds / 4):
                try:
                    self.renew_review(request_id, token, seconds=seconds)
                except Exception:
                    lost.set()
                    return

        thread = threading.Thread(target=heartbeat, name="bidsify-review", daemon=True)
        try:
            thread.start()
            yield token
            if lost.is_set():
                raise ValueError("Review lease renewal failed; reopen this session")
        finally:
            stopped.set()
            if thread.ident is not None:
                thread.join(timeout=1)
            self.release_review(request_id, token)

    def update(
        self,
        record: dict,
        *,
        expected_revision: int,
        review_token: str,
        event_files: dict[str, str] | None = None,
    ) -> dict:
        """Save decisions and event snapshots under a valid session lease and revision.

        event_files maps acquisition IDs to reviewed TSV text. Each snapshot
        has a content-addressed name, so a failed record write cannot replace
        an earlier decision's file. Rejected revisions/leases write no files.
        Missing BIDS labels can be filled before organization; resolved labels
        cannot be reassigned. Completing a destination checks its reservation.
        """
        with self.registry.connection(write=True):
            current = self.get(record["id"])
            self._require_review(record["id"], review_token)
            if current["revision"] != expected_revision:
                raise ValueError("Request changed concurrently; reopen it before editing")
            if current["state"] in {"running", "published", "cancelled"}:
                raise ValueError("Cannot edit an executing or finished ingestion request")
            record = deepcopy(record)
            for key in ("id", "server", "remote_session", "project", "config", "branch"):
                if record[key] != current[key]:
                    raise ValueError("Review cannot change request identity or its configuration")
            if record.get("execution") != current.get("execution"):
                raise ValueError("Review cannot replace an execution pin directly")
            changed = any(record[key] != current[key] for key in ("participant", "session"))
            if changed:
                if (
                    current["stage"] not in {"inspect", "prepare", "convert"}
                    or current["state"] == "awaiting_approval"
                ):
                    raise ValueError("BIDS labels cannot change after organization")
                for key in ("participant", "session"):
                    if current[key] is not None and record[key] != current[key]:
                        raise ValueError("Review cannot replace a resolved BIDS label")
                self._validate_destination_locked(record)
                record["approval"] = None
            if record["stage"] == "publish" or record["state"] == "awaiting_approval":
                if identity_issues(record):
                    raise ValueError("Resolve BIDS identity before publication")
            acquisitions = {item["id"]: item for item in record["acquisitions"]}
            if set(event_files or {}) - acquisitions.keys():
                raise ValueError("Events must belong to an acquisition in this request")
            for acquisition, text in (event_files or {}).items():
                from .paths import secure_directory

                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                provenance = acquisitions[acquisition].get("events_source")
                if provenance and provenance.get("sha256") != digest:
                    raise ValueError(
                        "Event catalog provenance does not match the reviewed snapshot"
                    )
                destination = (
                    Path(current["config"]["staging"])
                    / current["id"]
                    / "events"
                    / f"{identifier(acquisition)}-{digest}.tsv"
                )
                secure_directory(destination.parent)
                if destination.is_symlink():
                    raise ValueError("Event snapshots cannot be symbolic links")
                if destination.exists():
                    if destination.read_text() != text:
                        raise ValueError("An existing event snapshot has changed")
                else:
                    atomic_write_text(destination, text, mode=0o660, durable=True)
                acquisitions[acquisition]["events"] = str(destination)
            record["revision"] += 1
            record["executor_uid"] = os.getuid()
            if record["state"] == "queued" and self.execution is not None:
                record["execution"] = deepcopy(self.execution)
            self.write_locked(record)
            return record

    def summary(self, memory_gb: int | None = None) -> tuple[int, int, int]:
        """Return active count, runnable count, and requested concurrency under the caller's lock."""
        rows = self.rows()
        active = sum(r["state"] == "running" for r in rows)
        if self.branch != "main":
            from nro.orchestration.branch_store import BranchStore

            if (
                BranchStore(self.registry.paths.control)
                .read()
                .topology.records[self.branch]
                .retired
            ):
                return active, 0, 0
        ready = sum(
            r["state"] == "queued"
            and (self.branch == "main" or bool(r.get("execution")))
            and r["executor_uid"] == os.getuid()
            and not self._review_active(r["id"])
            and (memory_gb is None or r["config"]["memory_gb"] <= memory_gb)
            for r in rows
        )
        limit = max(
            (
                r["config"]["concurrency"]
                for r in rows
                if r["state"] in ACTIVE and (self.branch == "main" or r.get("execution"))
            ),
            default=0,
        )
        return active, ready, limit

    def admit_pending(self, request_id: str) -> dict:
        """Pin an unclaimed queue when its owner reopens the reviewer."""
        with self.registry.connection(write=True):
            row = self.get(request_id)
            if (
                self.execution
                and not row.get("execution")
                and row["state"] == "queued"
                and row["executor_uid"] == os.getuid()
                and not self._review_active(request_id)
            ):
                row.update(execution=deepcopy(self.execution), revision=row["revision"] + 1)
                self.write_locked(row)
            return row

    def _review_active(self, request_id: str) -> bool:
        lease = self._review(request_id)
        return bool(lease and lease["expires"] > time.time())

    def claim(self, worker: str, memory_gb: int) -> dict | None:
        """Claim one stage under the same global concurrency lock as derivative work."""
        if self.branch != "main":
            from nro.orchestration.branch_store import BranchStore

            owner = BranchStore(self.registry.paths.control).read().topology.records[self.branch]
            if owner.retired:
                return None
        with self.registry.connection(write=True) as db:
            owner = db.execute("SELECT state FROM workers WHERE id=?", (worker,)).fetchone()
            if owner is None or owner["state"] == "shutdown_requested":
                return None
            if db.execute("SELECT 1 FROM metadata WHERE key='maintenance_mode'").fetchone():
                return None
            from .index import IngestionIndex

            index = IngestionIndex(self.registry)
            active, _, limit = index.summary(memory_gb)
            if any(r["state"] == "running" and r["worker"] == worker for r in index.rows()):
                return None
            limit = max(
                limit,
                db.execute(
                    "SELECT COALESCE(MAX(concurrency),0) FROM requests WHERE state='active'"
                ).fetchone()[0],
            )
            active += db.execute(
                "SELECT COUNT(*) FROM attempts WHERE state IN ('queued','running','cancel_requested')"
            ).fetchone()[0]
            if active >= limit:
                return None
            for row in self.rows():
                if (
                    row["state"] != "queued"
                    or row["executor_uid"] != os.getuid()
                    or row["config"]["memory_gb"] > memory_gb
                    or self._review_active(row["id"])
                ):
                    continue
                if self.branch != "main" and not row.get("execution"):
                    continue
                row.update(
                    state="running",
                    worker=worker,
                    hostname=socket.gethostname(),
                    revision=row["revision"] + 1,
                )
                self.write_locked(row)
                return row
        return None

    def finish(
        self, request_id: str, worker: str, *, state: str, changes: dict | None = None
    ) -> None:
        """Finish only the stage still owned by this worker."""
        with self.registry.connection(write=True):
            row = self.get(request_id)
            if row["state"] != "running" or row["worker"] != worker:
                raise ValueError("Ingestion worker no longer owns this request")
            row.update(changes or {})
            row.update(state=state, worker=None, revision=row["revision"] + 1)
            self.write_locked(row)

    def recover_locked(self, dead_workers: set[str]) -> int:
        """Release attempts owned by workers already established to be dead."""
        count = 0
        for row in self.rows():
            if row["state"] == "running" and row["worker"] in dead_workers:
                row.update(state="interrupted", worker=None, revision=row["revision"] + 1)
                self.write_locked(row)
                count += 1
        return count
