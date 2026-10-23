"""Hold short recovery critical sections with owner records that survive NFS.

A recovery lock is a directory at a fixed path that contains ``owner.json``.
The owner prepares a private directory named after its host, process, and
token, writes its record there, and renames that directory onto the fixed
path. Renaming onto an existing non-empty directory fails, so one contender
wins, and the lock never exists without its owner record. Only ``mkdir`` and
``rename`` provide the exclusion, because advisory locks are unreliable on some
NFS configurations.

A holder that is killed leaves its lock behind. Another process may remove it
once the owner is shown to be gone: on the same host, its process no longer
exists or has a different start time; on another host, the lock is older than
its lease. A directory without a readable owner record, which earlier releases
created, is removed after ``UNOWNED_GRACE_SECONDS``.
"""

from __future__ import annotations

import errno
import os
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from nro.engine.io import atomic_write_json, read_json

PROTOCOL = 1
LEASE_SECONDS = 120.0
UNOWNED_GRACE_SECONDS = 30.0
_OWNER = "owner.json"
_OCCUPIED = frozenset({errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR, errno.ENOTDIR})


def process_start(pid: int) -> str | None:
    """Return a boot-scoped start time for a local process, or None if unknown.

    The value distinguishes a live owner from an unrelated process that reused
    its PID. It is available only where Linux ``/proc`` is mounted.
    """
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii", errors="replace")
        # Field 22 is the start time. The command name in field 2 may contain
        # spaces, so count fields after its closing parenthesis.
        return f"{boot}:{stat.rpartition(')')[2].split()[19]}"
    except (OSError, ValueError, IndexError):
        return None


class RecoveryLock:
    """Exclude other recoverers while one process replaces abandoned state.

    ``acquire`` does not wait. It returns False while another live owner holds
    the lock, and callers retry on their own schedule. A holder should call
    ``owned`` before each destructive step, because a holder that outlives its
    lease may be replaced.
    """

    def __init__(self, path: Path, *, lease_seconds: float = LEASE_SECONDS) -> None:
        """Configure the lock path and the lease, in seconds, for remote owners."""
        self.path = Path(path)
        self.lease_seconds = lease_seconds
        self.token = uuid.uuid4().hex
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self._held = False

    def acquire(self) -> bool:
        """Take the lock, first removing it if its owner is demonstrably gone."""
        if self._held:
            raise RuntimeError(f"Recovery lock is already held: {self.path}")
        label = self.host.split(".", 1)[0] or "host"
        candidate = self.path.with_name(f"{self.path.name}.{label}.{self.pid}.{self.token}")
        candidate.mkdir(mode=0o2775)
        try:
            candidate.chmod(0o2775)
            atomic_write_json(
                candidate / _OWNER,
                {
                    "protocol": PROTOCOL,
                    "token": self.token,
                    "host": self.host,
                    "pid": self.pid,
                    "process_start": process_start(self.pid),
                    "created_at": time.time(),
                },
                sort_keys=True,
                mode=0o664,
                durable=True,
            )
            self._held = self._install(candidate) or (
                self._remove_abandoned() and self._install(candidate)
            )
            return self._held
        finally:
            if not self._held:
                shutil.rmtree(candidate, ignore_errors=True)

    def owned(self) -> bool:
        """Return whether the lock on disk still carries this holder's token."""
        record = _read_owner(self.path)
        return record is not None and record.get("token") == self.token

    def release(self) -> None:
        """Remove the lock if this holder still owns it."""
        if not self._held:
            return
        self._held = False
        if not self.owned():
            return
        released = self.path.with_name(f"{self.path.name}.released-{self.token}")
        try:
            os.rename(self.path, released)
        except FileNotFoundError:
            return
        shutil.rmtree(released, ignore_errors=True)

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _install(self, candidate: Path) -> bool:
        """Rename the prepared owner directory onto the free lock path."""
        if (
            self.path.exists()
            and _read_owner(self.path) is None
            and not _unowned_expired(self.path)
        ):
            # A fresh unowned directory may belong to an older release that
            # creates the lock with mkdir. An empty directory would not block
            # rename, so wait for the grace period before replacing it.
            return False
        try:
            os.rename(candidate, self.path)
        except OSError as error:
            if error.errno not in _OCCUPIED:
                raise
            # An NFS client may retransmit a rename that already succeeded and
            # report the destination as occupied.
            return self.owned()
        return True

    def _remove_abandoned(self) -> bool:
        """Remove a lock whose owner is gone and confirm the removed token."""
        record = _read_owner(self.path)
        if not self._abandoned(record):
            return False
        stale = self.path.with_name(f"{self.path.name}.stale-{uuid.uuid4().hex}")
        try:
            os.rename(self.path, stale)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        removed = _read_owner(stale)
        if (removed or {}).get("token") != (record or {}).get("token"):
            # Another recoverer replaced the stale lock between the check and
            # the rename. Restore it when the path is still free. Otherwise its
            # holder fails its next ownership check and retries.
            try:
                os.rename(stale, self.path)
                return False
            except OSError:
                pass
            shutil.rmtree(stale, ignore_errors=True)
            return False
        shutil.rmtree(stale, ignore_errors=True)
        return True

    def _abandoned(self, record: dict[str, Any] | None) -> bool:
        if record is None:
            return _unowned_expired(self.path)
        if record.get("host") == self.host:
            try:
                pid = int(record["pid"])
            except (KeyError, TypeError, ValueError):
                return _unowned_expired(self.path)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                # The PID belongs to another user. Shared control directories
                # admit several users, so only the start time can tell a live
                # owner from a reused PID.
                pass
            except OSError:
                return False
            started = record.get("process_start")
            return started is not None and process_start(pid) not in {None, started}
        try:
            return time.time() - float(record["created_at"]) > self.lease_seconds
        except (KeyError, TypeError, ValueError):
            return _unowned_expired(self.path)


def _read_owner(path: Path) -> dict[str, Any] | None:
    try:
        record = read_json(path / _OWNER)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
        return None
    return record


def _unowned_expired(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime > UNOWNED_GRACE_SECONDS
    except OSError:
        return False
