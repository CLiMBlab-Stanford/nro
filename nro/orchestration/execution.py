"""Subprocess boundary for execution envelopes claimed by workers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Mapping, Protocol

from nro.orchestration.contracts import ExecutionEnvelope


def process_group_alive(group: int) -> bool:
    """Check a local Linux process group for non-zombie members before releasing inputs."""
    if not Path("/proc/self/stat").is_file():
        raise ShutdownUnconfirmed("Cannot inspect local process groups without /proc")
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] not in {"Z", "X"}:
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError) as error:
            raise ShutdownUnconfirmed(f"Cannot establish process-group shutdown: {path}") from error
    return False


class ShutdownUnconfirmed(RuntimeError):
    """A process group still has live members after cancellation escalation."""


@dataclass(frozen=True)
class ExecutionResult:
    """Process exit status and cancellation flags for a supervised attempt."""

    return_code: int
    cancelled: bool
    scheduler_cancelled: bool


class ExecutionLauncher(Protocol):
    """Launch and supervise one scientific module process."""

    def terminate(self) -> None:
        """Request termination of the launcher's current process, if any."""
        ...

    def run(
        self,
        envelope: ExecutionEnvelope,
        *,
        stdout: IO[str],
        environment: Mapping[str, str],
        poll_interval: float,
        cancellation_state: Callable[[], tuple[bool, bool]],
        heartbeat: Callable[[], None],
        process_started: Callable[[int], None] | None = None,
    ) -> ExecutionResult:
        """Execute an immutable envelope while polling cancellation and renewing heartbeats."""
        ...


class SubprocessExecutionLauncher:
    """Execute locally while keeping cancellation and process groups reliable."""

    def __init__(self) -> None:
        """Create a launcher with no running child process."""
        self.process: subprocess.Popen[str] | None = None

    def terminate(self) -> None:
        """Send SIGTERM to the active process group; ignore an already-ended process."""
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def run(
        self,
        envelope: ExecutionEnvelope,
        *,
        stdout: IO[str],
        environment: Mapping[str, str],
        poll_interval: float,
        cancellation_state: Callable[[], tuple[bool, bool]],
        heartbeat: Callable[[], None],
        process_started: Callable[[int], None] | None = None,
    ) -> ExecutionResult:
        """Launch the command in a new process group and supervise it.

        Merge stdout/stderr into the supplied stream. Poll cancellation and call
        heartbeat at poll_interval seconds. Cancellation escalates from SIGTERM
        to SIGKILL after a 20-second wait. Return the exit and cancellation flags.
        """
        self.process = subprocess.Popen(
            envelope.execution.command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=dict(environment),
        )
        if process_started is not None:
            try:
                process_started(self.process.pid)
            except BaseException:
                self._stop_group()
                raise
        try:
            cancelled = False
            scheduler_cancelled = False
            while self.process.poll() is None:
                cancel, scheduler_cancelled = cancellation_state()
                if cancel:
                    cancelled = True
                    self._stop_group()
                    break
                heartbeat()
                time.sleep(poll_interval)
            return_code = self.process.wait()
            if process_group_alive(self.process.pid):
                self._stop_group()
            self.process = None
            return ExecutionResult(return_code, cancelled, scheduler_cancelled)
        except ShutdownUnconfirmed:
            raise
        except BaseException:
            self._stop_group()
            raise

    def _stop_group(self) -> None:
        self.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 20
        while process_group_alive(self.process.pid):
            if time.monotonic() >= deadline:
                raise ShutdownUnconfirmed(f"Process group {self.process.pid} has not stopped")
            time.sleep(0.05)
        self.process.wait()
