"""Subprocess boundary for execution envelopes claimed by workers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, IO, Mapping, Protocol

from nro.orchestration.contracts import ExecutionEnvelope


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
        if self.process is None or self.process.poll() is not None:
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
        cancelled = False
        scheduler_cancelled = False
        while self.process.poll() is None:
            cancel, scheduler_cancelled = cancellation_state()
            if cancel:
                cancelled = True
                self.terminate()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            heartbeat()
            time.sleep(poll_interval)
        return_code = self.process.wait()
        self.process = None
        return ExecutionResult(
            return_code=return_code,
            cancelled=cancelled,
            scheduler_cancelled=scheduler_cancelled,
        )
