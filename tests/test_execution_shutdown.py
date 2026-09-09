"""Terminal execution means that managed processes can no longer read or write inputs."""

import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from nro.orchestration.execution import SubprocessExecutionLauncher, process_group_alive


def test_cancellation_kills_descendant_that_ignores_sigterm(tmp_path):
    output = tmp_path / "process.log"
    code = """import os, signal, time
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('child ready', flush=True)
while True:
    time.sleep(1)
"""
    envelope = SimpleNamespace(execution=SimpleNamespace(command=(sys.executable, "-c", code)))
    launcher = SubprocessExecutionLauncher()
    groups = []
    deadline = time.monotonic() + 5
    try:
        with output.open("w") as log:
            result = launcher.run(
                envelope,
                stdout=log,
                environment=os.environ,
                poll_interval=0.01,
                cancellation_state=lambda: (
                    "child ready" in output.read_text() or time.monotonic() > deadline,
                    True,
                ),
                heartbeat=lambda: None,
                process_started=groups.append,
            )
        assert "child ready" in output.read_text()
        assert result.cancelled and not process_group_alive(groups[0])
    finally:
        if groups:
            try:
                os.killpg(groups[0], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervision_error_stops_processes_before_propagating(tmp_path):
    launcher = SubprocessExecutionLauncher()
    envelope = SimpleNamespace(
        execution=SimpleNamespace(command=(sys.executable, "-c", "import time; time.sleep(60)"))
    )
    groups = []

    def fail():
        raise RuntimeError("lost registry connection")

    with (tmp_path / "process.log").open("w") as log:
        with pytest.raises(RuntimeError, match="lost registry"):
            launcher.run(
                envelope,
                stdout=log,
                environment=os.environ,
                poll_interval=0.01,
                cancellation_state=lambda: (False, False),
                heartbeat=fail,
                process_started=groups.append,
            )
    assert not process_group_alive(groups[0])
