"""Test the shared terminal spinner and its registry-wait caller."""

import io
import time

import pytest

from nro.engine import terminal
from nro.orchestration import registry


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def capture(monkeypatch) -> Terminal:
    # Patch inside each test body. Pytest reinstalls its own stderr capture
    # between fixture setup and the test call.
    stream = Terminal()
    monkeypatch.setattr(terminal.sys, "stderr", stream)
    return stream


def frames(rendered: str) -> int:
    return sum(rendered.count(frame) for frame in terminal.BRAILLE_SPINNER_FRAMES)


def test_spinner_advances_on_its_own_schedule(monkeypatch):
    stderr = capture(monkeypatch)
    spinner = terminal.SpinnerLine()
    spinner.update("Working...")
    time.sleep(terminal.SPINNER_INTERVAL_SECONDS * 6)
    spinner.clear()

    assert frames(stderr.getvalue()) >= 4
    assert stderr.getvalue().endswith(terminal.CLEAR_LINE)


def test_spinner_restarts_after_clear(monkeypatch):
    stderr = capture(monkeypatch)
    spinner = terminal.SpinnerLine(interval=0.01)
    spinner.update("First...")
    spinner.clear()
    drawn = frames(stderr.getvalue())
    spinner.update("Second...")
    time.sleep(0.15)
    spinner.clear()

    assert frames(stderr.getvalue()) - drawn >= 3


def test_spinner_is_silent_without_a_terminal(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(terminal.sys, "stderr", stream)
    spinner = terminal.SpinnerLine()
    spinner.update("Working...")
    spinner.clear()

    assert stream.getvalue() == ""


def test_registry_wait_animates_between_slow_polls(tmp_path, monkeypatch):
    stderr = capture(monkeypatch)
    path = tmp_path / "registry.lock"
    path.mkdir()
    monkeypatch.setattr(registry, "_WAIT_NOTICE_SECONDS", 0.0)
    lock = registry.RegistryLock(path, tmp_path / "registry.lock.recovery", timeout=0.6)

    with pytest.raises(registry.RegistryLockTimeout):
        lock.acquire()

    rendered = stderr.getvalue()
    assert "Waiting for registry access..." in rendered
    assert frames(rendered) >= 5
    assert rendered.endswith(terminal.CLEAR_LINE)
