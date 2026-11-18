"""Run stored nro commands with their installation's current interpreter.

Planning records the planner's interpreter in each execution recipe. An
installation may replace its dependency environment at any time, which can
remove that interpreter while work is still queued. The interpreter is not part
of a work item's scientific contract, so each claim runs the command with the
current environment of the installation that owns the recorded interpreter.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Sequence

INSTALLATION_RECORD = ".nro-installation.json"
# Installation records change only during maintenance, so a short cache spares
# a shared-filesystem read on every claim in a burst.
_CACHE_SECONDS = 30.0
_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()


def _is_nro_python_command(command: Sequence[str]) -> bool:
    if len(command) < 2:
        return False
    if Path(command[1]).name == "source_launcher.py":
        return True
    return len(command) >= 3 and command[1] == "-m" and command[2].startswith("nro.")


def current_interpreter(interpreter: str) -> str:
    """Return the current interpreter of the installation that owns ``interpreter``.

    The owner is the nearest parent directory whose installation record names
    that directory as its checkout. Without such a record, or when the recorded
    environment has no interpreter, the given path is returned unchanged.
    """
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(interpreter)
        if cached is not None and now - cached[0] < _CACHE_SECONDS:
            return cached[1]
    resolved = interpreter
    for parent in Path(interpreter).parents:
        record_path = parent / INSTALLATION_RECORD
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            break
        if record.get("checkout") == str(parent) and record.get("environment"):
            candidate = Path(record["environment"]) / "bin/python"
            if candidate.is_file():
                resolved = str(candidate)
        break
    with _cache_lock:
        _cache[interpreter] = (now, resolved)
    return resolved


def with_current_interpreter(command: Sequence[str]) -> tuple[str, ...]:
    """Return an nro Python command bound to its installation's current interpreter."""
    command = tuple(str(value) for value in command)
    if not _is_nro_python_command(command):
        return command
    return (current_interpreter(command[0]), *command[1:])
