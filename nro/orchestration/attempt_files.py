"""Name the files in a work item's log directory and update its step ledger.

A work item keeps one log directory, shared by its successive attempts. The
current-step ledger maps each step ID to its latest record. The events file
appends every record, so it also keeps the history.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from nro.engine.io import atomic_write_json


@dataclass(frozen=True)
class AttemptFiles:
    """The files that workers, runners, and status commands share for one work item."""

    directory: Path

    @classmethod
    def for_log(cls, log_path: Path) -> AttemptFiles:
        """Return the files beside a work item's log."""
        return cls(Path(log_path).parent)

    @property
    def log(self) -> Path:
        """Return the work item's current log, replaced by each attempt."""
        return self.directory / "work-item.log"

    @property
    def current_steps(self) -> Path:
        """Return the ledger of each step's latest record."""
        return self.directory / "current-steps.json"

    @property
    def step_events(self) -> Path:
        """Return the append-only record of step events."""
        return self.directory / "step-events.jsonl"

    @property
    def runner_graph(self) -> Path:
        """Return the frozen runner graph written for status reporting."""
        return self.directory / "runner-graph.json"

    @property
    def runner_contract(self) -> Path:
        """Return the runner contract that records completed nodes."""
        return self.directory / "runner-contract.json"


def record_step(
    ledger: Path,
    record: dict,
    *,
    merge_previous: bool = False,
    durable: bool = False,
) -> None:
    """Store one step record in the ledger at `ledger` and append it to the events file.

    With `merge_previous`, a finishing record ("success" or "error") inherits
    the outputs, command, working directory, and reason of the step's earlier
    record when it has none. A "skipping" record does not replace an earlier
    success in the ledger, but it is still appended to the events file. With
    `durable`, the events file is synced to disk before returning.
    """
    key = record["step_id"]
    try:
        current = json.loads(ledger.read_text(encoding="utf-8")) if ledger.is_file() else {}
        if not isinstance(current, dict):
            current = {}
    except (OSError, json.JSONDecodeError):
        current = {}
    previous = current.get(key, {})
    if merge_previous and record["status"] in {"success", "error"} and previous:
        for field in ("outputs", "command", "cwd", "reason"):
            if record.get(field) in (None, [], ""):
                record[field] = previous.get(field)
    if not (
        merge_previous and record["status"] == "skipping" and previous.get("status") == "success"
    ):
        current[key] = record
        atomic_write_json(ledger, current, sort_keys=True, mode=0o664)
    events = AttemptFiles(ledger.parent).step_events
    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    try:
        events.chmod(0o664)
    except PermissionError:
        pass
