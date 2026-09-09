"""Resolve standard event files by task name without guessing stimulus variants."""

import hashlib
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from nro.configuration.parsing import parse_mapping
from nro.configuration.site import definitions_root
from nro.configuration.store import validate_config_id
from nro.engine.events import validate_events


def task_key(value: str) -> str:
    """Compare complete task names ignoring case and separators, retaining digits."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


@dataclass(frozen=True)
class EventFile:
    """A logical TASK/VARIANT ID and its catalog-owned TSV path."""

    identifier: str
    path: Path

    def snapshot(self) -> tuple[str, dict]:
        """Validate current bytes and return text plus provenance for a review snapshot."""
        text = self.path.read_text(encoding="utf-8")
        validate_events(StringIO(text))
        return text, {
            "catalog_id": self.identifier,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }


class EventStore:
    """Read the task-organized standard-events catalog; never modify it during ingestion."""

    def __init__(self, root: Path | None = None):
        """Use the installation's configuration store unless another absolute root is supplied."""
        self.root = Path(root) if root is not None else definitions_root() / "events"
        if not self.root.is_absolute() or not self.root.is_dir():
            raise ValueError(f"Event store must be an existing absolute directory: {self.root}")
        self.root = self.root.resolve()

    def _entries(self, index: Path) -> tuple[list[str], list[EventFile]]:
        value = parse_mapping(index.read_text(), source=str(index))
        if (
            set(value) != {"tasks", "files"}
            or not isinstance(value["tasks"], list)
            or not all(isinstance(t, str) and task_key(t) for t in value["tasks"])
        ):
            raise ValueError(f"Event index needs tasks and files: {index}")
        if not isinstance(value["files"], dict):
            raise ValueError(f"Event files must be a mapping: {index}")
        entries = []
        task = validate_config_id(index.parent.name, kind="event task")
        for variant, entry in value["files"].items():
            validate_config_id(variant, kind="event variant")
            if not isinstance(entry, dict) or set(entry) != {"path", "source_names"}:
                raise ValueError(f"Event entry needs path and source_names: {index}")
            if (
                not isinstance(entry["path"], str)
                or not isinstance(entry["source_names"], list)
                or not all(isinstance(name, str) for name in entry["source_names"])
            ):
                raise ValueError(f"Event path and source names must be strings: {index}")
            relative = Path(entry["path"])
            path = (self.root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(self.root)
                or path.suffix != ".tsv"
                or not path.is_file()
            ):
                raise ValueError(f"Event file is missing or outside the catalog: {index}")
            if relative.parent != Path(task) or path.parent != index.parent.resolve():
                raise ValueError(f"Event file must belong to its task directory: {index}")
            entries.append(EventFile(f"{task}/{variant}", path))
        return value["tasks"], entries

    def candidates(self, task: str) -> list[EventFile]:
        """Return every variant for exact normalized task-name matches; never use substring matching."""
        if not task_key(task):
            return []
        matches = []
        for index in sorted(self.root.glob("*/index.yml")):
            tasks, entries = self._entries(index)
            if task_key(task) in {task_key(name) for name in tasks}:
                matches.extend(entries)
        return sorted(matches, key=lambda item: item.identifier)

    def resolve(self, identifier: str) -> EventFile:
        """Resolve one explicit TASK/VARIANT selection; missing IDs raise ValueError."""
        if len(identifier.split("/")) != 2:
            raise ValueError("Event IDs must be TASK/VARIANT")
        task, variant = identifier.split("/")
        validate_config_id(task, kind="event task")
        validate_config_id(variant, kind="event variant")
        index = self.root / task / "index.yml"
        if index.is_file():
            for entry in self._entries(index)[1]:
                if entry.identifier == identifier:
                    return entry
        raise ValueError(f"Unknown event ID: {identifier}")
