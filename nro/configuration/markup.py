"""Compile subject-specific source-BIDS inclusion and exclusion records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from nro.configuration.parsing import DefinitionError, parse_mapping
from nro.configuration.site import definitions_root
from nro.configuration.store import validate_config_id

SOURCE_MARKUP_ENV = "NRO_SOURCE_MARKUP"
_FIELDS = frozenset({"T1w", "T2w", "exclude"})


def _paths(value, *, location: str, list_only: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if list_only and not isinstance(value, list):
        raise DefinitionError(f"{location} must be a list of paths")
    items = value if isinstance(value, list) else [value]
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise DefinitionError(f"{location} must be a path or list of paths")
    result = []
    for raw in items:
        path = PurePosixPath(raw.strip())
        if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
            raise DefinitionError(f"{location} paths must be relative to the subject directory")
        normalized = path.as_posix()
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _contains(parent: str, child: str) -> bool:
    """Return whether one normalized relative path contains another."""
    parent_path, child_path = PurePosixPath(parent), PurePosixPath(child)
    return parent_path == child_path or parent_path in child_path.parents


@dataclass(frozen=True)
class SubjectMarkup:
    """Resolved source paths selected for one BIDS participant."""

    markup_id: str | None
    project: str
    subject_dir: Path
    t1w: tuple[Path, ...] = ()
    t2w: tuple[Path, ...] = ()
    excluded: tuple[Path, ...] = ()

    def is_excluded(self, path: Path) -> bool:
        """Return whether a path equals or descends from an excluded BIDS path."""
        candidate = Path(path).expanduser().absolute()
        return any(candidate == root or candidate.is_relative_to(root) for root in self.excluded)

    def filter(self, paths: Iterable[Path]) -> tuple[Path, ...]:
        """Remove excluded paths while preserving input order."""
        return tuple(path for path in paths if not self.is_excluded(path))

    def as_dict(self) -> dict:
        """Serialize the immutable subject selection stored in artifact contracts."""
        return {
            "id": self.markup_id,
            "project": self.project,
            "subject_dir": str(self.subject_dir),
            "T1w": [str(path) for path in self.t1w],
            "T2w": [str(path) for path in self.t2w],
            "exclude": [str(path) for path in self.excluded],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "SubjectMarkup":
        """Validate and restore a captured subject selection."""
        if not isinstance(value, Mapping) or set(value) != {
            "id",
            "project",
            "subject_dir",
            "T1w",
            "T2w",
            "exclude",
        }:
            raise ValueError("Invalid captured source markup")
        if value["id"] is not None and not isinstance(value["id"], str):
            raise ValueError("Invalid captured markup ID")
        if not isinstance(value["project"], str) or not value["project"]:
            raise ValueError("Invalid captured markup project")
        subject_dir = Path(value["subject_dir"]).expanduser().absolute()
        groups = {}
        for key in ("T1w", "T2w", "exclude"):
            if not isinstance(value[key], list) or any(
                not isinstance(item, str) for item in value[key]
            ):
                raise ValueError(f"Invalid captured markup {key} paths")
            paths = tuple(Path(item).expanduser().absolute() for item in value[key])
            if any(not path.is_relative_to(subject_dir) for path in paths):
                raise ValueError("Captured markup path escapes its BIDS subject directory")
            groups[key] = paths
        return cls(
            value["id"],
            value["project"],
            subject_dir,
            groups["T1w"],
            groups["T2w"],
            groups["exclude"],
        )


class MarkupStore:
    """Load named markup documents from a definitions store."""

    def __init__(self, root: Path | None = None) -> None:
        """Select an explicit definitions root or the configured site root."""
        self.root = Path(root).expanduser().resolve() if root is not None else definitions_root()

    def path(self, markup_id: str) -> Path:
        """Return the required path for a named markup document."""
        identifier = validate_config_id(markup_id, kind="markup")
        path = self.root / "markup" / f"{identifier}_markup.yml"
        if path.is_file():
            return path
        if identifier == "main":
            packaged = Path(__file__).parent / "starters" / "markup" / "main_markup.yml"
            if packaged.is_file():
                return packaged
        raise FileNotFoundError(f"Markup file was not found: {path}")

    def load(self, markup_id: str) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
        """Compile one document into canonical participant records."""
        path = self.path(markup_id)
        try:
            document = parse_mapping(path.read_text(encoding="utf-8"), source=str(path))
        except (OSError, DefinitionError) as error:
            raise ValueError(str(error)) from error
        return compile_markup(document, source=path)

    def subject(self, markup_id: str | None, project: str, subject_dir: Path) -> SubjectMarkup:
        """Resolve one participant's paths, or return an unrestricted selection."""
        root = Path(subject_dir).expanduser().absolute()
        if markup_id is None:
            return SubjectMarkup(None, project, root)
        identifier = validate_config_id(markup_id, kind="markup")
        participant = root.name.removeprefix("sub-")
        record = self.load(identifier).get(project, {}).get(participant, {})
        return SubjectMarkup(
            identifier,
            project,
            root,
            tuple(root / path for path in record.get("T1w", ())),
            tuple(root / path for path in record.get("T2w", ())),
            tuple(root / path for path in record.get("exclude", ())),
        )


def compile_markup(
    document: Mapping, *, source: str | Path = "markup"
) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    """Validate and normalize one parsed markup document."""
    path = str(source)
    if not isinstance(document, Mapping):
        raise DefinitionError(f"{path}: markup must contain a mapping")
    result = {}
    for raw_project, raw_subjects in document.items():
        if not isinstance(raw_project, str) or not raw_project.strip():
            raise DefinitionError(f"{path}: markup projects must be nonempty strings")
        project = raw_project.strip()
        if project in result:
            raise DefinitionError(f"{path}: duplicate markup project {project}")
        raw_subjects = {} if raw_subjects is None else raw_subjects
        if not isinstance(raw_subjects, Mapping):
            raise DefinitionError(f"{path}: project {project} must contain a subject mapping")
        subjects = {}
        for raw_subject, raw_record in raw_subjects.items():
            if not isinstance(raw_subject, str) or not raw_subject.strip():
                raise DefinitionError(f"{path}: markup subjects must be nonempty strings")
            subject = raw_subject.strip().removeprefix("sub-")
            if not subject:
                raise DefinitionError(f"{path}: invalid markup subject {raw_subject!r}")
            if subject in subjects:
                raise DefinitionError(f"{path}: duplicate markup subject {project}/sub-{subject}")
            raw_record = {} if raw_record is None else raw_record
            if not isinstance(raw_record, Mapping) or any(
                not isinstance(key, str) for key in raw_record
            ):
                raise DefinitionError(f"{path}: {project}/sub-{subject} must contain a mapping")
            unknown = set(raw_record) - _FIELDS
            if unknown:
                raise DefinitionError(
                    f"{path}: {project}/sub-{subject} has unknown field(s): "
                    + ", ".join(sorted(unknown))
                )
            record = {
                key: _paths(
                    raw_record.get(key),
                    location=f"{project}.sub-{subject}.{key}",
                    list_only=key == "exclude",
                )
                for key in ("T1w", "T2w", "exclude")
            }
            overlap = [
                anatomical
                for anatomical in (*record["T1w"], *record["T2w"])
                if any(_contains(excluded, anatomical) for excluded in record["exclude"])
            ]
            if overlap:
                raise DefinitionError(
                    f"{path}: {project}/sub-{subject} selects and excludes " + ", ".join(overlap)
                )
            subjects[subject] = record
        result[project] = subjects
    return result


def activate_source_markup(markup: SubjectMarkup) -> None:
    """Make a captured markup selection available to module-internal readers."""
    os.environ[SOURCE_MARKUP_ENV] = json.dumps(markup.as_dict(), sort_keys=True)


def active_source_markup(
    subject_dir: Path | None = None, *, project: str | None = None
) -> SubjectMarkup | None:
    """Return the worker's captured markup, optionally checking its subject root."""
    raw = os.environ.get(SOURCE_MARKUP_ENV)
    if not raw:
        return None
    try:
        markup = SubjectMarkup.from_dict(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {SOURCE_MARKUP_ENV}: {error}") from error
    if subject_dir is not None and markup.subject_dir != Path(subject_dir).expanduser().absolute():
        raise ValueError("Captured source markup belongs to a different BIDS participant")
    if project is not None and markup.project != project:
        raise ValueError("Captured source markup belongs to a different BIDS project")
    return markup


def load_source_markup(markup_id: str | None, project: str, subject_dir: Path) -> SubjectMarkup:
    """Use the worker snapshot when present, otherwise read the selected definition."""
    active = active_source_markup(subject_dir, project=project)
    if active is not None:
        if active.markup_id != markup_id:
            raise ValueError("Captured source markup differs from the runtime configuration")
        return active
    markup = MarkupStore().subject(markup_id, project, subject_dir)
    activate_source_markup(markup)
    return markup
