"""Copy catalog stimulus annotations into the BIDS projects whose events use them.

A project needs a stimulus when any of its run events files names it in
``stim_id``. Sync copies every annotation of each needed stimulus, adds the
stimulus's ``stimuli.tsv`` and ``annotations.tsv`` rows, and records the
copied version in each annotation sidecar's ``NROStimulusSource``.
"""

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nro.definitions.stimuli import (
    SOURCE_FIELD,
    STIMULUS_ID,
    CatalogAnnotation,
    StimulusCatalog,
    read_table,
)
from nro.engine.io import atomic_write_text
from nro.orchestration.project_scope import source_bids_walk

CURRENT = "current"
MISSING = "missing"
STALE = "stale"
LOCAL_EDIT = "local edit"
UNMANAGED = "not written by nro"
AHEAD = "ahead of catalog"
BLOCKING = {LOCAL_EDIT, UNMANAGED, AHEAD}


@dataclass(frozen=True)
class AnnotationStatus:
    """How one project annotation file compares with its catalog version."""

    annotation: CatalogAnnotation
    status: str
    project_version: int | None = None

    @property
    def name(self) -> str:
        """Return the annotation's TSV filename."""
        return self.annotation.table.name


@dataclass
class SyncPlan:
    """The comparison of one project with the catalog, before any file changes."""

    project_root: Path
    references: dict[str, list[Path]]
    unknown: list[str] = field(default_factory=list)
    annotations: list[AnnotationStatus] = field(default_factory=list)
    stimulus_rows: list[dict] = field(default_factory=list)
    annotation_rows: list[dict] = field(default_factory=list)

    @property
    def writes(self) -> bool:
        """Return whether applying the plan without ``--replace`` changes files."""
        return bool(
            self.stimulus_rows
            or self.annotation_rows
            or any(item.status == MISSING for item in self.annotations)
        )

    @property
    def stale(self) -> list[AnnotationStatus]:
        """Return copies that an older catalog version wrote."""
        return [item for item in self.annotations if item.status == STALE]

    @property
    def blocked(self) -> list[AnnotationStatus]:
        """Return project files that sync must not overwrite."""
        return [item for item in self.annotations if item.status in BLOCKING]


def referenced_stimuli(project_root: Path) -> dict[str, list[Path]]:
    """Map each ``stim_id`` in a project's run events files to the files naming it.

    Annotation files in ``stimuli/`` and derivative trees are not run events.
    """
    project_root = Path(project_root)
    references: dict[str, list[Path]] = {}
    for parent, _directories, names in source_bids_walk(project_root):
        relative = Path(parent).relative_to(project_root)
        if relative.parts and relative.parts[0] in {"stimuli", "sourcedata", "phenotype"}:
            continue
        for name in sorted(names):
            if not name.endswith("_events.tsv"):
                continue
            path = Path(parent) / name
            with path.open(encoding="utf-8", newline="") as stream:
                header = next(csv.reader(stream, delimiter="\t"), [])
            if "stim_id" not in header:
                continue
            values = read_table(path)["stim_id"]
            for value in dict.fromkeys(v for v in values if v not in {"", "n/a"}):
                references.setdefault(value, []).append(path)
    return references


def _project_status(annotation: CatalogAnnotation, directory: Path) -> AnnotationStatus:
    table = directory / annotation.table.name
    sidecar = directory / annotation.sidecar.name
    if not table.exists() and not sidecar.exists():
        return AnnotationStatus(annotation, MISSING)
    try:
        source = json.loads(sidecar.read_text(encoding="utf-8")).get(SOURCE_FIELD)
    except (OSError, json.JSONDecodeError, AttributeError):
        source = None
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("version"), int)
        or not isinstance(source.get("sha256"), str)
        or not table.is_file()
    ):
        return AnnotationStatus(annotation, UNMANAGED)
    version = source["version"]
    if hashlib.sha256(table.read_bytes()).hexdigest() != source["sha256"]:
        return AnnotationStatus(annotation, LOCAL_EDIT, version)
    if version > annotation.version:
        return AnnotationStatus(annotation, AHEAD, version)
    if version < annotation.version:
        return AnnotationStatus(annotation, STALE, version)
    return AnnotationStatus(annotation, CURRENT, version)


def plan_sync(project_root: Path, catalog: StimulusCatalog) -> SyncPlan:
    """Compare a project's stimuli with the catalog without changing files."""
    project_root = Path(project_root)
    directory = project_root / "stimuli"
    plan = SyncPlan(project_root, referenced_stimuli(project_root))
    existing_stimuli = row_keys(directory / "stimuli.tsv", "stimulus_id")
    existing_annotations = row_keys(directory / "annotations.tsv", "annot_id")
    added_labels = set()
    for identifier in sorted(plan.references):
        stimulus = catalog.get(identifier) if STIMULUS_ID.fullmatch(identifier) else None
        if stimulus is None:
            plan.unknown.append(identifier)
            continue
        if identifier not in existing_stimuli:
            plan.stimulus_rows.append(dict(stimulus.row))
        for label, annotation in sorted(stimulus.annotations.items()):
            plan.annotations.append(_project_status(annotation, directory))
            description = stimulus.descriptions.get(label)
            if (
                description is not None
                and label not in existing_annotations
                and label not in added_labels
            ):
                plan.annotation_rows.append(dict(description))
                added_labels.add(label)
    return plan


def row_keys(path: Path, key: str) -> set[str]:
    """Return the values of ``key`` in a project table, or an empty set."""
    if not path.is_file():
        return set()
    table = read_table(path)
    if key not in table:
        raise ValueError(f"{path} has no {key} column")
    return set(table[key])


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to a project table, creating it when missing; ``n/a`` fills gaps."""
    if not rows:
        return
    added = pd.DataFrame(rows, dtype=str)
    table = pd.concat([read_table(path), added], ignore_index=True) if path.is_file() else added
    atomic_write_text(path, table.fillna("n/a").to_csv(sep="\t", index=False), mode=0o664)


def _copy(annotation: CatalogAnnotation, directory: Path) -> None:
    data = annotation.table.read_bytes()
    sidecar = json.loads(annotation.sidecar.read_text(encoding="utf-8"))
    sidecar[SOURCE_FIELD] = {
        "path": f"stimuli/{annotation.table.name}",
        "version": annotation.version,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    # The TSV is written first. If the sidecar write fails, the old sidecar's
    # hash no longer matches and the next sync reports a local edit instead of
    # silently treating the file as current.
    atomic_write_text(directory / annotation.table.name, data.decode("utf-8"), mode=0o664)
    atomic_write_text(
        directory / annotation.sidecar.name, json.dumps(sidecar, indent=2) + "\n", mode=0o664
    )


def apply_sync(plan: SyncPlan, *, replace: bool = False) -> list[AnnotationStatus]:
    """Copy missing annotations, and stale ones when ``replace`` is true.

    Files that are local edits, unmanaged, or ahead of the catalog are never
    written. Return the annotation statuses that were copied.
    """
    directory = plan.project_root / "stimuli"
    copied = [
        item
        for item in plan.annotations
        if item.status == MISSING or (replace and item.status == STALE)
    ]
    if not copied and not plan.stimulus_rows and not plan.annotation_rows:
        return []
    directory.mkdir(mode=0o2775, exist_ok=True)
    append_rows(directory / "stimuli.tsv", plan.stimulus_rows)
    append_rows(directory / "annotations.tsv", plan.annotation_rows)
    for item in copied:
        _copy(item.annotation, directory)
    return copied


def active_attempts(db, project: str) -> list[str]:
    """Return ``MODULE:sub-PARTICIPANT`` for the project's queued or running attempts."""
    rows = db.execute(
        """SELECT DISTINCT item.module, item.participant
           FROM attempts attempt JOIN work_items item ON item.id=attempt.work_item_id
           WHERE item.project=? AND attempt.state IN ('queued','running','cancel_requested')
           ORDER BY item.module, item.participant""",
        (project,),
    ).fetchall()
    return [f"{row[0]}:sub-{row[1]}" for row in rows]


def describe(plan: SyncPlan, copied: list[AnnotationStatus], *, replace: bool) -> list[str]:
    """Return report lines for a plan and the files it copied."""
    lines = []
    copied_names = {item.name for item in copied}
    for identifier in plan.unknown:
        files = ", ".join(
            str(path.relative_to(plan.project_root)) for path in plan.references[identifier]
        )
        lines.append(f"unknown   {identifier}: not in the stimulus catalog (named in {files})")
    for item in plan.annotations:
        version = f"v{item.annotation.version}"
        if item.name in copied_names:
            lines.append(f"copied    {item.name} {version}")
        elif item.status == STALE:
            lines.append(
                f"stale     {item.name} v{item.project_version} -> {version}"
                + ("" if replace else "; rerun with --replace to update")
            )
        elif item.status == MISSING:
            lines.append(f"missing   {item.name} {version}")
        elif item.status == CURRENT:
            lines.append(f"current   {item.name} {version}")
        else:
            lines.append(f"blocked   {item.name}: {item.status}; sync will not overwrite it")
    for row in plan.stimulus_rows:
        lines.append(f"stimuli.tsv: add {row['stimulus_id']}")
    for row in plan.annotation_rows:
        lines.append(f"annotations.tsv: add {row['annot_id']}")
    return lines
