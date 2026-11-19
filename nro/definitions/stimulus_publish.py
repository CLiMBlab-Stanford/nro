"""Publish staged stimulus annotations into the definitions catalog.

A stimulus suite, such as a story collection with its own featurization code,
writes a staged directory in the flat layout of ``DEFINITIONS/stimuli``:
``stimuli.tsv`` and ``stim-<label>_annot-<level>_events.tsv`` files with JSON
sidecars. The sidecars carry no ``NROVersion``; publication assigns it by
comparing each staged annotation TSV with the published one. A new annotation
gets version 1, changed timing or values get the published version plus 1, and
an unchanged TSV keeps its version.

Staged ``stimuli.tsv`` rows are added to the published table; a staged row may
not differ from a published row with the same ID. Annotation levels are
defined by nro (``ANNOTATION_LEVELS``), not by suites: a staged annotation
must use one of them, and publication adds nro's description of each level to
the catalog's ``annotations.tsv`` the first time the level is used. The whole
batch is published in one definitions-store transaction, with the validation
of ``nro def apply``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nro.definitions.stimuli import (
    ANNOTATION_FILE,
    ANNOTATION_LEVELS,
    SOURCE_FIELD,
    STIMULI_COLUMNS,
    STIMULUS_ID,
    VERSION_FIELD,
    annotation_version,
    read_table,
)
from nro.engine.events import validate_events

CATALOG = Path("stimuli")
NA = "n/a"


@dataclass(frozen=True)
class AnnotationChange:
    """How publication changes one annotation file.

    ``status`` is ``new``, ``changed`` (the TSV differs), ``metadata`` (only
    the sidecar differs), or ``unchanged``.
    """

    name: str
    status: str
    version: int
    previous: int | None


@dataclass(frozen=True)
class PublishPlan:
    """The files a publication writes and what it changes."""

    root: Path
    updates: dict[Path, bytes]
    annotations: tuple[AnnotationChange, ...]
    added_rows: dict[str, tuple[str, ...]]

    @property
    def changed(self) -> bool:
        return bool(self.updates)


def _text(table, columns: list[str]) -> bytes:
    """Write a table of text values as a TSV, refusing tabs and line breaks in values."""
    lines = ["\t".join(columns)]
    for row in table:
        values = [row.get(column, NA) for column in columns]
        for value in values:
            if any(character in value for character in "\t\r\n"):
                raise ValueError(f"Table values may not contain tabs or line breaks: {value!r}")
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _merge_rows(
    name: str, key: str, rows: list[dict], published: Path, errors: list[str]
) -> tuple[bytes | None, tuple[str, ...]]:
    """Add rows to a published catalog table, keyed by ``key``.

    Return the merged table's bytes, or None when the published table already
    holds every row, and the IDs of the rows added. A row that differs from a
    published row with the same ID is an error.
    """
    old = read_table(published) if published.is_file() else None
    if old is None and not rows:
        return None, ()
    columns = list(old.columns) if old is not None else []
    for row in rows:
        columns += [column for column in row if column not in columns]
    merged = old.to_dict("records") if old is not None else []
    existing = {row[key]: row for row in merged}
    added: list[str] = []
    for row in rows:
        identifier = row[key]
        if identifier in added:
            errors.append(f"Staged {name}: duplicate {key} {identifier}")
        elif identifier in existing:
            different = sorted(c for c in row if existing[identifier].get(c, NA) != row[c])
            if different:
                errors.append(
                    f"Staged {name}: {identifier} differs from the published row in "
                    f"{different}; change published rows with `nro def apply`"
                )
        else:
            merged.append(row)
            added.append(identifier)
    if not added and old is not None and list(old.columns) == columns:
        return None, ()
    return _text(merged, columns), tuple(added)


def _staged_files(staged: Path, errors: list[str]) -> dict[str, dict[str, Path]]:
    """Group a staged directory's annotation files by base name, and check its entries."""
    annotations: dict[str, dict[str, Path]] = {}
    for path in sorted(staged.iterdir()):
        if path.name.startswith("."):
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"Staged entries must be regular files: {path}")
        elif path.name == "stimuli.tsv":
            continue
        elif path.name == "annotations.tsv":
            errors.append(
                f"{path}: remove it; nro defines annotation levels and writes the catalog's "
                "annotations.tsv"
            )
        elif (match := ANNOTATION_FILE.fullmatch(path.name)) is None:
            errors.append(
                f"Unsupported staged file: {path}. Publication takes stimuli.tsv and "
                "stim-<label>_annot-<level>_events.tsv files with JSON sidecars"
            )
        else:
            stimulus, level, extension = match.groups()
            if level not in ANNOTATION_LEVELS:
                errors.append(
                    f"{path}: unknown annotation level {level!r}; nro defines "
                    f"{', '.join(sorted(ANNOTATION_LEVELS))}"
                )
            annotations.setdefault(f"{stimulus}_annot-{level}_events", {})[extension] = path
    if not (staged / "stimuli.tsv").is_file():
        errors.append(f"Missing staged stimuli table: {staged / 'stimuli.tsv'}")
    return annotations


def _stimulus_rows(staged: Path, errors: list[str]) -> list[dict]:
    table = read_table(staged / "stimuli.tsv")
    missing = [column for column in STIMULI_COLUMNS if column not in table]
    if missing:
        errors.append(f"{staged / 'stimuli.tsv'}: missing required columns {missing}")
        return []
    rows = table.to_dict("records")
    errors.extend(
        f"{staged / 'stimuli.tsv'}: stimulus_id must have the form stim-<label>: "
        f"{row['stimulus_id']!r}"
        for row in rows
        if not STIMULUS_ID.fullmatch(row["stimulus_id"])
    )
    return rows


def _sidecar(path: Path, errors: list[str]) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append(f"Staged sidecar is not valid JSON: {path}")
        return None
    if not isinstance(value, dict):
        errors.append(f"Staged sidecar must be a JSON object: {path}")
        return None
    for field, reason in (
        (VERSION_FIELD, "publication assigns it"),
        (SOURCE_FIELD, "it belongs only in project copies"),
    ):
        if field in value:
            errors.append(f"{path}: remove {field}; {reason}")
    return value


def plan_publish(staged: Path, root: Path) -> PublishPlan:
    """Plan the publication of a staged directory into the store at ``root``.

    Raises ValueError listing every problem in the staged files.
    """
    staged, root = Path(staged), Path(root)
    if not staged.is_dir():
        raise ValueError(f"Staged stimulus directory does not exist: {staged}")
    errors: list[str] = []
    files = _staged_files(staged, errors)
    stimuli = _stimulus_rows(staged, errors) if (staged / "stimuli.tsv").is_file() else []
    identifiers = {row["stimulus_id"] for row in stimuli}
    published = root / CATALOG
    updates: dict[Path, bytes] = {}
    changes = []
    for base, paths in sorted(files.items()):
        table, sidecar = paths.get("tsv"), paths.get("json")
        if table is None or sidecar is None:
            errors.append(f"{staged}: {base} needs both an .tsv file and its JSON sidecar")
            continue
        if base.split("_annot-")[0] not in identifiers:
            errors.append(f"{staged}: {base} has no staged stimuli.tsv row")
        try:
            validate_events(table)
        except (OSError, ValueError) as error:
            errors.append(f"{table}: {error}")
            continue
        metadata = _sidecar(sidecar, errors)
        if metadata is None:
            continue
        content = table.read_bytes()
        old_table, old_sidecar = published / table.name, published / sidecar.name
        previous = None
        if old_table.is_file() and old_sidecar.is_file():
            try:
                previous = annotation_version(old_sidecar)
            except ValueError as error:
                errors.append(str(error))
                continue
        if previous is None:
            status, version = "new", 1
        elif old_table.read_bytes() != content:
            status, version = "changed", previous + 1
        else:
            status, version = "unchanged", previous
        text = json.dumps({**metadata, VERSION_FIELD: version}, indent=2, ensure_ascii=False)
        sidecar_bytes = (text + "\n").encode("utf-8")
        if status != "unchanged":
            updates[CATALOG / table.name] = content
        if not old_sidecar.is_file() or old_sidecar.read_bytes() != sidecar_bytes:
            updates[CATALOG / sidecar.name] = sidecar_bytes
            if status == "unchanged":
                status = "metadata"
        changes.append(AnnotationChange(base, status, version, previous))
    levels = sorted({base.split("_annot-")[1].removesuffix("_events") for base in files})
    tables = {
        "stimuli.tsv": ("stimulus_id", stimuli),
        "annotations.tsv": (
            "annot_id",
            [
                {"annot_id": level, "description": ANNOTATION_LEVELS[level]}
                for level in levels
                if level in ANNOTATION_LEVELS
            ],
        ),
    }
    added = {}
    for name, (key, rows) in tables.items():
        if name == "annotations.tsv":
            rows = _new_levels(rows, published / name)
        merged, added[name] = _merge_rows(name, key, rows, published / name, errors)
        if merged is not None:
            updates[CATALOG / name] = merged
    if errors:
        raise ValueError("The staged stimulus directory is invalid:\n" + "\n".join(errors))
    return PublishPlan(root, updates, tuple(changes), added)


def _new_levels(rows: list[dict], published: Path) -> list[dict]:
    """Keep the level rows that the catalog does not list yet.

    A level the catalog already lists keeps its published row, so a lab can
    refine the description with ``nro def apply``.
    """
    if not published.is_file():
        return rows
    listed = set(read_table(published).get("annot_id", []))
    return [row for row in rows if row["annot_id"] not in listed]


def publish_stimuli(
    staged: Path,
    *,
    dry_run: bool = False,
    root: Path | None = None,
    validate: Callable[[Path], None] | None = None,
) -> PublishPlan:
    """Publish a staged stimulus directory into the selected definitions store.

    ``root`` defaults to the selected store and ``validate`` to the checks of
    ``nro def apply``. With ``dry_run``, plan and return the publication
    without writing. Raises ValueError for invalid staged files, a store with
    direct changes, or a candidate store that fails validation.
    """
    from nro.definitions.migrations import update_store
    from nro.definitions.repository import store_validator
    from nro.site import configuration as site

    root = Path(root) if root is not None else site.definitions_root()
    with site.definition_write(root / CATALOG / "stimuli.tsv"):
        plan = plan_publish(staged, root)
        if dry_run or not plan.changed:
            return plan
        for relative in plan.updates:
            site.require_definition_write(root / relative)
        update_store(
            root, plan.updates, validate=validate if validate is not None else store_validator(root)
        )
    return plan


def describe(plan: PublishPlan, *, dry_run: bool) -> list[str]:
    """Return report lines for a publication plan."""
    lines = [
        f"{change.status:9}  {change.name}  version {change.version}"
        + (f" (was {change.previous})" if change.status == "changed" else "")
        for change in plan.annotations
    ]
    for name, identifiers in plan.added_rows.items():
        if identifiers:
            shown = ", ".join(identifiers[:5]) + (", ..." if len(identifiers) > 5 else "")
            lines.append(f"{name}: add {len(identifiers)} rows ({shown})")
    counts = {}
    for change in plan.annotations:
        counts[change.status] = counts.get(change.status, 0) + 1
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    if not plan.changed:
        lines.append(f"Nothing to publish; the catalog in {plan.root} is current ({summary}).")
    elif dry_run:
        lines.append(f"Dry run: would write {len(plan.updates)} files to {plan.root} ({summary}).")
    else:
        lines.append(f"Published {len(plan.updates)} files to {plan.root} ({summary}).")
    return lines
