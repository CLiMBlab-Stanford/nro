"""Read and validate the stimulus annotation catalog in a definitions store.

``DEFINITIONS/stimuli`` uses the flat layout of a BIDS ``stimuli/`` directory
from the BEP044 draft, so files copy into projects without renaming. The
catalog holds stimulus metadata and annotations, not media files.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nro.engine.events import validate_events

STIMULUS_ID = re.compile(r"stim-[A-Za-z0-9]+")
ANNOTATION_FILE = re.compile(r"(stim-[A-Za-z0-9]+)_annot-([A-Za-z0-9]+)_events\.(tsv|json)")
STIMULUS_TYPES = {"audio", "image", "video", "audiovideo"}
STIMULI_COLUMNS = ("stimulus_id", "type", "license", "present")
TABLES = {"stimuli.tsv", "stimuli.json", "annotations.tsv", "annotations.json"}
SOURCE_FIELD = "NROStimulusSource"
VERSION_FIELD = "NROVersion"


@dataclass(frozen=True)
class CatalogAnnotation:
    """One annotation file of a catalog stimulus and its sidecar metadata."""

    stimulus: str
    label: str
    table: Path
    sidecar: Path
    version: int

    @property
    def sha256(self) -> str:
        """Return the SHA-256 of the annotation TSV."""
        return hashlib.sha256(self.table.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CatalogStimulus:
    """A catalog stimulus: its ``stimuli.tsv`` row and its annotations by label."""

    identifier: str
    root: Path
    row: dict
    annotations: dict[str, CatalogAnnotation]
    descriptions: dict[str, dict]


def read_table(path: Path) -> pd.DataFrame:
    """Read a TSV as text, so values such as ``false`` and ``n/a`` stay unchanged."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def annotation_version(sidecar: Path) -> int:
    """Read the positive integer ``NROVersion`` from an annotation sidecar."""
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Annotation sidecar is not valid JSON: {sidecar}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Annotation sidecar must be a JSON object: {sidecar}")
    version = value.get(VERSION_FIELD)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"{VERSION_FIELD} must be a positive integer: {sidecar}")
    return version


def _catalog_files(directory: Path) -> tuple[dict, list[str]]:
    """Group a catalog directory's annotation files by stimulus, and collect errors."""
    errors = []
    annotations: dict[str, dict[str, dict[str, Path]]] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"Stimulus catalog entries must be regular files: {path}")
            continue
        if path.name in TABLES:
            continue
        match = ANNOTATION_FILE.fullmatch(path.name)
        if match is None:
            errors.append(
                f"Unsupported stimulus catalog file: {path}. The catalog holds stimuli.tsv, "
                "annotations.tsv, and stim-<label>_annot-<label>_events.tsv files with JSON "
                "sidecars; media files and part-<label> annotations are not supported yet"
            )
            continue
        stimulus, label, extension = match.groups()
        annotations.setdefault(stimulus, {}).setdefault(label, {})[extension] = path
    return annotations, errors


def _stimulus_rows(path: Path, errors: list[str]) -> dict[str, dict]:
    table = read_table(path)
    missing = [column for column in STIMULI_COLUMNS if column not in table]
    if missing:
        errors.append(f"{path}: missing required columns {missing}")
        return {}
    rows = {}
    for row in table.to_dict("records"):
        identifier = row["stimulus_id"]
        if not STIMULUS_ID.fullmatch(identifier):
            errors.append(f"{path}: stimulus_id must have the form stim-<label>: {identifier!r}")
        elif identifier in rows:
            errors.append(f"{path}: duplicate stimulus_id {identifier}")
        elif row["type"] not in STIMULUS_TYPES:
            errors.append(f"{path}: {identifier} type must be one of {sorted(STIMULUS_TYPES)}")
        elif row["present"] != "false":
            errors.append(
                f"{path}: {identifier} present must be false; the catalog holds no media files"
            )
        else:
            rows[identifier] = row
    return rows


def _annotation_rows(path: Path, errors: list[str]) -> dict[str, dict]:
    table = read_table(path)
    if "annot_id" not in table:
        errors.append(f"{path}: missing required column annot_id")
        return {}
    rows = {}
    for row in table.to_dict("records"):
        if row["annot_id"] in rows:
            errors.append(f"{path}: duplicate annot_id {row['annot_id']}")
        rows[row["annot_id"]] = row
    return rows


def load_catalog(root: Path) -> tuple[dict[str, CatalogStimulus], list[str]]:
    """Load one store's catalog, returning stimuli by ID and every problem found.

    A store without a ``stimuli`` directory has an empty catalog.
    """
    directory = Path(root) / "stimuli"
    if not directory.is_dir():
        return {}, []
    files, errors = _catalog_files(directory)
    if not files and not (directory / "stimuli.tsv").is_file():
        return {}, errors
    if not (directory / "stimuli.tsv").is_file():
        return {}, [*errors, f"Missing stimulus catalog table: {directory / 'stimuli.tsv'}"]
    rows = _stimulus_rows(directory / "stimuli.tsv", errors)
    descriptions = (
        _annotation_rows(directory / "annotations.tsv", errors)
        if (directory / "annotations.tsv").is_file()
        else None
    )
    result = {}
    for identifier in sorted(set(rows) | set(files)):
        if identifier not in rows:
            errors.append(f"{directory}: {identifier} has annotation files but no stimuli.tsv row")
            continue
        annotations = {}
        for label, paths in sorted(files.get(identifier, {}).items()):
            table, sidecar = paths.get("tsv"), paths.get("json")
            if table is None or sidecar is None:
                errors.append(
                    f"{directory}: {identifier}_annot-{label} needs both an _events.tsv file "
                    "and its JSON sidecar"
                )
                continue
            try:
                validate_events(table)
                version = annotation_version(sidecar)
                if SOURCE_FIELD in json.loads(sidecar.read_text(encoding="utf-8")):
                    raise ValueError(f"{SOURCE_FIELD} belongs only in project copies: {sidecar}")
            except (OSError, ValueError) as error:
                errors.append(f"{table}: {error}")
                continue
            if descriptions is not None and label not in descriptions:
                errors.append(f"{directory / 'annotations.tsv'}: annot_id {label} is not listed")
            annotations[label] = CatalogAnnotation(identifier, label, table, sidecar, version)
        result[identifier] = CatalogStimulus(
            identifier,
            Path(root),
            rows[identifier],
            annotations,
            {
                label: descriptions[label]
                for label in annotations
                if descriptions is not None and label in descriptions
            },
        )
    return result, errors


def validate_catalog(
    root: Path,
    *,
    inherited_roots: tuple[Path, ...] = (),
    published: Path | None = None,
) -> tuple[int, int, list[str]]:
    """Validate a store's catalog against its inherited stores and published copy.

    A stimulus ID defined in both this store and an inherited store is an
    error. When ``published`` names the store as it is currently published,
    each changed annotation TSV must carry a higher ``NROVersion``, an
    unchanged TSV must keep its version, and a new file must start at 1.
    Earlier versions are kept in the store's Git history, not in the catalog.
    A file that was changed directly in the published store is compared with
    its version at the store's Git ``HEAD``.

    Return the number of stimuli, the number of annotation files, and every
    problem found.
    """
    stimuli, errors = load_catalog(root)
    for inherited in inherited_roots:
        others, _ = load_catalog(inherited)
        for identifier in sorted(set(stimuli) & set(others)):
            errors.append(
                f"Stimulus {identifier} is defined in both {Path(root) / 'stimuli'} and "
                f"{Path(inherited) / 'stimuli'}; a stimulus ID must be unique across the "
                "definitions chain"
            )
    if published is not None:
        baseline = _PublishedFiles(Path(published))
        for stimulus in stimuli.values():
            for annotation in stimulus.annotations.values():
                try:
                    errors.extend(_version_errors(annotation, baseline.previous(annotation)))
                except ValueError as error:
                    errors.append(str(error))
    count = sum(len(stimulus.annotations) for stimulus in stimuli.values())
    return len(stimuli), count, errors


class _PublishedFiles:
    """Read the previous version of catalog files from a published store."""

    def __init__(self, root: Path):
        from nro.definitions.migrations import _read_manifest, changed_paths

        self.root = root
        manifest = _read_manifest(root)
        self.recorded = set(manifest["files"]) if manifest is not None else set()
        self.drift = {str(path) for path in changed_paths(root)} if manifest is not None else set()

    def _bytes(self, relative: str) -> bytes | None:
        path = self.root / relative
        if relative not in self.drift:
            return path.read_bytes() if path.is_file() else None
        committed = _git_head_file(self.root, relative)
        if committed is None and relative in self.recorded:
            raise ValueError(
                f"{relative}: cannot check {VERSION_FIELD} because the published file was "
                "changed directly and is not in the store's Git history; restore it and "
                "publish the new version with `nro def apply --file`"
            )
        return committed

    def previous(self, annotation: CatalogAnnotation) -> tuple[bytes, int] | None:
        """Return the previous TSV bytes and version, or None for a new annotation."""
        table = self._bytes(f"stimuli/{annotation.table.name}")
        sidecar = self._bytes(f"stimuli/{annotation.sidecar.name}")
        if table is None or sidecar is None:
            return None
        try:
            value = json.loads(sidecar)
            version = value[VERSION_FIELD]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"stimuli/{annotation.sidecar.name}: the previous sidecar has no readable "
                f"{VERSION_FIELD}"
            ) from error
        return table, int(version)


def _git_head_file(root: Path, relative: str) -> bytes | None:
    """Return a file's bytes at the store's Git HEAD, or None when unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _version_errors(annotation: CatalogAnnotation, previous: tuple[bytes, int] | None) -> list[str]:
    name = annotation.table.name
    if previous is None:
        if annotation.version != 1:
            return [f"{name}: a new annotation must start at {VERSION_FIELD} 1"]
        return []
    before, version = previous
    changed = annotation.table.read_bytes() != before
    if changed and annotation.version <= version:
        return [f"{name}: the annotation changed, so {VERSION_FIELD} must be higher than {version}"]
    if not changed and annotation.version != version:
        return [f"{name}: the annotation is unchanged, so {VERSION_FIELD} must stay {version}"]
    return []


class StimulusCatalog:
    """Look up catalog stimuli across a definitions chain, nearest store first."""

    def __init__(self, roots: tuple[Path, ...] | None = None):
        """Use the active definitions chain unless explicit store roots are given."""
        if roots is None:
            from nro.site.configuration import definitions_roots

            roots = definitions_roots()
        self.roots = tuple(Path(root) for root in roots)
        self._stimuli: dict[str, CatalogStimulus] = {}
        for root in reversed(self.roots):
            stimuli, errors = load_catalog(root)
            if errors:
                raise ValueError("Stimulus catalog is invalid:\n" + "\n".join(errors))
            self._stimuli.update(stimuli)

    def get(self, identifier: str) -> CatalogStimulus | None:
        """Return a stimulus by ``stim-<label>``, or None when no store defines it."""
        return self._stimuli.get(identifier)
