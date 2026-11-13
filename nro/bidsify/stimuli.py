"""Snapshot catalog stimulus annotations at review and copy them at publication.

When reviewed events name stimuli in ``stim_id``, review snapshots every
annotation of those stimuli from the definitions catalog. Publication copies
the snapshots into the project's ``stimuli/`` directory. As with events, a
later catalog edit does not change an accepted snapshot; ``nro stimuli sync``
updates projects explicitly.
"""

import hashlib
import json
from io import StringIO
from pathlib import Path

import pandas as pd

from nro.definitions.stimuli import SOURCE_FIELD, StimulusCatalog
from nro.definitions.stimulus_sync import append_rows, row_keys
from nro.engine.io import atomic_write_text

from .errors import BidsificationError


def stimulus_roots(config: dict) -> tuple[Path, ...]:
    """Return the definitions stores whose event catalogs the request uses."""
    stores = config["event_store"]
    stores = [stores] if isinstance(stores, str) else stores
    return tuple(Path(store).parent for store in stores)


def referenced_ids(text: str) -> tuple[str, ...]:
    """Return the distinct ``stim_id`` values named in reviewed events text."""
    table = pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    if "stim_id" not in table:
        return ()
    return tuple(dict.fromkeys(v for v in table["stim_id"] if v not in {"", "n/a"}))


def snapshot_stimuli(text: str, config: dict) -> tuple[dict, dict[str, str]]:
    """Resolve the stimuli that events name and return their snapshot.

    Return the snapshot description stored with the acquisition and the file
    texts to store. Raise BidsificationError when a stimulus is not in the
    catalog, so the reviewer can add it or choose other events.
    """
    identifiers = referenced_ids(text)
    if not identifiers:
        return {}, {}
    try:
        catalog = StimulusCatalog(stimulus_roots(config))
    except ValueError as error:
        raise BidsificationError(str(error)) from error
    unknown = [identifier for identifier in identifiers if catalog.get(identifier) is None]
    if unknown:
        raise BidsificationError(
            f"The events name stimuli that are not in the stimulus catalog: {unknown}. "
            "Add them to DEFINITIONS/stimuli with `nro def apply`, or choose other events."
        )
    files, annotations, rows, descriptions = {}, {}, [], []
    for identifier in identifiers:
        stimulus = catalog.get(identifier)
        rows.append(dict(stimulus.row))
        for label, annotation in sorted(stimulus.annotations.items()):
            table = annotation.table.read_text(encoding="utf-8")
            files[annotation.table.name] = table
            files[annotation.sidecar.name] = annotation.sidecar.read_text(encoding="utf-8")
            annotations[annotation.table.name] = {
                "sidecar": annotation.sidecar.name,
                "version": annotation.version,
                "sha256": hashlib.sha256(table.encode("utf-8")).hexdigest(),
            }
            if label in stimulus.descriptions:
                descriptions.append(dict(stimulus.descriptions[label]))
    return {
        "annotations": annotations,
        "stimulus_rows": rows,
        "annotation_rows": descriptions,
    }, files


def publish_stimuli(record: dict, project_root: Path) -> list[str]:
    """Copy a request's stimulus snapshots into the project's ``stimuli/`` directory.

    Missing annotations are copied with ``NROStimulusSource``. An existing copy
    of the same version and hash is left as is. A different existing copy is
    never replaced; it is reported so the operator can run
    ``nro stimuli sync PROJECT``. Missing ``stimuli.tsv`` and
    ``annotations.tsv`` rows are added. Repeating publication changes nothing.
    Return report lines.
    """
    directory = Path(project_root) / "stimuli"
    messages = []
    rows, descriptions, copies = {}, {}, {}
    for item in record["acquisitions"]:
        snapshot = item.get("stimuli")
        if not snapshot:
            continue
        for row in snapshot["stimulus_rows"]:
            rows.setdefault(row["stimulus_id"], row)
        for row in snapshot["annotation_rows"]:
            descriptions.setdefault(row["annot_id"], row)
        for name, source in snapshot["annotations"].items():
            copies.setdefault(name, (source, snapshot["files"]))
    if not copies and not rows:
        return []
    verified = {}
    for name, (source, files) in copies.items():
        data = Path(files[name]).read_bytes()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise BidsificationError(f"Stimulus snapshot changed after review: {files[name]}")
        verified[name] = data
    directory.mkdir(mode=0o2775, exist_ok=True)
    existing_rows = row_keys(directory / "stimuli.tsv", "stimulus_id")
    existing_descriptions = row_keys(directory / "annotations.tsv", "annot_id")
    append_rows(
        directory / "stimuli.tsv",
        [row for key, row in rows.items() if key not in existing_rows],
    )
    append_rows(
        directory / "annotations.tsv",
        [row for key, row in descriptions.items() if key not in existing_descriptions],
    )
    for name, (source, files) in sorted(copies.items()):
        table, sidecar = directory / name, directory / source["sidecar"]
        data = verified[name]
        if table.exists() or sidecar.exists():
            recorded = _recorded_source(sidecar)
            current = table.is_file() and hashlib.sha256(table.read_bytes()).hexdigest()
            if recorded != (source["version"], source["sha256"]) or current != source["sha256"]:
                messages.append(
                    f"Kept the existing {name}; it differs from the reviewed v{source['version']}. "
                    f"Run `nro stimuli sync {Path(project_root).name}` to compare it with the "
                    "catalog."
                )
            continue
        metadata = json.loads(Path(files[source["sidecar"]]).read_text(encoding="utf-8"))
        metadata[SOURCE_FIELD] = {
            "path": f"stimuli/{name}",
            "version": source["version"],
            "sha256": source["sha256"],
        }
        atomic_write_text(table, data.decode("utf-8"), mode=0o664)
        atomic_write_text(sidecar, json.dumps(metadata, indent=2) + "\n", mode=0o664)
        messages.append(f"Copied {name} v{source['version']}")
    return messages


def _recorded_source(sidecar: Path) -> tuple[int, str] | None:
    try:
        source = json.loads(sidecar.read_text(encoding="utf-8")).get(SOURCE_FIELD)
        return source["version"], source["sha256"]
    except (OSError, json.JSONDecodeError, AttributeError, KeyError, TypeError):
        return None
