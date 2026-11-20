"""Expand stimulus presentations into run-time events from stimulus annotations.

The layout follows the BEP044 (Stim-BIDS) draft. A run ``events.tsv`` row with
a ``stim_id`` value records when a stimulus started. An annotation file
``stimuli/<stim_id>_annot-<label>_events.tsv`` gives event timing relative to
the start of that stimulus.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from nro.engine.bids import bids_dataset_root

if TYPE_CHECKING:
    import pandas as pd

# numpy and pandas are imported inside functions: planning imports this module
# for its error type and label check, and planning must not load array stacks.

ANNOTATION_LABEL = re.compile(r"[A-Za-z0-9]+")
STIMULUS_ID = re.compile(r"stim-[A-Za-z0-9]+")
# Allowed overrun of an annotation past its presentation, in seconds, for
# rounding in exported alignments.
DURATION_TOLERANCE = 1e-6


class StimulusAnnotationError(ValueError):
    """A run cannot supply the stimulus annotation that its model requests."""


def validate_annotation_label(value) -> str:
    """Return a BIDS label for the ``annot`` entity, or raise ValueError."""
    if not isinstance(value, str) or not ANNOTATION_LABEL.fullmatch(value):
        raise ValueError("annotation must be a BIDS label of letters and digits")
    return value


def presentation_ids(events: pd.DataFrame, source: Path | str) -> tuple[str, ...]:
    """Return the distinct ``stim_id`` values of presentation rows, in row order.

    Raise ValueError when the table has no ``stim_id`` column, no presentation
    rows, or a value that is not of the form ``stim-<label>``.
    """
    if "stim_id" not in events:
        raise StimulusAnnotationError(
            f"The model names an annotation, but {source} has no stim_id column"
        )
    values = events["stim_id"].dropna().map(str)
    if values.empty:
        raise StimulusAnnotationError(
            f"The model names an annotation, but {source} has no stim_id values"
        )
    invalid = sorted({value for value in values if not STIMULUS_ID.fullmatch(value)})
    if invalid:
        raise StimulusAnnotationError(
            f"stim_id values must have the form stim-<label> in {source}: {invalid}"
        )
    return tuple(dict.fromkeys(values))


def annotation_path(dataset_root: Path, stim_id: str, annotation: str) -> Path:
    """Locate one stimulus annotation file in the dataset's ``stimuli`` directory.

    Raise StimulusAnnotationError when the file is missing, naming the
    annotations that do exist and the command that copies annotations into the
    project. Also raise it when ``part-<label>`` annotation files exist for the
    stimulus, because nro does not yet define how parts map onto presentations.
    """
    directory = Path(dataset_root) / "stimuli"
    parts = sorted(directory.glob(f"{stim_id}_part-*_annot-{annotation}_events.tsv"))
    if parts:
        raise StimulusAnnotationError(
            f"Annotation parts are not supported yet; found {[path.name for path in parts]}"
        )
    path = directory / f"{stim_id}_annot-{annotation}_events.tsv"
    if not path.is_file():
        project = Path(dataset_root).name
        present = available_annotations(dataset_root, (stim_id,))[stim_id]
        found = (
            "Annotations present: " + ", ".join(present) + "."
            if present
            else "No annotations for this stimulus are present."
        )
        raise StimulusAnnotationError(
            f"Stimulus {stim_id} has no annot-{annotation} annotation in project {project}: "
            f"{path} does not exist. {found} Run `nro stimuli sync {project}` to copy "
            "stimulus annotations from the definitions catalog, or check the model's "
            "annotation label."
        )
    return path


def available_annotations(dataset_root: Path, stim_ids) -> dict[str, tuple[str, ...]]:
    """List the ``annot-<label>`` values present for each stimulus in ``stimuli/``."""
    directory = Path(dataset_root) / "stimuli"
    result = {}
    for stim_id in stim_ids:
        labels = (
            re.fullmatch(rf"{re.escape(stim_id)}_annot-([A-Za-z0-9]+)_events\.tsv", path.name)
            for path in directory.glob(f"{stim_id}_annot-*_events.tsv")
        )
        result[stim_id] = tuple(sorted(match.group(1) for match in labels if match))
    return result


def annotation_inputs(events_path: Path, run_path: Path, annotation: str) -> tuple[Path, ...]:
    """Resolve the annotation files a run needs, for planning and step inputs.

    Missing annotations raise errors here so that they are reported before
    work is scheduled.
    """
    import pandas as pd

    events = pd.read_csv(events_path, sep="\t")
    root = bids_dataset_root(run_path)
    return tuple(
        annotation_path(root, stim_id, annotation)
        for stim_id in presentation_ids(events, events_path)
    )


def expand_presentations(
    events: pd.DataFrame,
    *,
    annotation: str,
    dataset_root: Path,
    source: Path | str = "events.tsv",
) -> tuple[pd.DataFrame, dict]:
    """Replace each presentation row with its stimulus's annotation rows.

    Run onsets are the presentation onset plus the annotation onset. Other
    presentation columns are copied onto each expanded row, and rows without a
    ``stim_id`` value are kept. The result is sorted by onset with a stable
    sort, so rows with equal onsets keep their input order.

    Return the expanded table and a mapping from each ``stim_id`` to the path
    and SHA-256 of the annotation file used. Raise ValueError for column
    collisions and for annotations that outlast a presentation with a positive
    duration.
    """
    import numpy as np
    import pandas as pd

    from nro.engine.events import validate_events

    presentation_ids(events, source)
    stimulus = events["stim_id"].notna()
    loaded = {}
    sources = {}
    pieces = []
    for index, row in events.iterrows():
        if not stimulus[index]:
            pieces.append(events.loc[[index]].drop(columns="stim_id"))
            continue
        stim_id = str(row["stim_id"])
        if stim_id not in loaded:
            path = annotation_path(dataset_root, stim_id, annotation)
            data = path.read_bytes()
            validate_events(path)
            table = pd.read_csv(path, sep="\t")
            collisions = sorted(set(table) & set(events) - {"onset", "duration"})
            if collisions:
                raise ValueError(
                    f"Columns {collisions} occur in both {source} and {path.name}; rename one"
                )
            loaded[stim_id] = table
            sources[stim_id] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
        table = loaded[stim_id]
        start = float(row["onset"])
        length = float(row["duration"])
        ends = table["onset"].to_numpy(float) + table["duration"].to_numpy(float)
        if length > 0 and (ends > length + DURATION_TOLERANCE).any():
            raise ValueError(
                f"annot-{annotation} events for {stim_id} end after its {length:g} s "
                f"presentation at onset {start:g} in {source}"
            )
        expanded = table.copy()
        expanded["onset"] = start + table["onset"].to_numpy(float)
        for column in events.columns:
            if column not in {"onset", "duration", "stim_id"}:
                expanded[column] = row[column]
        pieces.append(expanded)
    columns = ["onset", "duration"]
    columns += [c for c in events.columns if c not in {"onset", "duration", "stim_id"}]
    columns += [
        c for table in loaded.values() for c in table.columns if c not in {"onset", "duration"}
    ]
    result = pd.concat(pieces, ignore_index=True).reindex(columns=list(dict.fromkeys(columns)))
    order = np.argsort(result["onset"].to_numpy(float), kind="stable")
    return result.iloc[order].reset_index(drop=True), sources
