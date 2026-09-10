"""Propose task-model drafts from source BIDS event tables."""

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml

from nro.engine.bids import discover_raw_runs, resolve_bids_table
from nro.modules.firstlevels.task_models import validate_task_model


def discover_event_files(
    task: str,
    bids_root: Path,
    *,
    projects: Sequence[str] = (),
    participants: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Resolve events for matching source runs without loading images or a registry.

    A shared inherited table appears once. Missing tables and unknown project or
    participant selections raise errors instead of yielding a partial draft.
    """
    available = {
        path.name: path for path in bids_root.iterdir() if path.is_dir() and any(path.glob("sub-*"))
    }
    if set(projects) - available.keys():
        raise ValueError(f"Unknown BIDS project(s): {sorted(set(projects) - available.keys())}")
    selected = {value.removeprefix("sub-") for value in participants}
    found = set()
    paths = []
    for name in sorted(projects or available):
        for subject in sorted(available[name].glob("sub-*")):
            participant = subject.name.removeprefix("sub-")
            if not subject.is_dir() or (selected and participant not in selected):
                continue
            found.add(participant)
            for run in discover_raw_runs(subject):
                if run.entities.get("task") == task:
                    paths.append(resolve_bids_table(run.path, suffix="events"))
    if selected - found:
        raise ValueError(f"Unknown participant(s) in selected projects: {sorted(selected - found)}")
    if not paths:
        raise ValueError(
            f"No source BOLD runs for task {task!r}; supply --events FILE ... or --file FILE"
        )
    return tuple(dict.fromkeys(paths))


def model_draft(
    paths: Sequence[Path],
    *,
    conditions: str | None = None,
    choose: Callable[[tuple[str, ...]], str] | None = None,
    report: Callable[[str], None] = print,
) -> str:
    """Infer categorical effects from the union of event tables.

    Prefer trial_type; otherwise ask choose to select a common event column.
    Numeric predictors and between-condition contrasts are not inferred. Labels
    use the same pandas parsing as firstlevels execution. The draft has no model
    set membership and still needs scientific review.
    """
    tables = []
    for path in dict.fromkeys(paths):
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream, delimiter="\t"), [])
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate event column names: {path}")
        table = pd.read_csv(path, sep="\t")
        if table.empty or not {"onset", "duration"}.issubset(table.columns):
            raise ValueError(f"Events require rows, onset, and duration: {path}")
        for name in ("onset", "duration"):
            values = pd.to_numeric(table[name], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or (name == "duration" and (values < 0).any()):
                raise ValueError(f"Invalid {name} values: {path}")
        tables.append((path, table))
    if not tables:
        raise ValueError("At least one event file is required")
    common = set.intersection(*(set(table.columns) for _, table in tables)) - {"onset", "duration"}
    columns = set.union(*(set(table.columns) for _, table in tables)) - {"onset", "duration"}
    report(f"Inspected {len(tables)} distinct event table(s).")
    for name in sorted(columns):
        present = [table[name] for _, table in tables if name in table]
        kinds = {
            "numeric" if pd.api.types.is_numeric_dtype(values) else "categorical"
            for values in present
        }
        report(
            f"  {name}: {'/'.join(sorted(kinds))}; present in {len(present)}/{len(tables)} tables"
        )
    if conditions is None:
        if "trial_type" in common:
            conditions = "trial_type"
        elif "trial_type" in columns:
            raise ValueError(
                "trial_type is absent from some tables; narrow discovery or specify --conditions COLUMN"
            )
        elif choose is not None and common:
            conditions = choose(tuple(sorted(common)))
        else:
            raise ValueError(
                f"Choose --conditions from the common event columns: {', '.join(sorted(common))}"
            )
    if conditions not in common:
        raise ValueError(f"Condition column {conditions!r} must occur in every event table")
    counts = Counter()
    presence = Counter()
    missing = 0
    for _, table in tables:
        labels = table[conditions].dropna().map(str)
        counts.update(labels)
        presence.update(set(labels))
        missing += int(table[conditions].isna().sum())
    if not counts:
        raise ValueError(f"Condition column {conditions!r} has no observed values")
    contrasts = {}
    for label in sorted(counts):
        if any(char in label for char in "*?["):
            raise ValueError(
                f"Condition label {label!r} contains selector wildcards; write a model explicitly"
            )
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", label).lstrip("._-") or "condition"
        unique = name
        index = 2
        while unique in contrasts:
            unique = f"{name}-{index}"
            index += 1
        # A label containing the column prefix needs explicit qualification so
        # the compiler does not mistake part of its literal name for that prefix.
        predictor = f"{conditions}.{label}" if label.startswith(f"{conditions}.") else label
        contrasts[unique] = {predictor: 1}
        report(
            f"  Condition {label!r}: {counts[label]} events, {presence[label]}/{len(tables)} tables; contrast {unique}"
        )
    if missing:
        report(
            f"Warning: {missing} events have no condition label and receive no condition predictor."
        )
    report(
        "Other event columns were not added as predictors. Review comparisons, baseline, and HRF before saving."
    )
    model = {
        "model_set": [],
        "conditions": conditions,
        "hrf": "spm",
        "contrasts": contrasts,
    }
    validate_task_model(model)
    return (
        "# Draft inferred from event tables. Review the scientific design before saving.\n"
        "# Each contrast estimates one condition against the implicit baseline.\n"
        "# Add model_set: main only when this model should enter default requests.\n"
        + yaml.safe_dump(model, sort_keys=False, default_flow_style=None)
    )
