"""BIDS run discovery and entity-selector primitives."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import read_json

# BIDS entity order is also the deterministic tie-breaker when multiple equally
# small selector sets can identify every run.
ENTITY_ORDER = ("ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part", "chunk")
NON_RUN_ENTITIES = {
    "sub",
    "space",
    "scale",
    "smoothing",
    "res",
    "den",
    "hemi",
    "desc",
    "label",
    "from",
    "to",
    "mode",
}


def acquisition_time_seconds(raw: str) -> float:
    """Parse a BIDS acquisition time into seconds after midnight."""
    item = str(raw).strip()
    try:
        hour, minute, remainder = item.split(":", 2)
        if "." in remainder:
            second, fraction = remainder.split(".", 1)
        else:
            second, fraction = remainder, "0"
        microsecond = int((fraction + "000000")[:6])
        parsed = time(
            hour=int(hour),
            minute=int(minute),
            second=int(second),
            microsecond=microsecond,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unrecognized AcquisitionTime: {raw!r}") from error
    return parsed.hour * 3600.0 + parsed.minute * 60.0 + parsed.second + parsed.microsecond / 1e6


def acquisition_order_key(metadata: Mapping[str, Any], path: Path) -> tuple[str, float]:
    """Return the best available chronological key for an imaging acquisition."""
    if "AcquisitionDateTime" in metadata:
        return (
            "acqdt",
            datetime.fromisoformat(str(metadata["AcquisitionDateTime"])).timestamp(),
        )
    if "AcquisitionTime" in metadata:
        return (
            "acqtime",
            acquisition_time_seconds(str(metadata["AcquisitionTime"])),
        )
    if "SeriesNumber" in metadata:
        return "series", float(metadata["SeriesNumber"])
    if "AcquisitionNumber" in metadata:
        return "acqnum", float(metadata["AcquisitionNumber"])
    return "mtime", Path(path).stat().st_mtime


@dataclass(frozen=True)
class BidsMetadata:
    """Effective metadata and its ordered BIDS inheritance sources."""

    values: Mapping[str, Any]
    sources: tuple[Path, ...]


def strip_bids_prefix(value: str, entity: str) -> str:
    """Remove an optional BIDS entity prefix such as ``sub-``."""
    return str(value).removeprefix(f"{entity}-")


def parse_bids_entities(name: str) -> dict[str, str]:
    """Parse key-value entities from a BIDS-like filename or stem."""
    stem = Path(name).name
    for suffix in (".nii.gz", ".func.gii", ".shape.gii", ".surf.gii", ".json", ".tsv", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    entities: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        if key and value:
            entities[key] = value
    return entities


def bids_suffix(name: str | Path) -> str:
    """Return the final BIDS suffix from a data or sidecar filename."""
    stem = Path(name).name
    for extension in (
        ".nii.gz",
        ".func.gii",
        ".shape.gii",
        ".label.gii",
        ".surf.gii",
        ".json",
        ".tsv",
        ".nii",
    ):
        if stem.endswith(extension):
            stem = stem[: -len(extension)]
            break
    return stem.rsplit("_", 1)[-1]


def bids_dataset_root(path: Path) -> Path:
    """Find the source-BIDS root governing a path without resolving symlinks."""
    path = Path(path).expanduser().absolute()
    for directory in path.parents:
        if (directory / "dataset_description.json").is_file():
            return directory
    for directory in path.parents:
        if directory.name.startswith("sub-"):
            return directory.parent
    raise ValueError(f"Could not determine the BIDS dataset root for {path}")


def resolve_bids_metadata(
    path: Path,
    *,
    dataset_root: Path | None = None,
) -> BidsMetadata:
    """Resolve JSON metadata using the BIDS inheritance principle.

    Applicable sidecars have the same suffix as ``path`` and a subset of its
    filename entities. Metadata are applied from the dataset root toward the
    image and from less-specific to more-specific filenames. Conflicting
    sidecars at the same location and specificity are rejected as ambiguous.
    """
    path = Path(path).expanduser().absolute()
    root = (
        Path(dataset_root).expanduser().absolute()
        if dataset_root is not None
        else bids_dataset_root(path)
    )
    try:
        relative_parent = path.parent.relative_to(root)
    except ValueError as error:
        raise ValueError(f"BIDS path {path} is not under dataset root {root}") from error

    directories = [root]
    current = root
    for part in relative_parent.parts:
        current = current / part
        directories.append(current)

    target_suffix = bids_suffix(path)
    target_entities = parse_bids_entities(path.name)
    effective: dict[str, Any] = {}
    sources: list[Path] = []
    for directory in directories:
        applicable: list[tuple[int, Path, dict[str, Any]]] = []
        for candidate in sorted(directory.glob("*.json")):
            if bids_suffix(candidate) != target_suffix:
                continue
            entities = parse_bids_entities(candidate.name)
            if any(target_entities.get(key) != value for key, value in entities.items()):
                continue
            metadata = read_json(candidate)
            if not isinstance(metadata, dict):
                raise ValueError(f"BIDS metadata sidecar is not a JSON object: {candidate}")
            applicable.append((len(entities), candidate, metadata))

        for specificity in sorted({item[0] for item in applicable}):
            peers = [item for item in applicable if item[0] == specificity]
            peer_values: dict[str, tuple[Any, Path]] = {}
            for _count, candidate, metadata in peers:
                for key, value in metadata.items():
                    previous = peer_values.get(key)
                    if previous is not None and previous[0] != value:
                        raise ValueError(
                            "Ambiguous BIDS metadata inheritance for "
                            f"{path}: {previous[1]} and {candidate} assign "
                            f"different values to {key!r} at equal specificity"
                        )
                    peer_values[key] = (value, candidate)
                sources.append(candidate)
            effective.update({key: value for key, (value, _source) in peer_values.items()})

    ordered_sources = tuple(dict.fromkeys(sources))
    if not ordered_sources:
        raise FileNotFoundError(f"No applicable BIDS JSON metadata found for {path}")
    return BidsMetadata(values=effective, sources=ordered_sources)


def bids_entity(path: Path, name: str, *, default: str | None = None) -> str | None:
    """Return one parsed filename entity."""
    return parse_bids_entities(Path(path).name).get(name, default)


def resolve_bids_table(path: Path, *, suffix: str) -> Path:
    """Find the most specific inherited TSV for an image, without merging rows.

    Search from the image directory toward the dataset root. At a given level,
    prefer the largest matching entity set; ties are ambiguous and raise
    ValueError. Raise FileNotFoundError when no applicable table exists.
    """
    path = Path(path).expanduser().absolute()
    root = bids_dataset_root(path)
    target = parse_bids_entities(path.name)
    for directory in path.parents:
        matches = []
        for candidate in directory.glob(f"*{suffix}.tsv"):
            entities = parse_bids_entities(candidate.name)
            if bids_suffix(candidate) == suffix and all(
                target.get(k) == v for k, v in entities.items()
            ):
                matches.append((len(entities), candidate))
        if matches:
            specificity = max(count for count, _ in matches)
            selected = [candidate for count, candidate in matches if count == specificity]
            if len(selected) != 1:
                raise ValueError(f"Ambiguous BIDS {suffix} tables for {path}: {selected}")
            return selected[0]
        if directory == root:
            break
    raise FileNotFoundError(f"No applicable BIDS {suffix}.tsv found for {path}")


def replace_bids_entity_token(path: Path, old: str, new: str) -> Path:
    """Replace one complete BIDS filename entity token."""
    path = Path(path)
    parts = path.name.split("_")
    matches = [index for index, part in enumerate(parts) if part == old]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {old!r} entity token in filename: {path}")
    parts[matches[0]] = new
    return path.with_name("_".join(parts))


def bids_readout_time(metadata: Mapping[str, Any]) -> float:
    """Return total readout time from either valid BIDS representation."""
    if "TotalReadoutTime" in metadata:
        return float(metadata["TotalReadoutTime"])
    echo_spacing = metadata.get("EffectiveEchoSpacing")
    matrix_size = metadata.get("ReconMatrixPE") or metadata.get("AcquisitionMatrixPE")
    if echo_spacing is None or matrix_size is None:
        raise ValueError(
            "Need TotalReadoutTime or EffectiveEchoSpacing+ReconMatrixPE/AcquisitionMatrixPE"
        )
    return float(echo_spacing) * (int(matrix_size) - 1)


def phase_encoding_direction_from_json(path: Path | None) -> str | None:
    """Read an optional BIDS phase-encoding direction without raising."""
    if path is None:
        return None
    try:
        value = str(read_json(path).get("PhaseEncodingDirection", "")).strip()
    except Exception:
        return None
    return value or None


def readout_time_from_json(path: Path | None) -> float | None:
    """Read an optional BIDS total readout time without raising."""
    if path is None:
        return None
    try:
        return bids_readout_time(read_json(path))
    except Exception:
        return None


def repetition_time_from_json(path: Path | None) -> float | None:
    """Read an optional positive BIDS repetition time without raising."""
    if path is None:
        return None
    try:
        value = read_json(path).get("RepetitionTime")
        repetition_time = float(value) if value is not None else 0.0
    except Exception:
        return None
    return repetition_time if repetition_time > 0 else None


def _parse_selector_values(
    values: Sequence[str] | None, *, reject_non_run: bool
) -> dict[str, tuple[str, ...] | None]:
    selectors: dict[str, tuple[str, ...] | None] = {}
    for raw in values or ():
        if "=" not in raw:
            raise ValueError(f"Run selector must use entity=value syntax, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip().removeprefix("--")
        requested = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
        if not key or "_" in key:
            raise ValueError(f"Invalid BIDS entity selector: {raw!r}")
        if reject_non_run and key in NON_RUN_ENTITIES:
            raise ValueError(f"{key!r} is not a selectable run-level BIDS entity")
        if key in selectors:
            previous = selectors[key]
            if previous is None or not requested:
                raise ValueError(
                    f"Cannot combine an absent-value selector with other values for {key!r}"
                )
            selectors[key] = tuple(dict.fromkeys((*previous, *requested)))
        else:
            selectors[key] = requested or None
    return selectors


def parse_selectors(
    values: Sequence[str] | None,
) -> dict[str, tuple[str, ...] | None]:
    """Parse run selectors with comma-delimited alternatives per entity."""
    return _parse_selector_values(values, reject_non_run=True)


def _session_from_path(path: Path) -> str | None:
    for parent in path.parents:
        if parent.name.startswith("ses-"):
            return parent.name.removeprefix("ses-")
    return None


def raw_run_stem(path: Path) -> str:
    """Remove the NIfTI extension and ``_bold`` suffix from a BOLD path."""
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]
    if not name.endswith("_bold"):
        raise ValueError(f"Not a BOLD NIfTI: {path}")
    return name[: -len("_bold")]


@dataclass(frozen=True)
class BidsRun:
    """One discovered BOLD acquisition and its identifying BIDS entities."""

    participant: str
    session: str | None
    stem: str
    entities: Mapping[str, str]
    path: Path
    selectors: tuple[str, ...] = ()


def run_arguments(run: BidsRun) -> tuple[str, ...]:
    """Build deterministic command-line selectors for one BIDS run."""
    return ("--run", *(f"{key}={value}" for key, value in sorted(run.entities.items())))


def _record(path: Path) -> BidsRun:
    stem = raw_run_stem(path)
    entities = parse_bids_entities(stem)
    participant = entities.get("sub")
    if not participant:
        raise ValueError(f"BOLD filename lacks a sub entity: {path}")
    session = entities.get("ses") or _session_from_path(path)
    if session is not None and "ses" not in entities:
        entities["ses"] = session
    run_entities = {key: value for key, value in entities.items() if key not in NON_RUN_ENTITIES}
    return BidsRun(participant, session, stem, run_entities, path)


def discover_raw_runs(subject_dir: Path) -> tuple[BidsRun, ...]:
    """Discover source BOLD runs at the BIDS subject or session level."""
    # Source BIDS permits functional data directly below the subject or one
    # session level below it. Avoid a recursive glob here: derivative trees or
    # other nested copies must never expand the source run universe.
    paths = [
        *subject_dir.glob("func/*_bold.nii*"),
        *subject_dir.glob("ses-*/func/*_bold.nii*"),
    ]
    records = [_record(path) for path in sorted(paths)]
    by_stem: dict[str, BidsRun] = {}
    for record in records:
        if record.stem in by_stem:
            raise ValueError(f"Duplicate raw BOLD run stem {record.stem!r} under {subject_dir}")
        by_stem[record.stem] = record
    return tuple(by_stem.values())


def _matches(
    entities: Mapping[str, str],
    selectors: Mapping[str, Sequence[str] | str | None],
) -> bool:
    return all(
        (
            key not in entities
            if value is None
            else entities.get(key)
            in ({value} if isinstance(value, str) else {str(item) for item in value})
        )
        for key, value in selectors.items()
    )


def matches_selectors(
    entities: Mapping[str, str],
    selectors: Mapping[str, Sequence[str] | str | None],
) -> bool:
    """Return whether a run/entity mapping satisfies exact CLI selectors."""
    return _matches(entities, selectors)


def matches_filter(
    entities: Mapping[str, str],
    filters: Mapping[str, object] | None,
) -> bool:
    """Apply a compact BIDS-Stats-Models-style entity filter.

    Scalar values require equality, a list accepts any listed value, and null
    requires the entity to be absent. Entity names use BIDS abbreviations.
    """
    for raw_key, expected in (filters or {}).items():
        key = str(raw_key)
        actual = entities.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in {str(value) for value in expected}:
                return False
        elif expected is None:
            if actual is not None:
                return False
        elif actual != str(expected):
            return False
    return True


def resolve_run(
    runs: Sequence[BidsRun],
    selectors: Mapping[str, Sequence[str] | str | None],
) -> BidsRun:
    """Resolve selectors to exactly one run or report the ambiguous candidates."""
    matches = [run for run in runs if _matches(run.entities, selectors)]
    rendered = (
        " ".join(
            f"{key}={'' if value is None else ','.join(value) if not isinstance(value, str) else value}"
            for key, value in selectors.items()
        )
        or "<none>"
    )
    if len(matches) != 1:
        candidates = [run.stem for run in matches] if matches else [run.stem for run in runs]
        state = "ambiguous" if matches else "matched no runs"
        raise ValueError(f"Run selectors {rendered} are {state}; candidates: {candidates}")
    return matches[0]


def minimal_selectors(target: BidsRun, runs: Sequence[BidsRun]) -> tuple[str, ...]:
    """Return the smallest entity subset that uniquely identifies ``target``."""
    if len(runs) == 1:
        return ()
    keys = sorted(
        set().union(*(run.entities.keys() for run in runs)),
        key=lambda key: (
            ENTITY_ORDER.index(key) if key in ENTITY_ORDER else len(ENTITY_ORDER),
            key,
        ),
    )
    for size in range(1, len(keys) + 1):
        for subset in itertools.combinations(keys, size):
            selectors = {key: target.entities.get(key) for key in subset}
            if sum(_matches(run.entities, selectors) for run in runs) == 1:
                return tuple(f"{key}={selectors[key] or ''}" for key in subset)
    raise ValueError(f"BIDS entities do not uniquely identify raw run {target.stem!r}")
