"""Deterministic acquisition review, event validation, and association checks."""

import glob
import hashlib
import json
from copy import deepcopy
from io import StringIO
from pathlib import Path

import numpy as np

from nro.configuration.events import EventFile, EventStore
from nro.engine.events import validate_events

from .config import ALLOWED_TYPES, bids_label
from .identity import identity_issues


class SkipSession(Exception):
    """Leave the active session without changing its unfinished acquisition review."""


def event_candidates(config: dict, task: str) -> tuple[list[EventFile], list[str]]:
    """Return catalog entries and additional path candidates without choosing a variant."""
    entries = EventStore(Path(config["event_store"])).candidates(task)
    paths = sorted(
        {p for r in config["event_rules"] if r["task"] == task for p in glob.glob(r["pattern"])}
    )
    return entries, paths


def compatible(first: dict, second: dict, *, opposite: bool = False) -> bool:
    """Check EPI geometry and encoding before proposing a reference association."""
    a, b = first.get("metadata", {}), second.get("metadata", {})
    pa, pb = a.get("PhaseEncodingDirection"), b.get("PhaseEncodingDirection")
    if pa not in {"i", "i-", "j", "j-", "k", "k-"} or pb is None:
        return False
    desired = pa.rstrip("-") if pa.endswith("-") else pa + "-"
    if pb != (desired if opposite else pa):
        return False
    if "_shape" not in a or "_shape" not in b or a["_shape"][:3] != b["_shape"][:3]:
        return False
    if not np.allclose(a["_affine"], b["_affine"], atol=1e-3, rtol=0):
        return False
    if "TotalReadoutTime" not in a or "TotalReadoutTime" not in b:
        return False
    return bool(np.isclose(a["TotalReadoutTime"], b["TotalReadoutTime"], atol=1e-6, rtol=0))


def issues(record: dict, *, prepared: bool) -> list[str]:
    """Return unresolved publication requirements without applying heuristics."""
    unresolved = identity_issues(record) if prepared else []
    acquisitions = {a["id"]: a for a in record["acquisitions"]}
    names = set()
    used_sbrefs = set()
    fieldmap_groups = {}
    for item in acquisitions.values():
        name = item["id"]
        kind = item["datatype"], item["suffix"]
        if not item.get("confirmed") or kind not in ALLOWED_TYPES:
            unresolved.append(f"{name}: acquisition classification needs review")
            continue
        if item["datatype"] == "ignore":
            continue
        if item["datatype"] == "func" and item["suffix"] == "bold":
            task = item["entities"].get("task")
            if not task:
                unresolved.append(f"{name}: task is missing")
            if task != "rest":
                try:
                    metadata = item.get("metadata", {})
                    shape = metadata.get("_shape", [])
                    duration = (
                        shape[3] * metadata["RepetitionTime"]
                        if len(shape) == 4 and "RepetitionTime" in metadata
                        else None
                    )
                    validate_events(Path(item.get("events") or ""), duration=duration)
                except (ValueError, OSError, KeyError):
                    unresolved.append(f"{name}: valid events are required")
            if prepared:
                if "sbref" not in item:
                    unresolved.append(f"{name}: confirm SBRef or explicitly select none")
                elif item["sbref"] is not None:
                    ref = acquisitions.get(item["sbref"])
                    if ref is None or ref["suffix"] != "sbref" or not compatible(item, ref):
                        unresolved.append(f"{name}: incompatible SBRef")
                    if item["sbref"] in used_sbrefs:
                        unresolved.append(
                            f"{name}: SBRef already assigned to another BOLD; select a distinct reference or none"
                        )
                    used_sbrefs.add(item["sbref"])
                if "fieldmaps" not in item:
                    unresolved.append(f"{name}: confirm fieldmaps or explicitly select none")
                elif item["fieldmaps"]:
                    refs = [acquisitions.get(key) for key in item["fieldmaps"]]
                    if (
                        len(refs) != 2
                        or any(r is None or r["datatype"] != "fmap" for r in refs)
                        or not compatible(refs[0], refs[1], opposite=True)
                        or not any(compatible(item, r) for r in refs)
                    ):
                        unresolved.append(f"{name}: incompatible fieldmap pair")
                    pair = tuple(sorted(item["fieldmaps"]))
                    for key in pair:
                        if key in fieldmap_groups and fieldmap_groups[key] != pair:
                            unresolved.append(
                                f"{name}: a fieldmap is assigned to conflicting pairs"
                            )
                        fieldmap_groups[key] = pair
        if item["suffix"] != "sbref":
            entities = tuple(sorted(item["entities"].items()))
            target = item["datatype"], item["suffix"], entities
            if target in names:
                unresolved.append(f"{name}: duplicate output name; set distinct run/acq entities")
            names.add(target)
    if not any(a["datatype"] != "ignore" for a in acquisitions.values()):
        unresolved.append("No imaging acquisitions selected")
    return unresolved


def ask(prompt: str, default: str = "") -> str:
    """Read an answer; q exits and skip leaves the active session's saved decisions intact."""
    answer = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    if answer.lower() == "q":
        raise EOFError
    if answer.lower() == "skip":
        raise SkipSession
    return answer or default


def choose_indices(answer: str, length: int) -> list[int]:
    """Resolve numbered selections, rejecting zero, negative, and out-of-range values."""
    values = list(range(length)) if answer == "all" else [int(v) - 1 for v in answer.split()]
    if not values or any(v < 0 or v >= length for v in values):
        raise ValueError("Select one or more numbers from the displayed list")
    return list(dict.fromkeys(values))


def _entities(answer: str) -> dict:
    pairs = [part.split("=", 1) for part in answer.split()]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Use key=value for each entity")
    parsed = dict(pairs)
    if len(parsed) != len(pairs) or set(parsed) - {"task", "run", "acq", "dir", "echo"}:
        raise ValueError("Use each supported entity at most once")
    for key, value in parsed.items():
        bids_label(value)
        if key in {"run", "echo"} and (not value.isdigit() or int(value) < 1):
            raise ValueError(f"{key} must be a positive integer")
    return parsed


def _events(record: dict, item: dict) -> str | None:
    config = record["config"]
    catalog = EventStore(Path(config["event_store"]))
    task = item["entities"].get("task", "")
    entries, paths = event_candidates(config, task)
    choices = {entry.identifier: entry for entry in entries}
    for name in choices:
        print("Catalog event: " + name)
    if paths:
        print("Additional event candidates: " + ", ".join(paths))
    defaults = list(choices) + paths
    while True:
        answer = ask(
            "Event TASK/VARIANT ID or TSV path; s to defer",
            item.get("events") or (defaults[0] if len(defaults) == 1 else ""),
        )
        if answer == "s":
            return
        try:
            provenance = None
            if answer in choices or (
                len(answer.split("/")) == 2
                and not answer.endswith(".tsv")
                and not answer.startswith((".", "~", "/"))
            ):
                text, provenance = catalog.resolve(answer).snapshot()
            else:
                event_path = Path(answer).expanduser().resolve()
                text = event_path.read_text()
            metadata = item.get("metadata", {})
            shape = metadata.get("_shape", [])
            duration = (
                shape[3] * metadata["RepetitionTime"]
                if len(shape) == 4 and "RepetitionTime" in metadata
                else None
            )
            validate_events(StringIO(text), duration=duration)
        except (ValueError, OSError) as error:
            print(str(error))
            continue
        if (
            ask("Confirm event file contains no identifying columns or text? y/N", "n").lower()
            == "y"
        ):
            if provenance is not None:
                item["events_source"] = provenance
            elif (
                item.get("events") != str(event_path)
                or item.get("events_source", {}).get("sha256")
                != hashlib.sha256(text.encode("utf-8")).hexdigest()
            ):
                item.pop("events_source", None)
            return text
        return


def _references(
    prompt: str, refs: list[dict], current: list[str], *, count: int
) -> list[str] | None:
    for number, ref in enumerate(refs, 1):
        metadata = ref.get("metadata", {})
        print(
            f"{number}: {ref['id']} {ref['entities']} encoding={metadata.get('PhaseEncodingDirection')} time={metadata.get('AcquisitionTime')}"
        )
    default = " ".join(str(i + 1) for i, ref in enumerate(refs) if ref["id"] in current) or "none"
    while True:
        answer = ask(prompt + "; none or s to defer", default)
        if answer == "s":
            return None
        if answer == "none":
            return []
        try:
            selected = choose_indices(answer, len(refs))
            if len(selected) != count:
                raise ValueError(f"Select exactly {count} reference(s), or none")
            if count == 2 and not compatible(refs[selected[0]], refs[selected[1]], opposite=True):
                raise ValueError(
                    "Select an opposite-encoding pair with matching geometry and readout time"
                )
            return [refs[i]["id"] for i in selected]
        except ValueError as error:
            print(str(error))


def wizard(store, record: dict, *, review_token: str) -> dict:
    """Review one leased session, committing decisions and events after each acquisition."""
    prepared = record["stage"] == "convert"
    if prepared:
        for field in ("participant", "session"):
            if record[field] is not None:
                continue
            while True:
                answer = ask(f"BIDS {field} label; Enter to leave pending")
                if not answer:
                    break
                candidate = deepcopy(record)
                try:
                    candidate[field] = bids_label(answer)
                    candidate["issues"] = issues(candidate, prepared=True)
                    record = store.update(
                        candidate, expected_revision=record["revision"], review_token=review_token
                    )
                    break
                except ValueError as error:
                    print(str(error))
    source = None
    if not prepared:
        from .flywheel import FlywheelSource

        source = FlywheelSource(record["config"]["servers"][record["server"]])
    for index in range(len(record["acquisitions"])):
        event_files = {}
        item = deepcopy(record["acquisitions"][index])
        print(f"\nAcquisition {index + 1}: {item['id']}")
        if source is not None:
            print("Remote acquisition: " + source.describe(item))
        if prepared:
            print(json.dumps(item.get("metadata", {}), indent=2))
            if item["suffix"] != "bold":
                continue
            if item["entities"].get("task") != "rest":
                text = _events(record, item)
                if text is not None:
                    event_files[item["id"]] = text
            refs = [
                a for a in record["acquisitions"] if a["suffix"] == "sbref" and compatible(item, a)
            ]
            answer = _references("SBRef number", refs, [item.get("sbref")], count=1)
            if answer is not None:
                item["sbref"] = answer[0] if answer else None
            refs = [
                a
                for a in record["acquisitions"]
                if a["datatype"] == "fmap"
                and (compatible(item, a) or compatible(item, a, opposite=True))
            ]
            answer = _references(
                "Two fieldmap numbers separated by spaces", refs, item.get("fieldmaps", []), count=2
            )
            if answer is not None:
                item["fieldmaps"] = answer
        else:
            answer = ask(
                "Type: anat/T1w, anat/T2w, func/bold, func/sbref, fmap/epi, ignore, or s",
                f"{item['datatype']}/{item['suffix']}" if item["datatype"] else "",
            )
            if answer == "s":
                continue
            kind = ("ignore", "ignore") if answer == "ignore" else tuple(answer.split("/"))
            if kind not in ALLOWED_TYPES:
                print("Unsupported type; left unresolved.")
                continue
            item["datatype"], item["suffix"] = kind
            if kind[0] != "ignore":
                print(
                    "Confirm classification before download: raw anatomy is permitted only in /tmp."
                )
                if ask("Confirmed? y/N", "n").lower() != "y":
                    continue
                while True:
                    entities = ask(
                        "BIDS entities as key=value (task/run/acq/dir/echo); s to defer",
                        " ".join(f"{k}={v}" for k, v in item["entities"].items()),
                    )
                    try:
                        if entities != "s":
                            parsed = _entities(entities)
                        break
                    except ValueError as error:
                        print(str(error))
                if entities == "s":
                    continue
                item["entities"] = parsed
                if kind == ("func", "bold") and parsed.get("task") != "rest":
                    text = _events(record, item)
                    if text is not None:
                        event_files[item["id"]] = text
            item["confirmed"] = True
        record["acquisitions"][index] = item
        record["approval"] = None
        record["issues"] = issues(record, prepared=prepared)
        record = store.update(
            record,
            expected_revision=record["revision"],
            review_token=review_token,
            event_files=event_files,
        )
    record["issues"] = issues(record, prepared=prepared)
    if not record["issues"]:
        record.update(state="queued", stage="convert" if prepared else "prepare")
    else:
        print("\nStill unresolved:\n" + "\n".join(record["issues"]))
    return store.update(record, expected_revision=record["revision"], review_token=review_token)
