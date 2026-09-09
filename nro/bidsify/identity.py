"""BIDS identity requirements for staged ingestion requests."""

from .config import bids_label


def identity_issues(record: dict) -> list[str]:
    """List labels that must be resolved before BIDS organization and publication."""
    unresolved = []
    for field in ("participant", "session"):
        value = record[field]
        if value is None:
            unresolved.append(f"BIDS {field} label is required before publication")
        else:
            bids_label(value)
    return unresolved


def destination_label(record: dict) -> str:
    """Describe a proposed destination without treating missing labels as identifiers."""
    participant = (
        f"sub-{record['participant']}" if record["participant"] else "(participant pending)"
    )
    session = f"ses-{record['session']}" if record["session"] else "(session pending)"
    return f"{record['project']}/{participant}/{session}"
