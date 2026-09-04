"""Project discovery shared by orchestration commands."""

from pathlib import Path
from typing import Iterable

from nro.orchestration.registry import discover_registry_projects


def selected_projects(bids_root: Path, projects: Iterable[str] = ()) -> list[str]:
    """Return explicit projects or every project represented centrally."""
    requested = tuple(dict.fromkeys(projects))
    return list(requested) if requested else discover_registry_projects(bids_root)


def discover_bids_participants(project_root: Path) -> tuple[str, ...]:
    """Return participant IDs represented by source BIDS directories."""
    return tuple(
        sorted(
            path.name.removeprefix("sub-")
            for path in project_root.glob("sub-*")
            if path.is_dir()
        )
    )


def discover_bids_inventory(bids_root: Path) -> dict[str, tuple[str, ...]]:
    """Discover source BIDS project and participant directories without planning."""
    if not bids_root.is_dir():
        return {}
    inventory: dict[str, tuple[str, ...]] = {}
    for project in sorted(path for path in bids_root.iterdir() if path.is_dir()):
        participants = discover_bids_participants(project)
        if participants:
            inventory[project.name] = participants
    return inventory
