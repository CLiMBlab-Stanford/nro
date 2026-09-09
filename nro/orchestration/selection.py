"""Project discovery shared by orchestration commands."""

from pathlib import Path
from typing import Iterable

from nro.orchestration.registry import discover_registry_projects


def selected_projects(bids_root: Path, projects: Iterable[str] = ()) -> list[str]:
    """Return explicit projects or every project represented centrally."""
    requested = tuple(dict.fromkeys(projects))
    if requested:
        return list(requested)
    from nro.configuration.site import CHECKOUT, installation_record, settings
    from nro.orchestration.scheduler_implementation import implementation_path

    values = settings()[0]
    control = Path(values["registry"])
    if installation_record().get("mode") == "branch" or implementation_path(control).is_file():
        from nro.orchestration.scheduler_client import status

        if bids_root.resolve() != Path(values["bids"]).resolve():
            raise ValueError("Branch selection uses the shared site BIDS root")
        report = status(control, bids_root, checkout=CHECKOUT, mode="cached")
        visible = set(report["visible_ids"])
        return sorted({row["project"] for row in report["rows"] if row["id"] in visible})
    return discover_registry_projects(bids_root)


def discover_bids_participants(project_root: Path) -> tuple[str, ...]:
    """Return participant IDs represented by source BIDS directories."""
    return tuple(
        sorted(
            path.name.removeprefix("sub-") for path in project_root.glob("sub-*") if path.is_dir()
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
