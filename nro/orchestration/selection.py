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


def discover_bids_inventory(
    bids_root: Path,
    *,
    projects: Iterable[str] = (),
    participants: Iterable[str] = (),
) -> dict[str, tuple[str, ...]]:
    """Discover the requested source-BIDS directories without planning.

    Explicit selectors avoid walking unrelated projects and participants on the
    shared filesystem. An empty selector retains whole-site discovery.
    """
    if not bids_root.is_dir():
        return {}
    requested_projects = tuple(dict.fromkeys(projects))
    requested_participants = tuple(
        value.removeprefix("sub-") for value in dict.fromkeys(participants)
    )
    project_paths = (
        tuple(bids_root / project for project in requested_projects)
        if requested_projects
        else tuple(sorted(path for path in bids_root.iterdir() if path.is_dir()))
    )
    inventory: dict[str, tuple[str, ...]] = {}
    for project in project_paths:
        if not project.is_dir():
            continue
        available = (
            tuple(
                participant
                for participant in requested_participants
                if (project / f"sub-{participant}").is_dir()
            )
            if requested_participants
            else discover_bids_participants(project)
        )
        if available:
            inventory[project.name] = available
    return inventory
