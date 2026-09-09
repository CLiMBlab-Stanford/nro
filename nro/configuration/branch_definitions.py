"""Select development definitions without changing shared site settings."""

from __future__ import annotations

import json
from pathlib import Path

from nro.orchestration.branches import checkout_identity
from nro.orchestration.control_paths import ControlPaths


def _selection_file(control: Path, name: str) -> Path:
    path = ControlPaths(control).branch(name) / "definitions.json"
    if path.is_symlink():
        raise ValueError("Branch definitions selection cannot be a symbolic link")
    return path


def read_selection(control: Path, name: str, registry_id: str) -> Path | None:
    """Read a private selection, rejecting missing stores and mismatched registrations."""
    path = _selection_file(control, name)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"registry_id", "definitions"}
        or value["registry_id"] != registry_id
        or not isinstance(value["definitions"], str)
        or not Path(value["definitions"]).is_absolute()
    ):
        raise ValueError("Invalid branch definitions selection")
    selected = Path(value["definitions"])
    if selected.resolve() != selected or not selected.is_dir():
        raise ValueError(f"Branch definitions directory is missing or redirected: {selected}")
    return selected


def _disjoint(left: Path, right: Path) -> bool:
    return not (left.is_relative_to(right) or right.is_relative_to(left))


def selected_definitions(control: Path, installation: dict, shared: Path) -> Path | None:
    """Read a branch's explicit selection after checking its checkout binding.

    Return None for the shared, read-only default. This path is used while
    importing configuration, so it does not import or initialize registries.
    """
    paths = ControlPaths(control)
    paths.require_current_layout()
    root, name, _ = checkout_identity(Path(installation["checkout"]))
    records = json.loads(paths.catalog.read_text())
    record = records.get(name)
    if (
        name != installation.get("branch")
        or not isinstance(record, dict)
        or record.get("retired")
        or str(root) not in record.get("checkouts", ())
        or record.get("registry_id") != installation.get("registry_id")
    ):
        raise ValueError("Installation does not match an active branch registration")
    selected = read_selection(control, name, record["registry_id"])
    if selected is not None and not _disjoint(selected, shared.resolve()):
        raise ValueError("Development definitions must not overlap the shared definitions store")
    return selected


def require_private_store(control: Path, shared: Path, destination: Path, *, owner: str) -> None:
    """Reject overlap with shared definitions, control state, or another branch's store."""
    destination = destination.expanduser().absolute()
    if destination.resolve() != destination:
        raise ValueError("Development definitions cannot be redirected through symbolic links")
    if not _disjoint(destination, shared.resolve()) or not _disjoint(
        destination, control.resolve()
    ):
        raise ValueError(
            "Development definitions must be separate from shared definitions and control state"
        )
    records = json.loads(ControlPaths(control).catalog.read_text())
    for name, record in records.items():
        if name != owner:
            selected = read_selection(control, name, record["registry_id"])
            if selected is not None and not _disjoint(destination, selected):
                raise ValueError(f"Development definitions overlap the store selected by {name}")


def select_definitions(store, checkout: Path, shared: Path, destination: Path | None) -> Path:
    """Validate and publish a branch-wide selection; None restores shared read-only use.

    The destination must already be a valid definitions store. No files are
    copied, Git operations performed, or scientific observations changed.
    Cooperating selection updates are serialized with branch registration edits.
    """
    from nro.configuration.definitions import validate_store
    from nro.engine.io import atomic_write_json

    with store._lock():
        tree = store.read().topology
        name = tree.require_checkout(checkout)
        if name == "main":
            raise ValueError("Main uses the shared site definitions setting")
        record = tree.records[name]
        path = _selection_file(store.control, name)
        if destination is None:
            path.unlink(missing_ok=True)
            return shared.resolve()
        destination = destination.expanduser().absolute()
        require_private_store(store.control, shared, destination, owner=name)
        validate_store(destination)
        atomic_write_json(
            path,
            {"registry_id": record.registry_id, "definitions": str(destination)},
            mode=0o664,
            durable=True,
        )
        return destination
