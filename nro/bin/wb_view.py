"""Open Workbench scenes stored with completed derivatives."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import yaml

from nro.configuration.paths import BIDS_PATH, WB_COMMAND_PATH
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.engine.cli import matches_instance_selectors as matches_selectors
from nro.engine.workbench import resolve_workbench_command
from nro.orchestration.registry import Registry
from nro.orchestration.selection import selected_projects

DERIVATIVE_TYPES = ("dynconn", "microparcellation", "networks")
LOG = logging.getLogger(__name__)


def _resolve_path(value: object, manifest_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read derivative manifest: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Derivative manifest is not a mapping: {path}")
    return document


def _artifact_scene(manifest_path: Path) -> Path:
    """Return the scene recorded by a completed artifact."""
    manifest = _read_manifest(manifest_path)
    outputs = dict(manifest.get("outputs") or {})
    value = outputs.get("scene")
    if not value:
        raise ValueError(f"Derivative manifest does not record outputs.scene: {manifest_path}")
    scene = _resolve_path(value, manifest_path)
    if not scene.is_file():
        raise FileNotFoundError(f"Derivative scene is missing: {scene}")
    return scene


def _matching_rows(
    registry: Registry | None, args: argparse.Namespace, *, records=None
) -> list[dict]:
    selection = core_selection(args)
    projects = set(selected_projects(Path(args.bids_root), selection.projects))
    participants = set(selection.participants)
    workflows = set(selection.workflows)
    selectors = selection.instance_entities
    rows = []
    for row in registry.instance_rows(read_only=True) if records is None else records:
        if row["module"] != args.derivative_type or row["project"] not in projects:
            continue
        if participants and row["participant"] not in participants:
            continue
        if workflows and not workflows.intersection(str(row.get("workflow_ids") or "").split(",")):
            continue
        if selectors and not matches_selectors(json.loads(row["entities_json"]), selectors):
            continue
        rows.append(row)
    return rows


def build_parser(*, prog: str = "nro.bin.wb_view") -> argparse.ArgumentParser:
    """Construct the wb_view parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("derivative_type", choices=DERIVATIVE_TYPES)
    add_core_selection_arguments(parser, module_choices=DERIVATIVE_TYPES)
    parser.add_argument("--bids-root", default=BIDS_PATH)
    parser.add_argument(
        "--wb-command",
        default=WB_COMMAND_PATH,
        help="Path to Connectome Workbench's wb_command executable",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print matching scene paths without opening wb_view",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.wb_view") -> None:
    """Resolve completed scenes and open Workbench unless no-open is requested.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if selection.modules and selection.modules != (args.derivative_type,):
        raise SystemExit("--module must match DERIVATIVE_TYPE when it is supplied")
    from nro.orchestration.branch_views import registered_rows

    records = registered_rows(Path(args.bids_root))
    registry = (
        Registry.for_project("", bids_root=Path(args.bids_root).expanduser().resolve())
        if records is None
        else None
    )
    if registry is not None and not registry.existing_database_path().is_file():
        raise SystemExit("No central nro registry found")
    wb_command = resolve_workbench_command(args.wb_command)
    scenes: list[Path] = []
    for row in _matching_rows(registry, args, records=records):
        subject_directory = Path(row["output_root"])
        prefix = str(row["output_prefix"])
        description = (
            "dynamicConnectivity" if args.derivative_type == "dynconn" else args.derivative_type
        )
        manifest_name = f"{prefix}_desc-{description}_manifest.yaml"
        manifest_path = subject_directory / manifest_name
        if not manifest_path.is_file():
            LOG.warning("Skipping incomplete derivative: %s", manifest_path)
            continue
        try:
            scenes.append(_artifact_scene(manifest_path))
        except (FileNotFoundError, ValueError) as error:
            LOG.warning("Skipping incomplete derivative: %s", error)
    if not scenes:
        raise SystemExit("No completed derivatives match the requested selectors")
    for scene in scenes:
        print(scene)
    if args.no_open:
        return
    wb_view = Path(wb_command).with_name("wb_view")
    if not wb_view.is_file():
        raise SystemExit(f"Connectome Workbench viewer is not available: {wb_view}")
    subprocess.Popen([str(wb_view), *(str(scene) for scene in scenes)])


if __name__ == "__main__":
    main()
