"""Create combined Connectome Workbench scenes from completed derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from nro.configuration.paths import WB_COMMAND_PATH
from nro.engine.bids import ENTITY_ORDER, parse_bids_entities
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.engine.scenes import (
    SceneSource,
    build_scene_bundle,
    has_surface_data,
    manifest_anatomical_images,
    manifest_surface_families,
    public_manifest_paths,
    template_surface_family,
)
from nro.engine.slurm import run_x11
from nro.engine.targets import DEFAULT_SMOOTHING_MM, DEFAULT_SPACE
from nro.engine.workbench import resolve_workbench_command, surface_inventory
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import BranchPaths
from nro.orchestration.catalog import MODULES


def _mapping(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    try:
        decoded = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return (
        {str(key): str(item) for key, item in decoded.items()} if isinstance(decoded, dict) else {}
    )


def _values(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(map(str, value))
    return tuple(part for part in str(value or "").split(",") if part)


def _expected_paths(row: dict) -> tuple[Path, ...]:
    value = row.get("expected_outputs_json") or "[]"
    try:
        paths = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        paths = ()
    return tuple(Path(str(path)).expanduser() for path in paths or ())


def _manifest_candidates(row: dict) -> tuple[Path, ...]:
    candidates = [
        path
        for path in _expected_paths(row)
        if path.is_file() and "manifest" in path.name and path.suffix in {".json", ".yaml", ".yml"}
    ]
    root = Path(str(row["output_root"]))
    if not candidates and root.is_dir():
        candidates.extend(
            path
            for pattern in ("*manifest.json", "*manifest.yaml", "*manifest.yml")
            for path in root.glob(pattern)
            if path.is_file()
        )
    return tuple(dict.fromkeys(candidates))


def _row_matches(row: dict, selection, *, match_module: bool = True) -> bool:
    if row.get("status") not in {None, "Success", "success"}:
        return False
    if selection.projects and row.get("project") not in selection.projects:
        return False
    if selection.participants and row.get("participant") not in selection.participants:
        return False
    if match_module and selection.modules and row.get("module") not in selection.modules:
        return False
    if selection.workflows and not set(selection.workflows).intersection(
        _values(row.get("workflow_ids"))
    ):
        return False
    entities = _mapping(row.get("entities_json"))
    for key, accepted in selection.instance_entities.items():
        if key in entities and accepted is not None and entities[key] not in accepted:
            return False
    return True


def _source(row: dict, path: Path, *, role: str = "derivative") -> SceneSource:
    entities = _mapping(row.get("entities_json"))
    return SceneSource(
        module=str(row["module"]),
        role=role,
        path=path,
        directory_label=str(row.get("directory_label") or "main"),
        output_prefix=str(row.get("output_prefix") or "output"),
        output_root=Path(str(row["output_root"])),
        project=str(row["project"]),
        participant=str(row["participant"]),
        space=entities.get("space"),
    )


def _collect(
    rows: list[dict],
) -> tuple[list[tuple[SceneSource, dict[str, str]]], tuple[SceneSource, ...]]:
    data: list[tuple[SceneSource, dict[str, str]]] = []
    surfaces: list[SceneSource] = []
    seen_data: set[Path] = set()
    seen_surfaces: set[Path] = set()
    for row in rows:
        row_entities = _mapping(row.get("entities_json"))
        for manifest in _manifest_candidates(row):
            try:
                paths = public_manifest_paths(str(row["module"]), manifest)
            except ValueError:
                continue
            for path in paths:
                if path in seen_data or path.name.endswith(".surf.gii"):
                    continue
                seen_data.add(path)
                data.append(
                    (_source(row, path), {**row_entities, **parse_bids_entities(path.name)})
                )
            try:
                family = manifest_surface_families(manifest)
            except ValueError:
                family = ()
            for path in family:
                if path not in seen_surfaces:
                    seen_surfaces.add(path)
                    surfaces.append(_source(row, path, role="display_surface"))
    return data, tuple(surfaces)


def _smoothing(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"([0-9]+)(?:mm)?", str(value))
    return int(match.group(1)) if match else None


def _targets(data, selection) -> tuple[tuple[str, int], ...]:
    spaces = selection.spaces or tuple(
        dict.fromkeys(entities["space"] for _, entities in data if entities.get("space"))
    )
    smoothing = selection.smoothing or tuple(
        dict.fromkeys(
            value
            for _, entities in data
            if (value := _smoothing(entities.get("smoothing"))) is not None
        )
    )
    spaces = spaces or (DEFAULT_SPACE,)
    smoothing = smoothing or (DEFAULT_SMOOTHING_MM,)
    return tuple((space, amount) for space in spaces for amount in smoothing)


def _run_groups(data, selection) -> tuple[dict[str, str], ...]:
    selectors = set(selection.runs)
    if not selectors:
        return ({},)
    fields = tuple(selectors) if selectors <= {"ses", "task"} else ENTITY_ORDER
    groups = []
    for _, entities in data:
        group = {field: entities[field] for field in fields if field in entities}
        if group and all(
            accepted is None or group.get(key) in accepted
            for key, accepted in selection.runs.items()
            if key in group
        ):
            groups.append(group)
    return tuple(dict.fromkeys(tuple(group.items()) for group in groups)) or ({},)  # type: ignore[return-value]


def _normalized_groups(data, selection) -> tuple[dict[str, str], ...]:
    groups = _run_groups(data, selection)
    return tuple(dict(group) if not isinstance(group, dict) else group for group in groups)


def _compatible(
    entities: dict[str, str], space: str, smoothing: int, group: dict[str, str]
) -> bool:
    if entities.get("space") not in {None, space}:
        return False
    amount = _smoothing(entities.get("smoothing"))
    if amount is not None and amount != smoothing:
        return False
    return all(entities.get(key) in {None, value} for key, value in group.items())


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(value))
    return cleaned or "selected"


def scene_id(participant: str, space: str, smoothing: int, group: dict[str, str], selection) -> str:
    """Return a readable, bounded identifier for one requested view."""

    tokens = [f"sub-{_safe(participant)}"]
    tokens.extend(f"{key}-{_safe(value)}" for key, value in group.items())
    tokens.extend((f"space-{_safe(space)}", f"smoothing-{smoothing}mm"))
    qualifiers = {
        "modules": selection.modules,
        "workflows": selection.workflows,
        "models": selection.models,
        "model_sets": selection.model_sets,
    }
    active = {key: value for key, value in qualifiers.items() if value}
    if active:
        digest = hashlib.sha256(json.dumps(active, sort_keys=True).encode()).hexdigest()[:10]
        tokens.append(f"selection-{digest}")
    return "_".join(tokens)


def _viewer_command(viewer: Path, scenes: list[Path]) -> list[str]:
    if len(scenes) != 1:
        raise ValueError(
            "--open requires selectors that generate exactly one scene; "
            f"the current selection generated {len(scenes)}"
        )
    return [str(viewer), "-scene-load-hd", str(scenes[0]), "1"]


def build_parser(*, prog: str = "nro.bin.scene") -> argparse.ArgumentParser:
    """Construct the scene command parser."""

    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--wb-command", default=WB_COMMAND_PATH)
    parser.add_argument(
        "--publish", action="store_true", help="Copy every input into the scene bundle"
    )
    parser.add_argument("--open", action="store_true", help="Open generated scenes in wb_view")
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.scene") -> None:
    """Build all scenes selected by argv and optionally open them."""

    args = build_parser(prog=prog).parse_args(argv)
    try:
        selection = core_selection(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    from nro.configuration.site import CHECKOUT, bids_root, settings
    from nro.orchestration.branch_views import registered_rows

    selected_bids_root = bids_root()
    rows = registered_rows(selected_bids_root)
    if rows is None:
        raise SystemExit("No central nro registry found")
    matched = [row for row in rows if _row_matches(row, selection)]
    support_rows = [row for row in rows if _row_matches(row, selection, match_module=False)]
    data, _unused_surfaces = _collect(matched)
    support_data, surface_sources = _collect(support_rows)
    if not data:
        raise SystemExit("No completed derivatives match the requested selectors")
    wb_command = resolve_workbench_command(args.wb_command)
    values = settings()[0]
    branches = BranchStore(Path(values["registry"]))
    branch = branches.read().topology.require_checkout(CHECKOUT)
    branch_paths = BranchPaths(
        branch,
        Path(values["bids"]),
        Path(values["work"]),
        Path(values["development"]),
    )
    scenes: list[Path] = []
    projects = selection.projects or tuple(dict.fromkeys(str(row["project"]) for row in matched))
    participants = selection.participants or tuple(
        dict.fromkeys(str(row["participant"]) for row in matched)
    )
    for project in projects:
        for participant in participants:
            local = [
                item
                for item in data
                if item[0].project == project and item[0].participant == participant
            ]
            if not local:
                continue
            for space, smoothing in _targets(local, selection):
                for group in _normalized_groups(local, selection):
                    selected = tuple(
                        source
                        for source, entities in local
                        if _compatible(entities, space, smoothing, group)
                    )
                    if not selected:
                        continue
                    geometry = ()
                    available_surfaces = any(
                        source.project == project
                        and source.participant == participant
                        and (
                            source.space == space or (space == "fsnative" and source.space is None)
                        )
                        for source in surface_sources
                    )
                    surface_target = has_surface_data(tuple(item.path for item in selected)) or (
                        space == "fsnative" and available_surfaces
                    )
                    roots = tuple(
                        dict.fromkeys(
                            source.output_root
                            for source in surface_sources
                            if source.project == project
                            and source.participant == participant
                            and (
                                source.space == space
                                or (space == "fsnative" and source.space is None)
                            )
                        )
                    )
                    for source_root in roots if surface_target else ():
                        candidate = tuple(
                            source
                            for source in surface_sources
                            if source.project == project
                            and source.participant == participant
                            and source.output_root == source_root
                        )
                        try:
                            surface_inventory(source.path for source in candidate)
                        except ValueError:
                            continue
                        geometry = candidate
                        break
                    if surface_target and not geometry:
                        try:
                            generated = template_surface_family(
                                tuple(item.path for item in selected), space
                            )
                        except (OSError, ValueError, FileNotFoundError):
                            generated = ()
                        geometry = tuple(
                            SceneSource(
                                "anat",
                                "display_surface",
                                path,
                                project=project,
                                participant=participant,
                                space=space,
                            )
                            for path in generated
                        )
                    if surface_target and not geometry:
                        raise SystemExit(
                            "No complete bilateral display-surface family is available for "
                            f"{project} sub-{participant} space-{space}"
                        )
                    anatomical: tuple[SceneSource, ...] = ()
                    if not surface_target and space == "T1w":
                        images: list[SceneSource] = []
                        for row in support_rows:
                            if (
                                row.get("module") != "anat"
                                or row.get("project") != project
                                or row.get("participant") != participant
                            ):
                                continue
                            for manifest in _manifest_candidates(row):
                                for path in manifest_anatomical_images(manifest)[:1]:
                                    images.append(_source(row, path, role="anatomical_reference"))
                        anatomical = tuple(images[:1])
                    anchors: tuple[SceneSource, ...] = ()
                    if not surface_target and not any(
                        source.path.name.endswith(".dlabel.nii")
                        or (
                            source.path.name.endswith((".nii", ".nii.gz"))
                            and not any(
                                marker in source.path.name
                                for marker in (
                                    ".dscalar.nii",
                                    ".dtseries.nii",
                                    ".pconn.nii",
                                )
                            )
                        )
                        for source in (*selected, *anatomical)
                    ):
                        anchors = tuple(
                            source
                            for source, entities in support_data
                            if source.project == project
                            and source.participant == participant
                            and source.module == "microparcellation"
                            and _compatible(entities, space, smoothing, group)
                            and (
                                source.path.name.endswith(".dlabel.nii")
                                or source.path.name.endswith((".nii", ".nii.gz"))
                            )
                        )[:1]
                    identifier = scene_id(participant, space, smoothing, group, selection)
                    root = (
                        branch_paths.output_project(project)
                        / "derivatives"
                        / "scenes"
                        / f"space-{space}_smoothing-{smoothing}mm"
                        / f"sub-{participant}"
                    )
                    if "ses" in group:
                        root /= f"ses-{group['ses']}"
                    destination = root / identifier
                    branch_paths.require_output(destination, project)
                    scenes.append(
                        build_scene_bundle(
                            destination,
                            scene_id=identifier,
                            sources=(*selected, *anatomical, *anchors),
                            surfaces=geometry,
                            selection=asdict(selection),
                            context={
                                "project": project,
                                "participant": participant,
                                "space": space,
                                "smoothing_fwhm_mm": smoothing,
                                "group_entities": group,
                            },
                            wb_command=wb_command,
                            publish=args.publish,
                        )
                    )
    if not scenes:
        raise SystemExit("No completed derivatives match the requested target groups")
    for scene in scenes:
        print(scene)
    if args.open:
        viewer = Path(wb_command).with_name("wb_view")
        if not viewer.is_file():
            raise SystemExit(f"Connectome Workbench viewer is not available: {viewer}")
        try:
            run_x11(
                _viewer_command(viewer, scenes),
                partition=values["viewing_partition"],
                account=values["account"] or None,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
