"""Render selected Workbench scene maps as image files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from nro.bin.scene import generate_scenes
from nro.configuration.paths import WB_COMMAND_PATH
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.engine.rendering import RENDER_FORMATS, read_seed_file, render_scene
from nro.engine.workbench import resolve_workbench_command
from nro.orchestration.catalog import MODULES


def build_parser(*, prog: str = "nro.bin.render") -> argparse.ArgumentParser:
    """Construct the render command parser."""

    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(parser, module_choices=MODULES)
    parser.add_argument("--wb-command", default=WB_COMMAND_PATH)
    parser.add_argument(
        "--seeds",
        type=Path,
        help="YAML file containing subject-specific T1w xyz_mm seed coordinates",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2400,
        help="Rendered image width in pixels (default: 2400)",
    )
    parser.add_argument(
        "--format",
        choices=RENDER_FORMATS,
        default="png",
        dest="image_format",
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--no-scene-colors",
        action="store_true",
        help="Use Workbench's rendering colors instead of colors stored in the scene",
    )
    parser.add_argument(
        "--warn-seed-distance",
        type=float,
        default=10.0,
        metavar="MM",
        help="Warn when a resolved seed is farther than this distance (default: 10 mm)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Render destination; multiple scenes create one subdirectory per scene",
    )
    return parser


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.render") -> None:
    """Create scenes and render all selected finite maps and requested seeds."""

    args = build_parser(prog=prog).parse_args(argv)
    if args.width <= 0:
        raise SystemExit("--width must be positive")
    if args.warn_seed_distance < 0:
        raise SystemExit("--warn-seed-distance cannot be negative")
    try:
        selection = core_selection(args)
        seeds = read_seed_file(args.seeds) if args.seeds else {}
        wb_command = resolve_workbench_command(args.wb_command)
        scenes, _settings = generate_scenes(
            selection,
            wb_command=wb_command,
            publish=False,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    manifests = []
    multiple = len(scenes) > 1
    for scene in scenes:
        try:
            document = yaml.safe_load(
                (scene.parent / "scene_manifest.yaml").read_text(encoding="utf-8")
            )
            key = str(document["project"]), str(document["participant"])
            coordinates = seeds.get(key, ())
            manifest = render_scene(
                scene,
                wb_command=wb_command,
                coordinates=coordinates,
                width=args.width,
                image_format=args.image_format,
                no_scene_colors=args.no_scene_colors,
                warn_distance_mm=args.warn_seed_distance,
                destination=(
                    args.output_dir / f"{key[0]}_{scene.stem}"
                    if args.output_dir and multiple
                    else args.output_dir
                ),
                progress=_progress,
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as error:
            raise SystemExit(str(error)) from error
        rendered = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        for module in rendered.get("skipped_seed_modules", ()):
            _progress(
                f"Skipped seed-driven {module} maps for {key[0]} sub-{key[1]}: "
                "no matching --seeds coordinates"
            )
        manifests.append(manifest)
    for manifest in manifests:
        print(manifest)


if __name__ == "__main__":
    main()
