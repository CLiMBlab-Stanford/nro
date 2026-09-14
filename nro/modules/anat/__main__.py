"""Run anatomical preprocessing using the project's BIDS contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.configuration.markup import SubjectMarkup, load_source_markup
from nro.configuration.paths import BIDS_PATH
from nro.configuration.runtime import configure_anat, load_runtime_configuration
from nro.orchestration.runtime import select_runtime_config


def _bids_id(value: str, prefix: str) -> str:
    return value if value.startswith(f"{prefix}-") else f"{prefix}-{value}"


def _anatomicals(
    project: str,
    sub_id: str,
    *,
    bids_root: Path | None = None,
    markup: SubjectMarkup | None = None,
) -> tuple[list[Path], list[Path]]:
    subject_dir = Path(BIDS_PATH if bids_root is None else bids_root) / project / sub_id
    from nro.modules.anat.planning import raw_anatomical_images

    files = raw_anatomical_images(subject_dir, markup)
    t1w = [path for path in files if path.name.endswith(("_T1w.nii", "_T1w.nii.gz"))]
    t2w = [path for path in files if path.name.endswith(("_T2w.nii", "_T2w.nii.gz"))]
    if not t1w and not t2w:
        raise FileNotFoundError(f"No T1w or T2w images found under {subject_dir}")
    return t1w, t2w


def build_parser() -> argparse.ArgumentParser:
    """Build the direct anatomical-module parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("-w", "--workflow", default="main")
    return parser


def main(argv: list[str] | None = None, *, execution_context=None) -> None:
    """Resolve one anatomical instance and execute its runner graph."""
    args = build_parser().parse_args(argv)
    if execution_context is not None and execution_context.project != args.project:
        raise ValueError("Execution context project differs from the requested project")
    sub_id = _bids_id(args.participant, "sub")
    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        configuration_class="anat",
        execution_context=execution_context,
    )
    anat_id, cfg = load_runtime_configuration(runtime_config, "anat")
    configure_anat(args.project, anat_id, cfg)
    subject_dir = (
        Path(BIDS_PATH if execution_context is None else execution_context.paths.bids)
        / args.project
        / sub_id
    )
    markup = load_source_markup(cfg.get("markup"), args.project, subject_dir)
    t1w, t2w = _anatomicals(
        args.project,
        sub_id,
        bids_root=execution_context.paths.bids if execution_context else None,
        markup=markup,
    )

    from nro.modules.anat.module import main as run_anat

    module_argv = [
        "--project",
        args.project,
        "--anat-id",
        anat_id,
        "--sub-id",
        sub_id,
    ]
    for path in t1w:
        module_argv.extend(("--t1w", str(path)))
    for path in t2w:
        module_argv.extend(("--t2w", str(path)))
    run_anat(module_argv, **({"execution_context": execution_context} if execution_context else {}))


if __name__ == "__main__":
    main()
