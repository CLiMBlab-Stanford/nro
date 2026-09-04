"""Run anatomical preprocessing using the project's BIDS contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.runtime import select_runtime_config
from nro.configuration.runtime import configure_preprocessing, load_runtime_configuration


def _bids_id(value: str, prefix: str) -> str:
    return value if value.startswith(f"{prefix}-") else f"{prefix}-{value}"


def _anatomicals(project: str, sub_id: str) -> tuple[list[Path], list[Path]]:
    subject_dir = Path(BIDS_PATH) / project / sub_id
    files = sorted(subject_dir.glob("anat/*.nii*"))
    files.extend(sorted(subject_dir.glob("ses-*/anat/*.nii*")))
    t1w = [path for path in files if path.name.endswith(("_T1w.nii", "_T1w.nii.gz"))]
    t2w = [path for path in files if path.name.endswith(("_T2w.nii", "_T2w.nii.gz"))]
    if not t1w and not t2w:
        raise FileNotFoundError(f"No T1w or T2w images found under {subject_dir}")
    return t1w, t2w


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", required=True)
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("-w", "--workflow", default="main")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sub_id = _bids_id(args.participant, "sub")
    runtime_config = select_runtime_config(
        project=args.project, workflow_id=args.workflow, derivative_class="preprocessing"
    )
    preprocessing_id, cfg = load_runtime_configuration(runtime_config, "preprocessing")
    configure_preprocessing(args.project, preprocessing_id, cfg)
    t1w, t2w = _anatomicals(args.project, sub_id)

    from nro.anat.module import main as run_anat

    module_argv = [
        "--project", args.project,
        "--preprocessing-id", preprocessing_id,
        "--sub-id", sub_id,
        "--nthreads", str(cfg["anat"]["nthreads"]),
    ]
    for path in t1w:
        module_argv.extend(("--t1w", str(path)))
    for path in t2w:
        module_argv.extend(("--t2w", str(path)))
    run_anat(module_argv)


if __name__ == "__main__":
    main()
