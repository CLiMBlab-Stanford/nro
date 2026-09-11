"""Run the pinned official MARSS correction behind a process boundary."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
from pathlib import Path

import numpy as np


def _publish_nifti(source: Path, destination: Path) -> None:
    if destination.name.endswith(".nii.gz"):
        with source.open("rb") as input_stream, gzip.open(destination, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
        source.unlink()
    else:
        os.replace(source, destination)


def main() -> None:
    """Run the official correction function and publish its two 4D outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bold", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--multiband-factor", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    from MARSS.MARSS import MARSS_removeSliceArtifact

    args.work_dir.mkdir(parents=True, exist_ok=True)
    native_input = args.work_dir / "marss_input.nii"
    if args.bold.name.endswith(".nii.gz"):
        with gzip.open(args.bold, "rb") as source, native_input.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    else:
        shutil.copyfile(args.bold, native_input)
    corrected, _ = MARSS_removeSliceArtifact(
        str(native_input),
        args.multiband_factor,
        np.loadtxt(args.motion, ndmin=2),
        str(args.work_dir),
    )
    corrected_path = Path(corrected)
    artifact_path = args.work_dir / "marss_input_slcart.nii"
    _publish_nifti(corrected_path, args.corrected)
    _publish_nifti(artifact_path, args.artifact)


if __name__ == "__main__":
    main()
