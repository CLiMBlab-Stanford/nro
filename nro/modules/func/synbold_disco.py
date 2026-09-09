"""Validate and run the upstream SynBOLD-DisCo container."""

from __future__ import annotations

import filecmp
import importlib.resources
import shutil
from pathlib import Path
from typing import Callable, Optional

from nro.orchestration.runner_graph import Step


def ensure_image(
    *,
    image: Path,
    engine: str,
) -> Path:
    """Require an existing container image and container engine."""
    image = image.expanduser()
    if not image.is_file() or image.stat().st_size == 0:
        raise SystemExit(
            f"SynBOLD-DisCo image not found or empty: {image}. "
            "Provide it before starting preprocessing or configure "
            "synbold_disco_image."
        )
    if shutil.which(engine) is None:
        raise SystemExit(f"Container engine not found on PATH: {engine!r}")
    return image


def create_synthetic_reference_step(
    *,
    run_child: Callable[..., Optional[str]],
    distorted_reference: Path,
    skull_stripped_t1: Path,
    epi_to_t1_mat: Path,
    image: Path,
    license_file: Path,
    engine: str,
    work_dir: Path,
    force: bool,
) -> Step:
    """Create the synthetic-undistorted-EPI production step."""
    if not license_file.is_file():
        raise SystemExit(f"FreeSurfer license not found: {license_file}")

    inputs_dir = work_dir / "inputs"
    outputs_dir = work_dir / "outputs"
    bold_input = inputs_dir / "BOLD_d.nii.gz"
    t1_input = inputs_dir / "T1.nii.gz"
    transform_input = inputs_dir / "epi_reg_d.mat"
    synthetic = outputs_dir / "BOLD_s_3D.nii.gz"
    container_transform = outputs_dir / "epi_reg_d.mat"
    shim_verified = outputs_dir / ".nro_epi_reg_shim_v2"

    def synthesize() -> None:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for source, destination in (
            (distorted_reference, bold_input),
            (skull_stripped_t1, t1_input),
            (epi_to_t1_mat, transform_input),
        ):
            shutil.copy2(source, destination)
        shim_resource = importlib.resources.files(
            "nro.modules.func.resources.synbold_disco"
        ).joinpath("epi_reg")
        with importlib.resources.as_file(shim_resource) as shim:
            command = [
                engine,
                "run",
                "--cleanenv",
                "-B",
                f"{inputs_dir}:/INPUTS:ro",
                "-B",
                f"{outputs_dir}:/OUTPUTS",
                "-B",
                f"{shim}:/opt/fsl/bin/epi_reg:ro",
                "-B",
                f"{license_file}:/opt/freesurfer/license.txt:ro",
                "-B",
                "/tmp:/tmp",
                str(image),
                "--no_topup",
                "--no_smoothing",
                "--skull_stripped",
            ]
            try:
                run_child(command, direct=True)
            except SystemExit as error:
                log_path = outputs_dir / "output.log"
                detail = f" Upstream log: {log_path}." if log_path.exists() else ""
                raise SystemExit(f"{error}{detail}") from error
            if not container_transform.is_file() or not filecmp.cmp(
                transform_input, container_transform, shallow=False
            ):
                shim_verified.unlink(missing_ok=True)
                raise SystemExit(
                    "SynBOLD-DisCo did not honor nro's precomputed EPI-to-T1 "
                    f"transform: {container_transform} does not match {transform_input}."
                )
            shim_verified.write_text(
                "The container epi_reg call used nro's precomputed transform.\n",
                encoding="utf-8",
            )

    return Step.python(
        name="SynBOLD-DisCo EPI Synthesis",
        outputs=(synthetic, shim_verified),
        inputs=(distorted_reference, skull_stripped_t1, epi_to_t1_mat, image, license_file),
        force=force,
        action=synthesize,
    )
