"""Create HCP gradient-distortion corrections from resolved site policy."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from nro.configuration.hardware import (
    GRADIENT_UNWARP_IMAGE,
    GRADIENT_UNWARP_METHOD,
    GradientUnwarpingResolution,
)
from nro.engine.images import nifti_is_valid
from nro.engine.io import atomic_output_path, atomic_write_json
from nro.orchestration.runner_graph import Step


def _validate_outputs(
    source: Path,
    corrected: Path,
    warp: Path,
    metadata: Path,
) -> tuple[bool, str]:
    try:
        source_image = nib.load(str(source))
        corrected_image = nib.load(str(corrected))
        warp_image = nib.load(str(warp))
        if not metadata.is_file() or metadata.stat().st_size == 0:
            return False, f"Gradient-unwarping metadata is missing: {metadata}"
    except (OSError, ValueError) as error:
        return False, f"Gradient-unwarping output is unreadable: {error}"
    if corrected_image.shape != source_image.shape:
        return False, (
            f"Gradient correction changed image shape from {source_image.shape} "
            f"to {corrected_image.shape}"
        )
    if not np.array_equal(corrected_image.affine, source_image.affine):
        return False, "Gradient correction changed the image affine"
    expected_warp_shape = (*source_image.shape[:3], 3)
    if warp_image.shape != expected_warp_shape:
        return False, f"Gradient warp shape is {warp_image.shape}; expected {expected_warp_shape}"
    if not np.array_equal(warp_image.affine, source_image.affine):
        return False, "Gradient warp does not use the source image affine"
    return True, "Gradient-corrected image and relative warp are valid."


def create_gradient_unwarping_step(
    *,
    run_child: Callable[..., str | None],
    source: Path,
    corrected: Path,
    warp: Path,
    metadata: Path,
    resolution: GradientUnwarpingResolution,
    runtime: str,
    image: Path,
    force: bool,
) -> Step:
    """Create one corrected image and retain its source-grid relative warp."""
    if not resolution.applied or resolution.coefficients is None:
        raise ValueError("A gradient-unwarping step requires an applied policy resolution")

    provenance = {
        **resolution.scientific_record(),
        "tool": GRADIENT_UNWARP_METHOD,
        "tool_image": GRADIENT_UNWARP_IMAGE,
    }

    def unwarp() -> None:
        corrected.parent.mkdir(parents=True, exist_ok=True)
        warp.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".gradient-unwarp-", dir=corrected.parent) as raw:
            work = Path(raw)
            source_name = "source.nii.gz" if source.name.endswith(".nii.gz") else "source.nii"
            corrected_4d = work / "corrected.nii.gz"
            relative_warp = work / "gradient_warp.nii.gz"
            prefix = [
                runtime,
                "exec",
                "--cleanenv",
                "--bind",
                f"{source}:/input/{source_name}:ro",
                "--bind",
                f"{resolution.coefficients}:/input/coefficients.grad:ro",
                "--bind",
                f"{work}:/work",
                str(image),
            ]
            container_env = dict(os.environ)
            for prefix_name in ("SINGULARITYENV_", "APPTAINERENV_"):
                container_env[prefix_name + "HCPPIPEDIR"] = "/opt/HCP-Pipelines"
                container_env[prefix_name + "FSLDIR"] = "/usr/share/fsl"
                container_env[prefix_name + "FSLOUTPUTTYPE"] = "NIFTI_GZ"
                container_env[prefix_name + "PYTHONNOUSERSITE"] = "1"
            run_child(
                [*prefix, "fslmerge", "-t", "/work/source_as4d.nii.gz", f"/input/{source_name}"],
                env=container_env,
                direct=True,
            )
            run_child(
                [
                    *prefix,
                    "/opt/HCP-Pipelines/global/scripts/GradientDistortionUnwarp.sh",
                    "--workingdir=/work/hcp",
                    "--coeffs=/input/coefficients.grad",
                    "--in=/work/source_as4d",
                    "--out=/work/corrected",
                    "--owarp=/work/gradient_warp",
                ],
                env=container_env,
                direct=True,
                stream_output=True,
            )
            if not nifti_is_valid(corrected_4d) or not nifti_is_valid(relative_warp):
                raise RuntimeError(
                    f"{GRADIENT_UNWARP_METHOD} did not produce a readable image and warp"
                )
            source_image = nib.load(str(source))
            result_image = nib.load(str(corrected_4d))
            with atomic_output_path(corrected) as staged:
                if len(source_image.shape) == 3 and len(result_image.shape) == 4:
                    data = np.asarray(result_image.dataobj[..., 0], dtype=np.float32)
                    header = source_image.header.copy()
                    header.set_data_dtype(np.float32)
                    nib.save(nib.Nifti1Image(data, source_image.affine, header), str(staged))
                else:
                    shutil.copyfile(corrected_4d, staged)
            with atomic_output_path(warp) as staged:
                shutil.copyfile(relative_warp, staged)
            atomic_write_json(metadata, provenance)
        valid, reason = _validate_outputs(source, corrected, warp, metadata)
        if not valid:
            raise RuntimeError(reason)

    return Step.python(
        name="Gradient Distortion Unwarping",
        inputs=(source, resolution.coefficients, image),
        outputs=(corrected, warp, metadata),
        action=unwarp,
        force=force,
        validate=lambda: _validate_outputs(source, corrected, warp, metadata),
        parameters=provenance,
    )
