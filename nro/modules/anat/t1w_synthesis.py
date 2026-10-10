"""Construct a synthetic T1w reference when only T2w anatomy is available."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from nro.orchestration.runner_graph import Step

from .policy import t1w_support_mask_contract, t1w_synthesis_contract


def create_t1w_synthesis_step(
    *,
    run_child: Callable[..., Any],
    runtime: str,
    image: Path,
    license_file: Path,
    t2w: Path,
    output: Path,
    threads: int,
    env: Mapping[str, str],
    force: bool,
) -> Step:
    """Create a CPU SynthSR step with explicit input and software provenance."""

    def action() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        work = output.parent.resolve()
        source = t2w.resolve()
        if not source.is_relative_to(work):
            raise ValueError("T1w synthesis input and output must share a private work directory")
        command = [
            runtime,
            "exec",
            "--cleanenv",
            "--bind",
            f"{work}:/work,{license_file}:/license.txt:ro",
            str(image),
            "bash",
            "-lc",
            (
                "export FREESURFER_HOME=/usr/local/freesurfer; "
                "export FS_LICENSE=/license.txt; "
                'source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null; exec "$@"'
            ),
            "bash",
            "mri_synthsr",
            "--i",
            f"/work/{source.relative_to(work)}",
            "--o",
            f"/work/{output.resolve().relative_to(work)}",
            "--threads",
            str(max(1, threads)),
            "--cpu",
        ]
        run_child(command, direct=True, env=dict(env), discard_stdout=True)

    return Step.python(
        name="Synthesize T1w Reference from T2w",
        inputs=(t2w, image, license_file),
        outputs=(output,),
        action=action,
        force=force,
        parameters=t1w_synthesis_contract(),
    )


def create_t1w_support_mask_step(
    *,
    t2w: Path,
    synthetic_t1w: Path,
    output: Path,
    force: bool,
) -> Step:
    """Map the selected T2w brain support onto the synthetic T1w grid."""

    def action() -> None:
        source = nib.load(str(t2w))
        reference = nib.load(str(synthetic_t1w))
        if len(source.shape) != 3 or len(reference.shape) != 3:
            raise ValueError("Synthetic T1w support requires two 3D anatomical images")
        source_data = np.asarray(source.dataobj)
        support = np.isfinite(source_data) & (source_data != 0)
        if not np.any(support):
            raise ValueError(f"Selected T2w image has no finite nonzero brain support: {t2w}")

        source_mask = nib.Nifti1Image(support.astype(np.uint8), source.affine)
        resampled = resample_from_to(
            source_mask,
            (reference.shape, reference.affine),
            order=0,
            mode="constant",
            cval=0,
        )
        values = (np.asarray(resampled.dataobj) > 0.5).astype(np.uint8)
        if not np.any(values):
            raise ValueError("Selected T2w support does not overlap the synthetic T1w grid")

        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(values, reference.affine, header=header)
        image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
        image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".partial-{output.name}")
        temporary.unlink(missing_ok=True)
        nib.save(image, str(temporary))
        os.replace(temporary, output)

    return Step.python(
        name="Map T2w Brain Mask to Synthetic T1w",
        inputs=(t2w, synthetic_t1w),
        outputs=(output,),
        action=action,
        force=force,
        parameters=t1w_support_mask_contract(),
    )
