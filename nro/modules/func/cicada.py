"""Adapt nro functional intermediates to the external CICADA classifier."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from nro.engine.io import atomic_write_text

from .confounds import _dvars_metrics, _fd_power, _infer_mcflirt_order

RunCommand = Callable[[Sequence[str]], None]


def _component_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name.removesuffix(".nii.gz"))
    if match is None:
        raise ValueError(f"Cannot determine component number from {path}")
    return int(match.group(1))


def write_motion_metrics(
    *,
    bold: Path,
    mask: Path,
    motion_parameters: Path,
    output: Path,
    fd_radius_mm: float,
    dvars_statistical_alpha: float,
    dvars_practical_threshold_percent: float,
    dvars_power: float,
) -> None:
    """Write the pre-denoising FD and DVARS columns required by CICADA."""
    import nibabel as nib  # type: ignore
    import pandas as pd  # type: ignore

    image = nib.load(str(bold))
    if len(image.shape) != 4:
        raise ValueError(f"CICADA BOLD input must be 4D, got {image.shape}")
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    mask_image = nib.load(str(mask))
    if mask_image.shape[:3] != image.shape[:3]:
        raise ValueError(
            f"CICADA mask shape {mask_image.shape[:3]} does not match BOLD {image.shape[:3]}"
        )
    brain_mask = np.asanyarray(mask_image.dataobj, dtype=np.float32) > 0
    motion = np.loadtxt(motion_parameters, dtype=np.float64)
    if motion.ndim == 1:
        motion = motion.reshape(1, -1)
    motion = _infer_mcflirt_order(motion)
    if motion.shape[0] != image.shape[3]:
        raise ValueError(
            "CICADA motion rows do not match BOLD timepoints: "
            f"{motion.shape[0]} != {image.shape[3]}"
        )
    dvars = _dvars_metrics(
        data,
        brain_mask,
        statistical_alpha=dvars_statistical_alpha,
        practical_threshold_percent=dvars_practical_threshold_percent,
        power=dvars_power,
    )
    table = pd.DataFrame(
        {
            "framewise_displacement": _fd_power(motion, radius_mm=fd_radius_mm),
            "dvars": np.asarray(dvars["dvars"], dtype=np.float64),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False, na_rep="n/a")


def prepare_melodic_adapter(
    *,
    run_command: RunCommand,
    source_directory: Path,
    output_directory: Path,
    reference: Path,
    warp: Path,
    premat: Path,
) -> None:
    """Create a CICADA-readable MNI copy of an nro MELODIC decomposition."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for name in ("melodic_mix", "melodic_FTmix", "melodic_ICstats"):
        source = source_directory / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_directory / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

    probability_maps = sorted(
        (source_directory / "stats").glob("probmap_*.nii.gz"), key=_component_number
    )
    if not probability_maps:
        raise RuntimeError(f"MELODIC produced no component probability maps in {source_directory}")
    native_probabilities = output_directory / "ICprobabilities_native.nii.gz"
    probabilities = output_directory / "ICprobabilities.nii.gz"
    run_command(
        ["fslmerge", "-t", str(native_probabilities), *(str(path) for path in probability_maps)]
    )
    run_command(
        [
            "applywarp",
            f"--ref={reference}",
            f"--in={native_probabilities}",
            f"--out={probabilities}",
            f"--warp={warp}",
            f"--premat={premat}",
            "--interp=trilinear",
        ]
    )
    native_probabilities.unlink(missing_ok=True)
    run_command(
        [
            "applywarp",
            f"--ref={reference}",
            f"--in={source_directory / 'melodic_IC.nii.gz'}",
            f"--out={output_directory / 'melodic_IC.nii.gz'}",
            f"--warp={warp}",
            f"--premat={premat}",
            "--interp=trilinear",
        ]
    )


def parse_component_indices(path: Path) -> tuple[int, ...]:
    """Read and validate CICADA's one-based component index list."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    try:
        values = tuple(
            int(value.strip()) for value in text.replace("\n", ",").split(",") if value.strip()
        )
    except ValueError as error:
        raise ValueError(f"Invalid CICADA component list in {path}") from error
    if any(value < 1 for value in values) or len(values) != len(set(values)):
        raise ValueError(f"CICADA component indices must be distinct and one-based: {path}")
    return values


def write_result_manifest(
    *,
    output: Path,
    executable: Path,
    classification_directory: Path,
    tolerance: int,
    smoothing_retention_mode: str,
    mixing_matrix: Path,
) -> None:
    """Record the compact classifier result consumed by later func steps."""
    noise_path = classification_directory / "noise_components.txt"
    signal_path = classification_directory / "signal_components.txt"
    noise = parse_component_indices(noise_path)
    signal = parse_component_indices(signal_path)
    overlap = set(noise) & set(signal)
    if overlap:
        raise ValueError(f"CICADA labeled components as both signal and noise: {sorted(overlap)}")
    provenance_path = classification_directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload = {
        "classifier": "cicada",
        "executable": str(executable),
        "tolerance": int(tolerance),
        "smoothing_retention_mode": smoothing_retention_mode,
        "component_indexing": "one-based",
        "noise_components": list(noise),
        "signal_components": list(signal),
        "warnings": list(provenance.get("warnings", [])),
        "mixing_matrix": str(mixing_matrix),
        "external_provenance": str(provenance_path),
    }
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def reset_directory(path: Path) -> None:
    """Replace a private adapter directory without following symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
