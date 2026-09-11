"""Prepare, diagnose, and compact outputs from the official MARSS implementation."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from nro.engine.images import nifti_is_valid
from nro.engine.io import atomic_output_path, write_json
from nro.modules.func.contracts import MARSS_DIAGNOSTIC_METHOD
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

MARSS_PACKAGE_VERSION = "1.0.2"
MARSS_METHOD = "Multiband Artifact Regression in Simultaneous Slices"
MARSS_CITATION = "Tubiolo et al., Human Brain Mapping (2024), doi:10.1002/hbm.70066"


@dataclass(frozen=True)
class SliceGrouping:
    """Describe simultaneous slices in the NIfTI array coordinate system."""

    axis: int
    multiband_factor: int
    groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MarssOutputs:
    """Name the fixed outputs of one native-space MARSS stage."""

    bold: Path
    metadata: Path
    loadings: Path
    timecourses: Path
    mean_absolute_artifact: Path
    slice_score_map: Path
    correlations_before: Path
    correlations_after: Path
    heatmap: Path


def create_marss_motion_step(
    *,
    run_child: Callable[..., str | None],
    source_bold: Path,
    work_dir: Path,
    env: Mapping[str, str],
    force: bool,
) -> tuple[Step, Path]:
    """Estimate raw-space motion parameters for MARSS without retaining a resampled series."""
    reference = work_dir / "first_volume.nii.gz"
    temporary_corrected = work_dir / "motion_estimation.nii.gz"
    parameters = work_dir / "motion.par"

    def estimate() -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        run_child(["fslroi", str(source_bold), str(reference), "0", "1"], env=dict(env))
        run_child(
            [
                "mcflirt",
                "-in",
                str(source_bold),
                "-out",
                str(temporary_corrected),
                "-reffile",
                str(reference),
                "-plots",
            ],
            env=dict(env),
            cwd=work_dir,
        )
        generated = work_dir / f"{temporary_corrected.name}.par"
        if not generated.is_file() or generated.stat().st_size == 0:
            raise RuntimeError(f"MCFLIRT did not produce MARSS motion parameters: {generated}")
        os.replace(generated, parameters)
        temporary_corrected.unlink(missing_ok=True)

    def validate() -> tuple[bool, str]:
        if not parameters.is_file() or parameters.stat().st_size == 0:
            return False, "MARSS motion parameters are missing."
        return True, "MARSS motion parameters are available."

    return (
        Step.python(
            name="Estimate Native Motion for MARSS",
            inputs=(source_bold,),
            outputs=(reference, parameters),
            action=estimate,
            validate=validate,
            force=force,
        ),
        parameters,
    )


def derive_slice_grouping(
    metadata: Mapping[str, Any], spatial_shape: Sequence[int]
) -> SliceGrouping:
    """Derive simultaneous-slice groups and reject layouts unsupported by MARSS 1.0.2."""
    if len(spatial_shape) != 3:
        raise ValueError(f"Expected three spatial dimensions, got {tuple(spatial_shape)!r}")
    direction = str(metadata.get("SliceEncodingDirection", "k")).strip()
    axis_names = {"i": 0, "j": 1, "k": 2}
    axis = axis_names.get(direction.removesuffix("-"))
    if axis is None:
        raise ValueError(
            "SliceEncodingDirection must be i, j, k, or the corresponding negative direction"
        )
    if axis != 2:
        raise ValueError(
            "MARSS 1.0.2 requires the acquired slices on NIfTI axis k; "
            f"this run declares SliceEncodingDirection={direction!r}"
        )

    raw_factor = metadata.get("MultibandAccelerationFactor")
    if isinstance(raw_factor, bool):
        raise ValueError("MultibandAccelerationFactor must be an integer")
    try:
        factor = int(raw_factor)
    except (TypeError, ValueError) as error:
        raise ValueError("MultibandAccelerationFactor is required for MARSS") from error
    if factor < 2 or float(raw_factor) != factor:
        raise ValueError(
            "MARSS diagnostics require an integer MultibandAccelerationFactor of at least 2"
        )

    slice_count = int(spatial_shape[axis])
    if slice_count % factor:
        raise ValueError(
            f"{slice_count} slices are not divisible by MultibandAccelerationFactor={factor}"
        )
    raw_timing = metadata.get("SliceTiming")
    if not isinstance(raw_timing, list) or len(raw_timing) != slice_count:
        raise ValueError(
            f"SliceTiming must contain one value for each of the {slice_count} acquired slices"
        )
    try:
        timing = np.asarray(raw_timing, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("SliceTiming must contain finite numeric values") from error
    if not np.all(np.isfinite(timing)):
        raise ValueError("SliceTiming must contain finite numeric values")

    tolerance = max(1e-7, float(np.ptp(timing)) * 1e-7)
    groups: list[list[int]] = []
    group_times: list[float] = []
    for index, acquisition_time in enumerate(timing):
        match = next(
            (
                group_index
                for group_index, existing in enumerate(group_times)
                if abs(acquisition_time - existing) <= tolerance
            ),
            None,
        )
        if match is None:
            group_times.append(float(acquisition_time))
            groups.append([index])
        else:
            groups[match].append(index)
    if any(len(group) != factor for group in groups):
        sizes = sorted(len(group) for group in groups)
        raise ValueError(
            "SliceTiming does not define equal simultaneous groups matching the multiband "
            f"factor; observed group sizes: {sizes}"
        )

    # The official implementation accepts only the regular modulo layout. The
    # timing values can appear in any acquisition order, but each equal-time
    # group must contain slices separated by slice_count / factor.
    stride = slice_count // factor
    expected = {tuple(range(offset, slice_count, stride)) for offset in range(stride)}
    observed = {tuple(group) for group in groups}
    if observed != expected:
        raise ValueError(
            "SliceTiming defines simultaneous groups that MARSS 1.0.2 cannot represent"
        )
    ordered = tuple(sorted(observed, key=lambda group: group[0]))
    return SliceGrouping(axis=axis, multiband_factor=factor, groups=ordered)


def _load_motion(path: Path, volumes: int) -> np.ndarray:
    motion = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if motion.shape != (volumes, 6) or not np.all(np.isfinite(motion)):
        raise ValueError(
            f"MARSS motion parameters must have shape ({volumes}, 6), got {motion.shape}"
        )
    return motion


def _slice_means(
    path: Path, *, chunk_volumes: int
) -> tuple[np.ndarray, tuple[int, ...], dict[str, float]]:
    import nibabel as nib

    image = nib.load(str(path))
    if len(image.shape) != 4:
        raise ValueError(f"MARSS requires a 4D NIfTI image: {path}")
    nx, ny, nz, nt = (int(value) for value in image.shape)
    if nt < 3:
        raise ValueError(f"MARSS diagnostics require at least three volumes: {path}")
    means = np.empty((nt, nz), dtype=np.float64)
    signal_sum = np.zeros((nx, ny, nz), dtype=np.float64)
    signal_square_sum = np.zeros((nx, ny, nz), dtype=np.float64)
    for start in range(0, nt, chunk_volumes):
        stop = min(nt, start + chunk_volumes)
        block = np.asarray(image.dataobj[:, :, :, start:stop], dtype=np.float32)
        means[start:stop] = np.mean(block, axis=(0, 1), dtype=np.float64).T
        signal_sum += np.sum(block, axis=3, dtype=np.float64)
        signal_square_sum += np.einsum("...t,...t->...", block, block, dtype=np.float64)
    mean = signal_sum / nt
    variance = np.maximum((signal_square_sum - signal_sum * mean) / max(1, nt - 1), 0.0)
    valid = np.isfinite(mean) & np.isfinite(variance) & (variance > np.finfo(np.float32).eps)
    positive = valid & (mean > 0)
    tsnr = mean[positive] / np.sqrt(variance[positive])
    summary = {
        "mean_voxel_temporal_variance": float(np.mean(variance[valid])) if np.any(valid) else 0.0,
        "mean_voxel_tsnr": float(np.mean(tsnr)) if tsnr.size else 0.0,
        "median_voxel_tsnr": float(np.median(tsnr)) if tsnr.size else 0.0,
        "finite_nonconstant_voxel_fraction": float(np.mean(valid)),
    }
    return means, (nx, ny, nz, nt), summary


def _motion_residuals(slice_means: np.ndarray, motion: np.ndarray) -> np.ndarray:
    derivative = np.vstack((np.zeros((1, 6)), np.diff(motion, axis=0)))
    design = np.column_stack(
        (
            np.ones(slice_means.shape[0]),
            np.linspace(-1.0, 1.0, slice_means.shape[0]),
            motion,
            motion**2,
            derivative,
            derivative**2,
        )
    )
    return slice_means - design @ np.linalg.lstsq(design, slice_means, rcond=None)[0]


def _slice_correlation_summary(
    fisher: np.ndarray, grouping: SliceGrouping
) -> tuple[float, float, list[float]]:
    """Average simultaneous and comparison correlations with equal weight per slice."""
    group_by_slice = {slice_index: group for group in grouping.groups for slice_index in group}
    within_by_slice: list[float] = []
    adjacent_by_slice: list[float] = []
    per_slice: list[float] = []
    slice_count = fisher.shape[0]
    for slice_index in range(slice_count):
        peers = [value for value in group_by_slice[slice_index] if value != slice_index]
        neighbors = sorted(
            {
                neighbor
                for peer in peers
                for neighbor in (peer - 1, peer + 1)
                if 0 <= neighbor < slice_count and neighbor != slice_index
            }
        )
        within_z = float(np.mean(fisher[slice_index, peers]))
        adjacent_z = float(np.mean(fisher[slice_index, neighbors]))
        within_by_slice.append(within_z)
        adjacent_by_slice.append(adjacent_z)
        per_slice.append(float(np.tanh(within_z - adjacent_z)))
    return float(np.mean(within_by_slice)), float(np.mean(adjacent_by_slice)), per_slice


def _apply_correction(mode: str, multiband_factor: int, minimum_factor: int) -> bool:
    """Apply the published correction rule independently of the diagnostic score."""
    return mode == "auto" and multiband_factor >= minimum_factor


def slice_correlation_diagnostics(
    path: Path,
    motion_path: Path,
    grouping: SliceGrouping,
    *,
    chunk_volumes: int,
) -> dict[str, Any]:
    """Measure excess correlation among simultaneous slices after motion residualization."""
    slice_means, shape, temporal_summary = _slice_means(path, chunk_volumes=chunk_volumes)
    motion = _load_motion(motion_path, shape[3])
    residuals = _motion_residuals(slice_means, motion)
    correlation = np.corrcoef(residuals, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(correlation, -1 + 1e-7, 1 - 1e-7)
    fisher = np.arctanh(clipped)

    within_z, adjacent_z, per_slice = _slice_correlation_summary(fisher, grouping)
    delta_z = within_z - adjacent_z
    return {
        "shape": list(shape),
        "correlation": correlation,
        "per_slice_delta_r": per_slice,
        "mean_simultaneous_r": float(np.tanh(within_z)),
        "mean_adjacent_to_simultaneous_r": float(np.tanh(adjacent_z)),
        "simultaneous_minus_adjacent_delta_r": float(np.tanh(delta_z)),
        **temporal_summary,
    }


def _write_correlation_table(path: Path, correlation: np.ndarray) -> None:
    with atomic_output_path(path) as staged:
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["slice", *(f"slice_{index}" for index in range(correlation.shape[1]))])
            for index, row in enumerate(correlation):
                writer.writerow([index, *(f"{float(value):.9g}" for value in row)])


def _write_heatmap(path: Path, before: np.ndarray, after: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with atomic_output_path(path) as staged:
        figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        image = None
        for axis, matrix, title in zip(
            axes,
            (before, after),
            ("Source BOLD", "Selected native BOLD"),
        ):
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
            axis.set_title(title)
            axis.set_xlabel("Slice")
            axis.set_ylabel("Slice")
        assert image is not None
        figure.colorbar(image, ax=axes, label="Pearson correlation")
        figure.savefig(staged, dpi=140)
        plt.close(figure)


def _atomic_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(source.resolve())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _save_zero_artifact(
    source: Path, loadings: Path, timecourses: Path, mean_absolute: Path
) -> None:
    import nibabel as nib

    image = nib.load(str(source))
    spatial = np.zeros(image.shape[:3], dtype=np.float32)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    for destination in (loadings, mean_absolute):
        with atomic_output_path(destination) as staged:
            nib.save(nib.Nifti1Image(spatial, image.affine, header), str(staged))
    with atomic_output_path(timecourses) as staged:
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow([f"slice_{index}" for index in range(image.shape[2])])
            row = ["0"] * image.shape[2]
            for _ in range(image.shape[3]):
                writer.writerow(row)


def _save_slice_score_map(source: Path, scores: Sequence[float], destination: Path) -> None:
    import nibabel as nib

    image = nib.load(str(source))
    if len(scores) != image.shape[2]:
        raise ValueError("Per-slice MARSS scores do not match the BOLD slice count")
    data = np.broadcast_to(
        np.asarray(scores, dtype=np.float32)[None, None, :], image.shape[:3]
    ).copy()
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    with atomic_output_path(destination) as staged:
        nib.save(nib.Nifti1Image(data, image.affine, header), str(staged))


def _compact_artifact(
    source_bold: Path,
    corrected_bold: Path,
    artifact: Path,
    output_bold: Path,
    loadings_path: Path,
    timecourses_path: Path,
    mean_absolute_path: Path,
) -> tuple[float, float, float]:
    import nibabel as nib

    source_image = nib.load(str(source_bold))
    corrected_image = nib.load(str(corrected_bold))
    artifact_image = nib.load(str(artifact))
    if corrected_image.shape != source_image.shape or artifact_image.shape != source_image.shape:
        raise RuntimeError("MARSS outputs do not match the source BOLD dimensions")

    spatial_shape = tuple(int(value) for value in source_image.shape[:3])
    volumes = int(source_image.shape[3])
    loadings = np.zeros(spatial_shape, dtype=np.float32)
    mean_absolute = np.zeros(spatial_shape, dtype=np.float32)
    timecourses = np.zeros((volumes, spatial_shape[2]), dtype=np.float32)
    maximum_relative_error = 0.0
    maximum_correction_error = 0.0
    artifact_variance_sum = 0.0
    for slice_index in range(spatial_shape[2]):
        matrix = np.asarray(artifact_image.dataobj[:, :, slice_index, :], dtype=np.float32).reshape(
            -1, volumes
        )
        energies = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
        pivot = int(np.argmax(energies))
        norm = math.sqrt(float(energies[pivot]))
        if norm > 0:
            temporal = matrix[pivot] / norm
            spatial = matrix @ temporal
            reconstructed = spatial[:, None] * temporal[None, :]
            denominator = max(float(np.max(np.abs(matrix))), np.finfo(np.float32).eps)
            error = float(np.max(np.abs(matrix - reconstructed))) / denominator
            maximum_relative_error = max(maximum_relative_error, error)
            loadings[:, :, slice_index] = spatial.reshape(spatial_shape[:2])
            timecourses[:, slice_index] = temporal
        mean_absolute[:, :, slice_index] = np.mean(np.abs(matrix), axis=1).reshape(
            spatial_shape[:2]
        )
        artifact_variance_sum += float(np.sum(np.var(matrix, axis=1, ddof=1), dtype=np.float64))
        source_slice = np.asarray(
            source_image.dataobj[:, :, slice_index, :], dtype=np.float32
        ).reshape(-1, volumes)
        corrected_slice = np.asarray(
            corrected_image.dataobj[:, :, slice_index, :], dtype=np.float32
        ).reshape(-1, volumes)
        denominator = max(float(np.max(np.abs(matrix))), np.finfo(np.float32).eps)
        correction_error = float(np.max(np.abs(source_slice - corrected_slice - matrix)))
        maximum_correction_error = max(
            maximum_correction_error,
            correction_error / denominator,
        )
    if maximum_relative_error > 5e-5:
        raise RuntimeError(
            "The MARSS artifact is not slice-wise rank one within tolerance "
            f"(maximum relative error {maximum_relative_error:.3g})"
        )
    if maximum_correction_error > 5e-5:
        raise RuntimeError(
            "The corrected MARSS series does not equal source minus artifact within tolerance "
            f"(maximum relative error {maximum_correction_error:.3g})"
        )

    header = source_image.header.copy()
    header.set_data_dtype(np.float32)
    with atomic_output_path(output_bold) as staged:
        nib.save(
            nib.Nifti1Image(corrected_image.dataobj, source_image.affine, header),
            str(staged),
        )
    for array, destination in (
        (loadings, loadings_path),
        (mean_absolute, mean_absolute_path),
    ):
        with atomic_output_path(destination) as staged:
            nib.save(nib.Nifti1Image(array, source_image.affine, header), str(staged))
    with atomic_output_path(timecourses_path) as staged:
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow([f"slice_{index}" for index in range(spatial_shape[2])])
            writer.writerows(timecourses)
    mean_artifact_variance = artifact_variance_sum / int(np.prod(spatial_shape))
    return maximum_relative_error, maximum_correction_error, mean_artifact_variance


def _maximum_artifact_motion_correlation(timecourses_path: Path, motion_path: Path) -> float:
    """Return the largest absolute correlation between artifact and motion timecourses."""
    artifact = np.loadtxt(timecourses_path, delimiter="\t", skiprows=1, ndmin=2)
    motion = _load_motion(motion_path, artifact.shape[0])
    motion = np.column_stack((motion, np.vstack((np.zeros((1, 6)), np.diff(motion, axis=0)))))
    artifact = artifact - np.mean(artifact, axis=0)
    motion = motion - np.mean(motion, axis=0)
    denominator = np.sqrt(
        np.sum(artifact * artifact, axis=0)[:, None] * np.sum(motion * motion, axis=0)[None, :]
    )
    valid = denominator > np.finfo(np.float64).eps
    if not np.any(valid):
        return 0.0
    correlation = np.zeros_like(denominator)
    numerator = artifact.T @ motion
    correlation[valid] = numerator[valid] / denominator[valid]
    return float(np.max(np.abs(correlation)))


def _package_version() -> str:
    try:
        return version("MARSS")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "MARSS correction requires its Python dependency. Run ./install from this "
            "checkout, adding --maintain for a shared installation."
        ) from error


def create_marss_step(
    *,
    runner: Runner,
    source_bold: Path,
    metadata: Mapping[str, Any],
    metadata_sources: Sequence[Path],
    motion_parameters: Path,
    work_dir: Path,
    artifact_dir: Path,
    run_stem: str,
    mode: str,
    min_multiband_factor: int,
    chunk_volumes: int,
    force: bool,
) -> tuple[Step, MarssOutputs]:
    """Create one fixed-output native MARSS correction or passthrough step."""
    outputs = MarssOutputs(
        bold=work_dir / f"{run_stem}_desc-marss_bold.nii.gz",
        metadata=artifact_dir / f"{run_stem}_desc-marss_qc.json",
        loadings=artifact_dir / f"{run_stem}_desc-marssArtifact_loadings.nii.gz",
        timecourses=artifact_dir / f"{run_stem}_desc-marssArtifact_timeseries.tsv",
        mean_absolute_artifact=artifact_dir / f"{run_stem}_desc-marssArtifactMean_statmap.nii.gz",
        slice_score_map=artifact_dir / f"{run_stem}_desc-marssSliceScore_statmap.nii.gz",
        correlations_before=artifact_dir / f"{run_stem}_desc-marssBefore_correlation.tsv",
        correlations_after=artifact_dir / f"{run_stem}_desc-marssAfter_correlation.tsv",
        heatmap=artifact_dir / f"{run_stem}_desc-marssCorrelation_qc.png",
    )

    def run_stage() -> None:
        import nibabel as nib

        image = nib.load(str(source_bold))
        try:
            grouping = derive_slice_grouping(metadata, image.shape[:3])
        except ValueError as error:
            slice_count = int(image.shape[2])
            correlation = np.full((slice_count, slice_count), np.nan, dtype=np.float64)
            _atomic_symlink(source_bold, outputs.bold)
            _save_zero_artifact(
                source_bold,
                outputs.loadings,
                outputs.timecourses,
                outputs.mean_absolute_artifact,
            )
            _save_slice_score_map(
                source_bold,
                [float("nan")] * slice_count,
                outputs.slice_score_map,
            )
            _write_correlation_table(outputs.correlations_before, correlation)
            _write_correlation_table(outputs.correlations_after, correlation)
            _write_heatmap(outputs.heatmap, correlation, correlation)
            write_json(
                outputs.metadata,
                {
                    "Method": MARSS_METHOD,
                    "Citation": MARSS_CITATION,
                    "DiagnosticMethod": MARSS_DIAGNOSTIC_METHOD,
                    "DiagnosticAvailable": False,
                    "DiagnosticUnavailableReason": str(error),
                    "Mode": mode,
                    "Applied": False,
                    "Decision": "diagnostic_unavailable",
                    "MinimumCorrectionMultibandFactor": (
                        min_multiband_factor if mode == "auto" else None
                    ),
                    "OfficialPackageVersion": None,
                    "SliceEncodingAxis": None,
                    "MultibandAccelerationFactor": metadata.get("MultibandAccelerationFactor"),
                    "SimultaneousSliceGroups": [],
                    "Before": None,
                    "After": None,
                    "QualityChange": None,
                    "CompactArtifact": {
                        "Available": False,
                        "Representation": "slice_wise_rank_one",
                        "Reconstruction": ("artifact[v,t] = loading[v] * timecourse[slice(v),t]"),
                        "Scaling": "unit_l2_norm_timecourse_with_positive_pivot_loading",
                        "Loadings": str(outputs.loadings),
                        "Timecourses": str(outputs.timecourses),
                        "MeanAbsoluteMap": str(outputs.mean_absolute_artifact),
                        "SliceScoreMap": str(outputs.slice_score_map),
                        "MaximumRelativeReconstructionError": None,
                        "MaximumRelativeCorrectionError": None,
                    },
                },
            )
            return
        before = slice_correlation_diagnostics(
            source_bold,
            motion_parameters,
            grouping,
            chunk_volumes=chunk_volumes,
        )
        _save_slice_score_map(source_bold, before["per_slice_delta_r"], outputs.slice_score_map)
        correction_eligible = grouping.multiband_factor >= min_multiband_factor
        apply = _apply_correction(mode, grouping.multiband_factor, min_multiband_factor)
        package_version: str | None = None
        factorization_error: float | None = None
        correction_error: float | None = None
        mean_artifact_variance: float | None = None
        maximum_motion_correlation: float | None = None
        if apply:
            package_version = _package_version()
            if package_version != MARSS_PACKAGE_VERSION:
                raise RuntimeError(
                    f"MARSS {MARSS_PACKAGE_VERSION} is required; found {package_version}"
                )
            if importlib.util.find_spec("MARSS") is None:
                raise RuntimeError("The installed MARSS package cannot be imported")
            work_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=work_dir, prefix="official-") as temporary:
                temporary_path = Path(temporary)
                corrected = temporary_path / "corrected.nii.gz"
                artifact = temporary_path / "artifact.nii.gz"
                runner.run_direct(
                    (
                        sys.executable,
                        "-m",
                        "nro.modules.func.marss_worker",
                        "--bold",
                        str(source_bold),
                        "--motion",
                        str(motion_parameters),
                        "--multiband-factor",
                        str(grouping.multiband_factor),
                        "--work-dir",
                        str(temporary_path),
                        "--corrected",
                        str(corrected),
                        "--artifact",
                        str(artifact),
                    ),
                    step_name="Official MARSS Correction",
                    outputs=(corrected, artifact),
                    env={
                        **os.environ,
                        "MPLBACKEND": "Agg",
                        "MPLCONFIGDIR": str(temporary_path / "matplotlib"),
                    },
                )
                factorization_error, correction_error, mean_artifact_variance = _compact_artifact(
                    source_bold,
                    corrected,
                    artifact,
                    outputs.bold,
                    outputs.loadings,
                    outputs.timecourses,
                    outputs.mean_absolute_artifact,
                )
                maximum_motion_correlation = _maximum_artifact_motion_correlation(
                    outputs.timecourses, motion_parameters
                )
        else:
            _atomic_symlink(source_bold, outputs.bold)
            _save_zero_artifact(
                source_bold,
                outputs.loadings,
                outputs.timecourses,
                outputs.mean_absolute_artifact,
            )
        after = (
            slice_correlation_diagnostics(
                outputs.bold,
                motion_parameters,
                grouping,
                chunk_volumes=chunk_volumes,
            )
            if apply
            else deepcopy(before)
        )
        before_correlation = before.pop("correlation")
        after_correlation = after.pop("correlation")
        before_variance = float(before["mean_voxel_temporal_variance"])
        variance_retained = (
            float(after["mean_voxel_temporal_variance"]) / before_variance
            if before_variance > 0
            else None
        )
        adjacent_change = float(after["mean_adjacent_to_simultaneous_r"]) - float(
            before["mean_adjacent_to_simultaneous_r"]
        )
        overcorrection_reasons = []
        if apply and variance_retained is not None and variance_retained < 0.5:
            overcorrection_reasons.append("less_than_half_of_mean_temporal_variance_retained")
        if apply and adjacent_change < -0.1:
            overcorrection_reasons.append("broad_adjacent_slice_correlation_reduction")
        _write_correlation_table(outputs.correlations_before, before_correlation)
        _write_correlation_table(outputs.correlations_after, after_correlation)
        _write_heatmap(outputs.heatmap, before_correlation, after_correlation)
        write_json(
            outputs.metadata,
            {
                "Method": MARSS_METHOD,
                "Citation": MARSS_CITATION,
                "DiagnosticMethod": MARSS_DIAGNOSTIC_METHOD,
                "DiagnosticAvailable": True,
                "DiagnosticUnavailableReason": None,
                "Mode": mode,
                "Applied": apply,
                "Decision": (
                    "diagnosis_only"
                    if mode == "diagnose"
                    else "multiband_factor_below_recommended_minimum"
                    if not correction_eligible
                    else "multiband_factor_at_or_above_recommended_minimum"
                ),
                "MinimumCorrectionMultibandFactor": (
                    min_multiband_factor if mode == "auto" else None
                ),
                "OfficialPackageVersion": package_version,
                "SliceEncodingAxis": grouping.axis,
                "MultibandAccelerationFactor": grouping.multiband_factor,
                "SimultaneousSliceGroups": [list(group) for group in grouping.groups],
                "Before": before,
                "After": after,
                "QualityChange": {
                    "VarianceRetainedFraction": variance_retained,
                    "VarianceReductionFraction": (
                        1.0 - variance_retained if variance_retained is not None else None
                    ),
                    "ArtifactToSourceMeanVarianceRatio": (
                        mean_artifact_variance / before_variance
                        if mean_artifact_variance is not None and before_variance > 0
                        else None
                    ),
                    "MaximumAbsoluteArtifactMotionCorrelation": maximum_motion_correlation,
                    "MeanVoxelTSNRChange": float(after["mean_voxel_tsnr"])
                    - float(before["mean_voxel_tsnr"]),
                    "AdjacentSliceCorrelationChange": adjacent_change,
                    "PossibleOvercorrection": bool(overcorrection_reasons),
                    "PossibleOvercorrectionReasons": overcorrection_reasons,
                    "HeuristicThresholds": {
                        "MinimumVarianceRetainedFraction": 0.5,
                        "MinimumAdjacentSliceCorrelationChange": -0.1,
                    },
                },
                "CompactArtifact": {
                    "Available": apply,
                    "Representation": "slice_wise_rank_one",
                    "Reconstruction": "artifact[v,t] = loading[v] * timecourse[slice(v),t]",
                    "Scaling": "unit_l2_norm_timecourse_with_positive_pivot_loading",
                    "Loadings": str(outputs.loadings),
                    "Timecourses": str(outputs.timecourses),
                    "MeanAbsoluteMap": str(outputs.mean_absolute_artifact),
                    "SliceScoreMap": str(outputs.slice_score_map),
                    "MaximumRelativeReconstructionError": factorization_error,
                    "MaximumRelativeCorrectionError": correction_error,
                },
            },
        )

    def validate() -> tuple[bool, str]:
        required = tuple(Path(value) for value in vars(outputs).values())
        missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            return False, "MARSS stage is missing outputs: " + ", ".join(missing)
        if not nifti_is_valid(outputs.bold):
            return False, f"MARSS BOLD output is unreadable: {outputs.bold}"
        try:
            payload = json.loads(outputs.metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False, "MARSS metadata is unreadable."
        if payload.get("Mode") != mode:
            return False, "MARSS metadata was produced for a different mode."
        return True, "Native MARSS stage is complete."

    step = Step.python(
        name="Diagnose and Correct Simultaneous-Slice Artifact",
        inputs=(source_bold, motion_parameters, *metadata_sources),
        outputs=tuple(Path(value) for value in vars(outputs).values()),
        action=run_stage,
        validate=validate,
        force=force,
    )
    return step, outputs
