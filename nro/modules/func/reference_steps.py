"""Reference steps for functional preprocessing."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from nro.configuration.runtime import SETTINGS
from nro.engine.bids import (
    bids_readout_time,
)
from nro.engine.images import (
    copy_or_convert_nifti,
)
from nro.engine.io import read_json, write_json
from nro.engine.registration import rigid_transform_metrics
from nro.orchestration.runner import (
    write_completion_breadcrumb,
)
from nro.orchestration.runner_graph import Step

from .constants import (
    _NONSTEADY_DETECTION_POLICY_VERSION,
)
from .sdc_steps import _motion_matrix_breadcrumb


def _detect_initial_nonsteady_volumes(
    image_path: Path,
    *,
    max_vols: int,
    rel_thresh: float,
    stable_run: int,
) -> dict[str, object]:
    """Detect initial non-steady volumes after decompressing the 4D image once."""
    import nibabel as nib  # type: ignore
    import numpy as np

    from nro.modules.func.confounds import _nonsteady_spikes

    image = nib.load(str(image_path))
    total_volumes = int(image.shape[3]) if len(image.shape) > 3 else 1
    # Converting the complete proxy in one operation is important for .nii.gz:
    # indexing the proxy once per volume can restart gzip decompression hundreds
    # of times. float32 preserves the exact dtype used by the previous loop.
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    global_signal = np.zeros(total_volumes, dtype=np.float64)
    for volume_index in range(total_volumes):
        volume = data[..., volume_index]
        support = np.isfinite(volume) & (volume != 0)
        global_signal[volume_index] = float(np.median(volume[support])) if support.any() else 0.0
    nonsteady = set(
        _nonsteady_spikes(
            global_signal,
            max_vols=int(max_vols),
            rel_thresh=float(rel_thresh),
            stable_run=int(stable_run),
        )
    )
    dropped_initial_volumes = 0
    while dropped_initial_volumes in nonsteady:
        dropped_initial_volumes += 1
    if dropped_initial_volumes >= total_volumes:
        dropped_initial_volumes = 0
    return {
        "PolicyVersion": _NONSTEADY_DETECTION_POLICY_VERSION,
        "Input": str(image_path),
        "TotalBOLDVolumes": total_volumes,
        "InitialNonSteadyStateVolumesExcluded": dropped_initial_volumes,
        "DetectionParameters": {
            "MaxInitialVolumes": int(max_vols),
            "RelativeThreshold": float(rel_thresh),
            "StableRunLength": int(stable_run),
        },
        "PerVolumeNonzeroMedianSignal": global_signal.tolist(),
    }


@dataclass(frozen=True)
class _RobustBoldReferenceStep:
    step: Step
    reference: Path
    motion_corrected: Path
    metadata: Path


def _create_robust_bold_reference_step(
    *,
    run_child: Callable[..., Optional[str]],
    epi_in: Path,
    volume_count: int,
    run_stem: str,
    mc_dir: Path,
    env: dict[str, str],
    force: bool,
) -> _RobustBoldReferenceStep:
    """Add fixed two-pass motion/reference construction as one directory step."""
    bootstrap_ref = mc_dir / f"{run_stem}_mc_bootstrap_ref.nii.gz"
    provisional_dir = mc_dir / "provisional"
    provisional_mc = provisional_dir / f"{run_stem}_mc_provisional.nii.gz"
    robust_ref = mc_dir / f"{run_stem}_desc-robust_boldref.nii.gz"
    robust_metadata = mc_dir / f"{run_stem}_desc-robust_boldref.json"
    nonsteady_metadata = mc_dir / f"{run_stem}_desc-nonsteadyDetection_boldref.json"
    final_mc = mc_dir / f"{run_stem}_mc.nii.gz"
    final_par = mc_dir / "motion.par"
    final_mats = mc_dir / f"{final_mc.name}.mat"
    final_matrices_complete = _motion_matrix_breadcrumb(final_mats)
    nvols = int(volume_count)
    if nvols < 1:
        raise ValueError("Robust BOLD reference construction requires at least one volume")
    expected_matrices = tuple(final_mats / f"MAT_{index:04d}" for index in range(nvols))
    confound_cfg = SETTINGS.func_confounds
    detection_kwargs = {
        "max_vols": int(confound_cfg.nonsteady_max_vols),
        "rel_thresh": float(confound_cfg.nonsteady_rel_thresh),
        "stable_run": int(confound_cfg.nonsteady_stable_run),
    }

    def run_mcflirt(source: Path, reference: Path, output: Path, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        run_child(
            [
                "mcflirt",
                "-in",
                str(source),
                "-out",
                str(output),
                "-reffile",
                str(reference),
                "-mats",
                "-plots",
            ],
            env=env,
            cwd=directory,
        )
        raw_par = directory / f"{output.name}.par"
        canonical_par = directory / "motion.par"
        if not raw_par.is_file() or raw_par.stat().st_size == 0:
            raise SystemExit(f"MCFLIRT did not produce its motion-parameter file: {raw_par}")
        shutil.move(str(raw_par), str(canonical_par))

    def construct() -> None:
        mc_dir.mkdir(parents=True, exist_ok=True)
        run_child(["fslroi", str(epi_in), str(bootstrap_ref), "0", "1"], env=env)
        run_mcflirt(epi_in, bootstrap_ref, provisional_mc, provisional_dir)
        detection = _detect_initial_nonsteady_volumes(provisional_mc, **detection_kwargs)
        write_json(nonsteady_metadata, detection)
        total_volumes = int(detection["TotalBOLDVolumes"])
        dropped = int(detection["InitialNonSteadyStateVolumesExcluded"])
        median_input = provisional_mc
        if dropped:
            median_input = provisional_dir / f"{run_stem}_mc_provisional_steady.nii.gz"
            run_child(
                [
                    "fslroi",
                    str(provisional_mc),
                    str(median_input),
                    str(dropped),
                    str(total_volumes - dropped),
                ],
                env=env,
            )
        run_child(["fslmaths", str(median_input), "-Tmedian", str(robust_ref)], env=env)
        details = {
            "Type": "RobustBOLDReference",
            "Construction": "Temporal median of a provisional MCFLIRT-aligned BOLD series",
            "Statistic": "median",
            "BootstrapReference": "first BOLD volume",
            "MotionCorrection": {
                "Method": "MCFLIRT",
                "Passes": 2,
                "FinalTransformsEstimatedFromRawBOLD": True,
                "FinalInterpolationUsesOnlyFinalPassTransforms": True,
            },
            "WorkingImage": str(robust_ref),
            "TotalBOLDVolumes": total_volumes,
            "InitialNonSteadyStateVolumesExcluded": dropped,
            "VolumesUsed": total_volumes - dropped,
        }
        write_json(robust_metadata, details)
        run_mcflirt(epi_in, robust_ref, final_mc, mc_dir)
        write_completion_breadcrumb(final_matrices_complete, f"matrices={nvols}\n")

    def validate() -> tuple[bool, str]:
        required = (
            robust_ref,
            robust_metadata,
            nonsteady_metadata,
            final_mc,
            final_par,
            *expected_matrices,
        )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            f"Robust reference and fixed {nvols}-matrix motion inventory are complete."
            if not missing
            else "Robust reference construction is missing outputs: " + ", ".join(missing),
        )

    return _RobustBoldReferenceStep(
        step=Step.directory_step(
            name="Robust BOLD Reference and Motion Correction",
            directory=mc_dir,
            breadcrumb=mc_dir / ".nro_complete",
            outputs=(
                robust_ref,
                robust_metadata,
                nonsteady_metadata,
                final_mc,
                final_par,
                final_matrices_complete,
            ),
            inputs=(epi_in,),
            force=force,
            action=construct,
            validate=validate,
            breadcrumb_text=f"matrices={nvols}\n",
        ),
        reference=robust_ref,
        motion_corrected=final_mc,
        metadata=robust_metadata,
    )


def _image_overlap_and_correlation(
    *,
    moving_registered: Path,
    moving_mask_registered: Path,
    fixed: Path,
    fixed_mask: Path,
) -> dict[str, float]:
    import nibabel as nib  # type: ignore
    import numpy as np

    moving_data = np.asarray(nib.load(str(moving_registered)).dataobj, dtype=np.float32)
    fixed_data = np.asarray(nib.load(str(fixed)).dataobj, dtype=np.float32)
    moving_support = np.asarray(nib.load(str(moving_mask_registered)).dataobj) > 0
    fixed_support = np.asarray(nib.load(str(fixed_mask)).dataobj) > 0
    overlap = moving_support & fixed_support & np.isfinite(moving_data) & np.isfinite(fixed_data)
    overlap_count = int(overlap.sum())
    denominator = max(1, min(int(moving_support.sum()), int(fixed_support.sum())))
    support_overlap = float(overlap_count / denominator)
    correlation = 0.0
    if overlap_count >= 2:
        x = moving_data[overlap].astype(np.float64)
        y = fixed_data[overlap].astype(np.float64)
        if float(x.std()) > 0.0 and float(y.std()) > 0.0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    return {
        "SupportOverlapFraction": support_overlap,
        "IntensityCorrelation": correlation,
        "OverlapVoxels": overlap_count,
    }


@dataclass(frozen=True)
class _SelectedFunctionalReferenceStep:
    step: Step
    image: Path
    epi_to_reference: Path
    reference_to_epi: Path
    metadata: Path


def _create_functional_reference_selection_step(
    *,
    run_child: Callable[..., Optional[str]],
    robust_ref: Path,
    epi_metadata: dict[str, Any],
    epi_metadata_sources: tuple[Path, ...],
    sbref: Optional[Path],
    sbref_json: Optional[Path],
    sbref_metadata: Optional[dict[str, Any]],
    sbref_metadata_sources: tuple[Path, ...],
    sbref_metadata_inheritance: Optional[dict[str, Any]],
    work_dir: Path,
    env: dict[str, str],
    max_rotation_degrees: float,
    max_displacement_mm: float,
    min_support_overlap: float,
    min_correlation: float,
    force: bool,
) -> _SelectedFunctionalReferenceStep:
    """Create the fixed registration-reference selection step."""
    selected_image = work_dir / "selected_reference.nii.gz"
    epi_to_selected = work_dir / "epi_to_selected.mat"
    selected_to_epi = work_dir / "selected_to_epi.mat"
    metadata_path = work_dir / "selection.json"
    candidate_3d = work_dir / "sbref_3d.nii.gz"
    header_mat = work_dir / "header_init.mat"
    header_image = work_dir / "header_init_boldref.nii.gz"
    selected_mat = work_dir / "robust_to_sbref.mat"
    registered = work_dir / "robust_in_sbref.nii.gz"
    robust_mask = work_dir / "robust_support_mask.nii.gz"
    sbref_mask = work_dir / "sbref_support_mask.nii.gz"
    registered_mask = work_dir / "robust_support_in_sbref.nii.gz"

    identity = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"

    def choose_robust(details: dict[str, object]) -> None:
        copy_or_convert_nifti(robust_ref, selected_image)
        epi_to_selected.write_text(identity, encoding="utf-8")
        selected_to_epi.write_text(identity, encoding="utf-8")
        details["Selected"] = False
        details["SelectedRegistrationReference"] = "RobustBOLDReference"
        write_json(metadata_path, details)

    def support_mask(source: Path, destination: Path) -> None:
        import nibabel as nib  # type: ignore
        import numpy as np
        from scipy.ndimage import binary_erosion

        image = nib.load(str(source))
        data = np.asarray(image.dataobj)
        if data.ndim > 3:
            data = data[..., 0]
        support = binary_erosion(
            np.isfinite(data) & (np.abs(data) > np.finfo(np.float32).eps),
            iterations=1,
            border_value=0,
        )
        nib.save(
            nib.Nifti1Image(support.astype(np.uint8), image.affine, image.header),
            str(destination),
        )

    def select() -> None:
        import nibabel as nib  # type: ignore
        import numpy as np

        work_dir.mkdir(parents=True, exist_ok=True)
        details: dict[str, object] = {
            "Available": bool(sbref is not None and sbref_json is not None),
            "Selected": False,
            "InputImage": str(sbref) if sbref is not None else None,
            "InputMetadata": [str(path) for path in sbref_metadata_sources],
            "Reasons": [],
            "Metrics": {},
            "Thresholds": {
                "MaximumRigidRotationDegrees": float(max_rotation_degrees),
                "MaximumRigidCenterDisplacementMillimeters": float(max_displacement_mm),
                "MinimumSupportOverlapFraction": float(min_support_overlap),
                "MinimumIntensityCorrelation": float(min_correlation),
                "MaximumReadoutTimeRelativeDifference": 0.05,
            },
        }
        if sbref_metadata_inheritance is not None:
            details["MetadataInheritance"] = dict(sbref_metadata_inheritance)
        reasons = details["Reasons"]
        assert isinstance(reasons, list)
        if sbref is None or sbref_json is None:
            reasons.append("No matched SBRef image and JSON sidecar were available.")
            choose_robust(details)
            return
        try:
            image = nib.load(str(sbref))
            data = np.asarray(image.dataobj)
            if data.ndim == 4:
                data = data[..., 0]
            finite_nonzero = int(np.count_nonzero(np.isfinite(data) & (data != 0)))
            if data.ndim != 3 or finite_nonzero < 100:
                reasons.append(
                    f"SBRef image integrity check failed ({finite_nonzero} finite nonzero 3D voxels)."
                )
            else:
                nib.save(nib.Nifti1Image(data, image.affine, image.header), str(candidate_3d))
        except Exception as error:
            reasons.append(f"SBRef image could not be read: {type(error).__name__}: {error}")

        raw_epi_ped = epi_metadata.get("PhaseEncodingDirection")
        epi_ped = str(raw_epi_ped).strip() if raw_epi_ped is not None else None
        try:
            epi_readout = float(bids_readout_time(epi_metadata))
        except (KeyError, TypeError, ValueError):
            epi_readout = None
        effective_sbref_metadata = (
            dict(sbref_metadata) if sbref_metadata is not None else read_json(sbref_json)
        )
        raw_sbref_ped = effective_sbref_metadata.get("PhaseEncodingDirection")
        sbref_ped = str(raw_sbref_ped).strip() if raw_sbref_ped is not None else None
        try:
            sbref_readout = float(bids_readout_time(effective_sbref_metadata))
        except (KeyError, TypeError, ValueError):
            sbref_readout = None
        metrics = details["Metrics"]
        assert isinstance(metrics, dict)
        metrics.update(
            {
                "BOLDPhaseEncodingDirection": epi_ped,
                "SBRefPhaseEncodingDirection": sbref_ped,
                "BOLDTotalReadoutTime": epi_readout,
                "SBRefTotalReadoutTime": sbref_readout,
            }
        )
        if epi_ped and not sbref_ped:
            reasons.append("SBRef PhaseEncodingDirection metadata are missing.")
        elif epi_ped and sbref_ped and epi_ped != sbref_ped:
            reasons.append(f"PhaseEncodingDirection differs (BOLD={epi_ped}, SBRef={sbref_ped}).")
        if epi_readout is not None and sbref_readout is None:
            reasons.append("SBRef total-readout-time metadata are missing.")
        elif epi_readout is not None and sbref_readout is not None:
            relative = abs(epi_readout - sbref_readout) / max(abs(epi_readout), 1.0e-8)
            metrics["ReadoutTimeRelativeDifference"] = relative
            if relative > 0.05:
                reasons.append(f"TotalReadoutTime differs by {100.0 * relative:.1f}% (limit 5.0%).")
        if reasons:
            choose_robust(details)
            return

        support_mask(robust_ref, robust_mask)
        support_mask(candidate_3d, sbref_mask)
        run_child(
            [
                "flirt",
                "-in",
                str(robust_ref),
                "-ref",
                str(candidate_3d),
                "-usesqform",
                "-applyxfm",
                "-omat",
                str(header_mat),
                "-out",
                str(header_image),
            ],
            env=env,
        )
        search = str(float(max_rotation_degrees))
        run_child(
            [
                "flirt",
                "-in",
                str(robust_ref),
                "-ref",
                str(candidate_3d),
                "-init",
                str(header_mat),
                "-dof",
                "6",
                "-cost",
                "normcorr",
                "-searchrx",
                f"-{search}",
                search,
                "-searchry",
                f"-{search}",
                search,
                "-searchrz",
                f"-{search}",
                search,
                "-omat",
                str(selected_mat),
                "-out",
                str(registered),
                "-interp",
                "trilinear",
            ],
            env=env,
        )
        rigid = rigid_transform_metrics(
            matrix=selected_mat, initial_matrix=header_mat, center_mask=sbref_mask
        )
        used_local = (
            rigid["RotationDegrees"] > max_rotation_degrees
            or rigid["CenterDisplacementMillimeters"] > max_displacement_mm
        )
        if used_local:
            run_child(
                [
                    "flirt",
                    "-in",
                    str(robust_ref),
                    "-ref",
                    str(candidate_3d),
                    "-init",
                    str(header_mat),
                    "-dof",
                    "6",
                    "-cost",
                    "normcorr",
                    "-nosearch",
                    "-omat",
                    str(selected_mat),
                    "-out",
                    str(registered),
                    "-interp",
                    "trilinear",
                ],
                env=env,
            )
            rigid = rigid_transform_metrics(
                matrix=selected_mat, initial_matrix=header_mat, center_mask=sbref_mask
            )
        run_child(
            [
                "flirt",
                "-in",
                str(robust_mask),
                "-ref",
                str(candidate_3d),
                "-applyxfm",
                "-init",
                str(selected_mat),
                "-interp",
                "nearestneighbour",
                "-out",
                str(registered_mask),
            ],
            env=env,
        )
        rigid.update(
            _image_overlap_and_correlation(
                moving_registered=registered,
                moving_mask_registered=registered_mask,
                fixed=candidate_3d,
                fixed_mask=sbref_mask,
            )
        )
        rigid["UsedLocalNoSearchFallback"] = used_local
        metrics.update(rigid)
        if rigid["RotationDegrees"] > max_rotation_degrees:
            reasons.append("SBRef rigid rotation exceeds its configured limit.")
        if rigid["CenterDisplacementMillimeters"] > max_displacement_mm:
            reasons.append("SBRef rigid displacement exceeds its configured limit.")
        if rigid["SupportOverlapFraction"] < min_support_overlap:
            reasons.append("SBRef registered support overlap is below its configured minimum.")
        if rigid["IntensityCorrelation"] < min_correlation:
            reasons.append(
                "SBRef registered intensity correlation is below its configured minimum."
            )
        if reasons:
            choose_robust(details)
            return

        run_child(
            [
                "convert_xfm",
                "-inverse",
                "-omat",
                str(selected_to_epi),
                str(selected_mat),
            ],
            env=env,
        )
        run_child(
            [
                "flirt",
                "-in",
                str(candidate_3d),
                "-ref",
                str(robust_ref),
                "-applyxfm",
                "-init",
                str(selected_to_epi),
                "-interp",
                "sinc",
                "-out",
                str(selected_image),
            ],
            env=env,
        )
        epi_to_selected.write_text(identity, encoding="utf-8")
        selected_to_epi.write_text(identity, encoding="utf-8")
        details["Selected"] = True
        details["SelectedRegistrationReference"] = "SBRef"
        details["NormalizedToBOLDReferenceGrid"] = True
        details["OriginalBOLDToSBRefTransform"] = str(selected_mat)
        reasons.append("SBRef passed metadata, transform, overlap, and similarity checks.")
        write_json(metadata_path, details)

    def validate() -> tuple[bool, str]:
        required = (selected_image, epi_to_selected, selected_to_epi, metadata_path)
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            "Functional registration reference selection is complete."
            if not missing
            else "Functional reference selection is missing outputs: " + ", ".join(missing),
        )

    return _SelectedFunctionalReferenceStep(
        step=Step.python(
            name="Select Functional Registration Reference",
            outputs=(selected_image, epi_to_selected, selected_to_epi, metadata_path),
            inputs=(
                robust_ref,
                *epi_metadata_sources,
                sbref,
                *sbref_metadata_sources,
            ),
            force=force,
            action=select,
            validate=validate,
            parameters={
                "maximum_rotation_degrees": max_rotation_degrees,
                "maximum_displacement_millimeters": max_displacement_mm,
                "minimum_support_overlap": min_support_overlap,
                "minimum_intensity_correlation": min_correlation,
            },
        ),
        image=selected_image,
        epi_to_reference=epi_to_selected,
        reference_to_epi=selected_to_epi,
        metadata=metadata_path,
    )
