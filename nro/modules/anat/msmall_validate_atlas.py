"""Validate the mask-aware atlas transform used by the HCP MSMAll route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _geometry(image: nib.spatialimages.SpatialImage, mask: np.ndarray) -> dict[str, object]:
    indices = np.argwhere(mask)
    if not len(indices):
        raise ValueError("Atlas-registration mask is empty")
    points = nib.affines.apply_affine(image.affine, indices)
    low = points.min(axis=0)
    high = points.max(axis=0)
    return {
        "voxel_count": int(mask.sum()),
        "world_bbox_low_mm": low.tolist(),
        "world_bbox_high_mm": high.tolist(),
        "world_extent_mm": (high - low).tolist(),
    }


def validate(
    *,
    subject_path: Path,
    subject_mask_path: Path,
    reference_path: Path,
    reference_mask_path: Path,
    jacobian_path: Path,
) -> dict[str, object]:
    """Measure one transform and reject the gross failure modes seen in prototyping."""
    subject = nib.load(subject_path)
    subject_mask_image = nib.load(subject_mask_path)
    reference = nib.load(reference_path)
    reference_mask_image = nib.load(reference_mask_path)
    jacobian = np.asarray(nib.load(jacobian_path).dataobj, dtype=np.float32)

    subject_data = np.asarray(subject.dataobj, dtype=np.float32)
    reference_data = np.asarray(reference.dataobj, dtype=np.float32)
    subject_mask = np.asarray(subject_mask_image.dataobj) > 0.5
    reference_mask = np.asarray(reference_mask_image.dataobj) > 0.5
    if subject_data.shape != reference_data.shape or subject_mask.shape != reference_mask.shape:
        raise ValueError("Atlas-registration QC images do not share a grid")
    joint = subject_mask & reference_mask & np.isfinite(subject_data) & np.isfinite(reference_data)
    if np.count_nonzero(joint) < 100:
        raise ValueError("Atlas-registration masks have insufficient overlap")

    correlation = float(np.corrcoef(subject_data[joint], reference_data[joint])[0, 1])
    denominator = int(subject_mask.sum() + reference_mask.sum())
    dice = float(2 * np.count_nonzero(subject_mask & reference_mask) / denominator)
    finite_jacobian = jacobian[np.isfinite(jacobian)]
    if not len(finite_jacobian):
        raise ValueError("Atlas-registration Jacobian is entirely nonfinite")
    quantile_values = np.quantile(
        finite_jacobian,
        [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1],
    )
    quantiles = {
        name: float(value)
        for name, value in zip(
            ("min", "q001", "q01", "q05", "median", "q95", "q99", "q999", "max"),
            quantile_values,
            strict=True,
        )
    }
    nonpositive = float(np.mean(finite_jacobian <= 0))
    subject_geometry = _geometry(subject, subject_mask)
    reference_geometry = _geometry(reference, reference_mask)
    subject_extent = np.asarray(subject_geometry["world_extent_mm"])
    reference_extent = np.asarray(reference_geometry["world_extent_mm"])
    extent_ratio = subject_extent / reference_extent

    failures = []
    if not np.isfinite(correlation) or correlation < 0.3:
        failures.append(f"masked intensity correlation {correlation:.3f} is below 0.3")
    if dice < 0.75:
        failures.append(f"brain-mask Dice {dice:.3f} is below 0.75")
    if nonpositive:
        failures.append(f"Jacobian nonpositive fraction is {nonpositive:.6g}")
    if np.any(extent_ratio < 0.6) or np.any(extent_ratio > 1.4):
        failures.append(f"transformed-brain extent ratios are {extent_ratio.tolist()}")

    return {
        "subject": subject_geometry,
        "reference": reference_geometry,
        "mask_dice": dice,
        "masked_intensity_correlation": correlation,
        "jacobian_quantiles": quantiles,
        "jacobian_nonpositive_fraction": nonpositive,
        "extent_ratio": extent_ratio.tolist(),
        "criteria": {
            "minimum_mask_dice": 0.75,
            "minimum_masked_intensity_correlation": 0.3,
            "maximum_jacobian_nonpositive_fraction": 0.0,
            "extent_ratio_range": [0.6, 1.4],
        },
        "valid": not failures,
        "failures": failures,
    }


def main() -> None:
    """Validate paths supplied by the checkpointed shell driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--subject-mask", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--jacobian", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        subject_path=args.subject,
        subject_mask_path=args.subject_mask,
        reference_path=args.reference,
        reference_mask_path=args.reference_mask,
        jacobian_path=args.jacobian,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit("Invalid MSMAll atlas registration: " + "; ".join(report["failures"]))


if __name__ == "__main__":
    main()
