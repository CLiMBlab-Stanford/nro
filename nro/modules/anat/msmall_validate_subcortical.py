"""Validate the subject subcortical models required by HCP grayordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

REQUIRED_LABELS = {
    26: "ACCUMBENS_LEFT",
    58: "ACCUMBENS_RIGHT",
    18: "AMYGDALA_LEFT",
    54: "AMYGDALA_RIGHT",
    16: "BRAIN_STEM",
    11: "CAUDATE_LEFT",
    50: "CAUDATE_RIGHT",
    8: "CEREBELLUM_LEFT",
    47: "CEREBELLUM_RIGHT",
    28: "DIENCEPHALON_VENTRAL_LEFT",
    60: "DIENCEPHALON_VENTRAL_RIGHT",
    17: "HIPPOCAMPUS_LEFT",
    53: "HIPPOCAMPUS_RIGHT",
    13: "PALLIDUM_LEFT",
    52: "PALLIDUM_RIGHT",
    12: "PUTAMEN_LEFT",
    51: "PUTAMEN_RIGHT",
    10: "THALAMUS_LEFT",
    49: "THALAMUS_RIGHT",
}


def _counts(path: Path) -> dict[int, int]:
    values = np.rint(np.asarray(nib.load(path).dataobj)).astype(np.int32)
    return {label: int(np.count_nonzero(values == label)) for label in REQUIRED_LABELS}


def validate(*, subject_path: Path, reference_path: Path) -> dict[str, object]:
    """Require every subcortical structure expected by the HCP CIFTI template."""
    subject = _counts(subject_path)
    reference = _counts(reference_path)
    missing_subject = [REQUIRED_LABELS[label] for label, count in subject.items() if count == 0]
    missing_reference = [REQUIRED_LABELS[label] for label, count in reference.items() if count == 0]
    failures = []
    if missing_subject:
        failures.append("subject labels are empty: " + ", ".join(missing_subject))
    if missing_reference:
        failures.append("reference labels are empty: " + ", ".join(missing_reference))
    return {
        "subject_voxels": {REQUIRED_LABELS[label]: count for label, count in subject.items()},
        "reference_voxels": {REQUIRED_LABELS[label]: count for label, count in reference.items()},
        "valid": not failures,
        "failures": failures,
    }


def main() -> None:
    """Validate paths supplied by the staged MSMAll driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(subject_path=args.subject, reference_path=args.reference)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit("Invalid MSMAll subcortical models: " + "; ".join(report["failures"]))


if __name__ == "__main__":
    main()
