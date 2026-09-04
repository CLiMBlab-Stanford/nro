from pathlib import Path

import nibabel as nib
import numpy as np

from nro.qc import engine as qc_engine
from nro.qc.registration import (
    create_registration_audit,
    find_registered_bold,
    registration_output_dir,
)


def _write_nifti(path: Path, value: float, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((9, 10, 11, 2), value, dtype=np.float32)
    data[..., 1] = value + 100
    nib.save(nib.Nifti1Image(data, affine), path)


def _write_anatomy(subject_dir: Path, subject: str, affine: np.ndarray) -> None:
    anat = subject_dir / "anat"
    anat.mkdir(parents=True)
    anatomical = np.zeros((9, 10, 11), dtype=np.float32)
    nib.save(nib.Nifti1Image(anatomical, affine), anat / f"{subject}_desc-preproc_T1w.nii.gz")
    for hemi in ("L", "R"):
        for surface in ("white", "pial"):
            (anat / f"{subject}_space-fsnative_hemi-{hemi}_{surface}.surf.gii").write_text("surface")


def test_registration_audit_stacks_first_volumes_and_writes_scene(tmp_path: Path) -> None:
    subject = "sub-test"
    subject_dir = tmp_path / subject
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = (-8.0, -9.0, -10.0)
    first = subject_dir / "func" / f"{subject}_task-a_run-10_space-T1w_desc-preproc_bold.nii.gz"
    second = subject_dir / "ses-one" / "func" / f"{subject}_ses-one_task-a_run-2_space-T1w_desc-preproc_bold.nii.gz"
    _write_nifti(first, 10.0, affine)
    _write_nifti(second, 2.0, affine)
    _write_anatomy(subject_dir, subject, affine)

    assert find_registered_bold(subject_dir) == [first, second]
    outputs = create_registration_audit(
        subject_dir=subject_dir,
        output_dir=tmp_path / "audit",
        subject=subject,
        sagittal_coordinate=0.0,
        slab_thickness=3,
    )

    image = nib.load(outputs["volume"])
    assert image.shape == (3, 10, 11, 2)
    assert np.allclose(image.affine[:3, :3], np.diag([2.0, 2.0, 2.0]))
    assert image.affine[0, 3] + image.header.get_zooms()[0] == 0.0
    assert np.allclose(np.asarray(image.dataobj[..., 0]), 10.0)
    assert np.allclose(np.asarray(image.dataobj[..., 1]), 2.0)
    scene = outputs["scene"].read_text()
    assert outputs["volume"].name in scene
    assert "sub-c001_" not in scene
    assert 'Encoding="Base64"' not in scene
    assert "Functional registration audit" in scene
    assert outputs["metadata"].is_file()


def test_registration_audit_resamples_a_different_grid(tmp_path: Path) -> None:
    subject = "sub-test"
    subject_dir = tmp_path / subject
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shifted = affine.copy()
    shifted[0, 3] = 0.25
    for run, run_affine in ((1, affine), (2, affine), (3, shifted)):
        path = subject_dir / "func" / f"{subject}_task-a_run-{run}_space-T1w_desc-preproc_bold.nii.gz"
        _write_nifti(path, float(run), run_affine)
    _write_anatomy(subject_dir, subject, affine)

    outputs = create_registration_audit(
        subject_dir=subject_dir,
        output_dir=tmp_path / "audit",
        subject=subject,
        sagittal_coordinate=4.0,
        slab_thickness=1,
    )

    rows = outputs["index"].read_text().splitlines()
    assert "\tTrue\t" in rows[-1]
    assert nib.load(outputs["volume"]).shape == (1, 10, 11, 3)


def test_registration_output_is_a_nested_derivative_parallel_to_entities(
    tmp_path: Path,
) -> None:
    derivative_root = tmp_path / "derivatives" / "preprocessing" / "main"

    assert registration_output_dir(derivative_root, "sub-t20") == (
        derivative_root / "derivatives" / "qc" / "registration" / "sub-t20"
    )
    assert registration_output_dir(derivative_root, "sub-t20", "ses-one") == (
        derivative_root
        / "derivatives"
        / "qc"
        / "registration"
        / "sub-t20"
        / "ses-one"
    )


def test_qc_command_dispatches_registration_arguments(monkeypatch) -> None:
    received = []

    def registration(argv, *, prog):
        received.append((argv, prog))

    monkeypatch.setitem(qc_engine.QUALITY_CONTROLS, "registration", registration)

    qc_engine.main(
        ["registration", "t20", "-p", "nptl"], prog="nro qc"
    )

    assert received == [(["t20", "-p", "nptl"], "nro qc registration")]
