from pathlib import Path

import nibabel as nib
import numpy as np

from nro.microparcellation.__main__ import (
    infer_gray_matter_mask,
    infer_surface_geometry,
    target_output_names,
)
from nro.engine.bids import BidsRun
from nro.microparcellation.targets import expected_clean_target
from nro.engine.templates import find_fsaverage_surface, find_mni_gray_matter_mask


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_output_names_use_space_without_redundant_domain() -> None:
    directory, prefix = target_output_names("sub-01", "T1w", 2)
    assert directory == "space-T1w_smoothing-2mm"
    assert prefix == "sub-01_space-T1w_smoothing-2mm"
    assert "domain-" not in directory + prefix


def test_expected_target_is_derived_from_source_bids_and_requested_entities(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("nro.clean.paths.BIDS_PATH", str(tmp_path / "bids"))
    runs = (
        BidsRun(
            participant="01", session="a",
            stem="sub-01_ses-a_task-rest_run-01",
            entities={"ses": "a", "task": "rest", "run": "01"},
            path=tmp_path / "bids/demo/sub-01/ses-a/func/run.nii.gz",
        ),
    )
    target = expected_clean_target(
        runs,
        space="fsnative",
        smoothing_mm=2,
        project="demo",
        clean_id="main",
    )
    assert (target.domain, target.space, target.smoothing_mm) == (
        "surface", "fsnative", 2
    )
    assert [path.name for path in target.functional[0]] == [
        "sub-01_ses-a_task-rest_run-01_space-fsnative_smoothing-2mm_hemi-L_desc-clean_bold.func.gii",
        "sub-01_ses-a_task-rest_run-01_space-fsnative_smoothing-2mm_hemi-R_desc-clean_bold.func.gii",
    ]


def _write_functional(path: Path, vertices: int) -> None:
    image = nib.gifti.GiftiImage()
    for _ in range(4):
        image.add_gifti_data_array(
            nib.gifti.GiftiDataArray(np.arange(vertices, dtype=np.float32))
        )
    nib.save(image, path)


def _write_surface(path: Path, vertices: int) -> None:
    image = nib.gifti.GiftiImage()
    image.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            np.zeros((vertices, 3), dtype=np.float32), intent="NIFTI_INTENT_POINTSET"
        )
    )
    image.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            np.array([[0, 1, 2]], dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE"
        )
    )
    nib.save(image, path)


def test_template_surface_geometry_is_resolved_only_from_configured_templateflow(
    tmp_path: Path, monkeypatch
) -> None:
    templateflow = tmp_path / "templateflow" / "tpl-fsaverage"
    templateflow.mkdir(parents=True)
    clean_paths = []
    expected = []
    for hemi, vertices in (("L", 4), ("R", 5)):
        clean = tmp_path / f"sub-01_space-fsaverage_hemi-{hemi}_desc-clean_bold.func.gii"
        template = (
            templateflow
            / f"tpl-fsaverage_hemi-{hemi}_den-test_pial.surf.gii"
        )
        _write_functional(clean, vertices)
        _write_surface(template, vertices)
        clean_paths.append(clean)
        expected.append(template)
    monkeypatch.setenv("TEMPLATEFLOW_HOME", str(templateflow.parent))

    resolved = infer_surface_geometry(
        tmp_path / "empty-anat",
        "01",
        "pial",
        "fsaverage",
        tuple(clean_paths),
    )
    assert resolved == tuple(expected)


def test_fsaverage_surface_is_resolved_from_local_templateflow(tmp_path: Path, monkeypatch) -> None:
    templateflow = tmp_path / "templateflow" / "tpl-fsaverage"
    templateflow.mkdir(parents=True)
    expected = templateflow / "tpl-fsaverage_hemi-L_den-test_pial.surf.gii"
    _write_surface(expected, 7)
    monkeypatch.setenv("TEMPLATEFLOW_HOME", str(templateflow.parent))

    assert find_fsaverage_surface(hemi="L", surface="pial", n_vertices=7) == expected


def test_mni_mask_falls_back_to_matching_local_templateflow_grid(
    tmp_path: Path, monkeypatch
) -> None:
    functional = tmp_path / "sub-01_space-MNITest_desc-clean_bold.nii.gz"
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 3), dtype=np.float32), affine), functional)
    template_dir = tmp_path / "templateflow" / "tpl-MNITest"
    template_dir.mkdir(parents=True)
    wrong = template_dir / "tpl-MNITest_res-01_label-GM_probseg.nii.gz"
    expected = template_dir / "tpl-MNITest_res-02_label-GM_probseg.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 9, 10), dtype=np.float32), np.eye(4)), wrong)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), affine), expected)
    monkeypatch.setenv("TEMPLATEFLOW_HOME", str(template_dir.parent))

    assert infer_gray_matter_mask(
        ((functional,),), None, project="unused", space="MNITest"
    ) == expected


def test_mni_mask_accepts_cropped_functional_grid_at_template_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    functional = tmp_path / "sub-01_space-MNITest_desc-clean_bold.nii.gz"
    functional_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    functional_affine[:3, 3] = (-94.0, -132.0, -78.0)
    nib.save(
        nib.Nifti1Image(np.zeros((4, 5, 4, 2), dtype=np.float32), functional_affine),
        functional,
    )
    template_dir = tmp_path / "templateflow" / "tpl-MNITest"
    template_dir.mkdir(parents=True)
    one_mm = template_dir / "tpl-MNITest_res-01_label-GM_probseg.nii.gz"
    two_mm = template_dir / "tpl-MNITest_res-02_label-GM_probseg.nii.gz"
    one_mm_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    two_mm_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    one_mm_affine[:3, 3] = (-96.0, -132.0, -78.0)
    two_mm_affine[:3, 3] = (-96.5, -132.5, -78.5)
    nib.save(nib.Nifti1Image(np.ones((9, 11, 9), dtype=np.float32), one_mm_affine), one_mm)
    nib.save(nib.Nifti1Image(np.ones((5, 6, 5), dtype=np.float32), two_mm_affine), two_mm)
    monkeypatch.setenv("TEMPLATEFLOW_HOME", str(template_dir.parent))

    assert find_mni_gray_matter_mask(space="MNITest", functional=functional) == two_mm
