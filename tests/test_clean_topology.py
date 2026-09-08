from pathlib import Path

import pytest

from nro.clean.contract import (
    CLEAN_SIDECAR_FIELDS,
    clean_output_contract,
    validate_clean_sidecar,
)
from nro.clean.module import _expected_input_groups, _volume_gm_mask_path


def test_cleaning_inputs_are_fixed_by_run_and_preprocessing_config(
    tmp_path: Path,
) -> None:
    volume_variants = _expected_input_groups(
        func_dir=tmp_path,
        run_stem="sub-01_ses-a_task-rest_dir-AP_run-01",
        output_space="T1w",
        clean_ica_aroma=True,
    )
    surface_variants = _expected_input_groups(
        func_dir=tmp_path,
        run_stem="sub-01_ses-a_task-rest_dir-AP_run-01",
        output_space="fsnative",
        clean_ica_aroma=True,
    )

    assert [item["input_desc"] for item in volume_variants] == [
        "desc-preproc",
        "desc-preprocNoAROMA",
    ]
    assert [path.name for path in volume_variants[0]["vols"]] == [
        "sub-01_ses-a_task-rest_dir-AP_run-01_space-T1w_desc-preproc_bold.nii.gz",
    ]
    assert [path.name for path in surface_variants[0]["surfs"]] == [
        "sub-01_ses-a_task-rest_dir-AP_run-01_space-fsnative_hemi-L_desc-preproc_bold.func.gii",
    ]


def test_cleaning_topology_does_not_depend_on_derivative_directory_contents(
    tmp_path: Path,
) -> None:
    before = _expected_input_groups(
        func_dir=tmp_path,
        run_stem="sub-01_task-rest_run-01",
        output_space="T1w",
        clean_ica_aroma=False,
    )
    (tmp_path / "unrelated_space-fsnative_desc-preproc_bold.func.gii").touch()
    (tmp_path / "stale_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz").touch()
    after = _expected_input_groups(
        func_dir=tmp_path,
        run_stem="sub-01_task-rest_run-01",
        output_space="T1w",
        clean_ica_aroma=False,
    )
    assert after == before


def test_volume_gray_matter_mask_is_shared_across_cleaning_variants(
    tmp_path: Path,
) -> None:
    masks_by_space: dict[str, Path] = {}
    common = {
        "masks_by_space": masks_by_space,
        "work_dir": tmp_path / "work",
    }
    preproc = tmp_path / "sub-01_space-T1w_desc-preproc_bold.nii.gz"
    no_aroma = tmp_path / "sub-01_space-T1w_desc-preprocNoAROMA_bold.nii.gz"

    first, first_is_new = _volume_gm_mask_path(
        **common,
        target_img=preproc,
        input_desc="desc-preproc",
    )
    second, second_is_new = _volume_gm_mask_path(
        **common,
        target_img=no_aroma,
        input_desc="desc-preprocNoAROMA",
    )

    assert first == second
    assert first.name == "sub-01_space-T1w_desc-grayMatterMask_bold.nii.gz"
    assert first_is_new
    assert not second_is_new


def test_clean_output_contract_tracks_required_temporal_metadata() -> None:
    contract = clean_output_contract()
    fields = contract["sidecar_fields"]

    assert fields["Cleaning.TemporalMaskFile"] == "string"
    assert fields["Cleaning.RetainedFrames"] == "integer"
    assert (
        fields["Cleaning.QualityControl.ParticipationRatioEffectiveTemporalRank"]
        == "number"
    )


def test_clean_sidecar_validation_rejects_a_missing_contract_field() -> None:
    def sample(kind: str):
        return {
            "boolean": True,
            "integer": 1,
            "number": 1.0,
            "nullable_number": None,
            "string": "value",
            "nullable_string": None,
            "string_list": ["source"],
        }[kind]

    document: dict[str, object] = {}
    for path, kind in CLEAN_SIDECAR_FIELDS.items():
        target = document
        components = path.split(".")
        for component in components[:-1]:
            target = target.setdefault(component, {})
        target[components[-1]] = sample(kind)

    validate_clean_sidecar(document, volume=False)
    del document["Cleaning"]["TemporalMaskFile"]
    with pytest.raises(ValueError, match="Cleaning.TemporalMaskFile"):
        validate_clean_sidecar(document, volume=False)
