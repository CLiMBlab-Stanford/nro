import json
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.func import resolver as func_resolver


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.uint8), np.eye(4)), path)


def test_fieldmaps_are_resolved_when_sbref_has_no_json(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "sub-01" / "ses-a"
    bold = root / "func" / "sub-01_ses-a_task-rest_dir-RL_run-1_bold.nii.gz"
    sbref = root / "func" / "sub-01_ses-a_task-rest_dir-RL_run-1_sbref.nii.gz"
    _image(bold)
    _image(sbref)
    bold.with_suffix("").with_suffix(".json").write_text(
        json.dumps(
            {
                "PhaseEncodingDirection": "i-",
                "EffectiveEchoSpacing": 0.00058,
                "ReconMatrixPE": 104,
            }
        )
    )
    intended = "ses-a/func/" + bold.name
    for direction, ped in (("RL", "i-"), ("LR", "i")):
        fmap = root / "fmap" / f"sub-01_ses-a_acq-rest_dir-{direction}_run-1_epi.nii.gz"
        _image(fmap)
        fmap.with_suffix("").with_suffix(".json").write_text(
            json.dumps(
                {
                    "PhaseEncodingDirection": ped,
                    "TotalReadoutTime": 0.05974,
                    "IntendedFor": intended,
                }
            )
        )
    monkeypatch.setattr(func_resolver, "project_data_root", lambda _project: tmp_path)

    resolved = func_resolver.resolve_func_run_request(
        project="test",
        sub_id="sub-01",
        ses_id="ses-a",
        run_stem="sub-01_ses-a_task-rest_dir-RL_run-1",
        sdc_from_sbref_pair=False,
    )

    assert resolved.sbref is not None
    assert resolved.sbref.img == sbref
    assert resolved.sbref.ped == "i-"
    assert resolved.sbref.metadata_inheritance is not None
    assert resolved.sbref.metadata_inheritance["SourceMetadata"] == [
        str(bold.with_suffix("").with_suffix(".json"))
    ]
    assert resolved.pair is not None
    assert {resolved.pair.se1.ped, resolved.pair.se2.ped} == {"i", "i-"}
    assert all("run-1" in path.name for path in (resolved.pair.se1.img, resolved.pair.se2.img))
    assert resolved.pair.se1.readout == 0.05974
    assert resolved.pair.se2.readout == 0.05974

    assert resolved.sbref.metadata["PhaseEncodingDirection"] == "i-"
    assert resolved.sbref.metadata_inheritance["Applied"] is True
    assert resolved.sbref.metadata_inheritance["TargetImage"] == str(sbref)
    assert not sbref.with_suffix("").with_suffix(".json").exists()


def test_bold_uses_inherited_dataset_metadata(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dataset_description.json").write_text("{}")
    inherited = tmp_path / "task-story_bold.json"
    inherited.write_text(
        json.dumps(
            {
                "RepetitionTime": 2.0,
                "PhaseEncodingDirection": "j-",
                "TotalReadoutTime": 0.05,
            }
        )
    )
    func = tmp_path / "sub-01" / "ses-a" / "func"
    bold = func / "sub-01_ses-a_task-story_bold.nii.gz"
    _image(bold)
    monkeypatch.setattr(func_resolver, "project_data_root", lambda _project: tmp_path)

    resolved = func_resolver.resolve_func_run_request(
        project="test",
        sub_id="sub-01",
        ses_id="ses-a",
        run_stem="sub-01_ses-a_task-story",
        sdc_from_sbref_pair=False,
    )

    assert resolved.bold.metadata_path == inherited
    assert resolved.bold.metadata_sources == (inherited,)
    assert resolved.bold.metadata["RepetitionTime"] == 2.0


def test_sidecarless_sbref_inheritance_fails_closed_when_ambiguous(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "sub-01" / "func"
    bold = root / "sub-01_task-rest_run-1_bold.nii.gz"
    _image(bold)
    bold.with_suffix("").with_suffix(".json").write_text(
        json.dumps({"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05})
    )
    _image(root / "sub-01_task-rest_run-1_sbref.nii.gz")
    _image(root / "sub-01_task-rest_run-1_sbref.nii")
    monkeypatch.setattr(func_resolver, "project_data_root", lambda _project: tmp_path)

    resolved = func_resolver.resolve_func_run_request(
        project="test",
        sub_id="sub-01",
        ses_id=None,
        run_stem="sub-01_task-rest_run-1",
        sdc_from_sbref_pair=False,
    )

    assert resolved.sbref is None
    assert resolved.selection_warning is not None
    assert "multiple sidecarless SBRefs" in resolved.selection_warning
