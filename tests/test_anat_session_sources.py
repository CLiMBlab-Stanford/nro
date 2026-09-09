from pathlib import Path

import pytest

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.modules.anat import constants as anat_constants
from nro.modules.anat import steps as anat_steps
from nro.modules.anat.common import AnatImage


def test_gray_matter_aseg_labels_have_named_definitions() -> None:
    segmentations = anat_constants._FREESURFER_GRAY_MATTER_SEGMENTATIONS
    labels = anat_steps._aseg_label_ids(segmentations)

    assert len(segmentations) == len(set(segmentations))
    assert set(labels) == {
        3,
        8,
        10,
        11,
        12,
        13,
        17,
        18,
        26,
        28,
        42,
        47,
        49,
        50,
        51,
        52,
        53,
        54,
        58,
        60,
    }
    assert anat_constants._FREESURFER_ASEG_LABELS["Left-Cerebral-Cortex"] == 3
    assert anat_constants._FREESURFER_ASEG_LABELS["Right-VentralDC"] == 60


def test_session_source_brain_extraction_consumes_preprocessed_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw" / "sub-1_ses-1_T1w.nii.gz"
    raw.parent.mkdir()
    raw.write_bytes(b"raw")
    image = AnatImage(
        image=raw,
        json=None,
        modality="T1w",
        session_id="ses-1",
        entities={"sub": "1", "ses": "1"},
        time_kind="series",
        time_value=1.0,
    )
    session_dir = tmp_path / "derivatives" / "sub-1" / "ses-1" / "anat"
    work_dir = tmp_path / "work" / "sub-1" / "ses-1"
    monkeypatch.setattr(
        anat_steps,
        "preprocessing_session_anat_dir",
        lambda *_args, **_kwargs: session_dir,
    )
    monkeypatch.setattr(
        anat_steps,
        "preprocessing_session_work_dir",
        lambda *_args, **_kwargs: work_dir,
    )
    plans = anat_steps._plan_session_anatomicals(
        images=(image,),
        project="project",
        preprocessing_id="preprocessing",
        sub_id="sub-1",
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.staged_preprocessed == (
        work_dir / "anat" / "session_level" / "sub-1_ses-1_T1w_desc-preproc_T1w.nii.gz"
    )
    assert plan.final_source == plan.staged_preprocessed
    assert plan.output == session_dir / raw.name


def test_brain_extraction_reads_source_and_owns_its_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "destination.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    synthstrip = tmp_path / "synthstrip"

    step = anat_steps._brain_extract_anat_copy(
        synthstrip,
        env={},
        source=source,
        dst=destination,
        mask=mask,
        force=False,
    )

    assert step.command[2] == str(source)
    assert step.inputs == (source,)
    assert step.outputs == (destination, mask)
