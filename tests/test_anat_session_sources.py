from pathlib import Path

import pytest

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.modules.anat import constants as anat_constants
from nro.modules.anat import steps as anat_steps
from nro.modules.anat.inputs import AnatImage, load_anat_image


def test_anatomical_input_preserves_its_logical_bids_path(tmp_path: Path) -> None:
    source = tmp_path / "source" / "sub-1_T1w.nii.gz"
    source.parent.mkdir()
    source.write_bytes(b"image")
    logical = tmp_path / "dataset" / "sub-1" / "ses-1" / "anat" / "sub-1_ses-1_T1w.nii.gz"
    logical.parent.mkdir(parents=True)
    logical.symlink_to(source)

    image = load_anat_image(logical)

    assert image.image == logical
    assert image.image.resolve() == source


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
        "anat_session_dir",
        lambda *_args, **_kwargs: session_dir,
    )
    monkeypatch.setattr(
        anat_steps,
        "anat_session_work_dir",
        lambda *_args, **_kwargs: work_dir,
    )
    plans = anat_steps._plan_session_anatomicals(
        images=(image,),
        project="project",
        anat_id="main",
        sub_id="sub-1",
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.staged_preprocessed == (
        work_dir / "session_level" / "sub-1_ses-1_T1w_desc-preproc_T1w.nii.gz"
    )
    assert plan.output == session_dir / raw.name


def test_session_plans_keep_modalities_independent_and_repeated_acquisitions_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def image(modality: str, run: str, series: float) -> AnatImage:
        path = raw_dir / f"sub-1_ses-1_run-{run}_{modality}.nii.gz"
        path.write_bytes(b"raw")
        return AnatImage(
            image=path,
            json=None,
            modality=modality,
            session_id="ses-1",
            entities={"sub": "1", "ses": "1", "run": run},
            time_kind="series",
            time_value=series,
        )

    t1w_run_1 = image("T1w", "1", 1.0)
    t1w_run_2 = image("T1w", "2", 3.0)
    t2w_run_1 = image("T2w", "1", 2.0)
    t2w_run_2 = image("T2w", "2", 4.0)
    session_dir = tmp_path / "derivatives" / "sub-1" / "ses-1" / "anat"
    work_dir = tmp_path / "work" / "sub-1" / "ses-1"
    monkeypatch.setattr(
        anat_steps,
        "anat_session_dir",
        lambda *_args, **_kwargs: session_dir,
    )
    monkeypatch.setattr(
        anat_steps,
        "anat_session_work_dir",
        lambda *_args, **_kwargs: work_dir,
    )

    plans = anat_steps._plan_session_anatomicals(
        images=(t1w_run_1, t1w_run_2, t2w_run_1, t2w_run_2),
        project="project",
        anat_id="main",
        sub_id="sub-1",
    )

    assert len({plan.staged_preprocessed for plan in plans}) == 4
    for plan in plans:
        assert plan.output.name == plan.source.image.name
        assert "SpatialReference" not in plan.metadata
        assert "TransformToT1w" not in plan.metadata


def test_brain_extraction_reads_source_and_owns_its_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "missing" / "anat" / "destination.nii.gz"
    mask = tmp_path / "missing" / "anat" / "mask.nii.gz"
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
    assert step.prepare is not None
    step.prepare()
    assert destination.parent.is_dir()


def test_recon_all_uses_external_mask_and_pinned_container(tmp_path: Path) -> None:
    t1w = tmp_path / "input" / "sub-1_T1w.nii.gz"
    t1w.parent.mkdir()
    t1w.write_bytes(b"image")
    image = tmp_path / "fastsurfer.sif"
    image.write_bytes(b"container")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    subjects_dir = tmp_path / "subjects"
    subject_dir = subjects_dir / "sub-1"
    calls: list[list[str]] = []

    def run_child(command, **_kwargs):
        command = list(command)
        calls.append(command)
        if "mri_vol2vol" in command:
            path = subject_dir / "mri" / "brainmask.auto.mgz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mask")
        if "-autorecon3" in command:
            (subject_dir / "scripts").mkdir(parents=True, exist_ok=True)
            (subject_dir / "scripts" / "recon-all.done").write_text("done")
            for path in anat_steps._recon_required_outputs(subject_dir):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"output")

    step = anat_steps._create_recon_all_step(
        run_child=run_child,
        env={},
        t1w=t1w,
        t2w=None,
        subjects_dir=subjects_dir,
        fs_subject="sub-1",
        runtime="singularity",
        image=image,
        license_file=license_file,
        force=False,
    )
    assert step.action is not None
    step.action()

    assert len(calls) == 3
    assert all(call[:3] == ["singularity", "exec", "--cleanenv"] for call in calls)
    assert "-autorecon1" in calls[0]
    assert "-noskullstrip" in calls[0]
    assert "mri_vol2vol" in calls[1]
    assert "-autorecon2" in calls[2]
    assert "-autorecon3" in calls[2]
    assert "-noskullstrip" in calls[2]
    assert step.scientific_signature
