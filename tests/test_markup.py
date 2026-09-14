from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.configuration.definitions import create_store, validate_store
from nro.configuration.markup import MarkupStore, SubjectMarkup, compile_markup
from nro.configuration.parsing import DefinitionError
from nro.configuration.store import ConfigStore, WorkflowError
from nro.engine.bids import discover_raw_runs
from nro.modules.anat.inputs import load_anat_image
from nro.modules.anat.planning import raw_anatomical_images, raw_anatomical_inputs


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_markup_compiler_uses_project_and_subject_hierarchy() -> None:
    result = compile_markup(
        {
            "nptl": {
                "sub-t20": {
                    "T1w": "ses-anat/anat/sub-t20_ses-anat_T1w.nii.gz",
                    "exclude": ["ses-bad/func/sub-t20_ses-bad_task-rest_bold.nii.gz"],
                },
                "t12": None,
            }
        }
    )

    assert result["nptl"]["t20"]["T1w"] == ("ses-anat/anat/sub-t20_ses-anat_T1w.nii.gz",)
    assert result["nptl"]["t20"]["T2w"] == ()
    assert result["nptl"]["t12"] == {"T1w": (), "T2w": (), "exclude": ()}


def test_missing_markup_fields_retain_automatic_discovery(tmp_path: Path) -> None:
    subject = tmp_path / "BIDS/nptl/sub-t20"
    automatic_t1 = _write(subject / "ses-a/anat/sub-t20_ses-a_T1w.nii.gz")
    automatic_t2 = _write(subject / "ses-a/anat/sub-t20_ses-a_T2w.nii.gz")
    marked_t1 = _write(subject / "ses-b/anat/sub-t20_ses-b_T1w.nii.gz")
    excluded_run = _write(subject / "ses-a/func/sub-t20_ses-a_task-rest_run-1_bold.nii.gz")
    retained_run = _write(subject / "ses-a/func/sub-t20_ses-a_task-rest_run-2_bold.nii.gz")
    root = tmp_path / "definitions"
    _write(
        root / "markup/main_markup.yml",
        "nptl:\n  t20:\n    T1w: ses-b/anat/sub-t20_ses-b_T1w.nii.gz\n"
        "    exclude:\n      - ses-a/func/sub-t20_ses-a_task-rest_run-1_bold.nii.gz\n",
    )
    markup = MarkupStore(root).subject("main", "nptl", subject)

    assert raw_anatomical_images(subject, markup) == (marked_t1, automatic_t2)
    assert [run.path for run in discover_raw_runs(subject, markup=markup)] == [retained_run]
    assert markup.is_excluded(excluded_run)
    assert not markup.is_excluded(automatic_t1)


def test_missing_project_or_subject_produces_empty_markup(tmp_path: Path) -> None:
    root = tmp_path / "definitions"
    _write(root / "markup/main_markup.yml", "other:\n  someone: {}\n")
    subject = tmp_path / "BIDS/nptl/sub-t20"

    markup = MarkupStore(root).subject("main", "nptl", subject)

    assert markup == SubjectMarkup("main", "nptl", subject)


def test_captured_markup_rejects_paths_outside_subject(tmp_path: Path) -> None:
    value = SubjectMarkup("main", "nptl", tmp_path / "sub-01").as_dict()
    value["exclude"] = [str(tmp_path / "sub-02/bad.nii.gz")]

    with pytest.raises(ValueError, match="escapes"):
        SubjectMarkup.from_dict(json.loads(json.dumps(value)))


def test_markup_rejects_scalar_exclusions_and_anatomical_conflicts() -> None:
    with pytest.raises(DefinitionError, match="must be a list"):
        compile_markup({"nptl": {"t20": {"exclude": "ses-bad"}}})
    with pytest.raises(DefinitionError, match="selects and excludes"):
        compile_markup(
            {
                "nptl": {
                    "t20": {
                        "T1w": "ses-anat/anat/sub-t20_T1w.nii.gz",
                        "exclude": ["ses-anat"],
                    }
                }
            }
        )


def test_excluded_metadata_is_absent_from_planning_and_execution(
    tmp_path: Path, monkeypatch
) -> None:
    subject = tmp_path / "BIDS/nptl/sub-t20"
    image = _write(subject / "anat/sub-t20_T1w.nii.gz")
    sidecar = _write(subject / "anat/sub-t20_T1w.json", '{"AcquisitionTime": "12:00:00"}')
    markup = SubjectMarkup("main", "nptl", subject, excluded=(sidecar,))

    assert raw_anatomical_inputs(subject, markup) == (image,)
    monkeypatch.setenv("NRO_SOURCE_MARKUP", json.dumps(markup.as_dict()))
    assert load_anat_image(image, default_session="ses-default").json is None


def test_existing_store_may_omit_empty_main_markup(tmp_path: Path) -> None:
    root = create_store(tmp_path / "definitions")
    (root / "markup/main_markup.yml").unlink()
    (root / "markup").rmdir()

    assert validate_store(root)["markup"] == 0
    assert MarkupStore(root).subject("main", "nptl", tmp_path / "sub-t20").t1w == ()


def test_workflow_rejects_mixed_markup_views(tmp_path: Path) -> None:
    root = create_store(tmp_path / "definitions")
    _write(root / "markup/alternate_markup.yml", "{}\n")
    _write(root / "configs/anat/alternate_anat.yml", "markup: alternate\n")
    _write(root / "workflows/mixed_workflow.yml", "anat: alternate\n")

    with pytest.raises(WorkflowError, match="same markup"):
        ConfigStore(root).resolve("mixed")
