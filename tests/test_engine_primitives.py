import json
from pathlib import Path

from nro.engine.bids import (
    bids_entity,
    replace_bids_entity_token,
    resolve_bids_metadata,
    strip_bids_prefix,
)
from nro.engine.images import sidecar_json_path
from nro.engine.io import flatten_paths, manifest_value


def test_image_sidecars_preserve_gifti_type() -> None:
    assert sidecar_json_path(Path("sub-1_bold.nii.gz")) == Path("sub-1_bold.json")
    assert sidecar_json_path(Path("sub-1_hemi-L_bold.func.gii")) == Path(
        "sub-1_hemi-L_bold.func.json"
    )


def test_bids_metadata_inheritance_merges_root_to_image(tmp_path: Path) -> None:
    image = (
        tmp_path
        / "sub-01"
        / "ses-a"
        / "func"
        / "sub-01_ses-a_task-rest_run-1_bold.nii.gz"
    )
    image.parent.mkdir(parents=True)
    image.touch()
    (tmp_path / "dataset_description.json").write_text("{}")
    root_metadata = tmp_path / "task-rest_bold.json"
    root_metadata.write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}))
    session_metadata = image.parents[1] / "sub-01_ses-a_task-rest_bold.json"
    session_metadata.write_text(
        json.dumps({"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-"})
    )
    exact_metadata = image.with_suffix("").with_suffix(".json")
    exact_metadata.write_text(json.dumps({"TotalReadoutTime": 0.05}))
    (tmp_path / "task-other_bold.json").write_text(
        json.dumps({"RepetitionTime": 99.0})
    )

    resolved = resolve_bids_metadata(image)

    assert resolved.values == {
        "RepetitionTime": 1.5,
        "TaskName": "rest",
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.05,
    }
    assert resolved.sources == (
        root_metadata,
        session_metadata,
        exact_metadata,
    )


def test_nested_path_collections_have_one_shared_conversion() -> None:
    values = (Path("a"), [Path("b"), Path("c")])
    assert flatten_paths(values) == [Path("a"), Path("b"), Path("c")]
    assert manifest_value(values) == ["a", ["b", "c"]]


def test_bids_filename_primitives_use_complete_entity_tokens() -> None:
    path = Path("sub-1_space-fsnative_desc-preproc_bold.func.gii")
    replaced = replace_bids_entity_token(
        path,
        "desc-preproc",
        "desc-clean",
    )
    assert replaced.name == "sub-1_space-fsnative_desc-clean_bold.func.gii"
    assert bids_entity(replaced, "space") == "fsnative"
    assert strip_bids_prefix("sub-1", "sub") == "1"
