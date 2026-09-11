import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from nro.engine.bids import (
    bids_entity,
    replace_bids_entity_token,
    resolve_bids_metadata,
    strip_bids_prefix,
)
from nro.engine.cleaned_timeseries import (
    CleanedRunInclusionPolicy,
    cleaned_run_exclusions,
    load_cleaned_run_metadata,
    load_retained_frame_mask,
)
from nro.engine.connectivity import connectivity_run_weights
from nro.engine.images import sidecar_json_path
from nro.engine.io import flatten_paths, manifest_value


def test_image_sidecars_preserve_gifti_type() -> None:
    assert sidecar_json_path(Path("sub-1_bold.nii.gz")) == Path("sub-1_bold.json")
    assert sidecar_json_path(Path("sub-1_hemi-L_bold.func.gii")) == Path(
        "sub-1_hemi-L_bold.func.json"
    )


def test_bids_metadata_inheritance_merges_root_to_image(tmp_path: Path) -> None:
    image = tmp_path / "sub-01" / "ses-a" / "func" / "sub-01_ses-a_task-rest_run-1_bold.nii.gz"
    image.parent.mkdir(parents=True)
    image.touch()
    (tmp_path / "dataset_description.json").write_text("{}")
    root_metadata = tmp_path / "task-rest_bold.json"
    root_metadata.write_text(json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"}))
    session_metadata = image.parents[1] / "sub-01_ses-a_task-rest_bold.json"
    session_metadata.write_text(json.dumps({"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-"}))
    exact_metadata = image.with_suffix("").with_suffix(".json")
    exact_metadata.write_text(json.dumps({"TotalReadoutTime": 0.05}))
    (tmp_path / "task-other_bold.json").write_text(json.dumps({"RepetitionTime": 99.0}))

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
    assert flatten_paths((*values, {"nested": Path("d")})) == [
        Path("a"),
        Path("b"),
        Path("c"),
        Path("d"),
    ]
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


def _write_cleaned_run_contract(
    tmp_path: Path,
    *,
    right_retained_frames: int = 3,
) -> tuple[Path, Path]:
    mask = tmp_path / "sub-1_task-rest_desc-confounds_timeseries.tsv"
    mask.write_text("motion_outlier00\n0\n1\n0\n0\n", encoding="utf-8")
    files = tuple(
        tmp_path / f"sub-1_task-rest_hemi-{hemi}_desc-clean_bold.func.gii" for hemi in ("L", "R")
    )
    for file, retained_frames, effective_rank, dominant_fraction in (
        (files[0], 3, 12.0, 0.35),
        (files[1], right_retained_frames, 10.0, 0.45),
    ):
        file.touch()
        sidecar_json_path(file).write_text(
            json.dumps(
                {
                    "Cleaning": {
                        "CleaningDefined": True,
                        "TotalFrames": 4,
                        "RetainedFrames": retained_frames,
                        "CensoredFraction": 0.25,
                        "ResidualDesignDegreesOfFreedom": 55,
                        "AlgebraicTemporalRank": 40,
                        "TemporalMaskFile": str(mask),
                        "TemporalMaskRegex": ".*outlier.*",
                        "QualityControl": {
                            "ParticipationRatioEffectiveTemporalRank": effective_rank,
                            "DominantTemporalVarianceFraction": dominant_fraction,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
    return files


def test_cleaned_run_contract_drives_inclusion_and_temporal_mask(tmp_path: Path) -> None:
    files = _write_cleaned_run_contract(tmp_path)
    metadata = load_cleaned_run_metadata(files)
    policy = CleanedRunInclusionPolicy(
        minimum_retained_frames=3,
        minimum_retained_fraction=0.7,
        minimum_residual_design_dof=50,
        minimum_participation_effective_rank=10.0,
        maximum_dominant_temporal_variance_fraction=0.5,
    )

    assert metadata.participation_effective_rank == 10.0
    assert metadata.dominant_temporal_variance_fraction == 0.45
    assert cleaned_run_exclusions(metadata, policy) == ()
    np.testing.assert_array_equal(
        load_retained_frame_mask(metadata),
        np.asarray([True, False, True, True]),
    )


def test_connectivity_weights_are_equal_or_linear_in_effective_dof(tmp_path: Path) -> None:
    first = load_cleaned_run_metadata(_write_cleaned_run_contract(tmp_path))
    second = replace(first, algebraic_temporal_rank=20)

    equal = connectivity_run_weights((first, second), weighting="equal")
    precision = connectivity_run_weights(
        (first, second), weighting="precision", global_signal_regression=True
    )

    assert [item.normalized_weight for item in equal] == [0.5, 0.5]
    assert [item.effective_dof for item in precision] == [39, 19]
    np.testing.assert_allclose(
        [item.normalized_weight for item in precision],
        np.asarray([39, 19]) / 58,
    )


def test_cleaned_run_contract_rejects_disagreeing_surface_sidecars(
    tmp_path: Path,
) -> None:
    files = _write_cleaned_run_contract(tmp_path, right_retained_frames=2)

    try:
        load_cleaned_run_metadata(files)
    except ValueError as error:
        assert "disagree on Cleaning.RetainedFrames" in str(error)
    else:
        raise AssertionError("disagreeing cleaned-run sidecars were accepted")


def test_cleaned_run_policy_reports_sidecar_metrics_without_loading_data(
    tmp_path: Path,
) -> None:
    metadata = load_cleaned_run_metadata(_write_cleaned_run_contract(tmp_path))
    policy = CleanedRunInclusionPolicy(
        minimum_retained_frames=4,
        minimum_retained_fraction=0.9,
        minimum_residual_design_dof=60,
        minimum_participation_effective_rank=11.0,
        maximum_dominant_temporal_variance_fraction=0.4,
    )

    assert [record["reason"] for record in cleaned_run_exclusions(metadata, policy)] == [
        "retained_frames_below_minimum",
        "retained_fraction_below_minimum",
        "residual_design_dof_below_minimum",
        "participation_effective_rank_below_minimum",
        "dominant_temporal_variance_fraction_above_maximum",
    ]
