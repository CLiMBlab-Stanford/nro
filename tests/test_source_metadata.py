from __future__ import annotations

import json
from pathlib import Path

from nro.engine.source_metadata import semantic_metadata_snapshot


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "demo", "BIDSVersion": "1.10.0"})
    )
    image = root / "sub-01/func/sub-01_task-rest_bold.nii.gz"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    sidecar = image.with_name("sub-01_task-rest_bold.json")
    sidecar.write_text(
        json.dumps(
            {
                "RepetitionTime": 1.5,
                "TaskName": "rest",
                "EventsFile": "/legacy/events.tsv",
            }
        )
    )
    return image, sidecar


def test_semantic_metadata_ignores_unconsumed_sidecar_fields(tmp_path: Path) -> None:
    image, sidecar = _dataset(tmp_path)
    before = semantic_metadata_snapshot((image,), module="func")

    metadata = json.loads(sidecar.read_text())
    metadata["EventsFile"] = "/another/legacy/events.tsv"
    metadata["TaskName"] = "renamed"
    sidecar.write_text(json.dumps(metadata))
    after = semantic_metadata_snapshot((image,), module="func")

    assert (
        before
        == after
        == [
            {
                "source": "sub-01/func/sub-01_task-rest_bold.nii.gz",
                "fields": {"RepetitionTime": 1.5},
            }
        ]
    )


def test_semantic_metadata_tracks_consumed_sidecar_fields(tmp_path: Path) -> None:
    image, sidecar = _dataset(tmp_path)
    before = semantic_metadata_snapshot((image,), module="func")

    metadata = json.loads(sidecar.read_text())
    metadata["RepetitionTime"] = 2.0
    sidecar.write_text(json.dumps(metadata))

    assert semantic_metadata_snapshot((image,), module="func") != before
