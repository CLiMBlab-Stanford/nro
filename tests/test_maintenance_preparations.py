from __future__ import annotations

import json

import pytest

from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.maintenance_preparations import discard, load, retain


def test_preparation_survives_process_local_state(tmp_path) -> None:
    token = retain(
        tmp_path,
        kind="dataset_migration",
        scope="demo",
        payload={"paths": ["one", "two"]},
    )

    assert load(tmp_path, kind="dataset_migration", token=token) == {"paths": ["one", "two"]}
    record = json.loads(
        (ControlPaths(tmp_path).maintenance_preparations / f"{token}.json").read_text()
    )
    assert record["payload"] == {"paths": ["one", "two"]}


def test_new_preparation_supersedes_only_the_same_scope(tmp_path) -> None:
    old = retain(tmp_path, kind="dataset_migration", scope="demo", payload={"value": 1})
    other = retain(tmp_path, kind="dataset_migration", scope="other", payload={"value": 2})
    new = retain(tmp_path, kind="dataset_migration", scope="demo", payload={"value": 3})

    with pytest.raises(ValueError, match="unavailable"):
        load(tmp_path, kind="dataset_migration", token=old)
    assert load(tmp_path, kind="dataset_migration", token=other) == {"value": 2}
    assert load(tmp_path, kind="dataset_migration", token=new) == {"value": 3}


def test_discard_requires_the_matching_kind(tmp_path) -> None:
    token = retain(tmp_path, kind="project_rename", scope="old-new", payload={"value": 1})

    with pytest.raises(ValueError, match="does not match"):
        discard(tmp_path, kind="dataset_migration", token=token)
    assert load(tmp_path, kind="project_rename", token=token) == {"value": 1}

    discard(tmp_path, kind="project_rename", token=token)
    with pytest.raises(ValueError, match="unavailable"):
        load(tmp_path, kind="project_rename", token=token)
