import os
from pathlib import Path

from nro.orchestration import planning_cache


class MemoryCache:
    def __init__(self) -> None:
        self.records = {}

    def planning_file_records(self, paths):
        return {path: self.records[path] for path in paths if path in self.records}

    def record_planning_files(self, records):
        self.records.update(records)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_participant_manifest_reuses_unchanged_file_checksums(tmp_path, monkeypatch) -> None:
    project = tmp_path / "demo"
    subject = project / "sub-01"
    _write(project / "task-rest_bold.json", '{"RepetitionTime": 1.0}')
    _write(subject / "func/sub-01_task-rest_bold.nii.gz", "header")
    _write(subject / "func/sub-01_task-rest_bold.json", '{"EchoTime": 0.01}')
    cache = MemoryCache()

    first = planning_cache.participant_source_manifest(cache, project, subject)
    assert len(cache.records) == 3
    monkeypatch.setattr(
        planning_cache,
        "_digest",
        lambda _path, _kind: (_ for _ in ()).throw(
            AssertionError("reread an unchanged source file")
        ),
    )

    assert planning_cache.participant_source_manifest(cache, project, subject) == first


def test_participant_manifest_changes_with_metadata_and_inventory(tmp_path) -> None:
    project = tmp_path / "demo"
    subject = project / "sub-01"
    sidecar = subject / "func/sub-01_task-rest_bold.json"
    _write(subject / "func/sub-01_task-rest_bold.nii.gz", "header")
    _write(sidecar, '{"EchoTime": 0.01}')
    cache = MemoryCache()

    first = planning_cache.participant_source_manifest(cache, project, subject)
    _write(sidecar, '{"EchoTime": 0.02}')
    stat = sidecar.stat()
    os.utime(sidecar, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = planning_cache.participant_source_manifest(cache, project, subject)
    _write(subject / "func/sub-01_task-rest_run-02_bold.nii.gz", "other")
    third = planning_cache.participant_source_manifest(cache, project, subject)

    assert len({first, second, third}) == 3
