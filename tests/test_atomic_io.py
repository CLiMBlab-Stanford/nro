import errno
from pathlib import Path

import pytest

from nro.engine.io import atomic_output_path, read_json


def test_json_read_retries_stale_network_file_handle(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "control.json"
    target.write_text('{"state": "running"}\n', encoding="utf-8")
    original = Path.read_text
    attempts = 0

    def flaky_read_text(path: Path, *args, **kwargs) -> str:
        nonlocal attempts
        if path == target and attempts < 2:
            attempts += 1
            raise OSError(errno.ESTALE, "Stale file handle")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    assert read_json(target) == {"state": "running"}
    assert attempts == 2


def test_atomic_output_preserves_previous_target_when_producer_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "large.nii.gz"
    target.write_bytes(b"previous-complete-artifact")

    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_output_path(target) as staged:
            assert staged.name.endswith(".nii.gz")
            assert staged.parent == target.parent
            staged.write_bytes(b"partial")
            raise RuntimeError("interrupted")

    assert target.read_bytes() == b"previous-complete-artifact"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_atomic_output_publishes_only_after_success(tmp_path: Path) -> None:
    target = tmp_path / "large.nii.gz"

    with atomic_output_path(target) as staged:
        staged.write_bytes(b"complete")
        assert not target.exists()

    assert target.read_bytes() == b"complete"


def test_atomic_output_rejects_empty_staged_file(tmp_path: Path) -> None:
    target = tmp_path / "large.nii.gz"

    with pytest.raises(RuntimeError, match="missing or empty"):
        with atomic_output_path(target) as staged:
            staged.touch()

    assert not target.exists()
