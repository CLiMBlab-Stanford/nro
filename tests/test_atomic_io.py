from pathlib import Path

import pytest

from nro.engine.io import atomic_output_path


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
