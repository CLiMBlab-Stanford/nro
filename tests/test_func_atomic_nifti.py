from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.engine import images


def _write_nifti(path: Path, value: float) -> None:
    data = np.full((5, 4, 3, 2), value, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


def test_nifti_copy_does_not_replace_target_when_copy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "target.nii.gz"
    _write_nifti(source, 2.0)
    _write_nifti(target, 1.0)
    previous = target.read_bytes()

    def interrupted_copy(_source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(images.shutil, "copyfile", interrupted_copy)

    with pytest.raises(OSError, match="interrupted write"):
        images.copy_or_convert_nifti(source, target)

    assert target.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_nifti_copy_publishes_valid_result(tmp_path: Path) -> None:
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "target.nii.gz"
    _write_nifti(source, 3.0)

    images.copy_or_convert_nifti(source, target)

    assert images.nifti_is_valid(target)
    np.testing.assert_array_equal(
        np.asarray(nib.load(target).dataobj),
        np.asarray(nib.load(source).dataobj),
    )
