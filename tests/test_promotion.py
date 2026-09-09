"""Promotion reference rewriting rejects dependencies outside its transfer map."""

import pytest

from nro.orchestration.promotion import _relocate


def test_relocation_allows_accepting_development_parent(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text("<File>/dev/child/BIDS/demo/parcel.dlabel.nii</File>")
    _relocate(path, [("/dev/child/BIDS/demo", "/dev/parent/BIDS/demo")], "/dev/child")
    assert path.read_text() == "<File>/dev/parent/BIDS/demo/parcel.dlabel.nii</File>"


def test_unmapped_or_binary_references_cannot_be_promoted(tmp_path):
    path = tmp_path / "output"
    path.write_text("/dev/child/WORK/temporary")
    with pytest.raises(ValueError, match="Unresolved"):
        _relocate(path, [], "/dev/child")
    path.write_bytes(b"\xff/dev/child/WORK/temporary")
    with pytest.raises(ValueError, match="Binary"):
        _relocate(path, [], "/dev/child")
