from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from nro.modules.anat import synthstroke
from nro.modules.anat.lesion_policy import MASKER_MODEL, MASKER_REVISION


def _image(path: Path) -> nib.Nifti1Image:
    affine = np.diag([1.2, 1.3, 1.4, 1.0])
    image = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, path)
    return image


def test_adapter_restores_probability_and_mask_to_source_grid(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.nii.gz"
    source = _image(source_path)
    probability_path = tmp_path / "probability.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    working = nib.Nifti1Image(np.zeros(source.shape, dtype=np.float32), source.affine)

    monkeypatch.setattr(synthstroke, "_optional_stack", lambda: (object(), object(), None))
    monkeypatch.setattr(
        synthstroke,
        "_model_files",
        lambda *args: (tmp_path / "config.json", tmp_path / "weights.safetensors"),
    )
    monkeypatch.setattr(synthstroke, "_build_model", lambda *args: object())
    monkeypatch.setattr(synthstroke, "_prepare_image", lambda *args: working)
    monkeypatch.setattr(
        synthstroke,
        "_predict_probability",
        lambda *args, **kwargs: np.full(source.shape, 0.75, dtype=np.float32),
    )

    synthstroke.run(
        source_path=source_path,
        probability_path=probability_path,
        mask_path=mask_path,
        model_directory=tmp_path / "model",
        model_id=MASKER_MODEL,
        revision=MASKER_REVISION,
        threshold=0.5,
        use_tta=True,
        device="cpu",
    )

    probability = nib.load(probability_path)
    mask = nib.load(mask_path)
    assert probability.shape == mask.shape == source.shape
    assert np.allclose(probability.affine, source.affine)
    assert np.allclose(mask.affine, source.affine)
    assert np.allclose(np.asarray(probability.dataobj), 0.75)
    assert np.array_equal(np.asarray(mask.dataobj), np.ones(source.shape))


def test_adapter_uses_checksum_verified_local_model(tmp_path, monkeypatch) -> None:
    config = tmp_path / "model/config.json"
    weights = tmp_path / "model/model.safetensors"
    config.parent.mkdir()
    config.write_text("{}")
    weights.write_bytes(b"weights")
    digests = {
        config: synthstroke.MASKER_CONFIG_SHA256,
        weights: synthstroke.MASKER_WEIGHTS_SHA256,
    }
    monkeypatch.setattr(synthstroke, "_sha256", digests.__getitem__)

    assert synthstroke._model_files(config.parent, MASKER_MODEL, MASKER_REVISION) == (
        config,
        weights,
    )
    with pytest.raises(RuntimeError, match="only its pinned model"):
        synthstroke._model_files(config.parent, "example/model", MASKER_REVISION)


def test_adapter_rejects_invalid_threshold(tmp_path) -> None:
    with pytest.raises(ValueError, match="threshold"):
        synthstroke.run(
            source_path=tmp_path / "source.nii.gz",
            probability_path=tmp_path / "probability.nii.gz",
            mask_path=tmp_path / "mask.nii.gz",
            model_directory=tmp_path / "model",
            model_id=MASKER_MODEL,
            revision=MASKER_REVISION,
            threshold=1.0,
            use_tta=False,
            device="cpu",
        )
