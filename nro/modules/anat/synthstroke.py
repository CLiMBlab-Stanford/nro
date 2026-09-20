"""Run pinned SynthStroke inference and restore results to the source T1w grid.

The model architecture and inference policy follow SynthStroke revision
``f627b57b8f9291bfa870cb54cb21ddbeb4941d39``. Model configuration and weights
are installed centrally and checked before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

from nro.modules.anat.lesion_policy import (
    MASKER_CONFIG_SHA256,
    MASKER_MODEL,
    MASKER_PATCH_SIZE,
    MASKER_REVISION,
    MASKER_VOXEL_SIZE_MM,
    MASKER_WEIGHTS_SHA256,
    MASKER_WINDOW_OVERLAP,
    PROBABILITY_THRESHOLD,
)

_FLIPS = ((), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_stack() -> tuple[Any, Any, Any]:
    try:
        import monai
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "SynthStroke support is not installed. Run ./install --with-lesion, or "
            "./install --maintain --with-lesion for a shared installation."
        ) from error
    return torch, monai, load_file


def _model_files(model_directory: Path, model_id: str, revision: str) -> tuple[Path, Path]:
    """Resolve and verify the model revision supported by this adapter."""
    if model_id != MASKER_MODEL or revision != MASKER_REVISION:
        raise RuntimeError("The built-in SynthStroke adapter accepts only its pinned model")
    config = Path(model_directory) / "config.json"
    weights = Path(model_directory) / "model.safetensors"
    expected = {config: MASKER_CONFIG_SHA256, weights: MASKER_WEIGHTS_SHA256}
    for path, digest in expected.items():
        if not path.is_file():
            raise RuntimeError(
                f"SynthStroke resource is missing: {path}; run ./install --with-lesion"
            )
        if _sha256(path) != digest:
            raise RuntimeError(
                f"SynthStroke resource checksum does not match the pinned model: {path}"
            )
    return config, weights


def _build_model(config_path: Path, weights_path: Path, monai: Any, load_file: Any) -> Any:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "spatial_dims",
        "in_channels",
        "out_channels",
        "channels",
        "strides",
        "kernel_size",
        "up_kernel_size",
        "num_res_units",
        "act",
        "norm",
        "dropout",
        "bias",
        "adn_ordering",
    }
    missing = required - config.keys()
    if missing:
        raise RuntimeError(f"SynthStroke model configuration is missing {sorted(missing)}")
    if config["in_channels"] != 1 or config["out_channels"] not in {2, 6}:
        raise RuntimeError("SynthStroke model must accept one channel and emit two or six classes")
    model = monai.networks.nets.UNet(**{key: config[key] for key in required})
    state = load_file(str(weights_path), device="cpu")
    state = {key.removeprefix("unet."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def _prepare_image(source: nib.spatialimages.SpatialImage, monai: Any) -> nib.Nifti1Image:
    canonical = nib.as_closest_canonical(source)
    working = resample_to_output(
        canonical,
        voxel_sizes=(MASKER_VOXEL_SIZE_MM,) * 3,
        order=1,
    )
    data = np.asarray(working.dataobj, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("SynthStroke input must be one finite 3D image")
    transform = monai.transforms.Compose(
        [
            monai.transforms.HistogramNormalize(),
            monai.transforms.NormalizeIntensity(nonzero=False, channel_wise=True),
        ]
    )
    normalized = transform(data[np.newaxis, ...])
    normalized = np.asarray(normalized, dtype=np.float32)[0]
    return nib.Nifti1Image(normalized, working.affine)


def _predict_probability(
    image: nib.Nifti1Image,
    model: Any,
    torch: Any,
    monai: Any,
    *,
    use_tta: bool,
    device_name: str,
) -> np.ndarray:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SynthStroke requested CUDA, but PyTorch cannot access a CUDA device")
    device = torch.device(device_name)
    model = model.to(device).eval()
    data = np.asarray(image.dataobj, dtype=np.float32)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    def predict(value: Any) -> Any:
        return monai.inferers.sliding_window_inference(
            value,
            roi_size=(MASKER_PATCH_SIZE,) * 3,
            sw_batch_size=1,
            predictor=model,
            overlap=MASKER_WINDOW_OVERLAP,
            mode="gaussian",
        )

    with torch.inference_mode():
        logits = None
        flips = _FLIPS if use_tta else ((),)
        for dimensions in flips:
            augmented = torch.flip(tensor, dimensions) if dimensions else tensor
            prediction = predict(augmented)
            if dimensions:
                prediction = torch.flip(prediction, dimensions)
            logits = prediction if logits is None else logits + prediction
        probabilities = torch.softmax(logits / len(flips), dim=1)
        lesion_class = 5 if probabilities.shape[1] == 6 else 1
        result = probabilities[0, lesion_class].detach().cpu().numpy()
    return np.asarray(result, dtype=np.float32)


def _restore_source_grid(
    probability: np.ndarray,
    working: nib.Nifti1Image,
    source: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    image = nib.Nifti1Image(probability, working.affine)
    restored = resample_from_to(image, (source.shape[:3], source.affine), order=1)
    values = np.asarray(restored.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("SynthStroke produced nonfinite probabilities")
    return np.clip(values, 0.0, 1.0)


def _save_image(
    values: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = source.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(values.astype(dtype), source.affine, header=header)
    image.set_qform(source.get_qform(), int(source.header["qform_code"]))
    image.set_sform(source.get_sform(), int(source.header["sform_code"]))
    nib.save(image, str(path))


def run(
    *,
    source_path: Path,
    probability_path: Path,
    mask_path: Path,
    model_directory: Path,
    model_id: str,
    revision: str,
    threshold: float,
    use_tta: bool,
    device: str,
) -> None:
    """Estimate a lesion probability map and binary mask on the source image grid."""
    if not 0 < threshold < 1:
        raise ValueError("--threshold must be strictly between zero and one")
    torch, monai, load_file = _optional_stack()
    config_path, weights_path = _model_files(model_directory, model_id, revision)
    model = _build_model(config_path, weights_path, monai, load_file)
    source = nib.load(str(source_path))
    working = _prepare_image(source, monai)
    probability = _predict_probability(
        working,
        model,
        torch,
        monai,
        use_tta=use_tta,
        device_name=device,
    )
    probability = _restore_source_grid(probability, working, source)
    mask = probability >= threshold
    _save_image(probability, source, probability_path, np.dtype(np.float32))
    _save_image(mask, source, mask_path, np.dtype(np.uint8))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nro-synthstroke",
        description="Run nro's pinned SynthStroke lesion-mask adapter.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--probability", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model", default=MASKER_MODEL)
    parser.add_argument("--revision", default=MASKER_REVISION)
    parser.add_argument("--threshold", default=PROBABILITY_THRESHOLD, type=float)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse adapter arguments and run lesion-mask inference."""
    args = _parser().parse_args(argv)
    run(
        source_path=args.input,
        probability_path=args.probability,
        mask_path=args.mask,
        model_directory=args.model_directory,
        model_id=args.model,
        revision=args.revision,
        threshold=args.threshold,
        use_tta=args.tta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
