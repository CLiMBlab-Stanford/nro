"""Reusable runner-integrated neuroimaging operations."""

from __future__ import annotations

import logging
from pathlib import Path
from nro.orchestration.runner_graph import Step

from .images import copy_or_convert_nifti, nifti_volume_count
from .io import atomic_output_path, invalid_gzip_files


LOG = logging.getLogger(__name__)


def create_native_overlap_mask_step(
    *,
    images: tuple[Path, ...] | list[Path],
    out_mask: Path,
    erosion_voxels: int,
    minimum_voxels: int,
    force: bool,
) -> Step:
    """Create a step producing shared finite, nonzero image support."""
    def validate() -> tuple[bool, str]:
        import nibabel as nib
        import numpy as np

        try:
            count = int(
                np.count_nonzero(np.asarray(nib.load(str(out_mask)).dataobj) > 0)
            )
        except Exception as error:
            return False, f"Native overlap mask is unreadable: {error}"
        return (
            count >= int(minimum_voxels),
            f"Native-grid overlap mask contains {count} voxels; minimum is {minimum_voxels}.",
        )

    def create() -> None:
        import nibabel as nib
        import numpy as np
        from scipy.ndimage import binary_erosion

        loaded = [nib.load(str(path)) for path in images]
        first = loaded[0]
        shape = tuple(int(value) for value in first.shape[:3])
        overlap = np.ones(shape, dtype=bool)
        for path, image in zip(images, loaded):
            if (
                tuple(int(value) for value in image.shape[:3]) != shape
                or not np.allclose(image.affine, first.affine)
            ):
                raise SystemExit(f"Native overlap inputs are not in the same grid: {path}")
            data = np.asarray(image.dataobj)
            if data.ndim > 3:
                data = data[..., 0]
            overlap &= np.isfinite(data) & (
                np.abs(data) > np.finfo(np.float32).eps
            )
        if erosion_voxels > 0:
            overlap = binary_erosion(
                overlap,
                iterations=int(erosion_voxels),
                border_value=0,
            )
        count = int(overlap.sum())
        if count < int(minimum_voxels):
            raise SystemExit(
                f"Insufficient valid image overlap: {count} voxels after erosion; "
                f"minimum is {minimum_voxels}."
            )
        header = first.header.copy()
        header.set_data_dtype(np.uint8)
        with atomic_output_path(out_mask) as staged:
            nib.save(
                nib.Nifti1Image(overlap.astype(np.uint8), first.affine, header),
                str(staged),
            )
        LOG.info("Native-grid image overlap mask contains %d voxels", count)

    return Step.python(
        name="Native-Grid Image Overlap Mask",
        outputs=(out_mask,),
        inputs=tuple(images),
        force=force,
        action=create,
        validate=validate,
    )


def create_image_support_mask_step(
    *,
    image: Path,
    out_mask: Path,
    erosion_voxels: int,
    minimum_voxels: int,
    force: bool,
) -> Step:
    """Create a step producing one image's finite, nonzero support."""
    def validate() -> tuple[bool, str]:
        import nibabel as nib
        import numpy as np

        try:
            count = int(
                np.count_nonzero(np.asarray(nib.load(str(out_mask)).dataobj) > 0)
            )
        except Exception as error:
            return False, f"Image support mask is unreadable: {error}"
        return (
            count >= int(minimum_voxels),
            f"Image support mask contains {count} voxels; minimum is {minimum_voxels}.",
        )

    def create() -> None:
        import nibabel as nib
        import numpy as np
        from scipy.ndimage import binary_erosion

        reference = nib.load(str(image))
        data = np.asarray(reference.dataobj)
        if data.ndim > 3:
            data = data[..., 0]
        support = np.isfinite(data) & (
            np.abs(data) > np.finfo(np.float32).eps
        )
        if erosion_voxels > 0:
            support = binary_erosion(
                support,
                iterations=int(erosion_voxels),
                border_value=0,
            )
        count = int(support.sum())
        if count < int(minimum_voxels):
            raise SystemExit(
                f"Insufficient valid image support: {count} voxels; "
                f"minimum is {minimum_voxels}."
            )
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        with atomic_output_path(out_mask) as staged:
            nib.save(
                nib.Nifti1Image(support.astype(np.uint8), reference.affine, header),
                str(staged),
            )
        LOG.info("Image support mask contains %d voxels", count)

    return Step.python(
        name="Image Support Mask",
        outputs=(out_mask,),
        inputs=(image,),
        force=force,
        action=create,
        validate=validate,
    )


def create_copy_nifti_step(
    *,
    src: Path,
    dst: Path,
    force: bool,
    step_name: str = "Finalize Derivative",
) -> Step:
    """Create an atomic NIfTI-copy step."""
    return Step.python(
        name=step_name,
        outputs=(dst,),
        inputs=(src,),
        force=force,
        action=lambda: copy_or_convert_nifti(src, dst),
    )


def create_mask_resampling_step(
    *,
    src_mask: Path,
    ref_img: Path,
    out_mask: Path,
    force: bool,
) -> Step:
    """Create a nearest-neighbor mask-resampling step."""
    def resample() -> None:
        import nibabel as nib
        from nibabel.processing import resample_from_to

        source = nib.load(str(src_mask))
        reference = nib.load(str(ref_img))
        resampled = resample_from_to(
            source,
            (reference.shape[:3], reference.affine),
            order=0,
        )
        data = (resampled.get_fdata(dtype="float32") > 0.0).astype("uint8")
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(data, reference.affine, reference.header),
            str(out_mask),
        )

    return Step.python(
        name="Mask Resampling",
        outputs=(out_mask,),
        inputs=(src_mask, ref_img),
        force=force,
        action=resample,
    )


def create_nifti_volume_extraction_step(
    *,
    img: Path,
    index_zero_based: int,
    out_3d: Path,
    env: dict[str, str],
    force: bool,
    label: str | None = None,
) -> Step:
    """Create a step extracting one selected NIfTI volume."""
    index = int(index_zero_based)
    display_label = label or "NIfTI volume extraction"
    if index < 0:
        raise SystemExit(f"{display_label} volume index must be nonnegative (got {index})")
    volume_count = nifti_volume_count(img)
    if volume_count <= 1:
        if index != 0:
            raise SystemExit(f"{display_label} requested volume {index} from a 3D image: {img}")
        return Step.python(
            name=display_label,
            outputs=(out_3d,),
            inputs=(img,),
            force=force,
            action=lambda: copy_or_convert_nifti(img, out_3d),
        )
    if index >= volume_count:
        raise SystemExit(
            f"{display_label} requested volume {index} but {img} has {volume_count} volumes"
        )
    return Step.command_step(
        ["fslroi", str(img), str(out_3d), str(index), "1"],
        name=label,
        outputs=(out_3d,),
        inputs=(img,),
        force=force,
        env=env,
        prepare=lambda: out_3d.parent.mkdir(parents=True, exist_ok=True),
    )


def create_flirt_transform_step(
    *,
    in_img: Path,
    ref_img: Path,
    mat: Path,
    out_img: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Create a trilinear FSL affine-application step."""
    return Step.command_step(
        [
            "flirt",
            "-in",
            str(in_img),
            "-ref",
            str(ref_img),
            "-applyxfm",
            "-init",
            str(mat),
            "-out",
            str(out_img),
            "-interp",
            "trilinear",
        ],
        outputs=(out_img,),
        inputs=(in_img, ref_img, mat),
        force=force,
        env=env,
        prepare=lambda: out_img.parent.mkdir(parents=True, exist_ok=True),
    )


def create_n4_bias_correction_step(
    *,
    in_img: Path,
    out_img: Path,
    env: dict[str, str],
    force: bool,
    mask: Path | None = None,
    validate_gzip: bool = False,
    step_name: str | None = None,
) -> Step:
    """Create a tracked N4 bias-correction step."""
    command = [
        "N4BiasFieldCorrection",
        "-d",
        "3",
        "-i",
        str(in_img),
        "-o",
        str(out_img),
    ]
    inputs = [in_img]
    if mask is not None:
        command.extend(["-x", str(mask)])
        inputs.append(mask)

    def validate() -> tuple[bool, str]:
        invalid = invalid_gzip_files((out_img,))
        if invalid:
            return False, f"N4 output is not a readable gzip NIfTI: {invalid[0]}"
        return True, "N4 output is readable."

    return Step.command_step(
        command,
        name=step_name,
        outputs=(out_img,),
        inputs=tuple(inputs),
        force=force,
        env=env,
        prepare=lambda: (
            out_img.parent.mkdir(parents=True, exist_ok=True),
            out_img.unlink(missing_ok=True),
        ),
        validate=validate if validate_gzip else None,
    )


def create_identity_transform_step(path: Path) -> Step:
    """Create a step writing a four-by-four identity affine transform."""
    content = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"

    def validate() -> tuple[bool, str]:
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            return False, "Identity transform is missing or unreadable."
        return (
            current == content,
            "Identity transform has the expected coefficients."
            if current == content
            else "Identity transform coefficients are incorrect.",
        )

    def write_identity() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return Step.python(
        name="Write Identity Transform",
        outputs=(path,),
        action=write_identity,
        validate=validate,
    )
