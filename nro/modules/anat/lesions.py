"""Lesion-mask, reconstruction, and compact-surface helpers for anatomy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from nro.engine.io import atomic_write_text
from nro.modules.anat.lesion_policy import (
    MASKER_CONFIG_SHA256,
    MASKER_PATCH_SIZE,
    MASKER_SOURCE_REVISION,
    MASKER_VOXEL_SIZE_MM,
    MASKER_WEIGHTS_SHA256,
    MASKER_WINDOW_OVERLAP,
    NEUROLIT_CHECKPOINTS,
    NEUROLIT_VERSION,
)
from nro.orchestration.runner_graph import Step


def validate_lesion_mask(mask: Path, reference: Path) -> dict[str, object]:
    """Validate a binary lesion mask and return compact geometry statistics."""
    mask_image = nib.load(str(mask))
    reference_image = nib.load(str(reference))
    data = np.asanyarray(mask_image.dataobj)
    if data.shape != reference_image.shape[:3] or not np.allclose(
        mask_image.affine, reference_image.affine, atol=1e-4
    ):
        raise ValueError("Lesion mask does not share the selected T1w image grid")
    if not np.isfinite(data).all():
        raise ValueError("Lesion mask contains nonfinite values")
    lesion = data > 0
    count = int(lesion.sum())
    if count == 0:
        raise ValueError("Automatic lesion masking produced an empty mask")
    fraction = count / float(lesion.size)
    if fraction > 0.35:
        raise ValueError(f"Lesion mask occupies an implausible {fraction:.1%} of the image grid")
    reference_data = np.asanyarray(reference_image.dataobj)
    brain = np.isfinite(reference_data) & (reference_data != 0)
    outside_count = int((lesion & ~brain).sum())
    outside_fraction = outside_count / float(count)
    voxel_volume = float(abs(np.linalg.det(mask_image.affine[:3, :3])))
    from scipy import ndimage

    components, component_count = ndimage.label(lesion)
    component_voxels = np.bincount(components.reshape(-1))[1:]
    world_x = nib.affines.apply_affine(mask_image.affine, np.argwhere(lesion))[:, 0]
    return {
        "voxel_count": count,
        "volume_mm3": count * voxel_volume,
        "component_count": int(component_count),
        "component_volumes_mm3": sorted(
            (float(value) * voxel_volume for value in component_voxels), reverse=True
        ),
        "image_fraction": fraction,
        "outside_brain_fraction": outside_fraction,
        "laterality": ("left" if float(np.median(world_x)) < 0 else "right"),
    }


def validate_lesion_probability(
    probability: Path, mask: Path, reference: Path, threshold: float
) -> None:
    """Require a finite unit-interval score map consistent with the binary mask."""
    probability_image = nib.load(str(probability))
    mask_image = nib.load(str(mask))
    reference_image = nib.load(str(reference))
    if probability_image.shape != reference_image.shape[:3] or not np.allclose(
        probability_image.affine, reference_image.affine, atol=1e-4
    ):
        raise ValueError("Lesion probability image does not share the selected T1w grid")
    values = np.asanyarray(probability_image.dataobj)
    if not np.isfinite(values).all() or float(values.min()) < 0 or float(values.max()) > 1:
        raise ValueError("Lesion probabilities must be finite values between zero and one")
    binary = np.asanyarray(mask_image.dataobj) > 0
    if not np.array_equal(values >= threshold, binary):
        raise ValueError("Lesion mask does not equal the thresholded probability image")


def create_lesion_mask_step(
    *,
    run_child: Callable[..., Any],
    command: Path,
    source: Path,
    probability: Path,
    mask: Path,
    metadata: Path,
    model_directory: Path | None,
    model: str,
    revision: str,
    threshold: float,
    test_time_augmentation: bool,
    use_gpu: bool,
    force: bool,
    module: str | None = None,
) -> Step:
    """Create a fixed automatic-mask step through the configured SynthStroke adapter."""

    def action() -> None:
        probability.parent.mkdir(parents=True, exist_ok=True)
        partial_probability = probability.with_name(f".partial-{probability.name}")
        partial_mask = mask.with_name(f".partial-{mask.name}")
        partial_probability.unlink(missing_ok=True)
        partial_mask.unlink(missing_ok=True)
        invocation = [
            str(command),
            *(["-m", module] if module else []),
            "--input",
            str(source),
            "--probability",
            str(partial_probability),
            "--mask",
            str(partial_mask),
            *(["--model-directory", str(model_directory)] if model_directory is not None else []),
            "--model",
            model,
            "--revision",
            revision,
            "--threshold",
            str(threshold),
            "--device",
            "cuda" if use_gpu else "cpu",
        ]
        if test_time_augmentation:
            invocation.append("--tta")
        run_child(invocation, direct=True, stream_output=True)
        summary = validate_lesion_mask(partial_mask, source)
        validate_lesion_probability(partial_probability, partial_mask, source, threshold)
        os.replace(partial_probability, probability)
        os.replace(partial_mask, mask)
        atomic_write_text(
            metadata,
            json.dumps(
                {
                    "Type": "automatic stroke lesion mask",
                    "Sources": [str(source)],
                    "Model": model,
                    "ModelRevision": revision,
                    "ModelConfigurationSHA256": MASKER_CONFIG_SHA256,
                    "ModelWeightsSHA256": MASKER_WEIGHTS_SHA256,
                    "ImplementationRevision": MASKER_SOURCE_REVISION,
                    "InferenceVoxelSizeMM": MASKER_VOXEL_SIZE_MM,
                    "SlidingWindowPatchSize": MASKER_PATCH_SIZE,
                    "SlidingWindowOverlap": MASKER_WINDOW_OVERLAP,
                    "ProbabilityThreshold": threshold,
                    "TestTimeAugmentation": test_time_augmentation,
                    **summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def validate() -> tuple[bool, str]:
        try:
            validate_lesion_mask(mask, source)
            validate_lesion_probability(probability, mask, source, threshold)
            document = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            return False, f"Lesion mask is absent or invalid: {error}"
        if (
            document.get("Model") != model
            or document.get("ModelRevision") != revision
            or document.get("ImplementationRevision") != MASKER_SOURCE_REVISION
            or document.get("ModelConfigurationSHA256") != MASKER_CONFIG_SHA256
            or document.get("ModelWeightsSHA256") != MASKER_WEIGHTS_SHA256
        ):
            return False, "Lesion mask model provenance differs from the request."
        return True, "Automatic lesion mask and provenance are valid."

    return Step.python(
        name="Automatic Lesion Masking",
        inputs=(source,),
        outputs=(probability, mask, metadata),
        action=action,
        validate=validate,
        force=force,
        parameters={
            "backend": "SynthStroke",
            "adapter_module": module,
            "model": model,
            "revision": revision,
            "implementation_revision": MASKER_SOURCE_REVISION,
            "inference_voxel_size_mm": MASKER_VOXEL_SIZE_MM,
            "sliding_window_patch_size": MASKER_PATCH_SIZE,
            "sliding_window_overlap": MASKER_WINDOW_OVERLAP,
            "probability_threshold": threshold,
            "test_time_augmentation": test_time_augmentation,
        },
    )


def create_lesion_qc_step(
    *,
    source: Path,
    mask: Path,
    output: Path,
    force: bool,
) -> Step:
    """Render orthogonal source slices with the automatic lesion mask overlaid."""

    def action() -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        source_image = nib.as_closest_canonical(nib.load(str(source)))
        mask_image = nib.as_closest_canonical(nib.load(str(mask)))
        anatomy = np.asanyarray(source_image.dataobj, dtype=np.float32)
        lesion = np.asanyarray(mask_image.dataobj) > 0
        if anatomy.shape != lesion.shape:
            raise ValueError("Lesion QC inputs do not share a canonical grid")
        center = np.rint(np.argwhere(lesion).mean(axis=0)).astype(int)
        finite = anatomy[np.isfinite(anatomy)]
        nonzero = finite[finite != 0]
        display = nonzero if nonzero.size else finite
        lower, upper = np.percentile(display, (1, 99))
        if not upper > lower:
            upper = lower + 1.0
        labels = ("sagittal", "coronal", "axial")
        figure, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        for axis, index, label in zip(axes, center, labels, strict=True):
            anatomical_slice = np.rot90(np.take(anatomy, int(index), axis=labels.index(label)))
            lesion_slice = np.rot90(np.take(lesion, int(index), axis=labels.index(label)))
            axis.imshow(anatomical_slice, cmap="gray", vmin=lower, vmax=upper)
            axis.imshow(np.ma.masked_where(~lesion_slice, lesion_slice), cmap="autumn", alpha=0.55)
            axis.set_title(label)
            axis.axis("off")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".partial-{output.name}")
        temporary.unlink(missing_ok=True)
        figure.savefig(temporary, dpi=150, format="png")
        plt.close(figure)
        os.replace(temporary, output)

    def validate() -> tuple[bool, str]:
        try:
            signature = output.read_bytes()[:8]
        except OSError:
            return False, "Lesion QC image is absent or unreadable."
        if signature != b"\x89PNG\r\n\x1a\n":
            return False, "Lesion QC image is not a PNG file."
        return True, "Lesion QC image is readable."

    return Step.python(
        name="Render Lesion Mask QC",
        inputs=(source, mask),
        outputs=(output,),
        action=action,
        validate=validate,
        force=force,
        parameters={"views": ["sagittal", "coronal", "axial"], "overlay": "lesion_mask"},
    )


def create_lesion_excluded_mask_step(
    *,
    scaffold_mask: Path,
    lesion_mask: Path,
    output: Path,
    force: bool,
) -> Step:
    """Remove the lesion domain from one scaffold-derived volumetric mask."""

    def action() -> None:
        scaffold_image = nib.load(str(scaffold_mask))
        lesion_image = nib.load(str(lesion_mask))
        if scaffold_image.shape[:3] != lesion_image.shape[:3] or not np.allclose(
            scaffold_image.affine, lesion_image.affine, atol=1e-4
        ):
            raise ValueError("Scaffold and lesion masks do not share a spatial grid")
        scaffold = np.asanyarray(scaffold_image.dataobj) > 0
        lesion = np.asanyarray(lesion_image.dataobj) > 0
        values = np.asarray(scaffold & ~lesion, dtype=np.uint8)
        header = scaffold_image.header.copy()
        header.set_data_dtype(np.uint8)
        result = nib.Nifti1Image(values, scaffold_image.affine, header=header)
        result.set_qform(scaffold_image.get_qform(), int(scaffold_image.header["qform_code"]))
        result.set_sform(scaffold_image.get_sform(), int(scaffold_image.header["sform_code"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".partial-{output.name}")
        temporary.unlink(missing_ok=True)
        nib.save(result, str(temporary))
        os.replace(temporary, output)

    def validate() -> tuple[bool, str]:
        try:
            output_image = nib.load(str(output))
            scaffold_image = nib.load(str(scaffold_mask))
            lesion_image = nib.load(str(lesion_mask))
            values = np.asanyarray(output_image.dataobj)
            expected = (np.asanyarray(scaffold_image.dataobj) > 0) & ~(
                np.asanyarray(lesion_image.dataobj) > 0
            )
        except (OSError, ValueError) as error:
            return False, f"Lesion-excluded mask is absent or invalid: {error}"
        if output_image.shape[:3] != scaffold_image.shape[:3] or not np.allclose(
            output_image.affine, scaffold_image.affine, atol=1e-4
        ):
            return False, "Lesion-excluded mask does not preserve the scaffold grid."
        if not np.array_equal(values > 0, expected):
            return False, "Lesion-excluded mask differs from scaffold minus lesion."
        return True, "Lesion-excluded mask matches scaffold minus lesion."

    return Step.python(
        name="Exclude Lesion from Anatomical Mask",
        inputs=(scaffold_mask, lesion_mask),
        outputs=(output,),
        action=action,
        validate=validate,
        force=force,
        parameters={"operation": "scaffold_and_not_lesion"},
    )


@dataclass(frozen=True)
class LesionInpaintingPlan:
    """Describe NeuroLIT inpainting and its reusable private products."""

    step: Step
    image: Path
    brain_mask: Path
    original_mask: Path


def create_neurolit_inpainting_plan(
    *,
    run_child: Callable[..., Any],
    runtime: str,
    image: Path,
    data_directory: Path,
    t1w: Path,
    lesion_mask: Path,
    subjects_dir: Path,
    subject: str,
    use_gpu: bool,
    force: bool,
) -> LesionInpaintingPlan:
    """Create NeuroLIT inpainting without selecting a surface backend."""
    inpaint_dir = subjects_dir / ".nro-inpainting" / subject
    inpainted = inpaint_dir / "mri" / "inpainted.lit.nii.gz"
    inpaint_mask = inpaint_dir / "mri" / "mask.lit.nii.gz"
    original_mask = inpaint_dir / "mri" / "orig" / "mask.lit.nii.gz"
    checkpoints = tuple(data_directory / "LIT" / "weights" / name for name in NEUROLIT_CHECKPOINTS)

    def container_prefix() -> list[str]:
        subjects_dir.mkdir(parents=True, exist_ok=True)
        container = [runtime, "exec"]
        if use_gpu:
            container.append("--nv")
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if not visible_devices:
                raise RuntimeError(
                    "GPU lesion inpainting requires a Slurm-assigned CUDA_VISIBLE_DEVICES value"
                )
        else:
            visible_devices = None
        container.extend(
            [
                "--cleanenv",
                "--bind",
                f"{t1w.parent}:/input:ro",
                "--bind",
                f"{lesion_mask.parent}:/lesion:ro",
                "--bind",
                f"{subjects_dir}:/output",
                "--bind",
                f"{data_directory}:/nro-lit-data:ro",
                "--env",
                "XDG_DATA_HOME=/nro-lit-data",
            ]
        )
        if visible_devices is not None:
            container.extend(["--env", f"CUDA_VISIBLE_DEVICES={visible_devices}"])
        container.append(str(image))
        return container

    def inpaint_action() -> None:
        if inpaint_dir.exists():
            shutil.rmtree(inpaint_dir)
        device = "cuda" if use_gpu else "cpu"
        run_child(
            [
                *container_prefix(),
                "python3",
                "-s",
                "-m",
                "neurolit.cli",
                "--input_image",
                f"/input/{t1w.name}",
                "--lesion_mask",
                f"/lesion/{lesion_mask.name}",
                "--sd",
                f"/output/.nro-inpainting/{subject}",
                "--fastsurfer_dir",
                "--device",
                device,
                "--batch_size",
                "8",
            ],
            direct=True,
            stream_output=True,
        )

    def inpaint_valid() -> tuple[bool, str]:
        missing = [
            str(path)
            for path in (inpainted, inpaint_mask, original_mask)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "NeuroLIT outputs are incomplete: " + ", ".join(missing)
        return True, "NeuroLIT inpainting outputs are complete."

    inpainting_scientific = {
        "backend": "NeuroLIT",
        "neurolit_version": NEUROLIT_VERSION,
        "neurolit_checkpoint_sha256": NEUROLIT_CHECKPOINTS,
        "neurolit_batch_size": 8,
        "stage": "inpainting",
    }
    return LesionInpaintingPlan(
        step=Step.python(
            id="neurolit-inpainting",
            name="NeuroLIT Lesion Inpainting",
            inputs=(t1w, lesion_mask, image, *checkpoints),
            outputs=(inpainted, inpaint_mask, original_mask),
            action=inpaint_action,
            validate=inpaint_valid,
            force=force,
            resource_class="gpu" if use_gpu else None,
            parameters=inpainting_scientific,
        ),
        image=inpainted,
        brain_mask=inpaint_mask,
        original_mask=original_mask,
    )


def create_lesion_reconstruction_summary_step(
    *,
    subject_dir: Path,
    lesion_mask: Path,
    surface_reconstruction: Mapping[str, object],
    output: Path,
    force: bool,
) -> Step:
    """Record the intact scaffold used to derive lesion-excised surfaces."""
    reconciliation = subject_dir / "stats" / "nro-mask-reconciliation.json"
    required = (
        subject_dir / "mri" / "aseg.mgz",
        subject_dir / "mri" / "ribbon.mgz",
        subject_dir / "surf" / "lh.white",
        subject_dir / "surf" / "rh.white",
    )

    def payload() -> dict[str, object]:
        result: dict[str, object] = {
            "Type": "synthetic intact surface scaffold",
            "SurfaceReconstruction": dict(surface_reconstruction),
            "LesionMask": str(lesion_mask),
            "CompleteScaffold": True,
        }
        if surface_reconstruction.get("backend") == "FastSurfer":
            result["MaskReconciliation"] = json.loads(
                reconciliation.read_text(encoding="utf-8")
            )
        return result

    def action() -> None:
        atomic_write_text(output, yaml.safe_dump(payload(), sort_keys=False))

    def validate() -> tuple[bool, str]:
        try:
            current = yaml.safe_load(output.read_text(encoding="utf-8"))
            expected = payload()
        except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError) as error:
            return False, f"Lesion reconstruction summary is absent or invalid: {error}"
        if current != expected:
            return False, "Lesion reconstruction summary differs from its source evidence."
        return True, "Lesion reconstruction summary matches its source evidence."

    inputs = (*required, lesion_mask)
    if surface_reconstruction.get("backend") == "FastSurfer":
        inputs = (*inputs, reconciliation)
    return Step.python(
        name="Write Lesion Reconstruction Summary",
        inputs=inputs,
        outputs=(output,),
        action=action,
        validate=validate,
        force=force,
        parameters={"surface_reconstruction": dict(surface_reconstruction)},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_inpainted_metadata_step(
    *,
    image: Path,
    observed_t1w: Path,
    lesion_mask: Path,
    lesion_metadata: Path,
    output: Path,
    force: bool,
) -> Step:
    """Describe the synthetic image and bind it to its exact lesion mask."""

    def payload() -> dict[str, object]:
        mask_record = json.loads(lesion_metadata.read_text(encoding="utf-8"))
        return {
            "Type": "synthetic lesion-inpainted anatomical image",
            "Sources": [str(observed_t1w), str(lesion_mask)],
            "SyntheticTissue": True,
            "Method": "NeuroLIT",
            "SoftwareVersion": NEUROLIT_VERSION,
            "LesionMaskSHA256": _sha256(lesion_mask),
            "LesionMaskModel": mask_record.get("Model"),
            "LesionMaskModelRevision": mask_record.get("ModelRevision"),
            "Description": (
                "Intact-brain computational alternative for registration and surface "
                "reconstruction; not observed anatomy."
            ),
        }

    def action() -> None:
        atomic_write_text(output, json.dumps(payload(), indent=2, sort_keys=True) + "\n")

    def validate() -> tuple[bool, str]:
        try:
            current = json.loads(output.read_text(encoding="utf-8"))
            expected = payload()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            return False, f"Inpainted-anatomy metadata is absent or invalid: {error}"
        if current != expected:
            return False, "Inpainted-anatomy metadata differs from its source evidence."
        return True, "Inpainted-anatomy metadata matches its source evidence."

    return Step.python(
        name="Write Inpainted Anatomical Metadata",
        inputs=(image, observed_t1w, lesion_mask, lesion_metadata),
        outputs=(output,),
        action=action,
        validate=validate,
        force=force,
        parameters={"method": "NeuroLIT", "synthetic_tissue": True},
    )


def _surface_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, nib.GiftiImage]:
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError(f"Expected a GIFTI surface: {path}")
    points = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    triangles = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if len(points) != 1 or len(triangles) != 1:
        raise ValueError(f"Surface must contain one pointset and one triangle array: {path}")
    return np.asarray(points[0].data), np.asarray(triangles[0].data), image


def retained_surface_vertices(surface: Path, lesion_metric: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return retained scaffold indices and compact triangles after lesion removal."""
    points, triangles, _ = _surface_arrays(surface)
    metric_image = nib.load(str(lesion_metric))
    arrays = metric_image.darrays
    if len(arrays) != 1:
        raise ValueError(f"Lesion metric must contain one data array: {lesion_metric}")
    excluded = np.asarray(arrays[0].data).reshape(-1) > 0
    if excluded.shape[0] != points.shape[0]:
        raise ValueError("Lesion metric and scaffold surface have different vertex counts")
    surviving_faces = triangles[~excluded[triangles].any(axis=1)]
    retained = np.unique(surviving_faces.reshape(-1))
    if retained.size < 3 or surviving_faces.size == 0:
        raise ValueError("Lesion removal leaves no usable cortical surface")
    inverse = np.full(points.shape[0], -1, dtype=np.int64)
    inverse[retained] = np.arange(retained.size, dtype=np.int64)
    return retained, inverse[surviving_faces]


def _save_surface(source: Path, output: Path, retained: np.ndarray, triangles: np.ndarray) -> None:
    points, _, image = _surface_arrays(source)
    point_array = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0]
    triangle_array = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0]
    result = nib.GiftiImage(
        header=image.header,
        extra=image.extra,
        meta=image.meta,
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(points[retained], dtype=np.float32),
                intent=point_array.intent,
                datatype="NIFTI_TYPE_FLOAT32",
                coordsys=point_array.coordsys,
                meta=point_array.meta,
            ),
            nib.gifti.GiftiDataArray(
                np.asarray(triangles, dtype=np.int32),
                intent=triangle_array.intent,
                datatype="NIFTI_TYPE_INT32",
                meta=triangle_array.meta,
            ),
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".partial-{output.name}")
    temporary.unlink(missing_ok=True)
    nib.save(result, str(temporary))
    os.replace(temporary, output)


def _save_metric(source: Path, output: Path, retained: np.ndarray) -> None:
    image = nib.load(str(source))
    if not isinstance(image, nib.GiftiImage) or not image.darrays:
        raise ValueError(f"Expected a GIFTI metric: {source}")
    arrays = []
    for array in image.darrays:
        data = np.asarray(array.data)
        if data.shape[0] <= int(retained.max()):
            raise ValueError(f"Metric and scaffold surface have different vertex counts: {source}")
        arrays.append(
            nib.gifti.GiftiDataArray(
                np.asarray(data[retained]),
                intent=array.intent,
                datatype=array.datatype,
                coordsys=array.coordsys,
                meta=array.meta,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".partial-{output.name}")
    temporary.unlink(missing_ok=True)
    nib.save(
        nib.GiftiImage(header=image.header, extra=image.extra, meta=image.meta, darrays=arrays),
        str(temporary),
    )
    os.replace(temporary, output)


def create_cut_surfaces_step(
    *,
    scaffold_surface: Path,
    lesion_metric: Path,
    surfaces: Mapping[Path, Path],
    metrics: Mapping[Path, Path],
    mapping: Path,
    summary: Path,
    force: bool,
) -> Step:
    """Cut one hemisphere consistently and record public-to-scaffold indices."""
    outputs = (*surfaces.values(), *metrics.values(), mapping, summary)

    def action() -> None:
        retained, triangles = retained_surface_vertices(scaffold_surface, lesion_metric)
        for source, output in surfaces.items():
            _save_surface(source, output, retained, triangles)
        for source, output in metrics.items():
            _save_metric(source, output, retained)
        atomic_write_text(
            mapping,
            "public_vertex\tscaffold_vertex\n"
            + "".join(f"{public}\t{scaffold}\n" for public, scaffold in enumerate(retained)),
        )
        scaffold_points, scaffold_triangles, _ = _surface_arrays(scaffold_surface)
        atomic_write_text(
            summary,
            json.dumps(
                {
                    "Type": "surviving-cortex surface validity",
                    "ScaffoldSurface": str(scaffold_surface),
                    "LesionMetric": str(lesion_metric),
                    "VertexMapping": str(mapping),
                    "FaceExclusion": "any_lesion_vertex",
                    "ScaffoldVertexCount": int(scaffold_points.shape[0]),
                    "PublicVertexCount": int(retained.size),
                    "ExcludedVertexCount": int(scaffold_points.shape[0] - retained.size),
                    "ScaffoldFaceCount": int(scaffold_triangles.shape[0]),
                    "PublicFaceCount": int(triangles.shape[0]),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def validate() -> tuple[bool, str]:
        try:
            retained, triangles = retained_surface_vertices(scaffold_surface, lesion_metric)
            mapping_rows = mapping.read_text(encoding="utf-8").splitlines()
            record = json.loads(summary.read_text(encoding="utf-8"))
            for path in surfaces.values():
                points, faces, _ = _surface_arrays(path)
                if points.shape[0] != retained.size or not np.array_equal(faces, triangles):
                    raise ValueError(f"Cut surface has inconsistent compact topology: {path}")
            for path in metrics.values():
                image = nib.load(str(path))
                if not isinstance(image, nib.GiftiImage) or any(
                    np.asarray(array.data).shape[0] != retained.size for array in image.darrays
                ):
                    raise ValueError(f"Cut metric has inconsistent vertex count: {path}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            return False, f"Cut-surface publication is absent or invalid: {error}"
        expected_rows = ["public_vertex\tscaffold_vertex"] + [
            f"{public}\t{scaffold}" for public, scaffold in enumerate(retained)
        ]
        if mapping_rows != expected_rows:
            return False, "Cut-surface vertex mapping differs from the compact topology."
        if (
            record.get("PublicVertexCount") != int(retained.size)
            or record.get("PublicFaceCount") != int(triangles.shape[0])
            or record.get("VertexMapping") != str(mapping)
        ):
            return False, "Cut-surface validity summary differs from the compact topology."
        return True, "Cut surfaces, metrics, mapping, and validity summary agree."

    return Step.python(
        name="Cut Lesion from Cortical Surfaces",
        inputs=(scaffold_surface, lesion_metric, *surfaces.keys(), *metrics.keys()),
        outputs=outputs,
        action=action,
        validate=validate,
        force=force,
        parameters={"face_exclusion": "any_lesion_vertex", "compact_vertices": True},
    )
