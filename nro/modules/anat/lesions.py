"""Lesion-mask, reconstruction, and compact-surface helpers for anatomy."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from nro.engine.io import atomic_write_text
from nro.engine.manifests import create_json_step
from nro.modules.anat.lesion_policy import (
    MASKER_CONFIG_SHA256,
    MASKER_PATCH_SIZE,
    MASKER_SOURCE_REVISION,
    MASKER_VOXEL_SIZE_MM,
    MASKER_WEIGHTS_SHA256,
    MASKER_WINDOW_OVERLAP,
    synthsr_scaffold_contract,
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
class LesionScaffoldPlan:
    """Describe whole-brain SynthSR synthesis and its private image."""

    step: Step
    image: Path


def create_synthsr_scaffold_plan(
    *,
    run_child: Callable[..., Any],
    runtime: str,
    image: Path,
    license_file: Path,
    t1w: Path,
    output: Path,
    threads: int,
    env: Mapping[str, str],
    force: bool,
) -> LesionScaffoldPlan:
    """Create a CPU SynthSR scaffold without selecting a surface backend."""

    def action() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = output.with_name(f".raw-{output.name}")
        clipped = output.with_name(f".partial-{output.name}")
        raw.unlink(missing_ok=True)
        clipped.unlink(missing_ok=True)
        try:
            run_child(
                [
                    runtime,
                    "exec",
                    "--cleanenv",
                    "--bind",
                    f"{t1w.parent}:/input:ro,{output.parent}:/output,{license_file}:/license.txt:ro",
                    str(image),
                    "bash",
                    "-lc",
                    (
                        "export FREESURFER_HOME=/usr/local/freesurfer; "
                        "export FS_LICENSE=/license.txt; "
                        'source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null; exec "$@"'
                    ),
                    "bash",
                    "mri_synthsr",
                    "--i",
                    f"/input/{t1w.name}",
                    "--o",
                    f"/output/{raw.name}",
                    "--threads",
                    str(max(1, threads)),
                    "--cpu",
                ],
                direct=True,
                env=dict(env),
                discard_stdout=True,
            )
            source = nib.load(str(raw))
            values = np.asarray(source.dataobj, dtype=np.float32)
            if len(source.shape) != 3 or not np.isfinite(values).all():
                raise ValueError("SynthSR must produce one finite 3D image")
            if not np.allclose(source.header.get_zooms()[:3], 1.0, atol=1e-3):
                raise ValueError("SynthSR must produce a 1 mm isotropic image")
            values = np.maximum(values, 0)
            header = source.header.copy()
            header.set_data_dtype(np.float32)
            result = nib.Nifti1Image(values, source.affine, header=header)
            result.set_qform(source.get_qform(), int(source.header["qform_code"]))
            result.set_sform(source.get_sform(), int(source.header["sform_code"]))
            nib.save(result, str(clipped))
            os.replace(clipped, output)
        finally:
            raw.unlink(missing_ok=True)
            clipped.unlink(missing_ok=True)

    def validate() -> tuple[bool, str]:
        try:
            synthesized = nib.load(str(output))
            values = np.asarray(synthesized.dataobj)
        except (OSError, ValueError) as error:
            return False, f"SynthSR scaffold is absent or invalid: {error}"
        if len(synthesized.shape) != 3 or not np.isfinite(values).all():
            return False, "SynthSR scaffold must be one finite 3D image."
        if not np.allclose(synthesized.header.get_zooms()[:3], 1.0, atol=1e-3):
            return False, "SynthSR scaffold must use a 1 mm isotropic grid."
        if float(values.min()) < 0:
            return False, "SynthSR scaffold contains unclipped negative intensities."
        return True, "SynthSR scaffold is finite and nonnegative."

    return LesionScaffoldPlan(
        step=Step.python(
            id="synthsr-lesion-scaffold",
            name="Synthesize Intact Lesion Reconstruction Scaffold",
            inputs=(t1w, image, license_file),
            outputs=(output,),
            action=action,
            validate=validate,
            force=force,
            parameters=synthsr_scaffold_contract(),
        ),
        image=output,
    )


def create_lesion_reconstruction_summary_step(
    *,
    subject_dir: Path,
    lesion_mask: Path,
    scaffold_image: Path,
    scaffold_synthesis: Mapping[str, object],
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
            "ScaffoldSynthesis": dict(scaffold_synthesis),
            "SyntheticT1w": str(scaffold_image),
            "LesionMask": str(lesion_mask),
            "CompleteScaffold": True,
        }
        if surface_reconstruction.get("backend") == "FastSurfer":
            result["MaskReconciliation"] = json.loads(reconciliation.read_text(encoding="utf-8"))
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

    inputs = (*required, lesion_mask, scaffold_image)
    if surface_reconstruction.get("backend") == "FastSurfer":
        inputs = (*inputs, reconciliation)
    return Step.python(
        name="Write Lesion Reconstruction Summary",
        inputs=inputs,
        outputs=(output,),
        action=action,
        validate=validate,
        force=force,
        parameters={
            "scaffold_synthesis": dict(scaffold_synthesis),
            "surface_reconstruction": dict(surface_reconstruction),
        },
    )


def create_synthetic_scaffold_metadata_step(
    *,
    image: Path,
    source_t1w: Path,
    synthesis_input: Path,
    output: Path,
    force: bool,
) -> Step:
    """Describe the whole-brain synthetic image used for reconstruction."""
    method = synthsr_scaffold_contract()
    payload = {
        "Type": "synthetic intact anatomical reconstruction scaffold",
        "Sources": [str(source_t1w)],
        "SyntheticTissue": True,
        "Method": method["backend"],
        "SoftwareVersion": method["version"],
        "SoftwareBuild": method["build"],
        "Model": method["model"],
        "OutputResolutionMM": method["output_resolution_mm"],
        "NegativeIntensityPolicy": method["negative_intensity_policy"],
        "Description": (
            "Whole-brain computational alternative for registration and surface "
            "reconstruction. This image is synthetic, not observed anatomy."
        ),
    }
    return create_json_step(
        step_name="Write SynthSR Anatomical Metadata",
        path=output,
        payload=payload,
        inputs=(image, source_t1w, synthesis_input),
        force=force,
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
            for source, path in metrics.items():
                source_image = nib.load(str(source))
                image = nib.load(str(path))
                if not isinstance(source_image, nib.GiftiImage) or not isinstance(
                    image, nib.GiftiImage
                ):
                    raise ValueError(f"Cut metric source or output is not GIFTI: {path}")
                structure = source_image.meta.get("AnatomicalStructurePrimary")
                if not structure or image.meta.get("AnatomicalStructurePrimary") != structure:
                    raise ValueError(f"Cut metric lost its anatomical structure: {path}")
                if any(np.asarray(array.data).shape[0] != retained.size for array in image.darrays):
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
