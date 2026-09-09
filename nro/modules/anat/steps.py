"""Build anatomical preprocessing steps and their validation helpers."""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from nro.engine.execution import (
    create_copy_file_step,
    ensure_directory,
    new_step_counter,
)
from nro.engine.images import sidecar_json_path
from nro.engine.io import invalid_gzip_files
from nro.engine.manifests import create_json_step
from nro.engine.paths import (
    preprocessing_session_anat_dir,
    preprocessing_session_work_dir,
)
from nro.modules.anat.common import AnatImage, robust_template_cmd, sort_anat_images
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

from .constants import (
    _FREESURFER_ASEG_LABELS,
    _FS_GIFTI_VOLGEOM_META_PREFIXES,
)

next_step = new_step_counter()


@dataclass(frozen=True)
class _SessionAnatomicalPlan:
    source: AnatImage
    staged_raw: Path
    staged_preprocessed: Path
    registration_reference: Optional[Path]
    registration_matrix: Optional[Path]
    final_source: Path
    output: Path
    mask: Path
    metadata_output: Path
    metadata: dict[str, object]


def _plan_session_anatomicals(
    images: Sequence[AnatImage],
    *,
    project: str,
    preprocessing_id: str,
    sub_id: str,
    execution_context: ExecutionContext | None = None,
) -> tuple[_SessionAnatomicalPlan, ...]:
    by_session: dict[str, list[AnatImage]] = {}
    for image in images:
        by_session.setdefault(image.session_id, []).append(image)

    plans: list[_SessionAnatomicalPlan] = []
    for session_id, session_images in by_session.items():
        if execution_context is None:
            output_dir = preprocessing_session_anat_dir(
                sub_id, session_id, project=project, preprocessing_id=preprocessing_id
            )
            work_dir = (
                preprocessing_session_work_dir(
                    sub_id, session_id, project=project, preprocessing_id=preprocessing_id
                )
                / "anat"
                / "session_level"
            )
        else:
            relative = Path("derivatives/preprocessing") / preprocessing_id / sub_id / session_id
            output_dir = execution_context.paths.output_project(project) / relative / "anat"
            work_dir = (
                execution_context.paths.private_project(project) / relative / "anat/session_level"
            )
            execution_context.require_output(output_dir)
            execution_context.require_output(work_dir)
        staged = {
            image.modality: work_dir
            / f"{image.image.name.removesuffix('.nii.gz').removesuffix('.nii')}_desc-preproc_{image.modality}.nii.gz"
            for image in session_images
        }
        has_paired_anatomicals = "T1w" in staged and "T2w" in staged
        for image in session_images:
            payload: dict[str, object] = {}
            if image.json is not None and image.json.exists():
                payload = json.loads(image.json.read_text(encoding="utf-8"))
            payload["Sources"] = [str(image.image)]
            payload["BiasCorrection"] = "N4BiasFieldCorrection"
            registration_reference: Optional[Path] = None
            registration_matrix: Optional[Path] = None
            final_source = staged[image.modality]
            if image.modality == "T2w" and has_paired_anatomicals:
                registration_reference = staged["T1w"]
                final_source = work_dir / (
                    f"{image.image.name.removesuffix('.nii.gz').removesuffix('.nii')}_space-T1w.nii.gz"
                )
                registration_matrix = work_dir / (
                    f"{sub_id}_{session_id}_from-T2w_to-T1w_mode-image_xfm.mat"
                )
                payload["SpatialReference"] = "T1w"
                payload["TransformToT1w"] = str(registration_matrix)
            output = output_dir / image.image.name
            stem = output.name.removesuffix(".nii.gz").removesuffix(".nii")
            mask = output_dir / f"{stem}_desc-brain_mask.nii.gz"
            json_name = (
                image.json.name if image.json is not None else sidecar_json_path(output).name
            )
            payload["BrainExtraction"] = "mri_synthstrip"
            payload["BrainMask"] = str(mask)
            plans.append(
                _SessionAnatomicalPlan(
                    source=image,
                    staged_raw=work_dir / image.image.name,
                    staged_preprocessed=staged[image.modality],
                    registration_reference=registration_reference,
                    registration_matrix=registration_matrix,
                    final_source=final_source,
                    output=output,
                    mask=mask,
                    metadata_output=output_dir / json_name,
                    metadata=payload,
                )
            )
    return tuple(plans)


def _brain_extract_anat_copy(
    synthstrip_image: Path,
    *,
    env: dict[str, str],
    source: Path,
    dst: Path,
    mask: Path,
    force: bool,
) -> Step:
    cmd = [
        str(synthstrip_image),
        "-i",
        str(source),
        "-o",
        str(dst.parent / f"{dst.name}.tmp.nii.gz"),
        "-m",
        str(mask.parent / f"{mask.name}.tmp.nii.gz"),
    ]

    # ``dst`` is replaced atomically by SynthStrip, so it cannot also be an
    # input to its own freshness check.  Doing so makes write ordering look
    # like a changed input and can rerun the whole subject-anatomy chain.
    def validate() -> tuple[bool, str]:
        bad_outputs = invalid_gzip_files([dst, mask])
        if bad_outputs:
            return False, (
                "Brain-extraction outputs are not readable gzip NIfTIs: "
                + Runner._format_outputs([p.name for p in bad_outputs])
            )
        return True, "Brain-extraction outputs are readable."

    tmp_img = dst.parent / f"{dst.name}.tmp.nii.gz"
    tmp_mask = mask.parent / f"{mask.name}.tmp.nii.gz"

    def prepare() -> None:
        tmp_img.unlink(missing_ok=True)
        tmp_mask.unlink(missing_ok=True)

    def finalize() -> None:
        tmp_img.replace(dst)
        tmp_mask.replace(mask)

    return Step.command_step(
        cmd,
        env={**os.environ, **env},
        name="Brain Extraction",
        outputs=(dst, mask),
        inputs=(source,),
        force=force,
        direct=True,
        prepare=prepare,
        finalize=finalize,
        validate=validate,
    )


def _strip_freesurfer_volgeom_metadata(
    *,
    surf_in: Path,
    surf_out: Path,
    force: bool,
) -> Step:
    def normalize() -> None:
        try:
            import nibabel as nib  # type: ignore
        except Exception as e:
            raise SystemExit(
                f"Missing dependency: nibabel is required to normalize GIFTI surface metadata ({e})"
            ) from e
        img = nib.load(str(surf_in))
        for darray in img.darrays:
            meta_obj = getattr(darray, "meta", None)
            data_meta = getattr(meta_obj, "data", None)
            if isinstance(data_meta, dict):
                for key in list(data_meta.keys()):
                    if any(
                        str(key).startswith(prefix) for prefix in _FS_GIFTI_VOLGEOM_META_PREFIXES
                    ):
                        del data_meta[key]
            elif meta_obj is not None:
                for key in list(meta_obj.keys()):
                    if any(
                        str(key).startswith(prefix) for prefix in _FS_GIFTI_VOLGEOM_META_PREFIXES
                    ):
                        del meta_obj[key]
        img_meta = getattr(img, "meta", None)
        if img_meta is not None:
            for key in list(img_meta.keys()):
                if any(str(key).startswith(prefix) for prefix in _FS_GIFTI_VOLGEOM_META_PREFIXES):
                    del img_meta[key]
        ensure_directory(surf_out.parent)
        nib.save(img, str(surf_out))

    return Step.python(
        name="Normalize Surface Metadata",
        outputs=(surf_out,),
        inputs=(surf_in,),
        force=force,
        action=normalize,
    )


def _create_copy_or_average_step(
    *,
    env: dict[str, str],
    images: Sequence[AnatImage],
    modality: str,
    strategy: str,
    out_img: Path,
    work_dir: Path,
    force: bool,
) -> tuple[Step, dict[str, object]]:
    if not images:
        raise ValueError("An anatomical selection step requires at least one image.")

    ordered = sort_anat_images(images)
    out_meta = {
        "modality": modality,
        "strategy": strategy,
        "sources": [str(item.image) for item in ordered],
    }
    if strategy == "first" or len(ordered) == 1:
        src = ordered[0].image

        def validate() -> tuple[bool, str]:
            bad_outputs = invalid_gzip_files([out_img])
            if bad_outputs:
                return False, (
                    "Subject-level anatomical image is not a readable gzip NIfTI: "
                    + Runner._format_outputs([p.name for p in bad_outputs])
                )
            return True, "Subject-level anatomical image is readable."

        step = create_copy_file_step(
            src=src,
            dst=out_img,
            force=force,
            step_name=f"Select Subject {modality} Image",
            validate=validate,
        )
    else:
        tmp_dir = work_dir / f"robust_template_{modality}"
        cmd = robust_template_cmd(
            [item.image for item in ordered], out_img, tmp_dir / f"{modality}_"
        )

        def validate() -> tuple[bool, str]:
            bad_outputs = invalid_gzip_files([out_img])
            if bad_outputs:
                return False, (
                    "Existing robust-template output is not a readable gzip NIfTI: "
                    + Runner._format_outputs([p.name for p in bad_outputs])
                )
            return True, "Robust-template output is readable."

        step = Step.command_step(
            cmd,
            name=f"Construct Subject {modality} Robust Template",
            outputs=(out_img,),
            inputs=tuple(item.image for item in ordered),
            force=force,
            env=env,
            prepare=lambda: (
                tmp_dir.mkdir(parents=True, exist_ok=True),
                out_img.unlink(missing_ok=True),
            ),
            validate=validate,
        )
    return step, out_meta


def _subject_preproc_path(*, images: Sequence[AnatImage], modality: str, out_dir: Path) -> Path:
    """Return the canonical public location of the selected subject anatomy.

    Selection is itself a derivative-producing operation.  In particular, it
    must not first write an equivalent private copy and then copy that file to
    the public derivative: the private copy has an independent timestamp and
    can otherwise spuriously look like an anatomical change.
    """
    ordered = sort_anat_images(images)
    subject_label = str(ordered[0].entities.get("sub", "") or out_dir.parent.name).strip()
    if subject_label and not subject_label.startswith("sub-"):
        subject_label = f"sub-{subject_label}"
    return out_dir / f"{subject_label}_desc-preproc_{modality}.nii.gz"


def _register_t2_to_t1(
    *,
    env: dict[str, str],
    t2_src: Path,
    t1_ref: Path,
    out_t2: Path,
    out_mat: Path,
    force: bool,
) -> Step:
    cmd = [
        "flirt",
        "-in",
        str(t2_src),
        "-ref",
        str(t1_ref),
        "-omat",
        str(out_mat),
        "-out",
        str(out_t2),
        "-dof",
        "6",
    ]
    return Step.command_step(
        cmd,
        name="Register T2w to T1w",
        env=env,
        outputs=(out_t2, out_mat),
        inputs=(t2_src, t1_ref),
        force=force,
        prepare=lambda: (
            out_t2.parent.mkdir(parents=True, exist_ok=True),
            out_mat.parent.mkdir(parents=True, exist_ok=True),
        ),
    )


def _write_json_step(
    path: Path,
    payload: dict[str, object],
    *,
    step_name: str = "Write Anatomical Metadata",
) -> Step:
    sources = tuple(
        candidate
        for value in payload.get("Sources", [])
        if isinstance(value, str)
        for candidate in (Path(value),)
        if candidate.exists()
    )
    return create_json_step(
        step_name=step_name,
        path=path,
        payload=payload,
        inputs=sources,
        force=False,
    )


def _export_mgz(*, env: dict[str, str], src: Path, dst: Path, force: bool) -> Step:
    cmd = ["mri_convert", str(src), str(dst)]
    return Step.command_step(
        cmd,
        name="Export FreeSurfer Volume",
        outputs=(dst,),
        inputs=(src,),
        force=force,
        env=env,
        prepare=lambda: dst.parent.mkdir(parents=True, exist_ok=True),
    )


def _resample_mask_to_native(
    *,
    env: dict[str, str],
    src: Path,
    ref_image: Path,
    dst: Path,
    force: bool,
) -> Step:
    cmd = [
        "mri_vol2vol",
        "--mov",
        str(src),
        "--targ",
        str(ref_image),
        "--regheader",
        "--interp",
        "nearest",
        "--o",
        str(dst),
    ]
    return Step.command_step(
        cmd,
        name="Resample Mask to Native Anatomy",
        outputs=(dst,),
        inputs=(src, ref_image),
        force=force,
        env=env,
        prepare=lambda: dst.parent.mkdir(parents=True, exist_ok=True),
    )


def _aseg_label_ids(segmentations: Sequence[str]) -> list[int]:
    """Resolve named FreeSurfer segmentations to their standard aseg IDs."""
    return [_FREESURFER_ASEG_LABELS[name] for name in segmentations]


def _create_label_mask_step(
    *,
    source: Path,
    output: Path,
    labels: Sequence[int],
    name: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    command = ["mri_binarize", "--i", str(source), "--o", str(output)]
    for label in labels:
        command.extend(("--match", str(label)))
    return Step.command_step(
        command,
        name=name,
        outputs=(output,),
        inputs=(source,),
        force=force,
        env=env,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
    )


def _create_ribbon_mask_step(
    *,
    source: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        ["mri_binarize", "--i", str(source), "--min", "1", "--o", str(output)],
        name="Extract Cortical Ribbon FreeSurfer Labels",
        outputs=(output,),
        inputs=(source,),
        force=force,
        env=env,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
    )


def _recon_done(subject_dir: Path) -> bool:
    return (subject_dir / "scripts" / "recon-all.done").exists()


def _recon_breadcrumb(subject_dir: Path) -> Path:
    return subject_dir / ".nro_recon_all_complete"


def _recon_required_outputs(subject_dir: Path) -> list[Path]:
    return [
        subject_dir / "mri" / "orig" / "001.mgz",
        subject_dir / "mri" / "T1.mgz",
        subject_dir / "mri" / "aseg.mgz",
        subject_dir / "mri" / "brainmask.mgz",
        subject_dir / "mri" / "ribbon.mgz",
        subject_dir / "surf" / "lh.white",
        subject_dir / "surf" / "rh.white",
    ]


def _recon_valid(subject_dir: Path) -> bool:
    if not _recon_done(subject_dir):
        return False
    for path in _recon_required_outputs(subject_dir):
        if (not path.exists()) or path.stat().st_size <= 0:
            return False
    return True


def _recon_invalid_reason(subject_dir: Path) -> str:
    missing = [
        str(path)
        for path in _recon_required_outputs(subject_dir)
        if (not path.exists()) or path.stat().st_size <= 0
    ]
    if not missing:
        return "Re-running because existing recon-all outputs are incomplete."
    return "Re-running because existing recon-all outputs are incomplete or missing: " + ", ".join(
        missing
    )


def _create_recon_all_step(
    *,
    run_child: Callable[..., Optional[str]],
    env: dict[str, str],
    t1w: Optional[Path],
    t2w: Optional[Path],
    subjects_dir: Path,
    fs_subject: str,
    force: bool,
) -> Step:
    if t1w is None and t2w is None:
        raise SystemExit(
            "Need at least one subject-level T1w or T2w image for anatomical preprocessing."
        )
    subject_dir = subjects_dir / fs_subject
    recon_done = subject_dir / "scripts" / "recon-all.done"
    breadcrumb = _recon_breadcrumb(subject_dir)

    def validate() -> tuple[bool, str]:
        if _recon_valid(subject_dir):
            return True, "FreeSurfer recon-all outputs are complete."
        return False, _recon_invalid_reason(subject_dir)

    def action() -> None:
        cmd = ["recon-all", "-sd", str(subjects_dir), "-subjid", fs_subject]
        if t1w is not None:
            cmd += ["-i", str(t1w)]
        elif t2w is not None:
            cmd += ["-i", str(t2w)]
        if t2w is not None:
            cmd += ["-T2", str(t2w), "-T2pial"]
        run_child(cmd + ["-all"], env=env, stream_output=True)
        if not _recon_valid(subject_dir):
            raise SystemExit(f"recon-all completed without a valid output set under: {subject_dir}")

    return Step.directory_step(
        name="FreeSurfer Recon-All",
        directory=subject_dir,
        breadcrumb=breadcrumb,
        inputs=tuple(path for path in (t1w, t2w) if path is not None),
        outputs=(*_recon_required_outputs(subject_dir), recon_done),
        action=action,
        validate=validate,
        force=force,
        breadcrumb_text="recon-all complete\n",
    )


def _surface_names() -> dict[str, str]:
    return {
        "white": "white",
        "pial": "pial",
        "inflated": "inflated",
        "smoothwm": "smoothwm",
        "sphere": "sphere",
        "sphere.reg": "desc-reg_sphere",
    }


def _hemi_label(hemi: str) -> str:
    return {"lh": "L", "rh": "R"}[hemi]


def _metric_names() -> dict[str, tuple[str, str]]:
    return {
        "thickness": ("thickness", "cortical thickness"),
        "sulc": ("sulc", "sulcal depth"),
    }


def _create_t1_to_fsnative_affine_step(
    *,
    subject_t1: Path,
    fsnative_reference: Path,
    output: Path,
    registration_file: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        [
            "tkregister2",
            "--mov",
            str(subject_t1),
            "--targ",
            str(fsnative_reference),
            "--regheader",
            "--reg",
            str(registration_file),
            "--noedit",
            "--fslregout",
            str(output),
        ],
        name="Construct T1w-to-fsnative Affine",
        env=env,
        outputs=(output,),
        inputs=(subject_t1, fsnative_reference),
        force=force,
        finalize=lambda: registration_file.unlink(missing_ok=True),
    )


def _create_inverse_affine_step(
    *, source: Path, output: Path, env: dict[str, str], force: bool
) -> Step:
    return Step.command_step(
        ["convert_xfm", "-omat", str(output), "-inverse", str(source)],
        name="Invert T1w-to-fsnative Affine",
        env=env,
        outputs=(output,),
        inputs=(source,),
        force=force,
    )


def _create_mri_conversion_step(
    *,
    source: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
    name: str,
    to_scanner: bool = False,
    executable: str = "mri_convert",
) -> Step:
    command = [executable]
    if to_scanner:
        command.append("--to-scanner")
    command.extend((str(source), str(output)))
    return Step.command_step(
        command,
        name=name,
        env=env,
        outputs=(output,),
        inputs=(source,),
        force=force,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
    )


def _create_surface_affine_step(
    *,
    source: Path,
    affine: Path,
    source_volume: Path,
    target_volume: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        [
            "wb_command",
            "-surface-apply-affine",
            str(source),
            str(affine),
            str(output),
            "-flirt",
            str(source_volume),
            str(target_volume),
        ],
        name="Surface Affine Transform",
        env=env,
        outputs=(output,),
        inputs=(source, affine, source_volume, target_volume),
        force=force,
    )


def _create_metric_conversion_step(
    *,
    metric: Path,
    surface: Path,
    output: Path,
    description: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        ["mris_convert", "-c", str(metric), str(surface), str(output)],
        name=f"Convert FreeSurfer {description.title()} Metric",
        env=env,
        outputs=(output,),
        inputs=(metric, surface),
        force=force,
    )


def _create_midthickness_step(
    *,
    white: Path,
    pial: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        [
            "wb_command",
            "-surface-average",
            str(output),
            "-surf",
            str(white),
            "-surf",
            str(pial),
        ],
        name="Construct Midthickness Surface",
        env=env,
        outputs=(output,),
        inputs=(white, pial),
        force=force,
    )


def _create_mni_registration_step(
    *,
    subject_t1: Path,
    subject_brain_mask: Path,
    mni_brain_template: Path,
    mni_brain_mask: Path,
    mni_template: Path,
    prefix: Path,
    produced_forward: Path,
    produced_inverse: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    inputs = (
        subject_t1,
        subject_brain_mask,
        mni_template,
        mni_brain_template,
        mni_brain_mask,
    )
    return Step.command_step(
        [
            "antsRegistration",
            "--dimensionality",
            "3",
            "--float",
            "0",
            "--collapse-output-transforms",
            "1",
            "--write-composite-transform",
            "1",
            "--output",
            f"[{prefix}]",
            "--interpolation",
            "LanczosWindowedSinc",
            "--use-histogram-matching",
            "1",
            "--winsorize-image-intensities",
            "[0.005,0.995]",
            "--masks",
            f"[{mni_brain_mask},{subject_brain_mask}]",
            "--initial-moving-transform",
            f"[{mni_brain_template},{subject_t1},1]",
            "--transform",
            "Rigid[0.1]",
            "--metric",
            f"MI[{mni_brain_template},{subject_t1},1,32,Regular,0.25]",
            "--convergence",
            "[1000x500x250x0,1e-6,10]",
            "--shrink-factors",
            "8x4x2x1",
            "--smoothing-sigmas",
            "3x2x1x0vox",
            "--transform",
            "Affine[0.1]",
            "--metric",
            f"MI[{mni_brain_template},{subject_t1},1,32,Regular,0.25]",
            "--convergence",
            "[1000x500x250x0,1e-6,10]",
            "--shrink-factors",
            "8x4x2x1",
            "--smoothing-sigmas",
            "3x2x1x0vox",
            "--transform",
            "SyN[0.1,3,0]",
            "--metric",
            f"CC[{mni_brain_template},{subject_t1},1,4]",
            "--convergence",
            "[100x70x50x20,1e-6,10]",
            "--shrink-factors",
            "8x4x2x1",
            "--smoothing-sigmas",
            "3x2x1x0vox",
        ],
        name="SyN Registration",
        env=env,
        outputs=(produced_forward, produced_inverse),
        inputs=inputs,
        force=force,
        prepare=lambda: prefix.parent.mkdir(parents=True, exist_ok=True),
    )


def _create_transform_finalization_step(
    *,
    produced_forward: Path,
    produced_inverse: Path,
    forward: Path,
    inverse: Path,
    force: bool,
) -> Step:
    def finalize() -> None:
        shutil.copy2(produced_forward, forward)
        shutil.copy2(produced_inverse, inverse)

    return Step.python(
        name="Finalize MNI Composite Transforms",
        outputs=(forward, inverse),
        inputs=(produced_forward, produced_inverse),
        force=force,
        action=finalize,
    )


def _create_mni_qc_image_step(
    *,
    input_image: Path,
    reference: Path,
    transform: Path,
    output: Path,
    from_space: str,
    to_space: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(input_image),
            "-r",
            str(reference),
            "-o",
            str(output),
            "-n",
            "LanczosWindowedSinc",
            "-t",
            str(transform),
        ],
        name=f"Write {from_space}-to-{to_space} QC Image",
        env=env,
        outputs=(output,),
        inputs=(input_image, reference, transform),
        force=force,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
    )


def _nested_paths(value: object) -> tuple[Path, ...]:
    """Return path-valued leaves from a manifest-shaped value."""
    if isinstance(value, str) and value:
        return (Path(value),)
    if isinstance(value, dict):
        return tuple(path for item in value.values() for path in _nested_paths(item))
    if isinstance(value, (list, tuple)):
        return tuple(path for item in value for path in _nested_paths(item))
    return ()
