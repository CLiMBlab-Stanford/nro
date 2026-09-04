#!/usr/bin/env python3
"""Run anatomical preprocessing for one subject."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from nro.orchestration.runtime import selected_configuration_fingerprint
from nro.orchestration.runner_graph import Step
from nro.anat.common import AnatImage, load_anat_image, robust_template_cmd, sort_anat_images
from nro.orchestration.runner import ContainerSpec, Runner, write_completion_breadcrumb
from nro.engine.execution import (
    collect_bind_directories,
    create_copy_file_step,
    ensure_directory,
    new_step_counter,
    neuroimaging_environment,
)
from nro.engine.freesurfer import find_fsaverage_directory
from nro.engine.images import sidecar_json_path
from nro.engine.io import invalid_gzip_files, read_json, require_nonempty_file, write_json
from nro.engine.manifests import create_json_step
from nro.engine.neuroimaging import create_n4_bias_correction_step
from nro.configuration.runtime import SETTINGS
from nro.engine.paths import (
    anatomical_manifest_path,
    preprocessing_derivatives_root,
    preprocessing_session_anat_dir,
    preprocessing_session_work_dir,
    preprocessing_subject_anat_dir,
    preprocessing_subject_work_dir,
    resolve_project_path,
    resolve_project_work_path,
)


LOG = logging.getLogger("preprocess_anat")
DEFAULT_CONTAINER = Path(SETTINGS.common.qunex_container)
_FS_GIFTI_VOLGEOM_META_PREFIXES = ("VolGeom", "VolGeomC_")
# Standard aseg identifiers from FreeSurferColorLUT.txt:
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
_FREESURFER_ASEG_LABELS = {
    "Left-Cerebral-Cortex": 3,
    "Left-Cerebellum-White-Matter": 7,
    "Left-Cerebellum-Cortex": 8,
    "Left-Thalamus-Proper": 10,
    "Left-Caudate": 11,
    "Left-Putamen": 12,
    "Left-Pallidum": 13,
    "Brain-Stem": 16,
    "Left-Hippocampus": 17,
    "Left-Amygdala": 18,
    "Left-Accumbens-area": 26,
    "Left-VentralDC": 28,
    "Right-Cerebral-Cortex": 42,
    "Right-Cerebellum-White-Matter": 46,
    "Right-Cerebellum-Cortex": 47,
    "Right-Thalamus-Proper": 49,
    "Right-Caudate": 50,
    "Right-Putamen": 51,
    "Right-Pallidum": 52,
    "Right-Hippocampus": 53,
    "Right-Amygdala": 54,
    "Right-Accumbens-area": 58,
    "Right-VentralDC": 60,
}
_FREESURFER_GRAY_MATTER_SEGMENTATIONS = (
    "Left-Cerebral-Cortex",
    "Right-Cerebral-Cortex",
    "Left-Cerebellum-Cortex",
    "Right-Cerebellum-Cortex",
    "Left-Thalamus-Proper",
    "Right-Thalamus-Proper",
    "Left-Caudate",
    "Right-Caudate",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Pallidum",
    "Right-Pallidum",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Accumbens-area",
    "Right-Accumbens-area",
    "Left-VentralDC",
    "Right-VentralDC",
)
_FREESURFER_SUBCORTICAL_SEGMENTATIONS = {
    "cerebellum": (
        "Left-Cerebellum-White-Matter",
        "Left-Cerebellum-Cortex",
        "Right-Cerebellum-White-Matter",
        "Right-Cerebellum-Cortex",
    ),
    "thalamus": ("Left-Thalamus-Proper", "Right-Thalamus-Proper"),
    "caudate": ("Left-Caudate", "Right-Caudate"),
    "putamen": ("Left-Putamen", "Right-Putamen"),
    "pallidum": ("Left-Pallidum", "Right-Pallidum"),
    "brainStem": ("Brain-Stem",),
    "hippocampus": ("Left-Hippocampus", "Right-Hippocampus"),
    "amygdala": ("Left-Amygdala", "Right-Amygdala"),
}
next_step = new_step_counter()


@dataclass(frozen=True)
class Inputs:
    sub_id: str
    t1w: tuple[AnatImage, ...]
    t2w: tuple[AnatImage, ...]


@dataclass(frozen=True)
class Options:
    project: str
    preprocessing_id: str
    out_dir: Path
    work_dir: Path
    freesurfer_subjects_dir: Path
    fs_subject: str
    selection_strategy: str
    mni_template: Path
    container: Optional[ContainerSpec]
    synthstrip_image: Optional[Path]
    force: bool
    nthreads: int


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
) -> tuple[_SessionAnatomicalPlan, ...]:
    by_session: dict[str, list[AnatImage]] = {}
    for image in images:
        by_session.setdefault(image.session_id, []).append(image)

    plans: list[_SessionAnatomicalPlan] = []
    for session_id, session_images in by_session.items():
        output_dir = preprocessing_session_anat_dir(
            sub_id, session_id, project=project, preprocessing_id=preprocessing_id
        )
        work_dir = preprocessing_session_work_dir(
            sub_id, session_id, project=project, preprocessing_id=preprocessing_id
        ) / "anat" / "session_level"
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
                image.json.name
                if image.json is not None
                else sidecar_json_path(output).name
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
            raise SystemExit(f"Missing dependency: nibabel is required to normalize GIFTI surface metadata ({e})") from e
        img = nib.load(str(surf_in))
        for darray in img.darrays:
            meta_obj = getattr(darray, "meta", None)
            data_meta = getattr(meta_obj, "data", None)
            if isinstance(data_meta, dict):
                for key in list(data_meta.keys()):
                    if any(str(key).startswith(prefix) for prefix in _FS_GIFTI_VOLGEOM_META_PREFIXES):
                        del data_meta[key]
            elif meta_obj is not None:
                for key in list(meta_obj.keys()):
                    if any(str(key).startswith(prefix) for prefix in _FS_GIFTI_VOLGEOM_META_PREFIXES):
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
        cmd = robust_template_cmd([item.image for item in ordered], out_img, tmp_dir / f"{modality}_")
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
    *, source: Path, output: Path, labels: Sequence[int], name: str,
    env: dict[str, str], force: bool,
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
    *, source: Path, output: Path, env: dict[str, str], force: bool,
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
    missing = [str(path) for path in _recon_required_outputs(subject_dir) if (not path.exists()) or path.stat().st_size <= 0]
    if not missing:
        return "Re-running because existing recon-all outputs are incomplete."
    return "Re-running because existing recon-all outputs are incomplete or missing: " + ", ".join(missing)


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
        raise SystemExit("Need at least one subject-level T1w or T2w image for anatomical preprocessing.")
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


def build_module(inputs: Inputs, opts: Options) -> Runner:
    """Resolve anatomical inputs and construct the complete module DAG."""
    all_images = [*inputs.t1w, *inputs.t2w]
    if not all_images:
        raise SystemExit(f"No anatomical images provided for {inputs.sub_id}.")
    if opts.synthstrip_image is None:
        raise SystemExit("Missing SynthStrip image path.")
    require_nonempty_file(opts.synthstrip_image, "SynthStrip image")
    require_nonempty_file(opts.mni_template, "MNI template")

    public_inputs = [item.image for item in all_images]
    public_inputs.extend(item.json for item in all_images if item.json is not None)
    env = neuroimaging_environment(opts.nthreads, subjects_dir=opts.freesurfer_subjects_dir)
    binds = collect_bind_directories(
        [
            *(item.image for item in all_images),
            *(item.json for item in all_images if item.json is not None),
            opts.out_dir,
            opts.work_dir,
            opts.freesurfer_subjects_dir,
            opts.mni_template,
        ]
    )
    runner = Runner(
        module_name="Anatomical Preprocessing Module",
        container=opts.container,
        binds=binds,
        logger=LOG,
        next_step=next_step,
    )
    initialized = opts.work_dir / "initialized.complete"

    def initialize_outputs() -> None:
        ensure_directory(
            preprocessing_derivatives_root(
                project=opts.project, preprocessing_id=opts.preprocessing_id
            )
        )
        ensure_directory(opts.out_dir)
        ensure_directory(opts.work_dir)
        ensure_directory(opts.freesurfer_subjects_dir)
        write_completion_breadcrumb(initialized, "Anatomical output initialized\n")

    runner.add_step(
        Step.python(
            name="Initialize Anatomical Outputs",
            outputs=(initialized,),
            action=initialize_outputs,
        )
    )

    configuration = {
        "selection_strategy": opts.selection_strategy,
        "fs_subject": opts.fs_subject,
        "mni_template": str(opts.mni_template),
        "synthstrip_image": str(opts.synthstrip_image),
        "configuration_fingerprint": selected_configuration_fingerprint(),
    }
    configuration_snapshot = opts.work_dir / "configuration.json"

    def validate_configuration() -> tuple[bool, str]:
        try:
            current = read_json(configuration_snapshot)
        except (OSError, ValueError, TypeError):
            return False, "Anatomical configuration snapshot is missing or unreadable."
        if current != configuration:
            return False, "Anatomical configuration changed."
        return True, "Anatomical configuration is unchanged."

    runner.add_step(
        Step.python(
            name="Write Anatomical Configuration",
            outputs=(configuration_snapshot,),
            inputs=(initialized,),
            force=opts.force,
            action=lambda: write_json(configuration_snapshot, configuration),
            validate=validate_configuration,
        )
    )

    runner.set_definition_inputs((configuration_snapshot,))
    dependency_check = opts.work_dir / "dependencies.complete"

    def check_dependencies() -> None:
        runner.require_cmds([
            "mri_binarize",
            "mri_convert",
            "mri_vol2vol",
            "mri_robust_template",
            "N4BiasFieldCorrection",
            "recon-all",
            "mris_convert",
            "flirt",
            "fslmaths",
            "wb_command",
            "tkregister2",
            "convert_xfm",
            "antsRegistration",
            "antsApplyTransforms",
        ])
        write_completion_breadcrumb(
            dependency_check, "Anatomical dependencies available\n"
        )

    runner.add_step(Step.python(
        name="Check Anatomical Dependencies",
        outputs=(dependency_check,),
        action=check_dependencies,
        force=opts.force,
    ))

    session_plans = _plan_session_anatomicals(
        images=all_images,
        project=opts.project,
        preprocessing_id=opts.preprocessing_id,
        sub_id=inputs.sub_id,
    )
    for plan in session_plans:
        runner.add_step(create_copy_file_step(
            src=plan.source.image,
            dst=plan.staged_raw,
            force=opts.force,
            step_name="Stage Session Anatomical Image",
        ))
        runner.add_step(create_n4_bias_correction_step(
            env=env,
            in_img=plan.staged_raw,
            out_img=plan.staged_preprocessed,
            force=opts.force,
            validate_gzip=True,
            step_name="N4 Bias Field Correction",
        ))
    for plan in session_plans:
        if plan.registration_reference is not None:
            assert plan.registration_matrix is not None
            runner.add_step(_register_t2_to_t1(
                env=env,
                t2_src=plan.staged_preprocessed,
                t1_ref=plan.registration_reference,
                out_t2=plan.final_source,
                out_mat=plan.registration_matrix,
                force=opts.force,
            ))
        runner.add_step(_brain_extract_anat_copy(
            opts.synthstrip_image,
            env=env,
            source=plan.final_source,
            dst=plan.output,
            mask=plan.mask,
            force=opts.force,
        ))
        runner.add_step(create_json_step(
            step_name="Write Session Anatomical Metadata",
            path=plan.metadata_output,
            payload=plan.metadata,
            inputs=(plan.final_source, plan.output, plan.mask),
            force=opts.force,
        ))
    copied_images = [
        AnatImage(
            image=plan.output,
            json=plan.metadata_output,
            modality=plan.source.modality,
            entities=dict(plan.source.entities),
            session_id=plan.source.session_id,
            acq=plan.source.acq,
            run=plan.source.run,
            time_value=plan.source.time_value,
        )
        for plan in session_plans
    ]
    copied_session_files = [str(plan.output) for plan in session_plans]
    copied_t1 = [item for item in copied_images if item.modality == "T1w"]
    copied_t2 = [item for item in copied_images if item.modality == "T2w"]
    subj_t1_final: Optional[Path] = None
    subj_t2_final: Optional[Path] = None
    if copied_t1:
        subj_t1_final = _subject_preproc_path(
            images=copied_t1,
            modality="T1w",
            out_dir=opts.out_dir,
        )
    if copied_t2:
        subj_t2_final = _subject_preproc_path(
            images=copied_t2,
            modality="T2w",
            out_dir=opts.out_dir,
        )
    subj_t1 = subj_t1_final
    t1_meta: dict[str, object] = {
        "modality": "T1w", "sources": [], "strategy": opts.selection_strategy
    }
    if subj_t1 is not None:
        t1_step, t1_meta = _create_copy_or_average_step(
            env=env,
            images=copied_t1,
            modality="T1w",
            strategy=opts.selection_strategy,
            out_img=subj_t1,
            work_dir=opts.work_dir,
            force=opts.force,
        )
        runner.add_step(t1_step)
    subj_t2 = subj_t2_final
    t2_meta: dict[str, object] = {
        "modality": "T2w", "sources": [], "strategy": opts.selection_strategy
    }
    if subj_t2 is not None:
        t2_step, t2_meta = _create_copy_or_average_step(
            env=env,
            images=copied_t2,
            modality="T2w",
            strategy=opts.selection_strategy,
            out_img=subj_t2,
            work_dir=opts.work_dir,
            force=opts.force,
        )
        runner.add_step(t2_step)
    if subj_t1 is not None:
        runner.add_step(create_json_step(
            step_name="Write Subject T1w Metadata",
            path=subj_t1.with_suffix("").with_suffix(".json"),
            payload={
                "Sources": t1_meta["sources"],
                "SelectionStrategy": opts.selection_strategy,
                "BiasCorrection": "N4BiasFieldCorrection",
            },
            inputs=(subj_t1,),
            force=opts.force,
        ))
    if subj_t2 is not None:
        runner.add_step(create_json_step(
            step_name="Write Subject T2w Metadata",
            path=subj_t2.with_suffix("").with_suffix(".json"),
            payload={
                "Sources": t2_meta["sources"],
                "SelectionStrategy": opts.selection_strategy,
                "BiasCorrection": "N4BiasFieldCorrection",
                "SpatialReference": "T1w" if copied_t1 else None,
            },
            inputs=(subj_t2,),
            force=opts.force,
        ))
    myelin_map: Optional[Path] = None
    if subj_t1 is not None and subj_t2 is not None:
        myelin_map = opts.out_dir / f"{inputs.sub_id}_space-T1w_desc-myelinMap_T1w.nii.gz"
        t2_nonzero = opts.work_dir / "myelin_map" / f"{inputs.sub_id}_desc-T2wNonzero_T2w.nii.gz"
        runner.add_step(Step.command_step(
            ["fslmaths", str(subj_t2), "-thr", "0.001", str(t2_nonzero)],
            name="Prepare Nonzero T2w for Myelin Map",
            outputs=(t2_nonzero,),
            inputs=(subj_t2,),
            force=opts.force,
            env=env,
            prepare=lambda: t2_nonzero.parent.mkdir(parents=True, exist_ok=True),
        ))
        runner.add_step(Step.command_step(
            ["fslmaths", str(subj_t1), "-div", str(t2_nonzero), str(myelin_map)],
            name="Compute T1w/T2w Myelin Map",
            outputs=(myelin_map,),
            inputs=(subj_t1, subj_t2, t2_nonzero),
            force=opts.force,
            env=env,
        ))
        runner.add_step(_write_json_step(
            myelin_map.with_suffix("").with_suffix(".json"),
            {
                "Type": "myelin map",
                "Space": "T1w",
                "Sources": [str(subj_t1), str(subj_t2)],
                "Description": "T1w/T2w ratio using preprocessed T1w and T2w images already aligned in T1w geometry.",
            },
        ))

    subject_anat = subj_t1 or subj_t2
    if subject_anat is None:
        raise SystemExit("No subject-level anatomical image available after selection.")

    subject_dir = opts.freesurfer_subjects_dir / opts.fs_subject
    runner.add_step(_create_recon_all_step(
        run_child=runner.run_child,
        env=env,
        t1w=subj_t1,
        t2w=subj_t2,
        subjects_dir=opts.freesurfer_subjects_dir,
        fs_subject=opts.fs_subject,
        force=opts.force,
    ))
    aseg_mgz = subject_dir / "mri" / "aseg.mgz"
    subcortical_masks: dict[str, str] = {}
    for structure, segmentations in _FREESURFER_SUBCORTICAL_SEGMENTATIONS.items():
        labels = _aseg_label_ids(segmentations)
        output = opts.out_dir / f"{inputs.sub_id}_desc-{structure}_mask.nii.gz"
        temporary = opts.work_dir / "subcortical_masks" / (
            f"{inputs.sub_id}_desc-{structure}_mask_fs.nii.gz"
        )
        runner.add_step(_create_label_mask_step(
            source=aseg_mgz,
            output=temporary,
            labels=labels,
            name=f"Extract {structure} FreeSurfer Labels",
            env=env,
            force=opts.force,
        ))
        runner.add_step(_resample_mask_to_native(
            env=env, src=temporary, ref_image=subject_anat,
            dst=output, force=opts.force,
        ))
        runner.add_step(_write_json_step(
            output.with_suffix("").with_suffix(".json"),
            {
                "Type": "ROI mask",
                "Structure": structure,
                "Sources": [str(aseg_mgz), str(subject_anat)],
                "FreeSurferLabels": labels,
                "FreeSurferSegmentations": list(segmentations),
                "ReferenceImage": str(subject_anat),
            },
        ))
        subcortical_masks[structure] = str(output)

    brain_mask = opts.out_dir / f"{inputs.sub_id}_desc-brain_mask.nii.gz"
    gray_mask = opts.out_dir / f"{inputs.sub_id}_desc-grayMatter_mask.nii.gz"
    ribbon_mask = opts.out_dir / f"{inputs.sub_id}_desc-ribbon_mask.nii.gz"
    core_work = opts.work_dir / "core_masks"
    brain_temporary = core_work / f"{inputs.sub_id}_desc-brain_mask_fs.nii.gz"
    gray_temporary = core_work / f"{inputs.sub_id}_desc-grayMatter_mask_fs.nii.gz"
    ribbon_temporary = core_work / f"{inputs.sub_id}_desc-ribbon_mask_fs.nii.gz"
    brainmask_mgz = subject_dir / "mri" / "brainmask.mgz"
    ribbon_mgz = subject_dir / "mri" / "ribbon.mgz"
    runner.add_step(_export_mgz(
        env=env, src=brainmask_mgz, dst=brain_temporary, force=opts.force,
    ))
    runner.add_step(_resample_mask_to_native(
        env=env, src=brain_temporary, ref_image=subject_anat,
        dst=brain_mask, force=opts.force,
    ))
    runner.add_step(_write_json_step(
        brain_mask.with_suffix("").with_suffix(".json"),
        {
            "Type": "brain mask",
            "Sources": [str(brainmask_mgz), str(subject_anat)],
            "ReferenceImage": str(subject_anat),
        },
    ))
    gray_labels = _aseg_label_ids(_FREESURFER_GRAY_MATTER_SEGMENTATIONS)
    runner.add_step(_create_label_mask_step(
        source=aseg_mgz,
        output=gray_temporary,
        labels=gray_labels,
        name="Extract Gray Matter FreeSurfer Labels",
        env=env,
        force=opts.force,
    ))
    runner.add_step(_resample_mask_to_native(
        env=env, src=gray_temporary, ref_image=subject_anat,
        dst=gray_mask, force=opts.force,
    ))
    runner.add_step(_write_json_step(
        gray_mask.with_suffix("").with_suffix(".json"),
        {
            "Type": "gray matter mask",
            "Sources": [str(aseg_mgz), str(subject_anat)],
            "FreeSurferLabels": gray_labels,
            "FreeSurferSegmentations": list(
                _FREESURFER_GRAY_MATTER_SEGMENTATIONS
            ),
            "ReferenceImage": str(subject_anat),
        },
    ))
    runner.add_step(_create_ribbon_mask_step(
        source=ribbon_mgz,
        output=ribbon_temporary,
        env=env,
        force=opts.force,
    ))
    runner.add_step(_resample_mask_to_native(
        env=env, src=ribbon_temporary, ref_image=subject_anat,
        dst=ribbon_mask, force=opts.force,
    ))
    runner.add_step(_write_json_step(
        ribbon_mask.with_suffix("").with_suffix(".json"),
        {
            "Type": "cortical ribbon mask",
            "Sources": [str(ribbon_mgz), str(subject_anat)],
            "ReferenceImage": str(subject_anat),
        },
    ))
    core_masks = {
        "brain": str(brain_mask),
        "gray_matter": str(gray_mask),
        "cortical_ribbon": str(ribbon_mask),
    }
    subject_t1 = subj_t1 or subject_anat
    fsnative_ref = (
        opts.freesurfer_subjects_dir / opts.fs_subject / "mri" / "T1.mgz"
    )
    t1_to_fsnative = (
        opts.out_dir
        / f"{opts.fs_subject}_from-T1w_to-fsnative_mode-image_xfm.txt"
    )
    fsnative_to_t1 = (
        opts.out_dir
        / f"{opts.fs_subject}_from-fsnative_to-T1w_mode-image_xfm.txt"
    )
    t1_to_fsnative_reg = (
        opts.out_dir
        / f"{opts.fs_subject}_from-T1w_to-fsnative_mode-image_xfm.dat"
    )
    runner.add_step(
        _create_t1_to_fsnative_affine_step(
            subject_t1=subject_t1,
            fsnative_reference=fsnative_ref,
            output=t1_to_fsnative,
            registration_file=t1_to_fsnative_reg,
            env=env,
            force=opts.force,
        )
    )
    runner.add_step(
        _create_inverse_affine_step(
            source=t1_to_fsnative,
            output=fsnative_to_t1,
            env=env,
            force=opts.force,
        )
    )
    fsnative_xfms = {
        "t1_to_fsnative": str(t1_to_fsnative),
        "fsnative_to_t1": str(fsnative_to_t1),
    }
    for path, from_space, to_space in (
        (t1_to_fsnative, "T1w", "fsnative"),
        (fsnative_to_t1, "fsnative", "T1w"),
    ):
        runner.add_step(
            _write_json_step(
                path.with_suffix(".json"),
                {
                    "Type": "affine",
                    "Format": "FSL",
                    "Sources": [str(subject_t1), str(fsnative_ref)],
                    "SpatialReference": to_space,
                    "From": from_space,
                    "To": to_space,
                },
            )
        )

    surface_work_dir = opts.work_dir / "surface_export"
    surface_dir = opts.freesurfer_subjects_dir / opts.fs_subject / "surf"
    fs_t1_ref_nii = surface_work_dir / f"{opts.fs_subject}_fs_t1_ref.nii.gz"
    runner.add_step(
        _create_mri_conversion_step(
            source=fsnative_ref,
            output=fs_t1_ref_nii,
            env=env,
            force=opts.force,
            name="Convert FreeSurfer T1 Reference to NIfTI",
        )
    )
    exported_surfaces: dict[str, str] = {}
    for hemi in ("lh", "rh"):
        hemi_label = _hemi_label(hemi)
        hemi_paths: dict[str, Path] = {}
        white_source = surface_dir / f"{hemi}.white"
        for source_name, bids_suffix in _surface_names().items():
            source = surface_dir / f"{hemi}.{source_name}"
            output_name = (
                f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_"
                f"{bids_suffix}.surf.gii"
            )
            output = opts.out_dir / output_name
            is_sphere = source_name in {"sphere", "sphere.reg"}
            if is_sphere:
                runner.add_step(
                    _create_mri_conversion_step(
                        source=source,
                        output=output,
                        env=env,
                        force=opts.force,
                        name="Convert FreeSurfer Sphere",
                        executable="mris_convert",
                    )
                )
            else:
                raw = surface_work_dir / f"scanner_{output_name}"
                affined = surface_work_dir / f"affined_{output_name}"
                runner.add_step(
                    _create_mri_conversion_step(
                        source=source,
                        output=raw,
                        env=env,
                        force=opts.force,
                        name="Surface Conversion",
                        to_scanner=True,
                        executable="mris_convert",
                    )
                )
                runner.add_step(
                    _create_surface_affine_step(
                        source=raw,
                        affine=fsnative_to_t1,
                        source_volume=fs_t1_ref_nii,
                        target_volume=subject_t1,
                        output=affined,
                        env=env,
                        force=opts.force,
                    )
                )
                runner.add_step(
                    _strip_freesurfer_volgeom_metadata(
                        surf_in=affined,
                        surf_out=output,
                        force=opts.force,
                    )
                )
            runner.add_step(
                _write_json_step(
                    output.with_suffix(".json"),
                    {
                        "Hemisphere": hemi_label,
                        "Space": "fsnative",
                        "AnatomicalStructurePrimary": "Cortex",
                        "SurfaceType": bids_suffix,
                        "Sources": [str(source)],
                        "SpatialReference": "fsnative" if is_sphere else "T1w",
                    },
                )
            )
            exported_surfaces[f"{hemi}.{source_name}"] = str(output)
            hemi_paths[source_name] = output

        for metric_name, (bids_suffix, metric_description) in _metric_names().items():
            metric_source = surface_dir / f"{hemi}.{metric_name}"
            metric_output = (
                opts.out_dir
                / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_{bids_suffix}.shape.gii"
            )
            runner.add_step(
                _create_metric_conversion_step(
                    metric=metric_source,
                    surface=white_source,
                    output=metric_output,
                    description=metric_description,
                    env=env,
                    force=opts.force,
                )
            )
            runner.add_step(
                _write_json_step(
                    metric_output.with_suffix(".json"),
                    {
                        "Hemisphere": hemi_label,
                        "Space": "fsnative",
                        "AnatomicalStructurePrimary": "Cortex",
                        "MetricType": metric_description,
                        "Sources": [str(metric_source), str(white_source)],
                    },
                )
            )
            exported_surfaces[f"{hemi}.{metric_name}"] = str(metric_output)

        white = hemi_paths["white"]
        pial = hemi_paths["pial"]
        midthickness = (
            opts.out_dir
            / f"{opts.fs_subject}_space-fsnative_hemi-{hemi_label}_midthickness.surf.gii"
        )
        runner.add_step(
            _create_midthickness_step(
                white=white,
                pial=pial,
                output=midthickness,
                env=env,
                force=opts.force,
            )
        )
        runner.add_step(
            _write_json_step(
                midthickness.with_suffix(".json"),
                {
                    "Hemisphere": hemi_label,
                    "Space": "fsnative",
                    "AnatomicalStructurePrimary": "Cortex",
                    "SurfaceType": "midthickness",
                    "Sources": [str(white), str(pial)],
                },
            )
        )
        exported_surfaces[f"{hemi}.midthickness"] = str(midthickness)

    fsaverage_dir = find_fsaverage_directory(
        runner,
        environment=env,
        subjects_directory=opts.freesurfer_subjects_dir,
    )
    fsaverage_xfms: dict[str, str] = {}
    for hemi in ("lh", "rh"):
        hemi_label = _hemi_label(hemi)
        subject_registration = surface_dir / f"{hemi}.sphere.reg"
        fsaverage_sphere = fsaverage_dir / "surf" / f"{hemi}.sphere"
        forward = (
            opts.out_dir
            / f"{opts.fs_subject}_from-fsnative_to-fsaverage_hemi-{hemi_label}_mode-surface_xfm.surf.gii"
        )
        inverse = (
            opts.out_dir
            / f"{opts.fs_subject}_from-fsaverage_to-fsnative_hemi-{hemi_label}_mode-surface_xfm.surf.gii"
        )
        runner.add_step(
            _create_mri_conversion_step(
                source=subject_registration,
                output=forward,
                name=f"Export Hemisphere {hemi_label} fsnative-to-fsaverage Sphere",
                env=env,
                force=opts.force,
                executable="mris_convert",
            )
        )
        runner.add_step(
            _create_mri_conversion_step(
                source=fsaverage_sphere,
                output=inverse,
                name=f"Export Hemisphere {hemi_label} fsaverage-to-fsnative Sphere",
                env=env,
                force=opts.force,
                executable="mris_convert",
            )
        )
        for path, from_space, to_space, sources in (
            (
                forward,
                "fsnative",
                "fsaverage",
                (subject_registration, fsaverage_sphere),
            ),
            (
                inverse,
                "fsaverage",
                "fsnative",
                (fsaverage_sphere, subject_registration),
            ),
        ):
            runner.add_step(
                _write_json_step(
                    path.with_suffix(".json"),
                    {
                        "Type": "surface",
                        "Format": "FreeSurferSphereRegistration",
                        "Hemisphere": hemi_label,
                        "From": from_space,
                        "To": to_space,
                        "Sources": [str(source) for source in sources],
                    },
                )
            )
        fsaverage_xfms[f"hemi-{hemi_label}_fsnative_to_fsaverage"] = str(
            forward
        )
        fsaverage_xfms[f"hemi-{hemi_label}_fsaverage_to_fsnative"] = str(
            inverse
        )

    t1_fsaverage_xfms: dict[str, str] = {}
    for hemi_label in ("L", "R"):
        fsnative_to_fsaverage = Path(
            fsaverage_xfms[f"hemi-{hemi_label}_fsnative_to_fsaverage"]
        )
        fsaverage_to_fsnative = Path(
            fsaverage_xfms[f"hemi-{hemi_label}_fsaverage_to_fsnative"]
        )
        forward = (
            opts.out_dir
            / f"{opts.fs_subject}_from-T1w_to-fsaverage_hemi-{hemi_label}_mode-image+surface_xfm.json"
        )
        inverse = (
            opts.out_dir
            / f"{opts.fs_subject}_from-fsaverage_to-T1w_hemi-{hemi_label}_mode-surface+image_xfm.json"
        )
        runner.add_step(
            create_json_step(
                step_name=(
                    f"Write Hemisphere {hemi_label} T1w-to-fsaverage Transform Metadata"
                ),
                path=forward,
                payload={
                    "Type": "chain",
                    "Format": "surface",
                    "Hemisphere": hemi_label,
                    "From": "T1w",
                    "To": "fsaverage",
                    "Steps": [str(fsnative_to_fsaverage)],
                },
                inputs=(fsnative_to_fsaverage,),
                force=opts.force,
            )
        )
        runner.add_step(
            create_json_step(
                step_name=(
                    f"Write Hemisphere {hemi_label} fsaverage-to-T1w Transform Metadata"
                ),
                path=inverse,
                payload={
                    "Type": "chain",
                    "Format": "surface",
                    "Hemisphere": hemi_label,
                    "From": "fsaverage",
                    "To": "T1w",
                    "Steps": [str(fsaverage_to_fsnative)],
                },
                inputs=(fsaverage_to_fsnative,),
                force=opts.force,
            )
        )
        t1_fsaverage_xfms[f"hemi-{hemi_label}_t1_to_fsaverage"] = str(forward)
        t1_fsaverage_xfms[f"hemi-{hemi_label}_fsaverage_to_t1"] = str(inverse)

    mni_brain_template = Path(
        str(opts.mni_template).replace("_T1w.nii.gz", "_desc-brain_T1w.nii.gz")
    )
    mni_brain_mask = Path(
        str(opts.mni_template).replace("_T1w.nii.gz", "_desc-brain_mask.nii.gz")
    )
    mni_prefix = opts.work_dir / "ants_mni_" / f"{opts.fs_subject}_"
    t1_to_mni = (
        opts.out_dir
        / f"{opts.fs_subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )
    mni_to_t1 = (
        opts.out_dir
        / f"{opts.fs_subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )
    produced_t1_to_mni = mni_prefix.parent / f"{mni_prefix.name}Composite.h5"
    produced_mni_to_t1 = (
        mni_prefix.parent / f"{mni_prefix.name}InverseComposite.h5"
    )
    mni_registration_inputs = (
        subject_t1,
        Path(core_masks["brain"]),
        opts.mni_template,
        mni_brain_template,
        mni_brain_mask,
    )
    runner.add_step(
        _create_mni_registration_step(
            subject_t1=subject_t1,
            subject_brain_mask=Path(core_masks["brain"]),
            mni_brain_template=mni_brain_template,
            mni_brain_mask=mni_brain_mask,
            mni_template=opts.mni_template,
            prefix=mni_prefix,
            produced_forward=produced_t1_to_mni,
            produced_inverse=produced_mni_to_t1,
            env=env,
            force=opts.force,
        )
    )
    runner.add_step(
        _create_transform_finalization_step(
            produced_forward=produced_t1_to_mni,
            produced_inverse=produced_mni_to_t1,
            forward=t1_to_mni,
            inverse=mni_to_t1,
            force=opts.force,
        )
    )
    mni_xfms = {"t1_to_mni": str(t1_to_mni), "mni_to_t1": str(mni_to_t1)}
    for path, from_space, to_space in (
        (t1_to_mni, "T1w", "MNI152NLin2009cAsym"),
        (mni_to_t1, "MNI152NLin2009cAsym", "T1w"),
    ):
        runner.add_step(
            _write_json_step(
                path.with_suffix(".json"),
                {
                    "Type": "composite",
                    "Format": "ANTs",
                    "Sources": [str(value) for value in mni_registration_inputs],
                    "From": from_space,
                    "To": to_space,
                },
            )
        )

    mni_in_t1 = (
        opts.out_dir / f"{opts.fs_subject}_space-T1w_desc-mniToT1wQC.nii.gz"
    )
    t1_in_mni = (
        opts.out_dir
        / f"{opts.fs_subject}_space-MNI152NLin2009cAsym_desc-t1ToMNIQC.nii.gz"
    )
    mni_qc_images: dict[str, str] = {}
    for (
        key,
        output,
        input_image,
        reference,
        transform,
        from_space,
        to_space,
        description,
    ) in (
        (
            "mni_to_t1w",
            mni_in_t1,
            opts.mni_template,
            subject_t1,
            mni_to_t1,
            "MNI152NLin2009cAsym",
            "T1w",
            "Saved inverse composite transform applied in a single step to project the MNI template into subject T1w space for temporary QC.",
        ),
        (
            "t1w_to_mni",
            t1_in_mni,
            subject_t1,
            opts.mni_template,
            t1_to_mni,
            "T1w",
            "MNI152NLin2009cAsym",
            "Saved forward composite transform applied in a single step to project the preprocessed subject T1w image into MNI space for temporary QC.",
        ),
    ):
        runner.add_step(
            _create_mni_qc_image_step(
                input_image=input_image,
                reference=reference,
                transform=transform,
                output=output,
                from_space=from_space,
                to_space=to_space,
                env=env,
                force=opts.force,
            )
        )
        runner.add_step(
            _write_json_step(
                output.with_suffix("").with_suffix(".json"),
                {
                    "Type": "temporary QC image",
                    "From": from_space,
                    "To": to_space,
                    "Transform": str(transform),
                    "Sources": [str(input_image), str(reference)],
                    "Description": description,
                },
            )
        )
        mni_qc_images[key] = str(output)
    manifest = {
        "subject": inputs.sub_id,
        "fs_subject": opts.fs_subject,
        "selection_strategy": opts.selection_strategy,
        "inputs": {
            "t1w": [str(item.image) for item in inputs.t1w],
            "t2w": [str(item.image) for item in inputs.t2w],
        },
        "copied_session_files": copied_session_files,
        "outputs": {
            "subject_t1w": (str(subj_t1) if subj_t1 is not None else None),
            "subject_t2w": (str(subj_t2) if subj_t2 is not None else None),
            "myelin_map": (str(myelin_map) if myelin_map is not None else None),
            "brain_image": str(subject_anat),
            "brain_mask": core_masks["brain"],
            "gray_matter_mask": core_masks["gray_matter"],
            "cortical_ribbon_mask": core_masks["cortical_ribbon"],
            "subcortical_masks": subcortical_masks,
            "surfaces": exported_surfaces,
            "mni_qc_images": mni_qc_images,
            "xfms": {**fsnative_xfms, **fsaverage_xfms, **t1_fsaverage_xfms, **mni_xfms},
        },
        "freesurfer_subjects_dir": str(opts.freesurfer_subjects_dir),
        "mni_template": str(opts.mni_template),
        "options": {
            "synthstrip_image": (
                str(opts.synthstrip_image) if opts.synthstrip_image is not None else None
            ),
            "configuration": {
                "selection_strategy": opts.selection_strategy,
                "fs_subject": opts.fs_subject,
                "mni_template": str(opts.mni_template),
                "synthstrip_image": (
                    str(opts.synthstrip_image) if opts.synthstrip_image is not None else None
                ),
            },
            "configuration_fingerprint": selected_configuration_fingerprint(),
        },
        "complete": True,
    }
    manifest_path = anatomical_manifest_path(
        inputs.sub_id,
        project=opts.project,
        preprocessing_id=opts.preprocessing_id,
    )
    published_outputs = _nested_paths(manifest["outputs"]) + tuple(
        Path(path) for path in copied_session_files
    )

    def validate_publication() -> tuple[bool, str]:
        try:
            current = read_json(manifest_path)
        except (OSError, ValueError, TypeError):
            return False, "Anatomical publication manifest is missing or unreadable."
        if current != manifest:
            return False, "Anatomical publication manifest differs from the requested module."
        missing = [
            str(path)
            for path in published_outputs
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "Anatomical publication is missing outputs: " + ", ".join(missing)
        return True, "Anatomical publication is complete and current."

    runner.add_step(Step.python(
        name="Write Anatomical Publication Manifest",
        outputs=(manifest_path,),
        inputs=(*published_outputs, *public_inputs),
        force=opts.force,
        action=lambda: write_json(manifest_path, manifest),
        validate=validate_publication,
        completion_boundary=True,
    ))

    return runner


def run(inputs: Inputs, opts: Options) -> None:
    runner_started = time.perf_counter()
    runner = build_module(inputs, opts)
    with runner.run_context(started_at=runner_started):
        runner.execute()


def _build_argparser() -> argparse.ArgumentParser:
    cfg = SETTINGS.preprocess_anat
    p = argparse.ArgumentParser(prog="nro.anat.module")
    p.add_argument("--project", default=SETTINGS.common.project)
    p.add_argument("--preprocessing-id", default=SETTINGS.common.preprocessing_id)
    p.add_argument("--sub-id", required=True)
    p.add_argument("--fs-subject", default=cfg.fs_subject)
    p.add_argument("--t1w", action="append", default=[], type=Path)
    p.add_argument("--t2w", action="append", default=[], type=Path)
    p.add_argument("--selection-strategy", choices=["first", "robust_average"], default=cfg.selection_strategy)
    p.add_argument("--mni-template", type=Path, default=cfg.mni_template)
    p.add_argument("--synthstrip-container", type=Path, default=cfg.synthstrip_container)
    p.add_argument("--freesurfer-subjects-dir", type=Path, default=cfg.freesurfer_subjects_dir)
    p.add_argument("--out-dir", type=Path, default=cfg.out_dir)
    p.add_argument("--work-dir", type=Path, default=cfg.work_dir)
    p.add_argument("--nthreads", type=int, default=max(int(cfg.nthreads_min), (os.cpu_count() or 1) // int(cfg.nthreads_divisor)))
    p.add_argument("--force", action="store_true", default=cfg.force)
    p.add_argument("--verbose", action="store_true", default=cfg.verbose)
    p.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    p.add_argument("--no-container", action="store_true", default=cfg.no_container)
    p.add_argument("--container-engine", default=cfg.container_engine)
    p.add_argument("--container-no-cleanenv", action="store_true", default=not bool(cfg.container_cleanenv))
    p.add_argument("--container-bind", action="append", default=list(cfg.container_bind))
    p.add_argument("--container-home", type=Path, default=cfg.container_home)
    p.add_argument("--container-inner-setup", default=cfg.container_inner_setup)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    project = str(args.project)
    t1w = [load_anat_image(resolve_project_path(path, project=project)) for path in args.t1w]
    t2w = [load_anat_image(resolve_project_path(path, project=project)) for path in args.t2w]
    out_dir = resolve_project_path(
        args.out_dir or preprocessing_subject_anat_dir(str(args.sub_id), project=project, preprocessing_id=str(args.preprocessing_id)),
        project=project,
    )
    work_dir = resolve_project_work_path(
        args.work_dir or (preprocessing_subject_work_dir(str(args.sub_id), project=project, preprocessing_id=str(args.preprocessing_id)) / "anat"),
        project=project,
    )
    subjects_dir = resolve_project_path(
        args.freesurfer_subjects_dir or (preprocessing_derivatives_root(project=project, preprocessing_id=str(args.preprocessing_id)) / "code" / "freesurfer"),
        project=project,
    )
    mni_template = resolve_project_path(args.mni_template, project=project)
    if mni_template is None:
        raise SystemExit("Missing --mni-template path.")
    require_nonempty_file(mni_template, "MNI template")
    synthstrip_image = resolve_project_path(args.synthstrip_container, project=project)
    if synthstrip_image is None:
        raise SystemExit("Missing --synthstrip-container path.")
    container: Optional[ContainerSpec]
    if args.no_container:
        container = None
    else:
        container_home = args.container_home or (work_dir / "_qunex_home")
        container = ContainerSpec(
            image=Path(args.container),
            engine=str(args.container_engine),
            cleanenv=not bool(args.container_no_cleanenv),
            extra_binds=tuple(args.container_bind or []),
            home_dir=Path(container_home),
            inner_setup=str(args.container_inner_setup or ""),
        )
    run(
        Inputs(sub_id=str(args.sub_id), t1w=tuple(t1w), t2w=tuple(t2w)),
        Options(
            project=project,
            preprocessing_id=str(args.preprocessing_id),
            out_dir=out_dir,
            work_dir=work_dir,
            freesurfer_subjects_dir=subjects_dir,
            fs_subject=str(args.fs_subject or args.sub_id),
            selection_strategy=str(args.selection_strategy),
            mni_template=mni_template,
            container=container,
            synthstrip_image=synthstrip_image,
            force=bool(args.force),
            nthreads=int(args.nthreads),
        ),
    )


if __name__ == "__main__":
    main()
