"""Build anatomical preprocessing steps and their validation helpers."""

import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from nro.engine.execution import (
    create_copy_file_step,
    ensure_directory,
)
from nro.engine.image_paths import sidecar_json_path
from nro.engine.io import invalid_gzip_files, write_json
from nro.engine.manifests import create_json_step
from nro.engine.paths import (
    anat_session_dir,
    anat_session_work_dir,
    module_artifact_root,
)
from nro.engine.pose import create_pose_normalized_grid, invert_itk_affine, pose_quality
from nro.modules.anat.inputs import AnatImage, sort_anat_images
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

from .constants import (
    _FREESURFER_ASEG_LABELS,
    _FS_GIFTI_VOLGEOM_META_PREFIXES,
)
from .policy import FREESURFER_BUILD, FREESURFER_VERSION

_FREESURFER_STATUS_TIMESTAMP = re.compile(
    r"\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\d{4}$"
)


def _freesurfer_progress(marker: str) -> tuple[int, str]:
    marker = _FREESURFER_STATUS_TIMESTAMP.sub("", marker.removeprefix("#@# ").strip())
    lowered = marker.lower()
    hemisphere = ""
    if re.search(r"(?:^|\s)lh(?:\s|$)", lowered):
        hemisphere = ", left hemisphere"
    elif re.search(r"(?:^|\s)rh(?:\s|$)", lowered):
        hemisphere = ", right hemisphere"

    if any(
        token in lowered
        for token in (
            "em registration",
            "ca normalize",
            "ca reg",
            "subcort seg",
            "cc seg",
            "merge aseg",
            "intensity normalization2",
            "mask bfs",
            "wm segmentation",
            "fill",
        )
    ):
        phase = 2
    elif any(
        token in lowered
        for token in ("tessellate", "smooth1", "inflation1", "qsphere", "fix topology")
    ):
        phase = 3
    elif any(
        token in lowered
        for token in (
            "cortical parc 2",
            "cortical parc 3",
            "cortical parcellation 2",
            "cortical parcellation 3",
            "relabel hypointensities",
            "apas-to-aseg",
            "aparc-to-aseg",
            "wmparc",
        )
    ):
        phase = 6
    elif any(
        token in lowered
        for token in (
            "smooth2",
            "inflation2",
            "curv .h and .k",
            "sphere",
            "surf reg",
            "jacobian",
            "avgcurv",
            "cortical parc",
        )
    ):
        phase = 4
    elif any(
        token in lowered
        for token in (
            "refine pial",
            "white curv",
            "pial curv",
            "thickness",
            "area and vertex vol",
            "cortical ribbon",
        )
    ):
        phase = 5
    elif any(
        token in lowered
        for token in (
            "motioncor",
            "talairach",
            "nu intensity",
            "normalization",
            "skull strip",
            "t2/flair input",
        )
    ):
        phase = 1
    elif any(
        token in lowered
        for token in (
            "parcellation stats",
            "aseg stats",
            "ba_exvivo",
            "recon-all done",
        )
    ):
        phase = 7
    else:
        phase = 1
    return phase, f"{marker}{hemisphere}"


def _format_progress_elapsed(seconds: float) -> str:
    minutes = max(0, int(seconds)) // 60
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class _FreeSurferProgressMonitor:
    def __init__(
        self,
        subject_dir: Path,
        *,
        poll_seconds: float = 2.0,
        heartbeat_seconds: float = 300.0,
    ) -> None:
        self._status = subject_dir / "scripts" / "recon-all-status.log"
        self._detail = subject_dir / "scripts" / "recon-all.log"
        self._poll_seconds = poll_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._logger = logging.getLogger("anat")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen = 0
        self._current: tuple[int, str] | None = None
        self._started = time.monotonic()
        self._last_report = self._started

    def __enter__(self) -> "_FreeSurferProgressMonitor":
        self._logger.info("FreeSurfer detailed log: %s", self._detail)
        self._thread = threading.Thread(
            target=self._watch,
            name="nro-freesurfer-progress",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._poll_seconds * 2))
        self._consume()

    def report(self, phase: int, detail: str) -> None:
        if self._current is not None:
            phase = max(phase, self._current[0])
        self._current = (phase, detail)
        self._last_report = time.monotonic()
        self._logger.info("FreeSurfer %d/7: %s", phase, detail)

    def _consume(self) -> None:
        try:
            lines = self._status.read_text(errors="replace").splitlines()
        except OSError:
            return
        if len(lines) < self._seen:
            self._seen = 0
        new_lines = lines[self._seen :]
        self._seen = len(lines)
        for line in new_lines:
            if line.startswith("#@# "):
                self.report(*_freesurfer_progress(line))

    def _watch(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            self._consume()
            now = time.monotonic()
            if self._current is None or now - self._last_report < self._heartbeat_seconds:
                continue
            phase, detail = self._current
            self._logger.info(
                "FreeSurfer %d/7: %s; still running after %s",
                phase,
                detail,
                _format_progress_elapsed(now - self._started),
            )
            self._last_report = now


def _robust_template_command(
    inputs: Sequence[Path], out_template: Path, out_transform_prefix: Path
) -> list[str]:
    """Build an ``mri_robust_template`` command for anatomical averaging."""
    command = [
        "mri_robust_template",
        "--template",
        str(out_template),
        "--satit",
        "--mapmov",
        str(out_transform_prefix),
    ]
    for path in inputs:
        command.extend(("--mov", str(path)))
    return command


@dataclass(frozen=True)
class _SessionAnatomicalPlan:
    source: AnatImage
    staged_raw: Path
    staged_preprocessed: Path
    output: Path
    mask: Path
    metadata_output: Path
    metadata: dict[str, object]


def _plan_session_anatomicals(
    images: Sequence[AnatImage],
    *,
    project: str,
    anat_id: str,
    sub_id: str,
    execution_context: ExecutionContext | None = None,
) -> tuple[_SessionAnatomicalPlan, ...]:
    by_session: dict[str, list[AnatImage]] = {}
    for image in images:
        by_session.setdefault(image.session_id, []).append(image)

    plans: list[_SessionAnatomicalPlan] = []
    for session_id, session_images in by_session.items():
        if execution_context is None:
            output_dir = anat_session_dir(sub_id, session_id, project=project, anat_id=anat_id)
            work_dir = (
                anat_session_work_dir(sub_id, session_id, project=project, anat_id=anat_id)
                / "session_level"
            )
        else:
            output_root = module_artifact_root(
                execution_context.paths.output_project(project), "anat", anat_id
            )
            work_root = module_artifact_root(
                execution_context.paths.private_project(project), "anat", anat_id
            )
            output_dir = output_root / sub_id / session_id / "anat"
            work_dir = work_root / sub_id / session_id / "anat" / "session_level"
            execution_context.require_output(output_dir)
            execution_context.require_output(work_dir)
        staged = {
            image.image: work_dir
            / f"{image.image.name.removesuffix('.nii.gz').removesuffix('.nii')}_desc-preproc_{image.modality}.nii.gz"
            for image in session_images
        }
        for image in session_images:
            payload: dict[str, object] = dict(image.metadata)
            payload["Sources"] = [str(image.image)]
            payload["BiasCorrection"] = "N4BiasFieldCorrection"
            staged_preprocessed = staged[image.image]
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
                    staged_preprocessed=staged_preprocessed,
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
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
        parameters={"brain_extraction": "mri_synthstrip", "image": synthstrip_image},
    )


def _create_apply_brain_mask_step(
    *,
    source: Path,
    mask: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Apply a tracked binary brain mask without estimating it again."""
    return Step.command_step(
        ["fslmaths", str(source), "-mas", str(mask), str(output)],
        name="Apply Anatomical Brain Mask",
        outputs=(output,),
        inputs=(source, mask),
        force=force,
        env=env,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
        parameters={"mask_application": "binary_multiplication"},
    )


def _create_finalize_anatomy_step(
    *,
    source: Path,
    mask: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Restore a crisp brain boundary after continuous pose resampling."""
    return Step.command_step(
        ["fslmaths", str(source), "-mas", str(mask), str(output)],
        name="Finalize Pose-Normalized Anatomical Image",
        outputs=(output,),
        inputs=(source, mask),
        force=force,
        env=env,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
        parameters={"mask_application": "nearest_neighbor_binary_mask"},
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
            parameters={"selection_strategy": strategy},
        )
    else:
        tmp_dir = work_dir / f"robust_template_{modality}"
        cmd = _robust_template_command(
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
                out_img.parent.mkdir(parents=True, exist_ok=True),
                out_img.unlink(missing_ok=True),
            ),
            validate=validate,
            parameters={"selection_strategy": strategy},
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
        name="Register T2w to T1w Reference",
        env=env,
        outputs=(out_t2, out_mat),
        inputs=(t2_src, t1_ref),
        force=force,
        prepare=lambda: (
            out_t2.parent.mkdir(parents=True, exist_ok=True),
            out_mat.parent.mkdir(parents=True, exist_ok=True),
        ),
    )


def _create_pose_registration_step(
    *,
    source: Path,
    source_mask: Path,
    template: Path,
    template_mask: Path,
    prefix: Path,
    transform: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Estimate the six-degree-of-freedom transform into the standardized pose."""
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
            "0",
            "--output",
            f"[{prefix}]",
            "--interpolation",
            "BSpline[3]",
            "--use-histogram-matching",
            "1",
            "--winsorize-image-intensities",
            "[0.005,0.995]",
            "--masks",
            f"[{template_mask},{source_mask}]",
            "--initial-moving-transform",
            f"[{template},{source},1]",
            "--transform",
            "Rigid[0.1]",
            "--metric",
            f"MI[{template},{source},1,32,Regular,0.25]",
            "--convergence",
            "[1000x500x250x0,1e-6,10]",
            "--shrink-factors",
            "8x4x2x1",
            "--smoothing-sigmas",
            "3x2x1x0vox",
        ],
        name="Rigid Anatomical Pose Registration",
        env=env,
        outputs=(transform,),
        inputs=(source, source_mask, template, template_mask),
        force=force,
        prepare=lambda: prefix.parent.mkdir(parents=True, exist_ok=True),
        parameters={
            "degrees_of_freedom": 6,
            "metric": "mutual_information",
            "template_space": "MNI152NLin2009cAsym",
        },
    )


def _create_pose_grid_step(
    *,
    source: Path,
    source_mask: Path,
    template: Path,
    transform: Path,
    output: Path,
    force: bool,
    margin_mm: float = 5.0,
) -> Step:
    """Declare the source-resolution, template-oriented anatomical grid."""
    return Step.python(
        name="Construct Pose-Normalized Anatomical Grid",
        outputs=(output,),
        inputs=(source, source_mask, template, transform),
        force=force,
        action=lambda: create_pose_normalized_grid(
            source, source_mask, template, transform, output, margin_mm=margin_mm
        ),
        parameters={
            "extent": "anatomical_mask",
            "margin_mm": float(margin_mm),
            "resolution": "source",
            "transform_direction": "moving_to_fixed",
        },
    )


def _create_pose_resampling_step(
    *,
    source: Path,
    reference: Path,
    transform: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
    label: bool = False,
) -> Step:
    """Apply the rigid pose transform once on the declared anatomical grid."""
    return Step.command_step(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(source),
            "-r",
            str(reference),
            "-o",
            str(output),
            "-n",
            "NearestNeighbor" if label else "BSpline[3]",
            "-t",
            str(transform),
        ],
        name=(
            "Resample Anatomical Mask to Reference" if label else "Resample Anatomy to Reference"
        ),
        env=env,
        outputs=(output,),
        inputs=(source, reference, transform),
        force=force,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
        parameters={"interpolation": "nearest" if label else "cubic_bspline"},
    )


def _create_pose_inverse_step(*, source: Path, output: Path, force: bool) -> Step:
    """Declare the exact inverse of the rigid pose affine."""
    return Step.python(
        name="Invert Anatomical Pose Transform",
        outputs=(output,),
        inputs=(source,),
        force=force,
        action=lambda: invert_itk_affine(source, output),
    )


def _create_pose_qc_step(
    *,
    source: Path,
    source_mask: Path,
    aligned: Path,
    template: Path,
    transform: Path,
    grid: Path,
    output: Path,
    force: bool,
) -> Step:
    """Validate rigid geometry and record compact pose-alignment measures."""

    def action() -> None:
        write_json(output, pose_quality(source, source_mask, aligned, template, transform, grid))

    return Step.python(
        name="Validate Anatomical Pose Alignment",
        outputs=(output,),
        inputs=(source, source_mask, aligned, template, transform, grid),
        force=force,
        action=action,
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
    brain_mask: Path,
    subjects_dir: Path,
    fs_subject: str,
    runtime: str,
    image: Path,
    license_file: Path,
    force: bool,
) -> Step:
    if t1w is None and t2w is None:
        raise SystemExit(
            "Need at least one subject-level T1w or T2w image for anatomical preprocessing."
        )
    subject_dir = subjects_dir / fs_subject
    recon_done = subject_dir / "scripts" / "recon-all.done"
    breadcrumb = _recon_breadcrumb(subject_dir)

    def container_command(command: Sequence[str]) -> list[str]:
        binds = [
            f"{subjects_dir}:/subjects",
            f"{license_file}:/license.txt:ro",
        ]
        inputs = [path for path in (t1w, t2w, brain_mask) if path is not None]
        mounted: dict[Path, str] = {}
        for index, path in enumerate(inputs, start=1):
            parent = path.parent.resolve()
            if parent not in mounted:
                mounted[parent] = f"/input-{index}"
                binds.append(f"{parent}:{mounted[parent]}:ro")

        def translated(path: Path) -> str:
            return f"{mounted[path.parent.resolve()]}/{path.name}"

        translated_command = [
            translated(Path(item[4:])) if item.startswith("NRO:/") else item for item in command
        ]
        setup = (
            "export FREESURFER_HOME=/usr/local/freesurfer; "
            "export SUBJECTS_DIR=/subjects; export FS_LICENSE=/license.txt; "
            "source /usr/local/freesurfer/SetUpFreeSurfer.sh >/dev/null; "
            "export TMPDIR=/tmp; export TMP=/tmp; export TEMP=/tmp; "
            'exec "$@"'
        )
        return [
            runtime,
            "exec",
            "--cleanenv",
            "--bind",
            ",".join(binds),
            str(image),
            "bash",
            "-lc",
            setup,
            "bash",
            *translated_command,
        ]

    def validate() -> tuple[bool, str]:
        if _recon_valid(subject_dir):
            return True, "FreeSurfer recon-all outputs are complete."
        return False, _recon_invalid_reason(subject_dir)

    def action() -> None:
        primary = t1w or t2w
        assert primary is not None
        primary_arg = f"NRO:{primary}"
        first = [
            "recon-all",
            "-sd",
            "/subjects",
            "-subjid",
            fs_subject,
            "-i",
            primary_arg,
            "-autorecon1",
            "-noskullstrip",
        ]
        with _FreeSurferProgressMonitor(subject_dir) as progress:
            run_child(container_command(first), direct=True, env=env, discard_stdout=True)
            progress.report(2, "applying external brain mask")
            mask_arg = f"NRO:{brain_mask}"
            conformed_mask = f"/subjects/{fs_subject}/mri/brainmask.external.mgz"
            brainmask_auto = f"/subjects/{fs_subject}/mri/brainmask.auto.mgz"
            conformed_t1 = f"/subjects/{fs_subject}/mri/T1.mgz"
            run_child(
                container_command(
                    [
                        "mri_vol2vol",
                        "--mov",
                        mask_arg,
                        "--targ",
                        conformed_t1,
                        "--regheader",
                        "--interp",
                        "nearest",
                        "--o",
                        conformed_mask,
                        "--no-save-reg",
                    ]
                ),
                direct=True,
                env=env,
                discard_stdout=True,
            )
            run_child(
                container_command(["mri_mask", conformed_t1, conformed_mask, brainmask_auto]),
                direct=True,
                env=env,
                discard_stdout=True,
            )
            shutil.copy2(
                subject_dir / "mri" / "brainmask.auto.mgz",
                subject_dir / "mri" / "brainmask.mgz",
            )
            second = [
                "recon-all",
                "-sd",
                "/subjects",
                "-subjid",
                fs_subject,
            ]
            if t2w is not None and t1w is not None:
                second += ["-T2", f"NRO:{t2w}", "-T2pial"]
            second += ["-autorecon2", "-autorecon3", "-noskullstrip"]
            run_child(container_command(second), direct=True, env=env, discard_stdout=True)
        if not _recon_valid(subject_dir):
            raise SystemExit(f"recon-all completed without a valid output set under: {subject_dir}")

    return Step.directory_step(
        name="FreeSurfer Recon-All",
        directory=subject_dir,
        breadcrumb=breadcrumb,
        inputs=(
            *tuple(path for path in (t1w, t2w) if path is not None),
            brain_mask,
            image,
            license_file,
        ),
        outputs=(*_recon_required_outputs(subject_dir), recon_done),
        action=action,
        validate=validate,
        force=force,
        breadcrumb_text="recon-all complete\n",
        parameters={
            "backend": "FreeSurfer",
            "version": FREESURFER_VERSION,
            "build": FREESURFER_BUILD,
            "skull_stripping": "SynthStrip_external_mask",
            "external_mask_resampling": "nearest_neighbor",
            "brainmask_intensity_source": "FreeSurfer_normalized_T1",
        },
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
    *,
    source: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
    name: str = "Invert T1w-to-fsnative Affine",
) -> Step:
    return Step.command_step(
        ["convert_xfm", "-omat", str(output), "-inverse", str(source)],
        name=name,
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
    hemisphere = surface.name.split(".", 1)[0]
    converted = output.with_name(f"{hemisphere}.{output.name}")

    def prepare() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        converted.unlink(missing_ok=True)

    def finalize() -> None:
        os.replace(converted, output)

    return Step.command_step(
        ["mris_convert", "-c", str(metric), str(surface), str(converted)],
        name=f"Convert FreeSurfer {description.title()} Metric",
        env=env,
        outputs=(output,),
        inputs=(metric, surface),
        force=force,
        prepare=prepare,
        finalize=finalize,
    )


def _create_metric_structure_step(
    *,
    source: Path,
    output: Path,
    structure: str,
    description: str,
    force: bool,
) -> Step:
    """Assign a cortical hemisphere to a converted GIFTI metric."""

    def action() -> None:
        import nibabel as nib  # type: ignore

        image = nib.load(str(source))
        if not isinstance(image, nib.GiftiImage):
            raise ValueError(f"Expected a GIFTI metric: {source}")
        image.meta["AnatomicalStructurePrimary"] = structure
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".partial-{output.name}")
        temporary.unlink(missing_ok=True)
        nib.save(image, str(temporary))
        os.replace(temporary, output)

    def validate() -> tuple[bool, str]:
        try:
            import nibabel as nib  # type: ignore
            import numpy as np

            source_image = nib.load(str(source))
            output_image = nib.load(str(output))
            if not isinstance(source_image, nib.GiftiImage) or not isinstance(
                output_image, nib.GiftiImage
            ):
                raise ValueError("source or output is not GIFTI")
            if output_image.meta.get("AnatomicalStructurePrimary") != structure:
                raise ValueError(f"output is not assigned to {structure}")
            if len(source_image.darrays) != len(output_image.darrays) or any(
                not np.array_equal(source_array.data, output_array.data)
                for source_array, output_array in zip(
                    source_image.darrays, output_image.darrays, strict=True
                )
            ):
                raise ValueError("metric data changed while assigning its structure")
        except (OSError, ValueError, TypeError, nib.filebasedimages.ImageFileError) as error:
            return False, f"Structured {description} metric is absent or invalid: {error}"
        return True, f"{description.title()} metric is assigned to {structure}."

    return Step.python(
        name=f"Assign {description.title()} Metric Structure",
        inputs=(source,),
        outputs=(output,),
        action=action,
        validate=validate,
        force=force,
        parameters={"anatomical_structure_primary": structure},
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
            "BSpline[3]",
            "-t",
            str(transform),
        ],
        name=f"Write {from_space}-to-{to_space} QC Image",
        env=env,
        outputs=(output,),
        inputs=(input_image, reference, transform),
        force=force,
        prepare=lambda: output.parent.mkdir(parents=True, exist_ok=True),
        parameters={"interpolation": "cubic_bspline", "role": "registration_qc"},
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
