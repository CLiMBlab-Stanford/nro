"""Run anatomical preprocessing for one subject."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from nro.definitions.markup import active_source_markup
from nro.definitions.runtime import SETTINGS
from nro.engine.container import (
    ContainerSettings,
    add_container_arguments,
    bind_container_to_execution,
    build_container,
)
from nro.engine.execution import (
    collect_bind_directories,
    neuroimaging_environment,
)
from nro.engine.io import (
    require_nonempty_file,
)
from nro.engine.paths import (
    anat_subject_dir,
    anat_subject_work_dir,
    module_derivatives_root,
    resolve_project_path,
    resolve_project_work_path,
)
from nro.modules.anat.construction import (
    plan_fsaverage_registration,
    plan_initialization,
    plan_lesion_reconstruction_inputs,
    plan_lesion_surfaces,
    plan_mni_registration,
    plan_pose_normalization,
    plan_publication,
    plan_reference_outputs,
    plan_session_staging,
    plan_subject_selection,
    plan_surface_export,
    plan_surface_reconstruction,
    plan_template_transforms,
)
from nro.modules.anat.graph_policy import resolve_graph_policy
from nro.modules.anat.inputs import (
    AnatImage,
    load_anat_image,
)
from nro.modules.anat.layout import AnatLayout
from nro.orchestration.execution_context import ExecutionContext
from nro.orchestration.runner import ContainerSpec, Runner
from nro.site.configuration import settings as site_settings

from .msmall import MsmAllCalibration, add_msmall_plan, resolve_msmall_calibration
from .policy import (
    T1W_FALLBACK_DEFAULT,
)
from .steps import (
    _plan_session_anatomicals,
)

LOG = logging.getLogger("anat")
# Standard aseg identifiers from FreeSurferColorLUT.txt:
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT


@dataclass(frozen=True)
class Inputs:
    """Subject identity and available T1w/T2w acquisitions for anatomical construction."""

    sub_id: str
    t1w: tuple[AnatImage, ...]
    t2w: tuple[AnatImage, ...]


@dataclass(frozen=True)
class Options:
    """Resolved anatomy resources, output paths, selection strategy, and execution controls."""

    project: str
    anat_id: str
    out_dir: Path
    work_dir: Path
    freesurfer_subjects_dir: Path
    fs_subject: str
    fsaverage_template: str
    selection_strategy: str
    gradient_unwarping: str
    gradient_unwarp_image: Path
    container_runtime: str
    mni_template: Path
    container: Optional[ContainerSpec]
    synthstrip_image: Optional[Path]
    overwrite: bool
    surface_reconstruction_engine: str
    freesurfer_image: Optional[Path] = None
    lesion: bool = False
    lesion_masker_command: Optional[Path] = None
    synthstroke_data: Optional[Path] = None
    fastsurfer_image: Optional[Path] = None
    msmall: MsmAllCalibration | None = None
    t1w_fallback: str = T1W_FALLBACK_DEFAULT


def build_module(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> Runner:
    """Construct the anatomical DAG, optionally routing writes to a selected owner.

    The caller authorizes the context. Raw inputs stay in the shared BIDS tree;
    public outputs, session copies, FreeSurfer files, and work stay with the owner.
    """
    policy = resolve_graph_policy(inputs, opts, execution_context=execution_context)
    opts = policy.opts
    all_images = list(policy.all_images)
    synthetic_t1w = policy.synthetic_t1w
    lesion_masker_command = policy.lesion_masker_command
    lesion_masker_module = policy.lesion_masker_module
    synthstroke_data = policy.synthstroke_data
    mni_brain_template = policy.mni_brain_template
    mni_brain_mask = policy.mni_brain_mask
    gradient_resolutions = policy.gradient_resolutions
    public_inputs = list(policy.public_inputs)
    layout = AnatLayout.for_options(opts, inputs.sub_id)
    env = neuroimaging_environment(subjects_dir=opts.freesurfer_subjects_dir)
    require_nonempty_file(Path(env["FS_LICENSE"]), "FreeSurfer license")
    binds = collect_bind_directories(policy.bind_candidates(env))
    runner = Runner(
        module_name="Anatomical Module",
        container=bind_container_to_execution(opts.container, execution_context),
        binds=binds,
        logger=LOG,
        execution_context=execution_context,
    )
    derivative_root = module_derivatives_root(
        "anat",
        opts.anat_id,
        project=opts.project,
        bids_root=None if execution_context is None else execution_context.paths.bids,
    )
    if execution_context is not None:
        derivative_root = execution_context.output_path(derivative_root)

    initialization = plan_initialization(
        opts=opts,
        layout=layout,
        synthetic_t1w=synthetic_t1w,
        derivative_root=derivative_root,
        require_commands=runner.require_cmds,
    )
    runner.add_steps(initialization.steps)

    session_plans = _plan_session_anatomicals(
        images=all_images,
        project=opts.project,
        anat_id=opts.anat_id,
        sub_id=inputs.sub_id,
        execution_context=execution_context,
    )
    session_staging = plan_session_staging(
        opts=opts,
        session_plans=session_plans,
        gradient_resolutions=gradient_resolutions,
        env=env,
        run_child=runner.run_child,
    )
    runner.add_steps(session_staging.steps)
    gradient_corrected_sources = session_staging.products.gradient_corrected_sources
    selection = plan_subject_selection(
        opts=opts,
        inputs=inputs,
        layout=layout,
        session_plans=session_plans,
        gradient_corrected_sources=gradient_corrected_sources,
        synthetic_t1w=synthetic_t1w,
        env=env,
        run_child=runner.run_child,
    )
    runner.add_steps(selection.steps)
    copied_session_files = selection.products.copied_session_files
    lesion_raw_t1_selected = selection.products.lesion_raw_t1_selected
    subj_t1_selected = selection.products.subj_t1_selected
    subj_t2_selected = selection.products.subj_t2_selected
    t1_meta = selection.products.t1_meta
    t2_meta = selection.products.t2_meta
    pose_source = selection.products.pose_source
    pose_modality = selection.products.pose_modality
    pose_mask = selection.products.pose_mask
    pose_normalization = plan_pose_normalization(
        env=env,
        inputs=inputs,
        layout=layout,
        mni_brain_mask=mni_brain_mask,
        mni_brain_template=mni_brain_template,
        opts=opts,
        pose_mask=pose_mask,
        pose_modality=pose_modality,
        pose_source=pose_source,
        subj_t1_selected=subj_t1_selected,
        subj_t2_selected=subj_t2_selected,
    )
    runner.add_steps(pose_normalization.steps)
    pose_output = pose_normalization.products.pose_output
    published_source_to_reference = pose_normalization.products.published_source_to_reference
    reference_grid = pose_normalization.products.reference_grid
    reference_mask = pose_normalization.products.reference_mask
    reference_space = pose_normalization.products.reference_space
    reference_to_source = pose_normalization.products.reference_to_source
    source_to_reference = pose_normalization.products.source_to_reference
    subj_t1 = pose_normalization.products.subj_t1
    subj_t2 = pose_normalization.products.subj_t2
    reference_outputs = plan_reference_outputs(
        env=env,
        inputs=inputs,
        layout=layout,
        lesion_raw_t1_selected=lesion_raw_t1_selected,
        opts=opts,
        pose_mask=pose_mask,
        pose_output=pose_output,
        pose_source=pose_source,
        published_source_to_reference=published_source_to_reference,
        reference_grid=reference_grid,
        reference_space=reference_space,
        reference_to_source=reference_to_source,
        source_to_reference=source_to_reference,
        subj_t1=subj_t1,
        subj_t2=subj_t2,
        subj_t2_selected=subj_t2_selected,
        synthetic_t1w=synthetic_t1w,
        t1_meta=t1_meta,
        t2_meta=t2_meta,
    )
    runner.add_steps(reference_outputs.steps)
    lesion_synthsr_input = reference_outputs.products.lesion_synthsr_input
    myelin_map = reference_outputs.products.myelin_map
    pose_qc = reference_outputs.products.pose_qc
    pose_xfms = reference_outputs.products.pose_xfms
    subject_anat = reference_outputs.products.subject_anat
    t2w_to_t1w_xfms = reference_outputs.products.t2w_to_t1w_xfms

    lesion_reconstruction_inputs = plan_lesion_reconstruction_inputs(
        env=env,
        inputs=inputs,
        lesion_masker_command=lesion_masker_command,
        lesion_masker_module=lesion_masker_module,
        lesion_raw_t1_selected=lesion_raw_t1_selected,
        lesion_synthsr_input=lesion_synthsr_input,
        opts=opts,
        reference_mask=reference_mask,
        subj_t1=subj_t1,
        subj_t2=subj_t2,
        synthstroke_data=synthstroke_data,
        run_child=runner.run_child,
    )
    runner.add_steps(lesion_reconstruction_inputs.steps)
    lesion_mask = lesion_reconstruction_inputs.products.lesion_mask
    lesion_metadata = lesion_reconstruction_inputs.products.lesion_metadata
    lesion_probability = lesion_reconstruction_inputs.products.lesion_probability
    lesion_qc = lesion_reconstruction_inputs.products.lesion_qc
    lesion_reconstruction_summary = (
        lesion_reconstruction_inputs.products.lesion_reconstruction_summary
    )
    reconstruction_mask = lesion_reconstruction_inputs.products.reconstruction_mask
    reconstruction_t1 = lesion_reconstruction_inputs.products.reconstruction_t1
    reconstruction_t2 = lesion_reconstruction_inputs.products.reconstruction_t2
    synthetic_scaffold_t1 = lesion_reconstruction_inputs.products.synthetic_scaffold_t1
    surface_reconstruction = plan_surface_reconstruction(
        derivative_root=derivative_root,
        env=env,
        inputs=inputs,
        layout=layout,
        lesion_mask=lesion_mask,
        lesion_reconstruction_summary=lesion_reconstruction_summary,
        opts=opts,
        reconstruction_mask=reconstruction_mask,
        reconstruction_t1=reconstruction_t1,
        reconstruction_t2=reconstruction_t2,
        subj_t1=subj_t1,
        subject_anat=subject_anat,
        synthetic_scaffold_t1=synthetic_scaffold_t1,
        run_child=runner.run_child,
    )
    runner.add_steps(surface_reconstruction.steps)
    core_masks = surface_reconstruction.products.core_masks
    fsnative_xfms = surface_reconstruction.products.fsnative_xfms
    lesion_reconstruction_summary = surface_reconstruction.products.lesion_reconstruction_summary
    subcortical_masks = surface_reconstruction.products.subcortical_masks
    subject_t1 = surface_reconstruction.products.subject_t1
    surface_export = plan_surface_export(
        env=env,
        layout=layout,
        opts=opts,
        subject_t1=subject_t1,
    )
    runner.add_steps(surface_export.steps)
    exported_surfaces = surface_export.products.exported_surfaces
    intact_surfaces = surface_export.products.intact_surfaces
    lesion_scaffold_anatomy = surface_export.products.lesion_scaffold_anatomy
    lesion_scaffold_metrics = surface_export.products.lesion_scaffold_metrics
    lesion_scaffold_surfaces = surface_export.products.lesion_scaffold_surfaces

    fsaverage_registration = plan_fsaverage_registration(
        env=env,
        layout=layout,
        lesion_scaffold_surfaces=lesion_scaffold_surfaces,
        opts=opts,
    )
    runner.add_steps(fsaverage_registration.steps)
    fsaverage_space = fsaverage_registration.products.fsaverage_space
    fsaverage_xfms = fsaverage_registration.products.fsaverage_xfms

    msmall_outputs: dict[str, object] | None = None
    if opts.msmall is not None:
        assert subj_t1 is not None and subj_t2 is not None
        msmall_outputs = add_msmall_plan(
            runner,
            calibration=opts.msmall,
            subject=inputs.sub_id,
            out_dir=opts.out_dir,
            work_dir=opts.work_dir,
            license_path=Path(env["FS_LICENSE"]),
            structural_t1w=subj_t1,
            structural_t2w=subj_t2,
            structural_brain_mask=layout.brain_mask,
            native_registration_spheres={
                hemi: Path(exported_surfaces[f"{hemi}.sphere.reg"]) for hemi in ("lh", "rh")
            },
            env=env,
            force=opts.overwrite,
        )

    lesion_surfaces = plan_lesion_surfaces(
        env=env,
        layout=layout,
        lesion_mask=lesion_mask,
        lesion_scaffold_anatomy=lesion_scaffold_anatomy,
        lesion_scaffold_metrics=lesion_scaffold_metrics,
        lesion_scaffold_surfaces=lesion_scaffold_surfaces,
        opts=opts,
    )
    runner.add_steps(lesion_surfaces.steps)
    lesion_surface_validity = lesion_surfaces.products.lesion_surface_validity
    lesion_vertex_mappings = lesion_surfaces.products.lesion_vertex_mappings

    template_transforms = plan_template_transforms(
        fsaverage_space=fsaverage_space,
        fsaverage_xfms=fsaverage_xfms,
        opts=opts,
    )
    runner.add_steps(template_transforms.steps)
    t1_fsaverage_xfms = template_transforms.products.t1_fsaverage_xfms

    mni_registration = plan_mni_registration(
        core_masks=core_masks,
        env=env,
        layout=layout,
        mni_brain_mask=mni_brain_mask,
        mni_brain_template=mni_brain_template,
        opts=opts,
        subject_t1=subject_t1,
    )
    runner.add_steps(mni_registration.steps)
    mni_qc_images = mni_registration.products.mni_qc_images
    mni_xfms = mni_registration.products.mni_xfms
    publication = plan_publication(
        copied_session_files=copied_session_files,
        core_masks=core_masks,
        execution_context=execution_context,
        exported_surfaces=exported_surfaces,
        fsaverage_xfms=fsaverage_xfms,
        fsnative_xfms=fsnative_xfms,
        gradient_resolutions=gradient_resolutions,
        inputs=inputs,
        intact_surfaces=intact_surfaces,
        layout=layout,
        lesion_mask=lesion_mask,
        lesion_metadata=lesion_metadata,
        lesion_probability=lesion_probability,
        lesion_qc=lesion_qc,
        lesion_reconstruction_summary=lesion_reconstruction_summary,
        lesion_surface_validity=lesion_surface_validity,
        lesion_vertex_mappings=lesion_vertex_mappings,
        mni_qc_images=mni_qc_images,
        mni_xfms=mni_xfms,
        msmall_outputs=msmall_outputs,
        myelin_map=myelin_map,
        opts=opts,
        pose_qc=pose_qc,
        pose_xfms=pose_xfms,
        public_inputs=public_inputs,
        subcortical_masks=subcortical_masks,
        subj_t1=subj_t1,
        subj_t2=subj_t2,
        subject_anat=subject_anat,
        synthetic_scaffold_t1=synthetic_scaffold_t1,
        synthetic_t1w=synthetic_t1w,
        t1_fsaverage_xfms=t1_fsaverage_xfms,
        t2w_to_t1w_xfms=t2w_to_t1w_xfms,
    )
    runner.add_steps(publication.steps)

    return runner


def run(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> None:
    """Construct the module graph and execute it through the shared runner.

    Freshness is evaluated after graph construction. Processing and validation
    errors propagate to the caller; partial private outputs can support resumption.
    """
    runner_started = time.perf_counter()
    runner = build_module(inputs, opts, execution_context=execution_context)
    runner.execute_graph(started_at=runner_started)


def _build_argparser() -> argparse.ArgumentParser:
    cfg = SETTINGS.anat
    p = argparse.ArgumentParser(prog="nro.modules.anat.module")
    p.add_argument("--project", default=SETTINGS.common.project)
    p.add_argument("--anat-id", default=SETTINGS.common.anat_id)
    p.add_argument("--sub-id", required=True)
    p.add_argument("--fs-subject", default=cfg.fs_subject)
    p.add_argument("--fsaverage-template", default=cfg.fsaverage_template)
    p.add_argument("--t1w", action="append", default=[], type=Path)
    p.add_argument("--t2w", action="append", default=[], type=Path)
    p.add_argument(
        "--selection-strategy", choices=["first", "robust_average"], default=cfg.selection_strategy
    )
    p.add_argument(
        "--t1w-fallback",
        choices=["synthesize_from_t2w", "skip"],
        default=cfg.t1w_fallback,
    )
    p.add_argument(
        "--surface-reconstruction-engine",
        choices=["freesurfer", "fastsurfer"],
        default=cfg.surface_reconstruction_engine,
    )
    p.add_argument(
        "--gradient-unwarping",
        choices=["auto", "off"],
        default=cfg.gradient_unwarping,
    )
    p.add_argument("--mni-template", type=Path, default=cfg.mni_template)
    p.add_argument("--synthstrip-container", type=Path, default=cfg.synthstrip_container)
    p.add_argument("--freesurfer-container", type=Path, default=cfg.freesurfer_container)
    p.add_argument("--fastsurfer-container", type=Path, default=cfg.fastsurfer_container)
    p.add_argument("--lesion", action="store_true")
    p.add_argument("--lesion-masker-command", type=Path, default=cfg.lesion.masker_command)
    p.add_argument("--freesurfer-subjects-dir", type=Path, default=cfg.freesurfer_subjects_dir)
    p.add_argument("--out-dir", type=Path, default=cfg.out_dir)
    p.add_argument("--work-dir", type=Path, default=cfg.work_dir)
    p.add_argument("--overwrite", action="store_true", default=cfg.overwrite)
    p.add_argument("--verbose", action="store_true", default=cfg.verbose)
    add_container_arguments(p, cfg.container)
    return p


def source_image_path(path: Path, *, project: str) -> Path:
    """Resolve a source image's directory while keeping its BIDS file name.

    git-annex stores each image as a symlink to an object whose name lacks the
    BIDS suffix, so resolving the file itself would lose the modality.
    """
    return resolve_project_path(path.parent, project=project) / path.name


def main(
    argv: Optional[Sequence[str]] = None, *, execution_context: ExecutionContext | None = None
) -> None:
    """Parse and execute one anatomical module work item."""
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project = str(args.project)
    t1w = [load_anat_image(source_image_path(path, project=project)) for path in args.t1w]
    t2w = [load_anat_image(source_image_path(path, project=project)) for path in args.t2w]
    markup = active_source_markup(project=project)
    msmall = (
        resolve_msmall_calibration(
            markup=markup,
            t1w=[image.image for image in t1w],
            t2w=[image.image for image in t2w],
            parameters=SETTINGS.anat.msmall.as_dict(),
            surface_engine=str(args.surface_reconstruction_engine),
            selection_strategy=str(args.selection_strategy),
        )
        if markup is not None and t1w
        else None
    )
    out_dir = resolve_project_path(
        args.out_dir
        or anat_subject_dir(str(args.sub_id), project=project, anat_id=str(args.anat_id)),
        project=project,
    )
    work_dir = resolve_project_work_path(
        args.work_dir
        or (anat_subject_work_dir(str(args.sub_id), project=project, anat_id=str(args.anat_id))),
        project=project,
    )
    subjects_dir = resolve_project_path(
        args.freesurfer_subjects_dir
        or (
            module_derivatives_root("anat", str(args.anat_id), project=project)
            / "code"
            / "freesurfer"
        ),
        project=project,
    )
    mni_template = resolve_project_path(args.mni_template, project=project)
    if mni_template is None:
        raise SystemExit("Missing --mni-template path.")
    require_nonempty_file(mni_template, "MNI template")
    synthstrip_image = resolve_project_path(args.synthstrip_container, project=project)
    if synthstrip_image is None:
        raise SystemExit("Missing --synthstrip-container path.")
    freesurfer_image = resolve_project_path(args.freesurfer_container, project=project)
    site, _ = site_settings()
    lesion_masker_command = resolve_project_path(args.lesion_masker_command, project=project)
    ordinary_fastsurfer_image = resolve_project_path(args.fastsurfer_container, project=project)
    fastsurfer_image = ordinary_fastsurfer_image
    container = build_container(
        ContainerSettings.from_args(args),
        work_directory=work_dir,
        execution_context=execution_context,
    )
    run(
        Inputs(sub_id=str(args.sub_id), t1w=tuple(t1w), t2w=tuple(t2w)),
        Options(
            project=project,
            anat_id=str(args.anat_id),
            out_dir=out_dir,
            work_dir=work_dir,
            freesurfer_subjects_dir=subjects_dir,
            fs_subject=str(args.fs_subject or args.sub_id),
            fsaverage_template=str(args.fsaverage_template),
            selection_strategy=str(args.selection_strategy),
            surface_reconstruction_engine=str(args.surface_reconstruction_engine),
            gradient_unwarping=str(args.gradient_unwarping),
            gradient_unwarp_image=Path(site["gradient_unwarp"]),
            container_runtime=str(site["runtime"]),
            mni_template=mni_template,
            container=container,
            synthstrip_image=synthstrip_image,
            freesurfer_image=freesurfer_image,
            overwrite=bool(args.overwrite),
            lesion=bool(args.lesion),
            lesion_masker_command=lesion_masker_command,
            synthstroke_data=Path(site["synthstroke_data"]),
            fastsurfer_image=fastsurfer_image,
            msmall=msmall,
            t1w_fallback=str(args.t1w_fallback),
        ),
        execution_context=execution_context,
    )


if __name__ == "__main__":
    main()
