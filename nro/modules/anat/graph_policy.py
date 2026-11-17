"""Resolve and validate the inputs that shape an anatomical runner graph.

Graph construction starts by routing outputs to the attempt's owner, checking
that acquisitions and containers exist, and resolving gradient unwarping. The
result is an `AnatGraphPolicy` that the graph builder reads instead of a large
set of local variables.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from nro.definitions.hardware import GradientUnwarpingResolution, resolve_gradient_unwarping
from nro.engine.io import require_nonempty_file
from nro.modules.anat.inputs import AnatImage
from nro.orchestration.execution_context import ExecutionContext

if TYPE_CHECKING:
    from nro.modules.anat.module import Inputs, Options


@dataclass(frozen=True)
class AnatGraphPolicy:
    """Resolved acquisitions, resources, and choices for one anatomical graph."""

    opts: Options
    all_images: tuple[AnatImage, ...]
    synthetic_t1w: bool
    lesion_masker_command: Path | None
    lesion_masker_module: str | None
    synthstroke_data: Path | None
    mni_brain_template: Path
    mni_brain_mask: Path
    gradient_resolutions: Mapping[Path, GradientUnwarpingResolution]
    public_inputs: tuple[Path, ...]

    def bind_candidates(self, env: Mapping[str, str]) -> list[Path | None]:
        """Return the paths a container may need, including None for unused resources."""
        return [
            *(item.image for item in self.all_images),
            *(item.image.resolve() for item in self.all_images),
            *(item.json for item in self.all_images if item.json is not None),
            self.opts.out_dir,
            self.opts.work_dir,
            self.opts.freesurfer_subjects_dir,
            self.opts.mni_template,
            self.mni_brain_template,
            self.mni_brain_mask,
            self.opts.gradient_unwarp_image
            if any(resolution.applied for resolution in self.gradient_resolutions.values())
            else None,
            *(resolution.coefficients for resolution in self.gradient_resolutions.values()),
            Path(env["FS_LICENSE"]) if Path(env["FS_LICENSE"]).is_file() else None,
            self.lesion_masker_command,
            self.synthstroke_data,
            self.opts.freesurfer_image
            if self.opts.surface_reconstruction_engine == "freesurfer"
            or self.synthetic_t1w
            or self.opts.lesion
            else None,
            self.opts.fastsurfer_image
            if self.opts.surface_reconstruction_engine == "fastsurfer"
            else None,
            *(self.opts.msmall.input_paths if self.opts.msmall is not None else ()),
            Path(__file__).with_name("msmall_driver.sh") if self.opts.msmall is not None else None,
            Path(__file__).with_name("msmall_validate_atlas.py")
            if self.opts.msmall is not None
            else None,
            Path(__file__).with_name("msmall_validate_subcortical.py")
            if self.opts.msmall is not None
            else None,
        ]


def resolve_graph_policy(
    inputs: Inputs,
    opts: Options,
    *,
    execution_context: ExecutionContext | None = None,
) -> AnatGraphPolicy:
    """Route outputs to the attempt's owner and validate the graph's inputs and resources.

    With an execution context, output, work, and FreeSurfer directories are
    moved to the owner's locations, and every acquisition must be a shared
    source input. Raises SystemExit for missing inputs, unsupported choices, or
    missing resource files, and ValueError for inputs outside the shared
    source tree.
    """
    if execution_context is not None:
        if execution_context.project != opts.project:
            raise ValueError("Anatomical project differs from its execution context")
        opts = replace(
            opts,
            out_dir=execution_context.output_path(opts.out_dir),
            work_dir=execution_context.output_path(opts.work_dir, private=True),
            freesurfer_subjects_dir=execution_context.output_path(opts.freesurfer_subjects_dir),
        )
        for image in (*inputs.t1w, *inputs.t2w):
            for path in (image.image, image.json):
                if path is not None and (
                    not path.absolute().is_relative_to(
                        execution_context.paths.source_project(opts.project)
                    )
                    or execution_context.input_path(path) != path
                ):
                    raise ValueError("Anatomical acquisitions must be shared source inputs")
        if opts.msmall is not None:
            for path in opts.msmall.input_paths:
                if (
                    not path.absolute().is_relative_to(
                        execution_context.paths.source_project(opts.project)
                    )
                    or execution_context.input_path(path) != path
                ):
                    raise ValueError("MSMAll calibration files must be shared source inputs")
    all_images = [*inputs.t1w, *inputs.t2w]
    if not all_images:
        raise SystemExit(f"No anatomical images provided for {inputs.sub_id}.")
    if opts.synthstrip_image is None:
        raise SystemExit("Missing SynthStrip image path.")
    require_nonempty_file(opts.synthstrip_image, "SynthStrip image")
    if opts.surface_reconstruction_engine not in {"freesurfer", "fastsurfer"}:
        raise SystemExit(
            f"Unsupported surface-reconstruction engine: {opts.surface_reconstruction_engine}"
        )
    if opts.t1w_fallback not in {"synthesize_from_t2w", "skip"}:
        raise SystemExit(f"Unsupported T1w fallback policy: {opts.t1w_fallback}")
    synthetic_t1w = not inputs.t1w and bool(inputs.t2w)
    if synthetic_t1w and opts.t1w_fallback == "skip":
        raise SystemExit("No non-excluded T1w image is available and T1w fallback is disabled.")
    if synthetic_t1w and opts.lesion:
        raise SystemExit("Lesion-aware anatomy requires an acquired T1w image.")
    if opts.surface_reconstruction_engine == "freesurfer" or synthetic_t1w or opts.lesion:
        if opts.freesurfer_image is None:
            raise SystemExit("Missing FreeSurfer container path.")
        require_nonempty_file(opts.freesurfer_image, "FreeSurfer container")
    if opts.lesion:
        if not inputs.t1w:
            raise SystemExit("Lesion-aware anatomy requires at least one T1w image.")
        if opts.lesion_masker_command is None:
            lesion_masker_command = Path(sys.executable)
            lesion_masker_module = "nro.modules.anat.synthstroke"
            if opts.synthstroke_data is None:
                raise SystemExit("Lesion-aware anatomy requires installed SynthStroke data.")
            for name in ("config.json", "model.safetensors"):
                require_nonempty_file(
                    opts.synthstroke_data / name,
                    f"SynthStroke resource {name}",
                )
            synthstroke_data = opts.synthstroke_data
        else:
            lesion_masker_command = opts.lesion_masker_command
            lesion_masker_module = None
            require_nonempty_file(lesion_masker_command, "lesion masker command")
            synthstroke_data = None
    else:
        lesion_masker_command = None
        lesion_masker_module = None
        synthstroke_data = None
        if opts.surface_reconstruction_engine == "fastsurfer":
            if opts.fastsurfer_image is None:
                raise SystemExit("FastSurfer reconstruction requires a configured image.")
            require_nonempty_file(opts.fastsurfer_image, "FastSurfer image")
    if opts.msmall is not None:
        if opts.lesion:
            raise SystemExit("MSMAll calibration is not supported for lesion-aware anatomy.")
        if opts.surface_reconstruction_engine != "freesurfer":
            raise SystemExit("MSMAll calibration currently requires FreeSurfer reconstruction.")
        if synthetic_t1w:
            raise SystemExit("MSMAll calibration requires an acquired T1w image.")
    require_nonempty_file(opts.mni_template, "MNI template")
    mni_brain_template = Path(
        str(opts.mni_template).replace("_T1w.nii.gz", "_desc-brain_T1w.nii.gz")
    )
    mni_brain_mask = Path(str(opts.mni_template).replace("_T1w.nii.gz", "_desc-brain_mask.nii.gz"))
    require_nonempty_file(mni_brain_template, "MNI brain template")
    require_nonempty_file(mni_brain_mask, "MNI brain mask")

    gradient_resolutions = {
        image.image: resolve_gradient_unwarping(
            image.metadata,
            mode=opts.gradient_unwarping,
        )
        for image in all_images
    }
    if any(resolution.applied for resolution in gradient_resolutions.values()):
        require_nonempty_file(opts.gradient_unwarp_image, "gradient-unwarping image")

    public_inputs = [item.image for item in all_images]
    if opts.msmall is not None:
        public_inputs.extend(opts.msmall.input_paths)
    public_inputs = list(dict.fromkeys(public_inputs))
    return AnatGraphPolicy(
        opts=opts,
        all_images=tuple(all_images),
        synthetic_t1w=synthetic_t1w,
        lesion_masker_command=lesion_masker_command,
        lesion_masker_module=lesion_masker_module,
        synthstroke_data=synthstroke_data,
        mni_brain_template=mni_brain_template,
        mni_brain_mask=mni_brain_mask,
        gradient_resolutions=gradient_resolutions,
        public_inputs=tuple(public_inputs),
    )
