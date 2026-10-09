"""Resolve and execute explicitly calibrated MSMAll anatomical registration."""

from __future__ import annotations

import hashlib
import math
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nro.definitions.markup import SubjectMarkup
from nro.engine.execution import create_copy_file_step
from nro.engine.functional_references import (
    load_rec,
    load_reference_inventory,
    pe_axis_and_sign,
    resolve_func_references,
)
from nro.engine.image_paths import image_source_paths
from nro.engine.io import atomic_write_text, read_public_json, write_public_json
from nro.engine.manifests import create_json_step
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

from .contract import msmall_structural_input_contract


@dataclass(frozen=True)
class MsmAllRun:
    """One fixed calibration run and its distortion-correction inputs."""

    name: str
    bold: Path
    repetition_time: float
    echo_spacing: float
    fieldmap_echo_spacing: float
    phase_encoding_direction: str
    unwarp_direction: str
    se_negative: Path
    se_positive: Path
    input_paths: tuple[Path, ...]

    def contract(self, subject_dir: Path) -> dict[str, object]:
        """Return a portable scientific record for the selected run."""

        def relative(path: Path) -> str:
            return path.absolute().relative_to(subject_dir.absolute()).as_posix()

        return {
            "name": self.name,
            "bold": relative(self.bold),
            "repetition_time": self.repetition_time,
            "effective_echo_spacing": self.echo_spacing,
            "fieldmap_effective_echo_spacing": self.fieldmap_echo_spacing,
            "phase_encoding_direction": self.phase_encoding_direction,
            "unwarp_direction": self.unwarp_direction,
            "fieldmaps": [relative(self.se_negative), relative(self.se_positive)],
        }


@dataclass(frozen=True)
class MsmAllCalibration:
    """Resolved, immutable MSMAll calibration inputs for one participant."""

    subject_dir: Path
    t1w: tuple[Path, ...]
    t2w: tuple[Path, ...]
    runs: tuple[MsmAllRun, ...]
    parameters: Mapping[str, object]
    selection_strategy: str

    @property
    def input_paths(self) -> tuple[Path, ...]:
        """Return every source file read by the calibration route."""
        values: list[Path] = []
        for path in (*self.t1w, *self.t2w):
            values.extend(image_source_paths(path))
        for run in self.runs:
            values.extend(run.input_paths)
        return tuple(dict.fromkeys(values))

    @property
    def run_input_paths(self) -> tuple[Path, ...]:
        """Return the functional and fieldmap files read by the HCP route."""
        return tuple(dict.fromkeys(path for run in self.runs for path in run.input_paths))

    def contract(self, subject_dir: Path) -> dict[str, object]:
        """Return the scientific contract attached only to the MSMAll branch."""
        return {
            "anatomical_selection_strategy": self.selection_strategy,
            "structural_inputs": msmall_structural_input_contract(),
            "calibration_runs": [run.contract(subject_dir) for run in self.runs],
            "parameters": dict(self.parameters),
            "atlas_registration": {
                "moving_input": "skull_stripped_T1w",
                "moving_mask": "binarized_surface_reconstruction_brain_mask",
                "reference_mask": "HCP_MNI152_2mm_brain_mask_dil",
                "affine_degrees_of_freedom": 7,
                "fnirt_intensity_model": "disabled_for_binary_masks",
                "validation": {
                    "affine_singular_value_range": [0.8, 1.2],
                    "maximum_affine_anisotropy_ratio": 1.01,
                    "minimum_mask_dice": 0.75,
                    "minimum_masked_intensity_correlation": 0.3,
                    "maximum_jacobian_nonpositive_fraction": 0.0,
                    "extent_ratio_range": [0.6, 1.4],
                },
            },
            "missing_variance_policy": {
                "variance_floor": 0.001,
                "undefined_spatial_correlation_weight": 0.0,
            },
        }

    def publication_record(self) -> dict[str, object]:
        """Return path-typed calibration provenance for BIDS-URI publication."""
        return {
            "T1w": list(self.t1w),
            "T2w": list(self.t2w),
            "AnatomicalSelectionStrategy": self.selection_strategy,
            "Runs": [
                {
                    "Name": run.name,
                    "BOLD": run.bold,
                    "RepetitionTime": run.repetition_time,
                    "EffectiveEchoSpacing": run.echo_spacing,
                    "FieldmapEffectiveEchoSpacing": run.fieldmap_echo_spacing,
                    "PhaseEncodingDirection": run.phase_encoding_direction,
                    "UnwarpDirection": run.unwarp_direction,
                    "Fieldmaps": [run.se_negative, run.se_positive],
                }
                for run in self.runs
            ],
            "Parameters": dict(self.parameters),
        }


def _unwarp_direction(ped: str) -> str:
    axis, sign = pe_axis_and_sign(ped)
    return {"i": "x", "j": "y", "k": "z"}[axis] + ("-" if sign < 0 else "")


def _session_inventory(bold: Path, *, markup: SubjectMarkup):
    session = bold.parent.parent
    prefix = markup.subject_dir.name
    if session.name.startswith("ses-"):
        prefix += f"_{session.name}"
    return load_reference_inventory(
        session / "func",
        session / "fmap",
        prefix,
        markup=markup,
    )


def resolve_msmall_calibration(
    *,
    markup: SubjectMarkup,
    t1w: Sequence[Path],
    t2w: Sequence[Path],
    parameters: Mapping[str, object],
    surface_engine: str,
    selection_strategy: str,
) -> MsmAllCalibration | None:
    """Resolve a fixed markup calibration set or return no optional branch."""
    if not bool(parameters["enabled"]) or not markup.msmall_rest:
        return None
    if markup.lesion:
        raise ValueError("MSMAll calibration is not supported for lesion-aware anatomy")
    if surface_engine != "freesurfer":
        raise ValueError("MSMAll calibration currently requires FreeSurfer reconstruction")
    if not t1w or not t2w:
        raise ValueError("MSMAll calibration requires both T1w and T2w source images")
    if selection_strategy not in {"first", "robust_average"}:
        raise ValueError(f"Unsupported MSMAll anatomical selection: {selection_strategy}")

    selected_t1w = tuple(Path(path).absolute() for path in t1w)
    selected_t2w = tuple(Path(path).absolute() for path in t2w)
    if selection_strategy == "first":
        selected_t1w = selected_t1w[:1]
        selected_t2w = selected_t2w[:1]

    selected_records = []
    for selected in dict.fromkeys(markup.msmall_rest):
        bold_path = selected.expanduser().absolute()
        if markup.is_excluded(bold_path):
            raise ValueError(f"MSMAll calibration run is excluded by markup: {bold_path}")
        if not bold_path.is_file():
            raise FileNotFoundError(f"MSMAll calibration run does not exist: {bold_path}")
        if not bold_path.name.endswith(("_bold.nii", "_bold.nii.gz")):
            raise ValueError(f"MSMAll calibration path is not a BOLD image: {bold_path}")
        selected_records.append(load_rec(bold_path, markup=markup))
    selected_records.sort(key=lambda record: (*record.key, str(record.img)))

    runs: list[MsmAllRun] = []
    for index, bold in enumerate(selected_records, start=1):
        bold_path = bold.img
        repetition_time = bold.metadata.get("RepetitionTime")
        echo_spacing = bold.metadata.get("EffectiveEchoSpacing")
        if bold.ped is None or repetition_time is None or echo_spacing is None:
            raise ValueError(
                "MSMAll calibration requires RepetitionTime, EffectiveEchoSpacing, and "
                f"PhaseEncodingDirection: {bold_path}"
            )
        inventory = _session_inventory(bold_path, markup=markup)
        references = resolve_func_references(
            bold=bold,
            sbrefs=inventory.sbrefs,
            sidecarless_sbrefs=inventory.sidecarless_sbrefs,
            fmaps=inventory.fmaps,
            sdc_from_sbref_pair=False,
            selection_warning=inventory.fieldmap_warning,
            unusable_sbrefs=inventory.unusable_sbrefs,
        )
        if references.pair is None:
            detail = f" ({references.warning})" if references.warning else ""
            raise ValueError(
                f"MSMAll calibration requires a compatible opposite-PE fieldmap pair: "
                f"{bold_path}{detail}"
            )
        pair = references.pair
        fieldmap_echo_spacing = pair.se1.metadata.get("EffectiveEchoSpacing")
        second_echo_spacing = pair.se2.metadata.get("EffectiveEchoSpacing")
        if fieldmap_echo_spacing is None or second_echo_spacing is None:
            raise ValueError(
                f"MSMAll fieldmaps require EffectiveEchoSpacing: {pair.se1.img}, {pair.se2.img}"
            )
        if not math.isclose(
            float(fieldmap_echo_spacing),
            float(second_echo_spacing),
            rel_tol=1e-09,
            abs_tol=0.0,
        ):
            raise ValueError(
                "MSMAll opposite-PE fieldmaps must share EffectiveEchoSpacing: "
                f"{pair.se1.img}, {pair.se2.img}"
            )
        _, first_sign = pe_axis_and_sign(pair.se1.ped or "")
        negative, positive = (pair.se1, pair.se2) if first_sign < 0 else (pair.se2, pair.se1)
        sources: list[Path] = []
        for image in (bold.img, pair.se1.img, pair.se2.img):
            sources.extend(image_source_paths(image, markup=markup))
        runs.append(
            MsmAllRun(
                name=f"rfMRI_REST{index:03d}",
                bold=bold.img,
                repetition_time=float(repetition_time),
                echo_spacing=float(echo_spacing),
                fieldmap_echo_spacing=float(fieldmap_echo_spacing),
                phase_encoding_direction=bold.ped,
                unwarp_direction=_unwarp_direction(bold.ped),
                se_negative=negative.img,
                se_positive=positive.img,
                input_paths=tuple(dict.fromkeys(sources)),
            )
        )
    repetition_times = {round(run.repetition_time, 9) for run in runs}
    if len(repetition_times) != 1:
        raise ValueError("MSMAll calibration runs must share one repetition time")
    return MsmAllCalibration(
        subject_dir=markup.subject_dir,
        t1w=selected_t1w,
        t2w=selected_t2w,
        runs=tuple(runs),
        parameters=dict(parameters),
        selection_strategy=selection_strategy,
    )


def _shell_array(name: str, values: Sequence[object]) -> str:
    def render(value: object) -> str:
        if isinstance(value, float):
            return format(value, ".15g")
        return str(value)

    return f"{name}=({' '.join(shlex.quote(render(value)) for value in values)})"


def _content_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _structural_configuration_text(
    calibration: MsmAllCalibration,
    *,
    subject: str,
    work_dir: Path,
    license_path: Path,
    structural_t1w: Path,
    structural_t2w: Path,
    structural_brain_mask: Path,
) -> str:
    structural_lines = [
        f"subject={shlex.quote(subject)}",
        f"work_root={shlex.quote(str(work_dir))}",
        f"fs_license={shlex.quote(str(license_path))}",
        _shell_array("t1w", [structural_t1w]),
        _shell_array("t2w", [structural_t2w]),
        f"brain_mask={shlex.quote(str(structural_brain_mask))}",
    ]
    return "\n".join(structural_lines) + "\n"


def _surface_configuration_text(calibration: MsmAllCalibration) -> str:
    p = calibration.parameters
    lines = []
    for key in (
        "high_resolution_mesh",
        "low_resolution_mesh",
        "grayordinates_resolution_mm",
        "input_registration",
    ):
        value = p[key]
        rendered = format(value, ".15g") if isinstance(value, float) else str(value)
        lines.append(f"{key}={shlex.quote(rendered)}")
    return "\n".join(lines) + "\n"


def _calibration_configuration_text(
    calibration: MsmAllCalibration,
) -> str:
    p = calibration.parameters
    lines = [
        _shell_array("run_names", [run.name for run in calibration.runs]),
        _shell_array("run_paths", [run.bold for run in calibration.runs]),
        _shell_array("run_echo_spacing", [run.echo_spacing for run in calibration.runs]),
        _shell_array(
            "fieldmap_echo_spacing",
            [run.fieldmap_echo_spacing for run in calibration.runs],
        ),
        _shell_array("run_unwarp_direction", [run.unwarp_direction for run in calibration.runs]),
        _shell_array("se_negative", [run.se_negative for run in calibration.runs]),
        _shell_array("se_positive", [run.se_positive for run in calibration.runs]),
    ]
    for key in (
        "functional_resolution_mm",
        "surface_smoothing_fwhm_mm",
        "output_registration",
        "iteration_modes",
        "method",
        "ica_dimension",
        "high_pass_seconds",
        "fix_threshold",
        "fix_training_model",
        "matlab_run_mode",
    ):
        value = p[key]
        rendered = format(value, ".15g") if isinstance(value, float) else str(value)
        lines.append(f"{key}={shlex.quote(rendered)}")
    return "\n".join(lines) + "\n"


def _write_text_if_changed(path: Path, text: str) -> None:
    """Publish generated configuration without changing an identical file's timestamp."""
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    atomic_write_text(path, text)


def _remove_path(path: Path) -> None:
    """Remove one private stage-owned path without following symbolic links."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def add_msmall_plan(
    runner: Runner,
    *,
    calibration: MsmAllCalibration,
    subject: str,
    out_dir: Path,
    work_dir: Path,
    license_path: Path,
    structural_t1w: Path,
    structural_t2w: Path,
    structural_brain_mask: Path,
    native_registration_spheres: Mapping[str, Path],
    env: Mapping[str, str],
    force: bool,
) -> dict[str, object]:
    """Append the staged HCP route and publish canonical sphere transforms."""
    branch = work_dir / "msmall"
    structural_configuration = branch / "structural_configuration.sh"
    surface_configuration = branch / "surface_configuration.sh"
    calibration_configuration = branch / "calibration_configuration.sh"
    input_identities = out_dir / f"{subject}_desc-msmallInputs_provenance.json"
    driver = Path(__file__).with_name("msmall_driver.sh")
    atlas_validator = Path(__file__).with_name("msmall_validate_atlas.py")
    subcortical_validator = Path(__file__).with_name("msmall_validate_subcortical.py")
    hcp_inputs = tuple(
        dict.fromkeys(
            (
                *calibration.input_paths,
                structural_t1w,
                structural_t2w,
                structural_brain_mask,
            )
        )
    )
    structural_parameters = {
        "selection_strategy": calibration.selection_strategy,
        "structural_inputs": msmall_structural_input_contract(),
    }
    surface_parameters = {
        "high_resolution_mesh": calibration.parameters["high_resolution_mesh"],
        "low_resolution_mesh": calibration.parameters["low_resolution_mesh"],
        "grayordinates_resolution_mm": calibration.parameters["grayordinates_resolution_mm"],
        "input_registration": calibration.parameters["input_registration"],
    }
    grayordinates_resolution_value = calibration.parameters["grayordinates_resolution_mm"]
    grayordinates_resolution = (
        format(grayordinates_resolution_value, ".15g")
        if isinstance(grayordinates_resolution_value, float)
        else str(grayordinates_resolution_value)
    )

    structural_inputs = tuple(
        dict.fromkeys(
            path
            for image in (*calibration.t1w, *calibration.t2w)
            for path in image_source_paths(image)
        )
    ) + (structural_t1w, structural_t2w, structural_brain_mask)

    def write_structural_configuration() -> None:
        text = _structural_configuration_text(
            calibration,
            subject=subject,
            work_dir=branch,
            license_path=license_path,
            structural_t1w=structural_t1w,
            structural_t2w=structural_t2w,
            structural_brain_mask=structural_brain_mask,
        )
        _write_text_if_changed(structural_configuration, text)

    def write_calibration_configuration() -> None:
        text = _calibration_configuration_text(calibration)
        _write_text_if_changed(calibration_configuration, text)

    def write_surface_configuration() -> None:
        _write_text_if_changed(
            surface_configuration,
            _surface_configuration_text(calibration),
        )

    def write_input_identities() -> None:
        content_digests = {path: _content_digest(path) for path in hcp_inputs}
        write_public_json(
            input_identities,
            {
                "Subject": subject,
                "Inputs": [{"Path": path, "SHA256": content_digests[path]} for path in hcp_inputs],
                "Complete": True,
            },
        )

    structural_configuration_step = runner.add_step(
        Step.python(
            name="Write MSMAll Structural Configuration",
            outputs=(structural_configuration,),
            inputs=structural_inputs,
            action=write_structural_configuration,
            force=force,
            parameters=structural_parameters,
        )
    )
    calibration_configuration_step = runner.add_step(
        Step.python(
            name="Write MSMAll Calibration Configuration",
            outputs=(calibration_configuration,),
            inputs=calibration.run_input_paths,
            action=write_calibration_configuration,
            force=force,
            parameters=calibration.contract(calibration.subject_dir),
        )
    )
    surface_configuration_step = runner.add_step(
        Step.python(
            name="Write MSMAll Surface Configuration",
            outputs=(surface_configuration,),
            inputs=(),
            action=write_surface_configuration,
            force=force,
            parameters=surface_parameters,
        )
    )
    runner.add_step(
        Step.python(
            name="Write MSMAll Input Identities",
            outputs=(input_identities,),
            inputs=hcp_inputs,
            action=write_input_identities,
            force=force,
            parameters={"digest": "sha256"},
        )
    )
    hcp_subject = f"{subject.removeprefix('sub-')}_msmall"
    structural_session = branch / "structural" / hcp_subject
    session = branch / "study" / hcp_subject
    native = session / "MNINonLinear" / "Native"
    rois = session / "MNINonLinear" / "ROIs"
    atlas = (
        session / "MNINonLinear" / f"fsaverage_LR{calibration.parameters['low_resolution_mesh']}k"
    )
    complete = branch / "complete"
    hcp_native = {
        hemi: native / f"{hcp_subject}.{hemi}.sphere.MSMAll.native.surf.gii" for hemi in ("L", "R")
    }
    hcp_baseline = {
        hemi: native / f"{hcp_subject}.{hemi}.sphere.reg.reg_LR.native.surf.gii"
        for hemi in ("L", "R")
    }
    hcp_fsaverage = {
        hemi: native / f"{hcp_subject}.{hemi}.sphere.reg.native.surf.gii" for hemi in ("L", "R")
    }
    atlas_spheres = {
        hemi: atlas
        / f"{hcp_subject}.{hemi}.sphere.{calibration.parameters['low_resolution_mesh']}k_fs_LR.surf.gii"
        for hemi in ("L", "R")
    }
    hcp_atlas_surfaces = {
        (hemi, surface): (
            atlas
            / f"{hcp_subject}.{hemi}.{surface}_MSMAll.{calibration.parameters['low_resolution_mesh']}k_fs_LR.surf.gii"
        )
        for hemi in ("L", "R")
        for surface in ("white", "midthickness", "pial", "inflated")
    }
    stages = branch / "stages"

    def marker(name: str) -> Path:
        return stages / f"{name}.complete"

    def finalize(name: str):
        def write_marker() -> None:
            atomic_write_text(marker(name), "complete\n")

        return write_marker

    def prepare(name: str, *paths: Path, globs: tuple[tuple[Path, str], ...] = ()):
        def clean() -> None:
            marker(name).unlink(missing_ok=True)
            for path in paths:
                _remove_path(path)
            for root, pattern in globs:
                if root.is_dir():
                    for path in root.glob(pattern):
                        _remove_path(path)

        return clean

    def add_stage(
        stage: str,
        title: str,
        *,
        inputs: Sequence[Path],
        outputs: Sequence[Path],
        surface_stage: bool = False,
        calibration_stage: bool = False,
        index: int | None = None,
        cleanup: Sequence[Path] = (),
        cleanup_globs: tuple[tuple[Path, str], ...] = (),
        parameters: object | None = None,
        implementation_files: Sequence[Path] = (),
    ) -> None:
        if calibration_stage and not surface_stage:
            raise ValueError("An MSMAll calibration stage must include surface configuration")
        command = ["bash", str(driver), stage, str(structural_configuration)]
        dependencies = [structural_configuration_step.id]
        if surface_stage:
            command.append(str(surface_configuration))
            dependencies.append(surface_configuration_step.id)
        if calibration_stage:
            command.append(str(calibration_configuration))
            dependencies.append(calibration_configuration_step.id)
        if index is not None:
            command.append(str(index))
        implementations = (driver, *implementation_files)
        runner.add_step(
            Step.command_step(
                command,
                name=title,
                inputs=tuple(inputs),
                outputs=(*outputs, marker(stage)),
                after=tuple(dependencies),
                env=env,
                force=force,
                prepare=prepare(stage, *cleanup, globs=cleanup_globs),
                finalize=finalize(stage),
                parameters={
                    "scientific": parameters,
                    "implementation_sha256": {
                        path.name: _content_digest(path) for path in implementations
                    },
                },
            )
        )

    manifest = branch / "input_manifest.tsv"
    software = branch / "software_versions.txt"
    add_stage(
        "inventory",
        "Inventory MSMAll Inputs and Software",
        inputs=hcp_inputs,
        outputs=(manifest, software),
        surface_stage=True,
        calibration_stage=True,
        cleanup=(manifest, software),
        parameters={
            "structural": structural_parameters,
            "surface": surface_parameters,
            "calibration": calibration.contract(calibration.subject_dir),
        },
    )
    t1_dir = structural_session / "T1w"
    add_stage(
        "prefreesurfer",
        "MSMAll PreFreeSurfer",
        inputs=(structural_t1w, structural_t2w),
        outputs=(t1_dir / "T1w_acpc_dc_restore.nii.gz", t1_dir / "T2w_acpc_dc_restore.nii.gz"),
        cleanup=(structural_session,),
        parameters=structural_parameters,
    )
    source_mask = t1_dir / "nro_input_brain_mask.nii.gz"
    acpc_mask = t1_dir / "T1w_acpc_brain_mask.nii.gz"
    t1w_brain = t1_dir / "T1w_acpc_dc_restore_brain.nii.gz"
    t2w_brain = t1_dir / "T2w_acpc_dc_restore_brain.nii.gz"
    add_stage(
        "prefreesurfer_masks",
        "Restore MSMAll Source Brain Masks",
        inputs=(
            marker("prefreesurfer"),
            structural_brain_mask,
            t1_dir / "xfms/acpc.mat",
            t1_dir / "T1w_acpc_dc_restore.nii.gz",
            t1_dir / "T2w_acpc_dc_restore.nii.gz",
        ),
        outputs=(source_mask, acpc_mask, t1w_brain, t2w_brain),
        cleanup=(
            source_mask,
            acpc_mask,
            t1w_brain,
            t2w_brain,
        ),
        parameters={
            "source_mask": "nonzero_structural_T1w",
            "transform": "HCP_rigid_ACPC",
            "interpolation": "nearest_neighbor",
        },
    )
    add_stage(
        "masked_atlas",
        "MSMAll Mask-Aware Atlas Registration",
        inputs=(
            marker("prefreesurfer_masks"),
            t1_dir / "T1w_acpc_dc_restore.nii.gz",
            t1w_brain,
        ),
        outputs=(
            structural_session / "MNINonLinear/registration_qc.json",
            structural_session / "MNINonLinear/xfms/acpc2MNILinear.mat",
            structural_session / "MNINonLinear/xfms/acpc_dc2standard.nii.gz",
        ),
        cleanup=(structural_session / ".MNINonLinear.masked.tmp",),
        parameters=calibration.contract(calibration.subject_dir)["atlas_registration"],
        implementation_files=(atlas_validator,),
    )
    freesurfer_dir = t1_dir / hcp_subject
    add_stage(
        "freesurfer",
        "MSMAll FreeSurfer Reconstruction",
        inputs=(
            marker("prefreesurfer_masks"),
            t1_dir / "T1w_acpc_dc_restore.nii.gz",
            t1_dir / "T2w_acpc_dc_restore.nii.gz",
            t1w_brain,
            t2w_brain,
        ),
        outputs=(freesurfer_dir / "surf/lh.white", freesurfer_dir / "surf/rh.white"),
        cleanup=(freesurfer_dir,),
        parameters={"seed": 1234, "processing_mode": "HCPStyleData"},
    )
    add_stage(
        "postfreesurfer",
        "MSMAll PostFreeSurfer",
        inputs=(
            marker("freesurfer"),
            marker("masked_atlas"),
            freesurfer_dir / "surf/lh.white",
            freesurfer_dir / "surf/rh.white",
        ),
        outputs=(
            *hcp_baseline.values(),
            *hcp_fsaverage.values(),
            rois / f"ROIs.{grayordinates_resolution}.nii.gz",
            rois / f"Atlas_ROIs.{grayordinates_resolution}.nii.gz",
        ),
        surface_stage=True,
        cleanup=(session,),
        parameters=surface_parameters,
    )
    add_stage(
        "validate_subcortical",
        "Validate MSMAll Subcortical Models",
        inputs=(
            marker("postfreesurfer"),
            rois / f"ROIs.{grayordinates_resolution}.nii.gz",
            rois / f"Atlas_ROIs.{grayordinates_resolution}.nii.gz",
        ),
        outputs=(rois / "subcortical_qc.json",),
        surface_stage=True,
        cleanup=(rois / "subcortical_qc.json",),
        parameters={"required_labels": "HCP_FreeSurferSubcorticalLabelTableLut"},
        implementation_files=(subcortical_validator,),
    )
    results = session / "MNINonLinear/Results"
    surface_markers: list[Path] = []
    for index, run in enumerate(calibration.runs):
        run_dir = results / run.name
        volume_stage = f"fmri_volume_{run.name}"
        add_stage(
            volume_stage,
            f"MSMAll fMRI Volume {run.name}",
            inputs=(marker("validate_subcortical"), *run.input_paths),
            outputs=(run_dir / f"{run.name}.nii.gz",),
            surface_stage=True,
            calibration_stage=True,
            index=index,
            cleanup=(run_dir, session / run.name),
            parameters=run.contract(calibration.subject_dir),
        )
        surface_stage = f"fmri_surface_{run.name}"
        add_stage(
            surface_stage,
            f"MSMAll fMRI Surface {run.name}",
            inputs=(
                marker(volume_stage),
                run_dir / f"{run.name}.nii.gz",
                marker("validate_subcortical"),
            ),
            outputs=(run_dir / f"{run.name}_Atlas.dtseries.nii",),
            surface_stage=True,
            calibration_stage=True,
            index=index,
            cleanup_globs=((run_dir, f"{run.name}_Atlas*"),),
            parameters={
                "low_resolution_mesh": calibration.parameters["low_resolution_mesh"],
                "surface_smoothing_fwhm_mm": calibration.parameters["surface_smoothing_fwhm_mm"],
                "input_registration": calibration.parameters["input_registration"],
            },
        )
        surface_markers.append(marker(surface_stage))
    concat_dir = results / "rfMRI_REST_CONCAT"
    high_pass = calibration.parameters["high_pass_seconds"]
    hp = format(high_pass, ".15g") if isinstance(high_pass, float) else str(high_pass)
    concat = concat_dir / f"rfMRI_REST_CONCAT_Atlas_hp{hp}_clean.dtseries.nii"
    variance = concat_dir / f"rfMRI_REST_CONCAT_Atlas_hp{hp}_clean_vn.dscalar.nii"
    original_variance = (
        concat_dir / f"rfMRI_REST_CONCAT_Atlas_hp{hp}_clean_vn_before_floor.dscalar.nii"
    )
    add_stage(
        "multirun_fix",
        "MSMAll Multi-Run ICA-FIX",
        inputs=tuple(surface_markers),
        outputs=(concat, original_variance),
        surface_stage=True,
        calibration_stage=True,
        cleanup=(concat_dir,),
        parameters={
            key: calibration.parameters[key]
            for key in (
                "high_pass_seconds",
                "fix_threshold",
                "fix_training_model",
                "matlab_run_mode",
            )
        },
    )
    add_stage(
        "prepare_msmall",
        "Prepare MSMAll Functional Inputs",
        inputs=(marker("multirun_fix"), original_variance),
        outputs=(variance,),
        surface_stage=True,
        calibration_stage=True,
        cleanup=(variance,),
        parameters=calibration.contract(calibration.subject_dir)["missing_variance_policy"],
    )
    initial_registration = (
        f"{calibration.parameters['output_registration']}_InitialReg_2_"
        f"d{calibration.parameters['ica_dimension']}_{calibration.parameters['method']}"
    )
    initial_spheres = tuple(
        native / f"{hcp_subject}.{hemi}.sphere.{initial_registration}.native.surf.gii"
        for hemi in ("L", "R")
    )
    add_stage(
        "msmall",
        "Estimate MSMAll Registration",
        inputs=(marker("prepare_msmall"), variance, *hcp_baseline.values()),
        outputs=initial_spheres,
        surface_stage=True,
        calibration_stage=True,
        cleanup=initial_spheres,
        cleanup_globs=((native, f"*{calibration.parameters['output_registration']}_InitialReg*"),),
        parameters={
            key: calibration.parameters[key]
            for key in (
                "output_registration",
                "iteration_modes",
                "method",
                "ica_dimension",
                "matlab_run_mode",
            )
        },
    )
    final_outputs = (*hcp_native.values(), *atlas_spheres.values(), *hcp_atlas_surfaces.values())
    add_stage(
        "dedrift",
        "MSMAll Dedrift and Resample",
        inputs=(marker("msmall"), *initial_spheres),
        outputs=final_outputs,
        surface_stage=True,
        calibration_stage=True,
        cleanup=final_outputs,
        parameters={
            key: calibration.parameters[key]
            for key in (
                "output_registration",
                "ica_dimension",
                "method",
                "surface_smoothing_fwhm_mm",
                "high_pass_seconds",
            )
        },
    )
    add_stage(
        "validate",
        "Validate MSMAll Calibration Outputs",
        inputs=(marker("dedrift"), *final_outputs),
        outputs=(complete,),
        surface_stage=True,
        calibration_stage=True,
        cleanup=(complete,),
        parameters={"output_signature": "MSMAll-fsLR-surfaces-v1"},
    )

    outputs: dict[str, Any] = {
        "input_identities": str(input_identities),
        "transforms": {},
        "atlas_spheres": {},
        "atlas_surfaces": {},
        "valid_masks": {},
    }
    for hemi, long_hemi in (("L", "lh"), ("R", "rh")):
        baseline = branch / "bridge" / f"{subject}_hemi-{hemi}_desc-fsLRBaseline_sphere.surf.gii"
        forward = (
            out_dir / f"{subject}_from-fsnative_to-MSMAll_hemi-{hemi}_mode-surface_xfm.surf.gii"
        )
        runner.add_step(
            Step.command_step(
                [
                    "wb_command",
                    "-surface-sphere-project-unproject",
                    str(native_registration_spheres[long_hemi]),
                    str(hcp_fsaverage[hemi]),
                    str(hcp_baseline[hemi]),
                    str(baseline),
                ],
                name=f"Bridge Hemisphere {hemi} fsnative to fsLR",
                inputs=(
                    native_registration_spheres[long_hemi],
                    hcp_fsaverage[hemi],
                    hcp_baseline[hemi],
                ),
                outputs=(baseline,),
                env=env,
                force=force,
            )
        )
        runner.add_step(
            Step.command_step(
                [
                    "wb_command",
                    "-surface-sphere-project-unproject",
                    str(baseline),
                    str(hcp_baseline[hemi]),
                    str(hcp_native[hemi]),
                    str(forward),
                ],
                name=f"Bridge Hemisphere {hemi} fsnative to MSMAll",
                inputs=(baseline, hcp_baseline[hemi], hcp_native[hemi]),
                outputs=(forward,),
                env=env,
                force=force,
            )
        )
        atlas_output = (
            out_dir
            / f"{subject}_space-MSMAll_den-{calibration.parameters['low_resolution_mesh']}k_hemi-{hemi}_sphere.surf.gii"
        )
        runner.add_step(
            create_copy_file_step(
                src=atlas_spheres[hemi],
                dst=atlas_output,
                force=force,
                step_name=f"Publish Hemisphere {hemi} MSMAll Atlas Sphere",
            )
        )
        outputs["atlas_surfaces"][hemi] = {}
        for surface in ("white", "midthickness", "pial", "inflated"):
            hcp_surface = hcp_atlas_surfaces[(hemi, surface)]
            surface_output = (
                out_dir
                / f"{subject}_space-MSMAll_den-{calibration.parameters['low_resolution_mesh']}k_hemi-{hemi}_{surface}.surf.gii"
            )
            runner.add_step(
                create_copy_file_step(
                    src=hcp_surface,
                    dst=surface_output,
                    force=force,
                    step_name=f"Publish Hemisphere {hemi} MSMAll {surface.title()} Surface",
                )
            )
            runner.add_step(
                create_json_step(
                    step_name=f"Write Hemisphere {hemi} MSMAll {surface.title()} Metadata",
                    path=surface_output.with_suffix(".json"),
                    payload={
                        "AnatomicalStructurePrimary": (
                            "CortexLeft" if hemi == "L" else "CortexRight"
                        ),
                        "Hemisphere": hemi,
                        "Space": "MSMAll",
                        "SurfaceMesh": "fsLR",
                        "Density": f"{calibration.parameters['low_resolution_mesh']}k",
                        "SurfaceType": surface,
                    },
                    inputs=(surface_output,),
                    force=force,
                )
            )
            outputs["atlas_surfaces"][hemi][surface] = str(surface_output)
        valid = (
            out_dir
            / f"{subject}_space-MSMAll_den-{calibration.parameters['low_resolution_mesh']}k_hemi-{hemi}_desc-valid_mask.shape.gii"
        )
        runner.add_step(
            Step.command_step(
                [
                    "bash",
                    "-lc",
                    'wb_command -surface-coordinates-to-metric "$0" "$1.tmp" && '
                    'wb_command -metric-math "x*0+1" "$1" -var x "$1.tmp" '
                    "-select x 1 && "
                    'rm -f "$1.tmp"',
                    str(atlas_output),
                    str(valid),
                ],
                name=f"Create Hemisphere {hemi} MSMAll Valid Mask",
                inputs=(atlas_output,),
                outputs=(valid,),
                env=env,
                force=force,
            )
        )
        runner.add_step(
            create_json_step(
                step_name=f"Write Hemisphere {hemi} MSMAll Transform Metadata",
                path=forward.with_suffix(".json"),
                payload={
                    "Type": "surface",
                    "Format": "WorkbenchSphereRegistration",
                    "Hemisphere": hemi,
                    "From": "fsnative",
                    "To": "MSMAll",
                    "Carrier": "fsLR",
                    "CalibrationRuns": [run.name for run in calibration.runs],
                },
                inputs=(forward,),
                force=force,
            )
        )
        outputs["transforms"][hemi] = str(forward)
        outputs["atlas_spheres"][hemi] = str(atlas_output)
        outputs["valid_masks"][hemi] = str(valid)

    qc = out_dir / f"{subject}_space-MSMAll_desc-registration_qc.json"
    qc_inputs = (
        tuple(
            Path(path)
            for group in (
                outputs["transforms"],
                outputs["atlas_spheres"],
                outputs["valid_masks"],
            )
            for path in group.values()
        )
        + tuple(
            Path(path)
            for hemisphere in outputs["atlas_surfaces"].values()
            for path in hemisphere.values()
        )
        + tuple(native_registration_spheres.values())
    )

    def write_registration_qc() -> None:
        import nibabel as nib
        import numpy as np

        hemispheres: dict[str, object] = {}
        for hemi, long_hemi in (("L", "lh"), ("R", "rh")):
            transform = nib.load(Path(outputs["transforms"][hemi]))
            atlas_sphere = nib.load(Path(outputs["atlas_spheres"][hemi]))
            native_sphere = nib.load(native_registration_spheres[long_hemi])
            transform_coordinates = np.asarray(transform.darrays[0].data)
            atlas_coordinates = np.asarray(atlas_sphere.darrays[0].data)
            native_coordinates = np.asarray(native_sphere.darrays[0].data)
            surface_images = {
                name: nib.load(Path(path)) for name, path in outputs["atlas_surfaces"][hemi].items()
            }
            surface_counts = {
                name: int(len(image.darrays[0].data)) for name, image in surface_images.items()
            }
            atlas_count = int(len(atlas_coordinates))
            if not np.isfinite(transform_coordinates).all():
                raise ValueError(f"Hemisphere {hemi} MSMAll transform is nonfinite")
            if not np.isfinite(atlas_coordinates).all():
                raise ValueError(f"Hemisphere {hemi} MSMAll atlas sphere is nonfinite")
            if len(transform_coordinates) != len(native_coordinates):
                raise ValueError(
                    f"Hemisphere {hemi} MSMAll transform does not preserve native vertices"
                )
            if not np.array_equal(transform.darrays[1].data, native_sphere.darrays[1].data):
                raise ValueError(
                    f"Hemisphere {hemi} MSMAll transform does not preserve native topology"
                )
            if any(count != atlas_count for count in surface_counts.values()):
                raise ValueError(f"Hemisphere {hemi} MSMAll surfaces do not share atlas density")
            if any(
                not np.array_equal(image.darrays[1].data, atlas_sphere.darrays[1].data)
                for image in surface_images.values()
            ):
                raise ValueError(f"Hemisphere {hemi} MSMAll surfaces do not share atlas topology")
            hemispheres[hemi] = {
                "TransformVertices": int(len(transform_coordinates)),
                "NativeVertices": int(len(native_coordinates)),
                "AtlasVertices": atlas_count,
                "SurfaceVertices": surface_counts,
                "Finite": True,
            }
        write_public_json(
            qc,
            {
                "Space": "MSMAll",
                "Carrier": "fsLR",
                "CalibrationRuns": [run.name for run in calibration.runs],
                "Parameters": dict(calibration.parameters),
                "Transforms": outputs["transforms"],
                "AtlasSpheres": outputs["atlas_spheres"],
                "AtlasSurfaces": outputs["atlas_surfaces"],
                "ValidMasks": outputs["valid_masks"],
                "Hemispheres": hemispheres,
                "Complete": True,
            },
        )

    def validate_registration_qc() -> tuple[bool, str]:
        try:
            document = read_public_json(qc)
        except (OSError, ValueError):
            return False, "MSMAll registration QC is missing or invalid."
        valid = document.get("Complete") is True
        return valid, (
            "MSMAll registration QC is complete."
            if valid
            else "MSMAll registration QC is incomplete."
        )

    runner.add_step(
        Step.python(
            name="Validate MSMAll Registration",
            inputs=qc_inputs,
            outputs=(qc,),
            action=write_registration_qc,
            validate=validate_registration_qc,
            force=force,
            parameters={"calibration_runs": [run.name for run in calibration.runs]},
        )
    )
    outputs["qc"] = str(qc)
    software_versions = out_dir / f"{subject}_space-MSMAll_desc-software_versions.txt"
    runner.add_step(
        create_copy_file_step(
            src=branch / "software_versions.txt",
            dst=software_versions,
            force=force,
            step_name="Publish MSMAll Software Versions",
        )
    )
    outputs["software_versions"] = str(software_versions)
    manifest = out_dir / f"{subject}_desc-msmall_manifest.json"
    manifest_outputs = dict(outputs)
    runner.add_step(
        create_json_step(
            step_name="Write MSMAll Publication Manifest",
            path=manifest,
            payload={
                "Subject": subject,
                "Space": "MSMAll",
                "SurfaceMesh": "fsLR",
                "Density": f"{calibration.parameters['low_resolution_mesh']}k",
                "Calibration": calibration.publication_record(),
                "Outputs": manifest_outputs,
                "SoftwareVersions": str(software_versions),
                "Complete": True,
            },
            inputs=(*tuple(_published_paths(outputs)), software_versions),
            force=force,
        )
    )
    outputs["manifest"] = str(manifest)
    return outputs


def _published_paths(value: object):
    """Yield paths recursively from one MSMAll publication mapping."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _published_paths(item)
    elif isinstance(value, str):
        yield Path(value)
