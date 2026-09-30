"""Resolve and execute explicitly calibrated MSMAll anatomical registration."""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from nro.configuration.markup import SubjectMarkup
from nro.configuration.store import fingerprint
from nro.engine.execution import create_copy_file_step
from nro.engine.image_paths import image_source_paths
from nro.engine.io import atomic_write_text, read_public_json, write_public_json
from nro.engine.manifests import create_json_step
from nro.modules.func.resolver import (
    load_rec,
    load_reference_inventory,
    pe_axis_and_sign,
    resolve_func_references,
)
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step


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
            "calibration_runs": [run.contract(subject_dir) for run in self.runs],
            "parameters": dict(self.parameters),
            "atlas_registration": {
                "moving_input": "skull_stripped_T1w",
                "moving_mask": "explicit_binary_brain_mask",
                "reference_mask": "HCP_MNI152_2mm_brain_mask_dil",
                "fnirt_intensity_model": "disabled_for_binary_masks",
                "validation": {
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
        if not np.isclose(float(fieldmap_echo_spacing), float(second_echo_spacing)):
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


def _configuration_text(
    calibration: MsmAllCalibration,
    *,
    subject: str,
    work_dir: Path,
    license_path: Path,
    content_digests: Mapping[Path, str],
) -> str:
    p = calibration.parameters
    structural_identity = {
        "contract": {
            "selection_strategy": calibration.selection_strategy,
            "T1w": [str(path) for path in calibration.t1w],
            "T2w": [str(path) for path in calibration.t2w],
        },
        "content": [
            content_digests[source_path]
            for image_path in (*calibration.t1w, *calibration.t2w)
            for source_path in image_source_paths(image_path)
        ],
    }
    calibration_identity = {
        "contract": calibration.contract(calibration.subject_dir),
        "content": [content_digests[path] for path in calibration.run_input_paths],
    }
    lines = [
        f"subject={shlex.quote(subject)}",
        f"work_root={shlex.quote(str(work_dir))}",
        f"fs_license={shlex.quote(str(license_path))}",
        f"structural_fingerprint={fingerprint(structural_identity)}",
        f"calibration_fingerprint={fingerprint(calibration_identity)}",
        _shell_array("t1w", calibration.t1w),
        _shell_array("t2w", calibration.t2w),
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
        "high_resolution_mesh",
        "low_resolution_mesh",
        "grayordinates_resolution_mm",
        "functional_resolution_mm",
        "surface_smoothing_fwhm_mm",
        "input_registration",
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


def add_msmall_plan(
    runner: Runner,
    *,
    calibration: MsmAllCalibration,
    subject: str,
    out_dir: Path,
    work_dir: Path,
    license_path: Path,
    native_registration_spheres: Mapping[str, Path],
    env: Mapping[str, str],
    force: bool,
) -> dict[str, object]:
    """Append the checkpointed HCP route and publish canonical sphere transforms."""
    branch = work_dir / "msmall"
    configuration = branch / "configuration.sh"
    input_identities = out_dir / f"{subject}_desc-msmallInputs_provenance.json"
    driver = Path(__file__).with_name("msmall_driver.sh")
    atlas_validator = Path(__file__).with_name("msmall_validate_atlas.py")
    hcp_inputs = calibration.input_paths

    def write_configuration() -> None:
        content_digests = {path: _content_digest(path) for path in hcp_inputs}
        text = _configuration_text(
            calibration,
            subject=subject,
            work_dir=branch,
            license_path=license_path,
            content_digests=content_digests,
        )
        atomic_write_text(configuration, text)
        write_public_json(
            input_identities,
            {
                "Subject": subject,
                "Inputs": [{"Path": path, "SHA256": content_digests[path]} for path in hcp_inputs],
                "Complete": True,
            },
        )

    runner.add_step(
        Step.python(
            name="Write MSMAll Calibration Configuration",
            outputs=(configuration, input_identities),
            inputs=hcp_inputs,
            action=write_configuration,
            force=force,
            parameters=calibration.contract(calibration.subject_dir),
        )
    )
    hcp_subject = f"{subject.removeprefix('sub-')}_msmall"
    session = branch / "study" / hcp_subject
    native = session / "MNINonLinear" / "Native"
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
    runner.add_step(
        Step.command_step(
            ["bash", str(driver), str(configuration)],
            name="Estimate MSMAll Registration",
            inputs=(*hcp_inputs, configuration, driver, atlas_validator),
            outputs=(
                complete,
                *hcp_native.values(),
                *hcp_baseline.values(),
                *hcp_fsaverage.values(),
                *atlas_spheres.values(),
                *hcp_atlas_surfaces.values(),
                branch / "software_versions.txt",
            ),
            env=env,
            force=force,
            parameters=calibration.contract(calibration.subject_dir),
        )
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
