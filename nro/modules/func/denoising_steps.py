"""Denoising steps for functional preprocessing."""

import re
from pathlib import Path
from typing import Optional

from nro.engine.execution import (
    ensure_directory,
    resolve_runner_command,
    runner_path_exists,
)
from nro.engine.execution import (
    strip_ansi as _strip_ansi,
)
from nro.modules.func.cicada import (
    prepare_melodic_adapter,
    reset_directory,
    write_motion_metrics,
    write_result_manifest,
)
from nro.modules.func.confounds import get_confounds
from nro.modules.func.ica_aroma import denoising as run_ica_aroma_denoising
from nro.modules.func.ica_aroma import (
    make_dilated_anatomical_epi_mask,
    run_ica_aroma_workflow,
    run_melodic_decomposition,
)
from nro.orchestration.runner import (
    Runner,
    shlex_quote,
    write_completion_breadcrumb,
)
from nro.orchestration.runner_graph import Step

from .constants import (
    _ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD,
    _ICA_AROMA_ESTIMATION_POLICY_VERSION,
    _ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM,
    _ICA_AROMA_MELODIC_MASK_DILATION_MM,
    _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
)
from .step_support import LOG


def _create_confounds_step(
    *,
    epi_4d: Path,
    epi_mean_3d: Path,
    mc_dir: Path,
    subjects_dir: Path,
    fs_subject: str,
    brain_mask_in_epi: Path,
    out_tsv: Path,
    out_json: Path,
    force: bool,
) -> Step:
    par = mc_dir / "motion.par"

    def calculate() -> None:
        ensure_directory(out_tsv.parent)
        get_confounds(
            epi=epi_4d,
            epi_mean=epi_mean_3d,
            mcflirt_par=par,
            subjects_dir=subjects_dir,
            fs_subject=fs_subject,
            brain_mask_in_epi=brain_mask_in_epi,
            out_tsv=out_tsv,
            out_json=out_json,
        )

    return Step.python(
        name="Confounds",
        outputs=(out_tsv, out_json),
        inputs=(epi_4d, epi_mean_3d, par, brain_mask_in_epi),
        force=force,
        action=calculate,
    )


def _resolve_ica_aroma_cmd(
    runner: Runner,
    env: dict[str, str],
    *,
    configured_cmd: Optional[Path],
) -> Optional[str]:
    if configured_cmd is not None:
        if runner_path_exists(runner, env, configured_cmd):
            return str(configured_cmd)
        raise SystemExit(f"Configured ICA-AROMA command does not exist: {configured_cmd}")
    return resolve_runner_command(runner, env, ["ICA_AROMA.py", "ica_aroma.py"])


def _resolve_container_command_for_wrapper(
    *,
    runner: Runner,
    env: dict[str, str],
    command: str,
) -> Optional[str]:
    resolved = resolve_runner_command(runner, env, [command])
    if resolved is None:
        return None
    resolved_path = Path(resolved)
    versioned_out = (
        runner.run_child(
            [
                "bash",
                "-lc",
                (
                    f"find /opt/fsl -maxdepth 3 -type f -path '*/bin/{command}' 2>/dev/null | "
                    "sort -r || true"
                ),
            ],
            env=env,
            capture_stdout=True,
        )
        or ""
    )
    versioned_candidates: list[Path] = []
    for raw_line in _strip_ansi(versioned_out).splitlines():
        line = raw_line.strip()
        if not line.startswith("/"):
            continue
        candidate = Path(line)
        versioned_candidates.append(candidate)
    for candidate in versioned_candidates:
        candidate_str = str(candidate)
        if "/fsl-6." in candidate_str or re.search(r"/fsl-[0-9][^/]*/bin/", candidate_str):
            if runner_path_exists(runner, env, candidate):
                return candidate_str

    probe = (
        runner.run_child(
            [
                "bash",
                "-lc",
                (
                    f"if [ -f {shlex_quote(str(resolved_path))} ]; then "
                    f"sed -n '1,5p' {shlex_quote(str(resolved_path))}; "
                    "fi"
                ),
            ],
            env=env,
            capture_stdout=True,
        )
        or ""
    )
    cleaned = _strip_ansi(probe)
    pattern = re.compile(rf"(/[^\"' \t]+/{re.escape(command)})")
    for line in cleaned.splitlines():
        for match in pattern.finditer(line):
            candidate = Path(match.group(1))
            if candidate != resolved_path and runner_path_exists(runner, env, candidate):
                return str(candidate)
    for candidate in versioned_candidates:
        if candidate != resolved_path and runner_path_exists(runner, env, candidate):
            return str(candidate)
    return str(resolved_path)


def _resolve_ica_aroma_fsl_commands(runner: Runner, env: dict[str, str]) -> dict[str, str]:
    commands = {
        "applywarp": None,
        "bet": None,
        "flirt": None,
        "fsl_regfilt": None,
        "fslinfo": None,
        "fslmaths": None,
        "fslmerge": None,
        "fslroi": None,
        "fslstats": None,
        "melodic": None,
    }
    resolved: dict[str, str] = {}
    for name in commands:
        path = _resolve_container_command_for_wrapper(runner=runner, env=env, command=name)
        if path is None:
            raise SystemExit(f"Missing required FSL command for ICA-AROMA: {name}")
        resolved[name] = path
    return resolved


def _ica_aroma_policy_payload(
    *,
    input_is_mni: bool,
    denoise_type: str,
    repetition_time: Optional[float],
    external_aroma: bool,
) -> dict[str, object]:
    return {
        "version": _ICA_AROMA_ESTIMATION_POLICY_VERSION,
        "input_space": "MNI152NLin2009cAsym" if input_is_mni else "T1w",
        "workflow": "external" if external_aroma else "internal",
        "denoise_type": str(denoise_type).strip().lower(),
        "repetition_time": (float(repetition_time) if repetition_time is not None else None),
        "melodic_estimation_smoothing_fwhm_mm": _ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM,
        "melodic_mask_dilation_mm": _ICA_AROMA_MELODIC_MASK_DILATION_MM,
        "regression_mask_dilation_mm": _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
        "epi_support_bet_fractional_intensity_threshold": _ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD,
        "shared_across_output_spaces": True,
    }


def _ica_aroma_shared_regression_policy_payload(
    *,
    input_space: str,
    denoise_type: str,
    repetition_time: Optional[float],
    shared_work_dir: Path,
) -> dict[str, object]:
    return {
        "version": _ICA_AROMA_ESTIMATION_POLICY_VERSION,
        "input_space": str(input_space),
        "workflow": "shared-t1w-component-regression",
        "denoise_type": str(denoise_type).strip().lower(),
        "repetition_time": (float(repetition_time) if repetition_time is not None else None),
        "estimation_space": "T1w",
        "shared_estimation_work_dir": str(shared_work_dir),
        "regression_mask_dilation_mm": _ICA_AROMA_REGRESSION_MASK_DILATION_MM,
        "epi_support_bet_fractional_intensity_threshold": _ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD,
        "shared_across_output_spaces": True,
    }


def _create_epi_support_step(
    *,
    epi_mean: Path,
    support_brain: Path,
    support_mask: Path,
    work_dir: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    return Step.command_step(
        [
            "bet",
            str(epi_mean),
            str(support_brain),
            "-R",
            "-f",
            f"{_ICA_AROMA_BET_FRACTIONAL_INTENSITY_THRESHOLD:.8g}",
            "-g",
            "0",
            "-m",
        ],
        outputs=(support_brain, support_mask),
        inputs=(epi_mean,),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(work_dir),
    )


def _create_dilated_anatomical_mask_step(
    *,
    anatomical_mask: Path,
    support_mask: Path,
    output: Path,
    dilation_mm: float,
    role: str,
    force: bool,
) -> Step:
    def construct() -> None:
        make_dilated_anatomical_epi_mask(
            anatomical_mask=anatomical_mask,
            epi_support_mask=support_mask,
            dilation_mm=dilation_mm,
            out_mask=output,
        )

    return Step.python(
        name=f"Construct {role} Mask",
        outputs=(output,),
        inputs=(anatomical_mask, support_mask),
        force=force,
        action=construct,
    )


def _create_melodic_smoothing_step(
    *,
    epi: Path,
    output: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    sigma = _ICA_AROMA_ESTIMATION_SMOOTHING_FWHM_MM / 2.3548200450309493
    return Step.command_step(
        ["fslmaths", str(epi), "-s", f"{sigma:.8g}", str(output)],
        name="Prepare Smoothed MELODIC Input",
        outputs=(output,),
        inputs=(epi,),
        force=force,
        env=env,
    )


def _create_cicada_motion_metrics_step(
    *,
    epi: Path,
    mask: Path,
    motion_parameters: Path,
    output: Path,
    fd_radius_mm: float,
    dvars_statistical_alpha: float,
    dvars_practical_threshold_percent: float,
    dvars_power: float,
    force: bool,
) -> Step:
    def calculate() -> None:
        write_motion_metrics(
            bold=epi,
            mask=mask,
            motion_parameters=motion_parameters,
            output=output,
            fd_radius_mm=fd_radius_mm,
            dvars_statistical_alpha=dvars_statistical_alpha,
            dvars_practical_threshold_percent=dvars_practical_threshold_percent,
            dvars_power=dvars_power,
        )

    return Step.python(
        name="Compute CICADA Motion Metrics",
        inputs=(epi, mask, motion_parameters),
        outputs=(output,),
        force=force,
        action=calculate,
        parameters={
            "fd_radius_mm": fd_radius_mm,
            "dvars_statistical_alpha": dvars_statistical_alpha,
            "dvars_practical_threshold_percent": dvars_practical_threshold_percent,
            "dvars_power": dvars_power,
        },
    )


def _create_cicada_melodic_step(
    *,
    runner: Runner,
    melodic_input: Path,
    melodic_mask: Path,
    output_directory: Path,
    outputs: tuple[Path, ...],
    identity_transform: Path,
    t1_to_mni_warp: Path,
    mni_reference: Path,
    repetition_time: float,
    env: dict[str, str],
    force: bool,
) -> Step:
    def execute() -> None:
        fsl_commands = _resolve_ica_aroma_fsl_commands(runner, env)
        run_melodic_decomposition(
            run_cmd=lambda command: runner.run_child(list(command), env=env),
            run_out=lambda command: (
                runner.run_child(list(command), env=env, capture_stdout=True) or ""
            ),
            fsl_cmds=fsl_commands,
            in_file=melodic_input,
            out_dir=output_directory,
            mask=melodic_mask,
            tr=repetition_time,
            mni_ref=mni_reference,
            affmat=identity_transform,
            warp=t1_to_mni_warp,
            overwrite=True,
        )

    def validate() -> tuple[bool, str]:
        missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            "MELODIC directory contains all CICADA inputs."
            if not missing
            else "MELODIC directory is incomplete: " + ", ".join(missing),
        )

    return Step.directory_step(
        name="Estimate MELODIC Decomposition for CICADA",
        directory=output_directory,
        breadcrumb=output_directory / ".nro_complete",
        outputs=outputs,
        inputs=(
            melodic_input,
            melodic_mask,
            identity_transform,
            t1_to_mni_warp,
            mni_reference,
        ),
        force=force,
        action=execute,
        validate=validate,
        breadcrumb_text="CICADA MELODIC preparation complete\n",
    )


def _create_cicada_classification_step(
    *,
    runner: Runner,
    executable: Path,
    epi_mni: Path,
    mask_mni: Path,
    confounds: Path,
    source_melodic: Path,
    melodic_complete: Path,
    thresholded_components_mni: Path,
    t1_to_mni_warp: Path,
    identity_transform: Path,
    task_directory: Path,
    adapter_directory: Path,
    result_manifest: Path,
    tolerance: int,
    smoothing_retention_mode: str,
    repetition_time: float,
    fsl_image: Path,
    fsl_runtime: str,
    fsl_binds: tuple[str, ...],
    fsl_setup: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    classification = task_directory / "cicada_python"
    outputs = (
        classification / "noise_components.txt",
        classification / "signal_components.txt",
        classification / "component_labels.tsv",
        classification / "provenance.json",
        result_manifest,
    )

    def execute() -> None:
        if not executable.is_file():
            raise SystemExit(f"Configured CICADA executable does not exist: {executable}")
        reset_directory(adapter_directory)
        prepare_melodic_adapter(
            run_command=lambda command: runner.run_child(list(command), env=env),
            source_directory=source_melodic,
            output_directory=adapter_directory,
            reference=mask_mni,
            warp=t1_to_mni_warp,
            premat=identity_transform,
        )
        threshold_target = adapter_directory / "ICthresh_zstat.nii.gz"
        threshold_target.symlink_to(thresholded_components_mni)
        reset_directory(task_directory)
        command = [
            str(executable),
            "run",
            "--output-dir",
            str(task_directory),
            "--bold",
            str(epi_mni),
            "--mask",
            str(mask_mni),
            "--confounds",
            str(confounds),
            "--melodic-dir",
            str(adapter_directory),
            "--smoothing-retention-mode",
            smoothing_retention_mode,
            "--repetition-time",
            f"{repetition_time:.8g}",
            "--tolerance",
            str(tolerance),
            "--no-denoise",
            "--fsl-image",
            str(fsl_image),
            "--runtime",
            fsl_runtime,
            "--fsl-setup",
            fsl_setup,
        ]
        for bind in fsl_binds:
            command.extend(("--bind", bind))
        runner.run_direct(command, env=env)
        write_result_manifest(
            output=result_manifest,
            executable=executable,
            classification_directory=classification,
            tolerance=tolerance,
            smoothing_retention_mode=smoothing_retention_mode,
            mixing_matrix=source_melodic / "melodic_mix",
        )

    return Step.python(
        name="Run CICADA Component Classification",
        inputs=(
            executable,
            epi_mni,
            mask_mni,
            confounds,
            source_melodic / "melodic_IC.nii.gz",
            source_melodic / "melodic_mix",
            source_melodic / "melodic_FTmix",
            melodic_complete,
            thresholded_components_mni,
            t1_to_mni_warp,
            identity_transform,
        ),
        outputs=outputs,
        force=force,
        action=execute,
        parameters={
            "classifier": "cicada",
            "tolerance": tolerance,
            "smoothing_retention_mode": smoothing_retention_mode,
        },
    )


def _create_ica_aroma_workflow_step(
    *,
    runner: Runner,
    epi: Path,
    melodic_input: Path,
    motion_parameters: Path,
    melodic_mask: Path,
    regression_mask: Path,
    aroma_dir: Path,
    outputs: tuple[Path, ...],
    melodic_products: tuple[Path, ...],
    input_is_mni: bool,
    identity_transform: Path,
    t1_to_mni_warp: Optional[Path],
    mni_reference: Path,
    configured_command: Optional[Path],
    repetition_time: Optional[float],
    denoise_type: str,
    env: dict[str, str],
    force: bool,
) -> Step:
    def execute() -> None:
        aroma_command = (
            _resolve_ica_aroma_cmd(runner, env, configured_cmd=configured_command)
            if configured_command is not None
            else None
        )
        fsl_commands = _resolve_ica_aroma_fsl_commands(runner, env)
        if aroma_command is None:
            if repetition_time is None or repetition_time <= 0:
                raise SystemExit("ICA-AROMA requires a positive RepetitionTime in the EPI JSON.")
            run_ica_aroma_workflow(
                run_cmd=lambda cmd: runner.run_child(list(cmd), env=env),
                run_out=lambda cmd: runner.run_child(list(cmd), env=env, capture_stdout=True) or "",
                fsl_cmds=fsl_commands,
                in_file=epi,
                melodic_in_file=melodic_input,
                out_dir=aroma_dir,
                mc=motion_parameters,
                affmat=None if input_is_mni else identity_transform,
                warp=t1_to_mni_warp,
                mask=melodic_mask,
                regression_mask=regression_mask,
                tr=float(repetition_time),
                denoise_type=denoise_type,
                mni_ref=mni_reference,
                overwrite=True,
                melodic_dir=None,
            )
            return

        arguments = [
            "-in",
            str(melodic_input),
            "-out",
            str(aroma_dir),
            "-mc",
            str(motion_parameters),
            "-m",
            str(melodic_mask),
            "-den",
            "no",
        ]
        if not input_is_mni:
            arguments.extend(["-affmat", str(identity_transform), "-warp", str(t1_to_mni_warp)])
        if repetition_time is not None and repetition_time > 0:
            arguments.extend(["-tr", f"{float(repetition_time):.8g}"])
        runner.run_child([aroma_command, *arguments], env=env)
        for product in melodic_products[:-1]:
            if not product.is_file() or product.stat().st_size == 0:
                raise SystemExit(f"ICA-AROMA completed without required MELODIC product: {product}")
        write_completion_breadcrumb(melodic_products[-1], "MELODIC decomposition complete\n")
        classified = aroma_dir / "classified_motion_ICs.txt"
        indices = [
            int(value) - 1
            for value in classified.read_text(encoding="utf-8").strip().split(",")
            if value.strip()
        ]
        run_ica_aroma_denoising(
            run_cmd=lambda cmd: runner.run_child(list(cmd), env=env),
            fsl_cmds=fsl_commands,
            in_file=epi,
            mask=regression_mask,
            out_dir=aroma_dir,
            melmix=aroma_dir / "melodic.ica" / "melodic_mix",
            denoise_type=denoise_type,
            denoise_indices=indices,
        )

    def validate() -> tuple[bool, str]:
        missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
        return (
            not missing,
            "ICA-AROMA directory contains all required products."
            if not missing
            else "ICA-AROMA directory is incomplete: " + ", ".join(missing),
        )

    inputs = [epi, melodic_input, motion_parameters, melodic_mask, regression_mask]
    if configured_command is None:
        inputs.append(mni_reference)
    if not input_is_mni:
        inputs.extend((identity_transform, t1_to_mni_warp))
    return Step.directory_step(
        name="Run ICA-AROMA Classification and Denoising",
        directory=aroma_dir,
        breadcrumb=aroma_dir / ".nro_complete",
        outputs=outputs,
        inputs=inputs,
        force=force,
        action=execute,
        validate=validate,
        breadcrumb_text="ICA-AROMA workflow complete\n",
    )


def _create_shared_aroma_regression_step(
    *,
    runner: Runner,
    epi: Path,
    input_space: str,
    regression_mask: Path,
    mixing_matrix: Path,
    classified_components: Path,
    shared_policy: Path,
    aroma_dir: Path,
    outputs: tuple[Path, ...],
    denoise_type: str,
    classifier_name: str = "ICA-AROMA",
    env: dict[str, str],
    force: bool,
) -> Step:
    def regress() -> None:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore

        n_timepoints = int(nib.load(str(epi)).shape[3])
        mixing = np.loadtxt(mixing_matrix, ndmin=2)
        if int(mixing.shape[0]) != n_timepoints:
            raise SystemExit(
                f"Shared T1w {classifier_name} mixing matrix does not match registered "
                f"BOLD length: {mixing.shape[0]} != {n_timepoints}"
            )
        indices = [
            int(value)
            for value in classified_components.read_text(encoding="utf-8").strip().split(",")
            if value.strip()
        ]
        denoise_indices = np.asarray([value - 1 for value in indices], dtype=int)
        fsl_regfilt = _resolve_container_command_for_wrapper(
            runner=runner, env=env, command="fsl_regfilt"
        )
        if fsl_regfilt is None:
            raise SystemExit(
                f"Missing required FSL command for shared {classifier_name} regression: fsl_regfilt"
            )
        LOG.info(
            "Applying %d T1w-classified %s noise components to %s registered BOLD",
            len(indices),
            classifier_name,
            input_space,
        )
        aroma_dir.mkdir(parents=True, exist_ok=True)
        run_ica_aroma_denoising(
            run_cmd=lambda cmd: runner.run_child(list(cmd), env=env),
            fsl_cmds={"fsl_regfilt": fsl_regfilt},
            in_file=epi,
            mask=regression_mask,
            out_dir=aroma_dir,
            melmix=mixing_matrix,
            denoise_type=denoise_type,
            denoise_indices=denoise_indices,
        )

    return Step.python(
        name=f"Regress Shared {classifier_name} Components in {input_space}",
        outputs=outputs,
        inputs=(
            epi,
            regression_mask,
            mixing_matrix,
            classified_components,
            shared_policy,
        ),
        force=force,
        action=regress,
    )
