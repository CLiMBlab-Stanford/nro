"""Sdc steps for functional preprocessing."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from nro.engine.execution import (
    ensure_directory,
)
from nro.engine.io import atomic_write_text, read_json, write_json
from nro.orchestration.runner_graph import Step

from .constants import (
    _FIELDMAP_TRANSFER_POLICY_VERSION,
)
from .step_support import LOG


def _pe_to_topup_dir(phase_encoding_direction: str) -> Tuple[int, int, int]:
    ped = phase_encoding_direction.strip()
    mapping = {
        "i": (1, 0, 0),
        "i-": (-1, 0, 0),
        "j": (0, 1, 0),
        "j-": (0, -1, 0),
        "k": (0, 0, 1),
        "k-": (0, 0, -1),
    }
    if ped not in mapping:
        raise ValueError(f"Unsupported PhaseEncodingDirection: {phase_encoding_direction!r}")
    return mapping[ped]


def _pe_to_fsl_shift_direction(phase_encoding_direction: str) -> str:
    """Translate a BIDS PE direction into FSL's signed shift-axis syntax."""
    ped = str(phase_encoding_direction).strip()
    mapping = {
        "i": "x",
        "i-": "x-",
        "j": "y",
        "j-": "y-",
        "k": "z",
        "k-": "z-",
    }
    try:
        return mapping[ped]
    except KeyError as error:
        raise SystemExit(
            f"Unsupported PhaseEncodingDirection: {phase_encoding_direction!r}"
        ) from error


def _canonical_fieldmap_order(
    first: tuple[Path, dict[str, Any]],
    second: tuple[Path, dict[str, Any]],
) -> tuple[tuple[Path, dict[str, Any]], tuple[Path, dict[str, Any]]]:
    """Return a run-independent ordering for an opposite-PE fieldmap pair."""

    def key(item: tuple[Path, dict[str, Any]]) -> tuple[str, bool, str]:
        path, metadata = item
        ped = str(metadata.get("PhaseEncodingDirection", "")).strip()
        axis = ped.rstrip("-")
        return axis, ped.endswith("-"), str(path)

    return tuple(sorted((first, second), key=key))  # type: ignore[return-value]


def _write_topup_datain(
    *,
    out_txt: Path,
    ped_a: str,
    ped_b: str,
    readout_time: float,
    a_nvols: int,
    b_nvols: int,
    readout_time_b: Optional[float] = None,
) -> None:
    d1 = _pe_to_topup_dir(ped_a)
    d2 = _pe_to_topup_dir(ped_b)
    l1 = f"{d1[0]} {d1[1]} {d1[2]} {readout_time:.8f}"
    second_readout = readout_time if readout_time_b is None else float(readout_time_b)
    l2 = f"{d2[0]} {d2[1]} {d2[2]} {second_readout:.8f}"
    lines = ([l1] * max(1, int(a_nvols))) + ([l2] * max(1, int(b_nvols)))
    ensure_directory(out_txt.parent)
    atomic_write_text(out_txt, "\n".join(lines) + "\n")


def _motion_matrix_breadcrumb(matrix_dir: Path) -> Path:
    return matrix_dir.with_name(f"{matrix_dir.name}.complete")


@dataclass(frozen=True)
class TopupDfOutputs:
    """Declared TOPUP outputs and displacement-field paths for composing SDC transforms."""

    step: Step
    out_prefix: Path
    field_hz: Path
    iout: Path
    dfout: Path
    jacout: Path
    rbmout: Path
    a_nvols: int
    b_nvols: int


def _normalized_topup_matrix(prefix: Path, *, index_1based: int) -> Path:
    """Return one canonical matrix path from a normalized TOPUP directory."""
    index = int(index_1based)
    if index < 1:
        raise SystemExit(f"topup motion-matrix index must be >= 1 (got {index_1based})")
    return prefix.parent / f"{prefix.name}_{index:04d}.mat"


def _raw_topup_member(prefix: Path, *, index_1based: int, extension: str) -> Path:
    """Resolve a member inside TOPUP's opaque directory for normalization.

    FSL releases differ only in index padding.  These foreknown spellings are
    implementation details of the surrounding directory artifact and never
    become DAG outputs themselves.
    """
    index = int(index_1based)
    candidates = tuple(
        prefix.parent / f"{prefix.name}_{index:0{width}d}{extension}" for width in (2, 1, 3, 4)
    )
    for path in dict.fromkeys(candidates):
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise SystemExit(
        f"TOPUP did not produce indexed member {index} for {prefix}; checked: "
        + ", ".join(str(path) for path in dict.fromkeys(candidates))
    )


def _create_topup_dfout_step(
    *,
    run_child: Callable[..., Optional[str]],
    se_a: Path,
    se_b: Path,
    ped_a: str,
    ped_b: str,
    readout_time: float,
    topup_dir: Path,
    topup_config: str,
    env: dict[str, str],
    force: bool,
    spatial_shape: tuple[int, int, int],
    volumes_a: int,
    volumes_b: int,
    readout_time_b: Optional[float] = None,
) -> TopupDfOutputs:
    if topup_config.strip().lower() == "auto":
        topup_config = (
            "b02b0_2.cnf" if all(size % 2 == 0 for size in spatial_shape) else "b02b0_1.cnf"
        )
        LOG.info(
            "TOPUP configuration selected for image dimensions %s: %s",
            spatial_shape,
            topup_config,
        )
    merged = topup_dir / "se_merged.nii.gz"
    datain = topup_dir / "acqparams.txt"
    out_prefix = topup_dir / "topup_results"
    iout = topup_dir / "se_unwarped.nii.gz"
    fout = topup_dir / "fieldmap_Hz.nii.gz"
    # Use extension-less prefixes to avoid confusing FSL's output naming.
    dfout_prefix = topup_dir / "WarpField"
    jacout_prefix = topup_dir / "Jacobian"
    rbmout_prefix = topup_dir / "MotionMatrix"
    spec_path = topup_dir / "topup_spec.json"
    result_manifest = topup_dir / "topup_outputs.json"
    topup_complete = topup_dir / "topup.complete"

    a_nvols = int(volumes_a)
    b_nvols = int(volumes_b)
    if a_nvols < 1 or b_nvols < 1:
        raise ValueError("TOPUP inputs must each contain at least one volume")
    total_nvols = int(a_nvols) + int(b_nvols)
    second_readout = float(readout_time if readout_time_b is None else readout_time_b)
    spec: dict[str, object] = {
        "PolicyVersion": _FIELDMAP_TRANSFER_POLICY_VERSION,
        "InputA": str(se_a),
        "InputB": str(se_b),
        "PhaseEncodingDirectionA": str(ped_a),
        "PhaseEncodingDirectionB": str(ped_b),
        "TotalReadoutTimeA": float(readout_time),
        "TotalReadoutTimeB": second_readout,
        "VolumesA": int(a_nvols),
        "VolumesB": int(b_nvols),
        "Configuration": str(topup_config),
    }
    fieldcoef = topup_dir / "topup_results_fieldcoef.nii.gz"
    normalized_dir = topup_dir / "normalized"
    normalized_rbmout = normalized_dir / "MotionMatrix"
    normalized_warps = tuple(
        normalized_dir / f"WarpField_{index:04d}.nii.gz" for index in range(1, total_nvols + 1)
    )
    normalized_jacobians = tuple(
        normalized_dir / f"Jacobian_{index:04d}.nii.gz" for index in range(1, total_nvols + 1)
    )
    normalized_matrices = tuple(
        normalized_dir / f"MotionMatrix_{index:04d}.mat" for index in range(1, total_nvols + 1)
    )
    topup_cmd = [
        "topup",
        f"--imain={merged}",
        f"--datain={datain}",
        f"--config={topup_config}",
        f"--out={out_prefix}",
        f"--iout={iout}",
        f"--fout={fout}",
        f"--dfout={dfout_prefix}",
        f"--jacout={jacout_prefix}",
        f"--rbmout={rbmout_prefix}",
    ]

    def execute_topup_directory() -> None:
        ensure_directory(topup_dir)
        write_json(spec_path, spec)
        run_child(
            ["fslmerge", "-t", str(merged), str(se_a), str(se_b)],
            env=env,
        )
        _write_topup_datain(
            out_txt=datain,
            ped_a=ped_a,
            ped_b=ped_b,
            readout_time=float(readout_time),
            readout_time_b=readout_time_b,
            a_nvols=int(a_nvols),
            b_nvols=int(b_nvols),
        )
        LOG.info("SDC: running topup (dfout/jacout) in %s", topup_dir)
        run_child(topup_cmd, env=env)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        for index, destination in enumerate(normalized_warps, start=1):
            shutil.copy2(
                _raw_topup_member(dfout_prefix, index_1based=index, extension=".nii.gz"),
                destination,
            )
        for index, destination in enumerate(normalized_jacobians, start=1):
            shutil.copy2(
                _raw_topup_member(jacout_prefix, index_1based=index, extension=".nii.gz"),
                destination,
            )
        for index, destination in enumerate(normalized_matrices, start=1):
            shutil.copy2(
                _raw_topup_member(rbmout_prefix, index_1based=index, extension=".mat"),
                destination,
            )
        write_json(
            result_manifest,
            {
                "spec": spec,
                "field_hz": str(fout),
                "iout": str(iout),
                "dfout": str(normalized_warps[0]),
                "jacout": str(normalized_jacobians[0]),
                "rbmout": str(normalized_rbmout),
                "warps": [str(path) for path in normalized_warps],
                "jacobians": [str(path) for path in normalized_jacobians],
                "motion_matrices": [str(path) for path in normalized_matrices],
            },
        )

    def validate_topup_directory() -> tuple[bool, str]:
        required = (
            spec_path,
            result_manifest,
            fieldcoef,
            iout,
            fout,
            *normalized_warps,
            *normalized_jacobians,
            *normalized_matrices,
        )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            return False, "TOPUP directory is missing required output(s): " + ", ".join(missing)
        try:
            recorded = read_json(result_manifest)
        except Exception:
            return False, f"TOPUP output inventory is unreadable: {result_manifest}"
        if recorded.get("spec") != spec:
            return False, "TOPUP output inventory does not match the current specification."
        return True, f"TOPUP directory contains the fixed {total_nvols}-volume output inventory."

    return TopupDfOutputs(
        step=Step.directory_step(
            name="TOPUP Distortion Estimation Directory",
            directory=topup_dir,
            breadcrumb=topup_complete,
            outputs=(result_manifest,),
            inputs=(se_a, se_b),
            force=force,
            action=execute_topup_directory,
            validate=validate_topup_directory,
            breadcrumb_text=f"volumes={total_nvols}\n",
        ),
        out_prefix=out_prefix,
        field_hz=fout,
        iout=iout,
        dfout=normalized_warps[0],
        jacout=normalized_jacobians[0],
        rbmout=normalized_rbmout,
        a_nvols=int(a_nvols),
        b_nvols=int(b_nvols),
    )


def _create_convertwarp_conjugate_affine_step(
    *,
    ref: Path,
    warp1: Path,
    premat: Path,
    postmat: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--premat={premat}",
        f"--warp1={warp1}",
        f"--postmat={postmat}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd,
        outputs=(out_warp,),
        inputs=(ref, warp1, premat, postmat),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_postmat_step(
    *,
    ref: Path,
    warp1: Path,
    postmat: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--warp1={warp1}",
        f"--postmat={postmat}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd,
        outputs=(out_warp,),
        inputs=(ref, warp1, postmat),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_premat_step(
    *,
    ref: Path,
    premat: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--premat={premat}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd,
        outputs=(out_warp,),
        inputs=(ref, premat),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_premat_and_warp_step(
    *,
    ref: Path,
    premat: Path,
    warp1: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--premat={premat}",
        f"--warp1={warp1}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd,
        outputs=(out_warp,),
        inputs=(ref, premat, warp1),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_convertwarp_merge_warps_step(
    *,
    ref: Path,
    warp1: Path,
    warp2: Path,
    out_warp: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    cmd = [
        "convertwarp",
        "--relout",
        "--rel",
        f"--ref={ref}",
        f"--warp1={warp1}",
        f"--warp2={warp2}",
        f"--out={out_warp}",
    ]
    return Step.command_step(
        cmd,
        outputs=(out_warp,),
        inputs=(ref, warp1, warp2),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(out_warp.parent),
    )


def _create_warp_jacobian_step(
    *,
    warp: Path,
    ref: Path,
    temporary: Path,
    junk: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    convert_cmd = [
        "convertwarp",
        "--rel",
        "-w",
        str(warp),
        "-r",
        str(ref),
    ]
    convert_cmd.extend([f"--jacobian={temporary}", "-o", str(junk)])
    return Step.command_step(
        convert_cmd,
        name="Compute Full Warp Jacobian",
        outputs=(temporary, junk),
        inputs=(warp, ref),
        force=force,
        env=env,
        prepare=lambda: ensure_directory(temporary.parent),
    )


def _create_average_jacobian_step(
    *,
    temporary: Path,
    junk: Path,
    warp: Path,
    ref: Path,
    out_jac: Path,
    env: dict[str, str],
    force: bool,
) -> Step:
    """Create the HCP-style mean over convertwarp's eight Jacobian volumes."""
    return Step.command_step(
        ["fslmaths", str(temporary), "-Tmean", str(out_jac)],
        name="Average Warp Jacobian Components",
        outputs=(out_jac,),
        inputs=(temporary, warp, ref),
        force=force,
        env=env,
        finalize=lambda: (temporary.unlink(missing_ok=True), junk.unlink(missing_ok=True)),
    )
