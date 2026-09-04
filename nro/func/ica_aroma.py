from __future__ import annotations

import os
import shutil
import re
from importlib import resources
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from nro.orchestration.runner import write_completion_breadcrumb


RunCommand = Callable[[Sequence[str]], None]
RunOutCommand = Callable[[Sequence[str]], str]


def make_dilated_anatomical_epi_mask(
    *,
    anatomical_mask: Path,
    epi_support_mask: Path,
    dilation_mm: float,
    out_mask: Path,
) -> None:
    """Dilate an anatomical mask in physical units and clip it to EPI support."""
    import nibabel as nib  # type: ignore
    from scipy.ndimage import distance_transform_edt  # type: ignore

    radius_mm = float(dilation_mm)
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError(f"Mask dilation must be finite and nonnegative, got {dilation_mm!r}")

    anatomical_img = nib.load(str(anatomical_mask))
    support_img = nib.load(str(epi_support_mask))
    anatomical_shape = tuple(int(v) for v in anatomical_img.shape[:3])
    support_shape = tuple(int(v) for v in support_img.shape[:3])
    if anatomical_shape != support_shape or not np.allclose(
        anatomical_img.affine,
        support_img.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Anatomical and EPI support masks must share a grid: "
            f"anatomical={anatomical_mask} shape={anatomical_shape}, "
            f"support={epi_support_mask} shape={support_shape}"
        )

    anatomical = np.asarray(anatomical_img.dataobj) > 0
    support = np.asarray(support_img.dataobj) > 0
    zooms = tuple(float(v) for v in anatomical_img.header.get_zooms()[:3])
    distance_mm = distance_transform_edt(~anatomical, sampling=zooms)
    result = ((distance_mm <= radius_mm) & support).astype(np.uint8)
    if not np.any(result):
        raise RuntimeError(
            "Dilated anatomical mask has no overlap with liberal EPI support: "
            f"anatomical={anatomical_mask}, support={epi_support_mask}"
        )

    header = anatomical_img.header.copy()
    header.set_data_dtype(np.uint8)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    tmp_mask = out_mask.with_name(f".{out_mask.name}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(result, anatomical_img.affine, header), str(tmp_mask))
    os.replace(tmp_mask, out_mask)


def _resource_path(name: str) -> Path:
    return Path(resources.files("nro.func.resources.ica_aroma").joinpath(name))


def _parse_fslinfo_value(text: str, key: str) -> float:
    cleaned = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line.startswith(key):
            continue
        parts = line.split()
        if len(parts) >= 2:
            return float(parts[1])
    raise RuntimeError(f"Could not parse {key!r} from fslinfo output:\n{text}")


def _fslinfo_value(run_out: RunOutCommand, fsl_cmds: Mapping[str, str], img: Path, key: str) -> float:
    out = run_out([fsl_cmds["fslinfo"], str(img)])
    return _parse_fslinfo_value(out, key)


def _run_melodic_and_merge_thresholded_maps(
    *,
    run_cmd: RunCommand,
    run_out: RunOutCommand,
    fsl_cmds: Mapping[str, str],
    in_file: Path,
    out_dir: Path,
    mask: Path,
    dim: int,
    tr: float,
    melodic_dir: Path | None,
) -> None:
    try:
        import nibabel as nib  # type: ignore
    except Exception:
        nib = None  # type: ignore[assignment]

    mel_dir = out_dir / "melodic.ica"
    mel_ic = mel_dir / "melodic_IC.nii.gz"
    mel_ic_mix = mel_dir / "melodic_mix"
    mel_ic_thr = out_dir / "melodic_IC_thr.nii.gz"

    use_existing = (
        melodic_dir is not None
        and (melodic_dir / "melodic_IC.nii.gz").exists()
        and (melodic_dir / "melodic_FTmix").exists()
        and (melodic_dir / "melodic_mix").exists()
    )
    if use_existing:
        if mel_dir.exists() or mel_dir.is_symlink():
            if mel_dir.is_symlink() or mel_dir.is_file():
                mel_dir.unlink()
            else:
                if melodic_dir is None or melodic_dir.resolve() != mel_dir.resolve():
                    shutil.rmtree(mel_dir)
        if (melodic_dir / "stats").is_dir():
            if melodic_dir.resolve() != mel_dir.resolve():
                mel_dir.symlink_to(melodic_dir, target_is_directory=True)
        else:
            mel_dir.mkdir(parents=True, exist_ok=True)
            for item in melodic_dir.iterdir():
                (mel_dir / item.name).symlink_to(item)
            run_cmd(
                [
                    fsl_cmds["melodic"],
                    f"--in={mel_ic}",
                    f"--ICs={mel_ic}",
                    f"--mix={mel_ic_mix}",
                    f"--outdir={mel_dir}",
                    "--Ostats",
                    "--mmthresh=0.5",
                ]
            )
    else:
        run_cmd(
            [
                fsl_cmds["melodic"],
                f"--in={in_file}",
                f"--outdir={mel_dir}",
                f"--mask={mask}",
                f"--dim={int(dim)}",
                "--Ostats",
                "--nobet",
                "--mmthresh=0.5",
                f"--tr={float(tr):.8g}",
            ]
        )

    num_ics = int(round(_fslinfo_value(run_out, fsl_cmds, mel_ic, "dim4")))
    tmp_vols: list[Path] = []
    for i in range(1, num_ics + 1):
        z_temp = mel_dir / "stats" / f"thresh_zstat{i}.nii.gz"
        z_stat = out_dir / f"thr_zstat{i:04d}.nii.gz"
        tmp_vols.append(z_stat)
        len_ic: int | None = None
        if nib is not None:
            try:
                shape = nib.load(str(z_temp)).shape
                len_ic = int(shape[3]) if len(shape) >= 4 else 1
            except Exception:
                len_ic = None
        if len_ic is None:
            len_ic = int(round(_fslinfo_value(run_out, fsl_cmds, z_temp, "dim4")))
        if len_ic <= 1:
            shutil.copyfile(z_temp, z_stat)
        else:
            run_cmd([fsl_cmds["fslroi"], str(z_temp), str(z_stat), str(max(0, len_ic - 1)), "1"])

    run_cmd([fsl_cmds["fslmerge"], "-t", str(mel_ic_thr), *(str(p) for p in tmp_vols)])
    run_cmd([fsl_cmds["fslmaths"], str(mel_ic_thr), "-mas", str(mask), str(mel_ic_thr)])
    for tmp in tmp_vols:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()


def _register_to_mni(
    *,
    run_cmd: RunCommand,
    run_out: RunOutCommand,
    fsl_cmds: Mapping[str, str],
    in_file: Path,
    out_file: Path,
    ref: Path,
    affmat: Path | None,
    warp: Path | None,
) -> None:
    has_aff = affmat is not None
    has_warp = warp is not None
    if (not has_aff) and (not has_warp):
        pixdim1 = _fslinfo_value(run_out, fsl_cmds, in_file, "pixdim1")
        pixdim2 = _fslinfo_value(run_out, fsl_cmds, in_file, "pixdim2")
        pixdim3 = _fslinfo_value(run_out, fsl_cmds, in_file, "pixdim3")
        if (pixdim1 != 2.0) or (pixdim2 != 2.0) or (pixdim3 != 2.0):
            run_cmd(
                [
                    fsl_cmds["flirt"],
                    "-ref",
                    str(ref),
                    "-in",
                    str(in_file),
                    "-out",
                    str(out_file),
                    "-applyisoxfm",
                    "2",
                    "-interp",
                    "trilinear",
                ]
            )
        else:
            shutil.copyfile(in_file, out_file)
        return
    if (not has_aff) and has_warp:
        run_cmd(
            [
                fsl_cmds["applywarp"],
                f"--ref={ref}",
                f"--in={in_file}",
                f"--out={out_file}",
                f"--warp={warp}",
                "--interp=trilinear",
            ]
        )
        return
    if has_aff and (not has_warp):
        run_cmd(
            [
                fsl_cmds["flirt"],
                "-ref",
                str(ref),
                "-in",
                str(in_file),
                "-out",
                str(out_file),
                "-applyxfm",
                "-init",
                str(affmat),
                "-interp",
                "trilinear",
            ]
        )
        return
    run_cmd(
        [
            fsl_cmds["applywarp"],
            f"--ref={ref}",
            f"--in={in_file}",
            f"--out={out_file}",
            f"--warp={warp}",
            f"--premat={affmat}",
            "--interp=trilinear",
        ]
    )


def cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.ndim == b.ndim == 2
    _, ncols_a = a.shape
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


def feature_time_series(melmix: Path, mc: Path) -> np.ndarray:
    import random

    mix = np.loadtxt(melmix)
    rp6 = np.loadtxt(mc)
    rp6 = np.atleast_2d(rp6)
    if mix.ndim == 1:
        if mix.size != rp6.shape[0]:
            raise RuntimeError(
                f"Cannot orient one-dimensional MELODIC mixing matrix with {mix.size} values "
                f"against {rp6.shape[0]} motion rows: {melmix}"
            )
        mix = mix[:, np.newaxis]
    else:
        mix = np.atleast_2d(mix)
    if mix.shape[0] != rp6.shape[0]:
        raise RuntimeError(
            f"MELODIC mixing matrix has {mix.shape[0]} time points but motion parameters have "
            f"{rp6.shape[0]} rows: {melmix}, {mc}"
        )
    _, nparams = rp6.shape

    rp6_der = np.vstack((np.zeros(nparams), np.diff(rp6, axis=0)))
    rp12 = np.hstack((rp6, rp6_der))

    rp12_1fw = np.vstack((np.zeros(2 * nparams), rp12[:-1]))
    rp12_1bw = np.vstack((rp12[1:], np.zeros(2 * nparams)))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    nsplits = 1000
    nmixrows, nmixcols = mix.shape
    nrows_to_choose = int(round(0.9 * nmixrows))

    max_correls = np.empty((nsplits, nmixcols))
    for i in range(nsplits):
        chosen_rows = random.sample(population=range(nmixrows), k=nrows_to_choose)
        correl_nonsquared = cross_correlation(mix[chosen_rows], rp_model[chosen_rows])
        correl_squared = cross_correlation(mix[chosen_rows] ** 2, rp_model[chosen_rows] ** 2)
        correl_both = np.hstack((correl_squared, correl_nonsquared))
        max_correls[i] = np.abs(correl_both).max(axis=1)

    return np.nanmean(max_correls, axis=0)


def feature_frequency(mel_ftmix: Path, tr: float) -> np.ndarray:
    fs = 1.0 / tr
    nyquist = fs / 2.0
    ft = np.loadtxt(mel_ftmix)
    if ft.ndim == 1:
        ft = ft[:, np.newaxis]
    else:
        ft = np.atleast_2d(ft)

    f = nyquist * (np.arange(1, ft.shape[0] + 1)) / ft.shape[0]
    fincl = np.flatnonzero(f > 0.01)
    if fincl.size == 0:
        raise RuntimeError(f"MELODIC frequency spectrum contains no frequencies above 0.01 Hz: {mel_ftmix}")
    ft = ft[fincl, :]
    f = f[fincl]

    f_norm = (f - 0.01) / (nyquist - 0.01)
    fcumsum_fract = np.cumsum(ft, axis=0) / np.sum(ft, axis=0)
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)
    return f_norm[idx_cutoff]


def feature_spatial(
    *,
    run_cmd: RunCommand,
    run_out: RunOutCommand,
    fsl_cmds: Mapping[str, str],
    temp_dir: Path,
    mel_ic: Path,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ICA-AROMA spatial features require nibabel: {e}") from e

    edge_mask = _resource_path("mask_edge.nii.gz")
    csf_mask = _resource_path("mask_csf.nii.gz")
    out_mask = _resource_path("mask_out.nii.gz")
    mel_img = nib.load(str(mel_ic))
    mel_data = np.asarray(mel_img.get_fdata(dtype=np.float32))
    if mel_data.ndim == 3:
        mel_data = mel_data[..., np.newaxis]
    if mel_data.ndim != 4:
        raise RuntimeError(f"Expected 3D/4D ICA map image, got shape {mel_data.shape} for {mel_ic}")

    abs_data = np.abs(mel_data).reshape(-1, mel_data.shape[3])

    def _mask_on_mel_grid(mask_path: Path) -> np.ndarray:
        mask_img = nib.load(str(mask_path))
        mask_shape = tuple(int(v) for v in mask_img.shape[:3])
        mel_shape = tuple(int(v) for v in mel_img.shape[:3])
        if mask_shape != mel_shape or not np.allclose(mask_img.affine, mel_img.affine):
            mask_img = resample_from_to(mask_img, (mel_shape, mel_img.affine), order=0)
        return np.asarray(mask_img.get_fdata(dtype=np.float32) > 0, dtype=bool).reshape(-1)

    edge = _mask_on_mel_grid(edge_mask)
    csf = _mask_on_mel_grid(csf_mask)
    out = _mask_on_mel_grid(out_mask)

    tot_sum = abs_data.sum(axis=0)
    csf_sum = abs_data[csf, :].sum(axis=0)
    edge_sum = abs_data[edge, :].sum(axis=0)
    out_sum = abs_data[out, :].sum(axis=0)

    edge_fract = np.zeros(abs_data.shape[1], dtype=np.float64)
    csf_fract = np.zeros(abs_data.shape[1], dtype=np.float64)
    nonzero_tot = tot_sum != 0
    denom = tot_sum - csf_sum
    valid_edge = nonzero_tot & (denom != 0)
    edge_fract[valid_edge] = (out_sum[valid_edge] + edge_sum[valid_edge]) / denom[valid_edge]
    csf_fract[nonzero_tot] = csf_sum[nonzero_tot] / tot_sum[nonzero_tot]
    return edge_fract, csf_fract


def classification(out_dir: Path, max_rp_corr: np.ndarray, edge_fract: np.ndarray, hfc: np.ndarray, csf_fract: np.ndarray) -> np.ndarray:
    thr_csf = 0.10
    thr_hfc = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    x = np.array([max_rp_corr, edge_fract])
    proj = hyp[0] + np.dot(x.T, hyp[1:])
    motion_ics = np.squeeze(np.array(np.where((proj > 0) + (csf_fract > thr_csf) + (hfc > thr_hfc))))

    np.savetxt(out_dir / "feature_scores.txt", np.vstack((max_rp_corr, edge_fract, hfc, csf_fract)).T)

    classified = out_dir / "classified_motion_ICs.txt"
    with classified.open("w", encoding="utf-8") as txt:
        if np.size(motion_ics) > 1:
            txt.write(",".join(["{:.0f}".format(num) for num in (np.atleast_1d(motion_ics) + 1)]))
        elif np.size(motion_ics) == 1:
            txt.write("{:.0f}".format(float(np.atleast_1d(motion_ics)[0]) + 1))
        txt.write("\n")

    overview = out_dir / "classification_overview.txt"
    with overview.open("w", encoding="utf-8") as txt:
        txt.write(
            "\t".join(
                ["IC", "Motion/noise", "maximum RP correlation", "Edge-fraction", "High-frequency content", "CSF-fraction"]
            )
        )
        txt.write("\n")
        for i in range(len(csf_fract)):
            classif = "True" if (proj[i] > 0) or (csf_fract[i] > thr_csf) or (hfc[i] > thr_hfc) else "False"
            txt.write(
                "\t".join(
                    [
                        f"{i + 1:d}",
                        classif,
                        f"{max_rp_corr[i]:.2f}",
                        f"{edge_fract[i]:.2f}",
                        f"{hfc[i]:.2f}",
                        f"{csf_fract[i]:.2f}",
                    ]
                )
            )
            txt.write("\n")
    return np.atleast_1d(motion_ics)


def denoising(
    *,
    run_cmd: RunCommand,
    fsl_cmds: Mapping[str, str],
    in_file: Path,
    mask: Path,
    out_dir: Path,
    melmix: Path,
    denoise_type: str,
    denoise_indices: np.ndarray,
) -> None:
    has_motion = np.size(denoise_indices) > 0
    if has_motion:
        if np.size(denoise_indices) == 1:
            joined = f"{int(np.atleast_1d(denoise_indices)[0]) + 1:d}"
        else:
            joined = ",".join(np.char.mod("%i", np.atleast_1d(denoise_indices) + 1))

        if denoise_type in {"nonaggr", "both"}:
            run_cmd(
                [
                    fsl_cmds["fsl_regfilt"],
                    f"--in={in_file}",
                    f"--mask={mask}",
                    f"--design={melmix}",
                    f"--filter={joined}",
                    f"--out={out_dir / 'denoised_func_data_nonaggr.nii.gz'}",
                ]
            )
        if denoise_type in {"aggr", "both"}:
            run_cmd(
                [
                    fsl_cmds["fsl_regfilt"],
                    f"--in={in_file}",
                    f"--mask={mask}",
                    f"--design={melmix}",
                    f"--filter={joined}",
                    f"--out={out_dir / 'denoised_func_data_aggr.nii.gz'}",
                    "-a",
                ]
            )
        return

    if denoise_type in {"nonaggr", "both"}:
        out_nonaggr = out_dir / "denoised_func_data_nonaggr.nii.gz"
        if out_nonaggr.exists() or out_nonaggr.is_symlink():
            out_nonaggr.unlink()
        out_nonaggr.symlink_to(in_file)
    if denoise_type in {"aggr", "both"}:
        out_aggr = out_dir / "denoised_func_data_aggr.nii.gz"
        if out_aggr.exists() or out_aggr.is_symlink():
            out_aggr.unlink()
        out_aggr.symlink_to(in_file)


def run_ica_aroma_workflow(
    *,
    run_cmd: RunCommand,
    run_out: RunOutCommand,
    fsl_cmds: Mapping[str, str],
    in_file: Path,
    out_dir: Path,
    mc: Path,
    affmat: Path | None,
    warp: Path | None,
    mask: Path,
    regression_mask: Path,
    tr: float,
    denoise_type: str,
    mni_ref: Path,
    melodic_in_file: Path | None = None,
    overwrite: bool = False,
    dim: int = 0,
    melodic_dir: Path | None = None,
) -> None:
    if tr <= 0:
        raise RuntimeError(f"ICA-AROMA requires a positive TR, got {tr!r}")
    if denoise_type not in {"nonaggr", "aggr", "both", "no"}:
        raise RuntimeError(f"Unsupported ICA-AROMA denoise type: {denoise_type!r}")

    if out_dir.exists():
        if not overwrite:
            raise RuntimeError(f"ICA-AROMA output directory already exists: {out_dir}")
        reuse_existing_melodic = (
            melodic_dir is not None
            and melodic_dir.exists()
            and melodic_dir.resolve().is_relative_to(out_dir.resolve())
        )
        if reuse_existing_melodic:
            for child in list(out_dir.iterdir()):
                try:
                    if child.resolve() == melodic_dir.resolve():
                        continue
                except Exception:
                    pass
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_out = out_dir / "mask.nii.gz"
    shutil.copyfile(mask, mask_out)
    regression_mask_out = out_dir / "regression_mask.nii.gz"
    shutil.copyfile(regression_mask, regression_mask_out)
    estimation_file = melodic_in_file if melodic_in_file is not None else in_file

    _run_melodic_and_merge_thresholded_maps(
        run_cmd=run_cmd,
        run_out=run_out,
        fsl_cmds=fsl_cmds,
        in_file=estimation_file,
        out_dir=out_dir,
        mask=mask_out,
        dim=dim,
        tr=tr,
        melodic_dir=melodic_dir,
    )
    write_completion_breadcrumb(
        out_dir / "melodic.complete",
        "MELODIC decomposition and thresholded-map merge complete\n",
    )

    mel_ic = out_dir / "melodic_IC_thr.nii.gz"
    mel_ic_mni = out_dir / "melodic_IC_thr_MNI2mm.nii.gz"
    _register_to_mni(
        run_cmd=run_cmd,
        run_out=run_out,
        fsl_cmds=fsl_cmds,
        in_file=mel_ic,
        out_file=mel_ic_mni,
        ref=mni_ref,
        affmat=affmat,
        warp=warp,
    )

    edge_fract, csf_fract = feature_spatial(
        run_cmd=run_cmd,
        run_out=run_out,
        fsl_cmds=fsl_cmds,
        temp_dir=out_dir,
        mel_ic=mel_ic_mni,
    )
    melmix = out_dir / "melodic.ica" / "melodic_mix"
    mel_ftmix = out_dir / "melodic.ica" / "melodic_FTmix"
    max_rp_corr = feature_time_series(melmix, mc)
    hfc = feature_frequency(mel_ftmix, tr)
    motion_ics = classification(out_dir, max_rp_corr, edge_fract, hfc, csf_fract)
    if denoise_type != "no":
        denoising(
            run_cmd=run_cmd,
            fsl_cmds=fsl_cmds,
            in_file=in_file,
            mask=regression_mask_out,
            out_dir=out_dir,
            melmix=melmix,
            denoise_type=denoise_type,
            denoise_indices=motion_ics,
        )
