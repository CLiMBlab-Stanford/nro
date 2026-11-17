"""Name the planned files and directories of one functional graph.

Each path depends only on the functional options, the BOLD run, and the ICA
classifier, so the layout can be computed before any step is added. Paths
that depend on choices made while the graph is built stay with the code that
makes those choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .step_support import _with_suffix

if TYPE_CHECKING:
    from .config import Options

REG_REF_TAG = "regRef"
REG_REF_SPACE = "FunctionalReference"


@dataclass(frozen=True)
class FuncLayout:
    """Fixed paths used across the stages of one functional graph."""

    opts: Options
    run_stem: str
    classifier: str

    @property
    def run_base(self) -> str:
        """BOLD file stem without the `_bold` suffix."""
        return self.run_stem[: -len("_bold")] if self.run_stem.endswith("_bold") else self.run_stem

    @property
    def run_prefix(self) -> str:
        """Output name prefix for T1w-space files."""
        return f"{self.run_base}_space-T1w"

    @property
    def func_dir(self) -> Path:
        """Public functional output directory."""
        return self.opts.out_dir

    @property
    def fmap_dir(self) -> Path:
        """Public fieldmap output directory beside the functional outputs."""
        return self.func_dir.parent / "fmap"

    @property
    def sdc_dir(self) -> Path:
        """Work directory for distortion correction."""
        return self.opts.work_dir / "sdc"

    @property
    def reg_dir(self) -> Path:
        """Work directory for registration."""
        return self.opts.work_dir / "reg"

    @property
    def mc_dir(self) -> Path:
        """Work directory for motion correction."""
        return self.opts.work_dir / "mc"

    @property
    def surf_dir(self) -> Path:
        """Work directory for surface sampling."""
        return self.opts.work_dir / "surf"

    @property
    def qc_dir(self) -> Path:
        """Work directory for quality-control images."""
        return self.opts.work_dir / "qc"

    @property
    def preproc_t1_name(self) -> str:
        """File name of the preprocessed T1w-space BOLD series."""
        return _with_suffix(self.run_prefix, "_desc-preproc_bold.nii.gz")

    @property
    def preproc_t1(self) -> Path:
        """Published preprocessed BOLD series in T1w space."""
        return self.func_dir / self.preproc_t1_name

    @property
    def preproc_mni(self) -> Path:
        """Published preprocessed BOLD series in MNI space."""
        return self.func_dir / _with_suffix(
            f"{self.run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_bold.nii.gz"
        )

    @property
    def preproc_fsnative(self) -> dict[str, Path]:
        """Published fsnative surface series, by hemisphere."""
        return {
            "L": self.func_dir
            / _with_suffix(f"{self.run_base}_space-fsnative_hemi-L", "_desc-preproc_bold.func.gii"),
            "R": self.func_dir
            / _with_suffix(f"{self.run_base}_space-fsnative_hemi-R", "_desc-preproc_bold.func.gii"),
        }

    @property
    def epi_mean_t1(self) -> Path:
        """Mean preprocessed BOLD image in T1w space."""
        return self.qc_dir / _with_suffix(self.run_prefix, "_desc-preproc_mean.nii.gz")

    @property
    def epi_mean_mni(self) -> Path:
        """Mean preprocessed BOLD image in MNI space."""
        return self.qc_dir / _with_suffix(
            f"{self.run_base}_space-MNI152NLin2009cAsym", "_desc-preproc_mean.nii.gz"
        )

    @property
    def boldref_t1_out(self) -> Path:
        """Published BOLD reference in T1w space."""
        return self.func_dir / _with_suffix(self.run_prefix, "_boldref.nii.gz")

    @property
    def reg_prenonlinear_qc_out(self) -> Path:
        """Published BOLD reference before nonlinear registration, for QC."""
        return self.func_dir / _with_suffix(self.run_prefix, "_desc-preNonlinearReg_boldref.nii.gz")

    @property
    def reg_base_qc_out(self) -> Path:
        """Published BOLD reference after base registration, for QC."""
        return self.func_dir / _with_suffix(self.run_prefix, "_desc-baseReg_boldref.nii.gz")

    @property
    def reg_refine_qc_out(self) -> Path:
        """Published BOLD reference after anatomical refinement, for QC."""
        return self.func_dir / _with_suffix(self.run_prefix, "_desc-refineReg_boldref.nii.gz")

    @property
    def reg_mni_qc_out(self) -> Path:
        """Published BOLD reference in MNI space, for QC."""
        return self.func_dir / _with_suffix(
            f"{self.run_base}_space-MNI152NLin2009cAsym", "_boldref.nii.gz"
        )

    @property
    def melodic_ic_t1_out(self) -> Path:
        """Published MELODIC component maps in T1w space."""
        return self.func_dir / _with_suffix(self.run_prefix, "_desc-melodicIC_bold.nii.gz")

    @property
    def fieldmap_hz_in_t1_out(self) -> Path:
        """Published fieldmap in Hz, resampled to T1w space."""
        return self.fmap_dir / _with_suffix(self.run_prefix, "_desc-fieldmapHz_fieldmap.nii.gz")

    @property
    def fmap_field_hz_out(self) -> Path:
        """Published fieldmap in Hz in functional-reference space."""
        return self.fmap_dir / _with_suffix(
            f"{self.run_base}_space-{REG_REF_SPACE}", "_desc-fieldmapHz_fieldmap.nii.gz"
        )

    @property
    def fmap_sdc_warp_out(self) -> Path:
        """Published distortion-correction warp of the reference."""
        return self.fmap_dir / _with_suffix(
            f"{self.run_base}_space-{REG_REF_SPACE}", "_desc-sdcwarp_fieldmap.nii.gz"
        )

    @property
    def fmap_bold_sdc_warp_out(self) -> Path:
        """Published distortion-correction warp of the BOLD series."""
        return self.fmap_dir / _with_suffix(
            f"{self.run_base}_space-{REG_REF_SPACE}", "_desc-boldSDCwarp_fieldmap.nii.gz"
        )

    @property
    def fmap_jacobian_out(self) -> Path:
        """Published Jacobian of the distortion-correction warp."""
        return self.fmap_dir / _with_suffix(
            f"{self.run_base}_space-{REG_REF_SPACE}", "_desc-jacobian_fieldmap.nii.gz"
        )

    @property
    def fmap_topup_coeff_out(self) -> Path:
        """Published TOPUP field coefficients."""
        return self.fmap_dir / _with_suffix(self.run_base, "_desc-topupcoeff_fieldmap.nii.gz")

    @property
    def fmap_synbold_ref_out(self) -> Path:
        """Published SynBOLD-DisCo synthetic reference."""
        return self.fmap_dir / _with_suffix(
            f"{self.run_base}_space-{REG_REF_SPACE}", "_desc-synbold_boldref.nii.gz"
        )

    @property
    def fmap_synbold_rigid_out(self) -> Path:
        """Published rigid SynBOLD-DisCo reference in T1w space."""
        return self.fmap_dir / _with_suffix(
            f"{self.run_base}_space-T1w", "_desc-synboldRigid_boldref.nii.gz"
        )

    @property
    def anat_brain_mask_in_t1(self) -> Path:
        """Published anatomical brain mask on the functional T1w grid."""
        return self.func_dir / _with_suffix(self.run_prefix, "_desc-brain_mask.nii.gz")

    @property
    def anat_brain_mask_in_mni(self) -> Path:
        """Anatomical brain mask in MNI space."""
        return self.qc_dir / _with_suffix(
            f"{self.run_base}_space-MNI152NLin2009cAsym", "_desc-brain_mask.nii.gz"
        )

    @property
    def confounds_tsv(self) -> Path:
        """Published confound time series."""
        return self.func_dir / _with_suffix(self.run_stem, "_desc-confounds_timeseries.tsv")

    @property
    def confounds_json(self) -> Path:
        """Published confound descriptions."""
        return self.func_dir / _with_suffix(self.run_stem, "_desc-confounds_timeseries.json")

    @property
    def t1_to_mni_itk_warp(self) -> Path:
        """T1w-to-MNI warp in ITK format."""
        return self.reg_dir / f"{self.run_stem}_T1ToMNI_itk_warp.nii.gz"

    @property
    def t1_to_mni_fnirt_warp(self) -> Path:
        """T1w-to-MNI warp in FNIRT format."""
        return self.reg_dir / f"{self.run_stem}_T1ToMNI_fnirt_warp.nii.gz"

    @property
    def warp_epi2mni(self) -> Path:
        """Combined BOLD-to-MNI warp."""
        return self.reg_dir / f"{self.run_stem}_EPIToMNI_warp.nii.gz"

    @property
    def warp_regref2mni(self) -> Path:
        """Combined functional-reference-to-MNI warp."""
        return self.reg_dir / f"{self.run_stem}_RegRefToMNI_warp.nii.gz"

    @property
    def mni_ref(self) -> Path:
        """MNI target resampled to 2 mm."""
        return self.reg_dir / f"{self.run_stem}_mni_2mm.nii.gz"

    @property
    def bbr_t1_ref(self) -> Path:
        """FreeSurfer T1 reference for boundary-based registration, on the output grid."""
        return (
            self.reg_dir / f"{self.run_stem}_fs_t1_grid_epi_vox.nii.gz"
            if self.opts.output_grid == "t1_epi_vox"
            else self.reg_dir / f"{self.run_stem}_fs_t1_native.nii.gz"
        )

    @property
    def epi_mean_to_t1_fallback_mat(self) -> Path:
        """Affine from the mean BOLD image to T1w used by the SyN fallback."""
        return self.reg_dir / f"{self.run_stem}_epiMeanAffineToT1.mat"

    @property
    def reg_ref_in_t1_affine(self) -> Path:
        """Functional reference after the affine registration to T1w."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}_in_t1_affine.nii.gz"

    @property
    def fs_t1_to_t1w_mat(self) -> Path:
        """Affine from the FreeSurfer T1 to the T1w image."""
        return self.reg_dir / f"{self.run_stem}_fsT1ToT1w.mat"

    @property
    def reg_ref_to_t1w_mat(self) -> Path:
        """Affine from the functional reference to T1w."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}2t1w.mat"

    @property
    def epi_mean_nodc_t1_affine(self) -> Path:
        """Mean BOLD image without distortion correction after the affine registration."""
        return self.reg_dir / f"{self.run_stem}_epi_mean_nodc_t1_affine.nii.gz"

    @property
    def reg_ref_in_t1_linear(self) -> Path:
        """Functional reference after linear registration to T1w."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}_in_t1_linear.nii.gz"

    @property
    def reg_ref_dc_ref_n4(self) -> Path:
        """Bias-corrected, distortion-corrected functional reference."""
        return self.sdc_dir / f"{REG_REF_TAG}_dc_ref_n4.nii.gz"

    @property
    def reg_ref_in_t1_base_n4(self) -> Path:
        """Bias-corrected functional reference after base registration."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}_in_t1_base_n4.nii.gz"

    @property
    def reg_ref_in_t1_refine_n4(self) -> Path:
        """Bias-corrected functional reference used for refinement."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}_in_t1_refine_n4.nii.gz"

    @property
    def epi_to_t1_warp_planned(self) -> Path:
        """Planned BOLD-to-T1w warp."""
        return self.reg_dir / f"{self.run_stem}_EPIToT1w_warp.nii.gz"

    @property
    def fallback_affine_warp_planned(self) -> Path:
        """Planned affine warp of the SyN fallback."""
        return self.reg_dir / f"{self.run_stem}_SyNAffine_warp.nii.gz"

    @property
    def fallback_base_itk_planned(self) -> Path:
        """Planned SyN fallback warp in ITK format."""
        return self.reg_dir / f"{self.run_stem}_SyNBoldAff_itk_warp.nii.gz"

    @property
    def fallback_base_fnirt_planned(self) -> Path:
        """Planned SyN fallback warp in FNIRT format."""
        return self.reg_dir / f"{self.run_stem}_SyNBoldAff_fnirt_warp.nii.gz"

    @property
    def fallback_base_warp_planned(self) -> Path:
        """Planned base warp of the SyN fallback."""
        return self.reg_dir / f"{self.run_stem}_SyNBoldAffBase_warp.nii.gz"

    @property
    def warp_regref2t1_planned(self) -> Path:
        """Planned base functional-reference-to-T1w warp."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}ToT1wBase_warp.nii.gz"

    @property
    def syn_refine_fnirt_planned(self) -> Path:
        """Planned SyN refinement warp in FNIRT format."""
        return self.reg_dir / f"{self.run_stem}_SyNBoldAffRefine_fnirt_warp.nii.gz"

    @property
    def warp_regref2t1_refined_planned(self) -> Path:
        """Planned refined functional-reference-to-T1w warp."""
        return self.reg_dir / f"{self.run_stem}_{REG_REF_TAG}ToT1wRefined_warp.nii.gz"

    @property
    def field_hz_regref(self) -> Path:
        """Fieldmap in Hz in functional-reference space, in work storage."""
        return self.sdc_dir / f"{self.run_stem}_fieldmap_Hz_{REG_REF_TAG}Space.nii.gz"

    @property
    def mc_mat_dir(self) -> Path:
        """Directory of per-volume motion-correction matrices."""
        return self.mc_dir / f"{self.run_stem}_mc.nii.gz.mat"

    @property
    def syn_fixed_mask(self) -> Path:
        """T1w brain mask used as the fixed mask for ANTs SyN."""
        return self.reg_dir / f"{self.run_stem}_t1_brain_mask.nii.gz"

    @property
    def afni_motion_affines(self) -> Path:
        """Per-volume motion affines in AFNI format."""
        return self.mc_dir / f"{self.run_stem}_mc_pull.aff12.1D"
