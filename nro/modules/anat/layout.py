"""Name the planned files and directories of one anatomical graph.

Each path depends only on the anatomical options and the participant, so the
layout can be computed before any step is added. Paths that depend on
choices made while the graph is built stay with the stages that make them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnatLayout:
    """Fixed paths used across the stages of one anatomical graph."""

    sub_id: str
    fs_subject: str
    out_dir: Path
    work_dir: Path
    freesurfer_subjects_dir: Path

    @classmethod
    def for_options(cls, opts, sub_id: str) -> AnatLayout:
        """Return the layout for routed anatomical options and one participant."""
        return cls(
            sub_id=sub_id,
            fs_subject=opts.fs_subject,
            out_dir=opts.out_dir,
            work_dir=opts.work_dir,
            freesurfer_subjects_dir=opts.freesurfer_subjects_dir,
        )

    @property
    def initialized(self) -> Path:
        """Breadcrumb written once output directories exist."""
        return self.work_dir / "initialized.complete"

    @property
    def configuration_snapshot(self) -> Path:
        """Recorded anatomical configuration."""
        return self.work_dir / "configuration.json"

    @property
    def dependency_check(self) -> Path:
        """Breadcrumb written once required commands are available."""
        return self.work_dir / "dependencies.complete"

    @property
    def reference_work(self) -> Path:
        """Work directory for the subject reference template."""
        return self.work_dir / "subject_reference"

    @property
    def pose_work(self) -> Path:
        """Work directory for rigid pose normalization."""
        return self.work_dir / "pose_normalization"

    @property
    def subject_dir(self) -> Path:
        """FreeSurfer subject directory."""
        return self.freesurfer_subjects_dir / self.fs_subject

    @property
    def aseg_mgz(self) -> Path:
        """FreeSurfer subcortical segmentation."""
        return self.subject_dir / "mri" / "aseg.mgz"

    @property
    def brain_mask(self) -> Path:
        """Published brain mask."""
        return self.out_dir / f"{self.sub_id}_desc-brain_mask.nii.gz"

    @property
    def gray_mask(self) -> Path:
        """Published gray-matter mask."""
        return self.out_dir / f"{self.sub_id}_desc-grayMatter_mask.nii.gz"

    @property
    def ribbon_mask(self) -> Path:
        """Published cortical ribbon mask."""
        return self.out_dir / f"{self.sub_id}_desc-ribbon_mask.nii.gz"

    @property
    def core_work(self) -> Path:
        """Work directory for masks derived from FreeSurfer volumes."""
        return self.work_dir / "core_masks"

    @property
    def brain_temporary(self) -> Path:
        """Brain mask resampled from FreeSurfer before publication."""
        return self.core_work / f"{self.sub_id}_desc-brain_mask_fs.nii.gz"

    @property
    def gray_temporary(self) -> Path:
        """Gray-matter mask resampled from FreeSurfer before publication."""
        return self.core_work / f"{self.sub_id}_desc-grayMatter_mask_fs.nii.gz"

    @property
    def ribbon_temporary(self) -> Path:
        """Ribbon mask resampled from FreeSurfer before publication."""
        return self.core_work / f"{self.sub_id}_desc-ribbon_mask_fs.nii.gz"

    @property
    def brainmask_mgz(self) -> Path:
        """FreeSurfer brain mask volume."""
        return self.subject_dir / "mri" / "brainmask.mgz"

    @property
    def ribbon_mgz(self) -> Path:
        """FreeSurfer ribbon volume."""
        return self.subject_dir / "mri" / "ribbon.mgz"

    @property
    def fsnative_ref(self) -> Path:
        """FreeSurfer T1 volume that defines fsnative space."""
        return self.freesurfer_subjects_dir / self.fs_subject / "mri" / "T1.mgz"

    @property
    def t1_to_fsnative(self) -> Path:
        """Published T1w-to-fsnative transform."""
        return self.out_dir / f"{self.fs_subject}_from-T1w_to-fsnative_mode-image_xfm.txt"

    @property
    def fsnative_to_t1(self) -> Path:
        """Published fsnative-to-T1w transform."""
        return self.out_dir / f"{self.fs_subject}_from-fsnative_to-T1w_mode-image_xfm.txt"

    @property
    def t1_to_fsnative_reg(self) -> Path:
        """Published T1w-to-fsnative registration in FreeSurfer format."""
        return self.out_dir / f"{self.fs_subject}_from-T1w_to-fsnative_mode-image_xfm.dat"

    @property
    def surface_work_dir(self) -> Path:
        """Work directory for surface export."""
        return self.work_dir / "surface_export"

    @property
    def surface_dir(self) -> Path:
        """FreeSurfer surface directory."""
        return self.freesurfer_subjects_dir / self.fs_subject / "surf"

    @property
    def fs_t1_ref_nii(self) -> Path:
        """FreeSurfer T1 reference converted to NIfTI for surface export."""
        return self.surface_work_dir / f"{self.fs_subject}_fs_t1_ref.nii.gz"

    @property
    def mni_prefix(self) -> Path:
        """Output prefix for the ANTs T1w-to-MNI registration."""
        return self.work_dir / "ants_mni_" / f"{self.fs_subject}_"

    @property
    def t1_to_mni(self) -> Path:
        """Published T1w-to-MNI transform."""
        return self.out_dir / f"{self.fs_subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

    @property
    def mni_to_t1(self) -> Path:
        """Published MNI-to-T1w transform."""
        return self.out_dir / f"{self.fs_subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"

    @property
    def produced_t1_to_mni(self) -> Path:
        """Composite T1w-to-MNI transform written by ANTs."""
        return self.mni_prefix.parent / f"{self.mni_prefix.name}Composite.h5"

    @property
    def produced_mni_to_t1(self) -> Path:
        """Inverse composite transform written by ANTs."""
        return self.mni_prefix.parent / f"{self.mni_prefix.name}InverseComposite.h5"

    @property
    def mni_in_t1(self) -> Path:
        """MNI template resampled into T1w space for quality control."""
        return self.out_dir / f"{self.fs_subject}_space-T1w_desc-mniToT1wQC.nii.gz"

    @property
    def t1_in_mni(self) -> Path:
        """T1w image resampled into MNI space for quality control."""
        return self.out_dir / f"{self.fs_subject}_space-MNI152NLin2009cAsym_desc-t1ToMNIQC.nii.gz"
