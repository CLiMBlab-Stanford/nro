# Anatomical preprocessing

`anat` prepares a subject's anatomical references, masks, surfaces, and spatial
transforms. It belongs to the `preprocessing` derivative class and supplies
geometry to functional processing and subsequent analyses.

## Inputs and branch selection

Discovery collects T1w and T2w acquisitions across sessions. Missing anatomy
makes the subject unavailable to planning rather than preventing registry use.
`selection_strategy: first` selects the earliest ordered image for each modality;
the averaging strategy uses FreeSurfer `mri_robust_template` when several images
are available. Single images are copied. Acquisition metadata and filename
ordering resolve selection reproducibly.

T1w and T2w availability determines the graph. T2w images can be registered to
T1w; a T1w/T2w myelin proxy exists only when both modalities are available.
This ratio is not a quantitative myelin measurement. Review the manifest's
selected sources before comparing subjects with different acquisition schemes.

## Processing sequence

1. Stage acquisitions and apply ANTs N4 bias correction. Produce brain-extracted
   session copies and masks using SynthStrip. When appropriate, align T2w to
   the corresponding T1w using FSL FLIRT.
2. Select or combine the processed acquisitions into subject references. Save
   source lists, selection metadata, and the optional T1w/T2w ratio.
3. Run FreeSurfer `recon-all` with a validated directory completion boundary.
   Export anatomical volumes, cortical ribbon, subcortical masks, and the gray
   matter mask from FreeSurfer segmentation labels. The label names and numeric
   values are in `nro.engine.freesurfer`; they are not learned tissue probabilities.
4. Convert FreeSurfer geometry to GIFTI and construct white, pial, inflated,
   and midthickness surfaces. Export sphere registrations and surface metrics.
   FSL/FreeSurfer coordinate transforms and Workbench surface operations place
   geometry in the requested native and template coordinate systems.
5. Register the subject anatomy to the configured MNI reference with ANTs SyN.
   Publish forward and inverse composite transforms and registration-check images.
   The fixed schedule is rigid and affine MI (32 bins, regular 25% sampling),
   then SyN with radius-4 cross-correlation. Linear stages use
   `1000x500x250x0` iterations; SyN uses `100x70x50x20`. Both use shrink factors
   `8x4x2x1` and smoothing `3x2x1x0vox`. Registration uses brain masks,
   histogram matching, 0.5–99.5% winsorization, and Lanczos-windowed sinc
   interpolation. These schedules are implementation constants, not YAML keys.
6. Validate and publish the anatomical manifest. Session copies and subject-level
   results have separate paths; the subject manifest identifies the complete
   public result set, including the FreeSurfer directory.

The default container is QuNex, with a separate SynthStrip image. Host and
container paths are translated through `Runner`; the configured FreeSurfer
license is bound into processing. No anatomy is duplicated into network scenes.

## Public artifacts

Outputs live under `derivatives/preprocessing/LINEAGE/sub-ID/`, with session
acquisitions under `ses-ID/anat` where applicable. Subject `anat` contains
preprocessed anatomical references, brain and gray-matter masks, cortical
ribbon and subcortical masks, surfaces, metrics, and transforms. The publication
manifest records exact paths rather than requiring downstream filename guesses.

Required metadata includes `inputs`, `selection_strategy`, `outputs`,
`freesurfer_subjects_dir`, `mni_template`, configuration provenance, and
`complete`. Optional products have nullable paths. The full typed schema is
[`ANATOMICAL_MANIFEST_FIELDS`](../autoapi/nro/modules/anat/contract/index.rst).

## Configuration

`anat.selection_strategy` controls acquisition combination. `mni_template`
selects the registration target; `synthstrip_container` selects brain extraction.
`freesurfer_subjects_dir` and `fs_subject` override FreeSurfer storage and identity.
`nthreads`, `nthreads_divisor`, and `nthreads_min` determine tool thread allocation;
Slurm CPUs are a separate worker resource. `force` requests re-execution;
`verbose` changes logging. `container` controls the runtime, image, binds, home,
environment isolation, and inner setup command for preprocessing.

```{literalinclude} ../../nro/configuration/starters/configs/preprocessing/main_preprocessing.yml
:language: yaml
:start-at: container:
:end-before: func:
```

Implementation: [module construction](../autoapi/nro/modules/anat/module/index.rst),
[planning](../autoapi/nro/modules/anat/planning/index.rst).
See [software and methods sources](../methods/software.md).
