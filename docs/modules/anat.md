# Anatomical preprocessing

`anat` prepares a subject's anatomical references, masks, surfaces, and spatial
transforms. It supplies geometry to functional processing and subsequent
analyses.

## Inputs and processing choices

Discovery collects T1w and T2w acquisitions across sessions. Missing anatomy
makes the subject unavailable to planning rather than preventing registry use.
Markup can select either modality independently. Otherwise,
`selection_strategy: first` selects the earliest ordered image for each modality;
the averaging strategy uses FreeSurfer `mri_robust_template` when several images
are available. A missing modality is skipped. Single images are copied.
Acquisition metadata and filename ordering resolve selection reproducibly.

T1w and T2w availability determines the graph. Session outputs retain their
acquisition geometry. The module selects participant T1w and T2w references
independently, even when their sources come from different sessions. It rigidly
aligns the selected T1w reference to the pose of the configured MNI template and
resamples it once onto a source-resolution `ACPC` grid. The transform has six
degrees of freedom, so it changes head position without scaling or deforming the
brain. When both modalities exist, the module registers the selected T2w
reference directly to the ACPC T1w reference. A T1w/T2w myelin proxy exists only
when both modalities are available.
This ratio is not a quantitative myelin measurement. Review the manifest's
selected sources before comparing subjects with different acquisition schemes.

## Processing sequence

1. Resolve the site's gradient-unwarping policy from inherited BIDS metadata.
   A matching `unwarp` profile runs HCP gradient correction unless the metadata
   already reports `NonlinearGradientCorrection: true`. Unmatched acquisitions
   pass through unchanged. Apply ANTs N4 bias correction. Then produce
   brain-extracted session copies and masks with SynthStrip without changing
   their native grids.
2. Select or combine T1w and T2w acquisitions independently. Estimate a rigid
   T1w-to-ACPC transform with ANTs mutual-information registration. Construct a
   deterministic template-oriented grid at the selected source resolution,
   verify that it covers the transformed anatomy, and resample the T1w once.
   Publish both transform directions and numerical pose checks. If no T1w exists,
   use the selected T2w as the pose source.
3. If both modalities exist, align the selected T2w reference directly to the
   ACPC T1w reference with six-degree-of-freedom FSL FLIRT. Save the forward and
   inverse transforms and the optional T1w/T2w ratio.
4. Run FreeSurfer `recon-all` from the ACPC reference with a validated directory completion boundary.
   Export anatomical volumes, cortical ribbon, subcortical masks, and the gray
   matter mask from FreeSurfer segmentation labels. The label names and numeric
   values are in `nro.modules.anat.constants`; they are not learned tissue probabilities.
5. Convert FreeSurfer geometry to GIFTI and construct white, pial, inflated,
   and midthickness surfaces. Export sphere registrations and surface metrics.
   FSL/FreeSurfer coordinate transforms and Workbench surface operations place
   geometry in the requested native and template coordinate systems. The
   packaged configuration uses the pinned 41k-vertex-per-hemisphere
   `fsaverage6` geometry from the local TemplateFlow store.
6. Register the ACPC anatomy to the configured MNI reference with ANTs SyN.
   Publish forward and inverse composite transforms and registration-check images.
   The fixed schedule is rigid and affine MI (32 bins, regular 25% sampling),
   then SyN with radius-4 cross-correlation. Linear stages use
   `1000x500x250x0` iterations; SyN uses `100x70x50x20`. Both use shrink factors
   `8x4x2x1` and smoothing `3x2x1x0vox`. Registration uses brain masks,
   histogram matching, 0.5–99.5% winsorization, and Lanczos-windowed sinc
   interpolation. These schedules are implementation constants, not YAML keys.
7. Validate and publish the anatomical manifest. Session copies and subject-level
   results have separate paths; the subject manifest identifies the complete
   public result set, including the FreeSurfer directory.

The default container is QuNex, with a separate SynthStrip image. Host and
container paths are translated through `Runner`; the configured FreeSurfer
license is bound into processing. Linked scenes refer to these surfaces directly;
published scenes copy them only when requested.

## Public artifacts

Outputs live under
`derivatives/nro/anat/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`, with session
acquisitions under `ses-ID/anat` where applicable. Subject `anat` contains
ACPC-aligned anatomical references, brain and gray-matter masks, cortical
ribbon and subcortical masks, surfaces, metrics, and transforms. The publication
manifest records exact paths rather than requiring downstream filename guesses.
It also records the pose transforms, grid and registration checks, and the
separate ACPC-to-MNI transforms.

Required metadata includes `inputs`, `selection_strategy`, `outputs`,
`freesurfer_subjects_dir`, `mni_template`, configuration provenance, and
`complete`. Optional products have nullable paths. The full typed schema is
[`ANATOMICAL_MANIFEST_FIELDS`](../autoapi/nro/modules/anat/contract/index.rst).

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup); `null` ignores markup.
`gradient_unwarping` selects `auto` or `off`. In `auto` mode, only acquisitions
matched by the site's hardware catalog are eligible for correction. The catalog
and coefficient file are site resources, not module settings.
`fsaverage_template` selects either `fsaverage6`, the packaged default, or the
full-resolution `fsaverage` surface target. `selection_strategy` controls
acquisition combination. `mni_template`
selects the registration target; `synthstrip_container` selects brain extraction.
`freesurfer_subjects_dir` and `fs_subject` override FreeSurfer storage and identity.
Scientific tools use the worker's CPU allocation, which `nro run --cpus` sets.
`overwrite` requests re-execution;
`verbose` changes logging. `container` controls the runtime, image, binds, home,
environment isolation, and inner setup command for anatomy.

```{literalinclude} ../../nro/configuration/starters/configs/anat/main_anat.yml
:language: yaml
```

Implementation: [module construction](../autoapi/nro/modules/anat/module/index.rst),
[planning](../autoapi/nro/modules/anat/planning/index.rst).
See [software and methods sources](../methods/software.md).
