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
native geometry. The module builds participant T1w and T2w references
independently, even when their sources come from different sessions. When both
exist, it registers the selected participant T2w reference to the selected T1w
reference. A T1w/T2w myelin proxy exists only when both modalities are available.
This ratio is not a quantitative myelin measurement. Review the manifest's
selected sources before comparing subjects with different acquisition schemes.

## Processing sequence

1. Resolve the site's gradient-unwarping policy from inherited BIDS metadata.
   A matching `unwarp` profile runs HCP gradient correction unless the metadata
   already reports `NonlinearGradientCorrection: true`. Unmatched acquisitions
   pass through unchanged. Apply ANTs N4 bias correction. Then produce
   brain-extracted session copies and masks with SynthStrip without changing
   their native grids.
2. Select or combine T1w and T2w acquisitions independently into participant
   references. If both exist, align the participant T2w reference to the T1w
   reference with six-degree-of-freedom FSL FLIRT. Save source lists, selection
   metadata, the transform, and the optional T1w/T2w ratio.
3. Run FreeSurfer `recon-all` with a validated directory completion boundary.
   Export anatomical volumes, cortical ribbon, subcortical masks, and the gray
   matter mask from FreeSurfer segmentation labels. The label names and numeric
   values are in `nro.modules.anat.constants`; they are not learned tissue probabilities.
4. Convert FreeSurfer geometry to GIFTI and construct white, pial, inflated,
   and midthickness surfaces. Export sphere registrations and surface metrics.
   FSL/FreeSurfer coordinate transforms and Workbench surface operations place
   geometry in the requested native and template coordinate systems. The
   packaged configuration uses the pinned 41k-vertex-per-hemisphere
   `fsaverage6` geometry from the local TemplateFlow store.
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
license is bound into processing. Linked scenes refer to these surfaces directly;
published scenes copy them only when requested.

## Public artifacts

Outputs live under `derivatives/nro/anat/ANAT_ID/sub-ID/`, with session
acquisitions under `ses-ID/anat` where applicable. Subject `anat` contains
preprocessed anatomical references, brain and gray-matter masks, cortical
ribbon and subcortical masks, surfaces, metrics, and transforms. The publication
manifest records exact paths rather than requiring downstream filename guesses.

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
