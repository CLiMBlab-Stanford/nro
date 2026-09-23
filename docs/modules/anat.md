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
   pass through unchanged. Source BIDS anatomicals are already skull-stripped
   to prevent identifiable facial anatomy from entering the dataset. SynthStrip
   estimates a brain mask on each source image. ANTs N4 uses that mask to fit
   the bias field, retains the field as a private intermediate, and corrects the
   source image. Applying the same mask then standardizes the brain boundary
   without changing its grid.
2. Select or combine T1w and T2w acquisitions independently. Estimate a rigid
   T1w-to-ACPC transform with ANTs mutual-information registration. Construct a
   deterministic template-oriented grid around the transformed anatomical mask
   at the selected source resolution. Include a 5 mm margin, verify mask
   coverage, and resample the T1w once with cubic B-spline interpolation.
   Resample the mask separately with nearest-neighbor interpolation and reapply
   it before publication. ANTs affine files encode the
   fixed-to-moving map used for resampling, so grid construction inverts that
   map when projecting source-mask points into ACPC space.
   Publish both transform directions and numerical pose checks. If no T1w exists,
   use the selected T2w as the pose source.
3. If both modalities exist, align the selected T2w reference directly to the
   ACPC T1w reference with six-degree-of-freedom FSL FLIRT. Save the forward and
   inverse transforms and the optional T1w/T2w ratio.
4. Run FreeSurfer 7.4.1 from its pinned official image. `autorecon1` receives
   the ACPC reference with `-noskullstrip`; nro resamples the binary SynthStrip
   mask to FreeSurfer's conformed grid with nearest-neighbor interpolation and
   applies it to FreeSurfer's normalized `T1.mgz`. The result becomes
   `brainmask.auto.mgz` and `brainmask.mgz` before `autorecon2` and `autorecon3`.
   The reconstruction has a validated directory completion boundary.
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
   histogram matching, and 0.5–99.5% winsorization. Registration-check images use
   cubic B-spline interpolation. These schedules are implementation constants,
   not YAML keys.
7. Validate and publish the anatomical manifest. Session copies and subject-level
   results have separate paths; the subject manifest identifies the complete
   public result set, including the FreeSurfer directory.

For a participant marked `lesion: true`, steps 4 and 5 use a separate fixed
graph. nro's SynthStroke adapter estimates a stroke-lesion mask on the
selected, bias-corrected ACPC T1w image. A separately extracted brain mask
supports pose registration. FastSurfer receives an otherwise matched source
image before N4 bias correction, as required by its input contract.
Mechanical checks reject an empty, nonfinite,
misregistered, or implausibly large mask and report overlap with the nonzero
anatomical support for review. FastSurfer-LIT inpaints the mask, runs
FastSurferVINN and cortical reconstruction, and records its lesion-impact
summary. Cerebellar, hypothalamic, and corpus-callosum submodules are skipped
because they do not contribute to nro's cortical scaffold. nro then
projects the mask through the white-to-pial ribbon, removes every triangle that
touches the lesion, removes unused vertices, and applies the same compact
vertex mapping to every published surface, sphere, and metric. The complete
scaffold is not an anatomical observation. Its FreeSurfer directory remains
implementation support for closed-surface operations such as `bbregister`, but
scenes and surface-based analyses use the cut public meshes.

Most anatomical tools use the default QuNex container. SynthStrip and FreeSurfer
use separate pinned official images. Lesion-aware inpainting and reconstruction
also use the separately pinned FastSurfer 2.5.4 image. Host and
container paths are translated through `Runner`, and the configured FreeSurfer
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

Lesion-aware artifacts additionally contain the ACPC-grid mask and
probability image, an explicitly labeled synthetic inpainted T1w alternative,
a three-plane mask-overlay image, and one public-to-scaffold vertex table per
hemisphere. Each hemisphere also has a validity summary with its scaffold and
public vertex and face counts. The mask metadata records connected-component
sizes, lesion volume, laterality, model hashes, and overlap with nonzero
anatomical support. The FastSurfer-LIT lesion-impact summary is copied into the
public artifact. The ordinary ACPC T1w remains the primary anatomy. Published
surfaces contain surviving cortex only. Automatic masks and reconstructed
boundaries require visual review.

The functional module reads this public anatomical-domain contract. For
lesion-aware anatomy, registration tools that assume an intact brain use the
synthetic inpainted T1w and complete FreeSurfer scaffold. Functional sampling
and publication still use the observed anatomical grid and cut public surfaces.
When native functional data are resampled to fsaverage, Workbench also emits a
valid-output ROI from the cut native registration sphere. nro masks the
resampled metric with that ROI so interpolation cannot restore values over the
excised lesion.

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
`freesurfer_container` selects the image that supplies conventional FreeSurfer
7.4.1. It is an execution path; the pinned version and reconstruction policy are
part of the scientific artifact contract.
`freesurfer_subjects_dir` and `fs_subject` override FreeSurfer storage and identity.
The initial lesion method pins the SynthStroke model, source and model revisions,
model hashes, 1 mm inference grid, sliding-window settings, probability
threshold, test-time augmentation, FastSurfer version, 1 mm FastSurfer
reconstruction grid, and surface-boundary policy in code. The surface grid
matches conventional FreeSurfer conformation; lesion detection and the public
lesion mask retain the source anatomical grid. Install the optional Python stack
with `./install --with-lesion`.
When `masker_command` is `null`, anatomy runs its built-in adapter with the
installed Python environment and the checksum-verified model in
`synthstroke_data`. The worker never contacts Hugging Face. An override must
implement the same adapter interface:
`--input`, `--probability`, `--mask`, `--model`, `--revision`, `--threshold`,
`--device`, and optional `--tta`. `fastsurfer_image` points to a FastSurfer 2.5.4 image with
NeuroLIT 0.6.1 support. The image, its source revision, and all three inpainting
checkpoint hashes are pinned. Site setup stores the checkpoints under
`fastsurfer_data` and workers mount them read-only. The SynthStroke model is
stored separately under `synthstroke_data`. The override, image path,
and `use_gpu` are the lesion block's only configuration fields and are execution
settings. `use_gpu` controls both SynthStroke and FastSurfer-LIT execution.
Containerized NeuroLIT preserves Slurm's assigned CUDA device and uses its
version-pinned default batch size of eight slices. A lesion-marked participant
with `use_gpu: true` is assigned to a dedicated Slurm worker that requests one
GPU. That worker cannot claim ordinary work and exits when no GPU work is ready.
If `use_gpu` is false, the lesion-aware graph runs on a general worker with CPU
inference. Scheduling choices remain outside the scientific artifact contract.
The pinned scientific policy
enters the work-item contract only for lesion-marked participants, so adding the
feature does not rename or stale the ordinary anatomical lineage.
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
