# Anatomical preprocessing

`anat` builds a participant's anatomical references, brain masks, cortical
surfaces, and spatial transforms. Functional processing and every later
analysis take their geometry from these outputs. Read this page to see how nro
chooses anatomical images, what the default pipeline does, and how to enable
the optional lesion-aware and MSMAll routes.

## Inputs and image selection

Discovery collects T1w and T2w acquisitions from all of a participant's
sessions. A participant with no anatomy is unavailable to planning, but the
registry still accepts the participant.

nro chooses one T1w reference and one T2w reference independently, even when
they come from different sessions. Source markup can select either modality
directly. Without markup, `selection_strategy` decides:

* `first` takes the earliest image of each modality. Acquisition metadata and
  filename order make this choice reproducible.
* `robust_average` combines several images with FreeSurfer
  `mri_robust_template`. A single image is copied.

A missing modality is skipped. Session outputs keep their acquisition geometry.

When no usable T1w exists, `t1w_fallback` controls what happens. The default,
`synthesize_from_t2w`, uses FreeSurfer SynthSR to create a 1 mm T1-like
reference from the selected T2w image. Its sidecar and the publication manifest
mark the contrast as synthetic and name the source images. `skip` makes the
participant unavailable instead. Synthetic T1w anatomy cannot be used for
MSMAll calibration, lesion-aware processing, or the T1w/T2w myelin proxy.

The T1w/T2w ratio is produced only when both modalities exist. It is a proxy
and not a quantitative myelin measurement. Check the manifest's selected
sources before you compare participants whose acquisitions differ.

## Processing

The selected T1w image is rigidly aligned to the pose of the configured MNI
template. This six-degree-of-freedom transform changes head position without
scaling or deforming the brain, and the image is resampled only once. The steps
are:

1. **Correct each source image.** nro resolves the site's gradient-unwarping
   policy from inherited BIDS metadata. A matching `unwarp` profile runs HCP
   gradient correction unless the metadata already reports
   `NonlinearGradientCorrection: true`. Unmatched acquisitions pass through
   unchanged. Source BIDS anatomicals are already skull-stripped so that
   identifiable facial anatomy never enters the dataset. SynthStrip then
   estimates a brain mask on each image. ANTs N4 uses that mask to fit the
   bias field, keeps the field as a private intermediate, and corrects the
   image. Applying the same mask afterward standardizes the brain boundary
   without changing the grid.
2. **Build the participant T1w reference.** nro selects or combines the T1w
   and T2w acquisitions. If no acquired T1w exists, it first synthesizes one
   from the T2w. ANTs mutual-information registration estimates a rigid
   transform from the selected image into the participant reference. nro builds
   a deterministic, template-oriented grid around the transformed brain mask at
   the source resolution with a 5 mm margin and checks that the grid covers the
   mask. It resamples the T1w once with cubic B-spline interpolation, resamples
   the mask separately with nearest-neighbor interpolation, and reapplies the
   mask before publication. ANTs affine files store the fixed-to-moving map used
   for resampling, so grid construction inverts that map to project mask points
   into the reference. Both transform directions and numerical pose checks are
   published.
3. **Align the T2w.** When both modalities exist, FSL FLIRT aligns the T2w
   reference directly to the T1w reference with six degrees of freedom. The
   forward and inverse transforms and the optional T1w/T2w ratio are saved.
4. **Reconstruct cortical surfaces.** Both engines receive the acquired or
   synthesized T1w reference.
   * The default engine is FreeSurfer 7.4.1, run from its pinned official
     image. `autorecon1` receives the reference with `-noskullstrip`. nro
     resamples the binary SynthStrip mask to FreeSurfer's conformed grid with
     nearest-neighbor interpolation and applies it to FreeSurfer's normalized
     `T1.mgz`. The result becomes `brainmask.auto.mgz` and `brainmask.mgz`
     before `autorecon2` and `autorecon3`. Reconstruction uses seed 1234. If
     FreeSurfer fails during topology correction, nro discards the incomplete
     directory and retries once from the beginning with seed 5678. Other
     failures are not retried. The seed that succeeded is recorded in the
     FreeSurfer subject's `scripts` directory.
   * The alternative engine is FastSurfer 2.5.4. FastSurferVINN segments at
     1 mm on a GPU and stores a complete private archive in work storage. The
     work item then returns to a CPU worker for surface reconstruction, which
     publishes a validated directory.

   nro exports anatomical volumes, the cortical ribbon, subcortical masks, and
   the gray-matter mask from FreeSurfer segmentation labels. These masks come
   from fixed label names and values in `nro.modules.anat.constants`, not from
   learned tissue probabilities.
5. **Build surfaces in each space.** nro converts FreeSurfer geometry to GIFTI;
   builds white, pial, inflated, and midthickness surfaces; and exports sphere
   registrations and surface metrics. FSL and FreeSurfer coordinate transforms
   and Workbench surface operations place the geometry in the requested native
   and template coordinate systems. The packaged configuration uses the pinned
   `fsaverage6` geometry, with 41k vertices per hemisphere, from the local
   TemplateFlow store.
6. **Register to MNI.** ANTs SyN registers the T1w anatomy to the configured
   MNI reference. nro publishes forward and inverse composite transforms and
   registration-check images, which use cubic B-spline interpolation. The
   schedule is fixed in code, not in YAML:
   * rigid and affine stages with mutual information, 32 bins, and regular 25%
     sampling, using `1000x500x250x0` iterations;
   * a SyN stage with radius-4 cross-correlation, using `100x70x50x20`
     iterations;
   * shrink factors `8x4x2x1` and smoothing `3x2x1x0vox` for all stages;
   * brain masks, histogram matching, and 0.5–99.5% winsorization.
7. **Publish.** nro validates and publishes the anatomical manifest. Session
   copies and participant-level results have separate paths. The participant
   manifest lists the complete public result set, including the FreeSurfer
   directory.

## Lesion-aware anatomy

A participant marked `lesion: true` in source markup gets an intact synthetic
scaffold for surface reconstruction, and the published surfaces then have the
lesion cut out.

nro's SynthStroke adapter estimates a stroke-lesion mask on the selected,
bias-corrected T1w image. A separately extracted brain mask supports pose
registration. Mechanical checks reject a mask that is empty, nonfinite,
misregistered, or implausibly large, and report its overlap with nonzero
anatomical support for review.

FreeSurfer's `mri_synthsr` all-purpose model then synthesizes the whole brain
at 1 mm from the matched pose-normalized image before N4 correction. It does
not see the lesion mask. The configured `surface_reconstruction_engine`
reconstructs the scaffold from this synthetic T1w with the ordinary FreeSurfer
or FastSurfer runner. Reconstruction uses the whole-brain SynthStrip mask. The
SynthStroke mask only defines the lesion and is never used for skull stripping.

nro records the synthesis and reconstruction methods in the lesion-impact
summary. It then projects the mask through the white-to-pial ribbon, removes
every triangle that touches the lesion, and removes unused vertices. The same
compact vertex mapping is applied to every published surface, sphere, and
metric.

The scaffold is not an anatomical observation. Its public
FreeSurfer-compatible directory is labeled as support for closed-surface
operations such as `bbregister`. Scenes and surface-based analyses use the cut
public meshes.

`func` follows the same split. Registration tools that assume an intact brain
use the synthetic SynthSR T1w and the complete scaffold, while functional
sampling and publication use the observed anatomical grid and the cut surfaces.
When native functional data are resampled to fsaverage, Workbench also writes a
valid-output ROI from the cut native registration sphere. nro masks the
resampled metric with that ROI so interpolation cannot restore values over the
removed lesion.

Lesion-aware processing is mutually exclusive with MSMAll calibration.

## MSMAll calibration

When participant markup supplies a nonempty `msmall.rest` list, `anat` adds an
HCP Pipelines calibration branch after ordinary surface reconstruction. The
branch uses the marked runs and their opposite-phase-encoding fieldmaps to run
the HCP structural, minimal functional, multi-run FIX, MSMAll, and dedrifting
route. It is currently limited to FreeSurfer anatomy.

Because nro source anatomy is already skull-stripped, nonlinear atlas
registration uses explicit moving and reference masks and disables FNIRT's
whole-head intensity model. The moving mask is the binarized mask from the
completed surface reconstruction, so nonzero background in the T1w image does
not count as brain.

Each major HCP operation is a separate runner step. After preemption, the
branch resumes through the ordinary anatomical DAG and freshness rules. It runs
on a long CPU worker. The calibration time series are not published as
functional derivatives.

nro transfers the resulting registration onto its canonical `fsnative`
topology with Workbench spherical project and unproject operations. It then
publishes the forward registration sphere, the fsLR atlas sphere and surface
family, valid-domain metrics, software identities, and mechanical QC. The QC
rejects implausible atlas warps, nonfinite coordinates, and inconsistent atlas
vertex counts or topology.

## Software and containers

Most anatomical tools run in the default QuNex container. SynthStrip,
FreeSurfer, and FastSurfer use their own pinned official images. SynthSR runs
on a CPU from the pinned FreeSurfer image, whichever reconstruction engine is
selected. `Runner` translates host and container paths, and the configured
FreeSurfer license is bound into processing. Scientific tools use the worker's
CPU allocation, which `nro run --cpus` sets.

## Outputs

Outputs go under `derivatives/nro/anat/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/`,
with session acquisitions under `ses-ID/anat` where applicable. The
participant `anat` directory holds the pose-normalized T1w reference, the
aligned T2w reference, brain and gray-matter masks, cortical ribbon and
subcortical masks, surfaces, metrics, and transforms.

The publication manifest records exact paths, so downstream code does not
guess filenames. It also records the pose transforms, grid and registration
checks, and the separate T1w-to-MNI transforms. Linked scenes refer to these
surfaces directly, and published scenes copy them only when requested.

Required manifest metadata includes `inputs`, `selection_strategy`, `outputs`,
`freesurfer_subjects_dir`, `mni_template`, configuration provenance, and
`complete`. Optional products have nullable paths. The full typed schema is
[`ANATOMICAL_MANIFEST_FIELDS`](../autoapi/nro/modules/anat/contract/index.rst).

Lesion-aware results add:

* the T1w-grid lesion mask and probability image;
* an explicitly labeled synthetic SynthSR T1w alternative;
* labeled GIFTI copies of the intact synthetic scaffold;
* a three-plane mask-overlay image;
* one public-to-scaffold vertex table and one validity summary per hemisphere,
  with scaffold and public vertex and face counts;
* an nro-generated reconstruction summary.

The mask metadata records connected-component sizes, lesion volume,
laterality, model hashes, and overlap with nonzero anatomical support. The
reconstruction summary records the scaffold-synthesis method and the
reconstruction engine. With FastSurfer, it also records any voxels restored to
the FastSurfer mask from its segmentation. Anatomical scenes load these
diagnostic files and still display the cut surfaces. The ordinary T1w
reference remains the primary anatomy, and published surfaces contain only
surviving cortex. Review automatic masks and reconstructed boundaries visually.

MSMAll results add, for each hemisphere, a participant `fsnative`-to-MSMAll
registration sphere and the matching fsLR atlas sphere with white,
midthickness, pial, and inflated surfaces. They also add validity metrics,
registration QC, software versions, and a dedicated MSMAll manifest. That
manifest records the calibration inputs and parameters with portable BIDS
references and the inherited metadata fields that affect calibration. A
separate provenance document records a SHA-256 identity for every selected
source image. The ordinary anatomical manifest points to these products but
does not expose the calibration time series.

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.

**Image selection and geometry.** `selection_strategy` controls how
acquisitions are combined, and `t1w_fallback` chooses between T2w-to-T1w
synthesis and `skip`. `gradient_unwarping` is `auto` or `off`. In `auto` mode,
only acquisitions that match the site's hardware catalog are corrected. The
catalog and coefficient file are site resources, not module settings.
`fsaverage_template` selects `fsaverage6`, the packaged default, or the
full-resolution `fsaverage` target. `mni_template` selects the registration
target.

**Surface reconstruction.** `surface_reconstruction_engine` selects
`freesurfer`, the default, or `fastsurfer`. FastSurfer runs only its
neural-network segmentation on a GPU and finishes surfaces on a general CPU
worker. The engine choice, its version, the fixed 1 mm reconstruction grid,
and the scientific options are part of the artifact contract. Thread counts
and resource routing are not.

**Containers and storage.** `synthstrip_container` selects the brain-extraction
image. `freesurfer_container` and `fastsurfer_container` select the images for
their engines. These paths are execution settings, while the pinned versions
and reconstruction policies belong to the artifact contract. `container`
controls the runtime, image, binds, home, environment isolation, and inner
setup command for the rest of anatomy. Each BIDS project keeps one copy of
every required FreeSurfer `fsaverage` version under
`derivatives/nro/.nro/templates`. FreeSurfer-compatible subject directories
link to those copies with relative links, so the project stays portable.
`freesurfer_subjects_dir` and `fs_subject` override the shared
FreeSurfer-compatible storage location and subject identity.

**MSMAll.** The nested `msmall` block controls the HCP calibration route.
Markup, not discovery, decides whether a participant has the route, and
`enabled: false` turns it off even when markup selects runs. The other fields
pin the HCP mesh densities, functional resolution, smoothing, registration
modes, ICA dimension, high-pass setting, FIX threshold and model, and Octave
execution mode. These values affect only an enabled MSMAll branch. Existing
and ordinary anatomy inherit an empty calibration set and stay compatible.

**Lesion-aware processing.** Install the optional Python stack with
`./install --with-lesion`. The lesion method is pinned in code: the SynthStroke
model, source and model revisions, model hashes, 1 mm inference grid,
sliding-window settings, probability threshold, test-time augmentation, SynthSR
model, and surface-boundary policy. Reconstruction parameters come from the
selected surface engine. Lesion detection and the public lesion mask keep the
source anatomical grid. This pinned policy enters the work-item contract only
for lesion-marked participants, so adding the feature does not rename or stale
ordinary anatomical lineage.

`masker_command` is the lesion block's only configuration field, and it is an
execution setting. When it is `null`, anatomy runs its built-in adapter with
the installed Python environment and the checksum-verified SynthStroke model
stored in `synthstroke_data`. The worker never contacts Hugging Face. An
override must accept the same adapter options: `--input`, `--probability`,
`--mask`, `--model`, `--revision`, `--threshold`, `--device`, and optional
`--tta`. SynthStroke and SynthSR run on the general CPU worker. If FastSurfer
is the reconstruction engine, its segmentation phase still goes to the
scheduler's GPU pool, and that scheduling choice stays outside the artifact
contract.

**Execution.** `overwrite` requests re-execution, and `verbose` changes
logging.

```{literalinclude} ../../nro/definitions/starters/configs/anat/main_anat.yml
:language: yaml
```

Implementation: [module construction](../autoapi/nro/modules/anat/module/index.rst),
[planning](../autoapi/nro/modules/anat/planning/index.rst).
See [software and methods sources](../methods/software.md).
