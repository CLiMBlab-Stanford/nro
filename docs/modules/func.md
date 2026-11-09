# Functional preprocessing

`func` preprocesses one BOLD run using the participant's completed anatomy and
publishes the run in every supported output space along with its confounds.
Read this page to see how nro chooses distortion correction, what the default
pipeline does, and which settings change the published series.

One work item produces all output spaces for a run. Motion estimation,
distortion correction, and ICA classification are computed once and shared
across those spaces, so spaces are not separate functional work items. `func`
has its own configuration and output directory.

## Inputs and distortion-correction choice

The resolver reads inherited BIDS JSON metadata, the BOLD header, matching
SBRefs, and eligible spin-echo fieldmaps. It chooses references from run
entities, acquisition timing, phase-encoding direction, voxel size, and readout
compatibility. Sidecars do not need to sit next to the NIfTI file when BIDS
inheritance supplies the metadata, and the resolver can take TR from the header
where it permits.

nro picks a susceptibility distortion correction method in this order:

1. Associations written during bidsification take precedence.
2. An eligible opposite-phase fieldmap pair uses FSL TOPUP. With
   `sdc_from_sbref_pair`, automatic TOPUP selection uses opposite-phase SBRefs
   instead of dedicated fieldmaps. A malformed optional fieldmap does not hide
   other valid candidates. The resolver records a warning and continues with
   the usable files.
3. Otherwise `sdc_method` selects `synbold_disco`, the default, or anatomical
   `syn`. SynBOLD needs phase-encoding and readout metadata. Without it, the
   graph falls back to anatomical SyN and records the reason.

`fieldmap_syn_refine` adds the configured registration refinement to fieldmap
correction. The manifest records both the requested and the resolved methods,
so check it instead of assuming a run used the default path.

## Processing

1. **Validate inputs and measure slice artifacts.** nro validates the inputs,
   declares outputs, and optionally limits the data with `debug_first_nvols`.
   When `marss_mode` is not `off`, it derives simultaneous-slice groups from
   `SliceTiming`, checks them against `MultibandAccelerationFactor`, and
   measures excess correlation within the groups. The diagnostic averages each
   target slice's simultaneous and comparison-slice correlations in Fisher-z
   space and weights every target slice equally. `diagnose` records this
   measurement without correcting. `auto` applies MARSS at multiband factors of
   six or greater and passes other data through. A pass-through result aliases
   the source BOLD without copying it.
2. **Correct with MARSS.** The official MARSS package estimates and removes the
   correction in native scanner space, before any data used by later steps are
   resampled. Motion parameters are estimated without keeping the resampled
   motion-estimation series. nro stores MARSS's slice-wise rank-one artifact as
   a 3D loading map and a slice-by-time table, then deletes the temporary 4D
   artifact.
3. **Correct gradients and estimate motion.** Gradient correction is resolved
   separately for BOLD, SBRef, and spin-echo images. In `auto` mode, a matching
   site profile applies HCP gradient unwarping unless the input reports prior
   nonlinear-gradient correction, and unmatched acquisitions pass through. This
   step follows MARSS so that simultaneous-slice groups still refer to native
   scanner slices. nro then prepares BOLD and SBRef references and estimates
   rigid motion with FSL MCFLIRT on the corrected data. Reference-pose checks
   keep a badly aligned or incompatible SBRef from corrupting registration.
4. **Estimate distortion.** The resolved method estimates susceptibility
   distortion from TOPUP, the SynBOLD-DisCo synthetic reference, or anatomical
   ANTs registration. nro composes the pose and distortion transforms and
   adapts displacement fields to the BOLD readout when needed.
5. **Register to anatomy.** FreeSurfer boundary-based registration aligns the
   corrected EPI to the participant T1w anatomy. `bbregister_surf`,
   `bbregister_init`, and `bbregister_dof` select the boundary surface, the
   initialization, and rigid or affine degrees of freedom. nro composes the
   anatomy-to-MNI transforms for template output and publishes
   registration-check images.
6. **Resample once.** nro converts the composed spatial warp and per-frame
   motion transforms for AFNI `3dNwarpApply`. When BOLD gradient correction is
   active, its source-grid displacement field joins the same pull-transform
   chain. The post-MARSS source is sampled once, so gradient correction adds no
   extra interpolation to the published series. The 4D series is resampled
   without writing every TR to a permanent file. `output_grid` chooses native
   anatomical resolution or anatomical orientation at EPI voxel size, and
   `use_jacobian` controls intensity modulation. Warps use linear
   interpolation, and signal uses AFNI's `wsinc5`.
7. **Classify ICA noise components (optional).** When `ica_classifier` is
   `ica_aroma` or `cicada`, nro runs one MELODIC decomposition on the spatially
   smoothed T1w-space series. It transforms the component maps needed for
   classification to the 2 mm MNI reference. The classifier labels noise
   components, which are regressed from every output-space time course with
   the regression mode in `ica_regression`. CICADA runs through the separately
   installed, pinned pycicada process and takes its motion features from a
   private FD/DVARS table computed before denoising. The manifest records the
   classifier, labels, regression policy, and warnings.
8. **Compute confounds.** nro builds the published confounds from motion,
   anatomical segmentations, and the final BOLD series. They include motion
   parameters with their derivatives and squares, global, gray-matter, CSF,
   and white-matter signals with expansions, framewise displacement, aCompCor,
   DVARS, and numbered outlier families. `global_signal` averages the whole
   brain mask. `global_gm_signal` averages FreeSurfer aseg cerebral-cortex
   voxels after scaling each voxel's time course to unit standard deviation,
   so every cortical voxel contributes equally. It can replace `global_signal`
   in a cleaning or contrasts `confounds_regex`. See
   [denoising](../methods/denoising.md).
9. **Publish.** nro produces T1w and MNI volumes and left and right fsnative
   and template GIFTI time courses. Native surfaces and registration spheres
   come from anatomy, and the selected `anat` configuration defines the
   fsaverage target, which is `fsaverage6` in the packaged configuration.
   Native surface sampling is constrained to the ribbon between the white and
   pial surfaces. Sidecars, confounds, transforms, masks, and the run manifest
   are published only after the required products validate.

The current graph does not perform slice-timing correction.

Large published images are written in stages. Freshness checks do not protect
against arbitrary outside corruption. An interrupted write from older software
or a manual edit may require you to inspect and delete the affected artifact.

## Outputs

Run outputs go under
`derivatives/nro/func/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/[ses-ID/]func/`.
Filenames keep the run's BIDS entities and add space, hemisphere, and
processing descriptions. Each run has:

* preprocessed BOLD images and their JSON sidecars;
* brain masks;
* registration transforms and QC images;
* `desc-confounds_timeseries.tsv` with column metadata;
* a functional manifest that lists every published path.

Fieldmap-derived products appear only when a fieldmap was used.

Each workflow publishes one canonical `desc-preproc` series. Its sidecar
records whether ICA component regression was applied and which classifier
supplied the labels. To compare classifiers or regression modes, use a
separate workflow for each choice. Classification records stay in private work
storage.

When MARSS is enabled, a run also has native artifact loadings, artifact time
courses, a mean-absolute artifact map, slice-correlation tables and a heatmap,
and decision metadata. A pass-through run gets zero-valued placeholders so the
set of outputs is the same for every run.

The [functional contract](../autoapi/nro/modules/func/contract/index.rst)
defines the required manifest and sidecar fields. The
[resolver](../autoapi/nro/modules/func/resolver/index.rst) defines reference
selection.

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.
Thread, overwrite, and logging controls work as they do in `anat`.

Space is not a `func` setting. The selected `anat` `fsaverage_template` sets
the available template surface, and `--space` selects downstream work when you
make a request. Requesting a different space does not create a new `func`
configuration or directory.

**Registration.** The SynBOLD overlap, translation, rotation, and SBRef
support and correlation thresholds reject implausible reference alignment.
`syn_base_*` and `syn_refine_*` hold ANTs transform, convergence,
shrink-factor, and smoothing schedules, and each applies only to its
registration path. `topup_config` selects the TOPUP settings. `io_chunk_vols`
sets the I/O chunk size and has no effect on temporal filtering.

**Gradient unwarping.** `gradient_unwarping` is `auto` or `off`. `auto`
corrects only acquisitions matched by `hardware/gradient_unwarping.yml` and
passes others through. Contracts record the coefficient file's SHA-256 digest
instead of its absolute path, so moving an unchanged file does not change
scientific identity. A matching profile whose coefficient file is missing is a
site-configuration error.

**MARSS.** `marss_mode` defaults to `auto`, which follows the MARSS
publication's recommendation to correct at multiband factors of six or
greater. The slice-correlation score is diagnostic only and does not decide
correction. MARSS 1.0.2 supports the regular simultaneous-slice layout on NIfTI
axis `k`. Data below the cutoff, and data with missing or unsupported slice
metadata, produce a documented pass-through result.
`marss_min_multiband_factor` changes the cutoff. Values below six are
experimental because the estimate then averages fewer simultaneous slices. The
installer includes MARSS by default. Pass `--without-marss` only when every
functional configuration uses `marss_mode: off` or `diagnose`.

**ICA classification.** `ica_classifier` accepts `none`, `ica_aroma`, or
`cicada`, and defaults to `none`, which skips MELODIC. The two classifiers are
available for explicit comparison configurations. CICADA requires the MNI
functional output while its integration is under evaluation. Install it with
`./install --with-cicada`. A CICADA configuration reports that command if the
package is missing. `cicada_tolerance` and `cicada_smoothing_retention_mode`
control CICADA's classification and do not affect the artifact identity of
AROMA or no-classifier configurations.

**Confounds.** `confounds.aseg_in_epi` and `brain_mask_in_epi` override the
masks used for confound extraction. `n_acompcor` and `acompcor_max_voxels`
bound aCompCor. `cosine_high_pass_hz` sets the cutoff for the exported
nonconstant DCT-II cosine drift regressors and defaults to 1/128 Hz.
`fd_radius_mm` converts rotation to displacement, and
`motion_outlier_fd_thresh` is the extreme-motion threshold in mm.
`dvars_statistical_alpha`, `dvars_practical_threshold_percent`, and
`dvars_power` control DVARS inference. The `nonsteady_*` keys control detection
of initial unstable volumes. Which of these columns enter cleaning is decided
by the `clean` configuration. Its default uses global signal, FD, the base
motion parameters, the first five aCompCor components, cosine drifts, and
outlier columns.

```{literalinclude} ../../nro/definitions/starters/configs/func/main_func.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/func/module/index.rst),
[MARSS integration](../autoapi/nro/modules/func/marss/index.rst),
[resampling](../autoapi/nro/modules/func/resampling/index.rst),
[confounds](../autoapi/nro/modules/func/confounds/index.rst).
See [software sources](../methods/software.md).
