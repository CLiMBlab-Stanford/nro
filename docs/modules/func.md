# Functional preprocessing

`func` prepares one BOLD run using the subject's completed anatomy. It has its
own configuration and output directory. Computation across spaces shares motion
estimation, distortion correction, and ICA classification; spaces are not separate
functional work items.

## Inputs and processing choices

The resolver reads inherited BIDS JSON metadata, the BOLD header, matching
SBRefs, and eligible spin-echo fieldmaps. It uses run entities, acquisition
timing, phase-encoding direction, voxel size, and readout compatibility to choose
references. Sidecars need not be adjacent to the NIfTI if inheritance supplies
the metadata. Header timing can supply TR where permitted by the resolver.

An eligible opposite-phase fieldmap pair takes precedence and uses FSL TOPUP.
One malformed optional fieldmap does not hide other valid candidates; the
resolver records a warning and continues with the usable files. Otherwise
`sdc_method` selects `synbold_disco` or anatomical `syn`. SynBOLD requires
phase-encoding and readout metadata; if these are absent, the graph uses
anatomical SyN and records the reason. `sdc_from_sbref_pair` switches automatic
TOPUP selection from dedicated fieldmaps to opposite-phase SBRefs. Explicit
associations written during bidsification still take precedence.
`fieldmap_syn_refine` adds the configured registration refinement to fieldmap
correction. The manifest records requested and resolved methods; do not assume
every run used the default path.

## Processing sequence

1. Validate inputs and declare outputs; optionally restrict data with
   `debug_first_nvols`. When `marss_mode` is not `off`, derive simultaneous-slice
   groups from `SliceTiming`, validate them against `MultibandAccelerationFactor`,
   and measure excess correlation within those groups. `diagnose` records the
   measurement without correction. `auto` applies MARSS at multiband factors of
   six or greater and otherwise passes the source through. The
   diagnostic gives each target slice equal weight after averaging its
   simultaneous and comparison-slice correlations in Fisher-z space. A
   pass-through result aliases the source BOLD without copying it.
2. Estimate the MARSS correction in native scanner space before data used by the
   rest of preprocessing are resampled. Motion parameters are estimated without
   retaining the resampled motion-estimation series. The official MARSS package
   performs the correction. nro stores its slice-wise rank-one artifact as a 3D
   loading map and a slice-by-time table, then removes the temporary 4D artifact.
3. Resolve gradient correction separately for BOLD, SBRef, and spin-echo images.
   In `auto` mode, a matching site profile applies HCP gradient unwarping unless
   the input reports prior nonlinear-gradient correction. Unmatched acquisitions
   pass through. Gradient correction follows MARSS so simultaneous-slice groups
   still refer to native scanner slices. Prepare BOLD/SBRef references and
   estimate rigid motion with FSL MCFLIRT on corrected data.
   Reference-pose checks protect registration against badly aligned or
   incompatible SBRefs.
4. Estimate susceptibility distortion from TOPUP, SynBOLD-DisCo's synthetic
   reference, or anatomical ANTs registration. Compose pose and distortion
   transforms, adapting displacement fields to the BOLD readout when needed.
5. Align corrected EPI to the participant T1w anatomy with FreeSurfer boundary-based registration.
   `bbregister_surf`, `bbregister_init`, and `bbregister_dof` select the boundary,
   initialization, and rigid/affine degrees of freedom. Compose anatomy-to-MNI
   transforms for template output. Publish registration-check images.
6. Convert the composed spatial warp and per-frame motion transforms for AFNI
   `3dNwarpApply`. When BOLD gradient correction is active, include its retained
   source-grid displacement field in the same pull-transform chain. The final
   operation samples the post-MARSS source once, so gradient correction does not
   add another interpolation to the published series. Resample the 4D series
   without splitting every TR into a permanent file. `output_grid` chooses native
   anatomy resolution or anatomy orientation at EPI voxel size. `use_jacobian`
   controls intensity modulation. Warp interpolation is linear; signal
   interpolation uses AFNI's `wsinc5`.
7. When `ica_classifier` selects `ica_aroma` or `cicada`, estimate one shared
   MELODIC decomposition from the spatially smoothed T1w series. Transform the
   component maps needed for classification to the 2 mm MNI reference. The
   selected classifier labels noise components, which are then regressed from
   every output-space time course. `ica_regression` selects aggressive or
   nonaggressive regression. CICADA runs through the site-managed executable in
   `cicada_cmd`; `cicada_tolerance` and
   `cicada_smoothing_retention_mode` control its classification. A private
   pre-denoising FD/DVARS table supplies CICADA's motion features. The manifest
   records the classifier, labels, regression policy, and warnings.
8. Generate the published confounds from motion, anatomical segmentations, and
   the final BOLD series. These include motion parameters, their derivatives and
   squares, global/CSF/WM signals and expansions, framewise displacement,
   aCompCor, DVARS, and numbered outlier families. See
   [denoising](../methods/denoising.md).
9. Produce T1w/MNI volumes and left/right fsnative/template GIFTI time courses.
   The selected `anat` configuration defines the fsaverage target; the packaged
   target is `fsaverage6`. Filenames record each exact space. Native surfaces
   and registration spheres come from anatomy. Publish sidecars, confounds,
   transforms, masks, and the run manifest only after validating required
   products.

Large published images use staged writes. Freshness checks are not a guarantee
against arbitrary external corruption: interrupted writes from older software
or manual edits may require inspection and deletion of the affected artifact.

The current functional graph does not perform slice-timing correction.
Native surface sampling is ribbon-constrained between white and pial surfaces.

## Public artifacts

Run outputs are under
`derivatives/nro/func/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/[ses-ID/]func/`.
Names retain the
run's BIDS entities and add space, hemisphere, and processing descriptions.
Products include preprocessed BOLD images, their JSON sidecars, brain masks,
registration transforms/QC images, and `desc-confounds_timeseries.tsv` with
column metadata. Each workflow publishes one canonical `desc-preproc` series;
its sidecar records whether ICA component regression was applied and which
classifier supplied the labels. Use a separate workflow for each classifier or
regression choice. Classification records are retained in private work storage.
Fieldmap-derived products exist only when used.
MARSS-enabled runs also contain native artifact loadings, artifact
timecourses, a mean-absolute artifact map, slice-correlation tables and heatmap,
and decision metadata. Pass-through runs use zero-valued artifact placeholders
to keep the output signature fixed. The functional manifest identifies every
published path.

The [functional contract](../autoapi/nro/modules/func/contract/index.rst) defines the
required manifest and sidecar fields; the
[resolver](../autoapi/nro/modules/func/resolver/index.rst) defines reference selection.

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup); `null` ignores markup.
The SynBOLD overlap, translation, rotation, and SBRef support/correlation
thresholds reject implausible reference alignment. `syn_base_*` and
`syn_refine_*` are ANTs transform, convergence, shrink-factor, and smoothing
schedules; they apply only to the relevant registration path. `topup_config`
selects the TOPUP settings. `io_chunk_vols` bounds I/O chunks, not scientific
temporal filtering. Thread, overwrite, and logging controls have the same role as in anat.
Space is not a `func` configuration field. The selected `anat`
`fsaverage_template` defines the available template surface, while `--space`
selects downstream work at request time. Changing a requested space does not
create a new `func` configuration or directory.

`gradient_unwarping` selects `auto` or `off`. `auto` corrects only acquisitions
matched by `hardware/gradient_unwarping.yml`; it is otherwise a pass-through.
Moving an unchanged coefficient file does not change scientific identity because
contracts record its SHA-256 digest instead of its absolute path. A matching
profile with a missing coefficient is a site-configuration error.

`marss_mode` defaults to `auto`. Its correction rule follows the publication's
recommendation to apply MARSS at multiband factors of six or greater. The
slice-correlation score is diagnostic and does not control correction. MARSS
1.0.2 supports the regular simultaneous-slice layout on NIfTI axis `k`.
Lower-factor data and data with missing or unsupported slice metadata produce a
documented pass-through result. `marss_min_multiband_factor` changes the cutoff;
values below six are experimental because the correction estimate averages
fewer simultaneously acquired slices. The installer includes the MARSS
dependency by default. Pass `--without-marss` only when all functional
configurations use `marss_mode: off` or `diagnose`.

`ica_classifier` accepts `none`, `ica_aroma`, or `cicada`; the default is
`none`. ICA-AROMA and CICADA remain available for explicit comparison
configurations. Both classifiers require a MELODIC decomposition; the default
path skips it. CICADA requires the MNI
functional output while this integration is under evaluation. The site setting
`resources.pycicada` supplies its external executable. CICADA-specific settings
do not affect AROMA or no-classifier artifact identity.

`confounds.aseg_in_epi` and `brain_mask_in_epi` override confound extraction
masks. `n_acompcor` and `acompcor_max_voxels` bound aCompCor extraction.
`cosine_high_pass_hz` sets the cutoff for the exported nonconstant DCT-II
cosine drift regressors; its default is 1/128 Hz.
`fd_radius_mm` converts rotational motion to displacement;
`motion_outlier_fd_thresh` is the extreme-motion threshold in mm.
`dvars_statistical_alpha`, `dvars_practical_threshold_percent`, and
`dvars_power` control DVARS inference.
`nonsteady_*` controls initial-volume stabilization detection. The default clean
configuration selects global signal, FD, base motion parameters, the first five
aCompCor components, cosine drifts, and outlier columns.

```{literalinclude} ../../nro/configuration/starters/configs/func/main_func.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/func/module/index.rst),
[MARSS integration](../autoapi/nro/modules/func/marss/index.rst),
[resampling](../autoapi/nro/modules/func/resampling/index.rst),
[confounds](../autoapi/nro/modules/func/confounds/index.rst).
See [software sources](../methods/software.md).
