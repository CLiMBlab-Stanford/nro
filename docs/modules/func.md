# Functional preprocessing

`func` prepares one BOLD run using the subject's completed anatomy. Its outputs
share the `preprocessing` lineage with `anat`. Computation across spaces shares
motion estimation, distortion correction, and ICA-AROMA; spaces are not separate
functional instances.

## Inputs and branch selection

The resolver reads inherited BIDS JSON metadata, the BOLD header, matching
SBRefs, and eligible spin-echo fieldmaps. It uses run entities, acquisition
timing, phase-encoding direction, voxel size, and readout compatibility to choose
references. Sidecars need not be adjacent to the NIfTI if inheritance supplies
the metadata. Header timing can supply TR where permitted by the resolver.

An eligible opposite-phase fieldmap pair takes precedence and uses FSL TOPUP.
Otherwise `sdc_method` selects `synbold_disco` or anatomical `syn`. SynBOLD
requires phase-encoding and readout metadata; if these are absent, the graph
uses anatomical SyN and records the reason. `sdc_from_sbref_pair` permits the
resolver to consider SBRef pairs. `fieldmap_syn_refine` adds the configured
registration refinement to fieldmap correction. The manifest records requested
and resolved methods; do not assume every run used the default branch.

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
   performs the correction. Nro stores its slice-wise rank-one artifact as a 3D
   loading map and a slice-by-time table, then removes the temporary 4D artifact.
3. Prepare BOLD/SBRef references and estimate rigid motion with FSL MCFLIRT.
   Reference-pose checks protect registration against badly aligned or
   incompatible SBRefs.
4. Estimate susceptibility distortion from TOPUP, SynBOLD-DisCo's synthetic
   reference, or anatomical ANTs registration. Compose pose and distortion
   transforms, adapting displacement fields to the BOLD readout when needed.
5. Align corrected EPI to anatomy with FreeSurfer boundary-based registration.
   `bbregister_surf`, `bbregister_init`, and `bbregister_dof` select the boundary,
   initialization, and rigid/affine degrees of freedom. Compose anatomy-to-MNI
   transforms for template output. Publish registration-check images.
6. Convert the composed spatial warp and per-frame motion transforms for AFNI
   `3dNwarpApply`. Resample the 4D series without splitting every TR into a
   permanent file. `output_grid` chooses native anatomy resolution or anatomy
   orientation at EPI voxel size. `use_jacobian` controls intensity modulation.
   Warp interpolation is linear; signal interpolation uses AFNI's `wsinc5`.
7. Generate confounds from motion, anatomical segmentations, and BOLD signals.
   These include motion parameters, their derivatives and squares, global/CSF/WM
   signals and expansions, framewise displacement, aCompCor, DVARS, and numbered
   outlier families. See [denoising](../methods/denoising.md).
8. When `clean_ica_aroma` is enabled, estimate a shared MELODIC/ICA-AROMA model
   and noise classification. The estimation input is spatially smoothed; shared
   components are then regressed from each output-space time course. The
   `ica_aroma_denoise_type` selects aggressive or nonaggressive regression.
   `ica_aroma_cmd` can replace the bundled implementation with an external command.
   Classification and estimation policies are recorded with the outputs.
9. Produce T1w/MNI volumes and left/right fsnative/template GIFTI time courses
   according to `output_spaces`. The packaged template target is `fsaverage6`;
   filenames record that exact space. Native surfaces and registration spheres
   come from anatomy. Publish sidecars, confounds, transforms, masks, and the
   run manifest only after validating required products.

Large published images use staged writes. Freshness checks are not a guarantee
against arbitrary external corruption: interrupted writes from older software
or manual edits may require inspection and deletion of the affected artifact.

The current functional graph does not perform slice-timing correction.
Native surface sampling is ribbon-constrained between white and pial surfaces.
It also does not correct gradient nonlinearity. Data that require gradient
unwarping, including uncorrected 7 T acquisitions, are not yet supported.

## Public artifacts

Run outputs are under
`derivatives/preprocessing/LINEAGE/sub-ID/[ses-ID/]func/`. Names retain the
run's BIDS entities and add space, hemisphere, and processing descriptions.
Products include preprocessed BOLD images, their JSON sidecars, brain masks,
registration transforms/QC images, and `desc-confounds_timeseries.tsv` with
column metadata. AROMA-enabled runs also retain corresponding no-AROMA products
and classification records. Fieldmap-derived products exist only on applicable
branches. MARSS-enabled runs also contain native artifact loadings, artifact
timecourses, a mean-absolute artifact map, slice-correlation tables and heatmap,
and decision metadata. Pass-through runs use zero-valued artifact placeholders
to keep the output signature fixed. The functional manifest identifies every
published path.

The [functional contract](../autoapi/nro/modules/func/contracts/index.rst) defines the
required manifest and sidecar fields; the
[resolver](../autoapi/nro/modules/func/resolver/index.rst) defines reference selection.

## Configuration

The SynBOLD overlap, translation, rotation, and SBRef support/correlation
thresholds reject implausible reference alignment. `syn_base_*` and
`syn_refine_*` are ANTs transform, convergence, shrink-factor, and smoothing
schedules; they apply only to the relevant registration branch. `topup_config`
selects the TOPUP settings. `io_chunk_vols` bounds I/O chunks, not scientific
temporal filtering. Thread/force/logging controls have the same role as in anat.
The shared `fsaverage_template` setting selects the surface target and must
match any fsaverage-family entry in `func.output_spaces`.

`marss_mode` defaults to `auto`. Its correction rule follows the publication's
recommendation to apply MARSS at multiband factors of six or greater. The
slice-correlation score is diagnostic and does not control correction. MARSS
1.0.2 supports the regular simultaneous-slice layout on NIfTI axis `k`.
Lower-factor data and data with missing or unsupported slice metadata produce a
documented pass-through result. `marss_min_multiband_factor` changes the cutoff;
values below six are experimental because the correction estimate averages
fewer simultaneously acquired slices. The installer includes the MARSS dependency by default. Pass
`--without-marss` only when all functional configurations use `marss_mode: off`
or `diagnose`.

`confounds.aseg_in_epi` and `brain_mask_in_epi` override confound extraction
masks. `n_acompcor` and `acompcor_max_voxels` bound aCompCor extraction.
`fd_radius_mm` converts rotational motion to displacement;
`motion_outlier_fd_thresh` is the extreme-motion threshold in mm.
`dvars_statistical_alpha`, `dvars_practical_threshold_percent`, and
`dvars_power` control DVARS inference.
`nonsteady_*` controls initial-volume stabilization detection. aCompCor and FD
are available as confounds but are not selected by the default clean regex.

```{literalinclude} ../../nro/configuration/starters/configs/preprocessing/main_preprocessing.yml
:language: yaml
:start-at: func:
```

Implementation: [module](../autoapi/nro/modules/func/module/index.rst),
[MARSS integration](../autoapi/nro/modules/func/marss/index.rst),
[resampling](../autoapi/nro/modules/func/resampling/index.rst),
[confounds](../autoapi/nro/modules/func/confounds/index.rst).
See [software sources](../methods/software.md).
