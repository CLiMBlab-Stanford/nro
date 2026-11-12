# Cleaning

`clean` removes nuisance signal from one functional run in one space at one
smoothing FWHM, and band-pass filters the result. Read this page to see what
the cleaning model does, how censored frames are handled, and what the quality
metadata in each sidecar means.

Each run is processed independently. The inputs are the run's `func`
derivatives and confounds, plus the anatomy needed for masks or surface
smoothing. The output keeps every original frame and records which frames
downstream analyses must exclude.

## Processing

1. **Select confounds and the temporal mask.** nro resolves the run's
   functional derivatives and confounds. `confounds_regex` selects continuous
   nuisance candidates. `temporal_mask_regex` selects one-hot outlier columns,
   and the union of their marked frames is excluded from fitting.
2. **Smooth.** When FWHM is positive, Workbench smooths in the selected
   surface or volume geometry before any temporal modeling. Volumetric
   gray-matter masking uses `gm_mask_threshold`. Zero FWHM skips smoothing.
3. **Build the temporal model.** nro builds a real Fourier basis for the band
   between `high_pass` and `low_pass` Hz and fits it on retained frames. Task
   terms are added when `regress_out_task` is true and events exist. With
   `detrend`, detrending adds exact model terms.
4. **Reduce the nuisance design.** Exact and nuisance regressors are projected
   into passband coordinates. nro removes the exact task and detrending
   directions, standardizes the remaining nuisance design, and runs PCA. It
   keeps components up to the `nuisance_variance_explained` target while
   protecting both an absolute and a fractional amount of residual rank. If
   the rank before nuisance removal is already below the absolute floor, nro
   skips nuisance regression instead of spending the remaining dimensions.
5. **Fit and reconstruct.** The remaining clean-passband basis is fitted to
   retained BOLD frames and evaluated at every original frame. This takes one
   pass over the data with a combined projection. If the passband and exact
   model are unidentifiable on the retained frames, nro writes zero-valued
   data and sets `CleaningDefined: false` with a reason.
6. **Standardize and publish.** With `standardize`, location and scale are
   estimated from retained frames and applied to the whole result. nro then
   streams data-quality summaries and writes sidecars and the completion
   manifest.

Values at excluded frames are model predictions, not recovered observations.
Connectivity modules drop those frames, and any other downstream analysis must
honor the mask too. Zero-valued outputs from an undefined model are sentinels
and must not be read as signal.

The [denoising methods page](../methods/denoising.md) explains the rank
protections and why outliers are censored. The
[estimation equations](../methods/estimation.md) give the retained-frame
projection and separate undefined models from input-validation errors.

## Outputs and quality metadata

Outputs go under
`derivatives/nro/clean/<CONFIG_ID>-<LINEAGE_DIGEST>/sub-ID/[ses-ID/]` and keep
run, space, hemisphere where applicable, and `smoothing-Nmm` in their
filenames. Each run, space, and smoothing combination has cleaned NIfTI or
paired GIFTI time courses, JSON sidecars, and its own manifest. Combinations
share the configuration-lineage directory but have distinct files.

Sidecars report:

* the temporal mask and retained-frame statistics;
* passband and exact-design ranks;
* the nuisance selection and the final algebraic rank;
* participation effective rank and dominant temporal variance fraction;
* whether cleaning was defined;
* the spatial sample used for the quality estimates.

Downstream modules use these fields to admit or reject runs without reloading
the data. The [clean contract](../autoapi/nro/modules/clean/contract/index.rst)
defines the exact names and types.

## Configuration

`markup` selects the source-markup document described in
[definitions stores](../definitions.md#source-markup). `null` ignores markup.

**Nuisance selection.** `confounds_regex` selects continuous nuisance
candidates, and `temporal_mask_regex` selects the censoring columns. The
default candidates are the cerebral-cortex signal `global_gm_signal`, FD, the
six base motion parameters, the
first five aCompCor components, cosine drifts, and outlier columns. The outlier
columns also define the temporal mask, so they are zero on every retained row
and are dropped before nuisance PCA. `nuisance_variance_explained` is a
fraction, not a percentage.

**Rank and input limits.** `minimum_temporal_rank` and
`minimum_temporal_rank_fraction` constrain nuisance PCA. They are not
connectivity admission thresholds. `min_trs` rejects runs with too few original
frames before graph construction.

**Model.** `gm_mask_threshold` sets gray-matter support. `regress_out_task`,
`detrend`, `standardize`, `high_pass`, and `low_pass` change the model.
Smoothing is selected per work item and is not a `clean` configuration key.

**Execution.** Container settings select Workbench and other external
execution resources. `wb_command` is the Workbench command inside that
environment. `overwrite` and `verbose` control re-execution and logging.

```{literalinclude} ../../nro/definitions/starters/configs/clean/main_clean.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/modules/clean/module/index.rst),
[downstream reader](../autoapi/nro/engine/cleaned_timeseries/index.rst).
