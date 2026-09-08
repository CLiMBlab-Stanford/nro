# Cleaning

`clean` processes each run independently in one space at one smoothing FWHM.
Its direct inputs are functional derivatives and the anatomy needed for masks
or surface smoothing. It retains the full frame count and records which frames
downstream analyses must exclude.

## Processing sequence and branches

1. Resolve matching functional derivatives and confounds for the run. The
   default regex selects the 36 expanded motion and tissue/global-signal
   regressors; it does not select outlier one-hot columns. A separate
   `temporal_mask_regex` selects the union of excluded frames.
2. Spatial smoothing precedes temporal projection when FWHM is positive.
   Workbench performs surface or volume smoothing in the selected geometry.
   Volumetric gray-matter masking uses the configured threshold. Zero FWHM
   omits smoothing.
3. Build the real Fourier basis inside `high_pass`–`low_pass` Hz. Fit it using
   retained frames. Construct task terms when `regress_out_task` is true and
   events exist; detrending contributes exact model terms when enabled.
4. Project exact and nuisance regressors into passband coordinates. Remove
   exact task/detrending directions, standardize the remaining nuisance design,
   and perform PCA. Retain the variance target subject to both absolute and
   fractional residual-rank protections. If the pre-nuisance rank is already
   below the absolute floor, skip nuisance regression rather than spend the
   remaining dimensions.
5. Fit the remaining clean-passband basis to retained BOLD frames and evaluate
   it at every original frame. This is one physical data pass with a combined
   projection, not three successive full-volume regressions. If the retained
   passband/exact model is unidentifiable, write zero-valued data and mark
   `CleaningDefined: false` with a reason.
6. With `standardize`, estimate location/scale from retained frames and apply
   it to the full result. Stream data-quality summaries and write sidecars and
   the completion manifest.

The full reconstruction at excluded frames is a model prediction, not a
recovered observation. Connectivity drops those frames. Other downstream
analyses must honor the mask too. Undefined zero outputs are sentinels, not
valid signals. The [denoising methods page](../methods/denoising.md) explains
the rank protections and the outlier rationale.
The [estimation equations](../methods/estimation.md) give the retained-frame
projection and distinguish undefined models from input-validation errors.

## Public artifacts and quality metadata

Outputs are under `derivatives/clean/LINEAGE/sub-ID/[ses-ID/]/` and retain run,
space, hemisphere where applicable, and `smoothing-Nmm` in their filenames.
The output set includes NIfTI or paired GIFTI cleaned time courses, JSON
sidecars, and a run/space/smoothing-specific manifest. Different combinations
share the derivative lineage directory but not file identities.

Sidecars report the temporal mask, retained-frame statistics, passband and
exact-design ranks, nuisance selection, final algebraic rank, participation
effective rank, dominant temporal variance fraction, and whether cleaning was
defined. Quality estimates also identify the spatial sample used. These fields
support downstream admission decisions without reloading all data. Exact names
and types are defined in the [clean contract](../autoapi/nro/clean/contract/index.rst).

## Configuration

`confounds_regex` controls continuous nuisance selection; `temporal_mask_regex`
controls censoring. `nuisance_variance_explained` is a fraction, not a percentage.
`minimum_temporal_rank` and `minimum_temporal_rank_fraction` constrain nuisance
PCA; they are not connectivity admission thresholds. `min_trs` rejects inputs
with too few original frames before graph construction. `gm_mask_threshold`
selects gray-matter support. `regress_out_task`, `detrend`, `standardize`, and passband bounds alter
the model. Smoothing is an instance selector, not a clean configuration key.

Container settings select Workbench and other external execution resources;
`force` and `verbose` control execution and reporting. `wb_command` is the tool
command used inside the configured execution environment.

```{literalinclude} ../../nro/configuration/starters/configs/clean/main_clean.yml
:language: yaml
```

Implementation: [module](../autoapi/nro/clean/module/index.rst),
[downstream reader](../autoapi/nro/engine/cleaned_timeseries/index.rst).
