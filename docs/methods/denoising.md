# Denoising and temporal censoring

Use this page to describe and justify how nro marks motion outliers, removes
nuisance signal, and decides whether a run is usable for connectivity. Each
choice is given with its motivation and sources.

## Motion outliers

nro marks outlier frames in three families of one-hot columns. Each family is
numbered independently within the run.

- `extreme_fd_outlierNN` marks frames whose Power framewise displacement is
  greater than 1 mm by default.
- `dvars_outlierNN` marks frames that meet both the statistical and the
  practical DVARS criteria of Afyouni and Nichols. By default, the upper-tail
  DVARS test must be significant at a 5% Bonferroni family-wise error rate
  within the run, and the change in percent D-var must exceed 5%. When the
  difference between two adjacent frames is anomalous, both frames are marked.
- `motion_outlierNN` marks the unique union of the two families above.

The DVARS rule requires both criteria on purpose. In high-dimensional data,
statistical significance alone can flag negligible changes. A fixed,
unstandardized DVARS cutoff has the opposite weakness: it depends on signal
scaling and on the run's baseline variability. Afyouni and Nichols
([2018](https://doi.org/10.1016/j.neuroimage.2017.12.098)) proposed combining
statistical and practical significance. They reported that the 5% practical
cutoff worked adequately in the HCP data they studied and noted that other data
sources might need a different value. The `dvars_statistical_alpha`,
`dvars_practical_threshold_percent`, and `dvars_power` configuration keys
control the test.

For connectivity, the temporal mask is the row-wise union of
`non_steady_state_outlierNN` and `motion_outlierNN`. This is logically the same
as the union of every `*_outlierNN` column, because the FD and DVARS columns
cannot mark a frame that `motion_outlierNN` leaves unmarked. nro keeps those
reason-specific columns for provenance and quality control.

First-level modeling selects retained rows with the same union. It does not fit
the reason-specific columns and the union columns as duplicate scan-nulling
regressors.

## Confound regression

By default, the connectivity nuisance model contains global signal, framewise
displacement, the six rigid-body motion parameters, the first five aCompCor
components, and DCT-II cosine drifts. aCompCor is estimated from combined
white-matter and CSF voxels. The source confounds table also contains the
36-parameter model of Satterthwaite et al.
([2013](https://doi.org/10.1016/j.neuroimage.2012.08.052)). Alternative clean
configurations can select that model without repeating functional
preprocessing.

The default selector also matches the numbered outlier columns. Because those
columns define the temporal mask, they are zero in every retained fitting row.
Cleaning discards them as unusable nuisance directions instead of fitting
scan-nulling regressors alongside censoring.

### How cleaning fits the model

Cleaning builds a real Fourier basis for the frequencies inside the configured
passband. It projects task and nuisance regressors into that basis using only
retained frames. Task and detrending terms stay in the model as explicit
columns.

The nuisance design is then made orthogonal to those task and detrending terms,
standardized, and reduced by principal-component analysis. nro keeps the
smallest number of components that reaches the configured variance target,
which is 99% by default. Two limits cap how many components can be kept: at
least 30 dimensions and at least 50% of the passband rank left after task
removal must remain. Both limits are configurable. If fewer than 30 dimensions
remain before nuisance regression, nuisance regression is skipped.

nro removes the task directions and the retained nuisance components from the
passband basis. It fits each BOLD series to the remaining clean-passband basis
using only retained frames, then evaluates the fitted model at every original
frame. Fitting this way avoids the rank saturation that can occur when a large
stopband basis is fitted on an irregular, censored grid. It also keeps the
stated passband and removes the task terms exactly, so PCA cannot trade either
away. The equations are in [Temporal projection and streaming
estimators](estimation.md).

The output keeps the original time axis. Values at censored frames are finite
model reconstructions. They are not observed or repaired data, and downstream
analyses must still apply the temporal mask. Location and scale after cleaning
are estimated from retained frames and applied to the complete output.

### Cleaning diagnostics

Each cleaned time-course sidecar reports:

- the temporal-mask burden, retained duration, and longest censored interval;
- the passband dimension, rank, and condition number;
- the rank of the task and detrending design;
- the nuisance columns and retained PCA components;
- the joint regression rank, residual design degrees of freedom, and final
  algebraic temporal rank;
- the numerical and effective ranks of the cleaned data and the dominant
  temporal variance fraction;
- the number of spatial locations used for those data-derived estimates and how
  they were selected.

These values are descriptive. The cleaning module does not assign a run-level
pass or fail status from them.

### Undefined cleaning

Cleaning is mathematically undefined in two cases: censoring makes the passband
basis unidentifiable, or the task and detrending design uses up the whole
estimable passband. The module still writes its usual outputs. It writes an
all-zero time course, sets `CleaningDefined` to false in the sidecar, and
records the cause in `CleaningUndefinedReason`. This output is a sentinel and
contains no usable functional data. Downstream modules must exclude it and
record the exclusion in their own metadata.

## Run usability

A run is usable for connectivity only if it meets all of these conditions:

- it retains at least 100 frames;
- it retains at least 70% of its original frames;
- the clean model reports at least 50 residual design degrees of freedom;
- the clean sidecar reports a participation effective rank of at least 10;
- no single component accounts for more than 50% of the variance.

Modules can add aggregate requirements. The `main` microparcellation
configuration requires at least two usable runs and 1,100 retained frames in
total. The `main` dynconn configuration requires one usable run and 100
retained frames in total. Module configuration is the authority for all of
these thresholds.

## References

The citation records for this page are in [`../references.bib`](../references.bib).
